import json
import os
import sys
from threading import Thread
import time
import sqlite3 as sqlite
import multiprocessing as mp

from pipelines.pipeline_naturezas import PipelineNaturezasAPI, PipelineNaturezasAPIb25

class EmergencyInterpreter:

    hardware_to_interpreter = {
        "azure-api": "azure-api",
        "gcp-api": "gcp-api",
        "triton-server": "triton-server",
        "vllm-api": "vllm-api",
        "serpro-llm": "serpro-llm"
    }
    SUPPORTED_INTERPRETERS = [
        "azure-api",
        "gcp-api",
        "triton-server",
        "vllm-api",
        "serpro-llm"
    ]
    EMBEDDING_DEVICES = ["cpu", "cuda"]

    def __init__(
        self,
        model_name,
        emb_model,
        hardware_config="cpu",
        embedding_device="cpu",
        sqlite_path="",
        n_cpus=6,
        max_similar_natures=16,
        b25_weight=0.15
    ):
        assert hardware_config in EmergencyInterpreter.hardware_to_interpreter.keys()
        interpreter_type = EmergencyInterpreter.hardware_to_interpreter[hardware_config]
        assert interpreter_type in EmergencyInterpreter.SUPPORTED_INTERPRETERS
        assert embedding_device in EmergencyInterpreter.EMBEDDING_DEVICES
        print(
            f"Iniciando EmergencyInterpreter com: {hardware_config} e {interpreter_type}"
        )
        self.sqlite_path = sqlite_path
        if not os.path.exists(self.sqlite_path):
            raise FileNotFoundError(f"SQLite database not found at {self.sqlite_path}")
        if interpreter_type in ["azure-api", "triton-server", "gcp-api", "vllm-api", "serpro-llm"]:

            self.inf_pipeline = PipelineNaturezasAPIb25(
                model_name, emb_model, client_type=interpreter_type, embedding_device=embedding_device
            )
            self.concurrency_mode = "async"
        else:
            raise ValueError(f"Interpreter type {interpreter_type} not supported.")
        self.sqlite_conn = sqlite.connect(self.sqlite_path)
        self.max_similar_natures = max_similar_natures
        self.redo_queue = None
        self.b25_weight = b25_weight

    def set_agent_can_redo_queue(self, redo_queue: mp.Queue):
        self.redo_queue = redo_queue

    def close(self):
        self.sqlite_conn.close()

    def send_interpretations(self, interpretations):
        # Campos necessários: id_emergencia, resultado, tipo, meta
        try:
            conn_to_use = self.sqlite_conn
            cursor = self.sqlite_conn.cursor()
        except sqlite.ProgrammingError as err:
            conn_to_use = sqlite.connect(self.sqlite_path)
            cursor = conn_to_use.cursor()
        tuples = []
        cols = """id_emergencia, horario_contexto, horario_fim, 
            tipo_de_inferencia, resultado, duracao_inferencia, 
            duracao_outros_processamentos, input_tokens, output_tokens, 
            modelo_utilizado"""
        for a in interpretations:
            id_emergencia = int(a["id_emergencia"])
            horario_contexto = a["horario_contexto"]
            horario_fim = a["horario_fim"]
            tipo_de_inferencia = a["tipo"]
            resultado = json.dumps(a["resultado"], ensure_ascii=False)
            duracao_inferencia = a["meta"].get("processing_time", None)
            duracao_outros_processamentos = a["meta"].get("no_gpu_time", None)
            input_tokens = a["meta"].get("input_tokens", None)
            output_tokens = a["meta"].get("output_tokens", None)
            modelo_utilizado = a["meta"].get("model_name", None)
            tuples.append(
                (
                    id_emergencia,
                    horario_contexto,
                    horario_fim,
                    tipo_de_inferencia,
                    resultado,
                    duracao_inferencia,
                    duracao_outros_processamentos,
                    input_tokens,
                    output_tokens,
                    modelo_utilizado,
                )
            )
        cursor.executemany(
            f"""INSERT INTO resultados_inferencia 
            ({cols}) VALUES (?,?,?,?,?,?,?,?,?,?)""",
            tuples,
        )

        conn_to_use.commit()

    def process_emergency(self, context: dict, result_queue: list, max_naturezas: int):
        sent = False
        assert self.redo_queue is not None
        try:
            if max_naturezas is None:
                max_naturezas = self.max_similar_natures
            horario = context["horario_ultima"]
            transcription = context["transcription"]

            # Reduzir caso o comprimento maximo de contexto seja pequeno
            c_len = (
                self.inf_pipeline.context_len
                if self.inf_pipeline.context_len != None
                else 10000
            )
            transcription = (
                transcription if len(transcription) <= c_len else transcription[-c_len:]
            )

            interpretation_a = self.inf_pipeline.create_interpretation_a([transcription])
            interpretation_a, meta_dict = interpretation_a[0]

            desc = None
            id_emergencia = context["id_emergencia"]
            m = meta_dict["meta"]
            if "error" in m:
                interpretation_1 = None
                msg = (
                    "Error while running interpreter.create_interpretation_a:"
                    + str(type(m["error"]))
                    + " "
                    + m["error"]
                )
                print(msg, file=sys.stderr)
                self.redo_queue.put((id_emergencia, msg))
                sent = True
                raise Exception(msg)
            else:
                desc = interpretation_a["descricao_breve"]
                interpretation_1 = {
                    "id_emergencia": id_emergencia,
                    "horario_contexto": horario,
                    "horario_fim": time.time(),
                    "resultado": {
                        "descricao_breve": desc,
                        "outras_observacoes": interpretation_a["outras_observacoes"],
                        "ponto_de_referencia": interpretation_a["ponto_de_referencia"],
                        "nome_do_solicitante": interpretation_a["nome_do_solicitante"],
                    },
                    "tipo": "descricao_e_observacao",
                    "meta": m,
                }

                self.send_interpretations([interpretation_1])

                similares = self.inf_pipeline.find_similar_natures([desc], max_n=max_naturezas, 
                    transcripts=[transcription], bm25_weight=self.b25_weight)
                similares = similares[desc]
                similares_vec = similares["sims_dict"]
                meta_vec_similares = similares["meta"]
                interpretation_2 = {
                    "id_emergencia": id_emergencia,
                    "horario_contexto": horario,
                    "horario_fim": time.time(),
                    "resultado": {
                        "classificacoes_provaveis": similares_vec,
                    },
                    "tipo": "lista_de_naturezas",
                    "meta": meta_vec_similares,
                }
                self.send_interpretations([interpretation_2])

                interpretations_b = self.inf_pipeline.create_interpretation_b(
                    [transcription], [similares_vec]
                )
                meta_dict_b = [m for _, m in interpretations_b][0]
                m2 = meta_dict_b["meta"]
                interpretation_b = [x for x, m in interpretations_b][0]

                interpretation_3 = None
                final_result = None

                if "error" in m:
                    msg = (
                        "Error while running interpreter.create_interpretation_b:"
                        + str(type(m["error"]))
                        + " "
                        + m["error"]
                    )
                    print(msg, file=sys.stderr)
                    self.redo_queue.put((id_emergencia, msg))
                    sent = True
                    raise Exception(msg)
                else:
                    interpretation_3 = {
                        "id_emergencia": id_emergencia,
                        "horario_contexto": horario,
                        "horario_fim": float(time.time()),
                        "resultado": {
                            "classificacao_decisiva": interpretation_b,
                        },
                        "tipo": "natureza_decisiva",
                        "meta": m2,
                    }

                    self.send_interpretations([interpretation_3])
                    sent = True
                    final_result = {
                        "id_emergencia": id_emergencia,
                        "transcricao": transcription,
                        "resultados": {
                            "classificacoes_provaveis": similares_vec,
                            "classificacao_decisiva": interpretation_b,
                            "descricao_breve": desc,
                            "outras_observacoes": interpretation_a[
                                "outras_observacoes"
                            ],
                            "ponto_de_referencia": interpretation_a[
                                "ponto_de_referencia"
                            ],
                            "nome_do_solicitante": interpretation_a[
                                "nome_do_solicitante"
                            ],
                        },
                        "meta": {"a": m, "b": meta_vec_similares, "c": m2},
                    }
                result_queue.append(final_result)

            interpretation_env = self.inf_pipeline.create_envolvimentos([transcription])
            interpretation_env, meta_dict = interpretation_env[0]
            m = meta_dict["meta"]
            if "error" in m:
                interpretation_env = None
                msg = (
                    "Error while running interpreter.create_envolvimentos:"
                    + str(type(m["error"]))
                    + " "
                    + m["error"]
                )
                print(msg, file=sys.stderr)
                self.redo_queue.put((id_emergencia, msg))
                sent = True
                raise Exception(msg)
            else:
                envolvimentos = interpretation_env["envolvimentos_emergencia"]
                interpretation_part = {
                    "id_emergencia": id_emergencia,
                    "horario_contexto": horario,
                    "horario_fim": float(time.time()),
                    "resultado": {
                        "envolvimentos": envolvimentos,
                    },
                    "tipo": "envolvimentos",
                    "meta": m,
                }

                self.send_interpretations([interpretation_part])
                self.redo_queue.put((id_emergencia, "envolvimentos"))

        except Exception as err:
            if not sent:
                self.redo_queue.put(
                    (
                        context["id_emergencia"],
                        "Erro desconhecido: " + str(err) + " " + str(type(err)),
                    )
                )
                print(err, file=sys.stderr)
            raise (err)

    def process_emergencies_async(self, contexts, max_naturezas=None, join=False):
        threads = []
        results = []
        for c in contexts:
            t = Thread(target=self.process_emergency, args=(c, results, max_naturezas))
            t.start()
            threads.append(t)
        return results

    def process_emergencies_batch(self, contexts, max_naturezas=None):
        sent_to_redo = set()
        if max_naturezas is None:
            max_naturezas = self.max_similar_natures
        horarios = [em["horario_ultima"] for em in contexts]
        transcriptions = [x["transcription"] for x in contexts]

        # Reduzir caso o comprimento maximo de contexto seja pequeno
        c_len = self.inf_pipeline.context_len
        transcriptions = [t if len(t) <= c_len else t[-c_len:] for t in transcriptions]

        interpretations_a = self.inf_pipeline.create_interpretation_a(transcriptions)
        meta_dicts_a = [m for _, m in interpretations_a]
        interpretations_a = [x for x, m in interpretations_a]
        descs = [r["descricao_breve"] for r in interpretations_a]
        interpretations_vec_1 = []
        for n, transcription in enumerate(transcriptions):
            desc = descs[n]
            id_emergencia = contexts[n]["id_emergencia"]
            m = meta_dicts_a[n]["meta"]
            if "error" in m:
                interpretations_vec_1.append(None)
                if id_emergencia not in sent_to_redo:
                    self.redo_queue.put(
                        (
                            id_emergencia,
                            "Error while running interpreter.create_interpretation_a",
                        )
                    )
                    sent_to_redo.add(id_emergencia)
                raise Exception(
                    "Error while running interpreter.create_interpretation_a:"
                    + type(m["error"])
                    + " "
                    + m["error"]
                )
            else:
                interpretations_vec_1.append(
                    {
                        "id_emergencia": id_emergencia,
                        "horario_contexto": horarios[n],
                        "horario_fim": float(time.time()),
                        "resultado": {
                            "descricao_breve": desc,
                            "outras_observacoes": interpretations_a[n][
                                "outras_observacoes"
                            ],
                            "ponto_de_referencia": interpretations_a[n][
                                "ponto_de_referencia"
                            ],
                            "nome_do_solicitante": interpretations_a[n][
                                "nome_do_solicitante"
                            ],
                        },
                        "tipo": "descricao_e_observacao",
                        "meta": meta_dicts_a[n]["meta"],
                    }
                )
        transcriptions_all = [x for x in transcriptions]
        transcriptions = [
            t
            for n, t in enumerate(transcriptions)
            if interpretations_vec_1[n] is not None
        ]
        contexts = [
            c for n, c in enumerate(contexts) if interpretations_vec_1[n] is not None
        ]
        interpretations_vec_1 = [t for t in interpretations_vec_1 if t is not None]
        self.send_interpretations(interpretations_vec_1)

        similares = self.inf_pipeline.find_similar_natures(descs, max_n=max_naturezas, 
            transcripts=transcriptions_all, bm25_weight=self.b25_weight)
        similares_vecs = [similares[desc]["sims_dict"] for desc in descs]
        meta_vecs = [similares[desc]["meta"] for desc in descs]
        interpretations_vec_2 = []
        for n, transcription in enumerate(transcriptions):
            desc = descs[n]
            interpretations_vec_2.append(
                {
                    "id_emergencia": contexts[n]["id_emergencia"],
                    "horario_contexto": horarios[n],
                    "horario_fim": float(time.time()),
                    "resultado": {
                        "classificacoes_provaveis": similares[desc],
                    },
                    "tipo": "lista_de_naturezas",
                    "meta": meta_vecs[n],
                }
            )
        self.send_interpretations(interpretations_vec_2)

        interpretations_b = self.inf_pipeline.create_interpretation_b(
            transcriptions, similares_vecs
        )
        meta_dicts_b = [m for _, m in interpretations_b]
        interpretations_b = [x for x, m in interpretations_b]

        interpretations_vec_3 = []
        results_final = []
        for n, transcription in enumerate(transcriptions):
            desc = descs[n]
            m = meta_dicts_b[n]["meta"]
            id_emergencia = contexts[n]["id_emergencia"]
            if "error" in m:
                interpretations_vec_3.append(None)
                msg = (
                    "Error while running interpreter.create_interpretation_b:"
                    + str(type(m["error"]))
                    + " "
                    + m["error"]
                )
                if id_emergencia not in sent_to_redo:
                    self.redo_queue.put((id_emergencia, msg))
                    sent_to_redo.add(id_emergencia)
                raise Exception(msg)
            else:
                interpretations_vec_3.append(
                    {
                        "id_emergencia": contexts[n]["id_emergencia"],
                        "horario_contexto": horarios[n],
                        "horario_fim": float(time.time()),
                        "resultado": {
                            "classificacao_decisiva": interpretations_b[n],
                        },
                        "tipo": "natureza_decisiva",
                        "meta": meta_dicts_b[n]["meta"],
                    }
                )

                results_final.append(
                    {
                        "id_emergencia": contexts[n]["id_emergencia"],
                        "transcricao": transcription,
                        "resultados": {
                            "classificacoes_provaveis": similares[desc],
                            "classificacao_decisiva": interpretations_b[n],
                            "descricao_breve": desc,
                            "outras_observacoes": interpretations_a[n][
                                "outras_observacoes"
                            ],
                            "ponto_de_referencia": interpretations_a[n][
                                "ponto_de_referencia"
                            ],
                            "nome_do_solicitante": interpretations_a[n][
                                "nome_do_solicitante"
                            ],
                        },
                        "meta": {
                            "a": meta_dicts_a[n]["meta"],
                            "b": meta_vecs[n],
                            "c": meta_dicts_b[n]["meta"],
                        },
                    }
                )
        interpretations_vec_3 = [t for t in interpretations_vec_3 if t is not None]
        self.send_interpretations(interpretations_vec_3)

        interpretations_env = self.inf_pipeline.create_envolvimentos(transcriptions)
        meta_dicts_env = [m for _, m in interpretations_env]
        interpretations_env = [x for x, m in interpretations_env]
        envolvimentos = [r["envolvimentos_emergencia"] for r in interpretations_env]
        envolvimentos_vec = []

        for n, transcription in enumerate(transcriptions_all):
            envolvimento = envolvimentos[n]
            id_emergencia = contexts[n]["id_emergencia"]
            m = meta_dicts_env[n]["meta"]
            if "error" in m:
                envolvimentos_vec.append(None)
                if id_emergencia not in sent_to_redo:
                    self.redo_queue.put(
                        (
                            id_emergencia,
                            "Error while running interpreter.create_envolvimentos",
                        )
                    )
                    sent_to_redo.add(id_emergencia)
                raise Exception(
                    "Error while running interpreter.create_envolvimentos:"
                    + type(m["error"])
                    + " "
                    + m["error"]
                )
            else:
                envolvimentos_vec.append(
                    {
                        "id_emergencia": id_emergencia,
                        "horario_contexto": horarios[n],
                        "horario_fim": float(time.time()),
                        "resultado": {
                            "envolvimentos": envolvimento,
                        },
                        "tipo": "envolvimentos",
                        "meta": meta_dicts_env[n]["meta"],
                    }
                )

        for x in interpretations_vec_3:
            id_emergencia = x["id_emergencia"]
            if id_emergencia not in sent_to_redo:
                self.redo_queue.put((id_emergencia, "interpretação_finalizada"))

        return results_final

    def process_emergencies(self, contexts, max_naturezas=None):
        if self.concurrency_mode == "async":
            return self.process_emergencies_async(
                contexts, max_naturezas=max_naturezas, join=False
            )
        elif self.concurrency_mode == "batch":
            return self.process_emergencies_batch(contexts, max_naturezas=max_naturezas)
        else:
            return None