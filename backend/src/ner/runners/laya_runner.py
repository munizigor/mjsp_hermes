"""
Runner experimental de NER usando o Laya (modelo de decisão "System 1",
não autorregressivo) para as classificações booleanas da ocorrência.

O Laya responde perguntas tipadas (choice, score, noul) com probabilidades
calibradas em uma única passada, mas não extrai trechos de texto. Por isso
este runner usa o Laya apenas para o bloco EmergencyClassifications
(fato ocorrendo, autor no local, autor armado, feridos com risco de morte,
risco de tumulto e Lei Maria da Penha). As entidades (endereço, nome etc.)
podem vir do GLiNER local ou ficar vazias, conforme LAYA_ENTITIES_BACKEND.

Ativação: "hardware_config": "laya" na seção "ner" do config.json.

Variáveis de ambiente (todas opcionais):
    LAYA_MODEL             repositório no Hugging Face ou caminho local do
                           checkpoint (padrão: convaiinnovations/laya)
    LAYA_SUBFOLDER         subpasta do checkpoint (padrão: multilingual;
                           vazio para usar a raiz, que é o modelo em inglês)
    LAYA_DEVICE            cpu, cuda, mps ou xpu (padrão: detecção automática
                           do próprio Laya)
    LAYA_MAX_LEN           limite de tokens da entrada (padrão: o do próprio
                           checkpoint, 1024 no multilingual, que aceita até 8192)
    LAYA_THRESHOLD         probabilidade a partir da qual a resposta é "Sim"
                           (padrão: 0.5)
    LAYA_ENTITIES_BACKEND  "none" (padrão) ou "gliner". Com "gliner", as
                           entidades são extraídas pelo GLiNER local, usando
                           o modelo definido em "gliner_mname".
"""

import json
import os
import sqlite3 as sqlite
import sys
import time
from queue import Empty as QueueEmptyError

from runners.gliner_utilities import (
    EmergencyClassifications,
    EmergencyEntities,
    many_values,
    ner_labels,
    schema_key_translations,
)

LAYA_MODEL = os.getenv("LAYA_MODEL", "convaiinnovations/laya")
LAYA_SUBFOLDER = os.getenv("LAYA_SUBFOLDER", "multilingual")
LAYA_DEVICE = os.getenv("LAYA_DEVICE", "")
LAYA_MAX_LEN = os.getenv("LAYA_MAX_LEN", "").strip()
LAYA_THRESHOLD = float(os.getenv("LAYA_THRESHOLD", "0.5"))
LAYA_ENTITIES_BACKEND = os.getenv("LAYA_ENTITIES_BACKEND", "none").strip().lower()

# Contexto comum às perguntas, para o modelo saber de que tipo de texto se trata.
CONTEXTO_PERGUNTAS = (
    "O texto é a transcrição automática de uma ligação para um serviço de "
    "emergência (polícia, bombeiros ou atendimento médico). "
)


def save_ner_to_db(to_save: dict, redo_queue):
    """
    Mesmo comportamento de runners.gcp_endpoint.save_ner_to_db. Fica replicado
    aqui porque aquele módulo encerra o processo na importação quando
    GEMINI_API_KEY não está definida, e este runner não depende do Gemini.
    """
    redo = True
    meta = to_save["meta"]
    if "error" in meta:
        print("laya_runner: erro na inferência, nada salvo no banco")
        print(to_save)
    elif int(to_save["id_emergencia"]) > 0 and int(to_save["horario_contexto"]) > 0:
        sqlite_conn = sqlite.connect(os.environ["SQLITE_DB_PATH"])
        sqlite_conn.execute(
            """INSERT INTO resultados_inferencia
            (id_emergencia, horario_contexto, horario_fim,
            tipo_de_inferencia, resultado, duracao_inferencia,
            duracao_outros_processamentos, input_tokens, output_tokens,
            modelo_utilizado) VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                int(to_save["id_emergencia"]),
                to_save["horario_contexto"],
                to_save["horario_fim"],
                "ner",
                json.dumps(to_save["entities"], ensure_ascii=False),
                meta.get("processing_time", None),
                meta.get("no_gpu_time", None),
                meta.get("input_tokens", None),
                meta.get("output_tokens", None),
                meta.get("model_name", None),
            ),
        )
        sqlite_conn.commit()
        sqlite_conn.close()
    else:
        print(f"It looks like {to_save} was a test", file=sys.stderr)
        redo = False

    if redo:
        redo_queue.put(int(to_save["id_emergencia"]))


def build_laya_questions() -> dict:
    """
    Monta as perguntas do Laya a partir das descrições do schema
    EmergencyClassifications, para que o Laya e os runners baseados em LLM
    respondam exatamente às mesmas perguntas.
    """
    questions = {}
    for field_name, field in EmergencyClassifications.model_fields.items():
        descricao = field.description or field_name.replace("_", " ")
        questions[field_name] = {
            "type": "noul",
            "instructions": CONTEXTO_PERGUNTAS + descricao + ".",
        }
    return questions


LAYA_QUESTIONS = build_laya_questions()


def load_laya_agent():
    import laya

    kwargs = {}
    if LAYA_SUBFOLDER:
        kwargs["subfolder"] = LAYA_SUBFOLDER
    if LAYA_DEVICE:
        kwargs["device"] = LAYA_DEVICE

    print(f"laya_runner: carregando {LAYA_MODEL} (subfolder={LAYA_SUBFOLDER or '-'})")
    agent = laya.load(LAYA_MODEL, **kwargs)

    # O checkpoint multilingual vem com limite de 1024 tokens; transcrições
    # mais longas são truncadas se o limite não for ampliado.
    if LAYA_MAX_LEN:
        try:
            agent.cfg["max_len"] = int(LAYA_MAX_LEN)
        except Exception as err:
            print(f"laya_runner: não foi possível ajustar max_len: {err}", file=sys.stderr)

    print("laya_runner: modelo carregado")
    return agent


def empty_entities() -> dict:
    return {key: [] for key in EmergencyEntities.model_fields.keys()}


def classify_with_laya(agent, transcripts: list, batch_size: int):
    """
    Retorna, para cada transcrição, um dict {campo: probabilidade de "Sim"},
    além do tempo total da inferência.
    """
    start = time.time()
    if len(transcripts) == 1:
        outputs = [agent.predict(transcripts[0], LAYA_QUESTIONS)]
    else:
        outputs = agent.predict_batch(
            transcripts, LAYA_QUESTIONS, batch_size=max(1, batch_size)
        )
    duration = time.time() - start

    probabilities = []
    input_tokens = []
    for out in outputs:
        answers = out.get("answers", {})
        probs = {}
        for field_name in LAYA_QUESTIONS.keys():
            answer = answers.get(field_name, {})
            p = answer.get("noul", None)
            probs[field_name] = float(p) if p is not None else None
        probabilities.append(probs)
        usage = out.get("usage", {}) or {}
        input_tokens.append(usage.get("input_tokens", None))
    return probabilities, input_tokens, duration


def probabilities_to_response(probs: dict) -> dict:
    """
    Converte as probabilidades no formato que a API e o frontend já esperam:
    chave traduzida (ex.: "Autor Do Fato Armado ?") com valor 1.0 ou 0.0.
    """
    response = {}
    for field_name, p in probs.items():
        key = schema_key_translations.get(field_name, field_name)
        if p is None:
            continue
        response[key] = 1.0 if p >= LAYA_THRESHOLD else 0.0
    return response


def laya_worker_process(
    gliner_mname,
    transcript_queue,
    redo_queue,
    result_queue,
    hardware_config,
    batch_size: int,
    labels_set,
    max_tokens: int,
    n_cpus: int,
):
    print("Starting laya worker process for NER")
    try:
        import torch

        torch.set_num_threads(n_cpus)
    except Exception:
        pass

    try:
        agent = load_laya_agent()
    except ImportError:
        print(
            "laya_runner: pacote 'laya' não instalado. Construa a imagem do NER com "
            "INSTALL_LAYA=true (veja backend/README.md).",
            file=sys.stderr,
        )
        quit(1)

    gliner_model = None
    word_splitter = None
    if LAYA_ENTITIES_BACKEND == "gliner":
        from runners.gliner_utilities import extract_labels_parallel, load_gliner_cuda

        use_cuda = LAYA_DEVICE == "cuda"
        gliner_model, word_splitter = load_gliner_cuda(gliner_mname, use_cuda=use_cuda)
    elif LAYA_ENTITIES_BACKEND != "none":
        print(
            f"laya_runner: LAYA_ENTITIES_BACKEND desconhecido '{LAYA_ENTITIES_BACKEND}', "
            "usando 'none'",
            file=sys.stderr,
        )

    model_name = f"laya:{LAYA_MODEL}" + (f"/{LAYA_SUBFOLDER}" if LAYA_SUBFOLDER else "")
    max_loading_time = 0.1

    while True:
        to_process_list = []
        loading_started = time.time()
        while (
            len(to_process_list) < max(1, batch_size)
            and time.time() - loading_started < max_loading_time
        ):
            try:
                to_process_list.append(transcript_queue.get(timeout=0.01))
            except QueueEmptyError:
                break

        if len(to_process_list) == 0:
            time.sleep(0.05)
            continue

        should_stop = any(tp[2] == "STOP" for tp in to_process_list)
        to_process_list = [tp for tp in to_process_list if tp[2] != "STOP"]

        if len(to_process_list) > 0:
            proc_start = time.time()
            transcripts = [tp[2] for tp in to_process_list]
            try:
                probabilities, input_tokens, laya_duration = classify_with_laya(
                    agent, transcripts, batch_size
                )

                entities_per_call = [empty_entities() for _ in transcripts]
                gliner_duration = 0.0
                if gliner_model is not None:
                    gliner_start = time.time()
                    entities_per_call, _ = extract_labels_parallel(
                        transcripts,
                        gliner_model,
                        word_splitter,
                        labels=ner_labels["emergency_basic"],
                        top_to_keep=3,
                        not_keep_top=many_values,
                        batch_size=max(1, batch_size),
                        max_tokens=max_tokens,
                    )
                    gliner_duration = time.time() - gliner_start

                processing_time = laya_duration + gliner_duration
                total_time = time.time() - proc_start
                n = len(to_process_list)

                for tp, probs, n_tokens, entities in zip(
                    to_process_list, probabilities, input_tokens, entities_per_call
                ):
                    id_emergencia, horario_contexto, _ = tp
                    response_json = dict(entities)
                    response_json.update(probabilities_to_response(probs))
                    print(
                        f"laya_runner: emergência {id_emergencia} probabilidades: "
                        + json.dumps(probs, ensure_ascii=False)
                    )
                    to_save = {
                        "id_emergencia": id_emergencia,
                        "horario_contexto": horario_contexto,
                        "entities": response_json,
                        "horario_fim": float(time.time()),
                        "meta": {
                            "processing_time": processing_time / n,
                            "no_gpu_time": (total_time - processing_time) / n,
                            "input_tokens": n_tokens,
                            "output_tokens": 0,
                            "model_name": model_name,
                        },
                    }
                    save_ner_to_db(to_save, redo_queue)
            except Exception as err:
                print(f"laya_runner: erro na inferência: {err}", file=sys.stderr)
                for tp in to_process_list:
                    id_emergencia, horario_contexto, _ = tp
                    save_ner_to_db(
                        {
                            "id_emergencia": id_emergencia,
                            "horario_contexto": horario_contexto,
                            "entities": None,
                            "horario_fim": float(time.time()),
                            "meta": {
                                "error": str(err),
                                "model_name": model_name,
                            },
                        },
                        redo_queue,
                    )

        if should_stop:
            break
