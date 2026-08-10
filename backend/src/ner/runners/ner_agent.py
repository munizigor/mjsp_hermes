import json
import time
import sqlite3 as sqlite
import os
import multiprocessing as mp
from queue import Empty as QueueEmptyError

from runners.gliner_utilities import GLINER_MODEL_NAME

print("Iniciando modulo do NERAgent")


def ultimos_horarios_de_transcricao(sqlite_path):
    query = """SELECT 
        e.id AS emergency_id,
        MAX(et.horario_de_transcricao) AS ultimo_horario_de_transcricao
    FROM 
        emergencies e
    LEFT JOIN 
        emergency_transcripts et
    ON 
        e.id = et.id_emergencia
    GROUP BY 
        e.id;"""

    sqlite_conn = sqlite.connect(sqlite_path)
    cursor = sqlite_conn.cursor()
    cursor.execute(query)
    horarios_trans = cursor.fetchall()
    if horarios_trans == None:
        results = []
    else:
        results = [
            {"id_emergencia": x[0], "ultimo_horario_de_transcricao": x[1]}
            for x in horarios_trans
        ]
    sqlite_conn.close()

    return results


def horario_ultima_ner(sqlite_path, em_id):
    # listar linhas de resultados_inferencia associadas com a emergência
    sqlite_conn = sqlite.connect(sqlite_path)
    cursor = sqlite_conn.cursor()
    cursor.execute(
        """
        SELECT horario_contexto FROM resultados_inferencia
        WHERE tipo_de_inferencia = 'ner'
            AND id_emergencia = ?
    """,
        (em_id,),
    )
    horario_contexto = cursor.fetchall()
    sqlite_conn.close()
    if horario_contexto == None:
        horario_contexto = []
    if len(horario_contexto) > 0:
        ultimo_ponto = max([x[0] for x in horario_contexto])
        # desde_ner = time.time() - ultimo_ponto
        return ultimo_ponto
    else:
        return 0


def emergencias_nao_analizadas(sqlite_path):
    horarios_ultimas_transcricoes = ultimos_horarios_de_transcricao(sqlite_path)
    emergencias = []
    for h in horarios_ultimas_transcricoes:
        id_emergencia = int(h["id_emergencia"])
        if h["ultimo_horario_de_transcricao"] != None:
            ultimo_horario = float(h["ultimo_horario_de_transcricao"])
            horario_ner = horario_ultima_ner(sqlite_path, id_emergencia)
            if horario_ner < ultimo_horario:
                emergencias.append([id_emergencia, horario_ner])

    emergencias.sort(key=lambda x: x[1])
    id_emergencias = [x[0] for x in emergencias]
    return id_emergencias


def coletar_transcricao(sqlite_path, em_id):
    sqlite_conn = sqlite.connect(sqlite_path)
    cursor = sqlite_conn.cursor()
    cursor.execute(
        """
        SELECT et.part, et.horario_de_transcricao 
        FROM emergency_transcripts et
        WHERE et.id_emergencia = ?
        ORDER BY et.start_time ASC
    """,
        (em_id,),
    )
    transcript_parts = cursor.fetchall()
    sqlite_conn.close()
    if transcript_parts == None:
        transcript_parts = []
    if len(transcript_parts) > 0:
        context = " ".join([part[0] for part in transcript_parts])
        horario = max([float(part[1]) for part in transcript_parts])
        # context = {'transcription': context, 'id_emergencia': em_id}
        return context, horario
    else:
        return "Sem transcrição", 0


def worker_sequence_workload(transcript_queue, sqlite_path, running_flag, redo_queue):
    running_flag.value = 1
    print("Esperando para realizar NERs...")
    redo_timeout = 0.05
    processando = set()
    try:
        while running_flag.value == 1:
            ner_feita = False
            em_ids = emergencias_nao_analizadas(sqlite_path)
            not_processing = [em for em in em_ids if em not in processando]
            # print(f'Processando: {processando}')
            # print(f'Não processando, mas precisam: {not_processing}')

            if len(not_processing) > 0:
                para_processar = []
                tempo_atual = time.time()
                for em_id in not_processing:
                    texto, horario = coletar_transcricao(sqlite_path, em_id)
                    para_processar.append(
                        {
                            "id_emergencia": em_id,
                            "horario_ultima": horario,
                            "delay": tempo_atual - horario,
                            "transcription": texto,
                        }
                    )
                    processando.add(em_id)

                print("\nEmergencias com transcricao sem NER:")
                for em in para_processar:
                    print(
                        "Emergencia: "
                        + json.dumps(em, indent=4, ensure_ascii=False)
                        + "\n"
                    )

                para_processar.sort(key=lambda x: (-x["delay"], x["horario_ultima"]))

                for tp in para_processar:
                    transcript_queue.put(
                        (tp["id_emergencia"], tp["horario_ultima"], tp["transcription"])
                    )
                    ner_feita = True

            try:
                redo_item = redo_queue.get(timeout=redo_timeout)
                if redo_item in processando:
                    processando.remove(redo_item)
                print(f"worker_sequence_workload: Re-enfileirando para NER {redo_item}")
            except QueueEmptyError as err:
                pass

            if not ner_feita:
                time.sleep(0.25)  # waiting for input

        for _ in range(100):
            transcript_queue.put((0, 0, "STOP"))
    except Exception as err:
        running_flag.value = 0
        for _ in range(100):
            transcript_queue.put((0, 0, "STOP"))
        raise (err)


class NERAgent:
    def __init__(
        self,
        sqlite_path: str,
        gliner_mname: str = GLINER_MODEL_NAME,
        n_gliner_workers: int = 1,
        hardware_config: str = "cpu",
        label_set_name: str = "emergency_gliclass",
        batch_size=5,
        max_tokens=150,
        n_cpus=5,
    ):
        # self.interpreter = interpreter
        self.sqlite_path = sqlite_path
        self.stop_flag = False
        self.running = False
        # self.min_transcricao = min_transcricao

        transcript_queue = mp.Queue()
        result_queue = mp.Queue()
        # calls_being_processed_lockset = mp.Array('i', range(n_gliner_workers))
        workers = []

        # stop_flag = mp.Value('H', 0)
        self.label_set_name = label_set_name

        running_flag = mp.Value("H", 0)
        self.running_flag = running_flag

        redo_queue = mp.Queue()
        if hardware_config in ["cpu", "cuda"]:
            from runners.local_gliner import (
                gliner_inference_saver as local_gliner_inference_saver,
            )
            from runners.local_gliner import (
                gliner_worker_process as local_gliner_worker_process,
            )

            gliner_worker_process = local_gliner_worker_process
            gliner_inference_saver = local_gliner_inference_saver
        else:
            gliner_inference_saver = None
            if hardware_config in ["triton-server"]:
                from runners.triton import triton_worker_process as worker_process_func
            elif hardware_config in ["gcp-endpoint"]:
                from runners.gcp_endpoint import (
                    gcp_worker_process as worker_process_func,
                )
            elif hardware_config in ["azure-endpoint"]:
                from runners.azure_endpoint import (
                    azure_worker_process as worker_process_func,
                )
            elif hardware_config in ["vllm-api"]:
                from runners.vllm_endpoint import (
                    vllm_worker_process as worker_process_func,
                )
            else:
                worker_process_func = None

            gliner_worker_process = worker_process_func

        for worker_index in range(n_gliner_workers):
            print("GlinerRunner: creating process")
            p = mp.Process(
                target=gliner_worker_process,
                args=(
                    gliner_mname,
                    transcript_queue,
                    redo_queue,
                    result_queue,
                    hardware_config,
                    batch_size,
                    self.label_set_name,
                    max_tokens,
                    n_cpus,
                ),
            )
            print("GlinerRunner: starting process")
            p.start()
            print("GlinerRunner: started process")
            workers.append(p)

        if gliner_inference_saver is not None:
            saver_worker = mp.Process(
                target=gliner_inference_saver, args=(result_queue, redo_queue)
            )
            saver_worker.start()

        sequencer_worker = mp.Process(
            target=worker_sequence_workload,
            args=(transcript_queue, sqlite_path, running_flag, redo_queue),
        )
        sequencer_worker.start()

    def stop(self, max_wait: float = 8.0):
        print("Stopping NER agent...")
        self.running_flag.value = 0
        while self.running_flag.value == 1 and max_wait > 0:
            time.sleep(0.5)
            max_wait -= 0.5
