
import json
import os
import sys
import time
import sqlite3 as sqlite
import multiprocessing as mp
from queue import Empty as QueueEmptyError
from runners.gline_interpreter import (extract_labels, extract_labels_parallel, load_gliner_cuda, GLINER_MODEL_NAME, 
    ner_labels, many_values)

import logging
logging.basicConfig(filename='/logs/main.log', level=logging.INFO)
logger = logging.getLogger(__name__)

def worker_sequence_workload(transcript_queue, label_set_name):
    sqlite_conn = sqlite.connect(os.environ["SQLITE_DB_PATH"])
    cursor = sqlite_conn.cursor()
    while True:
        last_not_analyzed_transcript = cursor.execute('''
            SELECT et.id, et.id_emergencia, et.start_time
            FROM emergency_transcripts et
            WHERE et.is_analyzed = FALSE
            AND et.id_emergencia IN (
                SELECT em.id FROM emergencies as em
                WHERE em.processing_running = FALSE
            )
            ORDER BY et.start_time ASC
            LIMIT 1
        ''').fetchone()
        if last_not_analyzed_transcript:
            _, id_emergencia, _ = last_not_analyzed_transcript
            transcript_parts_not_analyzed = cursor.execute(f'''
                SELECT et.id, et.part
                FROM emergency_transcripts et
                WHERE et.id_emergencia = {id_emergencia}
                ORDER BY et.start_time ASC
            ''').fetchall()

            transcript_ids = [t_id for t_id, _ in transcript_parts_not_analyzed]
            text_parts = [text_part for _, text_part in transcript_parts_not_analyzed]
            text = ' '.join(text_parts)

            new_task = (id_emergencia, transcript_ids, text, label_set_name)

            cursor.execute('''
                UPDATE emergencies SET processing_running = TRUE WHERE id = ?
            ''', (id_emergencia,))
            sqlite_conn.commit()

            transcript_queue.put(new_task)

        sleep(0.05)
    for _ in range(99):
        transcript_queue.put((None, None, 'STOP', None))
    sqlite_conn.close()

def worker_save_everything(result_queue):
    
    while os.path.exists(os.environ["SQLITE_DB_PATH"]):
        try:
            to_save = result_queue.get(timeout=0.1)
        except Exception as err:
            to_save = None
        if to_save is not None:
            if isinstance(to_save, dict):
                sqlite_conn = sqlite.connect(os.environ["SQLITE_DB_PATH"])
                cursor = sqlite_conn.cursor()
                if len(to_save['transcript_part_ids']) > 0:
                    
                    id_emergencia = to_save["id_emergencia"]
                    transcript_ids = ', '.join([str(x) for x in to_save['transcript_part_ids']])
                    try:
                        previous_ent_str = cursor.execute(f'''
                            SELECT json_data FROM emergencies
                            WHERE id = {id_emergencia}
                        ''').fetchone()[0]
                        if previous_ent_str == None:
                            previous_entities = {}
                        else:
                            previous_entities = json.loads(previous_ent_str)
                    except Exception as err:
                        print(err, file=sys.stderr)
                        print(previous_ent_str)
                        print(err.with_traceback(None), file=sys.stderr)
                        pass
                    cursor.execute(f'''
                        UPDATE emergency_transcripts SET is_analyzed = TRUE 
                        WHERE id IN ({transcript_ids})
                    ''')
                    print(f'worker_save_everything: {to_save['meta']}')

                    for ent_key in to_save['entities']:
                        new_values = to_save['entities'][ent_key]
                        previous = []
                        if ent_key in previous_entities:
                            previous = previous_entities[ent_key]
                        value_points = {}
                        for value, points in new_values + previous:
                            if value in value_points:
                                if points > value_points[value]:
                                    value_points[value] = points
                            else:
                                value_points[value] = points
                        updated_values = [(key, p) for key, p in value_points.items()]
                        updated_values.sort(key= lambda xy: xy[1], reverse=True)
                        to_save['entities'][ent_key] = updated_values
                    
                    json_data_str = json.dumps(to_save["entities"])
                    cursor.execute(f'''
                        UPDATE emergencies SET json_data = '{json_data_str}' 
                        WHERE id = {id_emergencia}
                    ''')
                    cursor.execute('''
                        UPDATE emergencies SET processing_running = FALSE WHERE id = ?
                    ''', (id_emergencia,))
                else:
                    id_emergencia = to_save["id_emergencia"]
                    cursor.execute('''
                        UPDATE emergencies SET processing_running = FALSE WHERE id = ?
                    ''', (id_emergencia,))
                sqlite_conn.commit()
                sqlite_conn.close()
            else:
                print('worker_save_everything: Unkhown format to save')
                print(to_save)
                print(type(to_save))
        time.sleep(0.05)

def gliner_worker_process(gliner_mname, transcript_queue, result_queue, hardware_config, batch_size: int, labels_set):
    # Cada worker carrega seu próprio modelo
    #calls_being_processed_lockset[worker_index] = -1
    print('gliner_worker_process: Starting model loading')
    if hardware_config == 'cuda':
        gliner_model, word_splitter = load_gliner_cuda(gliner_mname)
    else:
        gliner_model, word_splitter = load_gliner_cuda(gliner_mname, use_cuda=False)
    max_loading_time = 0.1
    print('Loaded model')
    while True:
        to_process_list = []
        loading_started = time.time()
        loading_time = time.time() - loading_started
        while len(to_process_list) < batch_size and loading_time < max_loading_time:
            #tp = id_emergencia, horario_contexto, transcript
            try:
                tp = transcript_queue.get(timeout=0.01)
                to_process_list.append(tp)
            except QueueEmptyError as err:
                break
            loading_time = time.time() - loading_started
        if len(to_process_list) > 0:
            try:
                label_list = ner_labels[labels_set]
                transcripts = [tp[2] for tp in to_process_list]
                should_stop = "STOP" in transcripts
                
                if should_stop:
                    to_process_list = [tp for tp in to_process_list if tp[2] != 'STOP']
                    transcripts = [tp[2] for tp in to_process_list]

                if len(to_process_list) > 0:
                    results, metas = extract_labels_parallel(transcripts, gliner_model, word_splitter,
                        labels=label_list, top_to_keep=3, not_keep_top=many_values, batch_size=batch_size)
                    #result, response_meta = extract_labels(transcript, gliner_model, word_splitter,
                    #    labels=label_list, top_to_keep=3, not_keep_top=many_values)
                    for tp, result, response_meta in zip(to_process_list, results, metas):
                        print(f'result: {result}')
                        id_emergencia, horario_contexto, _ = tp
                        result_queue.put({
                            "id_emergencia": id_emergencia,
                            "horario_contexto": horario_contexto,
                            "entities": result,
                            "horario_fim": float(time.time()),
                            "meta": response_meta
                        })
                
                if should_stop:
                    break
            except Exception as err:
                print('Error processing transcript, removing processing_running flag')
                print('transcript: {transcript}')
                print('Erro: {err}')
                print(err.with_traceback(None))
                time.sleep(0.1)
                result_queue.put({
                    "id_emergencia": id_emergencia,
                    "horario_contexto": horario_contexto,
                    "entities": {},
                    "horario_fim": float(time.time()),
                    "meta": {'error': str(err)}
                })

        time.sleep(0.05)

class GlinerRunner:

    def __init__(self, gliner_mname: str = GLINER_MODEL_NAME, 
            n_gliner_workers: int = 1, 
            hardware_config: str = 'cpu',
            label_set_name: str = 'emergency_gliclass',
            batch_size=8):
        transcript_queue = mp.Queue()
        result_queue = mp.Queue()
        #calls_being_processed_lockset = mp.Array('i', range(n_gliner_workers))
        workers = []

        #stop_flag = mp.Value('H', 0)
        self.label_set_name = label_set_name

        for worker_index in range(n_gliner_workers):
            print('GlinerRunner: creating process')
            p = mp.Process(
                target=gliner_worker_process, 
                args=(gliner_mname, transcript_queue, result_queue, hardware_config, batch_size, self.label_set_name)
            )
            print('GlinerRunner: starting process')
            p.start()
            print('GlinerRunner: started process')
            workers.append(p)
        
        saver_worker = mp.Process(
            target=worker_save_everything, 
            args=(result_queue,)
        )

        saver_worker.start()

    def add_to_queue(self, id_emergencia, horario_contexto, transcription):
        self.transcript_queue.put((id_emergencia, horario_contexto, transcription))