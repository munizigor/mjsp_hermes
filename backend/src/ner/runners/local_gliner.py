import json
import time
import sqlite3 as sqlite
import os
from queue import Empty as QueueEmptyError

from runners.gliner_utilities import (load_gliner_cuda, 
    extract_labels_parallel, ner_labels, many_values)

def gliner_inference_saver(result_queue, redo_queue):

    #cols = '''id_emergencia, horario_contexto, horario_fim, 
    #        tipo_de_inferencia, resultado, duracao_inferencia, 
    #        duracao_outros_processamentos, input_tokens, output_tokens, 
    #        modelo_utilizado'''
    '''    for a in interpretations:
            id_emergencia = int(a['id_emergencia'])
            horario_contexto = a['horario_contexto']
            horario_fim = a['horario_fim']
            tipo_de_inferencia = a['tipo']
            resultado = json.dumps(a['resultado'], ensure_ascii=False)
            duracao_inferencia = a['meta'].get('processing_time', None)
            duracao_outros_processamentos = a['meta'].get('no_gpu_time', None)
            input_tokens = a['meta'].get('input_tokens', None)
            output_tokens = a['meta'].get('output_tokens', None)
            modelo_utilizado = a['meta'].get('model_name', None)
            tuples.append((id_emergencia, horario_contexto, horario_fim, 
                           tipo_de_inferencia, resultado, duracao_inferencia, 
                            duracao_outros_processamentos, input_tokens, output_tokens, 
                            modelo_utilizado))
    '''

    while os.path.exists(os.environ["SQLITE_DB_PATH"]):
        cols = '''id_emergencia, horario_contexto, horario_fim, 
            tipo_de_inferencia, resultado, duracao_inferencia, 
            duracao_outros_processamentos, input_tokens, output_tokens, 
            modelo_utilizado'''

        try:
            to_save = result_queue.get(timeout=0.05)
        except QueueEmptyError as err:
            to_save = None
        if to_save is not None:
            if isinstance(to_save, dict):
                sqlite_conn = sqlite.connect(os.environ["SQLITE_DB_PATH"])
                cursor = sqlite_conn.cursor()
                if 'error' not in to_save['meta']:
                    id_emergencia = to_save["id_emergencia"]
                    
                    print(f'worker_save_everything: {to_save['meta']}')
                    
                    id_emergencia = int(to_save['id_emergencia'])
                    horario_contexto = to_save['horario_contexto']
                    horario_fim = to_save['horario_fim']
                    tipo_de_inferencia = 'ner'
                    resultado = json.dumps(to_save['entities'], ensure_ascii=False)
                    duracao_inferencia = to_save['meta'].get('processing_time', None)
                    duracao_outros_processamentos = to_save['meta'].get('no_gpu_time', None)
                    input_tokens = to_save['meta'].get('input_tokens', None)
                    output_tokens = to_save['meta'].get('output_tokens', None)
                    modelo_utilizado = to_save['meta'].get('model_name', None)
                    new_line = (id_emergencia, horario_contexto, horario_fim, 
                                tipo_de_inferencia, resultado, duracao_inferencia, 
                                    duracao_outros_processamentos, input_tokens, output_tokens, 
                                    modelo_utilizado)
                    cursor.execute(f'''INSERT INTO resultados_inferencia 
                        ({cols}) VALUES (?,?,?,?,?,?,?,?,?,?)''', new_line)
                else:
                    id_emergencia = to_save["id_emergencia"]
                    print('gliner_inference_saver: Error in inference, not saving to DB')
                    print(to_save)
                    
                sqlite_conn.commit()
                sqlite_conn.close()
                redo_queue.put(int(to_save['id_emergencia']))
            else:
                print('gliner_inference_saver: Unkhown format to save')
                print(to_save)
                print(type(to_save))
        time.sleep(0.05)

def gliner_worker_process(gliner_mname, transcript_queue, redo_queue, result_queue, 
                          hardware_config, batch_size: int, labels_set, max_tokens: int, 
                          n_cpus: int):
    import torch
    torch.set_num_threads(n_cpus)
    # Cada worker carrega seu próprio modelo
    #calls_being_processed_lockset[worker_index] = -1
    print('gliner_worker_process: Starting model loading')
    if hardware_config == 'cuda':
        gliner_model, word_splitter = load_gliner_cuda(gliner_mname)
    else:
        gliner_model, word_splitter = load_gliner_cuda(gliner_mname, use_cuda=False)
    max_loading_time = 0.1
    torch.set_num_threads(n_cpus)
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
                        labels=label_list, top_to_keep=3, not_keep_top=many_values, batch_size=batch_size, 
                        max_tokens=max_tokens)
                    
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
                print('Error processing transcript')
                print(f'Erro: {err}')
                print(err.with_traceback(None))

                for tp in to_process_list:
                    print(id_emergencia, horario_contexto)
                    id_emergencia, horario_contexto, _ = tp
                    time.sleep(0.1)
                    result_queue.put({
                        "id_emergencia": id_emergencia,
                        "horario_contexto": horario_contexto,
                        "entities": {},
                        "horario_fim": float(time.time()),
                        "meta": {'error': str(err)}
                    })
                    redo_queue.put(id_emergencia)

        time.sleep(0.05)