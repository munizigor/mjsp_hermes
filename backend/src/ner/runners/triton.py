import json
import time
import sqlite3 as sqlite
import os
import multiprocessing as mp
from queue import Empty as QueueEmptyError
import threading
import requests

import numpy as np
import tritonclient.http as http_client

from runners.gliner_utilities import (ner_labels, many_values,
    values_to_ignore, env_vals)

if not os.getenv("TRITON_SERVER_URL_NER"):
    print('Environment variable TRITON_SERVER_URL_NER not set, quitting')
    quit(1)
if not os.getenv("TRITON_SERVER_NER_TIMEOUT_SECS"):
    os.environ["TRITON_SERVER_NER_TIMEOUT_SECS"] = "5"

timeout_secs = int(os.environ["TRITON_SERVER_NER_TIMEOUT_SECS"])

def save_ner_to_db(to_save: dict, redo_queue: mp.Queue):
    '''
    Args:
    to_save: Dictionary with fields:
        meta:
            processing_time;
            no_gpu_time;
            input_tokens;
            output_tokens;
            model_name;
        id_emergencia;
        horario_contexto;
        horario_fim;
        entities;
    '''
    cols = '''id_emergencia, horario_contexto, horario_fim, 
            tipo_de_inferencia, resultado, duracao_inferencia, 
            duracao_outros_processamentos, input_tokens, output_tokens, 
            modelo_utilizado'''
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

'''def call_triton_server_thread(params_dict: dict, redo_queue: mp.Queue, TRITON_SERVER_URL_NER: str):

    try:
        client = http_client.InferenceServerClient(url=TRITON_SERVER_URL_NER,
            connection_timeout=timeout_secs,
            network_timeout=timeout_secs)

        #test connection:
        if not client.is_server_live():
            print('call_triton_server_thread: Triton server is not live')
            redo_queue.put(int(params_dict["id_emergencia"]))
            quit(1)
        if not client.is_server_ready():
            print('call_triton_server_thread: Triton server is not ready')
            redo_queue.put(int(params_dict["id_emergencia"]))
            quit(1)
        if not client.is_model_ready(params_dict["gliner_mname"]):
            print('call_triton_server_thread: Triton model is not ready')
            redo_queue.put(int(params_dict["id_emergencia"]))
            quit(1)

    except Exception as err:
        print('call_triton_server_thread: Unable to create Triton client')
        print(TRITON_SERVER_URL_NER, timeout_secs)
        print(err)
        print(err.with_traceback(None))
        time.sleep(5)
        redo_queue.put(int(params_dict["id_emergencia"]))
        quit(1)
    
    inputs = [
        http_client.InferInput(
            "PROMPT", [1, 1], "BYTES"
        ),
        http_client.InferInput(
            "LABEL_LIST", [1, 1], "BYTES"
        )
    ]
    transcript = params_dict["transcript"]
    #inputs[0].set_data_from_numpy(np.array([transcript.encode('utf-8')]))
    inputs[0].set_data_from_numpy(np.array([[transcript.encode('utf-8')]]))
    #inputs[0].set_data_from_numpy(np.array([[transcript.encode('utf-8')]], dtype=object))
    label_list = params_dict["labels"]
    if type(label_list) == list:
        label_list_str = ','.join(label_list)
    else:
        label_list_str = json.dumps(label_list, ensure_ascii=False, indent=2)
        label_list_str = label_list_str.replace('"', '\"')
    #inputs[1].set_data_from_numpy(np.array([label_list_str.encode('utf-8')]))
    inputs[1].set_data_from_numpy(np.array([[label_list_str.encode('utf-8')]]))
    print("clf schema used:")
    print(json.dumps(label_list, ensure_ascii=False).replace('"', '\\"'))
    print("transcript used:")
    print(transcript)
    #inputs[1].set_data_from_numpy(np.array([[label_list_str.encode('utf-8')]], dtype=object))

    try:
        results = client.infer(model_name=params_dict["gliner_mname"], inputs=inputs, 
            timeout=timeout_secs*1000)
    except Exception as err:
        print('call_triton_server_thread: Error during inference call to Triton server')
        print(err)
        print(err.with_traceback(None))
        print('Inputs were:')
        for a in inputs:
            print(a)
        print('Model:', params_dict["gliner_mname"])
        time.sleep(1)
        redo_queue.put(int(params_dict["id_emergencia"]))
        return
    
    id_emergencia = params_dict["id_emergencia"]
    horario_contexto = params_dict["horario_contexto"]

    response_json_str = results.as_numpy("ENTITIES_JSON")[0].decode('utf-8')
    print("Raw NER result:", response_json_str)
    #response_json_str = response_json_str.replace("\\n", "").replace('\\"', '"')
    #print("Cleaned NER result:", response_json_str)
    response_json = json.loads(response_json_str)
    json_keys = list(response_json.keys())
    print("JSON Keys:", json_keys)
    print("JSON Values:", response_json[json_keys[0]])
    if len(json_keys) == 1 and len(response_json[json_keys[0]]) == 0:
        response_json_str = response_json[json_keys[0]]
        response_json = json.loads(response_json_str)
        print("Cleaned NER result:", response_json)
    else:
        print("NER result is already in the correct format")
    response_meta = json.loads(results.as_numpy("META_INFO")[0].decode('utf-8'))

    for label in values_to_ignore.keys():
        if label in response_json:
            not_ignore = []
            for value, score in response_json[label]:
                if value.lower() not in values_to_ignore[label]:
                    not_ignore.append([value, score])
            response_json[label] = not_ignore
    
    top_to_keep = params_dict['top_to_keep']
    not_keep_top = params_dict['not_keep_top']
    for label in response_json.keys():
        if type(response_json[label]) == list:
            response_json[label].sort(key=lambda x: x[1], reverse=True)
            if label not in not_keep_top and len(response_json[label]) > top_to_keep:
                response_json[label] = response_json[label][:top_to_keep]  # Keep only the top 3 values for each label
    
    to_save = {
        "id_emergencia": id_emergencia,
        "horario_contexto": horario_contexto,
        "entities": response_json,
        "horario_fim": float(time.time()),
        "meta": response_meta
    }
    print("New NER result:", json.dumps(to_save, ensure_ascii=False, indent=3))
    save_ner_to_db(to_save, redo_queue)'''

def call_triton_server_thread(params_dict: dict, redo_queue: mp.Queue, TRITON_SERVER_URL_NER: str):
    '''
    After getting the results, calls the saver function.
    
    Args:
    params_dict: dict with the following keys:
        - gliner_mname: name of the gliner model
        - labels: list of labels
        - top_to_keep: number of top labels to keep
        - not_keep_top: list of labels to not keep
        - batch_size: batch size
        - max_tokens: max tokens
        - transcript: transcript
        - id_emergencia:
        - horario_contexto: 
    
    '''
    
    transcript = params_dict["transcript"]
    schema_dict = params_dict["labels"]
    schema_as_str = json.dumps(schema_dict)
    payload = {
        "inputs": [
            {
                "name": "PROMPT",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [transcript]
            },
            {
                "name": "LABEL_LIST",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [schema_as_str] # Passando como string dentro da lista
            }
        ]
    }

    print("transcript used:")
    print(transcript)
    #inputs[1].set_data_from_numpy(np.array([[label_list_str.encode('utf-8')]], dtype=object))

    try:
        base_url = TRITON_SERVER_URL_NER
        if not base_url.startswith("http"):
            base_url = f"http://{base_url}"
        
        url = f"{base_url}/v2/models/{params_dict['gliner_mname']}/infer"
        response = requests.post(
            url,
            json=payload,
            timeout=timeout_secs
        )
        response.raise_for_status()
        results = response.json()
    except Exception as err:
        print('call_triton_server_thread: Error during inference call to Triton server')
        print(err)
        print(err.with_traceback(None))
        print('Inputs were:', payload)
        print('Model:', params_dict["gliner_mname"])
        time.sleep(1)
        redo_queue.put(int(params_dict["id_emergencia"]))
        return
    
    id_emergencia = params_dict["id_emergencia"]
    horario_contexto = params_dict["horario_contexto"]

    outputs = {output_dict["name"]: output_dict["data"][0] for output_dict in results["outputs"]}
    try:
        print("Raw NER result:", outputs)
        entities_json_str = outputs["ENTITIES_JSON"].replace('\\"', '"').replace('\\n', '\n')
        print("Raw NER result 2:", entities_json_str)
        response_json = json.loads(entities_json_str)
    except json.JSONDecodeError as err:
        entities_json_str = outputs["ENTITIES_JSON"]
        first_parse_round = json.loads(entities_json_str)
        response_json = {}
        for key, value in first_parse_round.items():
            print(key)
            print(value)
            response_json[key] = json.loads(value)

    meta_info_str = outputs["META_INFO"].replace('\\"', '"').replace('\\n', '\n')
    response_meta = json.loads(meta_info_str)

    print("Clean NER result:", response_json)
    
    for label in values_to_ignore.keys():
        if label in response_json:
            not_ignore = []
            for value, score in response_json[label]:
                if value.lower() not in values_to_ignore[label]:
                    not_ignore.append([value, score])
            response_json[label] = not_ignore
    
    top_to_keep = params_dict['top_to_keep']
    not_keep_top = params_dict['not_keep_top']
    for label in response_json.keys():
        if type(response_json[label]) == list:
            response_json[label].sort(key=lambda x: x[1], reverse=True)
            if label not in not_keep_top and len(response_json[label]) > top_to_keep:
                response_json[label] = response_json[label][:top_to_keep]  # Keep only the top 3 values for each label
    
    print("Processed NER result:", response_json)

    to_save = {
        "id_emergencia": id_emergencia,
        "horario_contexto": horario_contexto,
        "entities": response_json,
        "horario_fim": float(time.time()),
        "meta": response_meta
    }
    print("New NER result:", json.dumps(to_save, ensure_ascii=False, indent=3))
    save_ner_to_db(to_save, redo_queue)

def triton_worker_process(gliner_mname, transcript_queue, redo_queue, result_queue, 
                          hardware_config, batch_size: int, labels_set, max_tokens: int, 
                          n_cpus: int):
    TRITON_SERVER_URL_NER = os.environ["TRITON_SERVER_URL_NER"] 
    while True:
        try:
            tp = transcript_queue.get()
        except QueueEmptyError as err:
            tp = None
        
        if tp != None:
            try:
                id_emergencia, horario_contexto, transcript = tp
                label_list = ner_labels[labels_set]
                should_stop = "STOP" == transcript
                
                if should_stop:
                    #treat stop signal
                    break
                else:
                    params = {
                        "id_emergencia": id_emergencia,
                        "horario_contexto": horario_contexto,
                        "gliner_mname": gliner_mname,
                        "labels": label_list,
                        "top_to_keep": 3,
                        "not_keep_top": many_values,
                        "batch_size": batch_size,
                        "max_tokens": max_tokens,
                        "transcript": transcript,
                    }
                    ner_thread = threading.Thread(target=call_triton_server_thread, 
                                                args=(params, redo_queue, TRITON_SERVER_URL_NER))
                    ner_thread.start()
                    
            except Exception as err:
                print('Error processing transcript')
                print(f'Erro: {err}')
                print(err.with_traceback(None))
                id_emergencia, horario_contexto, _ = tp
                
                print(id_emergencia, horario_contexto)
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