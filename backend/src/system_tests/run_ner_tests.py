from unittest import TestProgram
import sys
import json
import requests
import threading
import time
import os
from shutil import copy as cp
from glob import glob
from shutil import copytree

import polars as pl
import numpy as np
from tqdm import tqdm

from ner.long_test import download_eval_dataset

from docker_control import safely_start_docker
from test_utils.call_handlers import start_emergency, send_transcript, end_emergency
from startup_routine import (proj_dir, env_vals, api_key, hermes_addr, hermes_port, sqlite_path, 
    docker_ctrl, config_path, configs, output_dir_base, header, dataset_name, 
    fake_calls_dataset_path, hermes_url, hermes_headers, config_name)
"""
Example: python run_ner_tests.py ../envs/config.gcp.glinerx_ministral.json
"""
print("Argumentos recebidos:", sys.argv)


test_parameters = {
    "n_ligacoes": 60
}
quick_test_parameters = {
    "n_ligacoes": 16
}

if len(sys.argv) >= 3:
    quick_test = sys.argv[2] == "quick"
else:
    quick_test = False
if quick_test:
    test_parameters = quick_test_parameters

ner_eval_df = download_eval_dataset(total_rows=test_parameters["n_ligacoes"])

newname = f"ner_eval-{config_name}-n_{test_parameters['n_ligacoes']}"
newname_timestamp = (
    newname
    + f'-{time.strftime("%d-%m-%Y_%H:%M:%S", time.gmtime())}'.replace(":", "-")
)
test_config = {
    "n_ligacoes": test_parameters["n_ligacoes"],
    "test_name": newname,
    "test_name_timestamp": newname_timestamp,
    "output_dir": f"{output_dir_base}/{newname_timestamp}"
}

#save config
if not os.path.exists(test_config['output_dir']):
    os.mkdir(test_config['output_dir'])

with open(f"{test_config['output_dir']}/test_config.json", "w") as f:
    json.dump(test_config, f, indent=4)

docker_ctrl.set_output_dir(test_config["output_dir"])

measurer, readings, stop_flag, start_time = safely_start_docker(docker_ctrl, config_path,
    test_config["output_dir"], hermes_url, hermes_headers)

get_delay_secs = 3

results_json_path = f"{test_config['output_dir']}/results.json"
results_raw = []

global_bar = tqdm(total=len(ner_eval_df))

info_columns = [
    "endereco_completo",
    "nome_solicitante",
    "rua",
    "numero",
    "complemento",
    "bairro",
    "cidade",
    "estado",
    "CEP",
    "ponto_de_referencia",
    "fatoOcorrendoNesteMomento",
    "autorDoFatoNoLocal",
    "autorDoFatoArmado",
    "feridosComRiscoDeMorte",
    "riscoDeTumulto",
    "leiMariaPenha",
]

for row in ner_eval_df.rows(named=True):
    original_info = {k: row[k] for k in info_columns}
    test_id = row["ID"]
    operator_code = f"operator_{test_id:03d}"
    transcript = row["roteiro_segmentado"]
    print(f"Processing {test_id}")
    lines = [(speaker, l2.strip()) for speaker, l2 in transcript if len(l2.strip()) > 0]
    time_per_char = 0.005
    total_chars = sum([len(line[1]) for line in lines])
    total_time_secs = time_per_char * total_chars
    print(f"\tTotal characters: {total_chars}; total time for this transcript: {total_time_secs}s")
    call_start = time.time()
    # Inicia a emergência
    emergency_id = start_emergency(hermes_url, hermes_headers, operator_code)
    if emergency_id is None:
        print(f"Falha ao iniciar emergência para áudio {test_id}")
        continue
    inf_results = {}
    for i, (speaker, line) in enumerate(lines):
        to_sleep = len(line) * time_per_char
        time.sleep(to_sleep)
        sent = send_transcript(hermes_url, hermes_headers, emergency_id, transcript)
    
    print(f"Encerrando emergência {emergency_id}")
    end_emergency(hermes_url, emergency_id)

    delay_local = 0.0
    delay_wait_time = 1
    max_wait_time = 16  # Tempo máximo de espera em segundos
    wait_start = time.time()
    while True:
        if time.time() - wait_start > max_wait_time:
            print(f"Tempo máximo de espera excedido para emergência {emergency_id}")
            break
        end_status = requests.get(
            f"{hermes_url}/processing_finished/",
            params={"id_emergencia": emergency_id},
        )
        content = end_status.json()
        if content["terminado"] == False:
            print(emergency_id, content)
            houve_delay = True
            time.sleep(delay_wait_time)
        else:
            delay_local = time.time() - wait_start
            break

    if delay_local <= 2:
        delay_local = 0.0

    print(f"Chamada {test_id} processada com sucesso")

    try:
        inf_results = requests.get(
            f"{hermes_url}/get_all_inference_results/",
            params={"id_emergencia": emergency_id},
        ).json()
        print(inf_results)
    except Exception as err:
        print("Erro tentando obter resultados:", err, err.__context__)
        pass

    final_result = {
        "dataset_index": test_id,
        "id_emergencia": emergency_id,
        "delay_local": delay_local,
        "metadata": original_info,
        "results": inf_results,
    }

    results_raw.append(final_result)
    with open(results_json_path, "w") as f:
        json.dump(results_raw, f, ensure_ascii=False, indent=4)
    global_bar.update(1)

        
    