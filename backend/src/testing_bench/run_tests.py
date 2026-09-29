import sys
import json
import requests
import threading
import time
import os
from shutil import copy as cp
import signal
import atexit
import subprocess
import shutil
from glob import glob
from shutil import copytree

import polars as pl
import numpy as np

from test_utils.call_handlers import process_audio_call_hf
from test_utils.data_processing import get_inputs, get_input_selection
from test_utils.audio_processing import prepare_api_audio_files
from test_utils.scheduling import get_docker_measurer, calcular_delay_hf
from docker_control import DockerController
"""
Example: python run_tests.py ../envs/config.gcp.glinerx_ministral.json
"""
print("Argumentos recebidos:", sys.argv)

# Handle constants and environment variables

proj_dir = "../.."
mandatory_keys = ["LOCAL_SQL_DB_PATH", "API_KEY", "HERMES_ADDR", "HERMES_PORT"]
env_path = f"{proj_dir}/.env"
env_vals = {
    rawline.split("=")[0]: rawline.split("=")[1].rstrip("\n")
    for rawline in open(env_path, "r").read().split("\n")
    if "=" in rawline
}
for key, value in env_vals.items():
    os.environ[key] = value

if not all(key in os.environ for key in mandatory_keys):
    print("❌ Missing mandatory keys in .env file")
    print("Missing:", [key for key in mandatory_keys if key not in os.environ])
    sys.exit(1)

print(env_vals)
api_key = os.environ["API_KEY"]
hermes_addr = os.environ["HERMES_ADDR"]
hermes_port = os.environ["HERMES_PORT"]
sqlite_path = f"{proj_dir}/{os.environ['LOCAL_SQL_DB_PATH']}/database.sqlite"

docker_ctrl = DockerController(proj_dir=proj_dir, env_vals=env_vals)

'''test_parameters = [
    {
        "n_ligacoes": 40,
        "carga_de_ligacoes": 5,
    },
    {
        "n_ligacoes": 80,
        "carga_de_ligacoes": 8,
    },
    {
        "n_ligacoes": 100,
        "carga_de_ligacoes": 12,
    },
    {
        "n_ligacoes": 100,
        "carga_de_ligacoes": 16,
    },
]'''

test_parameters = [
    {
        "n_ligacoes": 120,
        "carga_de_ligacoes": 20,
    },
    {
        "n_ligacoes": 120,
        "carga_de_ligacoes": 25,
    },
    {
        "n_ligacoes": 120,
        "carga_de_ligacoes": 30,
    },
]

quick_test_parameters = [
    {
        "n_ligacoes": 5,
        "carga_de_ligacoes": 4,
    },
    {
        "n_ligacoes": 7,
        "carga_de_ligacoes": 5,
    },
    {
        "n_ligacoes": 9,
        "carga_de_ligacoes": 6,
    },
    {
        "n_ligacoes": 11,
        "carga_de_ligacoes": 7,
    },
    {
        "n_ligacoes": 13,
        "carga_de_ligacoes": 8,
    },
]


if len(sys.argv) not in [2, 3]:
    sys.exit(1)
config_path = sys.argv[1]
if len(sys.argv) == 3:
    quick_test = sys.argv[2] == "quick"
else:
    quick_test = False

if quick_test:
    test_parameters = quick_test_parameters

total_to_download = sum([t['n_ligacoes'] for t in test_parameters]) + 1

configs = json.load(open(config_path, "r"))
output_dir_base = f"{proj_dir}/{configs['test_output_dir']}"
header = f"X-Hermes-API-Key: {api_key}"
dataset_name = "fake-emergencies-br"
fake_calls_dataset_path = "pitagoras-alves/fake-emergencies-br"
hermes_url = f"http://{hermes_addr}:{hermes_port}"
hermes_headers = {"X-Hermes-API-Key": api_key}

print(f"Endereço Hermes: {hermes_addr}:{hermes_port}")
print(f"URL Hermes: {hermes_url}")
print(f"Caminho do dataset: {fake_calls_dataset_path}")

config_name = os.path.basename(config_path).replace("config.", "").replace(".json", "")

for test_config in test_parameters:

    test_config["n_ligacoes"] = test_config["n_ligacoes"]
    test_config["carga_de_ligacoes"] = test_config["carga_de_ligacoes"]
    newname = f"{config_name}-n_{test_config['n_ligacoes']}-cl_{test_config['carga_de_ligacoes']}"
    newname_timestamp = (
        newname
        + f'-{time.strftime("%d-%m-%Y_%H:%M:%S", time.gmtime())}'.replace(":", "-")
    )
    test_config["test_name"] = newname
    test_config["test_name_timestamp"] = newname_timestamp
    test_config["output_dir"] = (
        f"{output_dir_base}/{test_config['test_name_timestamp']}"
    )

# Download and prepare all inputs once
all_calls_dataset, all_calls_dataset_path = get_inputs(total_to_download, fake_calls_dataset_path)

def perform_test(test_config, previous_n):
    output_dir = test_config["output_dir"]
    docker_ctrl.set_output_dir(output_dir)
    n_ligacoes = test_config["n_ligacoes"]
    test_name = test_config["test_name"]
    carga_de_ligacoes = test_config["carga_de_ligacoes"]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print(f"Nome do teste: {test_name}")

    audios_ds, dataset_path = get_input_selection(
        all_calls_dataset, previous_n, n_ligacoes, output_dir)

    print("Columns:", audios_ds[0].keys())
    audios_ds = calcular_delay_hf(audios_ds, carga_de_ligacoes, output_dir)
    short_audio_paths = []
    audio_lengths_col = []
    for audio_data in audios_ds:
        # print(audio_data["audio"].keys())
        new_paths, audio_lengths = prepare_api_audio_files(
            audio_data["audio"]["array"],
            audio_data["audio"]["sampling_rate"],
            verbose=True,
        )
        short_audio_paths.append(new_paths)
        audio_lengths_col.append(np.array(audio_lengths))
    audios_ds = audios_ds.add_column("audio_paths", short_audio_paths)
    audios_ds = audios_ds.add_column("audio_lengths", audio_lengths_col)
    print("Columns:", audios_ds[0].keys())

    docker_ctrl.start_docker(config_path, output_dir=output_dir)

    # Verifica se a API do Hermes está rodando
    delays = [2, 4, 5]
    running = False
    for delay in delays:
        try:
            response = requests.get(f"{hermes_url}/", headers=hermes_headers)
            if response.status_code == 200:
                print("API do Hermes está rodando e acessível")
                running = True
                break
            else:
                print(f"API do Hermes retornou status {response.status_code}")
                time.sleep(delay)
        except Exception as e:
            print(f"Erro ao conectar com API do Hermes: {e}")
            time.sleep(delay)
    if running:
        time.sleep(12)
    else:
        print("API do Hermes não está rodando")
        docker_ctrl.docker_stop()
        sys.exit(1)

    measurer, readings, stop_flag = get_docker_measurer()
    measurer.start()
    start_time = time.time()

    # Processa as chamadas em threads paralelas
    print(f"\nIniciando processamento de {len(audios_ds)} chamadas em paralelo...")

    threads = []
    i = 1
    for audio_data in audios_ds:
        operator_code = f"operator_{i+1:03d}"

        # Cria uma thread para cada chamada
        thread = threading.Thread(
            target=process_audio_call_hf,
            args=(hermes_url, hermes_headers, audio_data, operator_code, output_dir),
        )
        threads.append(thread)
        thread.start()

        # Pequeno delay entre inícios para evitar sobrecarga
        time.sleep(0.2)
        i += 1

    # Aguarda todas as threads terminarem
    print("Aguardando todas as chamadas terminarem...")
    for i, thread in enumerate(threads):
        thread.join()
        print(f"Thread {i+1} finalizada")
    stop_flag.set()

    readings_parquet = pl.DataFrame(readings)
    readings_parquet.write_parquet(f"{output_dir}/container_stats.parquet")

    cp(sqlite_path, output_dir + "/sqlite.db")
    # save_to_parquet(sqlite_path, output_dir, results_by_index)

    print("\nTodas as chamadas foram processadas!")
    print(f"Total de chamadas processadas: {len(audios_ds)}")

    """for p in thread_result_paths:
        os.remove(p)"""

    # Stop containers first without locking permissions so logs are flushed before copying
    docker_ctrl.docker_stop(apply_permissions=False)

    cp(
        "/tmp/hermes_queue-interpretation.tsv",
        f"{output_dir}/hermes_queue-interpretation.tsv",
    )
    cp(config_path, f"{output_dir}/config.json")
    cp(f"{proj_dir}/start_stdout.log", f"{output_dir}/start_stdout.log")
    cp(f"{proj_dir}/start_stderr.log", f"{output_dir}/start_stderr.log")
    cp(f"{proj_dir}/containers.log", f"{output_dir}/containers.log")

    copytree(dataset_path, f"{output_dir}/test_dataset_finalcopy")
    json.dump(
        test_config,
        open(f"{output_dir}/load_parameters.json", "w"),
        indent=4,
        ensure_ascii=False,
    )

    docker_ctrl.set_permissions(output_dir)


# Actual analysis
if __name__ == "__main__":

    if not os.path.exists(output_dir_base):
        os.mkdir(output_dir_base)

    existing_tests_params = glob(f"{output_dir_base}/*/load_parameters.json")
    tested_names = [json.load(open(p, "r"))["test_name"] for p in existing_tests_params]
    previous_n = 0
    for test_config in test_parameters:
        test_name = test_config["test_name"]
        if test_name in tested_names and not quick_test:
            print(f"Test {test_name} already exists, skipping...")
            continue
        perform_test(test_config, previous_n)
        previous_n += test_config["n_ligacoes"]
