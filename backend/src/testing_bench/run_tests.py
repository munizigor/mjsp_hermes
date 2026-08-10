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
from test_utils.data_processing import get_inputs
from test_utils.audio_processing import prepare_api_audio_files
from test_utils.scheduling import get_docker_measurer, calcular_delay_hf


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

global current_output_dir
current_output_dir = ""

test_parameters = [
    {
        "n_ligacoes": 72,
        "carga_de_ligacoes": 24,
    },
    {
        "n_ligacoes": 72,
        "carga_de_ligacoes": 36,
    },
    {
        "n_ligacoes": 72,
        "carga_de_ligacoes": 12,
    },
    {
        "n_ligacoes": 72,
        "carga_de_ligacoes": 16,
    },
    {
        "n_ligacoes": 24,
        "carga_de_ligacoes": 1.8,
    },
    {
        "n_ligacoes": 48,
        "carga_de_ligacoes": 4.8,
    },
    {
        "n_ligacoes": 72,
        "carga_de_ligacoes": 7.8,
    },
]

quick_test_parameters = [
    {
        "n_ligacoes": 32,
        "carga_de_ligacoes": 8,
    }
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


def set_output_permissions(output_dir: str):
    """
    Recursively grants all users write access to directories (and the execute bit to enter them),
    and read-only access to files. Assumes execution as sudo.
    """
    if not os.path.exists(output_dir):
        print(f"⚠️ Directory not found, skipping permissions: {output_dir}")
        return

    # Sanity check: Ensure we are actually running as root (sudo)
    if os.geteuid() != 0:
        print(
            "⚠️ Warning: Script is not running as root. os.chmod may fail on files you don't own."
        )

    # --- PERMISSION MASKS ---
    # 0o777 (rwxrwxrwx): Read, Write, and Execute for Owner, Group, and Others.
    dir_mode = 0o777

    # 0o644 (rw-r--r--): Owner can read/write, Group/Others are read-only.
    # (Note: Change this to 0o444 if you want to lock out root/owner from writing too).
    file_mode = 0o644

    try:
        # 1. Apply permissions to the root output directory itself
        os.chmod(output_dir, dir_mode)

        # 2. Traverse the directory tree top-down
        for root, dirs, files in os.walk(output_dir):

            # Apply 777 to all subdirectories
            for d in dirs:
                dir_path = os.path.join(root, d)
                os.chmod(dir_path, dir_mode)

            # Apply 644 (or 444) to all files
            for f in files:
                file_path = os.path.join(root, f)
                os.chmod(file_path, file_mode)

        print(f"✅ Permissions successfully applied to {output_dir}")

    except Exception as e:
        print(f"❌ Failed to set permissions in {output_dir}: {e}")


# Docker handling


def docker_stop(proj_dir=proj_dir, output_dir=current_output_dir):
    print("🧹 Tearing down Docker Compose stack...")
    subprocess.run(
        ["docker", "compose", "down"],
        cwd=proj_dir,
        check=False,  # We don't want to crash during cleanup if it fails
    )
    print("🏁 Cleanup complete.")
    if output_dir:
        if os.path.exists(output_dir):
            set_output_permissions(output_dir)


def handle_interrupt(signum, frame):
    """
    Catches system signals (like CTRL+C) and forces a graceful exit.
    """
    docker_stop()
    print(f"\n⚠️ Received interrupt signal ({signum}). Exiting gracefully...")
    # sys.exit triggers the atexit registered functions
    sys.exit(1)


def docker_run(proj_dir):
    print("Making sure docker is not running yet...")
    docker_stop(proj_dir)

    print("🚀 Starting Docker Compose stack...")

    success = False
    try:
        # Pass os.environ to docker compose using a big export
        run_env = os.environ.copy()
        run_env.update(env_vals)
        out_log_path = os.path.join(proj_dir, "start_stdout.log")
        err_log_path = os.path.join(proj_dir, "start_stderr.log")
        export_str = "export "
        for key, value in env_vals.items():
            if " " in value:
                value = f'"{value}"'
            export_str += f"{key}={value} "
        export_str = export_str.rstrip(" ")

        # 1. Start containers and wait for them to be healthy
        # -d: run in detached mode
        # --wait: block until all containers report as healthy
        to_remove = [
            f"{proj_dir}/datasets/naturezas_cache_vllm.json",
            f"{proj_dir}/{os.environ['LOCAL_SQL_DB_PATH']}",
        ]
        for d in to_remove:
            if os.path.exists(d):
                shutil.rmtree(d, ignore_errors=True)
        cmd_vec = ["docker", "compose", "up", "-d", "--wait"]
        print("Running command:", " ".join(cmd_vec))
        with open(out_log_path, "w") as out_log, open(err_log_path, "w") as err_log:
            subprocess.run(
                cmd_vec,
                check=True,
                text=True,
                cwd=proj_dir,
                env=run_env,  # Injects variables seamlessly
                stdout=out_log,  # Redirects stdout to file
                stderr=err_log,  # Redirects stderr to file
            )
        print("✅ All containers are up and healthy!")

        # 2. Start a background process to stream the actual container logs to a file
        log_path = os.path.join(proj_dir, "containers.log")
        log_file = open(log_path, "w")

        print(f"📝 Streaming container application logs to {log_path}...")
        log_process = subprocess.Popen(
            ["docker", "compose", "logs", "-f"],  # -f follows the logs in real time
            cwd=proj_dir,
            env=run_env,
            stdout=log_file,  # The OS handles writing this directly to the file
            stderr=subprocess.STDOUT,  # Combine stderr and stdout into the same file
        )

        success = True
    except subprocess.CalledProcessError as e:
        print(f"❌ Docker Compose failed to start or health checks timed out: {e}")
    except Exception as e:
        print(f"❌ An error occurred during request execution: {e}")
    finally:
        if not success:
            docker_stop(proj_dir)
            sys.exit(1)


def start_docker(proj_dir, config_json_path):
    # Copy config.json to proj_dir/config.json
    cp(config_json_path, proj_dir + "/config.json")

    docker_run(proj_dir)


# 1. Register the cleanup function for normal exits and Python crashes
atexit.register(docker_stop)
# 2. Wire up the signal handlers
# SIGINT captures CTRL+C
signal.signal(signal.SIGINT, handle_interrupt)
# SIGTERM captures standard OS kill commands
signal.signal(signal.SIGTERM, handle_interrupt)


def perform_test(test_config):
    output_dir = test_config["output_dir"]
    global current_output_dir
    current_output_dir = output_dir
    n_ligacoes = test_config["n_ligacoes"]
    test_name = test_config["test_name"]
    carga_de_ligacoes = test_config["carga_de_ligacoes"]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print(f"Nome do teste: {test_name}")

    audios_ds, dataset_path = get_inputs(n_ligacoes, fake_calls_dataset_path)

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

    start_docker(proj_dir, config_path)

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
        docker_stop(proj_dir)
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

    # docker_stop(proj_dir)

    cp(
        "/tmp/hermes_queue-interpretation.tsv",
        f"{output_dir}/hermes_queue-interpretation.tsv",
    )
    cp(config_path, f"{output_dir}/config.json")
    cp(f"{proj_dir}/start_stdout.log", f"{output_dir}/start_stdout.log")
    cp(f"{proj_dir}/start_stderr.log", f"{output_dir}/start_stderr.log")
    cp(f"{proj_dir}/containers.log", f"{output_dir}/containers.log")

    copytree(dataset_path, f"{output_dir}/test_dataset")
    json.dump(
        test_config,
        open(f"{output_dir}/load_parameters.json", "w"),
        indent=4,
        ensure_ascii=False,
    )


# Actual analysis
if __name__ == "__main__":

    if not os.path.exists(output_dir_base):
        os.mkdir(output_dir_base)

    existing_tests_params = glob(f"{output_dir_base}/*/load_parameters.json")
    tested_names = [json.load(open(p, "r"))["test_name"] for p in existing_tests_params]

    for test_config in test_parameters:
        test_name = test_config["test_name"]
        if test_name in tested_names and not quick_test:
            print(f"Test {test_name} already exists, skipping...")
            continue
        perform_test(test_config)
