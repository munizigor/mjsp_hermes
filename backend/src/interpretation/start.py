import os
import sys
import json
from time import sleep
import signal

from interpreters import EmergencyInterpreter
from agent import InterpretationAgent


def start_and_listen(
    hardware_config,
    embedding_hardware,
    model,
    delay=5,
    n_in_batch=10,
    max_similar_natures=16,
    n_cpus=6,
    min_transcricao=10,
):
    ei = EmergencyInterpreter(
        model,
        hardware_config=hardware_config,
        embedding_device=embedding_hardware,
        sqlite_path=os.environ["SQLITE_DB_PATH"],
        max_similar_natures=max_similar_natures,
        n_cpus=n_cpus,
    )
    i_agent = InterpretationAgent(
        ei,
        os.environ["SQLITE_DB_PATH"],
        n_in_batch=n_in_batch,
        max_similar_natures=max_similar_natures,
        min_transcricao=min_transcricao,
    )

    # Captura SIGTERM e SIGINT para parar o agente
    def handle_sigterm(signum, frame):
        print("Recebido sinal de finalização (SIGTERM/SIGINT). Parando agente...")
        i_agent.stop()

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    i_agent.escutar_async()


if __name__ == "__main__":
    config_json = json.load(open("/config.json", "r"))
    interpretation_config = config_json.get("interpretation", {})
    hardware_config = interpretation_config.get("hardware_config", "cpu")
    embedding_hardware = interpretation_config.get("embedding_hardware", "cpu")
    model = interpretation_config.get("model", "gaia")
    n_cpus = interpretation_config.get("n_cpus", 6)
    max_similar_natures = interpretation_config.get("max_similar_natures", 16)
    delay = interpretation_config.get("delay", 5)
    n_in_batch = interpretation_config.get("n_in_batch", 10)
    min_transcricao = interpretation_config.get("min_transcricao", 10)

    while not os.path.exists(os.environ["SQLITE_DB_PATH"]):
        print(f'asr/start.py waiting 4s for {os.environ["SQLITE_DB_PATH"]}')
        sleep(4)

    start_and_listen(
        hardware_config,
        embedding_hardware,
        model,
        delay=delay,
        n_in_batch=n_in_batch,
        max_similar_natures=max_similar_natures,
        n_cpus=n_cpus,
        min_transcricao=min_transcricao,
    )
