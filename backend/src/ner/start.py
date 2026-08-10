import json
import os
from time import sleep
from glob import glob

print("Starting NER module...")

from runners.gliner_utilities import GLINER_MODEL_NAME
from runners.ner_agent import NERAgent

if __name__ == "__main__":
    config_json = json.load(open("/config.json", "r"))
    ner_config = config_json.get("ner", {})
    n_gliner_workers = ner_config.get("n_gliner_workers", 1)
    batch_size = ner_config.get("batch_size", 5)
    max_tokens = ner_config.get("max_tokens", 150)
    n_cpus = ner_config.get("n_cpus", 5)

    hardware_config = ner_config.get("hardware_config", "cpu")
    gliner_mname = ner_config.get("gliner_mname", GLINER_MODEL_NAME)

    while not os.path.exists(os.environ["SQLITE_DB_PATH"]):
        print(f'ner/start.py waiting 3s for {os.environ["SQLITE_DB_PATH"]}')
        print(glob("/app/*"))
        print(glob("/app/sqlite_data/*"))
        sleep(3)
    # sqlite_conn = sqlite3.connect(os.environ["SQLITE_DB_PATH"])
    # start_ner_database(sqlite_conn)
    print("DB Initialization success!")

    try:
        print("Agent Initializing...")
        print(f"Hardware config: {config_json}")
        ner_agent = NERAgent(
            os.environ["SQLITE_DB_PATH"],
            gliner_mname=gliner_mname,
            n_gliner_workers=n_gliner_workers,
            hardware_config=hardware_config,
            batch_size=batch_size,
            max_tokens=max_tokens,
            n_cpus=n_cpus,
        )
        print("Agent Initialization success!")
        while True:
            sleep(30)
    except Exception as err:
        print("Error at ner/start.py startup")
        print(err)
        ner_agent.stop()
        quit(1)
