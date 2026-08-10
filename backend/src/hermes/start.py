import os

env_vals = {
    rawline.split("=")[0]: rawline.split("=")[1].rstrip("\n")
    for rawline in open(".env", "r").read().split("\n")
    if "=" in rawline
}
for k, v in env_vals.items():
    os.environ[k] = v

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from fastapi import Security, HTTPException, Depends
from fastapi.security.api_key import APIKeyHeader
from starlette import status

from contextlib import asynccontextmanager
import sqlite3 as sqlite

from rendering import render_main_log_html2 as render_main_log_html3
from monitors.clusters_monitor import start_clusters_monitor
from monitors.asterisk_monitor_dependent import start_asterisk_monitor

from routers import (
    get_endpoints,
    post_endpoints,
    emergency_post_endpoints,
    emergency_get_endpoints,
)

API_KEY_NAME = "X-Hermes-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
HERMES_API_KEY = os.environ.get("HERMES_API_KEY")


async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == HERMES_API_KEY:
        return api_key_header
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Acesso não autorizado."
        )


# Start variables that should exist for all workers
try:
    print("Connecting to DB", os.environ["SQLITE_DB_PATH"])
    sqlite_conn = sqlite.connect(os.environ["SQLITE_DB_PATH"])
    sqlite_conn.close()
    print("DB Pre Initialization success!")
except Exception as err:
    print("Exception before startup")
    print(err)
    print(err.with_traceback())
    quit(1)


def parse_triton_hosts():
    triton_hosts = []
    vllm_hosts = []
    flask_hosts = []
    for key, value in os.environ.items():
        print(key, value)
        if "TRITON_SERVER_" in key and not ("TIMEOUT_SECS" in key):
            if "_URL_" in key:
                servertype = key.split("_URL_")[1]
            else:
                servertype = "ASR"
            hostname = value.split(":")[0]
            triton_hosts.append({"hostname": hostname, "servertype": servertype})
        if "_VLLM_" in key:
            key_parts = key.split("_VLLM_")
            inf_type_name = key_parts[0]
            info_name = key_parts[1]
            if info_name == "HOST":
                hostname = value
                vllm_hosts.append(
                    {
                        "hostname": hostname,
                        "servertype": inf_type_name.upper(),
                    }
                )
        if "FLASK" in key:
            key_parts = key.split("_FLASK_")
            inf_type_name = key_parts[0]
            info_name = key_parts[1]
            if info_name == "SERVER_IP":
                hostname = value
                flask_hosts.append(
                    {
                        "hostname": hostname,
                        "servertype": inf_type_name.upper(),
                    }
                )
    return triton_hosts, vllm_hosts, flask_hosts


@asynccontextmanager
async def lifespan(app: FastAPI):
    # INICIO DO SERVIDOR
    try:
        sqlite_conn = sqlite.connect(os.environ["SQLITE_DB_PATH"])
        cursor = sqlite_conn.cursor()
        cursor.executescript(open("/init.sql", "r").read())
        cursor.close()
        sqlite_conn.commit()
        sqlite_conn.close()
        # start_ner_database(sqlite_conn)
        print("DB Initialization success!")

        # Emptyes the metrics reading table "metrics" in DB
        sqlite_conn = sqlite.connect(os.environ["SQLITE_DB_PATH"])
        cursor = sqlite_conn.cursor()
        cursor.execute("DELETE FROM metrics")
        cursor.close()
        sqlite_conn.commit()
        sqlite_conn.close()
        print("Metrics table cleared!")

        use_asterisk = os.environ.get("USE_ASTERISK", "FALSE") == "TRUE"
        if use_asterisk:
            print("Starting Asterisk Monitor")
            monitor, monitor_process, stop_signal = start_asterisk_monitor()

        print("Starting Clusters Monitor")
        triton_hosts, vllm_hosts, flask_hosts = parse_triton_hosts()
        for host in triton_hosts:
            print("\tTRITON", host)
        for host in vllm_hosts:
            print("\tvLLM", host)
        for host in flask_hosts:
            print("\tFLASK", host)
        triton_monitor, triton_monitor_process, triton_stop_signal = (
            start_clusters_monitor(triton_hosts, vllm_hosts, flask_hosts, 8002)
        )

        print("Initialization success!")
    except Exception as err:
        print("Error at startup")
        print(err)
        quit(1)
    yield
    # FIM DO SERVIDOR
    print("Stopping Asterisk Monitor")
    stop_signal.value = 1
    monitor_process.join()
    print("Exiting FASTAPI Lifespan")


try:

    app = FastAPI(lifespan=lifespan, dependencies=[Depends(get_api_key)])
    app.include_router(get_endpoints.router)
    app.include_router(post_endpoints.router)
    app.include_router(emergency_post_endpoints.router)
    app.include_router(emergency_get_endpoints.router)
    print("App Initialization success!")
except Exception as err:
    print("Exception before startup")
    print(err)
    print(err.with_traceback())
    quit(1)


@app.get("/", response_class=HTMLResponse)
def read_root():
    # Render the main log HTML page
    sqlite_conn = sqlite.connect(os.environ["SQLITE_DB_PATH"])
    cursor = sqlite_conn.cursor()
    # Busca todas as linhas das tabelas
    emergency_rows = cursor.execute("SELECT * FROM emergencies").fetchall()
    emergency_columns = [desc[0] for desc in cursor.description]
    transcript_rows = cursor.execute("SELECT * FROM emergency_transcripts").fetchall()
    cursor.execute("SELECT * FROM emergency_transcripts LIMIT 1")
    transcript_columns = [desc[0] for desc in cursor.description]

    audio_rows = cursor.execute("SELECT * FROM emergency_audios").fetchall()
    cursor.execute("SELECT * FROM emergency_audios LIMIT 1")
    audio_columns = [desc[0] for desc in cursor.description]

    inference_rows = cursor.execute("SELECT * FROM resultados_inferencia").fetchall()
    cursor.execute("SELECT * FROM resultados_inferencia LIMIT 1")
    inference_columns = [desc[0] for desc in cursor.description]

    log_html = render_main_log_html3(
        emergency_columns,
        transcript_columns,
        emergency_rows,
        transcript_rows,
        audio_columns,
        audio_rows,
        # inference_columns,
        # inference_rows,
    )
    sqlite_conn.close()
    return log_html
