import atexit
import os
import sys

env_file_path = "/.env" if os.path.exists("/.env") else ".env"
if os.path.exists(env_file_path):
    with open(env_file_path, "r") as f:
        env_vals = {
            rawline.split("=", 1)[0]: rawline.split("=", 1)[1].rstrip("\r\n")
            for rawline in f.read().split("\n")
            if "=" in rawline and not rawline.lstrip().startswith("#")
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

from time import sleep
from rendering import render_main_log_html2 as render_main_log_html3
from monitors.clusters_monitor import start_clusters_monitor

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
        key_start = api_key_header[:5]
        key_end = api_key_header[-5:]
        print(f"Tentativa de acesso não autorizado: {key_start}...{key_end}", file=sys.stderr)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Acesso não autorizado."
        )


# Start variables that should exist for all workers
sqlite_conn = None
try:
    while not os.path.exists(os.environ["SQLITE_DB_PATH"]):
        print(f'hermes/start.py waiting 2s for {os.environ["SQLITE_DB_PATH"]}')
        sleep(2)
    print("Connecting to DB", os.environ["SQLITE_DB_PATH"])
    sqlite_conn = sqlite.connect(os.environ["SQLITE_DB_PATH"])
    print("DB Pre Initialization success!")
except Exception as err:
    print("Exception before startup")
    print(err)
    quit(1)
finally:
    if sqlite_conn is not None:
        sqlite_conn.close()


def parse_triton_hosts():
    triton_hosts = []
    vllm_hosts = []
    flask_hosts = []
    for key, value in os.environ.items():
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
                port_key = key.replace("HOST", "PORT")
                '''if port_key in os.environ:
                    port = os.environ[port_key]
                else:'''
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


def _stop_monitor_process(proc, sig, name: str, timeout: float = 5.0):
    if sig is not None:
        try:
            sig.value = 1
        except Exception as err:
            print(f"Error setting stop signal for {name}: {err}", file=sys.stderr)
    if proc is not None:
        try:
            if proc.is_alive():
                print(f"Stopping {name}...")
                proc.join(timeout=timeout)
            if proc.is_alive():
                print(f"{name} did not exit within {timeout}s, terminating...", file=sys.stderr)
                proc.terminate()
                proc.join(timeout=2.0)
            if proc.is_alive():
                print(f"{name} still alive after terminate, killing...", file=sys.stderr)
                proc.kill()
                proc.join(timeout=1.0)
        except Exception as err:
            print(f"Error stopping {name}: {err}", file=sys.stderr)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # INICIO DO SERVIDOR
    triton_monitor = None
    triton_monitor_process = None
    triton_stop_signal = None

    def _cleanup_all_monitors():
        _stop_monitor_process(triton_monitor_process, triton_stop_signal, "Clusters Monitor")

    atexit.register(_cleanup_all_monitors)

    try:
        sqlite_conn = None
        cursor = None
        try:
            sqlite_conn = sqlite.connect(os.environ["SQLITE_DB_PATH"])
            cursor = sqlite_conn.cursor()
            # Empties the metrics reading table "metrics" in DB
            cursor.execute("DELETE FROM metrics")
            sqlite_conn.commit()
            print("Metrics table cleared!")
        finally:
            if cursor is not None:
                cursor.close()
            if sqlite_conn is not None:
                sqlite_conn.close()

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
        _cleanup_all_monitors()
        quit(1)

    try:
        yield
    finally:
        # FIM DO SERVIDOR
        _cleanup_all_monitors()
        atexit.unregister(_cleanup_all_monitors)
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
    sqlite_conn = None
    cursor = None
    try:
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
    finally:
        if cursor is not None:
            cursor.close()
        if sqlite_conn is not None:
            sqlite_conn.close()

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
    return log_html
