import socket
import time
import requests
import os
import argparse
import json
import uuid
from pathlib import Path
from typing import Dict, Any

# =============================
# Configurações AMI Asterisk
# =============================
AMI_HOST = "127.0.0.1"
AMI_PORT = 5038
AMI_USER = "hermes-sip-sign"
AMI_PASS = "inserir_senha_ami_aqui"

# =============================
# Configurações API Hermes
# =============================
API_HOST = "inserir_api_host_aqui"
API_PORT = 8001
API_KEY = "inserir_senha_api_aqui"
API_TIMEOUT = 5

COMM_MODES = ["API", "FILE_SYSTEM"]

class NetworkJSONSender:
    def __init__(self, shared_dir: str):
        self.shared_dir = Path(shared_dir)
        self.tmp_dir = self.shared_dir / "tmp"
        self.new_dir = self.shared_dir / "new"
        
        # Ensure directories exist once upon initialization
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.new_dir.mkdir(parents=True, exist_ok=True)

    def send(self, payload: Dict[str, Any], prefix: str = None) -> str:
        filename = f"{prefix if prefix else ''}-{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}.json"
        if "unix_timestamp" not in payload:
            payload["unix_timestamp"] = time.time()
        tmp_path = self.tmp_dir / filename
        new_path = self.new_dir / filename

        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp_path, new_path)
            return filename

        except Exception as e:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise RuntimeError(f"Failed to send message over network share: {e}") from e

# =============================
# Estado interno & Instância Global do Sender
# =============================
calls = {}
fs_sender = None  # Instância única inicializada no main

# =============================
# Funções auxiliares
# =============================
def send(sock, msg):
    sock.sendall((msg + "\r\n").encode())

def now():
    return int(time.time())

def send_api_request(endpoint, params=None):
    url = f"http://{API_HOST}:{API_PORT}{endpoint}"
    headers = {"X-Hermes-API-Key": API_KEY}

    try:
        r = requests.post(url, params=params, headers=headers, timeout=API_TIMEOUT)
        if r.status_code >= 300:
            print(f"[WARN] API {endpoint} {r.status_code}: {r.text}")
        else:
            print(f"[OK] API {endpoint}")
    except Exception as e:
        print(f"[ERROR] API {endpoint} falhou: {e}")

def send_fs_request(endpoint, params=None):
    global fs_sender
    if fs_sender is None:
        return

    try:
        payload = {
            "endpoint": endpoint,
            "params": params or {},
        }
        filename = fs_sender.send(payload, prefix=endpoint.strip('/'))
        print(f"[OK] FILE_SYSTEM {endpoint} ({filename})")
    except Exception as e:
        print(f"[ERROR] FILE_SYSTEM {endpoint} falhou: {e}")

def signal_start_call(origin, destination, timestamp, comm_modes=COMM_MODES):
    params = {
        "source_number": origin,
        "destination_number": destination,
        "call_timestamp": str(timestamp),
    }
    if "API" in comm_modes:
        send_api_request("/start_call/", params=params)
    if "FILE_SYSTEM" in comm_modes:
        send_fs_request("/start_call/", params=params)

def signal_end_call(origin, destination, comm_modes=COMM_MODES):
    params = {
        "source_number": origin,
        "destination_number": destination,
    }
    if "API" in comm_modes:
        send_api_request("/end_call_by_number/", params=params)
    if "FILE_SYSTEM" in comm_modes:
        send_fs_request("/end_call_by_number/", params=params)

def test_loop(comm_modes=COMM_MODES):
    print("Iniciando modo de teste (3 chamadas)...")
    call_durations = 3
    test_calls = [
        ("301", "401"),
        ("302", "402"),
        ("303", "403"),
    ]

    for i, (origin, destination) in enumerate(test_calls, start=1):
        start_ts = now()
        print(f"[CALL_ANSWERED] (TEST {i}/3) {origin} -> {destination}")
        signal_start_call(origin, destination, start_ts, comm_modes=comm_modes)
        time.sleep(call_durations)
        duration = now() - start_ts
        print(f"[CALL_END] (TEST {i}/3) {origin} -> {destination} | {duration}s")
        signal_end_call(origin, destination, comm_modes=comm_modes)
        time.sleep(1.5)

    print("Modo de teste finalizado.")

def socket_loop(comm_modes=COMM_MODES):
    """Loop principal com reconexão automática e tratamento de falhas de socket."""
    while True:
        sock = None
        try:
            print(furnish_msg := "Conectando ao AMI do Asterisk...")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(30)  # Timeout para detectar travamentos de rede
            sock.connect((AMI_HOST, AMI_PORT))
            sock.settimeout(None) # Remove timeout para o recv contínuo

            send(sock, "Action: Login")
            send(sock, f"Username: {AMI_USER}")
            send(sock, f"Secret: {AMI_PASS}")
            send(sock, "")

            send(sock, "Action: Events")
            send(sock, "EventMask: on")
            send(sock, "")

            print("Conectado ao AMI com sucesso, aguardando eventos...")
            buffer = ""

            while True:
                data = sock.recv(4096).decode(errors="ignore")
                if not data:
                    print("[WARN] Conexão fechada pelo servidor AMI. Tentando reconectar...")
                    break

                buffer += data

                while "\r\n\r\n" in buffer:
                    raw_event, buffer = buffer.split("\r\n\r\n", 1)

                    if "Event:" not in raw_event:
                        continue

                    event = {}
                    for line in raw_event.splitlines():
                        if ": " in line:
                            k, v = line.split(": ", 1)
                            event[k] = v

                    event_name = event.get("Event")
                    uniqueid = event.get("Uniqueid")
                    channel = event.get("Channel")

                    if not uniqueid or not channel:
                        continue

                    call_id = uniqueid.split(".")[0]

                    call = calls.setdefault(
                        call_id,
                        {
                            "channels": set(),
                            "answered": False,
                            "answered_at": None,
                            "origin": None,
                            "destination": None,
                        },
                    )

                    caller = event.get("CallerIDNum")
                    callee = event.get("ConnectedLineNum")
                    exten = event.get("Exten")

                    if caller in (None, "", "<unknown>"):
                        caller = None
                    if callee in (None, "", "<unknown>"):
                        callee = None
                    if not callee and exten:
                        callee = exten

                    if caller:
                        call["origin"] = caller
                    if callee:
                        call["destination"] = callee

                    if event_name == "Newstate" and event.get("ChannelStateDesc") == "Up":
                        call["channels"].add(channel)

                        if not call["answered"]:
                            call["answered"] = True
                            call["answered_at"] = now()

                            if call["origin"] and call["destination"]:
                                print(f"[CALL_ANSWERED] {call['origin']} -> {call['destination']} | CallID={call_id}")
                                signal_start_call(call["origin"], call["destination"], call["answered_at"], comm_modes=comm_modes)
                            else:
                                print(f"[SKIP START] Dados incompletos: {call}")

                    elif event_name == "Hangup":
                        call["channels"].discard(channel)

                        if call["answered"] and not call["channels"]:
                            ended_at = now()
                            duration = ended_at - call["answered_at"]

                            if call["origin"] and call["destination"]:
                                print(f"[CALL_END] {call['origin']} -> {call['destination']} | {duration}s")
                                signal_end_call(call["origin"], call["destination"], comm_modes=comm_modes)
                            else:
                                print(f"[SKIP END] Dados incompletos: {call}")

                            del calls[call_id]

        except (socket.error, socket.timeout) as e:
            print(f"[ERROR] Erro na conexão com o socket AMI: {e}. Reconectando em 5 segundos...")
            time.sleep(5)
        except KeyboardInterrupt:
            print("Encerrando listener AMI...")
            if sock:
                sock.close()
            break
        finally:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass

def main():
    global fs_sender
    parser = argparse.ArgumentParser(description='Sinaliza inicio e fim de ligações Asterisk para o Backend-Hermes')
    parser.add_argument('monitor_dir', help='Diretório de gravações do Asterisk a ser monitorado')
    parser.add_argument('--output-dir', '-o', help='Diretório para salvar os segmentos')
    parser.add_argument('--communication-modes', '-c', type=str, default='FILE_SYSTEM',
        help=f'Modos de comunicação com o Backend-Hermes "{",".join(COMM_MODES)}" (separados por ,)')
    parser.add_argument('--test', '-t', action='store_true',
        help='Executa o script em modo de teste, simulando o início e fim de 3 chamadas')

    args = parser.parse_args()
    
    output_dir = args.output_dir or os.path.join(args.monitor_dir, 'segments')
    events_dir = os.path.join(output_dir, 'events')
    os.makedirs(events_dir, exist_ok=True)
    comm_modes = [m.strip() for m in args.communication_modes.split(",") if m.strip()]
    
    fs_sender = NetworkJSONSender(events_dir)

    if args.test:
        test_loop(comm_modes=comm_modes)
    else:
        socket_loop(comm_modes=comm_modes)

if __name__ == "__main__":
    main()