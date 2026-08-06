import socket
import time
import requests

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

# =============================
# Estado interno
# =============================
calls = {}

# =============================
# Funções auxilixares
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


def signal_start_call(origin, destination, timestamp):
    params = {
        "source_number": origin,
        "destination_number": destination,
        "call_timestamp": str(timestamp),
    }
    send_api_request("/start_call/", params=params)


def signal_end_call(origin, destination):
    params = {
        "source_number": origin,
        "destination_number": destination,
    }
    send_api_request("/end_call_by_number/", params=params)


# =============================
# Conexão AMI
# =============================
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((AMI_HOST, AMI_PORT))

send(sock, "Action: Login")
send(sock, f"Username: {AMI_USER}")
send(sock, f"Secret: {AMI_PASS}")
send(sock, "")

send(sock, "Action: Events")
send(sock, "EventMask: on")
send(sock, "")

print("Conectado ao AMI, aguardando eventos...")

buffer = ""

# =============================
# Loop principal
# =============================
try:
    while True:
        data = sock.recv(4096).decode(errors="ignore")
        if not data:
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

            # =============================
            # EXTRAÇÃO CORRETA DOS DADOS
            # =============================
            caller = event.get("CallerIDNum")
            callee = event.get("ConnectedLineNum")
            exten = event.get("Exten")

            # limpar valores inválidos
            if caller in (None, "", "<unknown>"):
                caller = None

            if callee in (None, "", "<unknown>"):
                callee = None

            # fallback para destino
            if not callee and exten:
                callee = exten

            # salva apenas se válido
            if caller:
                call["origin"] = caller

            if callee:
                call["destination"] = callee

            # =============================
            # NEWSTATE → CALL_ANSWERED
            # =============================
            if event_name == "Newstate" and event.get("ChannelStateDesc") == "Up":
                call["channels"].add(channel)

                if not call["answered"]:
                    call["answered"] = True
                    call["answered_at"] = now()

                    if call["origin"] and call["destination"]:
                        print(
                            f"[CALL_ANSWERED] {call['origin']} -> {call['destination']} | CallID={call_id}"
                        )

                        signal_start_call(
                            call["origin"],
                            call["destination"],
                            call["answered_at"],
                        )
                    else:
                        print(f"[SKIP START] Dados incompletos: {call}")

            # =============================
            # HANGUP → CALL_END
            # =============================
            elif event_name == "Hangup":
                call["channels"].discard(channel)

                if call["answered"] and not call["channels"]:
                    ended_at = now()
                    duration = ended_at - call["answered_at"]

                    if call["origin"] and call["destination"]:
                        print(
                            f"[CALL_END] {call['origin']} -> {call['destination']} | {duration}s"
                        )

                        signal_end_call(
                            call["origin"],
                            call["destination"],
                        )
                    else:
                        print(f"[SKIP END] Dados incompletos: {call}")

                    del calls[call_id]

except KeyboardInterrupt:
    print("Encerrando listener AMI...")

finally:
    sock.close()
