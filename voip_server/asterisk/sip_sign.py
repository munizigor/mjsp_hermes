from pooch.downloaders import tqdm
from ntpath import basename
from scipy.io import wavfile
import argparse
import json
import os
import sys
import shutil
import socket
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from glob import glob
from datetime import datetime

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

COMM_MODES = ["API", "FILE_SYSTEM"]
DEFAULT_MAX_ATTEMPTS = 3
RECOVERY_INTERVAL_SECONDS = 30 * 60  # 30 minutos
SPEED_MULTI = 1.0
# =============================
# Filtros de ramais
# =============================
# Ramais cujas chamadas devem ser sinalizadas ao Hermes.
# Set vazio = permite todos os ramais, mas continua ignorando canais de fila.
# Esta lista pode ser substituída pela opção --allowed-extensions.
ALLOWED_EXTENSIONS = {"501", "502", "503"}

# Canais internos/de fila não devem disparar sinalização.
QUEUE_CHANNEL_PREFIXES = (
    "Local/",
    "PJSIP/queue",
    "SIP/queue",
)
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEGMENTED_AUDIOS_DIR = f"{PROJ_DIR}/backend/test_audios/segmentado"


def is_queue_channel(channel: str) -> bool:
    """Ignora canais internos ou de filas, como no v5."""
    return channel.startswith(QUEUE_CHANNEL_PREFIXES)


def parse_allowed_extensions(value: str) -> set:
    """Interpreta ramais separados por vírgula; ALL ou * remove a restrição."""
    if value.strip().upper() in ("", "ALL", "*"):
        return set()
    extensions = {number.strip() for number in value.split(",") if number.strip()}
    if not extensions:
        raise ValueError("Informe ramais ou ALL para permitir todos.")
    return extensions


REQUIRED_STATUS_KEYS = {
    "open_calls",
    "closed_calls",
    "audios_received",
    "transcriptions_done",
    "inferences_done",
}


def now() -> int:
    return int(time.time())


# =============================
# Comunicadores
# =============================
class APICommunicator:
    name = "API"

    def __init__(self, host: str, port: int, api_key: str, timeout: int = 5):
        self.base_url = f"http://{host}:{port}"
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Hermes-API-Key": api_key,
        }
        self.timeout = timeout

    def is_usable(self) -> bool:
        url = f"{self.base_url}/status"
        r = requests.get(
            url,
            params={"last_hours": 1},
            headers=self.headers,
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, dict) or not REQUIRED_STATUS_KEYS.issubset(data.keys()):
            raise ValueError(f"Resposta de /status inválida: {data}")
        return True

    def send(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> None:
        url = f"{self.base_url}{endpoint}"
        r = requests.post(
            url,
            params=params,
            headers={"X-Hermes-API-Key": self.headers["X-Hermes-API-Key"]},
            timeout=self.timeout,
        )
        if r.status_code >= 300:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
        print(f"[OK] API {endpoint}")


class FileSystemCommunicator:
    name = "FILE_SYSTEM"

    def __init__(self, shared_dir: Optional[str]):
        self.shared_dir = Path(shared_dir) if shared_dir else None
        self.tmp_dir = self.shared_dir / "tmp" if self.shared_dir else None
        self.new_dir = self.shared_dir / "new" if self.shared_dir else None

    def _ensure_dirs(self) -> None:
        if not self.shared_dir:
            raise RuntimeError("Diretório de eventos (events_dir) não definido.")
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.new_dir.mkdir(parents=True, exist_ok=True)

    def is_usable(self) -> bool:
        self._ensure_dirs()
        probe_file = self.tmp_dir / f".probe_{uuid.uuid4().hex[:8]}"
        probe_token = uuid.uuid4().hex
        try:
            with open(probe_file, "w", encoding="utf-8") as f:
                f.write(probe_token)
                f.flush()
                os.fsync(f.fileno())
            with open(probe_file, "r", encoding="utf-8") as f:
                if f.read() != probe_token:
                    raise IOError("Falha de verificação de leitura no diretório de eventos.")
            return True
        finally:
            if probe_file.exists():
                try:
                    probe_file.unlink()
                except OSError:
                    pass

    def send(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> None:
        self._ensure_dirs()
        prefix = endpoint.strip("/")
        filename = f"{prefix}-{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}.json"
        tmp_path = self.tmp_dir / filename
        new_path = self.new_dir / filename

        payload = {
            "endpoint": endpoint,
            "params": params or {},
            "unix_timestamp": time.time(),
        }

        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, new_path)
            print(f"[OK] FILE_SYSTEM {endpoint} ({filename})")
        except Exception as e:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise RuntimeError(f"Falha ao gravar evento no sistema de arquivos: {e}") from e


# =============================
# Gerenciador de Prioridade e Fallback
# =============================
class FallbackCommunicationManager:
    def __init__(
        self,
        communicators: List[Any],
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        recovery_interval: int = RECOVERY_INTERVAL_SECONDS,
    ):
        if not communicators:
            raise ValueError("Pelo menos um modo de comunicação deve ser configurado.")
        self.communicators = communicators
        self.max_attempts = max_attempts
        self.recovery_interval = recovery_interval
        self.current_index = 0
        self.last_recovery_check = time.time()

        self._select_available_communicator()

    def _retry(self, action: Callable[[], Any], label: str) -> bool:
        for attempt in range(1, self.max_attempts + 1):
            try:
                action()
                return True
            except Exception as e:
                wait_time = attempt * 2
                print(
                    f"[WARN] {label} falhou (tentativa {attempt}/{self.max_attempts}): {e}"
                )
                if attempt < self.max_attempts:
                    print(f"[INFO] Aguardando {wait_time}s antes da próxima tentativa...")
                    time.sleep(wait_time)
        return False

    def _select_available_communicator(self, exclude_index: Optional[int] = None) -> None:
        for idx, comm in enumerate(self.communicators):
            if idx == exclude_index:
                continue
            print(f"[INFO] Verificando disponibilidade do método '{comm.name}'...")
            if self._retry(comm.is_usable, f"Health check [{comm.name}]"):
                self.current_index = idx
                self.last_recovery_check = time.time()
                print(f"[OK] Método de comunicação ativo: '{comm.name}' (prioridade #{idx + 1})")
                return

        self._raise_fatal_error("Nenhum método de comunicação da lista de prioridade está acessível!")

    def maybe_recover_primary(self) -> None:
        if self.current_index == 0:
            return
        if (time.time() - self.last_recovery_check) < self.recovery_interval:
            return

        self.last_recovery_check = time.time()
        primary = self.communicators[0]
        print(f"[INFO] Tentando retornar ao método prioritário '{primary.name}'...")
        if self._retry(primary.is_usable, f"Recuperação [{primary.name}]"):
            self.current_index = 0
            print(f"[OK] Retornou ao método prioritário '{primary.name}'!")
        else:
            current = self.communicators[self.current_index]
            print(f"[WARN] Método prioritário '{primary.name}' indisponível. Mantendo '{current.name}'.")

    def send_event(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> None:
        self.maybe_recover_primary()

        # Ordem de tentativa: método atual primeiro, depois os demais em ordem de prioridade
        order = [self.current_index] + [
            i for i in range(len(self.communicators)) if i != self.current_index
        ]

        for idx in order:
            comm = self.communicators[idx]
            if idx != self.current_index:
                print(f"[WARN] Alternando por fallback para '{comm.name}'...")
                if not self._retry(comm.is_usable, f"Health check [{comm.name}]"):
                    continue

            if self._retry(lambda: comm.send(endpoint, params), f"Envio {endpoint} [{comm.name}]"):
                if idx != self.current_index:
                    self.current_index = idx
                    self.last_recovery_check = time.time()
                    print(f"[OK] Novo método ativo após fallback: '{comm.name}'")
                return

        self._raise_fatal_error(f"Falha ao enviar evento '{endpoint}' em todos os métodos configurados!")

    def signal_start_call(self, origin: str, destination: str, asterisk_id: str) -> None:
        self.send_event(
            "/start_call_asterisk/",
            params={
                "source_number": origin,
                "destination_number": destination,
                "asterisk_id": str(asterisk_id),
            },
        )

    def signal_transfer(self, asterisk_id: str, new_destination: str) -> None:
        self.send_event(
            "/change_destination_number/",
            params={
                "asterisk_id": asterisk_id,
                "new_destination_number": new_destination,
            },
        )

    def signal_end_call_by_number(self, origin: str, destination: str) -> None:
        self.send_event(
            "/end_call_by_number/",
            params={
                "source_number": origin,
                "destination_number": destination,
            },
        )

    def signal_end_call_asterisk(self, asterisk_id: str) -> None:
        self.send_event(
            "/end_call_asterisk/",
            params={
                "asterisk_id": asterisk_id,
            },
        )

    @staticmethod
    def _raise_fatal_error(msg: str) -> None:
        banner = "=" * 70
        error_msg = f"\n{banner}\n[ERRO CRÍTICO DE COMUNICAÇÃO]\n{msg}\n{banner}"
        print(error_msg)
        raise RuntimeError(error_msg)


# =============================
# Cliente / Listener AMI Asterisk
# =============================
class AsteriskAMIListener:
    def __init__(
        self,
        comm_manager: FallbackCommunicationManager,
        host: str = AMI_HOST,
        port: int = AMI_PORT,
        username: str = AMI_USER,
        secret: str = AMI_PASS,
        allowed_extensions: Optional[set] = None,
    ):
        self.comm = comm_manager
        self.host = host
        self.port = port
        self.username = username
        self.secret = secret
        # None = configuração do arquivo; set() = todos os ramais.
        self.allowed_extensions = set(
            ALLOWED_EXTENSIONS if allowed_extensions is None else allowed_extensions
        )
        self.calls: Dict[str, Dict[str, Any]] = {}

    def _is_allowed_extension(self, number: Optional[str]) -> bool:
        """Um número válido passa se estiver na whitelist ou se ela estiver vazia."""
        return bool(number) and (
            not self.allowed_extensions or number in self.allowed_extensions
        )

    def _call_is_allowed(self, call: Dict[str, Any]) -> bool:
        """Mesmo critério do v5: origem OU destino deve estar na whitelist."""
        return self._is_allowed_extension(call.get("origin")) or self._is_allowed_extension(
            call.get("destination")
        )

    @staticmethod
    def _send_line(sock: socket.socket, msg: str) -> None:
        sock.sendall((msg + "\r\n").encode())

    def _connect_and_login(self) -> socket.socket:
        print("Conectando ao AMI do Asterisk...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(30)
        sock.connect((self.host, self.port))
        sock.settimeout(10)  # Timeout curto para permitir checagem periódica de recuperação

        for line in (
            "Action: Login",
            f"Username: {self.username}",
            f"Secret: {self.secret}",
            "",
            "Action: Events",
            "EventMask: on",
            "",
        ):
            self._send_line(sock, line)

        print("Conectado ao AMI com sucesso, aguardando eventos...")
        return sock

    @staticmethod
    def _parse_event(raw_event: str) -> Dict[str, str]:
        event = {}
        for line in raw_event.splitlines():
            if ": " in line:
                k, v = line.split(": ", 1)
                event[k] = v
        return event

    @staticmethod
    def _clean_number(value: Optional[str]) -> Optional[str]:
        return None if value in (None, "", "<unknown>") else value

    def _get_or_create_call(self, call_id: str) -> Dict[str, Any]:
        return self.calls.setdefault(
            call_id,
            {
                "channels": set(),
                "answered": False,
                "answered_at": None,
                "origin": None,
                "destination": None,
                # Guarda o par efetivamente sinalizado, mesmo se eventos
                # subsequentes atualizarem CallerIDNum/ConnectedLineNum.
                "signaled_origin": None,
                "signaled_destination": None,
            },
        )

    def _update_participants(self, call: Dict[str, Any], event: Dict[str, str]) -> None:
        caller = self._clean_number(event.get("CallerIDNum"))
        callee = self._clean_number(event.get("ConnectedLineNum")) or event.get("Exten")

        if caller:
            call["origin"] = caller
        if callee:
            call["destination"] = callee

    def _handle_newstate(self, call_id: str, call: Dict[str, Any], channel: str) -> None:
        if call["answered"]:
            call["channels"].add(channel)
            return

        # Não inicia o registro até que ambos os números estejam disponíveis.
        if not call["origin"] or not call["destination"]:
            print(f"[SKIP START] Dados incompletos: {call}")
            return

        call["channels"].add(channel)
        call["answered_at"] = now()
        print(f"[CALL_ANSWERED] {call['origin']} -> {call['destination']} | CallID={call_id}")
        self.comm.signal_start_call(call["origin"], call["destination"], call_id)
        call["answered"] = True
        call["signaled_origin"] = call["origin"]
        call["signaled_destination"] = call["destination"]

    def _handle_hangup(self, call_id: str, call: Dict[str, Any], channel: str) -> None:
        call["channels"].discard(channel)
        if not call["answered"]:
            # Evita reter estado de chamadas que nunca foram sinalizadas.
            if not call["channels"]:
                self.calls.pop(call_id, None)
            return
        if call["channels"]:
            return

        duration = now() - call["answered_at"]
        origin = call["signaled_origin"]
        destination = call["signaled_destination"]
        if origin and destination:
            print(f"[CALL_END] {origin} -> {destination} | {duration}s")
            self.comm.signal_end_call_by_number(origin, destination)
        else:
            print(f"[SKIP END] Dados incompletos: {call}")

        del self.calls[call_id]

    def _process_raw_event(self, raw_event: str) -> None:
        if "Event:" not in raw_event:
            return

        event = self._parse_event(raw_event)
        # Linkedid identifica a chamada ao longo das pernas/ramais.
        call_id = event.get("Linkedid") or event.get("Uniqueid")
        channel = event.get("Channel")
        if not call_id or not channel:
            return

        # Filtro 1 do v5: desconsidera canais de fila/internos.
        if is_queue_channel(channel):
            return

        call = self._get_or_create_call(call_id)
        self._update_participants(call, event)
        event_name = event.get("Event")

        if event_name == "Newstate" and event.get("ChannelStateDesc") == "Up":
            # Filtro 2 do v5: exige ao menos um ramal permitido.
            if not self._call_is_allowed(call):
                return
            self._handle_newstate(call_id, call, channel)
        elif event_name == "Hangup":
            # Uma chamada já sinalizada deve sempre receber o encerramento,
            # mesmo que o último evento traga outros números (transferências).
            self._handle_hangup(call_id, call, channel)

    def _consume_stream(self, sock: socket.socket) -> None:
        buffer = ""
        while True:
            self.comm.maybe_recover_primary()
            try:
                data = sock.recv(4096).decode(errors="ignore")
            except socket.timeout:
                continue

            if not data:
                print("[WARN] Conexão fechada pelo servidor AMI. Tentando reconectar...")
                break

            buffer += data
            while "\r\n\r\n" in buffer:
                raw_event, buffer = buffer.split("\r\n\r\n", 1)
                self._process_raw_event(raw_event)

    def run(self) -> None:
        while True:
            sock = None
            try:
                sock = self._connect_and_login()
                self._consume_stream(sock)
            except KeyboardInterrupt:
                print("Encerrando listener AMI...")
                break
            except socket.error as e:
                print(f"[ERROR] Erro no socket AMI: {e}. Reconectando em 5 segundos...")
                time.sleep(5)
            finally:
                if sock:
                    try:
                        sock.close()
                    except Exception:
                        pass


# =============================
# Modo de Teste e Inicialização
# =============================

def call_test_cycle(comm_manager: FallbackCommunicationManager, call_info: dict, 
        seg_dir: str, speed, has_transfer=False) -> None:
    from pydub import AudioSegment

    destination = call_info["destination"]
    origin = call_info["origin"]
    call_id = call_info["call_id"]
    start_ts = now() #float
    wavs = call_info["wav_paths"]
    test_original_dir = wavs[0].rsplit("/")[:-1]
    print(f"[TEST CALL] Origem: {origin}, Destino: {destination}, Call ID: {call_id}, Call length: {len(wavs)} audios, located on {'/'.join(test_original_dir)}")

    start_ts_date = datetime.fromtimestamp(start_ts)
    yyyymmdd = start_ts_date.strftime("%Y%m%d")
    hhmm = start_ts_date.strftime("%H%M")
    #{asterisk_id}-{caller}-{call_date}-{call_hour}-{call_timestamp}-{mix|solicitante|atendente}/segment_*.wav
    segments_dir = f"{seg_dir}/{call_id}-{origin}-{yyyymmdd}-{hhmm}-{start_ts}-mix"
    os.makedirs(segments_dir, exist_ok=True)
    try:
        comm_manager.signal_start_call(origin, destination, call_id)

        print(f"[CALL_ANSWERED]{origin} -> {destination}")
        if has_transfer:
            destination, old_dest = (str(int(destination) + 300), destination)
            print(f"[CALL_TRANSFER] {origin} -> {old_dest} -> {destination}")
            time.sleep(2)
            comm_manager.signal_transfer(call_id, destination)
            time.sleep(1)
        for wav_file in call_info["wav_paths"]:
            basename = os.path.basename(wav_file)
            segment = AudioSegment.from_file(wav_file)
            duration = segment.duration_seconds
            shutil.copy(wav_file, f"{segments_dir}/{basename}")
            time.sleep(duration*speed)
        duration = now() - start_ts
        print(f"[CALL_END]{origin} -> {destination} | {duration}s")
        comm_manager.signal_end_call_by_number(origin, destination)
    except Exception as err:
        shutil.rmtree(segments_dir)
        raise err
    shutil.rmtree(segments_dir)

def run_test_mode(comm_manager: FallbackCommunicationManager, 
        test_audios_seg_dir: str, seg_dir: str, speed=SPEED_MULTI) -> None:
    print("Iniciando modo de teste...")
    #we only have 5 audios, so this is safe:
    call_meta_infos = [("301", "401", "17200.22"), 
                ("302", "402", "17301.25"), 
                ("303", "403", "17503.26"),
                ("304", "404", "17700.22"),
                ("305", "405", "17801.25"),
                ("306", "406", "17903.26")]
    
    test_calls = []
    for i, call_dir in enumerate(glob(test_audios_seg_dir+"/*")):
        wav_files = glob(os.path.join(call_dir, "*.wav"))
        if len(wav_files) > 1 and i < len(call_meta_infos):
            metainfo = call_meta_infos[i]
            call_info = {
                "wav_paths": wav_files,
                "call_id": metainfo[2],
                "origin": metainfo[0],
                "destination": metainfo[1]
            }
            test_calls.append(call_info)

    n_transfer = int(len(test_calls) / 2)
    if n_transfer >= 1:
        simple_calls = test_calls[:n_transfer]
        queue_transfer_calls = test_calls[n_transfer:]
    else:
        simple_calls = test_calls
        queue_transfer_calls = []

    bar = tqdm(total=len(test_calls), desc="Testando chamadas")
    print("Testando chamada simples...")
    for i, call_info in enumerate(simple_calls, start=1):
        call_test_cycle(comm_manager, call_info, seg_dir, speed, has_transfer=False)
        bar.update(1)
    
    print("Testando chamada com fila...")
    for i, call_info in enumerate(queue_transfer_calls, start=1):
        call_test_cycle(comm_manager, call_info, seg_dir, speed, has_transfer=True)
        bar.update(1)
    bar.close()
        
    print("Modo de teste finalizado.")


def build_communicators(modes: List[str], events_dir: Optional[str]) -> List[Any]:
    registry = {
        "API": lambda: APICommunicator(API_HOST, API_PORT, API_KEY, API_TIMEOUT),
        "FILE_SYSTEM": lambda: FileSystemCommunicator(events_dir),
    }
    communicators = []
    for mode in modes:
        if mode not in registry:
            raise ValueError(f"Modo de comunicação inválido '{mode}'. Opções: {COMM_MODES}")
        communicators.append(registry[mode]())
    return communicators


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sinaliza início e fim de ligações Asterisk para o Backend-Hermes"
    )
    parser.add_argument(
        "monitor_dir",
        nargs="?",
        default=None,
        help="Diretório de gravações do Asterisk a ser monitorado",
    )
    parser.add_argument(
        "--output-dir", "-o", help="Diretório para salvar os segmentos"
    )
    parser.add_argument(
        "--communication-modes",
        "-c",
        type=str,
        default="API,FILE_SYSTEM",
        help=f'Lista de prioridade de modos de comunicação "{",".join(COMM_MODES)}" (separados por vírgula)',
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help=f"Número de tentativas antes de fallback (padrão: {DEFAULT_MAX_ATTEMPTS})",
    )
    parser.add_argument(
        "--allowed-extensions",
        metavar="RAMAIS",
        default=None,
        help=(
            "Ramais autorizados, separados por vírgula (ex.: 501,502,503). "
            "Padrão: ALLOWED_EXTENSIONS no arquivo. "
            "Use ALL para permitir todos os ramais, exceto canais de fila."
        ),
    )
    parser.add_argument(
        "--test",
        "-t",
        action="store_true",
        help="Executa o script em modo de teste, simulando o início e fim de 3 chamadas",
    )

    parser.add_argument(
        "--seg-dir",
        default=SEGMENTED_AUDIOS_DIR,
        help="Diretório onde estão os segmentos de áudio para teste",
    )
    parser.add_argument("--test-speed", "-ts", type=float, default=SPEED_MULTI)

    args = parser.parse_args()

    events_dir = None
    if args.monitor_dir or args.output_dir:
        output_dir = args.output_dir or os.path.join(args.monitor_dir, "segments")
        events_dir = os.path.join(output_dir, "events")
        os.makedirs(events_dir, exist_ok=True)

    modes = [m.strip().upper() for m in args.communication_modes.split(",") if m.strip()]
    communicators = build_communicators(modes, events_dir)
    comm_manager = FallbackCommunicationManager(
        communicators=communicators,
        max_attempts=args.max_attempts,
    )

    if args.test:
        # O modo --test testa o transporte com chamadas sintéticas,
        # não o filtro AMI de ramais.
        run_test_mode(comm_manager, args.seg_dir, output_dir, speed=args.test_speed)
    else:
        allowed = (
            parse_allowed_extensions(args.allowed_extensions)
            if args.allowed_extensions is not None
            else None
        )
        listener = AsteriskAMIListener(comm_manager, allowed_extensions=allowed)
        if listener.allowed_extensions:
            print(f"[INFO] Ramais sinalizados: {', '.join(sorted(listener.allowed_extensions))}")
        else:
            print("[INFO] Todos os ramais permitidos (exceto canais de fila/internos)")
        listener.run()


if __name__ == "__main__":
    main()
