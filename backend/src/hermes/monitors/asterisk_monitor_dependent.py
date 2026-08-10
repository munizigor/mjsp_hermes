import os
from shutil import copyfile
from glob import glob
import time
import sqlite3 as sqlite
from uuid import uuid1
import multiprocessing as mp
import sys

import soundfile as sf

ASTERISK_DIR = "/asterisk_recordings"
ALLOW_MONO_CHANNEL = True
"""
Diretórios: /asterisk_recordings/exten-102-101-20251107-181136-1762549896.161, 
    /asterisk_recordings/exten-101-102-20251106-164440-1762458280.2656;
Arquivos:
    exten-102-101-20251107-174557-1762548357.69/:
    segment_20251107-174619_0.wav  segment_20251107-174639_4.wav  segment_20251107-174712_8.wav
    segment_20251107-174629_1.wav  segment_20251107-174643_5.wav  segment_20251107-174714_9.wav
    segment_20251107-174635_2.wav  segment_20251107-174702_6.wav
    segment_20251107-174637_3.wav  segment_20251107-174704_7.wav
"""


class CallMonitor:
    def __init__(
        self,
        asterisk_dir,
        origem,
        destino,
        id_emergencia,
        unique_id,
        file_stable_seconds=0.4,
    ):
        self.asterisk_dir = asterisk_dir
        self.file_stable_seconds = file_stable_seconds
        self.origem = origem
        self.destino = destino
        self.id_emergencia = id_emergencia
        self.unique_id = unique_id
        self.base_call_dir = "/tmp/None"
        self.mix_dir = "/tmp/None-mix"
        self.atendente_dir = "/tmp/None-atendente"
        self.solicitante_dir = "/tmp/None-solicitante"
        self.inserted_audios = set()

    def _is_file_ready(self, path):
        """Retorna True se o arquivo existe, tem tamanho > 0 e não foi modificado
        nos últimos self.file_stable_seconds segundos."""
        try:
            st = os.stat(path)
        except FileNotFoundError:
            return False
        if st.st_size == 0:
            return False
        age = time.time() - st.st_mtime
        return age >= self.file_stable_seconds

    def find_call_dir(self):
        if os.path.exists(self.mix_dir) or (
            os.path.exists(self.atendente_dir) and os.path.exists(self.solicitante_dir)
        ):
            return

        if not (
            os.path.exists(self.atendente_dir) and os.path.exists(self.solicitante_dir)
        ):
            print(
                "Call directories not found:",
                self.atendente_dir,
                self.solicitante_dir,
                file=sys.stderr,
            )
        elif not os.path.exists(self.mix_dir):
            print("Call directory not found:", self.mix_dir, file=sys.stderr)

        call_dir_pattern = os.path.join(
            self.asterisk_dir, f"exten-{self.origem}-{self.destino}-*"
        )
        call_dir_pattern2 = os.path.join(
            self.asterisk_dir, f"q-{self.origem}-{self.destino}-*"
        )
        call_dir_pattern3 = os.path.join(
            self.asterisk_dir, f"q-{self.destino}-{self.origem}-*"
        )
        patterns = [call_dir_pattern, call_dir_pattern2, call_dir_pattern3]
        options = []
        for pattern in patterns:
            options.extend(glob(pattern))

        print("Call dir patterns:", patterns, file=sys.stderr)
        print(
            "Options for call",
            self.origem,
            self.destino,
            ":\n",
            options,
            file=sys.stderr,
        )

        options = [
            o
            for o in options
            if abs(float(os.path.basename(o).split("-")[5]) - float(self.unique_id))
            <= 1
        ]

        print(
            "Options for call after filtering:",
            self.origem,
            self.destino,
            ":\n",
            options,
            file=sys.stderr,
        )

        if len(options) > 0:
            self.base_call_dir = (
                options[0]
                .replace("-mix", "")
                .replace("-atendente", "")
                .replace("-solicitante", "")
            )
            self.mix_dir = self.base_call_dir + "-mix"
            if "q-" in self.base_call_dir:
                self.mix_dir = self.base_call_dir
            self.atendente_dir = self.base_call_dir + "-atendente"
            self.solicitante_dir = self.base_call_dir + "-solicitante"

    @property
    def all_channels_dir(self):
        if os.path.exists(self.mix_dir):
            return self.mix_dir
        elif os.path.exists(self.base_call_dir):
            return self.base_call_dir
        else:
            raise Exception(
                "No valid directory found for recording", self.base_call_dir
            )

    def is_dual_channel(self):
        if os.path.exists(self.atendente_dir) and os.path.exists(self.solicitante_dir):
            return True
        return False

    def get_channels(self):
        if self.is_dual_channel():
            return [
                ["Atendente", self.atendente_dir],
                ["Solicitante", self.solicitante_dir],
            ]
        else:
            return [[None, self.all_channels_dir]]

    def insert_previous_audios(self, sqlite_conn):
        cursor = sqlite_conn.cursor()
        audio_paths_result = cursor.execute(
            """
            SELECT audio_path
            FROM emergency_audios
        """
        ).fetchall()
        if audio_paths_result is not None:
            audios_in_db = set([os.path.basename(row[0]) for row in audio_paths_result])
        else:
            audios_in_db = set()
        self.inserted_audios.update(audios_in_db)
        cursor.close()

    def load_audios(self, sqlite_conn):
        try:
            self.find_call_dir()
            channels = self.get_channels()
        except Exception as e:
            print("Error getting all channels dir:", e, file=sys.stderr)
            return

        if self.base_call_dir != None:
            for channel_name, channel_dir in self.get_channels():
                if channel_name == None:
                    if not ALLOW_MONO_CHANNEL:
                        continue
                # print('Reading channel:', channel_dir)

                wavs = glob(os.path.join(channel_dir, "segment_*.wav"))
                for w in wavs:
                    # print(w)
                    filename = os.path.basename(w)
                    if filename in self.inserted_audios:
                        pass
                    elif filename in self.inserted_audios:
                        self.inserted_audios.add(filename)
                    else:
                        # verificar se o arquivo parece estar finalizado (não sendo escrito)
                        if not self._is_file_ready(w):
                            # pular; ficará para a próxima varredura
                            # não marcar como inserido para tentar novamente depois
                            print(
                                "Arquivo ainda em escrita, pulando por enquanto:",
                                w,
                                file=sys.stderr,
                            )
                            continue

                        audio_id = str(uuid1())
                        id_emergencia = self.id_emergencia
                        date_str = filename.split("_")[1]
                        # tentar ler o arquivo de áudio com tratamento de exceção
                        try:
                            print("Loading", w, date_str, file=sys.stderr)
                            dia, hora = date_str.split("-")  # YYYYMMDD HHMMSS
                            start_time = time.strptime(f"{dia} {hora}", "%Y%m%d %H%M%S")
                            start_time_float = time.mktime(start_time)
                            y, sr = sf.read(w)
                        except Exception as e:
                            # se ainda não for legível, pular e tentar novamente depois
                            print(
                                "Não foi possível ler o arquivo (pular por enquanto):",
                                w,
                                str(e),
                                file=sys.stderr,
                            )
                            continue
                        audio_duration = len(y) / sr
                        if audio_duration > 1:
                            db_audios_dir = os.path.join(
                                os.path.dirname(os.environ["SQLITE_DB_PATH"]),
                                str(id_emergencia),
                            )
                            audio_copy_path = db_audios_dir + "/" + filename
                            print(
                                "Original audio shape:",
                                y.shape,
                                "Sample rate:",
                                sr,
                                "Data type:",
                                y.dtype,
                                file=sys.stderr,
                            )
                            cursor = sqlite_conn.cursor()
                            insert_str = f"""
                                INSERT INTO emergency_audios (id, id_emergencia, start_time, transcription_status, sampling_rate, audio_length_seconds, audio_path, channel) 
                                    VALUES ('{audio_id}', {id_emergencia}, {start_time_float}, {0}, {sr}, {audio_duration}, '{audio_copy_path}', '{channel_name}')
                            """
                            # print(insert_str)
                            cursor.execute(insert_str)
                            new_audio_id = cursor.lastrowid

                            if not os.path.exists(db_audios_dir):
                                os.mkdir(db_audios_dir)
                            copyfile(w, audio_copy_path)
                            sqlite_conn.commit()
                        self.inserted_audios.add(filename)


class AsteriskMonitor:
    def __init__(
        self, directory=ASTERISK_DIR, poll_interval_ms=3000, file_stable_seconds=0.4
    ):
        self.directory = directory
        self.poll_interval_ms = poll_interval_ms
        self.file_stable_seconds = file_stable_seconds
        self.calls = {}
        self.cannot_track = set()
        self.stop_signal = mp.Value("H", 0)
        self.dirs_to_ignore = set()
        print("Monitor created!", file=sys.stderr)

    def list_calls(self, sqlite_conn):
        cursor = sqlite_conn.cursor()
        query_results = cursor.execute(
            """
            SELECT source_phone_number, destination_phone_number, asterisk_id, id 
            FROM emergencies 
            WHERE end_time IS NULL
        """
        ).fetchall()
        for open_call in query_results:
            source_phone_number, destination_phone_number, asterisk_id, em_id = (
                open_call
            )
            if em_id in self.calls:
                print("Call already being tracked:", open_call, file=sys.stderr)
                continue
            elif em_id in self.cannot_track:
                print("Call cannot be tracked:", open_call, file=sys.stderr)
                continue
            else:
                print("New call found:", open_call, file=sys.stderr)
                print(
                    "Data types for new call:",
                    [type(x) for x in open_call],
                    file=sys.stderr,
                )
                if (
                    source_phone_number is not None
                    and destination_phone_number is not None
                ):
                    try:
                        new_call = CallMonitor(
                            self.directory,
                            source_phone_number,
                            destination_phone_number,
                            em_id,
                            asterisk_id,
                            file_stable_seconds=self.file_stable_seconds,
                        )
                        new_call.find_call_dir()
                        new_call.insert_previous_audios(sqlite_conn)
                        self.calls[em_id] = new_call
                        print("Call added to self.calls:", self.calls, file=sys.stderr)
                    except Exception as e:
                        print("Error creating new call:", e, file=sys.stderr)
                        continue
                else:
                    self.cannot_track.add(em_id)
                    print(
                        "Call has no source or destination phone number, cannot track:",
                        open_call,
                    )

    def monitor_recordings(self):
        # Logic to monitor the Asterisk recordings directory
        while self.stop_signal.value == 0:
            operations_start = time.time()
            sqlite_conn = sqlite.connect(os.environ["SQLITE_DB_PATH"])
            print("Polling Asterisk recordings directory...", file=sys.stderr)
            self.list_calls(sqlite_conn)
            # print('Linking calls to emergencies...', file=sys.stderr)
            print("Inserting new audios into database...", file=sys.stderr)
            for em_id, call in self.calls.items():
                try:
                    print("Loading audios for call:", em_id, file=sys.stderr)
                    channels = call.get_channels()
                    print("Channels for call:", channels, file=sys.stderr)
                    call.load_audios(sqlite_conn)
                except Exception as e:
                    print(
                        "Error loading audios for call:",
                        em_id,
                        "Error:",
                        e,
                        file=sys.stderr,
                    )
                    continue
            sqlite_conn.close()
            operations_duration = time.time() - operations_start
            # print('Asterisk Monitor operations duration:', operations_duration, file=sys.stderr)
            time_left = self.poll_interval_ms - operations_duration * 1000
            if time_left > 0:
                time.sleep(time_left / 1000.0)
        print("Asterisk Monitor stopping...", file=sys.stderr)

    def stop(self):
        self.stop_signal.value = 1


def start_asterisk_monitor():

    monitor = AsteriskMonitor()
    stop_signal = monitor.stop_signal

    # Start process to monitor recordings, without blocking
    monitor_process = mp.Process(target=monitor.monitor_recordings)
    monitor_process.start()
    print("Monitor started!", file=sys.stderr)

    return monitor, monitor_process, stop_signal
