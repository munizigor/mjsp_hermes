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
ALLOW_MONO_CHANNEL = False
'''
Diretórios: /asterisk_recordings/exten-102-101-20251107-181136-1762549896.161, 
    /asterisk_recordings/exten-101-102-20251106-164440-1762458280.2656;
Arquivos:
    exten-102-101-20251107-174557-1762548357.69/:
    segment_20251107-174619_0.wav  segment_20251107-174639_4.wav  segment_20251107-174712_8.wav
    segment_20251107-174629_1.wav  segment_20251107-174643_5.wav  segment_20251107-174714_9.wav
    segment_20251107-174635_2.wav  segment_20251107-174702_6.wav
    segment_20251107-174637_3.wav  segment_20251107-174704_7.wav
'''

class SplittedRecording:
    def __init__(self, path, file_stable_seconds=0.4):
        self.file_stable_seconds = file_stable_seconds
        self.path = path
        self.mix_dir = path + '-mix'
        self.atendente_dir = path + '-atendente'
        self.solicitante_dir = path + '-solicitante'
        parts = os.path.basename(path).split("-")
        assert len(parts) == 6, "Invalid recording filename format"
        self.origem = parts[1]
        self.destino = parts[2]
        self.dia = parts[3] #AAAAMMDD
        self.hora = parts[4] # HHMMSS
        self.unique_id = parts[5]
        self.id_emergencia = None
        self.inserted_audios = set()
        self.closed = False
        self.short_audios_cache = []

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
    
    @property
    def all_channels_dir(self):
        if os.path.exists(self.mix_dir):
            return self.mix_dir
        elif os.path.exists(self.path):
            return self.path
        else:
            raise Exception("No valid directory found for recording", self.path)
    
    def is_dual_channel(self):
        if os.path.exists(self.atendente_dir) and os.path.exists(self.solicitante_dir):
            return True
        return False
    
    def get_channels(self):
        if self.is_dual_channel():
            return [['Atendente', self.atendente_dir], ['Solicitante', self.solicitante_dir]]
        else:
            return [[None, self.all_channels_dir]]

    def load_audios(self, sqlite_conn, audios_in_db):
        for channel_name, channel_dir in self.get_channels():
            if channel_name == None:
                if not ALLOW_MONO_CHANNEL:
                    continue
            #print('Reading channel:', channel_dir)

            wavs = glob(os.path.join(channel_dir, "segment_*.wav"))
            for w in wavs:
                #print(w)
                filename = os.path.basename(w)
                if filename in self.inserted_audios:
                    pass
                elif filename in audios_in_db:
                    self.inserted_audios.add(filename)
                else:
                    # verificar se o arquivo parece estar finalizado (não sendo escrito)
                    if not self._is_file_ready(w):
                        # pular; ficará para a próxima varredura
                        # não marcar como inserido para tentar novamente depois
                        print('Arquivo ainda em escrita, pulando por enquanto:', 
                            w, file=sys.stderr)
                        continue

                    audio_id = str(uuid1())
                    id_emergencia = self.id_emergencia
                    date_str = filename.split('_')[1]
                    # tentar ler o arquivo de áudio com tratamento de exceção
                    try:
                        print('Loading', w, date_str, file=sys.stderr)
                        dia, hora = date_str.split('-') #YYYYMMDD HHMMSS
                        start_time = time.strptime(f"{dia} {hora}", "%Y%m%d %H%M%S")
                        start_time_float = time.mktime(start_time)
                        y, sr = sf.read(w)
                    except Exception as e:
                        # se ainda não for legível, pular e tentar novamente depois
                        print('Não foi possível ler o arquivo (pular por enquanto):', 
                            w, str(e), file=sys.stderr)
                        continue
                    audio_duration = len(y) / sr
                    if audio_duration > 1:
                        db_audios_dir = os.path.join(os.path.dirname(os.environ["SQLITE_DB_PATH"]), 
                            str(id_emergencia))
                        audio_copy_path = db_audios_dir + '/' + filename
                        print('Original audio shape:', y.shape, 'Sample rate:', sr, 
                            'Data type:', y.dtype, file=sys.stderr)
                        cursor = sqlite_conn.cursor()
                        insert_str = f'''
                            INSERT INTO emergency_audios (id, id_emergencia, start_time, transcription_status, sampling_rate, audio_length_seconds, audio_path, channel) 
                                VALUES ('{audio_id}', {id_emergencia}, {start_time_float}, {0}, {sr}, {audio_duration}, '{audio_copy_path}', '{channel_name}')
                        '''
                        #print(insert_str)
                        cursor.execute(insert_str)
                        new_audio_id = cursor.lastrowid
                        
                        if not os.path.exists(db_audios_dir):
                            os.mkdir(db_audios_dir)
                        copyfile(w, audio_copy_path)
                        sqlite_conn.commit()
                    self.inserted_audios.add(filename)

class AsteriskMonitor:
    def __init__(self, directory=ASTERISK_DIR, poll_interval_ms=300,
            file_stable_seconds=0.4):
        self.directory = directory
        self.poll_interval_ms = poll_interval_ms
        self.file_stable_seconds = file_stable_seconds
        self.calls = {}
        self.stop_signal = mp.Value('H', 0)
        self.dirs_to_ignore = set()
        print('Monitor created!', file=sys.stderr)

    def list_calls(self, sqlite_conn):
        #exten-<number_1>-<number_2>-<data>-<hora>-<timestamp>[-[mix|atendente|solicitante]]
        recording_dirs = glob(os.path.join(self.directory, "exten-*"))
        recording_dirs_no_redundancy = [d.replace('-mix', '').replace('-atendente', '').replace('-solicitante', '') for d in recording_dirs]
        for rec_dir in recording_dirs_no_redundancy:
            if not (rec_dir in self.calls or rec_dir in self.dirs_to_ignore):
                print(rec_dir, file=sys.stderr)
                try:
                    recording = SplittedRecording(rec_dir, self.file_stable_seconds)
                    if not ALLOW_MONO_CHANNEL and not recording.is_dual_channel():
                        print('Chamada não é dual channel, pulando', file=sys.stderr)
                        self.dirs_to_ignore.add(rec_dir)
                        continue
                    else:
                        self.calls[rec_dir] = recording
                        print('Diretorio de chamada detectado:', 
                            recording.unique_id, file=sys.stderr)
                except AssertionError:
                    print('Formato de nome de gravação inválido:', rec_dir, file=sys.stderr)
                    continue
    
    def link_calls_to_emergencies(self, sqlite_conn):
        # Logic to link recordings to call records in the database

        cursor = sqlite_conn.cursor()
        query_results = cursor.execute('''
            SELECT asterisk_id, id
            FROM emergencies
        ''').fetchall()
        if query_results is not None:
            asterisk2emergencia = {row[0]: row[1] for row in query_results}
        else:
            asterisk2emergencia = {}
        
        not_linked1 = [r for r in self.calls.values() if r.id_emergencia is None]
        for asterisk_id, id_emergencia in asterisk2emergencia.items():
            for r in not_linked1:
                if r.unique_id == asterisk_id:
                    r.id_emergencia = id_emergencia
                    print('Linked call', r.unique_id, 'to existing emergency id', r.id_emergencia, file=sys.stderr)

        not_linked2 = [r for r in self.calls.values() if r.id_emergencia is None]
        for r in not_linked2:
            operator_unique_code = r.destino
            asterisk_id = r.unique_id
            cursor = sqlite_conn.cursor()
            cursor.execute('''
                INSERT INTO emergencies (operator_unique_code,asterisk_id,source_phone_number,destination_phone_number) VALUES (?,?,?,?)
            ''', (operator_unique_code,asterisk_id,r.origem,r.destino))
            r.id_emergencia = cursor.lastrowid
            sqlite_conn.commit()
            print('Linked call', r.unique_id, 'to new emergency id', r.id_emergencia, file=sys.stderr)

    def insert_new_audios(self, sqlite_conn):
        cursor = sqlite_conn.cursor()
        audio_paths_result = cursor.execute('''
            SELECT audio_path
            FROM emergency_audios
        ''').fetchall()
        if audio_paths_result is not None:
            audios_in_db = set([os.path.basename(row[0]) for row in audio_paths_result])
        else:
            audios_in_db = set()
        cursor.close()
        for call_data in self.calls.values():
            call_data.load_audios(sqlite_conn, audios_in_db)

    def monitor_recordings(self):
        # Logic to monitor the Asterisk recordings directory
        while self.stop_signal.value == 0:
            operations_start = time.time()
            sqlite_conn = sqlite.connect(os.environ["SQLITE_DB_PATH"])
            #print('Polling Asterisk recordings directory...', file=sys.stderr)
            self.list_calls(sqlite_conn)
            #print('Linking calls to emergencies...', file=sys.stderr)
            self.link_calls_to_emergencies(sqlite_conn)
            #print('Inserting new audios into database...', file=sys.stderr)
            self.insert_new_audios(sqlite_conn)
            sqlite_conn.close()
            operations_duration = time.time() - operations_start
            #print('Asterisk Monitor operations duration:', operations_duration, file=sys.stderr)
            time_left = self.poll_interval_ms - operations_duration*1000
            if time_left > 0:
                time.sleep(time_left / 1000.0)
        print('Asterisk Monitor stopping...', file=sys.stderr)

    def stop(self):
        self.stop_signal.value = 1

def start_asterisk_monitor():
    
    monitor = AsteriskMonitor()
    stop_signal = monitor.stop_signal

    #Start process to monitor recordings, without blocking
    monitor_process = mp.Process(target=monitor.monitor_recordings)
    monitor_process.start()
    print('Monitor started!', file=sys.stderr)

    return monitor, monitor_process, stop_signal
