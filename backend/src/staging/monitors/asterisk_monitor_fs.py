import os
import time
import json
import sqlite3 as sqlite
import sys
import multiprocessing as mp

# Reuse the dependent monitor's classes and constants
from monitors.asterisk_monitor_dependent import AsteriskMonitor, CallMonitor, ALLOW_MONO_CHANNEL

class NetworkJSONReader:
    """Reads file-based JSON messages atomically using the tmp/ -> new/ -> cur/ pattern."""
    def __init__(self, shared_dir: str):
        self.shared_dir = os.path.abspath(shared_dir)
        self.new_dir = os.path.join(self.shared_dir, "new")
        self.cur_dir = os.path.join(self.shared_dir, "cur")
        
        os.makedirs(self.new_dir, exist_ok=True)
        os.makedirs(self.cur_dir, exist_ok=True)

    def fetch_pending(self):
        claimed_messages = []
        try:
            with os.scandir(self.new_dir) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name.endswith(".json"):
                        new_path = entry.path
                        cur_path = os.path.join(self.cur_dir, entry.name)
                        try:
                            # Atomic claim
                            os.replace(new_path, cur_path)
                            with open(cur_path, "r", encoding="utf-8") as f:
                                payload = json.load(f)
                            claimed_messages.append((entry.name, payload))
                        except Exception:
                            continue
        except FileNotFoundError:
            pass
        if len(claimed_messages) > 0:
            #sort
            claimed_messages.sort(key=lambda x: (x[1].get("unix_timestamp", 0.0), x[0]))
        return claimed_messages

    def acknowledge(self, msg_id: str):
        cur_path = os.path.join(self.cur_dir, msg_id)
        try:
            if os.path.exists(cur_path):
                os.unlink(cur_path)
        except OSError:
            pass


class FilesystemEventDrivenAsteriskMonitor(AsteriskMonitor):
    """
    Inherits from AsteriskMonitor (which waits for calls to exist in SQLite 
    via open emergencies) and adds filesystem-based event processing to 
    start and end those calls.
    """
    def __init__(self, directory, events_dir=None, poll_interval_ms=3000, file_stable_seconds=0.4):
        super().__init__(directory=directory, poll_interval_ms=poll_interval_ms, file_stable_seconds=file_stable_seconds)
        self.events_dir = events_dir or os.path.join(directory, "events")
        self.json_reader = NetworkJSONReader(self.events_dir)
        print("Filesystem Event-Driven Asterisk Monitor created!", file=sys.stderr)

    def process_incoming_events(self, sqlite_conn):
        """Processes file-based start/end call control events matching endpoint behaviors."""
        messages = self.json_reader.fetch_pending()
        for msg_id, payload in messages:
            endpoint = payload.get("endpoint")
            params = payload.get("params", {})
            
            try:
                if endpoint == "/start_call/":
                    source_number = params.get("source_number")
                    destination_number = params.get("destination_number")
                    call_timestamp = params.get("call_timestamp")
                    
                    if source_number and destination_number and str(source_number).isdigit() and str(destination_number).isdigit():
                        cursor = sqlite_conn.cursor()
                        cursor.execute("""
                            INSERT INTO emergencies (source_phone_number, destination_phone_number, asterisk_id) 
                            VALUES (?, ?, ?)
                        """, (str(source_number), str(destination_number), str(call_timestamp)))
                        new_emergency_id = cursor.lastrowid
                        sqlite_conn.commit()

                        cursor.execute("""
                            INSERT INTO call_logs (id_emergencia, event_type, source_phone_number, destination_phone_number) 
                            VALUES (?, ?, ?, ?)
                        """, (new_emergency_id, "START", str(source_number), str(destination_number)))
                        sqlite_conn.commit()
                        print(f"[FILE_EVENT] Nova emergencia entre numeros {source_number} {destination_number} (ID: {new_emergency_id})", file=sys.stderr)
                    else:
                        print(f"[FILE_EVENT ERROR] /start_call/ com números inválidos: {params}", file=sys.stderr)

                elif endpoint == "/end_call_by_number/":
                    source_number = params.get("source_number")
                    destination_number = params.get("destination_number")
                    
                    if source_number and destination_number:
                        cursor = sqlite_conn.cursor()
                        cursor.execute("""
                            UPDATE emergencies
                            SET end_time = CURRENT_TIMESTAMP, current_status = 2
                            WHERE end_time IS NULL 
                                AND source_phone_number = ? 
                                AND destination_phone_number = ?
                        """, (str(source_number), str(destination_number)))
                        sqlite_conn.commit()

                        cursor.execute("""
                            INSERT INTO call_logs (event_type, source_phone_number, destination_phone_number) 
                            VALUES (?, ?, ?)
                        """, ("END", str(source_number), str(destination_number)))
                        sqlite_conn.commit()
                        print(f"[FILE_EVENT] Emergency ended {source_number} {destination_number}", file=sys.stderr)
                
                # Acknowledge/delete the successfully processed event file
                self.json_reader.acknowledge(msg_id)
            except Exception as e:
                print(f"[FILE_EVENT ERROR] Falha ao processar evento {endpoint}: {e}", file=sys.stderr)

    def monitor_recordings(self):
        """Main polling loop extending the dependent monitor behavior with filesystem events."""
        while self.stop_signal.value == 0:
            operations_start = time.time()
            sqlite_conn = sqlite.connect(os.environ["SQLITE_DB_PATH"])
            
            # 1. Process control events from filesystem first (inserts/updates emergencies in SQLite)
            self.process_incoming_events(sqlite_conn)
            
            # 2. Re-use dependent monitor logic: list open calls from SQLite & load audios
            self.list_calls(sqlite_conn)
            for em_id, call in self.calls.items():
                try:
                    call.load_audios(sqlite_conn)
                except Exception as e:
                    print(f"Error loading audios for call {em_id}: {e}", file=sys.stderr)
                    continue
                    
            sqlite_conn.close()
            
            operations_duration = time.time() - operations_start
            time_left = self.poll_interval_ms - operations_duration * 1000
            if time_left > 0:
                time.sleep(time_left / 1000.0)
        print("Filesystem Event-Driven Monitor stopping...", file=sys.stderr)


def start_asterisk_monitor(recordings_dir: str, events_dir=None):
    events_dir = os.path.join(recordings_dir, "events")
    monitor = FilesystemEventDrivenAsteriskMonitor(recordings_dir, events_dir=events_dir)
    stop_signal = monitor.stop_signal

    # Determine the target method cleanly
    target_method = monitor.monitor_loop if hasattr(monitor, 'monitor_loop') else monitor.monitor_recordings

    # Start process to monitor recordings, without blocking
    monitor_process = mp.Process(target=target_method)
    monitor_process.start()
    print("Filesystem Event-Driven Monitor process started!", file=sys.stderr)

    return monitor, monitor_process, stop_signal