import sys
import queue
import multiprocessing as mp
import sqlite3 as sqlite
from time import sleep, time
import os

import tempfile
from pydub import AudioSegment

from runners.asr_utils import env_vals


def worker_sequence_workload(audio_queue):
    sqlite_conn = sqlite.connect(os.environ["SQLITE_DB_PATH"])
    cursor = sqlite_conn.cursor()

    # Reset any items that were left marked as "processing"
    try:
        cursor.execute(
            """
            UPDATE emergency_audios
            SET transcription_status = 0
            WHERE transcription_status = 1
        """
        )
        sqlite_conn.commit()
        print("worker_sequence_workload: reset transcription_status 1 -> 0")
    except sqlite.OperationalError as err:
        print(
            "worker_sequence_workload: Error resetting transcription_status:",
            err,
            file=sys.stderr,
        )

    # Dictionary to hold buffers per emergency
    # Format: { id_emergencia: {"audios": [], "duration": 0.0, "first_added": timestamp} }
    buffers = {}

    # Dictionary with temp files created and their creation times
    temp_files = {}

    last_buffer_print_time = time()
    buffer_print_interval = 10

    while True:

        for temp_path in list(temp_files.keys()):
            if os.path.exists(temp_path):
                age = time() - temp_files[temp_path]
                if age > 30 * 60:
                    try:
                        os.remove(temp_path)
                    except Exception as err:
                        pass
                    del temp_files[temp_path]
            else:
                del temp_files[temp_path]

        try:
            # Fetch a batch instead of LIMIT 1 to group them efficiently
            pending_audios = cursor.execute(
                """
                UPDATE emergency_audios
                SET transcription_status = 1
                WHERE id IN (
                    SELECT id FROM emergency_audios
                    WHERE transcription_status = 0
                    ORDER BY start_time ASC
                    LIMIT 20
                )
                RETURNING id, id_emergencia, start_time, sampling_rate, audio_path, channel
            """
            ).fetchall()

            for row in pending_audios:
                (
                    audio_id,
                    id_emergencia,
                    start_time,
                    sampling_rate,
                    audio_path,
                    channel,
                ) = row

                # Immediately mark as processing so we don't fetch it again
                # cursor.execute('UPDATE emergency_audios SET transcription_status = 1 WHERE id = ?', (audio_id,))
                buffer_key = (id_emergencia, channel)
                if buffer_key not in buffers:
                    buffers[buffer_key] = {
                        "audios": [],
                        "duration": 0.0,
                        "first_added": time(),
                        "id_emergencia": id_emergencia,
                        "channel": channel,
                    }

                try:
                    # Load audio to get duration and prep for potential concatenation
                    segment = AudioSegment.from_file(audio_path)
                    duration = segment.duration_seconds
                except Exception as e:
                    print(f"Error loading {audio_path}: {e}", file=sys.stderr)
                    # If it fails, revert status to 0 or handle error appropriately
                    cursor.execute(
                        "UPDATE emergency_audios SET transcription_status = 0 WHERE id = ?",
                        (audio_id,),
                    )
                    continue

                buffers[buffer_key]["audios"].append(
                    {
                        "id": audio_id,
                        "start_time": start_time,
                        "added_at": time(),
                        "sampling_rate": sampling_rate,
                        "audio_path": audio_path,
                        "segment": segment,
                        "duration": duration,
                    }
                )
                buffers[buffer_key]["duration"] += duration

            sqlite_conn.commit()

            # Check all buffers to see if they meet dispatch conditions
            now = time()
            # list() is used because we delete keys during iteration
            max_wait = 10.0
            min_duration = 7.0
            max_duration = 29.9  # Maximum length for the chunk
            buffer_keys = buffers.keys()
            for buffer_key in list(buffer_keys):
                buf = buffers[buffer_key]
                id_emergencia, channel = buffer_key
                # Condition: >= 4 seconds of audio OR oldest audio waiting > 7 seconds
                if (
                    buf["duration"] >= min_duration
                    or (now - buf["first_added"]) > max_wait
                ):
                    if not buf["audios"]:
                        del buffers[buffer_key]
                        continue

                    # Inherit metadata from the chronologically first audio segment
                    buf["audios"].sort(key=lambda x: x["start_time"])

                    combined_audio = AudioSegment.empty()
                    audio_ids = []
                    current_dur = 0.0

                    # Pack audios until we hit the 29-second limit
                    for item in buf["audios"]:
                        # If adding this audio exceeds the max length (and we already have at least 1 audio), stop packing.
                        if (
                            current_dur + item["duration"] > max_duration
                            and len(audio_ids) > 0
                        ):
                            break

                        combined_audio += item["segment"]
                        audio_ids.append(item["id"])
                        current_dur += item["duration"]

                    # Optimization: If only 1 audio was packed, just use its original file path
                    if len(audio_ids) == 1:
                        # Find the packed audio to get its path
                        packed_audio = next(
                            item for item in buf["audios"] if item["id"] == audio_ids[0]
                        )
                        temp_path = packed_audio["audio_path"]
                    else:
                        # Export combined audio to a temporary file
                        temp_fd, temp_path = tempfile.mkstemp(suffix=".temp_concat.wav")
                        os.close(
                            temp_fd
                        )  # Close file descriptor so pydub can write to it
                        combined_audio.export(temp_path, format="wav")
                        temp_files[temp_path] = time()

                    # Grab metadata from the first audio that made it into this chunk
                    first_packed_audio = next(
                        item for item in buf["audios"] if item["id"] == audio_ids[0]
                    )
                    start_time = first_packed_audio["start_time"]
                    sampling_rate = first_packed_audio["sampling_rate"]

                    # Note: We are passing a list of audio_ids as the first element now
                    new_task = (
                        audio_ids,
                        id_emergencia,
                        start_time,
                        sampling_rate,
                        temp_path,
                        channel,
                    )
                    print("Enfileirando batch:", temp_path, "IDs:", audio_ids)

                    audio_queue.put(new_task)

                    # Keep whatever didn't fit in the 29s chunk in the buffer for the next loop iteration
                    remaining_buffer = [
                        item for item in buf["audios"] if item["id"] not in audio_ids
                    ]
                    if len(remaining_buffer) > 0:
                        buffers[buffer_key] = {
                            "audios": remaining_buffer,
                            "duration": sum([r["duration"] for r in remaining_buffer]),
                            "first_added": min(
                                [r["added_at"] for r in remaining_buffer]
                            ),
                            "id_emergencia": id_emergencia,
                            "channel": channel,
                        }
                    else:
                        del buffers[buffer_key]

            # Print current size of buffers and queue every 10 seconds
            if time() - last_buffer_print_time > buffer_print_interval:
                total_items = 0
                total_duration = 0

                for buf in buffers.values():
                    total_items += len(buf["audios"])
                    total_duration += buf["duration"]

                print(
                    f"Buffers size: {total_items} | Buffers duration: {total_duration} | Queue size: {audio_queue.qsize()}"
                )
                last_buffer_print_time = time()

        except sqlite.OperationalError as err:
            print(err, file=sys.stderr)
            print("Sqlite error", file=sys.stderr)
            sqlite_conn.close()
            sqlite_conn = sqlite.connect(os.environ["SQLITE_DB_PATH"])
            cursor = sqlite_conn.cursor()
            sleep(1)

        sleep(0.05)


def worker_save_everything(result_queue):
    sqlite_conn = sqlite.connect(os.environ["SQLITE_DB_PATH"])
    cursor = sqlite_conn.cursor()
    while True:
        try:
            to_save = result_queue.get(timeout=0.1)
        except queue.Empty:
            to_save = None

        if to_save is not None:
            if isinstance(to_save, dict):
                audio_ids = to_save["audio_id"]
                id_emergencia = to_save["id_emergencia"]

                # Standardize to list so we can iterate safely
                if not isinstance(audio_ids, list):
                    audio_ids = [audio_ids]

                if "error" not in to_save:
                    transcript = to_save["part"]
                    # Attach the combined transcript to the first audio ID in the DB
                    first_audio_id = audio_ids[0]

                    insert_str = """
                        INSERT INTO emergency_transcripts (id_emergencia, audio_id, part, start_time, 
                                horario_de_transcricao, is_analyzed, transcription_seconds, transcription_model, actor) 
                            VALUES (?,?,?,?,?,?,?,?,?)
                    """
                    cursor.execute(
                        insert_str,
                        (
                            id_emergencia,
                            first_audio_id,
                            transcript,
                            to_save["start_time"],
                            time(),
                            False,
                            to_save["transcription_seconds"],
                            to_save["transcription_model"],
                            to_save["actor"],
                        ),
                    )

                    # Mark all concatenated chunks as successfully transcribed
                    for aid in audio_ids:
                        cursor.execute(
                            "UPDATE emergency_audios SET transcription_status = 2 WHERE id = ?",
                            (aid,),
                        )
                else:
                    # Revert all items to 0 if the ASR model errored out
                    for aid in audio_ids:
                        cursor.execute(
                            "UPDATE emergency_audios SET transcription_status = 0 WHERE id = ?",
                            (aid,),
                        )

                sqlite_conn.commit()

                # Optional but highly recommended: Clean up the temporary concatenated audio file
                # Assuming your ASR worker packs the temp 'audio_path' into the dict
                try:
                    if "audio_path" in to_save and os.path.exists(
                        to_save["audio_path"]
                    ):
                        if "temp_concat" in to_save["audio_path"]:
                            os.remove(to_save["audio_path"])
                except Exception as e:
                    print(f"Temp file cleanup failed: {e}", file=sys.stderr)

            else:
                print("worker_save_everything: Unknown format to save")
        sleep(0.025)


class AsrRunner:

    def __init__(
        self,
        whisper_mname: str,
        n_asr_workers: int = 1,
        hardware_config: str = "cuda",
        language: str = "pt",
        n_cpus: int = 6,
    ):
        transcript_queue = mp.Queue()
        result_queue = mp.Queue()
        # calls_being_processed_lockset = mp.Array('i', range(n_asr_workers))
        workers = []

        # stop_flag = mp.Value('H', 0)

        # mp.set_start_method('spawn')

        if hardware_config in ["cpu", "cuda"]:
            from runners.local_whisper import whisper_worker_process

            worker_func = whisper_worker_process
        elif hardware_config == "azure-api":
            from runners.azure import azure_caller_process

            worker_func = azure_caller_process
        elif hardware_config == "triton-server":
            from runners.triton import triton_caller_process

            worker_func = triton_caller_process
        elif hardware_config == "gcp-api":
            from runners.chirp import google_caller_process

            worker_func = google_caller_process
        elif hardware_config == "flask-server":
            from runners.flask_server import flask_caller_process

            worker_func = flask_caller_process
        elif hardware_config == "vllm-api":
            from runners.vllm_server import vllm_caller_process

            worker_func = vllm_caller_process
        else:
            raise Exception("Invalid hardware-config for ASR")

        for worker_index in range(n_asr_workers):
            print("AsrRunner: creating processes")
            p = mp.Process(
                target=worker_func,
                args=(
                    whisper_mname,
                    transcript_queue,
                    result_queue,
                    hardware_config,
                    language,
                    n_cpus,
                ),
            )
            workers.append(p)
        # mp.set_start_method('fork')
        sequencer_worker = mp.Process(
            target=worker_sequence_workload, args=(transcript_queue,)
        )
        saver_worker = mp.Process(target=worker_save_everything, args=(result_queue,))

        try:
            # Setting processes as daemons, in order to prevent becoming zombies
            for p in workers:
                p.daemon = True
            sequencer_worker.daemon = True
            saver_worker.daemon = True
        except Exception as err:
            print(err)
            print("Failed to set future processes as daemons")

        print("AsrRunner: starting process")
        for p in workers:
            p.start()
        sequencer_worker.start()
        saver_worker.start()

        print("AsrRunner: started process")


if __name__ == "__main__":
    from runners.azure import call_azure_fast

    audio_path = sys.argv[1]
    resp = call_azure_fast(audio_path)
    print(resp)
