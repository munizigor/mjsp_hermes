import json
import queue
import multiprocessing as mp
import threading
import time
from glob import glob

from pydub import AudioSegment

from names import asr_models
from runners.worker_selector import select_worker

'''
Test the ASR inference server. Mounting requirements:
config.json -> /config.json
src/asr -> /src && cd src/ && python3 test_quick.py
test_audios/ -> /test_audios
'''

config_json = json.load(open('/config.json', 'r'))
asr_config = config_json.get('asr', {})
hardware_config = asr_config.get('hardware_config', {})
asr_model = asr_config.get('model', 'large')
language = asr_config.get('language', 'pt')
n_asr_workers = asr_config.get('n_asr_workers', 1)
n_cpus = asr_config.get('n_cpus', 6)

if asr_model in asr_models:
    full_mname = asr_models[asr_model]
else:
    full_mname = asr_model

worker_func = select_worker(hardware_config)
transcript_queue = mp.Queue()
result_queue = mp.Queue()

'''worker_proc = mp.Process(
    target=worker_func,
    args=(
        full_mname,
        transcript_queue,
        result_queue,
        hardware_config,
        language,
        n_cpus,
    ),
)'''
worker_thread = threading.Thread(
    target=worker_func,
    args=(
        full_mname,
        transcript_queue,
        result_queue,
        hardware_config,
        language,
        n_cpus,
    ),
)
worker_thread.start()

audios = glob('test_audios/*.wav')

durations = []
errors = []

for audio in audios:
    print(f"Testing {audio}...")
    segment = AudioSegment.from_file(audio)
    duration = segment.duration_seconds
    wait_time_secs = duration*1.25

    new_task = (
        [1],
        '1',
        time.time(),
        16000,
        audio,
        'Canal Misto',
    )
    transcription_start = time.time()
    transcript_queue.put(new_task)

    '''
    {
        "audio_id": audio_id,
        "id_emergencia": id_emergencia,
        "part": transcript,
        "start_time": inf_start,
        "transcription_seconds": total_inf_time,
        "transcription_model": model_name,
        "actor": channel,
    }
    '''
    try:
        transcription_result = result_queue.get(block=True, timeout=wait_time_secs)
        transcription_end = time.time()
        transcription_duration = transcription_end - transcription_start

        print(f"Transcription: {transcription_result['part']}")
        print(f"Transcription Duration: {transcription_duration}")
        print(f"Audio Duration: {duration}")
        print(f"Speedup: {duration/transcription_duration}")
        durations.append((audio, duration, transcription_duration))
    except queue.Empty as err:
        print(f"Timeout waiting for transcription of {audio}: {wait_time_secs}s")
        errors.append((audio, f"Timed out waiting of {wait_time_secs}s", -1))

for audio, error, _ in errors:
    print(f"Audio: {audio}")
    print(f"Error: {error}")

for audio, duration, transcription_duration in durations:
    print(
        f"Audio: {audio}, "
        f"Duration: {duration}, "
        f"Transcription Duration: {transcription_duration}, "
        f"Speedup: {duration/transcription_duration}")

success_rate = len(durations)/(len(durations)+len(errors))

all_durations = sum([d[1] for d in durations])
all_transcription_times = sum([d[2] for d in durations])
mean_speedup = all_durations/all_transcription_times
print(f"Total Audio Duration: {all_durations}")
print(f"Total Transcription Time: {all_transcription_times}")
print(f"Mean Speedup: {mean_speedup}")

assert success_rate >= 0.75
assert mean_speedup > 1.25

transcript_queue.put("STOP")
time.sleep(1.5)
transcript_queue.close()
result_queue.close()
worker_thread.join()
print("Done!")