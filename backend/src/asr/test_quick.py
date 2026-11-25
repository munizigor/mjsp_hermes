import json
import queue
import multiprocessing as mp
import threading
import time
from glob import glob
import sys

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
MAX_AUDIO_FILES = 25
MIN_SPEEDUP = 1.25
MIN_SUCCESS_RATE = 0.8
TIMEOUT_COEF = 0.9

if asr_model in asr_models:
    full_mname = asr_models[asr_model]
else:
    full_mname = asr_model

worker_func = select_worker(hardware_config)

def high_timeout_worker(full_mname,
        transcript_queue,
        result_queue,
        hardware_config,
        language,
        n_cpus,):
    worker_func(full_mname, transcript_queue, result_queue, hardware_config, language, n_cpus, timeout=32)

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
    target=high_timeout_worker,
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

total_files = 0
audios = [] #glob('test_audios/*.wav')
for audio_dir in glob("test_audios/segmentado/*"):
    if MAX_AUDIO_FILES > 0:
        wavs = glob(audio_dir + "/*.wav")
        wavs.sort()
        if len(wavs) > MAX_AUDIO_FILES:
            wavs = wavs[:MAX_AUDIO_FILES]
        audios.append((audio_dir, wavs))
        total_files += len(wavs)
        MAX_AUDIO_FILES -= len(wavs)

durations = []
errors = []

i = 0
for audio_dir, wav_files in audios:
    print(f"Testing {audio_dir}...", file=sys.stderr)

    for wav_file in wav_files:
        i += 1
        segment = AudioSegment.from_file(wav_file)
        duration = segment.duration_seconds
        wait_time_secs = duration*TIMEOUT_COEF

        new_task = (
            [1],
            '1',
            time.time(),
            16000,
            wav_file,
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
            #print("WAITING FOR", wav_file, file=sys.stderr)
            transcription_result = result_queue.get(block=True, timeout=wait_time_secs)
            transcription_end = time.time()
            transcription_duration = transcription_end - transcription_start

            #print(f"Transcription: {transcription_result['part']}", file=sys.stderr)
            #print(f"Transcription Duration: {transcription_duration}", file=sys.stderr)
            #print(f"Audio Duration: {duration}", file=sys.stderr)
            #print(f"Speedup: {duration/transcription_duration}", file=sys.stderr)
            durations.append((wav_file, duration, transcription_duration))
        except queue.Empty as err:
            print(f"Timeout waiting for transcription of {wav_file}: {wait_time_secs}s", file=sys.stderr)
            errors.append((wav_file, f"Timed out waiting of {wait_time_secs}s", -1, wait_time_secs))
        print(f"{i}/{total_files} DONE", file=sys.stderr)

print("Errors:")
for audio, error, _, timeout in errors:
    print(f"Audio: {audio}", file=sys.stderr)
    print(f"Error: {error}", file=sys.stderr)

print("Successes:")
for audio, duration, transcription_duration in durations:
    print(
        f"Audio: {audio}, "
        f"Duration: {duration}, "
        f"Transcription Duration: {transcription_duration}, "
        f"Speedup: {duration/transcription_duration}", file=sys.stderr)

success_rate = len(durations)/(len(durations)+len(errors))

all_durations = sum([d[1] for d in durations])
all_transcription_times = sum([d[2] for d in durations])
mean_speedup = all_durations/all_transcription_times
print(f"Total Audio Duration: {all_durations}", file=sys.stderr)
print(f"Total Transcription Time: {all_transcription_times}", file=sys.stderr)
print(f"Mean Speedup: {mean_speedup}", file=sys.stderr)
print(f"Success Rate: {success_rate}", file=sys.stderr)
print(f"Errors: {len(errors)}", file=sys.stderr)

transcript_queue.put("STOP")
time.sleep(2)
transcript_queue.close()
worker_thread.join()
time.sleep(1)
result_queue.close()
print("Done!")

assert success_rate >= MIN_SUCCESS_RATE, f"Success rate too low: {success_rate} < {MIN_SUCCESS_RATE}"
assert mean_speedup > MIN_SPEEDUP, f"Mean speedup too low: {mean_speedup} < {MIN_SPEEDUP}"