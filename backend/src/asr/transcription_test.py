import sys
from glob import glob
import os
import json
from threading import Thread
from time import time, sleep
from multiprocessing import Event
import torch
from tqdm import tqdm
from transformers import pipeline
import soundfile as sf
import pandas as pd
import numpy as np

def audio_len(audio_path):
    y, sr = sf.read(audio_path)
    return y.shape[0] / sr

def watch_gpu_memory(readings_vec: list, flag):
    print('Thread inside')
    waittime = 0.25
    device = torch.device('cuda:0')

    while True:
        free, total = torch.cuda.mem_get_info(device)
        mem_used_MB = (total - free) / 1024 ** 2
        readings_vec.append(mem_used_MB)
        if flag.is_set():
            #print('Stoping thread')
            return None
        else:
            #print(readings)
            sleep(waittime)

def get_gpu_measurer():
    flag = Event()
    flag.clear()
    readings = []
    t = Thread(target = watch_gpu_memory, args = (readings, flag))

    return t, readings, flag

def transcription_test_1(audios_dir):
    measurer0, normal_readings, stop_flag0 = get_gpu_measurer()
    measurer0.start()
    stop_flag0.set()
    normal_gpu_use = np.mean(normal_readings)

    models = [
        #'thiagobarbosa/whisper-base-common-voice-16-pt-v6',
        'my-north-ai/whisper-medium-pt',
        'nilc-nlp/distil-whisper-coraa-mupe-asr',
        'openai/whisper-large-v3-turbo'
    ]
    audios = glob(audios_dir+'/*.wav')
    audio_lens = {
        p: round(audio_len(p), 2) for p in audios
    }
    transcriptions = []
    for model_name in models:
        model = pipeline("automatic-speech-recognition", 
            model=model_name,
            return_timestamps=True,
            generate_kwargs={"language": "portuguese"})
        #whisper_prompt = "Esta é uma transcrição de chamada de emergência: "
        #prompt_ids = model.tokenizer.get_prompt_ids(whisper_prompt, return_tensors='pt').to('cuda:0')
        n_tries = 5
        for _ in range(n_tries):
            for audio_path in tqdm(reversed(audios)):
                audio_name = os.path.basename(audio_path)
                measurer, mem_readings, stop_flag = get_gpu_measurer()
                measurer.start()
                start_time = time()
                print(audio_name, file=sys.stderr)
                result = model(audio_path,
                    #generate_kwargs={"prompt_ids": prompt_ids}
                )
                time_spent = time() - start_time
                stop_flag.set()
                mean_gpu_mem = np.mean(mem_readings) - normal_gpu_use
                gpu_peak = max(mem_readings) - normal_gpu_use
                speed_up = audio_lens[audio_path] / time_spent
                transcriptions.append({
                    'audio_name': audio_name,
                    'model_name': model_name,
                    'full_text': result['text'],
                    'processing_duration': str(round(time_spent, 3)),
                    'audio_duration': str(audio_lens[audio_path]),
                    'speed_up': str(round(speed_up, 3))+'x',
                    'mean_gpu_mem': str(round(mean_gpu_mem, 3)),
                    'gpu_peak': str(round(gpu_peak, 3)),
                #    'segments': [{'start_end': s, 'content': t} for s, t in chunks]
                })
                print(json.dumps(transcriptions[-1], indent=2, ensure_ascii=False), file=sys.stderr)
        del model
    
    df = pd.DataFrame(transcriptions)
    df.to_csv(audios_dir + '/transcriptions_df.csv', sep=',')



if __name__ == "__main__":
    print(sys.argv)
    audios_dir = sys.argv[1]
    transcription_test_1(audios_dir)

    df = pd.read_csv(audios_dir + '/transcriptions_df.csv')
    stats = []
    for model_name, lines in df.groupby('model_name'):
        processing_times = lines['processing_duration'].tolist()
        speed_ups = [float(x.rstrip('x')) for x in lines['speed_up'].tolist()]
        mean_gpu_mem = lines['mean_gpu_mem'].tolist()
        gpu_peak = lines['gpu_peak'].tolist()
        stats.append({
            'model_name': model_name,
            'mean_processing_time': str(round(float(np.mean(processing_times)), 3)),
            'mean_speed_up': str(round(float(np.mean(speed_ups)), 3)),
            'mean_gpu_mem': str(round(float(np.mean(mean_gpu_mem)), 3)),
            'gpu_peak': str(round(float(np.mean(gpu_peak)), 3)),
        })
        print(stats[-1])

    df2 = pd.DataFrame(stats)
    df2.to_csv(audios_dir + '/model_stats.csv', sep=',')
    