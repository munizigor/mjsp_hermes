import argparse
import base64
import glob
import os
import threading
from time import time, sleep
from typing import List
import librosa
import numpy as np
import tritonclient.http as http_client
import io
import soundfile as sf
from pydub import AudioSegment

from asr_lib.audio_processing import prepare_audios_for_triton

if not os.getenv("TRITON_SERVER_URL"):
    os.environ["TRITON_SERVER_URL"] = "localhost:8000"

#MODEL_NAME = "turbo_cuda"
MODEL_NAME = "qwen3_asr_1.7b"
TRITON_SERVER_URL = os.environ["TRITON_SERVER_URL"] 
timeout_secs = 240
#Baseado em: https://cloud.google.com/products/calculator?hl=pt_br&dl=CjhDaVF6TXpZNFl6RmlNQzB6TnpSaUxUUTVNVEl0T0RWaVl5MWxNbVkzWW1VeFkyRmtaallRQVE9PRAIGiRDODBGMjI5RS02MjExLTQ5QUMtOUU2Ri0zNzBBRTJFODkyOEM
gpu_hour_cost_dollars = 0.23
#gpu_hour_cost_dollars_normal = 0.7659

def preprocess_audio(audio_path, audio_format):
    # Load audio
    try:
        audio, sr = librosa.load(audio_path, sr=16000)  # Whisper expects 16kHz
    except ValueError as err:
        print(f"Error loading audio file {audio_path}: {err}")
        # Use pydub to open audio and convert to a standardized WAV in memory
        audio_segment = AudioSegment.from_file(audio_path)
        audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
        
        # Export to a buffer
        buffer = io.BytesIO()
        audio_segment.export(buffer, format="wav")
        buffer.seek(0)
        
        # Load with librosa from the buffer
        audio, sr = librosa.load(buffer, sr=16000)
        print(audio.shape, sr)
        
    length_seconds = len(audio) / sr
    np_audio = audio.astype(np.float32)
    audios, lengths = prepare_audios_for_triton(np_audio, 16000, max_len_sec=24.0)
    return audios, lengths

def process_audios(filepaths: List[str]):
    results = []
    for filepath in filepaths:
        audio_input_datas, lengths_seconds = preprocess_audio(filepath, None)
        results.append((filepath, audio_input_datas, lengths_seconds))
    return results

def asr_thread(filepath, audio_input_datas, lengths_seconds, results_list):
    #try:
    '''outputs = [
        http_client.InferRequestedOutput("OUTPUT_0", binary_data=True)
    ]'''
    '''outputs = [
        http_client.InferRequestedOutput("OUTPUT_0", binary_data=False)
    ]'''
    client = http_client.InferenceServerClient(url=TRITON_SERVER_URL,
        connection_timeout=timeout_secs,
        network_timeout=timeout_secs)
    
    audios_iter = zip(audio_input_datas, lengths_seconds)

    index = 0
    full_transcription = ""
    for audio_input_data, length_seconds in audios_iter:
        index += 1
    
        # Create input tensor
        inputs = [
            http_client.InferInput(
                "INPUT_0", audio_input_data.shape, "FP32"
            )
        ]
        inputs[0].set_data_from_numpy(audio_input_data)

        # Send request
        infer_start_time = time()
        results = client.infer(model_name=MODEL_NAME, inputs=inputs,
                               outputs=[http_client.InferRequestedOutput("OUTPUT_0", binary_data=True)],
                               timeout=timeout_secs*1000)
        infer_time_spent = time() - infer_start_time
        print(f"Inference took {infer_time_spent} seconds")
        print(f"Length of audio: {length_seconds} seconds")
        speed = length_seconds / infer_time_spent
        print(f"current speed: {speed} seconds per second")
        hour_cost = gpu_hour_cost_dollars / speed
        print(f"hour cost: {hour_cost} dollars per hour of transcription")

        # Get output
        print(results)
        print("RAW RESPONSE:", results.get_response())
        print("Full results object:", results)
        print("Output names:", results.get_response()['outputs'])

        raw_json = results.get_response()
        out_info = next(o for o in raw_json['outputs'] if o['name'] == 'OUTPUT_0')
        print(out_info)

        res1 = results.get_output("OUTPUT_0")
        print(res1)

        output_data = results.as_numpy("OUTPUT_0")

        print(output_data)
        print(output_data.shape)
        print(output_data.dtype)

        transcription = bytes(output_data.tolist()).decode("utf-8")
        full_transcription += "\n" +  transcription
        print(f"Transcription for audio {filepath}[{index}]: {transcription}")
        quit(1)

    # Optionally, save transcription to a file
    output_txt_file = os.path.splitext(filepath)[0] + ".txt"
    with open(output_txt_file, "w", encoding="utf-8") as f:
        f.write(transcription)
    print(f"Transcription saved to {output_txt_file}")
    results_list.append((infer_time_spent, length_seconds))

    '''except Exception as e:
        print(f"Error processing audio {filepath}: {e}")
        
        infer_time_spent = time() - infer_start_time
        print(f"Inference took {infer_time_spent} seconds")
        
        results_list.append((None, length_seconds))'''

def main():
    parser = argparse.ArgumentParser(description="Triton Whisper Client for WAV files.")
    parser.add_argument("directory", type=str, help="Directory containing .wav files.")
    args = parser.parse_args()

    wav_files = glob.glob(os.path.join(args.directory, "*.wav")) + glob.glob(os.path.join(args.directory, "*.WAV"))
    
    if not wav_files:
        wav_files = []
    mp3_files = glob.glob(os.path.join(args.directory, "*.mp3")) + glob.glob(os.path.join(args.directory, "*.MP3"))
    if not mp3_files:
        mp3_files = []
    wav_files = [(w, 'wav') for w in wav_files if 'part' not in w]
    mp3_files = [(w, 'mp3') for w in mp3_files if 'part' not in w]

    audio_files = wav_files + mp3_files

    print(f"Found {len(audio_files)} files. Sending to Triton server...")
    audio_files = [audio_files[0]]
    processed_audios = process_audios([p for p, f in audio_files])
    index = 0
    threads: List[threading.Thread] = []
    results = []
    inferences_start = time()
    for filepath, audio_input_data, length_seconds in processed_audios:
        print(f"Starting thread for audio of length {length_seconds} seconds...")

        thread = threading.Thread(target=asr_thread, args=(filepath, audio_input_data, length_seconds, results))
        thread.start()
        threads.append(thread)
        sleep(1.0)

        index += 1
        quit(1)
    for thread in threads:
        thread.join()
    seconds_of_inference = time() - inferences_start
    audio_lens = [audio_len for infer_time, audio_len in results if infer_time is not None]
    infer_times = [infer_time for infer_time, audio_len in results if infer_time is not None]
    seconds_transcribed = sum(audio_lens)
    print("All files processed.")
    speed = seconds_transcribed / seconds_of_inference
    print(f"Total transcription time: {seconds_of_inference} seconds")
    print(f"Total audio transcribed: {seconds_transcribed} seconds")
    print(f"current speed: {speed} seconds per second")

    hour_cost = gpu_hour_cost_dollars / speed
    print(f"hour cost: {hour_cost} dollars per hour of transcription")

if __name__ == "__main__":
    main()
