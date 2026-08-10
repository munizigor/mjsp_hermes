import os
from time import time, sleep
import io
import sys
from pydub import AudioSegment
import librosa
import numpy as np
import tritonclient.http as http_client

from runners.asr_utils import (
    remove_duplicates_regex,
    remove_duplicates_regex_simple,
    env_vals,
    normalize_smart_light,
)

if not os.getenv("TRITON_SERVER_URL"):
    os.environ["TRITON_SERVER_URL"] = "localhost:8000"
if not os.getenv("TRITON_SERVER_TIMEOUT_SECS"):
    os.environ["TRITON_SERVER_TIMEOUT_SECS"] = "240"

# MODEL_NAME = "turbo_cuda"
TRITON_SERVER_URL = os.environ["TRITON_SERVER_URL"]
timeout_secs = int(os.environ["TRITON_SERVER_TIMEOUT_SECS"])


def preprocess_audio(audio_path, audio_format, original_sr=16000, target_sr=16000):
    # Load audio
    try:
        audio, sr = librosa.load(audio_path, sr=target_sr)  # Whisper expects 16kHz
    except ValueError as err:
        print(f"Error loading audio file {audio_path}: {err}", file=sys.stderr)
        # Use pydub to open audio and convert to a standardized WAV in memory
        audio_segment = AudioSegment.from_file(audio_path)
        audio_segment = audio_segment.set_frame_rate(target_sr).set_channels(1)

        # Export to a buffer
        buffer = io.BytesIO()
        audio_segment.export(buffer, format="wav")
        buffer.seek(0)

        # Load with librosa from the buffer
        audio, sr = librosa.load(buffer, sr=target_sr)
        print(audio.shape, sr)

    length_seconds = len(audio) / sr
    return audio.astype(np.float32), length_seconds


def call_triton_server(inputs, model_name, client):
    # Send request
    results = client.infer(
        model_name=model_name, inputs=inputs, timeout=timeout_secs * 1000
    )
    output_data = results.as_numpy("OUTPUT_0")
    transcription = output_data[0].decode("utf-8")

    return {"status_code": 200, "location": None, "text": transcription}


def triton_caller_process(
    model_name, audio_queue, result_queue, hardware_config, language, n_cpus
):

    client = http_client.InferenceServerClient(
        url=TRITON_SERVER_URL,
        connection_timeout=timeout_secs,
        network_timeout=timeout_secs,
    )
    while True:
        audio_id, id_emergencia, start_time, sampling_rate, audio_path, channel = (
            audio_queue.get()
        )
        n_audios = len(audio_id)

        try:
            print(audio_path)
            if audio_path == "STOP":
                break
            audio_input_data, audio_secs = preprocess_audio(
                audio_path, "wav", original_sr=sampling_rate
            )
            # Create input tensor
            inputs = [http_client.InferInput("INPUT_0", audio_input_data.shape, "FP32")]
            inputs[0].set_data_from_numpy(audio_input_data)

            print(
                f"Calling ASR model {model_name} on server {TRITON_SERVER_URL}"
                f" for {n_audios} audio(s) from emergency {id_emergencia} with length {audio_secs}s. Path = {audio_path}."
            )
            start_time = time()
            result = call_triton_server(inputs, model_name, client)
            time_spent = time() - start_time

            if "text" in result:
                print("Text found in response:", result["text"])
                transcript = result["text"]
                transcript = normalize_smart_light(transcript)
                transcript = remove_duplicates_regex_simple(transcript)
                result = {
                    "audio_id": audio_id,
                    "id_emergencia": id_emergencia,
                    "part": transcript,
                    "start_time": start_time,
                    "transcription_seconds": time_spent,
                    "transcription_model": model_name,
                    "actor": channel,
                }
                print("result:", result)
                result_queue.put(result)
            else:
                status = result["status_code"]
                print("No text found in response:", result)
                raise Exception(f"Status code {status} on transcription request")
        except Exception as err:
            print(
                f"Error processing audio {audio_id} from emergencia {id_emergencia}.",
                file=sys.stderr,
            )
            print(f"Error: {err}", file=sys.stderr)
            print(err, file=sys.stderr)
            print(err.with_traceback(None), file=sys.stderr)
            time_spent = time() - start_time
            print(f"Time spent on ASR request: {time_spent}", file=sys.stderr)

            # print(f'transcript: {transcript}')
            result_queue.put(
                {
                    "audio_id": audio_id,
                    "id_emergencia": id_emergencia,
                    "part": "",
                    "start_time": None,
                    "transcription_seconds": 0.0,
                    "transcription_model": model_name,
                    "error": str(err),
                    "actor": channel,
                }
            )
            sleep(5)

        sleep(0.05)
