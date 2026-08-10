import os
import io
from time import time, sleep

# Google Cloud Speech V2 imports
from google import auth
from google.api_core import client_options  # Importante para definir o endpoint
from google.cloud import speech_v2
from google.cloud.speech_v2 import types as speech_types

from runners.asr_utils import (
    remove_duplicates_regex,
    remove_duplicates_regex_simple,
    env_vals,
    normalize_smart_light,
)
from runners.audio_processing import get_audio_duration, generate_safe_chunks_iterative

# Configurações globais
PROJECT_ID = os.environ.get("GOOGLE_PROJECT_ID")
REGION = os.environ.get("GOOGLE_RECOGNIZER_REGION")


def call_google_chirp(recognizer_path, audio_path, client):
    """
    Chama o Google Cloud Speech V2 (Chirp) mantendo a assinatura da função anterior.
    """
    total_inf_time = 0.0
    inf_start = None
    try:

        full_transcription_parts = []
        total_duration = get_audio_duration(audio_path)
        audio_chunks = generate_safe_chunks_iterative(audio_path)

        for i, chunk_buffer in enumerate(audio_chunks):
            content = chunk_buffer.getvalue()

            # Prepare Request

            request = speech_types.RecognizeRequest(
                recognizer=recognizer_path,
                config=speech_types.RecognitionConfig(
                    auto_decoding_config=speech_types.AutoDetectDecodingConfig(),
                    features=speech_types.RecognitionFeatures(
                        enable_word_time_offsets=False,
                    ),
                ),
                content=content,
            )

            # API Call
            # We use a 60s timeout per chunk
            try:
                start_time = time()
                if inf_start == None:
                    inf_start = start_time
                response = client.recognize(request=request, timeout=60)
                time_spent = time() - start_time
                total_inf_time += time_spent

                for result in response.results:
                    if len(result.alternatives) > 0:
                        full_transcription_parts.append(
                            result.alternatives[0].transcript
                        )
            except Exception as e_chunk:
                print(f"Error transcribing chunk {i+1}/{len(audio_chunks)}: {e_chunk}")
                # We continue to the next chunk instead of failing everything
                continue

        final_text = "\n".join(full_transcription_parts)

        return (
            total_inf_time,
            inf_start,
            {
                "status_code": 200,
                "location": REGION,
                "text": final_text,
                "audio_duration": total_duration,
            },
        )

    except Exception as e:
        print(f"Google Speech API Error: {e}")
        return total_inf_time, inf_start, {"status_code": 500, "error": str(e)}


def google_caller_process(
    recognizer_id, audio_queue, result_queue, hardware_config, language, n_cpus
):

    print("Loaded Google Chirp model (Soundfile version)!")
    api_endpoint = f"{REGION}-speech.googleapis.com"
    client_opts = client_options.ClientOptions(api_endpoint=api_endpoint)

    # Inicializa o cliente com as opções regionais
    client = speech_v2.SpeechClient(client_options=client_opts)

    recognizer_path = (
        f"projects/{PROJECT_ID}/locations/{REGION}/recognizers/{recognizer_id}"
    )
    while True:
        # Pega item da fila
        audio_id, id_emergencia, start_time_q, sampling_rate, audio_path, channel = (
            audio_queue.get()
        )

        try:
            print(audio_path)
            if audio_path == "STOP":
                break

            total_inf_time, inf_start, result = call_google_chirp(
                recognizer_path, audio_path, client
            )

            if "text" in result:
                print(result["text"])
                transcript = result["text"]

                # Limpeza de texto
                transcript = normalize_smart_light(transcript)
                transcript = remove_duplicates_regex_simple(transcript)

                final_result = {
                    "audio_id": audio_id,
                    "id_emergencia": id_emergencia,
                    "part": transcript,
                    "start_time": inf_start,
                    "transcription_seconds": total_inf_time,
                    "transcription_model": recognizer_id,
                    "actor": channel,
                }
                result_queue.put(final_result)
            else:
                status = result.get("status_code", "Unknown")
                err_msg = result.get("error", "Unknown Error")
                print(f"Failed result: {status} - {err_msg}")
                raise Exception(f"Status code {status} on transcription request")

        except Exception as err:
            print(
                f"Error processing audio {audio_path}, removing processing_running flag"
            )
            print(f"Erro: {err}")

            result_queue.put(
                {
                    "audio_id": audio_id,
                    "id_emergencia": id_emergencia,
                    "part": "",
                    "start_time": None,
                    "transcription_seconds": 0.0,
                    "transcription_model": recognizer_id,
                    "error": str(err),
                    "actor": channel,
                }
            )
            sleep(5)

        sleep(0.05)
