import os
import re
import requests
from time import time, sleep

from runners.asr_utils import (
    remove_duplicates_regex_simple,
    normalize_smart_light,
)

'''
Exemplo de requisição no shell e resposta:
curl http://127.0.0.1:8080/inference
    -H "Content-Type: multipart/form-data"
    -F file="@test_audios/senhora_ataque_cachorro.wav"
    -F response_format="json"

{
    "text":" pode me dizer o que está acontecendo?\n estou com muito medo, [...], já estão a caminhar\n"
}
'''

def call_whispercpp_api(api_url, audio_path, model_name):
    """
    Chama a API do Whisper CPP
    """
    total_inf_time = 0.0
    inf_start = time()

    try:
        with open(audio_path, "rb") as f:
            files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
            response = requests.post(api_url, files=files, timeout=28)
        total_inf_time = time() - inf_start

        if response.status_code == 200:
            result_data = response.json()
            if "text" in result_data:
                transcript = result_data["text"]
                return (
                    total_inf_time,
                    inf_start,
                    {
                        "status_code": 200,
                        "text": transcript,
                    },
                )
            else:
                return (
                    total_inf_time,
                    inf_start,
                    {
                        "status_code": 500,
                        "error": "'text' field not found; " + response.text,
                    },
                )
        else:
            return (
                total_inf_time,
                inf_start,
                {"status_code": response.status_code, "error": response.text},
            )

    except Exception as e:
        print(f"Whisper CPP API Error: {e}")
        total_inf_time = time() - inf_start
        return total_inf_time, inf_start, {"status_code": 500, "error": str(e)}


def whisper_cpp_caller_process(
    model_name, audio_queue, result_queue, hardware_config, language, n_cpus
):
    """
    Processo worker.
    """
    server_ip = os.environ.get("ASR_VLLM_HOST", None)
    server_port = os.environ.get("ASR_VLLM_PORT", None)

    assert server_ip is not None, "ASR_VLLM_HOST not found in environment variables"
    assert server_port is not None, "ASR_VLLM_PORT not found in environment variables"

    api_url = f"http://{server_ip}:{server_port}/inference"
    print(f"Loaded Whisper CPP ASR worker! Targeting {api_url}")

    while True:
        # Pega item da fila
        queue_item = audio_queue.get()

        # Prevenção caso o sinal de STOP venha direto
        if queue_item == "STOP":
            break

        audio_id, id_emergencia, start_time_q, sampling_rate, audio_path, channel = (
            queue_item
        )

        try:
            print(audio_path)
            if audio_path == "STOP":
                break

            # 2. Chama a função de requisição modularizada
            total_inf_time, inf_start, result = call_whispercpp_api(
                api_url, audio_path, model_name
            )

            if result.get("status_code") == 200 and "text" in result:
                transcript = result["text"]
                print(transcript)

                # Limpeza de texto nativa do seu pipeline
                transcript = normalize_smart_light(transcript)
                transcript = remove_duplicates_regex_simple(transcript)

                final_result = {
                    "audio_id": audio_id,
                    "id_emergencia": id_emergencia,
                    "part": transcript,
                    "start_time": inf_start,
                    "transcription_seconds": total_inf_time,
                    "transcription_model": model_name,
                    "actor": channel,
                }
                result_queue.put(final_result)
            else:
                status = result.get("status_code", "Unknown")
                err_msg = result.get("error", "Unknown Error")
                print(f"Failed result: {status} - {err_msg}")
                raise Exception(
                    f"Status code {status} on transcription request: {err_msg}"
                )

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
                    "transcription_model": model_name,
                    "error": str(err),
                    "actor": channel,
                }
            )
            sleep(0.5)

        sleep(0.025)