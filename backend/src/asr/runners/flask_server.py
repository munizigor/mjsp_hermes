import os
import requests
from time import time, sleep

# Importações mantendo a compatibilidade com o ecossistema do seu projeto
from runners.asr_utils import (
    remove_duplicates_regex_simple,
    normalize_smart_light,
)


def call_flask_api(api_url, audio_path):
    """
    Chama a nossa API customizada Flask (Qwen3-ASR) mantendo a assinatura
    de retorno esperada: total_inf_time, inf_start, result_dict.
    """
    total_inf_time = 0.0
    inf_start = time()

    try:
        # Abre o arquivo de áudio para enviar via multipart/form-data
        with open(audio_path, "rb") as f:
            files = {"audio": (os.path.basename(audio_path), f, "audio/wav")}

            # Usamos timeout de 300s (5 minutos) para garantir que áudios
            # grandes não derrubem a conexão pelo lado do cliente.
            response = requests.post(api_url, files=files, timeout=300)

        total_inf_time = time() - inf_start

        if response.status_code == 200:
            result_data = response.json()

            if result_data.get("status") == "success":
                return (
                    total_inf_time,
                    inf_start,
                    {
                        "status_code": 200,
                        "text": result_data.get("transcription", ""),
                    },
                )
            else:
                err_msg = result_data.get("message", "Unknown Error in payload")
                return total_inf_time, inf_start, {"status_code": 500, "error": err_msg}
        else:
            return (
                total_inf_time,
                inf_start,
                {"status_code": response.status_code, "error": response.text},
            )

    except Exception as e:
        print(f"Flask API Error: {e}")
        total_inf_time = time() - inf_start
        return total_inf_time, inf_start, {"status_code": 500, "error": str(e)}


def flask_caller_process(
    model_name, audio_queue, result_queue, hardware_config, language, n_cpus
):
    """
    Processo worker equivalente ao google_caller_process.
    """
    # 1. Busca dinâmica de IP e Porta contendo "FLASK" nas variáveis de ambiente
    flask_ip = None
    flask_port = None

    for key, value in os.environ.items():
        key_upper = key.upper()
        if "FLASK" in key_upper and "ASR" in key_upper:
            if "IP" in key_upper or "HOST" in key_upper:
                flask_ip = value
            elif "PORT" in key_upper:
                flask_port = value

    assert flask_ip is not None, "FLASK_IP not found in environment variables"
    assert flask_port is not None, "FLASK_PORT not found in environment variables"

    api_url = f"http://{flask_ip}:{flask_port}/transcribe"
    print(f"Loaded Flask ASR worker! Targeting {api_url}")

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
            total_inf_time, inf_start, result = call_flask_api(api_url, audio_path)

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
            sleep(5)

        sleep(0.05)
