import os
import json
import requests
from time import time, sleep

from runners.asr_utils import (
    remove_duplicates_regex,
    remove_duplicates_regex_simple,
    env_vals,
    normalize_smart_light,
)


def call_azure_fast(audio_path):
    locale = "pt-br"
    yourserviceregion = os.environ["AZURE_SERVICE_REGION"]
    yourspeechresourcekey = os.environ["AZURE_AI_RESOURCE_KEY"]
    url = f"https://{yourserviceregion}.api.cognitive.microsoft.com/speechtotext/transcriptions:transcribe?api-version=2024-11-15"
    headers = {
        "Ocp-Apim-Subscription-Key": yourspeechresourcekey,
        # NOTA: não defina Content-Type aqui; o requests configura multipart/form-data automaticamente
    }

    definition = {"locales": [locale]}

    with open(audio_path, "rb") as f:
        files = {
            # nome do campo 'audio' e tupla (filename, fileobj, content_type)
            "audio": (os.path.basename(audio_path), f, "audio/wav"),
        }
        data = {
            # 'definition' deve ser uma string JSON no campo de formulário
            "definition": json.dumps(definition, ensure_ascii=False)
        }
        # print(url)
        # print(headers)
        # print(files)
        # print(data)
        resp = requests.post(url, headers=headers, files=files, data=data, timeout=120)

    # Levanta exceção para códigos de erro HTTP
    try:
        resp.raise_for_status()
    except requests.HTTPError:
        # mostra corpo da resposta para debugging
        raise

    # Muitos endpoints de criação retornam 201 Created com Location para operação assíncrona
    if resp.status_code in (200, 201):
        # tenta retornar JSON se houver
        try:
            result_json = json.loads(resp.text)
            if "json" in result_json:
                json_body = result_json["json"]
            elif (
                "durationMilliseconds" in result_json
                and "combinedPhrases" in result_json
            ):
                json_body = result_json
            else:
                json_body = None
            audio_duration = json_body["durationMilliseconds"] / 1000
            transcription = "\n".join([x["text"] for x in json_body["combinedPhrases"]])
            return {
                "status_code": resp.status_code,
                "location": resp.headers.get("Location"),
                "text": transcription,
                "audio_duration": audio_duration,
            }
        except Exception as err:
            print(err)
            print("Unknown problem while parsing azure transcribe result")
            print("response:")
            print(resp.text)
            quit(1)
    else:
        return {"status_code": resp.status_code}


def azure_caller_process(
    model_name, audio_queue, result_queue, hardware_config, language, n_cpus
):
    if model_name == "azure-fast-transcribe":
        api_caller = call_azure_fast
    else:
        api_caller = None
    print("Loaded model!")
    while True:
        audio_id, id_emergencia, start_time, sampling_rate, audio_path, channel = (
            audio_queue.get()
        )
        try:
            print(audio_path)
            if audio_path == "STOP":
                break
            start_time = time()
            result = api_caller(audio_path)
            time_spent = time() - start_time
            if "text" in result:
                print(result["text"])
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
                # print(f'result: {result}')
                result_queue.put(result)
            else:
                status = result["status_code"]
                print(result)
                raise Exception(f"Status code {status} on transcription request")
        except Exception as err:
            print(
                f"Error processing audio {audio_path}, removing processing_running flag"
            )
            # print(f'transcript: {transcript}')
            print(f"Erro: {err}")
            print(err)
            print(err.with_traceback(None))
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
