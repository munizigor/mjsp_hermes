import json
import time
import os
import sys
import requests
from queue import Empty as QueueEmptyError
import threading

from pydantic import ValidationError

from openai import OpenAI, APITimeoutError, APIConnectionError
from runners.gcp_endpoint import save_ner_to_db
from runners.gliner_utilities import (
    ner_labels_pydantic,
    schema_key_translations,
    many_values,
    values_to_ignore,
)

if not os.getenv("TRITON_SERVER_NER_TIMEOUT_SECS"):
    os.environ["TRITON_SERVER_NER_TIMEOUT_SECS"] = "5"

USER_CODE = os.getenv("SERPRO_USER_CODE")
SERPRO_BASE_URL = "https://e-api-serprollm.ni.estaleiro.serpro.gov.br"
SERPRO_LLM_BASE_URL = f"{SERPRO_BASE_URL}/gateway/v1"
SERPRO_OAUTH_URL = f"{SERPRO_BASE_URL}/oauth2/token"

timeout_secs = int(os.environ["TRITON_SERVER_NER_TIMEOUT_SECS"])

user_template_with_schema = """
Você é um assistente especializado em extrair informações de transcrições de chamadas de emergência.
Sua tarefa é analisar uma transcrição e produzir um JSON estritamente válido de acordo com o seguinte schema:
[[[[schema_placeholder]]]]

Você não deve discursar sobre o significado do schema, nem incluir elementos pré-textuais antes das informações dele. 
Apenas dê a saída em JSON puro e estritamente válido.

Você deve analisar a seguinte transcrição:
[[[[transcript_content]]]]
"""
system_prompt = "Você é um assistente que sempre responde estritamente no formato JSON especificado."

def smart_json_loads(raw: str, verbose=False):
    try:
        return json.loads(raw)
    except Exception as e:
        #if verbose:
        #    print("Initial json parsing failed for: ", raw, file=sys.stderr)
        #    print("Attempting to strip markdown and parse again...", file=sys.stderr)
        clean_raw = raw.strip().strip("`")
        if clean_raw.startswith("json"):
            clean_raw = clean_raw[4:]
        if clean_raw.endswith("json"):
            clean_raw = clean_raw[:-4]
        if verbose:
            print("Cleaned json: ", clean_raw, file=sys.stderr)

        try:
            d = json.loads(clean_raw)
            if verbose:
                print("Successfully parsed json: ", d, file=sys.stderr)
            return d
        except Exception as e2:
            print("Failed to parse cleaned json: ", clean_raw, file=sys.stderr)
            raise e2


def get_serpro_token():
    """
    curl -k -d "grant_type=client_credentials" --user "ABCDE:EXAMPLE" \
        https://e-api-serprollm.ni.estaleiro.serpro.gov.br/oauth2/token
    {"expires_in":7200,"token_type":"bearer","access_token":"XXXXXXXXXXXXXXXXXX"}
    """
    base_url = "https://e-api-serprollm.ni.estaleiro.serpro.gov.br/oauth2/token"
    payload = "grant_type=client_credentials"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    # add user
    response = requests.request(
        "POST", base_url, headers=headers, data=payload, auth=(USER_CODE, "")
    )
    return json.loads(response.text)["access_token"]

def request_serpro_generic(
    model, access_token, prompt, model_class, max_tokens=1200, verbose=False
):
    base_url = (
        "https://e-api-serprollm.ni.estaleiro.serpro.gov.br/gateway/v1/chat/completions"
    )
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
            "guided_json": model_class.model_json_schema(),
            "reasoning_effort": "low",  # Can also use "low" to minimize rather than disable
            "chat_template_kwargs": {
                "enable_thinking": False,  # Use this for Qwen3 models
                "thinking": False,  # Use this for Granite or Holo2 models
            },
        }
    )
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + access_token,
    }

    max_tokens_reached_flags = [
        "'finish_reason': 'length'",
        '"finish_reason": "length"',
        "'finish_reason': 'max_tokens'",
        '"finish_reason": "max_tokens"',
    ]

    attempts_left = 3
    failures = 0
    req_latency = 0.0
    response_json = None
    reasonings = []
    failure_texts = []
    jsons = []
    while attempts_left > 0:
        try:
            req_start = time.time()
            response = requests.request(
                "POST", base_url, headers=headers, data=payload, timeout=16
            )
            req_end = time.time()
            req_latency += req_end - req_start
            response_json = json.loads(response.text)
            for c in response_json["choices"]:
                raw = c["message"]["content"]
                if any([f in raw for f in max_tokens_reached_flags]):
                    raise Exception("Response truncated: " + raw)
                if raw is not None:
                    try:
                        jsons.append(smart_json_loads(raw, verbose=verbose))
                    except Exception as e:
                        if verbose:
                            print("Could not parse json: ", raw)
                        raise Exception("Cannot parse json: \n" + raw)
                    reasoning = c["message"].get("reasoning_content", None)
                    if reasoning is None:
                        reasonings.append("")
                    else:
                        reasonings.append(reasoning)
                else:
                    if verbose:
                        print("raw is None, skipping", file=sys.stderr)
            break
        except Exception as e:
            if verbose:
                print(f"Attempt failed ({attempts_left}). Retrying...", file=sys.stderr)
                print(e, file=sys.stderr)
                print("full raw:", response_json, file=sys.stderr)
            failure_texts.append(str(e))
            failures += 1
            attempts_left -= 1
            continue

    if response_json and len(jsons) > 0:
        if type(response_json) is dict:
            answer = jsons[0]
            prompt_tokens = response_json["usage"].get("prompt_tokens", None)
            compl_tokens = response_json["usage"].get("completion_tokens", None)
            total_tokens = response_json["usage"].get("total_tokens", None)
            meta = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": compl_tokens,
                "total_tokens": total_tokens,
                "request_time": req_latency / (failures + 1),
                "failures": failure_texts,
            }
            return answer, meta, response_json
    # failure reached
    meta = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "request_time": req_latency / failures,
        "failures": failure_texts,
    }
    return None, meta, response_json

def call_serpro_llm_with_schema(
    params_dict: dict,
    redo_queue,
):
    """
    Interpret a call transcript and extract relevant information.

    Args:
        transcript (str): The call transcript to interpret.

    Returns:
        dict: A dictionary containing the extracted information.
    """
    proc_start = time.time()

    fmt_class = params_dict["labels"]
    transcript = params_dict["transcript"]
    model_name = params_dict["model_name"]
    access_token = params_dict["access_token"]

    processing_time = 0.0
    fmd_json = fmt_class.model_json_schema()
    prompt = user_template_with_schema.replace(
        "[[[[schema_placeholder]]]]\n", json.dumps(fmd_json, ensure_ascii=False)
    )
    prompt = prompt.replace("[[[[transcript_content]]]]", transcript)
    response_meta = {}
    result = None
    input_tokens = None
    output_tokens = None

    try:
        start_time = time.time()
        output_json, meta, raw_response = request_serpro_generic(
            model_name,
            access_token,
            prompt,
            fmt_class,
            max_tokens=3600,
            verbose=True,
        )

        processing_time = time.time() - start_time

        if meta is not None:
            input_tokens = meta.get("prompt_tokens", None)
            output_tokens = meta.get("completion_tokens", None)
        
        error_type = "No error in request"
    except ValidationError as err:
        print(
            "Pydantic Validation error for prompt:\n",
            params_dict["transcript"],
            file=sys.stderr,
        )
        error_type = "Pydantic_validation_error"
        output_json = None
    except (APITimeoutError, APIConnectionError) as e:
        print(f"Attempt failed: {e}")
        error_type = "api_timeout_or_connection_error"
        output_json = None
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        error_type = "unknown_error"
        output_json = None

    id_emergencia = params_dict["id_emergencia"]
    horario_contexto = params_dict["horario_contexto"]
    if output_json is not None:
        print("Raw NER result:", output_json)

        response_json = {}
        for key, val in output_json["entities"].items():
            if not key in response_json:
                response_json[key] = []
            if val is None:
                continue
            else:
                if isinstance(val, list):
                    for item in val:
                        response_json[key].append([item, 1.0])
                else:
                    response_json[key].append([val, 1.0])
        for key, val in output_json["classifications"].items():
            if key in schema_key_translations:
                key = schema_key_translations[key]
            if val == "Sim":
                response_json[key] = 1.0
            elif val == "Não":
                response_json[key] = 0.0
            else:
                print(f"Unknown classification value for {key}: {val}", file=sys.stderr)
                pass
        print("Cleaned NER result:", response_json)

        for label in values_to_ignore.keys():
            if label in response_json:
                not_ignore = []
                if response_json[label] is not None:
                    for value, score in response_json[label]:
                        if value.lower() not in values_to_ignore[label]:
                            not_ignore.append([value, score])
                response_json[label] = not_ignore

        top_to_keep = params_dict["top_to_keep"]
        not_keep_top = params_dict["not_keep_top"]
        for label in response_json.keys():
            if type(response_json[label]) == list:
                response_json[label].sort(key=lambda x: x[1], reverse=True)
                if (
                    label not in not_keep_top
                    and len(response_json[label]) > top_to_keep
                ):
                    response_json[label] = response_json[label][
                        :top_to_keep
                    ]  # Keep only the top 3 values for each label
        print("Processed NER result:", response_json)
        all_processing_time = time.time() - proc_start
        response_meta = {
            "no_gpu_time": all_processing_time - processing_time,
            "processing_time": processing_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model_name": params_dict["model_name"],
        }

        to_save = {
            "id_emergencia": id_emergencia,
            "horario_contexto": horario_contexto,
            "entities": response_json,
            "horario_fim": float(time.time()),
            "meta": response_meta,
        }
        print("New NER result:", json.dumps(to_save, ensure_ascii=False, indent=3))
        save_ner_to_db(to_save, redo_queue)
    else:
        print(f"Error in NER inference for id_emergencia: {id_emergencia}")
        all_processing_time = time.time() - proc_start
        to_save = {
            "id_emergencia": id_emergencia,
            "horario_contexto": horario_contexto,
            "entities": None,
            "horario_fim": float(time.time()),
            "meta": {
                "no_gpu_time": all_processing_time - processing_time,
                "processing_time": processing_time,
                "error": error_type,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "model_name": params_dict["gliner_mname"],
            },
        }
        save_ner_to_db(to_save, redo_queue)


def serpro_llm_worker_process(
    gliner_mname,
    transcript_queue,
    redo_queue,
    result_queue,
    hardware_config,
    batch_size: int,
    labels_set,
    max_tokens: int,
    n_cpus: int,
):
    print("Starting serpro llm worker process for NER")
    access_token = get_serpro_token()
    while True:
        try:
            tp = transcript_queue.get()
        except QueueEmptyError as err:
            tp = None

        if tp != None:
            try:
                id_emergencia, horario_contexto, transcript = tp
                label_list = ner_labels_pydantic[labels_set]
                should_stop = "STOP" == transcript

                if should_stop:
                    # treat stop signal
                    break
                else:
                    params = {
                        "id_emergencia": id_emergencia,
                        "horario_contexto": horario_contexto,
                        "model_name": gliner_mname,
                        "labels": label_list,
                        "top_to_keep": 3,
                        "not_keep_top": many_values,
                        "batch_size": batch_size,
                        "max_tokens": max_tokens,
                        "transcript": transcript,
                        "access_token": access_token,
                    }
                    ner_thread = threading.Thread(
                        target=call_serpro_llm_with_schema,
                        args=(params, redo_queue),
                    )
                    ner_thread.start()

            except Exception as err:
                print("Error processing transcript")
                print(f"Erro: {err}")
                print(err.with_traceback(None))
                id_emergencia, horario_contexto, _ = tp

                print(id_emergencia, horario_contexto)
                time.sleep(0.1)
                result_queue.put(
                    {
                        "id_emergencia": id_emergencia,
                        "horario_contexto": horario_contexto,
                        "entities": {},
                        "horario_fim": float(time.time()),
                        "meta": {"error": str(err)},
                    }
                )
                redo_queue.put(id_emergencia)

        time.sleep(0.05)
