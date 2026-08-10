import json
import time
import sys
import os
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

if not os.getenv("TRITON_SERVER_URL_NER"):
    print("Environment variable TRITON_SERVER_URL_NER not set, quitting")
    quit(1)
if not os.getenv("TRITON_SERVER_NER_TIMEOUT_SECS"):
    os.environ["TRITON_SERVER_NER_TIMEOUT_SECS"] = "5"

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


def call_vllm_server_thread_with_schema(
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
    endpoint = os.environ["TRITON_SERVER_URL_NER"]
    base_url = f"http://{endpoint}/v1"
    try:
        vllm_client = OpenAI(base_url=base_url, api_key="EMPTY")
    except Exception as err:
        print("call_vllm_server_thread: Unable to create vllm openai client")
        print(err)
        print(err.with_traceback(None))
        time.sleep(5)
        redo_queue.put(int(params_dict["id_emergencia"]))
        quit(1)

    fmt_class = params_dict["labels"]
    transcript = params_dict["transcript"]
    model_name = params_dict["gliner_mname"]

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
        response = vllm_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "schema": fmt_class.model_json_schema(),
                },
            },
            temperature=0,
            timeout=12,
            extra_body={"enable_thinking": False},
        )

        processing_time = time.time() - start_time
        if response.usage:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

        content_str = response.choices[0].message.content
        raw_response = response
        try:
            parsed_output = fmt_class.model_validate_json(content_str)
            output_json = parsed_output.model_dump()
        except Exception:
            # fallback if invalid JSON
            parsed_dict = json.loads(content_str)
            parsed_output = fmt_class.model_validate(parsed_dict)
            output_json = parsed_output.model_dump()
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
            "model_name": params_dict["gliner_mname"],
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


def vllm_worker_process(
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
    print("Starting vllm worker process for NER")
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
                        "gliner_mname": gliner_mname,
                        "labels": label_list,
                        "top_to_keep": 3,
                        "not_keep_top": many_values,
                        "batch_size": batch_size,
                        "max_tokens": max_tokens,
                        "transcript": transcript,
                    }
                    ner_thread = threading.Thread(
                        target=call_vllm_server_thread_with_schema,
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
