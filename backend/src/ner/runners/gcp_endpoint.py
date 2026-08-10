import json
import time
import sys
import sqlite3 as sqlite
import os
import multiprocessing as mp
from queue import Empty as QueueEmptyError
import threading

import numpy as np
from google import genai
from google.genai import types  # Import types for Config
from pydantic import ValidationError

from runners.gliner_utilities import (
    ner_labels_pydantic,
    schema_key_translations,
    many_values,
    values_to_ignore,
    env_vals,
)

if not os.getenv("GEMINI_API_KEY"):
    print("Environment variable GEMINI_API_KEY not set, quitting")
    quit(1)
if not os.getenv("GEMINI_NER_TIMEOUT_SECS"):
    os.environ["GEMINI_NER_TIMEOUT_SECS"] = "5"

timeout_secs = int(os.environ["GEMINI_NER_TIMEOUT_SECS"])

system_prompt = "Você é um assistente que sempre responde estritamente no formato JSON especificado."


def save_ner_to_db(to_save: dict, redo_queue: mp.Queue):
    """
    Args:
    to_save: Dictionary with fields:
        meta:
            processing_time;
            no_gpu_time;
            input_tokens;
            output_tokens;
            model_name;
        id_emergencia;
        horario_contexto;
        horario_fim;
        entities;
    """
    cols = """id_emergencia, horario_contexto, horario_fim, 
            tipo_de_inferencia, resultado, duracao_inferencia, 
            duracao_outros_processamentos, input_tokens, output_tokens, 
            modelo_utilizado"""
    sqlite_conn = sqlite.connect(os.environ["SQLITE_DB_PATH"])
    cursor = sqlite_conn.cursor()
    if "error" not in to_save["meta"]:
        id_emergencia = to_save["id_emergencia"]
        meta = to_save["meta"]
        print(f"worker_save_everything: {meta}")

        id_emergencia = int(to_save["id_emergencia"])
        horario_contexto = to_save["horario_contexto"]
        horario_fim = to_save["horario_fim"]
        tipo_de_inferencia = "ner"
        resultado = json.dumps(to_save["entities"], ensure_ascii=False)
        duracao_inferencia = meta.get("processing_time", None)
        duracao_outros_processamentos = meta.get("no_gpu_time", None)
        input_tokens = meta.get("input_tokens", None)
        output_tokens = meta.get("output_tokens", None)
        modelo_utilizado = meta.get("model_name", None)
        new_line = (
            id_emergencia,
            horario_contexto,
            horario_fim,
            tipo_de_inferencia,
            resultado,
            duracao_inferencia,
            duracao_outros_processamentos,
            input_tokens,
            output_tokens,
            modelo_utilizado,
        )
        cursor.execute(
            f"""INSERT INTO resultados_inferencia 
            ({cols}) VALUES (?,?,?,?,?,?,?,?,?,?)""",
            new_line,
        )
    else:
        id_emergencia = to_save["id_emergencia"]
        print("gliner_inference_saver: Error in inference, not saving to DB")
        print(to_save)

    sqlite_conn.commit()
    sqlite_conn.close()
    redo_queue.put(int(to_save["id_emergencia"]))


def call_gcp_server_thread(
    params_dict: dict, redo_queue: mp.Queue, GEMINI_API_KEY: str
):
    """Runs NER and Classification using an endpoint in GCP.
    After getting the results, calls the saver function.

    The API key is in the environment variable GEMINI_API_KEY.

    Args:
    params_dict: dict with the following keys:
        - gliner_mname: name of the gcp model
        - labels: list of labels
        - top_to_keep: number of top labels to keep
        - not_keep_top: list of labels to not keep
        - batch_size: batch size
        - max_tokens: max tokens
        - transcript: transcript
        - id_emergencia:
        - horario_contexto:

    """
    proc_start = time.time()
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

    except Exception as err:
        print("call_gcp_server_thread: Unable to create GCP client")
        print(err)
        print(err.with_traceback(None))
        time.sleep(5)
        redo_queue.put(int(params_dict["id_emergencia"]))
        quit(1)

    format_schema = params_dict["labels"]
    input_tokens = None
    output_tokens = None
    processing_time = 0.0
    try:
        inf_start = time.time()
        response = client.models.generate_content(
            model=params_dict["gliner_mname"],
            contents=params_dict["transcript"],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                # You can pass the Pydantic class directly in the new SDK!
                response_schema=format_schema,
                system_instruction=system_prompt,
            ),
        )
        processing_time = time.time() - inf_start

        # The SDK can often parse automatically into the Pydantic object:
        # parsed_output = response.parsed
        # However, sticking to manual validation is safe and robust:
        parsed_output = format_schema.model_validate_json(response.text)
        output_json = parsed_output.model_dump()
        # print('Parsed Output:', parsed_output, file=sys.stderr)
        input_tokens = None
        output_tokens = None
        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count
        error_type = None
    except ValidationError as err:
        print(
            "Pydantic Validation error for prompt:\n",
            params_dict["transcript"],
            file=sys.stderr,
        )
        error_type = "Pydantic_validation_error"
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


def gcp_worker_process(
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
    GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
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
                        target=call_gcp_server_thread,
                        args=(params, redo_queue, GEMINI_API_KEY),
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
