import json
import os
import sys
import time
import sqlite3 as sqlite
import multiprocessing as mp
import threading
from typing import List, Tuple, Any, Optional
from abc import ABC, abstractmethod

from openai import AzureOpenAI
from pydantic import ValidationError

from runners.gliner_utilities import (
    ner_labels_pydantic,
    schema_key_translations,
    many_values,
    values_to_ignore,
    env_vals,
)

azure_context_lengths = {
    "gpt-5-nano": 400000,
}


class InterpretationClient(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.context_len: Optional[int] = 1001

    @abstractmethod
    def extract_structured(
        self, input_dict: dict
    ) -> Tuple[Any, Optional[int], Optional[int], float, Any]:
        """
        Deve executar uma única chamada e retornar:
        (structured_resp, input_tokens, output_tokens, processing_time, raw_response)
        """
        raise NotImplementedError


class AzureAPIRunner(InterpretationClient):
    system_prompt = "Você é um assistente que sempre responde estritamente no formato JSON especificado."

    def __init__(self, model_name):
        super().__init__(model_name)

        self.api_key = os.environ["AZURE_AI_RESOURCE_KEY_" + model_name]
        try:
            self.api_version = os.environ["AZURE_AI_RESOURCE_VERSION_" + model_name]
        except KeyError:
            self.api_version = os.environ["AZURE_AI_RESOURCE_VERSION"]
        try:
            self.endpoint = os.environ["AZURE_AI_RESOURCE_ENDPOINT_" + model_name]
        except KeyError:
            self.endpoint = os.environ["AZURE_AI_RESOURCE_ENDPOINT"]
        self.model_name = model_name

        self.client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
        )

        self.context_len = azure_context_lengths.get(model_name, 400000)

    def extract_structured(self, input_dict):
        prompt = input_dict["prompt"]
        format_schema = input_dict["format"]

        inf_start = time.time()
        try:
            response = self.client.responses.parse(
                model=self.model_name,
                input=prompt,
                text_format=format_schema,
                reasoning={"effort": "minimal"},
            )
        except ValidationError as err:
            print("Pydantic Validation error for prompt:\n", prompt, file=sys.stderr)
            print(type(err))
            print(err)
            print(err.args)
            print(err.__traceback__)
            print(err.__context__)
            return None, None, None, time.time() - inf_start, err

        except Exception as err:
            inf_time = time.time() - inf_start
            print(
                f"\n🚨 [AzureAPIRunner] API Connection/Execution Error!",
                file=sys.stderr,
            )
            print(f"   Model: {self.model_name}", file=sys.stderr)
            print(f"   Endpoint: '{self.endpoint}'", file=sys.stderr)
            print(f"   API Version: {self.api_version}", file=sys.stderr)
            print(f"   Error Type: {type(err).__name__}", file=sys.stderr)
            print(f"   Message: {err}\n", file=sys.stderr)

            return None, None, None, inf_time, err

        inf_time = time.time() - inf_start

        resp0 = response.output_parsed
        if resp0 is not None:
            usage = getattr(response, "usage", None)
            output_tokens = getattr(usage, "output_tokens", None) if usage else None
            input_tokens = getattr(usage, "input_tokens", None) if usage else None
            return resp0, input_tokens, output_tokens, inf_time, response

        return None, None, None, inf_time, response


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


def call_azure_server_thread(params_dict: dict, redo_queue: mp.Queue):
    """
    Azure version of the structured inference worker.

    GEMINI_API_KEY is kept only to preserve the existing call signature.
    Azure credentials are loaded from:
      AZURE_AI_RESOURCE_KEY_<model_name>
      AZURE_AI_RESOURCE_VERSION_<model_name>
      AZURE_AI_RESOURCE_ENDPOINT_<model_name>
    """
    proc_start = time.time()

    try:
        runner = AzureAPIRunner(params_dict["gliner_mname"])
    except Exception as err:
        print("call_azure_server_thread: Unable to create Azure client")
        print(err)
        print(err.with_traceback(None))
        time.sleep(5)
        redo_queue.put(int(params_dict["id_emergencia"]))
        return

    format_schema = params_dict["labels"]
    input_tokens = None
    output_tokens = None
    processing_time = 0.0

    try:
        output_json, input_tokens, output_tokens, processing_time, response_obj = (
            runner.extract_structured(
                {
                    "prompt": params_dict["transcript"],
                    "format": format_schema,
                }
            )
        )

        if output_json is not None:
            if hasattr(output_json, "model_dump"):
                output_json = output_json.model_dump()
            elif isinstance(output_json, dict):
                output_json = output_json
            else:
                # Last resort: try to serialize pydantic/dataclass-like objects
                output_json = json.loads(
                    json.dumps(output_json, ensure_ascii=False, default=str)
                )

            error_type = None
        else:
            error_type = "unknown_error"

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
            if key not in response_json:
                response_json[key] = []
            if val is None:
                continue
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
            if isinstance(response_json[label], list):
                response_json[label].sort(key=lambda x: x[1], reverse=True)
                if (
                    label not in not_keep_top
                    and len(response_json[label]) > top_to_keep
                ):
                    response_json[label] = response_json[label][:top_to_keep]

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


def azure_worker_process(
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
    while True:
        try:
            tp = transcript_queue.get()
        except Exception:
            tp = None

        if tp is not None:
            try:
                id_emergencia, horario_contexto, transcript = tp
                label_list = ner_labels_pydantic[labels_set]
                should_stop = "STOP" == transcript

                if should_stop:
                    break

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
                    target=call_azure_server_thread,
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
