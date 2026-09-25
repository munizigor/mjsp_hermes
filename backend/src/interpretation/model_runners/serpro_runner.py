from enum import Enum
import json
import time
import os
import sys
import hashlib
import re
from typing import List, Optional, Tuple, Any

from pydantic import BaseModel
import requests
from tqdm import tqdm
import numpy as np

from model_runners.templates import InterpretationClient


if os.path.exists(".env"):
    env_vals = {
        rawline.split("=")[0]: rawline.split("=")[1].rstrip("\n")
        for rawline in open(".env", "r").read().split("\n")
        if "=" in rawline
    }
    for k, v in env_vals.items():
        os.environ[k] = v

USER_CODE = os.getenv("SERPRO_USER_CODE")
SERPRO_BASE_URL = "https://e-api-serprollm.ni.estaleiro.serpro.gov.br"
SERPRO_LLM_BASE_URL = f"{SERPRO_BASE_URL}/gateway/v1"
SERPRO_OAUTH_URL = f"{SERPRO_BASE_URL}/oauth2/token"
#Examples for guided decoding by JSON using Pydantic schema
class CarType(str, Enum):
    sedan = "sedan"
    suv = "SUV"
    truck = "Truck"
    coupe = "Coupe"

class CarDescription(BaseModel):
    brand: str
    model: str
    car_type: CarType


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


def list_models(access_token):
    """
    curl -k -X GET -H "Authorization: Bearer XXXXXXXXXXXXXXXXXX" https://e-api-serprollm.ni.estaleiro.serpro.gov.br/gateway/v1/models
    """
    base_url = "https://e-api-serprollm.ni.estaleiro.serpro.gov.br/gateway/v1/models"
    headers = {"Authorization": "Bearer " + access_token}
    response = requests.request("GET", base_url, headers=headers)
    content = json.loads(response.text)
    models = []
    for m in content["data"]:
        name = m["id"]
        type = m["owned_by"]
        models.append({"name": name, "type": type})
    return models


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


def serpro_test_model(model, access_token):
    prompt = "Generate a JSON with the brand, model and car_type of the most iconic car from the 90's"
    model_class = CarDescription
    answer, meta = request_serpro_generic(
        model, access_token, prompt, model_class, max_tokens=3000
    )
    if answer is None:
        return [], [], meta
    else:
        return [answer], [""], meta


def serpro_initial_model_tests(access_token):
    models_available = list_models(access_token)
    for m in models_available:
        print(m)

    model_stopwords = ["guard", "devstral", "pixtral"]
    vllm_models = [
        m
        for m in models_available
        if m["type"] == "vllm"
        and not any([word in m["name"] for word in model_stopwords])
    ]
    results = []
    for m in reversed(vllm_models):
        print("Testing: " + m["name"])
        try:
            answers, reasonings, meta = serpro_test_model(m["name"], access_token)
            print(json.dumps(answers, indent=2))
            results.append(
                {
                    "model": m["name"],
                    "structured_response": True,
                    # "reasoning": resp[1] != "",
                    "request_time": meta["request_time"],
                    "total_tokens": meta["total_tokens"],
                }
            )
        except Exception as e:
            print(e)
            results.append(
                {
                    "model": m["name"],
                    "structured_response": False,
                    # "reasoning": None,
                    "request_time": 0,
                    "total_tokens": 0,
                }
            )
        print("\n")

    print("\n\nModel Test Results:\n\n")
    print(json.dumps(results, indent=2))

    results.sort(key=lambda x: x["request_time"])

    structured_models = [
        {"name": r["model"], "latency": r["request_time"]}
        for r in results
        if r["structured_response"]
    ]
    return structured_models

class SerproLLMTextAPIRunner(InterpretationClient):
    """
    Runner that encapsulates vLLM OpenAI-compatible API calls.
    """

    system_prompt = "Você é um assistente que sempre responde estritamente no formato JSON especificado."

    def __init__(
        self,
        model_name: str,
    ):
        super().__init__(model_name)
        self.model_name = model_name
        self.base_url = SERPRO_LLM_BASE_URL
        self.access_token = get_serpro_token()
        self.context_len = 1000000
    
    def extract_structured(
        self, input_dict: dict, verbose: bool = True
    ) -> Tuple[Any, Optional[int], Optional[int], float, Any]:
        """
        Executes a single structured extraction call.

        Expected input_dict format:
        {
            "prompt": str,
            "format": Pydantic BaseModel class
        }"""
        prompt = input_dict["prompt"]
        format_schema = input_dict["format"]
        fmd_json = format_schema.model_json_schema()

        answer, meta, response_json = request_serpro_generic(
            self.model_name,
            self.access_token,
            prompt,
            format_schema,
            max_tokens=3600,
            verbose=verbose,
        )

        if len(meta["failures"]) > 0:
            meta["failures"] = meta["failures"][0]

        if answer is not None:
            return (
                answer,
                meta["prompt_tokens"],
                meta["completion_tokens"],
                meta["request_time"],
                response_json,
            )
        else:
            if verbose:
                print("Failure: not saved")
            return (None, None, None, meta["request_time"], response_json)

        return parsed_output, input_tokens, output_tokens, latency, response_json