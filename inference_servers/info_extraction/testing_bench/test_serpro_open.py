from enum import Enum
from typing import Optional
import json
import time
import os
import sys
import signal
import hashlib
import re
from pathlib import Path

from pydantic import BaseModel
from dotenv import load_dotenv
import requests
from tqdm import tqdm
import numpy as np

from local_server.ollama_consult import (
    InformacoesOcorrencia,
    answer_to_value,
    question_mapping,
    prompt_a,
)

from local_server.vllm_consult import process_vllm_response

from testing import (
    get_testing_inputs,
    find_fmax_per_col_parallel,
    find_max_jw_sim,
    USE_PERC,
)

load_dotenv()

USER_CODE = os.getenv("SERPRO_USER_CODE")


# Guided decoding by JSON using Pydantic schema
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
        if verbose:
            print("Initial json parsing failed for: ", raw, file=sys.stderr)
            print("Attempting to strip markdown and parse again...", file=sys.stderr)
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
            return answer, meta
    # failure reached
    meta = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "request_time": req_latency / failures,
        "failures": failure_texts,
    }
    return None, meta


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


class SerproAPIExtract:
    def __init__(self, model: str, offline_mode: bool = False):
        self.model = model
        if not offline_mode:
            self.access_token = get_serpro_token()
        else:
            self.access_token = None
        self.safe_model_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", model)
        self.cache_dir = "testing_bench/serpro_caches/" + self.safe_model_name
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_cache_path(self, transcript: str) -> str:
        transcript_hash = hashlib.md5(transcript.encode("utf-8")).hexdigest()
        return self.cache_dir + "/" + f"{transcript_hash}.json"

    def get_cached_response(self, transcript: str) -> Optional[dict]:
        cache_path = self.get_cache_path(transcript)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return None
        return None

    def save_cached_response(self, transcript: str, response_data: dict):
        cache_path = self.get_cache_path(transcript)
        try:
            with open(cache_path, "w") as f:
                json.dump(response_data, f, indent=4)
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}", file=sys.stderr)

    def extract(self, transcript: str, use_cache=True, verbose=False):
        if use_cache:
            cached = self.get_cached_response(transcript)
            if cached:
                if verbose:
                    print("Success: using cached response")
                prompt_tokens = cached.get("prompt_tokens", None)
                if prompt_tokens is None:
                    prompt_tokens = cached.get("input_tokens", None)
                completion_tokens = cached.get("completion_tokens", None)
                if completion_tokens is None:
                    completion_tokens = cached.get("output_tokens", None)
                request_time = cached.get("request_time", None)
                if request_time is None:
                    request_time = cached.get("latency", None)
                if "failures" in cached:
                    if len(cached["failures"]) > 0:
                        cached["failures"] = cached["failures"][0]
                else:
                    cached["failures"] = []
                return (
                    cached["entities"],
                    prompt_tokens,
                    completion_tokens,
                    request_time,
                    cached.get("failures", []),
                )

        if self.access_token is None:
            return (None, None, None, None, ["No access token"])

        prompt = prompt_a.replace("<transcript_placeholder>", transcript)

        fmd_json = InformacoesOcorrencia.model_json_schema()
        prompt = prompt.replace(
            "<template_placeholder>\n", json.dumps(fmd_json, ensure_ascii=False)
        )

        answer, meta = request_serpro_generic(
            self.model,
            self.access_token,
            prompt,
            InformacoesOcorrencia,
            max_tokens=2800,
            verbose=verbose,
        )

        if len(meta["failures"]) > 0:
            meta["failures"] = meta["failures"][0]

        if answer is not None:
            result_data = process_vllm_response(
                answer,
                meta["prompt_tokens"],
                meta["completion_tokens"],
                meta["request_time"],
            )
            result_data["failures"] = meta["failures"]
            if use_cache:
                self.save_cached_response(transcript, result_data)
                if verbose:
                    print("Success: saved")
            return (
                result_data,
                meta["prompt_tokens"],
                meta["completion_tokens"],
                meta["request_time"],
                meta["failures"],
            )
        else:
            if verbose:
                print("Failure: not saved")
            return (None, None, None, meta["request_time"], meta["failures"])


def test_local_model(
    model_name: str, testing_perc: float, use_cache=True, verbose=False
):
    (
        ml_categories,
        all_texts,
        clfnames,
        clfs_true,
        full_schema_dict,
        entities_true,
    ) = get_testing_inputs(usage_perc=testing_perc)

    entity_names = list(full_schema_dict["entities"].keys())

    # Identify non-redundant names for metrics
    redundancies = {
        "rua": "rua_ou_logradouro",
        "municipio": "cidade",
        "street_number": "numero",
        "number": "numero",
        "endereço_complemento": "complemento",
    }

    # Setup extractor
    try:
        extractor = SerproAPIExtract(model=model_name, offline_mode=True)
    except Exception as e:
        print(f"Failed to initialize extractor for {model_name}: {e}")
        return {}

    # Lists to store results
    clfs_scores = []
    entities_found = []
    gpu_usage_total = 0.0
    p_tokens_total = 0
    c_tokens_total = 0
    start_time = time.time()
    n_failures = 0
    results = []
    success_idx = []

    bar = tqdm(total=len(all_texts))

    failure_texts = []
    n_failures = 0

    for idx, transcript in enumerate(all_texts):
        res = extractor.extract(transcript, use_cache=use_cache, verbose=verbose)
        # print(res)
        # n_failures += len(res[4])
        if res[0] is not None:
            results.append(res)
            success_idx.append(idx)
        else:
            failure_texts.append(res[4])
            n_failures += 1
        bar.update(1)

    if len(results) == 0:
        print(f"No successful inferences for {model_name}. Skipping metrics.")
        return {}

    for res in results:
        final_json, p_tokens, c_tokens, latency, failures = res
        if failures == []:
            # failure_texts.append("No failures")
            pass
        else:
            failure_texts.append(failures)
        p_tokens_total += p_tokens
        c_tokens_total += c_tokens
        gpu_usage_total += latency

        # Flatten scores for classification
        scores_line = []
        for clf in clfnames:
            scores_line.append(final_json.get(clf, 0.0))

        # Flatten entities
        entities_line = {}
        for name in entity_names:
            val = final_json.get(name, [])
            entities_line[name] = val

        clfs_scores.append(scores_line)
        entities_found.append(entities_line)
    n_requests = len(results) + n_failures
    failures_perc = n_failures / n_requests
    clfs_true_no_err = np.asarray([clfs_true[i] for i in success_idx])

    if gpu_usage_total > 0:
        tokens_per_second_in = p_tokens_total / gpu_usage_total
        tokens_per_second_out = c_tokens_total / gpu_usage_total
    else:
        tokens_per_second_in = 0
        tokens_per_second_out = 0

    fmax_per_col, recalls_at_good_precisions, recalls, precisions, best_thresholds = (
        find_fmax_per_col_parallel(
            np.array(clfs_scores),
            clfs_true_no_err,
            clfnames,
            n_jobs=4,
        )
    )

    non_redundant_entity_names = [
        name for name in entity_names if name not in redundancies.keys()
    ]

    similarities_per_entity = {}
    similarities_per_entity_simple = {}
    recalls_simple = {}
    precisions_simple = {}
    verbosity_ratios = {}

    for entity_name in non_redundant_entity_names:
        pred_values = [
            entities_found[i].get(entity_name, []) for i in range(len(entities_found))
        ]

        true_values = [entities_true[i].get(entity_name, []) for i in success_idx]

        fmax1, jw_sim_max1, recall1, precision1, _ = find_max_jw_sim(
            pred_values, true_values, field_name=entity_name, use_simple=True
        )
        similarities_per_entity_simple[entity_name] = jw_sim_max1
        recalls_simple[entity_name] = recall1
        precisions_simple[entity_name] = precision1

        fmax2, jw_sim_max2, recall2, precision2, vr = find_max_jw_sim(
            pred_values, true_values, field_name=entity_name, use_simple=False
        )
        similarities_per_entity[entity_name] = jw_sim_max2
        recalls[entity_name] = recall2
        precisions[entity_name] = precision2
        verbosity_ratios[entity_name] = vr

    return {
        "fmax_per_col": fmax_per_col,
        "similarities_per_entity": similarities_per_entity,
        "similarities_per_entity_simple": similarities_per_entity_simple,
        "recalls_at_good_precisions": recalls_at_good_precisions,
        "recalls": recalls,
        "recalls_simple": recalls_simple,
        "precisions": precisions,
        "precisions_simple": precisions_simple,
        "best_thresholds": best_thresholds,
        "verbosity_ratios": verbosity_ratios,
        "meta": {
            "tokens_per_second_in": tokens_per_second_in,
            "tokens_per_second_out": tokens_per_second_out,
            "latency_sum": gpu_usage_total,
            "gpu_seconds": gpu_usage_total,
            "samples": len(success_idx),
            "tokens_total_in": p_tokens_total,
            "tokens_total_out": c_tokens_total,
            "failures": n_failures,
            "failures_perc": failures_perc,
            "n_requests": n_requests,
            "failure_texts": failure_texts,
        },
    }


model_sizes = [
    {"llama-3.1-8B-instruct": 8},
    {"qwen3.5-35b": 35},
    {"mistral-small-3.2-24b-instruct": 24},
    {"magistral-small": 24},
    {"gemma-3n-e4b-it": 4},
    {"gpt-oss-120b": 120},  # 117B parameters with 5.1B active parameters
    {"gemma-3-4b-it": 4},
    {"deepseek-r1-distill-qwen-14b": 14},
]

if __name__ == "__main__":
    testing_perc = USE_PERC
    if len(sys.argv) > 1:
        testing_perc = float(sys.argv[1])
    results_path = "results/serpro_open_results.json"
    """access_token = get_serpro_token()
    print("Got temp access token")
    structured_models = serpro_initial_model_tests(access_token)"""
    structured_models = [
        {"name": "qwen3.5-35b", "latency": 0.8253350257873535},
        {"name": "llama-3.1-8B-instruct", "latency": 0.7867984771728516},
        {"name": "mistral-small-3.2-24b-instruct", "latency": 1.0027306079864502},
        {"name": "magistral-small", "latency": 1.075148582458496},
        {"name": "gemma-3n-e4b-it", "latency": 1.1260874271392822},
        {"name": "gpt-oss-120b", "latency": 1.3318655490875244},
        {"name": "gemma-3-4b-it", "latency": 3.1739470958709717},
        {"name": "deepseek-r1-distill-qwen-14b", "latency": 7.509040117263794},
    ]

    print("Structured llms found: ", structured_models)

    models_to_test = [d["name"] for d in structured_models]

    # models_to_test = [models_to_test[0]]

    model_results = []

    for model_name in models_to_test:

        print(f"=== Testing {model_name} ===")

        try:
            metrics = test_local_model(
                model_name, testing_perc, use_cache=True, verbose=False
            )
            if not metrics:
                print(f"Skipping result save for {model_name} due to lack of metrics.")
                continue

            mean_fmax = np.mean(list(metrics["fmax_per_col"].values()))
            mean_jw_sim = np.mean(list(metrics["similarities_per_entity"].values()))
            mean_jw_sim_simple = np.mean(
                list(metrics["similarities_per_entity_simple"].values())
            )
            mean_recall = np.mean(list(metrics["recalls"].values()))
            mean_recall_simple = np.mean(list(metrics["recalls_simple"].values()))
            mean_precision = np.mean(list(metrics["precisions"].values()))
            mean_precision_simple = np.mean(list(metrics["precisions_simple"].values()))
            mean_verbosity_ratio = np.mean(list(metrics["verbosity_ratios"].values()))

            print(f"\tMean Fmax: {mean_fmax}")
            print(f"\tMean JW Sim: {mean_jw_sim}")
            print(f"\tMean JW Sim Simple: {mean_jw_sim_simple}")
            print(f"\tMean Recall: {mean_recall}")
            print(f"\tMean Recall Simple: {mean_recall_simple}")
            print(f"\tMean Precision: {mean_precision}")
            print(f"\tMean Precision Simple: {mean_precision_simple}")
            print(f"\tMean Verbosity Ratio: {mean_verbosity_ratio}")
            print(f"\tMeta: {metrics['meta']}")

            result_entry = {
                "model": model_name,
                "mean_metrics": {
                    "fmax": mean_fmax,
                    "jw_sim": mean_jw_sim,
                    "jw_sim_simple": mean_jw_sim_simple,
                    "recall": mean_recall,
                    "recall_simple": mean_recall_simple,
                    "precision": mean_precision,
                    "precision_simple": mean_precision_simple,
                    "verbosity_ratio": mean_verbosity_ratio,
                },
                "meta": metrics["meta"],
                "fmax": metrics["fmax_per_col"],
                "jw_sim": metrics["similarities_per_entity"],
                "jw_sim_simple": metrics["similarities_per_entity_simple"],
                "recall": metrics["recalls"],
                "recall_simple": metrics["recalls_simple"],
                "precision": metrics["precisions"],
                "precision_simple": metrics["precisions_simple"],
                "verbosity_ratio": metrics["verbosity_ratios"],
                "best_thresholds": metrics["best_thresholds"],
                "detailed_results": metrics,
            }

            model_results.append(result_entry)

            # Ensure dir exists
            os.makedirs(os.path.dirname(results_path), exist_ok=True)

            with open(results_path, "w") as f:
                json.dump(model_results, f, indent=4)

        except Exception as e:
            print(f"Failed to test {model_name}: {e}")
            raise e
