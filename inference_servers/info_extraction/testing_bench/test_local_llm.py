import os
import json
import sys
import time
import hashlib
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from pathlib import Path
import signal

import numpy as np
from tqdm import tqdm
from local_server.vllm_consult import consult_vllm_emergency

from testing import (
    get_testing_inputs,
    find_fmax_per_col_parallel,
    find_max_jw_sim,
)

CONCURRENT_REQUESTS=int(sys.argv[2])
#CONCURRENT_REQUESTS=6

# Keep track of how many times CTRL+C is pressed
_ctrl_c_count = 0

def force_exit_handler(sig, frame):
    global _ctrl_c_count
    _ctrl_c_count += 1
    
    if _ctrl_c_count == 1:
        print("\n[!] CTRL+C pressed. Attempting to stop... (Press again to force kill immediately)")
        # This raises the interrupt so your try/except blocks can try to handle it gracefully
        raise KeyboardInterrupt 
    else:
        print("\n[!] Multiple CTRL+C detected. Forcing hard exit!")
        # os._exit() terminates the process immediately, skipping the thread join wait
        os._exit(1)

# Register the signal handler
signal.signal(signal.SIGINT, force_exit_handler)

class GemmaExtract:
    def __init__(self, model: str):
        self.model = model
        # Cache setup
        self.cache_dir = Path("testing_bench/gemini_cache") / self.safe_model_name(
            model
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def safe_model_name(self, model_name: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_\-]", "_", model_name)

    def get_cache_path(self, transcript: str) -> Path:
        transcript_hash = hashlib.md5(transcript.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{transcript_hash}.json"

    def get_cached_response(self, transcript: str) -> Optional[dict]:
        cache_path = self.get_cache_path(transcript)
        if cache_path.exists():
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

    def extract(self, transcript: str):
        # Check cache first
        '''cached = self.get_cached_response(transcript)
        if cached:
            return (
                cached["entities"],
                cached.get("input_tokens", 0),
                cached.get("output_tokens", 0),
                cached.get("latency", 0.0),
            )'''

        start_time = time.time()
        try:
            result_data = consult_vllm_emergency(transcript, self.model)

            end_time = time.time()
            latency2 = end_time - start_time

            # Save to cache
            #self.save_cached_response(transcript, result_data)

            #print(json.dumps(result_data['entities'], indent=2, ensure_ascii=False))

            return (
                result_data["entities"],
                result_data["input_tokens"],
                result_data["output_tokens"],
                result_data["latency"],
            )

        except Exception as e:
            # Propagate error to let caller handle it (or not)
            raise e


def test_local_model(model_name: str) -> dict:
    (
        ml_categories,
        all_texts,
        clfnames,
        clfs_true,
        full_schema_dict,
        entities_true,
    ) = get_testing_inputs()

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
        extractor = GemmaExtract(model=model_name)
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
    successful_inferences_indices = []

    results_map = {}  # index -> result

    # Using 20 threads as requested
    with ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor:
        future_to_idx = {
            executor.submit(extractor.extract, transcript): idx
            for idx, transcript in enumerate(all_texts)
        }

        for future in tqdm(
            as_completed(future_to_idx),
            total=len(all_texts),
            desc=f"Testing {model_name}",
        ):
            idx = future_to_idx[future]
            try:
                final_json, p_tokens, c_tokens, latency = future.result(timeout=18)
                results_map[idx] = (final_json, p_tokens, c_tokens, latency)
            except Exception as e:
                print(f"Sample {idx} failed: {e}")
                pass
    
    end_time = time.time()
    raw_time = end_time - start_time

    # Reconstruct lists in order, skipping failed ones
    for idx in range(len(all_texts)):
        if idx in results_map:
            final_json, p_tokens, c_tokens, latency = results_map[idx]

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
            successful_inferences_indices.append(idx)

    # Metrics Calculation
    if not successful_inferences_indices:
        print(f"No successful inferences for {model_name}. Skipping metrics.")
        return {}

    clfs_true_no_err = np.asarray([clfs_true[i] for i in successful_inferences_indices])

    if raw_time > 0:
        tokens_per_second_in = p_tokens_total / raw_time
        tokens_per_second_out = c_tokens_total / raw_time
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

    for entity_name in non_redundant_entity_names:
        pred_values = [
            entities_found[i].get(entity_name, []) for i in range(len(entities_found))
        ]

        true_values = [
            entities_true[i].get(entity_name, []) for i in successful_inferences_indices
        ]

        # Guard against empty lists if needed, but find_max_jw_sim should handle it
        fmax, jw_sim_max, recall, precision = find_max_jw_sim(
            pred_values, true_values, field_name=entity_name
        )
        similarities_per_entity[entity_name] = jw_sim_max
        recalls[entity_name] = recall
        precisions[entity_name] = precision

    return {
        "fmax_per_col": fmax_per_col,
        "similarities_per_entity": similarities_per_entity,
        "recalls_at_good_precisions": recalls_at_good_precisions,
        "recalls": recalls,
        "precisions": precisions,
        "best_thresholds": best_thresholds,
        "meta": {
            "tokens_per_second_in": tokens_per_second_in,
            "tokens_per_second_out": tokens_per_second_out,
            "latency_sum": gpu_usage_total,
            "gpu_seconds": raw_time,
            "samples": len(successful_inferences_indices),
            "tokens_total_in": p_tokens_total,
            "tokens_total_out": c_tokens_total,
        },
    }


if __name__ == "__main__":
    results_path = "results/local_llm_results1.json"
    # Models ordered from best quality to lowest quality
    models_to_test = [
        sys.argv[1],
    ]

    models_to_test = list(reversed(models_to_test))

    results = []
    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                results = json.load(f)
        except json.JSONDecodeError:
            results = []

    existing_models = [r.get("model") for r in results]

    for model_name in models_to_test:
        """if model_name in existing_models:
        print(f"Model {model_name} already in results. Skipping.")
        continue"""

        print(f"=== Testing {model_name} ===")

        try:
            metrics = test_local_model(model_name)
            if not metrics:
                print(f"Skipping result save for {model_name} due to lack of metrics.")
                continue

            mean_fmax = np.mean(list(metrics["fmax_per_col"].values()))
            mean_jw_sim = np.mean(list(metrics["similarities_per_entity"].values()))
            mean_recall = np.mean(list(metrics["recalls"].values()))
            mean_precision = np.mean(list(metrics["precisions"].values()))

            print(f"\tMean Fmax: {mean_fmax}")
            print(f"\tMean JW Sim: {mean_jw_sim}")
            print(f"\tMeta: {metrics['meta']}")

            result_entry = {
                "model": model_name,
                "mean_metrics": {
                    "fmax": mean_fmax,
                    "jw_sim": mean_jw_sim,
                    "recall": mean_recall,
                    "precision": mean_precision,
                },
                "meta": metrics["meta"],
                "fmax": metrics["fmax_per_col"],
                "jw_sim": metrics["similarities_per_entity"],
                "recall": metrics["recalls"],
                "precision": metrics["precisions"],
                "best_thresholds": metrics["best_thresholds"],
            }

            results.append(result_entry)

            # Ensure dir exists
            os.makedirs(os.path.dirname(results_path), exist_ok=True)

            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)

        except Exception as e:
            print(f"Failed to test {model_name}: {e}")
