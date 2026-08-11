import os
import sys
import json
import time
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import tritonclient.http as http_client

from testing import (
    get_testing_inputs
)

def call_triton_server(transcript, schema_str, endpoint, model="gliner_x_large"):
    # 1. Initialize the client
    try:
        client = http_client.InferenceServerClient(url=endpoint)
    except Exception as e:
        print(f"Error initializing client: {e}")
        time.sleep(2)
        quit(1)
    
    # 2. Define expected inputs
    inputs = [
        http_client.InferInput("PROMPT", [1, 1], "BYTES"),
        http_client.InferInput("LABEL_LIST", [1, 1], "BYTES")
    ]
    
    # 3. Format data correctly for Triton's BYTES datatype
    inputs[0].set_data_from_numpy(np.array([[transcript.encode('utf-8')]]))
    inputs[1].set_data_from_numpy(np.array([[schema_str.encode('utf-8')]]))
    
    # 4. Execute and time the request
    start_time = time.time()
    results = client.infer(model_name=model, inputs=inputs)
    latency = time.time() - start_time
    
    # 5. Decode the raw bytes back into JSON
    try:
        final_json = json.loads(results.as_numpy("ENTITIES_JSON")[0].decode('utf-8'))
        meta_info = json.loads(results.as_numpy("META_INFO")[0].decode('utf-8'))
        p_tokens = meta_info.get("input_tokens", 0)
        c_tokens = meta_info.get("output_tokens", 0)
        no_gpu_time = meta_info.get("no_gpu_time", 0)
        processing_time = meta_info.get("processing_time", 0)

        return final_json, p_tokens, c_tokens, latency, no_gpu_time, processing_time

    except Exception as e:
        print(f"Error decoding response: {e}")
        time.sleep(2)
        quit(1)

def test_triton_runtimes(model="gliner_x_large", endpoint="0.0.0.0:8003", n_tests=160):
    ml_categories, all_texts, clfnames, clfs_true, full_schema_dict, entities_true = (
        get_testing_inputs()
    )
    schema_str = json.dumps(full_schema_dict, ensure_ascii=False, indent=2)

    results_map = {}  # index -> result

    # Using 16 threads as requested
    less_texts = all_texts[:n_tests]
    requesting_start = time.time()
    
    with ThreadPoolExecutor(max_workers=16) as executor:
        future_to_idx = {
            executor.submit(call_triton_server, transcript, schema_str, endpoint, model): idx
            for idx, transcript in enumerate(less_texts)
        }

        for future in tqdm(
            as_completed(future_to_idx),
            total=len(less_texts),
            desc=f"Testing {model}",
        ):
            idx = future_to_idx[future]
            try:
                final_json, p_tokens, c_tokens, latency, no_gpu_time, processing_time = future.result()
                results_map[idx] = (final_json, p_tokens, c_tokens, latency, no_gpu_time, processing_time)
                #print(json.dumps(final_json, ensure_ascii=False, indent=2))
            except Exception as e:
                print(f"Sample {idx} failed: {e}")
                pass
    
    requesting_end = time.time()
    requesting_time = requesting_end - requesting_start
    time_per_transcript = requesting_time / len(less_texts)

    no_gpu_time_sum = 0
    gpu_time_sum = 0
    for final_json, p_tokens, c_tokens, latency, no_gpu_time, processing_time in results_map.values():
        no_gpu_time_sum += no_gpu_time
        gpu_time_sum += processing_time
    
    print(f"Total Requesting time: {requesting_time:.3f}s")
    print(f"Mean Time per transcript: {time_per_transcript:.3f}s")
    print(f"Total No GPU time: {no_gpu_time_sum:.3f}s")
    print(f"Mean No GPU time: {no_gpu_time_sum / len(less_texts):.3f}s")
    print(f"Total GPU time: {gpu_time_sum:.3f}s")
    print(f"Mean GPU time: {gpu_time_sum / len(less_texts):.3f}s")

    return requesting_time, time_per_transcript

if __name__ == "__main__":
    time1, time_mean = test_triton_runtimes()