import time
import os
import json
import sys
from typing import List
from collections import deque
from itertools import islice
from typing import List
from multiprocessing import Pool
import gc
import re
from copy import copy
import numpy as np
from tqdm import tqdm
import torch
import polars as pl
from openai import OpenAI, APITimeoutError, APIConnectionError

from testing import (entity_redundancies, SIM_TH, NAME_SIM_TH, NUMBER_TH, HONORIFICS, basic_norm, 
    get_testing_inputs, find_fmax_per_col_parallel, find_max_jw_sim)

# Initialize client pointing to your local vLLM instance
# Note: "api_key" can be anything for local vLLM, but must be present.

class NuExtractCaller:
    def __init__(self, model: str = "numind/NuExtract-2.0-2B", timeout: int = 10,
            endpoint="http://localhost:8000/v1", api_key="putarealsecret"):
        self.model = model
        self.client = OpenAI(
            base_url=endpoint,
            api_key=api_key
        )

        self.entity_redundancies = {
            "rua": "rua_ou_logradouro",
            "municipio": "cidade",
            "street_number": "numero",
            "number": "numero",
            "endereço_complemento": "complemento"
        }

        self.answer_to_value = {
            "Certamente Sim": 1.0,
            "Sim": 0.8,
            "Provavelmente Sim": 0.65,
            "Não sei": 0.5,
            "Provavelmente Não": 0.35,
            "Não": 0.2,
            "Certamente Não": 0.0
        }

    def extract_with_retry(self, transcript: str, schema: dict):
        """
        Extracts entities based on a schema using NuExtract with a retry mechanism.
        
        Args:
            transcript (str): The text to process.
            schema (dict): The dictionary defining the extraction structure.

        Returns:
            tuple: (recognized_entities (dict), prompt_tokens (int), completion_tokens (int), latency (float))
        
        Raises:
            Exception: If the request fails after 2 attempts.
        """
        
        # NuExtract expects the schema to be passed as a stringified JSON in the template
        # We use 'extra_body' to pass 'chat_template_kwargs' which the OpenAI client usually filters out.
        payload_extra = {
            "chat_template_kwargs": {
                "template": json.dumps(schema),
                # You can add few-shot examples here if needed
                "examples": [] 
            }
        }

        messages = [
            {"role": "user", "content": transcript}
        ]

        max_retries = 2
        
        for attempt in range(1, max_retries + 1):
            start_time = time.time()
            try:
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    extra_body=payload_extra,  # Critical for passing NuExtract specific args
                    temperature=0.1,           # Low temp for extraction
                    timeout=self.timeout
                )
                
                end_time = time.time()
                latency = end_time - start_time

                # 1. Parse Usage
                usage = response.usage
                input_tokens = usage.prompt_tokens
                output_tokens = usage.completion_tokens
                
                # 2. Parse Content (The model returns a JSON string)
                content_str = response.choices[0].message.content
                try:
                    recognized_entities = json.loads(content_str)
                except json.JSONDecodeError:
                    # Fallback if model outputs markdown or bad JSON
                    recognized_entities = {"error": "Failed to parse JSON", "raw_content": content_str}

                return recognized_entities, input_tokens, output_tokens, latency

            except (APITimeoutError, APIConnectionError) as e:
                print(f"Attempt {attempt} failed: {e}")
                if attempt == max_retries:
                    raise Exception(f"Failed after {max_retries} attempts. Last error: {e}")
            except Exception as e:
                # Immediate fail for non-network errors (e.g., bad request)
                raise e

    def get_entities(self, full_schema_dict: dict, transcript: str):
        

        nuextract_schema = {}
        for key, value in full_schema_dict['entities'].items():
            nuextract_schema[key] = "verbatim-string"
        
        ml_categories = set()
        for key, value in full_schema_dict['boolean'].items():
            for clf_group, clfs in value.items():
                for clf in clfs:
                    nuextract_schema[clf] = list(self.answer_to_value.keys())
                    ml_categories.add(clf)
        
        entity_names = list(full_schema_dict['entities'].keys())
        
        new_entities, p_tokens, c_tokens, req_duration1 = self.extract_with_retry(
            transcript=transcript,
            schema=nuextract_schema
        )

        processed_entities = {}

        for clf in ml_categories:
            if clf not in new_entities:
                new_entities[clf] = "Não sei"
            elif new_entities[clf] is None:
                new_entities[clf] = "Não sei"
            val_float = self.answer_to_value[new_entities[clf]]
            processed_entities[clf] = val_float

        for name in entity_names:
            if name in new_entities:
                raw_val = new_entities[name]
                if raw_val is None:
                    raw_val = []
                if type(raw_val) == str:
                    raw_val = [raw_val]
                no_repeats = list(set(raw_val))
                processed_entities[name] = [(w, 1.0) for w in no_repeats]
        
        for redundant, correct_name in entity_redundancies.items():
            if redundant in processed_entities:
                if not correct_name in processed_entities:
                    processed_entities[correct_name] = []
                processed_entities[correct_name].extend(processed_entities[redundant])
                del processed_entities[redundant]

        return processed_entities, p_tokens, c_tokens, req_duration1


def test_nuextract_model(model, endpoint) -> dict:
    ml_categories, all_texts, clfnames, clfs_true, full_schema_dict, entities_true = get_testing_inputs()
    #clfnames_no_inter = [c.replace(' ?', '') for c in clfnames]
    nuextract_schema = {
        "rua_ou_logradouro": "verbatim-string", 
        "rua": "verbatim-string", 
        "bairro": "verbatim-string", 
        "municipio": "verbatim-string", 
        "cidade": "verbatim-string", 
        "ponto_de_referencia": "verbatim-string", 
        "nome_do_solicitante": "verbatim-string", 
        "pessoa": "verbatim-string", 
        "numero": "verbatim-string", 
        "street_number": "verbatim-string", 
        "number": "verbatim-string", 
        "complemento": "verbatim-string", 
        "endereço_complemento": "verbatim-string",
    }

    api_caller = NuExtractCaller(model=model, endpoint=endpoint)

    non_redundant_entity_names = [name for name in entity_names 
            if name not in api_caller.entity_redundancies.keys()]

    entity_names = list(full_schema_dict['entities'].keys())

    clfs_scores = []
    entities_found = []
    gpu_usage_total = 0.0
    p_tokens_total = 0
    c_tokens_total = 0
    successfull_inferences = set()
    current_index = 0
    for transcript in tqdm(all_texts):
        try:
            new_entities, p_tokens, c_tokens, req_duration1 = api_caller(
                transcript=transcript,
                schema=nuextract_schema
            )
            
            p_tokens_total += p_tokens
            c_tokens_total += c_tokens
            gpu_usage_total += req_duration1
            
            scores_line = []
            for clf in clfnames:
                scores_line.append(new_entities[clf])
            entities_line = {}
            for name in entity_names:
                entities_line[name] = new_entities[name]
            
            clfs_scores.append(scores_line)
            entities_found.append(entities_line)
            print(entities_line)
            print(scores_line)
            successfull_inferences.add(current_index)

        except Exception as err:
            print(f"Critical Error: {err}")
            #raise(err)
            #continue
        current_index += 1

    clfs_true_no_err = np.asarray([clfs_true[i] for i in successfull_inferences])

    tokens_per_second_in = p_tokens_total / gpu_usage_total
    tokens_per_second_out = c_tokens_total / gpu_usage_total

    fmax_per_col, recalls_at_good_precisions, recalls, precisions, best_thresholds = find_fmax_per_col_parallel(
        np.array(clfs_scores),
        clfs_true_no_err,
        clfnames,
        n_jobs=4,
    )

    similarities_per_entity = {}
    for entity_name in non_redundant_entity_names:
        pred_values = [entities_found[i][entity_name] if entity_name in entities_found[i] else []
            for i in range(len(entities_found))]
        true_values = [entities_true[i][entity_name] for i in range(len(entities_true))]

        fmax, jw_sim_max, recall, precision = find_max_jw_sim(pred_values, 
            true_values, field_name=entity_name)
        similarities_per_entity[entity_name] = jw_sim_max
        #fmax_per_col[entity_name] = fmax
        recalls[entity_name] = recall
        precisions[entity_name] = precision

    result_dicts = {
        "fmax_per_col": fmax_per_col,
        "similarities_per_entity": similarities_per_entity,
        "recalls_at_good_precisions": recalls_at_good_precisions,
        "recalls": recalls,
        "precisions": precisions,
        "best_thresholds": best_thresholds,
        "meta": {
            "tokens_per_second_in": tokens_per_second_in,
            "tokens_per_second_out": tokens_per_second_out,
            "gpu_seconds": gpu_usage_total,
            "samples": len(successfull_inferences)
        }
    }

    return result_dicts
    
if __name__ == "__main__":
    results = []

    results_path = sys.argv[1]
    #"numind/NuExtract-2.0-4B"
    nuextract_model = sys.argv[2]

    prev_res = None
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            prev_res = json.load(f)
    
    if prev_res is None:
        prev_res = []

    for prev_calc in prev_res:
        results.append(prev_calc)

    models_calculated_quality = [r["model"] for r in prev_res]
    '''if nuextract_model in models_calculated_quality:
        quit(0)'''

    result_dicts = test_nuextract_model(nuextract_model, 'http://localhost:8000/v1')

    for col_name, fmax in result_dicts["fmax_per_col"].items():
        print(f"\tFmax para {col_name}: {fmax}")

    for col_name, sim in result_dicts["similarities_per_entity"].items():
        print(f"\tJW Sim para {col_name}: {sim}")

    mean_fmax = np.mean(list(result_dicts["fmax_per_col"].values()))
    mean_jw_sim = np.mean(list(result_dicts["similarities_per_entity"].values()))
    mean_recall = np.mean(list(result_dicts["recalls"].values()))
    mean_precision = np.mean(list(result_dicts["precisions"].values()))
    
    print(f"\tMedia Fmax: {mean_fmax}")
    print(f"\tMedia JW Sim: {mean_jw_sim}")
    print(f"\tMedia Recall: {mean_recall}")
    print(f"\tMedia Precision: {mean_precision}")
    results.append({"model": nuextract_model, 
                    "mean_metrics":{
                        "fmax": mean_fmax,
                        "jw_sim": mean_jw_sim,
                        "recall": mean_recall,
                        "precision": mean_precision
                    },
                    "meta": result_dicts["meta"],
                    "fmax": result_dicts["fmax_per_col"], 
                    "jw_sim": result_dicts["similarities_per_entity"],
                    "recall": result_dicts["recalls"],
                    "precision": result_dicts["precisions"],
                    "best_thresholds": result_dicts["best_thresholds"]})
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
        