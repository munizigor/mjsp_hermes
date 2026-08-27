import os
import json
import sys
import time
import hashlib
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm
import pydantic
from enum import Enum
from google import genai
from google.genai import types

from testing import (
    get_testing_inputs,
    find_fmax_per_col_parallel,
    find_max_jw_sim,
)

# --- Environment Loading ---
if os.path.exists(".env"):
    try:
        env_vals = {
            rawline.split("=")[0]: rawline.split("=")[1].rstrip("\n")
            for rawline in open(".env", "r").read().split("\n")
            if "=" in rawline
        }
        for k, v in env_vals.items():
            if k not in os.environ:
                os.environ[k] = v
    except Exception as e:
        print(f"Warning: Failed to load .env: {e}", file=sys.stderr)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


class RespostaDeClassificacao(str, Enum):
    # CERTAMENTE_SIM = "Certamente Sim"
    SIM = "Sim"
    # PROVAVELMENTE_SIM = "Provavelmente Sim"
    # NAO_SEI = "Não sei"
    # PROVAVELMENTE_NAO = "Provavelmente Não"
    NAO = "Não"
    # CERTAMENTE_NAO = "Certamente Não"


class DetalhesGravidadeOcorrencia(pydantic.BaseModel):
    fato_ocorrendo_neste_momento: Optional[RespostaDeClassificacao] = None
    autor_do_fato_no_local: Optional[RespostaDeClassificacao] = None
    autor_do_fato_armado: Optional[RespostaDeClassificacao] = None
    feridos_com_risco_de_morte: Optional[RespostaDeClassificacao] = None
    risco_de_tumulto: Optional[RespostaDeClassificacao] = None
    lei_maria_da_penha: Optional[RespostaDeClassificacao] = None


class InformacoesOcorrencia(pydantic.BaseModel):
    detalhes_gravidade_ocorrencia: Optional[DetalhesGravidadeOcorrencia] = None
    rua_ou_logradouro: List[str] = pydantic.Field(default_factory=list)
    street_number: List[str] = pydantic.Field(default_factory=list)
    complemento: List[str] = pydantic.Field(default_factory=list)
    bairro: List[str] = pydantic.Field(default_factory=list)
    cidade: List[str] = pydantic.Field(default_factory=list)
    ponto_de_referencia: List[str] = pydantic.Field(default_factory=list)
    nome_do_solicitante: List[str] = pydantic.Field(default_factory=list)
    pessoa: List[str] = pydantic.Field(default_factory=list)


class GeminiExtract:
    def __init__(self, model: str, offline_mode: bool = False):
        self.model = model

        if offline_mode:
            self.client = None
        else:
            if not GEMINI_API_KEY:
                raise ValueError(
                    "GEMINI_API_KEY not found. Please set it in .env or environment variables."
                )

            self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.system_prompt = "Você é um assistente que sempre responde estritamente no formato JSON especificado. Extraia informações da transcrição da chamada de emergência."

        self.answer_to_value = {
            "Certamente Sim": 1.0,
            "Sim": 0.95,
            "Provavelmente Sim": 0.65,
            "Não sei": 0.5,
            "Provavelmente Não": 0.35,
            "Não": 0.05,
            "Certamente Não": 0.0,
        }

        self.question_mapping = {
            "fato_ocorrendo_neste_momento": "Fato Ocorrendo Neste Momento ?",
            "autor_do_fato_no_local": "Autor Do Fato No Local ?",
            "autor_do_fato_armado": "Autor Do Fato Armado ?",
            "feridos_com_risco_de_morte": "Feridos Com Risco de Morte ?",
            "risco_de_tumulto": "Risco De Tumulto ?",
            "lei_maria_da_penha": "Lei Maria da Penha ?",
        }

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

    def convert_to_json_schema(self, pydantic_obj: InformacoesOcorrencia) -> dict:
        output = {}

        # Helper to get value from enum safely
        def get_clf_score(val: Optional[RespostaDeClassificacao]) -> float:
            if val is None:
                return 0.0
            return self.answer_to_value.get(val.value, 0.0)

        # Process Classifications (Questions)
        details = pydantic_obj.detalhes_gravidade_ocorrencia
        if details:
            for attr_name, question_text in self.question_mapping.items():
                val = getattr(details, attr_name, None)
                output[question_text] = get_clf_score(val)
        else:
            # If details object is missing, set all to 0.0
            for question_text in self.question_mapping.values():
                output[question_text] = 0.0

        # Process Entities (Lists)
        entity_field_mapping = {
            "rua_ou_logradouro": "rua_ou_logradouro",
            "street_number": "numero",
            "complemento": "complemento",
            "bairro": "bairro",
            "cidade": "cidade",
            "ponto_de_referencia": "ponto_de_referencia",
            "nome_do_solicitante": "nome_do_solicitante",
            "pessoa": "pessoa",
        }

        for field_name, json_key in entity_field_mapping.items():
            val_list = getattr(pydantic_obj, field_name, [])
            if val_list is None:
                val_list = []

            # Format as list of tuples (value, confidence)
            formatted_list = [(str(item), 1.0) for item in val_list]
            output[json_key] = formatted_list

        return output

    def extract_with_retry(self, transcript: str):
        # Check cache first
        cached = self.get_cached_response(transcript)
        if cached:
            return (
                cached["entities"],
                cached.get("input_tokens", 0),
                cached.get("output_tokens", 0),
                cached.get("latency", 0.0),
            )

        if self.client is None:
            raise ValueError(
                "Client not initialized. Please set offline_mode=False to perform requests to the LLM."
            )

        start_time = time.time()
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=transcript,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=InformacoesOcorrencia,
                    system_instruction=self.system_prompt,
                    temperature=0.1,
                ),
            )

            if not response.text:
                raise ValueError("Empty response text")

            parsed_obj = InformacoesOcorrencia.model_validate_json(response.text)
            final_json = self.convert_to_json_schema(parsed_obj)

            end_time = time.time()
            latency = end_time - start_time

            # Usage metadata
            input_tokens = 0
            output_tokens = 0
            if response.usage_metadata:
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count

            # Create result object with metadata
            result_data = {
                "entities": final_json,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency": latency,
            }

            # Save to cache
            self.save_cached_response(transcript, result_data)

            return final_json, input_tokens, output_tokens, latency

        except Exception as e:
            # Propagate error to let caller handle it (or not)
            raise e


def test_gemini_model(model_name: str) -> dict:
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
        extractor = GeminiExtract(model=model_name, offline_mode=True)
    except Exception as e:
        print(f"Failed to initialize extractor for {model_name}: {e}")
        return {}

    # Lists to store results
    clfs_scores = []
    entities_found = []
    gpu_usage_total = 0.0
    p_tokens_total = 0
    c_tokens_total = 0
    successful_inferences_indices = []

    results_map = {}  # index -> result

    # Using 20 threads as requested
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_idx = {
            executor.submit(extractor.extract_with_retry, transcript): idx
            for idx, transcript in enumerate(all_texts)
        }

        for future in tqdm(
            as_completed(future_to_idx),
            total=len(all_texts),
            desc=f"Testing {model_name}",
        ):
            idx = future_to_idx[future]
            try:
                final_json, p_tokens, c_tokens, latency = future.result()
                results_map[idx] = (final_json, p_tokens, c_tokens, latency)
            except Exception as e:
                print(f"Sample {idx} failed: {e}")
                pass

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
    verbosity_ratios = {}

    for entity_name in non_redundant_entity_names:
        pred_values = [
            entities_found[i].get(entity_name, []) for i in range(len(entities_found))
        ]

        true_values = [
            entities_true[i].get(entity_name, []) for i in successful_inferences_indices
        ]

        # Guard against empty lists if needed, but find_max_jw_sim should handle it
        fmax, jw_sim_max, recall, precision, vr = find_max_jw_sim(
            pred_values, true_values, field_name=entity_name
        )
        similarities_per_entity[entity_name] = jw_sim_max
        recalls[entity_name] = recall
        precisions[entity_name] = precision
        verbosity_ratios[entity_name] = vr

        _, jw_sim_max2, _, _, _ = find_max_jw_sim(
            pred_values, true_values, field_name=entity_name, use_simple=True
        )
        similarities_per_entity_simple[entity_name] = jw_sim_max2

    return {
        "fmax_per_col": fmax_per_col,
        "similarities_per_entity": similarities_per_entity,
        "similarities_per_entity_simple": similarities_per_entity_simple,
        "recalls_at_good_precisions": recalls_at_good_precisions,
        "recalls": recalls,
        "precisions": precisions,
        "verbosity_ratios": verbosity_ratios,
        "best_thresholds": best_thresholds,
        "meta": {
            "tokens_per_second_in": tokens_per_second_in,
            "tokens_per_second_out": tokens_per_second_out,
            "gpu_seconds": gpu_usage_total,
            "samples": len(successful_inferences_indices),
            "tokens_total_in": p_tokens_total,
            "tokens_total_out": c_tokens_total,
        },
    }


if __name__ == "__main__":
    results_path = "results/gemini_results3.json"

    # Models ordered from best quality to lowest quality
    models_to_test = [
        "gemini-3-flash-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        # "gemini-1.5-flash",
    ]

    models_to_test = list(reversed(models_to_test))

    results = []
    """if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                results = json.load(f)
        except json.JSONDecodeError:
            results = []

    existing_models = [r.get("model") for r in results]"""

    for model_name in models_to_test:
        """if model_name in existing_models:
        print(f"Model {model_name} already in results. Skipping.")
        continue"""

        print(f"=== Testing {model_name} ===")

        try:
            metrics = test_gemini_model(model_name)
            if not metrics:
                print(f"Skipping result save for {model_name} due to lack of metrics.")
                continue

            mean_fmax = np.mean(list(metrics["fmax_per_col"].values()))
            mean_jw_sim = np.mean(list(metrics["similarities_per_entity"].values()))
            mean_jw_sim_simple = np.mean(
                list(metrics["similarities_per_entity_simple"].values())
            )
            mean_recall = np.mean(list(metrics["recalls"].values()))
            mean_precision = np.mean(list(metrics["precisions"].values()))
            mean_verbosity = np.mean(list(metrics["verbosity_ratios"].values()))

            print(f"\tMean Fmax: {mean_fmax}")
            print(f"\tMean JW Sim: {mean_jw_sim}")
            print(f"\tMean JW Sim Simple: {mean_jw_sim_simple}")
            print(f"\tMean Recall: {mean_recall}")
            print(f"\tMean Precision: {mean_precision}")
            print(f"\tMean Verborrity: {mean_verbosity}")

            result_entry = {
                "model": model_name,
                "mean_metrics": {
                    "fmax": mean_fmax,
                    "jw_sim": mean_jw_sim,
                    "jw_sim_simple": mean_jw_sim_simple,
                    "recall": mean_recall,
                    "precision": mean_precision,
                    "verbosity": mean_verbosity,
                },
                "meta": metrics["meta"],
                "fmax": metrics["fmax_per_col"],
                "jw_sim": metrics["similarities_per_entity"],
                "jw_sim_simple": metrics["similarities_per_entity_simple"],
                "recall": metrics["recalls"],
                "precision": metrics["precisions"],
                "verbosity_ratios": metrics["verbosity_ratios"],
                "best_thresholds": metrics["best_thresholds"],
            }

            results.append(result_entry)

            # Ensure dir exists
            os.makedirs(os.path.dirname(results_path), exist_ok=True)

            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)

        except Exception as e:
            print(f"Failed to test {model_name}: {e}")
