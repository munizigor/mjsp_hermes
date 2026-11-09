import sys
import os
from typing import List
import time
import json

from model_runners.templates import RemoteInferenceAPIRunner
from pipelines.pipeline_naturezas import PipelineNaturezasAPI, PipelineNaturezasAPIb25

import polars as pl
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(script_dir))

DATASETS_DIR = f"{PROJECT_DIR}/datasets"
OUTPUT_DIR = f"{PROJECT_DIR}/results"

naturezas_dump_path = f"{DATASETS_DIR}/naturezas_cache_vllm.json"

print(f"PROJECT_DIR: {PROJECT_DIR}")
print(f"DATASETS_DIR: {DATASETS_DIR}")
print(f"OUTPUT_DIR: {OUTPUT_DIR}")

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def download_eval_dataset(total_rows=36):
    expected_path = f"{DATASETS_DIR}/df_emergencies_eval_naturezas-n={total_rows}.parquet"
    expected_csv = f"{DATASETS_DIR}/df_emergencies_eval_naturezas-n={total_rows}.csv"
    if os.path.exists(expected_path) and os.path.exists(expected_csv):
        print(f"NER eval dataset already exists at {expected_path}")
        return pl.read_parquet(expected_path)

    COLUMNS = [
        "ID",
        "roteiro",
        "nome_solicitante",
        "participacoes",
        "natureza_inicial",
    ]

    # Lazy scan across all Parquet shards on the Hub
    lf = pl.scan_parquet("hf://datasets/pitagoras-alves/fake-emergencies-br/data/train-*.parquet")

    #AttributeError: 'LazyFrame' object has no attribute 'sample'
    df_eval = lf.select(COLUMNS).unique(subset=["roteiro"], keep="first", maintain_order=True).collect()
    df_eval = df_eval.sample(n=total_rows, seed=1337)

    # Save locally if needed
    df_eval.write_parquet(expected_path)
    print(df_eval.schema)
    #convert roteiro_segmentado to str
    participacoes_raw_lists = df_eval["participacoes"].to_list()
    participacoes_strs = [str(part) for part in participacoes_raw_lists]
    df_eval2 = df_eval.with_columns(pl.Series("participacoes", participacoes_strs))
    print(df_eval2.schema)
    df_eval2.write_csv(expected_path.replace('.parquet', '.csv'))
    return df_eval

def pipeline_runner(pipeline: RemoteInferenceAPIRunner, roteiro: str, max_naturezas: int, tqdm_bar=None):
    proc_start = time.time()
    res = pipeline.create_interpretation_a([roteiro])
    interpretation_a, meta_dict_a = res[0]
    desc = interpretation_a["descricao_breve"] if interpretation_a is not None else None
    summarization_len = time.time() - proc_start
    meta_dicts_b = {}
    if desc is None:
        outras_observacoes = None
        similares_vec = []
        interpretation_b = None
        envolvimento = []
        filtering_total = None
        decision_total = None
        env_total = None
    else:
        outras_observacoes = interpretation_a["outras_observacoes"]
        filter_start = time.time()
        similares = pipeline.find_similar_natures([desc], max_n=max_naturezas)
        filter_end = time.time()
        filtering_total = filter_end - filter_start
        #print(f"Filtering total: {filtering_total}")
        '''for desc, similar in similares.items():
            print(f"Description {desc}:")
            for s in similar["sims_dict"]:
                print(f"\t{s}")'''
        success = similares != None
        if success:
            success = desc in list(similares.keys())
        
        if not success:
            similares_vec = []
            meta_vec_a2 = {}
            interpretation_b = None
            envolvimento = []
            decision_total = None
            env_total = None
        else:
            similares_vec = similares[desc]["sims_dict"]
            meta_vec_a2 = similares[desc]["meta"]

            decision_start = time.time()
            interpretations_b = pipeline.create_interpretation_b([roteiro], [similares_vec])
            decision_end = time.time()
            decision_total = decision_end - decision_start
            #print(f"Decision total: {decision_total}")
            success_dec = interpretations_b is not None
            if success_dec:
                success_dec = len(interpretations_b) == 1
                if success_dec:
                    success_dec = interpretations_b[0][0] is not None

            if not success_dec:
                interpretation_b = None
                meta_dicts_b = {}
            else:
                interpretation_b, meta_dicts_b = interpretations_b[0]
            
            env_start = time.time()
            interpretations_env = pipeline.create_envolvimentos([roteiro])
            env_total = time.time() - env_start
            interp_env, meta = interpretations_env[0]
            success_env = type(interp_env) == dict
            envolvimento = interp_env["envolvimentos_emergencia"] if success_env else []
            
    proc_end = time.time()
    proc_total = proc_end - proc_start
    if tqdm_bar:
        tqdm_bar.update(1)

    newline = {
        "descricao_breve": desc,
        "outras_observacoes": outras_observacoes,
        "classificacoes_provaveis": similares_vec,
        "classificacao_decisiva": interpretation_b,
        "envolvimentos": envolvimento,
        "processing_seconds": proc_total,
        "summarization_seconds": summarization_len,
        "filtering_seconds": filtering_total,
        "decision_seconds": decision_total,
        "env_seconds": env_total,
        "decision_metadata": meta_dicts_b
    }
    #print(json.dumps(newline, ensure_ascii=False, indent=4))

    return newline


def test_b25_integration(pipeline: RemoteInferenceAPIRunner, roteiro: str, max_naturezas: int, 
        b25_weights: List[float], tqdm_bar=None):
    proc_start = time.time()
    res = pipeline.create_interpretation_a([roteiro])
    interpretation_a, meta_dict_a = res[0]
    desc = interpretation_a["descricao_breve"] if interpretation_a is not None else None
    summarization_len = time.time() - proc_start

    newlines = []
    
    if desc is None:
        outras_observacoes = None
        similares_vec = []
        interpretation_b = None
        envolvimento = []
        filtering_total = None
        decision_total = None
        env_total = None
    else:
        outras_observacoes = interpretation_a["outras_observacoes"]
        for b25_w in b25_weights:
            proc_start = time.time()
            meta_dicts_b = {}
            newline = {
                "descricao_breve": desc,
                "outras_observacoes": outras_observacoes,
                "classificacoes_provaveis": [],
                "classificacao_decisiva": None,
                "envolvimentos": {},
                "processing_seconds": None,
                "summarization_seconds": summarization_len,
                "filtering_seconds": None,
                "decision_seconds": None,
                "env_seconds": None,
                "decision_metadata": meta_dicts_b,
                "filtering_metadata": {},
                "b25_w": b25_w
            }
            filter_start = time.time()
            similares = pipeline.find_similar_natures([desc], [roteiro], max_n=max_naturezas, bm25_weight=b25_w)
            filter_end = time.time()
            filtering_total = filter_end - filter_start
            #print(f"Filtering total: {filtering_total}")
            '''for desc, similar in similares.items():
                print(f"Description {desc}:")
                for s in similar["sims_dict"]:
                    print(f"\t{s}")'''
            newline["filtering_seconds"] = filtering_total
            success = similares != None
            if success:
                success = desc in list(similares.keys())
            if success:
                similares_vec = similares[desc]["sims_dict"]
                meta_vec_a2 = similares[desc]["meta"]
                newline["classificacoes_provaveis"] = similares_vec
                newline["filtering_metadata"] = meta_vec_a2

                decision_start = time.time()
                interpretations_b = pipeline.create_interpretation_b([roteiro], [similares_vec])
                decision_end = time.time()
                decision_total = decision_end - decision_start
                newline["decision_seconds"] = decision_total
                success_dec = interpretations_b is not None
                if success_dec:
                    success_dec = len(interpretations_b) == 1
                    if success_dec:
                        success_dec = interpretations_b[0][0] is not None

                if success_dec:
                    interpretation_b, meta_dicts_b = interpretations_b[0]
                    newline["classificacao_decisiva"] = interpretation_b
                    newline["decision_metadata"] = meta_dicts_b
                
            proc_end = time.time()
            proc_total = proc_end - proc_start + summarization_len
            newline["processing_seconds"] = proc_total
            newlines.append(newline)

    if tqdm_bar:
        tqdm_bar.update(1)
    
    return newlines

all_b25_w = [0.0, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]

tests = {
    "embedding_test": {
        "emb_models": [
            "PORTULAN/serafim-335m-portuguese-pt-sentence-encoder-ir",
            "google/embeddinggemma-300m",
            "PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "rufimelo/Legal-BERTimbau-sts-base-ma-v2",
            "iara-project/ModBERTBr-matryoshka-sts-pt",
            "ibm-granite/granite-embedding-311m-multilingual-r2"
        ],
        "pipelines": [
            {
                "interp_config": {
                    "hardware_config": "vllm-api",
                    "embedding_hardware": "cpu",
                    "model": "cyankiwi/Ministral-3-3B-Instruct-2512-AWQ-4bit",
                    "n_cpus": 6,
                    "min_transcricao": 120,
                    "n_in_batch": 6,
                    "max_similar_natures": 16
                },
                "Nome": "PipelineNaturezasAPI-v1",
                "Classe": PipelineNaturezasAPI
            }
        ]
    },
    "max_similar_natures_test": {
        "emb_models": [
            "PORTULAN/serafim-335m-portuguese-pt-sentence-encoder-ir",
            "PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ],
        "pipelines": [
            {
                "interp_config": {
                    "hardware_config": "vllm-api",
                    "embedding_hardware": "cpu",
                    "model": "cyankiwi/Ministral-3-3B-Instruct-2512-AWQ-4bit",
                    "n_cpus": 6,
                    "min_transcricao": 120,
                    "n_in_batch": 6,
                    "max_similar_natures": 8
                },
                "Nome": "PipelineNaturezasAPI-v1-top8",
                "Classe": PipelineNaturezasAPI
            },
            {
                "interp_config": {
                    "hardware_config": "vllm-api",
                    "embedding_hardware": "cpu",
                    "model": "cyankiwi/Ministral-3-3B-Instruct-2512-AWQ-4bit",
                    "n_cpus": 6,
                    "min_transcricao": 120,
                    "n_in_batch": 6,
                    "max_similar_natures": 16
                },
                "Nome": "PipelineNaturezasAPI-v1-top16",
                "Classe": PipelineNaturezasAPI
            },
            {
                "interp_config": {
                    "hardware_config": "vllm-api",
                    "embedding_hardware": "cpu",
                    "model": "cyankiwi/Ministral-3-3B-Instruct-2512-AWQ-4bit",
                    "n_cpus": 6,
                    "min_transcricao": 120,
                    "n_in_batch": 6,
                    "max_similar_natures": 24
                },
                "Nome": "PipelineNaturezasAPI-v1-top24",
                "Classe": PipelineNaturezasAPI
            },
            {
                "interp_config": {
                    "hardware_config": "vllm-api",
                    "embedding_hardware": "cpu",
                    "model": "cyankiwi/Ministral-3-3B-Instruct-2512-AWQ-4bit",
                    "n_cpus": 6,
                    "min_transcricao": 120,
                    "n_in_batch": 6,
                    "max_similar_natures": 32
                },
                "Nome": "PipelineNaturezasAPI-v1-top32",
                "Classe": PipelineNaturezasAPI
            },
            {
                "interp_config": {
                    "hardware_config": "vllm-api",
                    "embedding_hardware": "cpu",
                    "model": "cyankiwi/Ministral-3-3B-Instruct-2512-AWQ-4bit",
                    "n_cpus": 6,
                    "min_transcricao": 120,
                    "n_in_batch": 6,
                    "max_similar_natures": 64
                },
                "Nome": "PipelineNaturezasAPI-v1-top64",
                "Classe": PipelineNaturezasAPI
            },
            {
                "interp_config": {
                    "hardware_config": "vllm-api",
                    "embedding_hardware": "cpu",
                    "model": "cyankiwi/Ministral-3-3B-Instruct-2512-AWQ-4bit",
                    "n_cpus": 6,
                    "min_transcricao": 120,
                    "n_in_batch": 6,
                    "max_similar_natures": 96
                },
                "Nome": "PipelineNaturezasAPI-v1-top96",
                "Classe": PipelineNaturezasAPI
            },
            {
                "interp_config": {
                    "hardware_config": "vllm-api",
                    "embedding_hardware": "cpu",
                    "model": "cyankiwi/Ministral-3-3B-Instruct-2512-AWQ-4bit",
                    "n_cpus": 6,
                    "min_transcricao": 120,
                    "n_in_batch": 6,
                    "max_similar_natures": 128
                },
                "Nome": "PipelineNaturezasAPI-v1-top128",
                "Classe": PipelineNaturezasAPI
            }
        ]
    },
    "substr_models": {
        "emb_models": [
            "PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir",
        ],
        "pipelines": [
            {
                "interp_config": {
                    "hardware_config": "vllm-api",
                    "embedding_hardware": "cpu",
                    "model": "cyankiwi/Ministral-3-3B-Instruct-2512-AWQ-4bit",
                    "n_cpus": 6,
                    "min_transcricao": 120,
                    "n_in_batch": 6,
                    "max_similar_natures": 16
                },
                "b25_weights": all_b25_w,
                "Nome": "PipelineNaturezasAPIb25-v1-top16",
                "Classe": PipelineNaturezasAPIb25
            },
            {
                "interp_config": {
                    "hardware_config": "vllm-api",
                    "embedding_hardware": "cpu",
                    "model": "cyankiwi/Ministral-3-3B-Instruct-2512-AWQ-4bit",
                    "n_cpus": 6,
                    "min_transcricao": 120,
                    "n_in_batch": 6,
                    "max_similar_natures": 32
                },
                "b25_weights": all_b25_w,
                "Nome": "PipelineNaturezasAPIb25-v1-top32",
                "Classe": PipelineNaturezasAPIb25
            },
            {
                "interp_config": {
                    "hardware_config": "vllm-api",
                    "embedding_hardware": "cpu",
                    "model": "cyankiwi/Ministral-3-3B-Instruct-2512-AWQ-4bit",
                    "n_cpus": 6,
                    "min_transcricao": 120,
                    "n_in_batch": 6,
                    "max_similar_natures": 64
                },
                "b25_weights": all_b25_w,
                "Nome": "PipelineNaturezasAPIb25-v1-top64",
                "Classe": PipelineNaturezasAPIb25
            },
        ]
    }
}

if __name__ == "__main__":
    test_timestamp = f'-{time.strftime("%d-%m-%Y_%H:%M:%S", time.gmtime())}'.replace(":", "-")
    n_roteiros = 175
    test_name = sys.argv[1]
    assert test_name in tests.keys()
    if len(sys.argv) > 2:
        n_roteiros = int(sys.argv[2])
    if len(sys.argv) > 3:
        output_path = sys.argv[3]
    else:
        output_path = f"{OUTPUT_DIR}/eval_interp-n={n_roteiros}-{test_timestamp}.json"
    
    print(f"Output path: {output_path}")
    test_dict = tests[test_name]
    df = download_eval_dataset(total_rows=n_roteiros)
    emb_models = test_dict["emb_models"]
    pipelines = test_dict["pipelines"]

    roteiro_cols = {}
    for row in df.rows(named=True): 
        new_dict = {k: v for k, v in row.items()}
        roteiro_cols[row['ID']] = new_dict

    all_results = []
    for pipeline_dict in reversed(pipelines):
        pipeline_class = pipeline_dict["Classe"]
        pipeline_name = pipeline_dict["Nome"]
        interp_config = pipeline_dict["interp_config"]
        interpreter_type = interp_config['hardware_config']
        embedding_device = interp_config['embedding_hardware']
        max_similar_natures = interp_config['max_similar_natures']
        model_name = interp_config['model']
        print(f"Loading {pipeline_name} with {model_name}...")
        for emb_model in emb_models:
            interpreter = pipeline_class(
                model_name, client_type=interpreter_type,
                embedding_device=embedding_device, nat_dump_path=naturezas_dump_path, emb_model=emb_model
            )
            #test each one
            ids = df['ID'].to_list()
            roteiros = df['roteiro'].to_list()
            tqdm_bar = tqdm(total=len(roteiros))

            results_batch_s = []
            batch_n = 0
            for idx, r in zip(ids, roteiros):
                time.sleep(0.1)
                if test_name == "substr_models":
                    b25_weights = pipeline_dict["b25_weights"]
                    newlines = test_b25_integration(interpreter, r, interp_config['max_similar_natures'], b25_weights,
                        tqdm_bar=tqdm_bar)
                    if len(newlines) > 0:
                        print(f"Found {len(newlines)} infs in batch {batch_n}")
                        to_append = [(idx, r) for r in newlines]
                        results_batch_s.extend(to_append)
                    else:
                        print("A BATCH WAS NOT DONE")
                else:
                    results_batch = pipeline_runner(interpreter, r, interp_config['max_similar_natures'], tqdm_bar=tqdm_bar)
                    results_batch_s.append((idx, results_batch))
                batch_n += 1
            
            for r_idx, r_inf in results_batch_s:
                original_cols = roteiro_cols[r_idx]
                new_dict = {k: v for k, v in original_cols.items()}
                new_dict["inferences"] = r_inf
                new_dict["pipeline_name"] = pipeline_name
                new_dict["model"] = model_name
                new_dict["embedding_model"] = emb_model
                new_dict["max_similar_natures"] = max_similar_natures
                all_results.append(new_dict)
            tqdm_bar.close()

            json.dump(all_results, open(output_path, "w"), ensure_ascii=False, indent=4)
        

        
        
        