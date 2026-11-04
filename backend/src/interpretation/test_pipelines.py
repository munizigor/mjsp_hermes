from torch.distributed.tensor._ops import NewDim
import sys
import os
from typing import List
import time
import json

from model_runners.templates import RemoteInferenceAPIRunner
from pipelines.pipeline_naturezas import PipelineNaturezasAPI

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

def pipeline_runner(pipeline: RemoteInferenceAPIRunner, roteiros: List[str], max_naturezas: int, tqdm_bar=None):
    interpretations_a = pipeline.create_interpretation_a(roteiros)
    meta_dicts_a = [m for _, m in interpretations_a]
    interpretations_a = [x for x, m in interpretations_a]
    descs = [r["descricao_breve"] for r in interpretations_a]
    #for idx, desc in enumerate(descs):
    #    print(f"Description {idx}: {desc}")
    if tqdm_bar:
        tqdm_bar.update(len(roteiros))

    similares = pipeline.find_similar_natures(descs, max_n=max_naturezas)
    for desc, similar in similares.items():
        print(f"Description {desc}:")
        for s in similar["sims_dict"]:
            print(f"\t{s}")
    similares_vecs = [similares[desc]["sims_dict"] for desc in descs]
    meta_vecs_a2 = [similares[desc]["meta"] for desc in descs]
    if tqdm_bar:
        tqdm_bar.update(len(roteiros))

    interpretations_b = pipeline.create_interpretation_b(roteiros, similares_vecs)
    meta_dicts_b = [m for _, m in interpretations_b]
    interpretations_b = [x for x, m in interpretations_b]
    for idx, simil_vec in enumerate(similares_vecs):
        nat_idx = meta_dicts_b[idx]["meta"]["nat_id"]
        def_nat = interpretations_b[idx]
        print(f"idx={idx}")
        for s in simil_vec:
            print(f"\t{s}")
        print(f"\tnat_idx: {nat_idx}")
        print(f"\tDefinitive nat: {def_nat}")
    if tqdm_bar:
        tqdm_bar.update(len(roteiros))

    interpretations_env = pipeline.create_envolvimentos(roteiros)
    meta_dicts_env = [m for _, m in interpretations_env]
    interpretations_env = [x for x, m in interpretations_env]
    envolvimentos = [r["envolvimentos_emergencia"] for r in interpretations_env]
    if tqdm_bar:
        tqdm_bar.update(len(roteiros))

    results = []
    for idx, roteiro in enumerate(roteiros):
        interpretation_a = interpretations_a[idx]
        desc = descs[idx]
        similares_vec = similares_vecs[idx]
        interpretation_b = interpretations_b[idx]
        envolvimento = envolvimentos[idx]

        newline = {
            "descricao_breve": desc,
            "outras_observacoes": interpretation_a["outras_observacoes"],
            "ponto_de_referencia": interpretation_a["ponto_de_referencia"],
            "nome_do_solicitante": interpretation_a["nome_do_solicitante"],
            "classificacoes_provaveis": similares_vec,
            "classificacao_decisiva": interpretation_b,
            "envolvimentos": envolvimento,
        }
        results.append(newline)
    quit(1)
    
    return results



if __name__ == "__main__":
    test_timestamp = f'-{time.strftime("%d-%m-%Y_%H:%M:%S", time.gmtime())}'.replace(":", "-")
    n_roteiros = 8
    if len(sys.argv) > 1:
        n_roteiros = int(sys.argv[1])
    llm_name = "cyankiwi/Ministral-3-3B-Instruct-2512-AWQ-4bit"
    if len(sys.argv) > 2:
        llm_name = sys.argv[2]
    
    if len(sys.argv) > 3:
        output_path = sys.argv[3]
    else:
        llm_name_safe = llm_name.replace("/", "--")
        output_path = f"{OUTPUT_DIR}/eval_interp-n={n_roteiros}-{llm_name_safe}{test_timestamp}.json"

    interp_config_localvllm = {
        "hardware_config": "vllm-api",
        "embedding_hardware": "cpu",
        "model": llm_name,
        "n_cpus": 4,
        "min_transcricao": 120,
        "n_in_batch": 6,
        "max_similar_natures": 16
    }
    df = download_eval_dataset(total_rows=n_roteiros)

    pipelines = [
        (PipelineNaturezasAPI, "PipelineNaturezasAPI-v1", interp_config_localvllm),
    ]

    all_results = []
    max_batch = 4
    for pipeline_class, pipeline_name, interp_config in pipelines:
        interpreter_type = interp_config['hardware_config']
        embedding_device = interp_config['embedding_hardware']
        model_name = interp_config['model']
        print(f"Loading {pipeline_name} with {model_name}...")
        interpreter = pipeline_class(
            model_name, client_type=interpreter_type,
            embedding_device=embedding_device, nat_dump_path=naturezas_dump_path
        )
        #test each one
        roteiros = df['roteiro'].to_list()

        tqdm_bar = tqdm(total=len(roteiros)*4)

        #batchs = [roteiros]
        batchs = list(chunks(roteiros, max_batch))

        results_batch_s = []
        for batch in batchs:
            results_batch = pipeline_runner(interpreter, batch, interp_config['max_similar_natures'], tqdm_bar=tqdm_bar)
            results_batch_s.extend(results_batch)

        idx = 0
        for row in df.rows(named=True):
            new_dict = {k: v for k, v in row.items()}
            new_dict["inferences"] = results_batch_s[idx]
            new_dict["pipeline_name"] = pipeline_name
            new_dict["model"] = model_name
            all_results.append(new_dict)
            idx += 1
            print(json.dumps(new_dict, ensure_ascii=False, indent=4))
        tqdm_bar.close()

        json.dump(all_results, open(output_path, "w"), ensure_ascii=False, indent=4)
        

        
        
        