import sys
import os
from glob import glob
import pandas as pd
import json
from tqdm import tqdm

from asr_lib.utils import (
    eval_transcriptions, normalize_special, normalize_punct, normalize_smart_light
)
from asr_lib.utils import normalize_smart_light as normalize_light
from asr_lib.utils import remove_asr_hallucination_loops_fast as normalize_repeats

results_required_columns = [
    "audio_len",
    "processing_time",
    "original",
    "transcribed",
    "model_id"
]

columns_fillable = {
    "weight": 1.0,
    "chunking_speedup": 1.0
}

def sort_paths(paths):
    #Sort first by last modification time and then by number at raw_results.{N}.csv
    mod_times = {}
    for path in paths:
        mod_times[path] = os.path.getmtime(path)
    
    sorted_paths = sorted(paths, key=lambda x: (int(x.split('.')[-2]), mod_times[x]))
    return sorted_paths

def load_all_raw_results(results_dir: str):
    results_files = glob(f"{results_dir}/raw_results.*.csv")
    results_files = sort_paths(results_files)

    model_dfs = {}
    for results_file in results_files:
        df = pd.read_csv(results_file)
        if all(col in df.columns for col in results_required_columns):
            print(f"Loading {results_file}")
            for col, default_val in columns_fillable.items():
                if col not in df.columns:
                    df[col] = default_val
            new_df = df[results_required_columns + list(columns_fillable.keys())]
            for model_id, model_rows in new_df.groupby("model_id"):
                model_dfs[model_id] = model_rows
                print(f"\tLoaded {model_id}")
        else:
            print(f"Skipping {results_file} due to missing columns")
    
    dfs = [v for k, v in model_dfs.items()]
    all_raw_results = pd.concat(dfs, ignore_index=True)
    return all_raw_results

def apply_normalizations(all_raw_results: pd.DataFrame):
    '''normalize_light, normalize_special, normalize_punct, normalize_repeats'''
    print("applying basic normalizations")
    all_raw_results["original_normalized"] = all_raw_results["original"].apply(normalize_light)
    all_raw_results["transcribed_normalized"] = all_raw_results["transcribed"].apply(normalize_light)
    
    print("applying special character removals")
    all_raw_results["original_normalized_special"] = all_raw_results["original_normalized"].apply(normalize_special)
    all_raw_results["transcribed_normalized_special"] = all_raw_results["transcribed_normalized"].apply(normalize_special)
    
    print("applying punctuation removals")
    all_raw_results["original_normalized_punct"] = all_raw_results["original_normalized"].apply(normalize_punct)
    all_raw_results["transcribed_normalized_punct"] = all_raw_results["transcribed_normalized"].apply(normalize_punct)
    
    print("applying repeat removals")
    all_raw_results["original_normalized_repeats"] = all_raw_results["original_normalized"].apply(normalize_repeats)
    all_raw_results["transcribed_normalized_repeats"] = all_raw_results["transcribed_normalized"].apply(normalize_repeats)

    return all_raw_results

def sample_normalizations(df, max_rows=66):
    '''
    sample an equal amount of rows for each model_id, to use for 
    manual inspection and AI assisted analysis of normalization functions
    '''

    model_ids = df["model_id"].unique()
    n_models = len(model_ids)
    n_per_model = max_rows // n_models
    sampled_dfs = []
    for model_id in tqdm(model_ids, desc="Sampling normalizations"):
        model_df = df[df["model_id"] == model_id]
        #calculate difference in str length between transcribed_normalized and transcribed_normalized_repeats
        repeats_len_diff = model_df["transcribed_normalized"].str.len() - model_df["transcribed_normalized_repeats"].str.len()
        #print(repeats_len_diff.describe())
        model_df["repeats_len_diff"] = repeats_len_diff
        #Sample n_per_model-1 and the row with highest repeats_len_diff
        print(model_df)
        print(f"\tSampling {n_per_model} rows for {model_id} with {model_df.shape[0]} available rows")
        to_sample = n_per_model-1
        if to_sample > model_df.shape[0]:
            to_sample = model_df.shape[0]
        sampled_df = model_df.sample(n=to_sample, replace=False)
        sampled_df = pd.concat([sampled_df, model_df.nlargest(1, "repeats_len_diff")], ignore_index=True)
        cols_to_keep = ["model_id", "original", "transcribed", "transcribed_normalized", "transcribed_normalized_special", 
                        "transcribed_normalized_punct", "transcribed_normalized_repeats"]
        sampled_df = sampled_df[cols_to_keep]
        sampled_dfs.append(sampled_df)
    return pd.concat(sampled_dfs, ignore_index=True)

def calc_wer_per_model(df):
    print("Calculating WER per model")
    col_pairs = [("raw", "original", "transcribed"), 
                 ("normalized", "original_normalized", "transcribed_normalized"), 
                 ("no_special", "original_normalized_special", "transcribed_normalized_special"), 
                 ("no_punct", "original_normalized_punct", "transcribed_normalized_punct"), 
                 ("no_repeats", "original_normalized_repeats", "transcribed_normalized_repeats")]
    wer_per_model = []
    n_models = len(df["model_id"].unique())
    n_evals = len(col_pairs)
    teste_items = n_models * n_evals
    bar = tqdm(total=teste_items)
    for model_id, model_rows in df.groupby("model_id"):
        #eval_transcriptions
        new_row = {"model_id": model_id}
        print(model_id)
        for norm_name, orig_col, trans_col in col_pairs:
            orig_list = model_rows[orig_col].tolist()
            transc_list = model_rows[trans_col].tolist()
            print("\t"+norm_name)
            new_row["wer_"+norm_name] = eval_transcriptions(orig_list, transc_list)
            bar.update(1)
        new_row["n_samples"] = len(model_rows)
        new_row["total_audio_seconds"] = model_rows["audio_len"].sum()
        new_row["total_seconds_pipeline"] = model_rows["processing_time"].sum()
        model_rows["processing_time_chunked"] = model_rows["processing_time"] / model_rows["chunking_speedup"]
        new_row["total_seconds_chunks"] = model_rows["processing_time_chunked"].sum()
        new_row["speed_pipeline"] = new_row["total_audio_seconds"] / new_row["total_seconds_pipeline"]
        new_row["speed_chunks"] = new_row["total_audio_seconds"] / new_row["total_seconds_chunks"]
        wer_per_model.append(new_row)
    bar.close()
    eval_df = pd.DataFrame(wer_per_model)
    #sort by score
    eval_df = eval_df.sort_values(by="wer_normalized", ascending=False)

    pricing_path = "pricing.json"
    instance_type = "cpu8_gput4_mem30gb"
    pricing = json.load(open(pricing_path))
    on_demand = [pricing[instance_type][cloud]["on_demand"] for cloud in pricing[instance_type]]
    spot = [pricing[instance_type][cloud]["spot"] for cloud in pricing[instance_type]]
    demand_h = min(on_demand)
    spot_h = min(spot)

    commercial_libs = pricing["commercial_apis"]

    def add_demand_cost(row):
        if row["model_id"] in commercial_libs:
            return commercial_libs[row["model_id"]]
        else:
            return (1/row["speed_chunks"]) * demand_h
    
    def add_spot_cost(row):
        if row["model_id"] in commercial_libs:
            return None
        else:
            return (1/row["speed_chunks"]) * spot_h
    
    def add_test_cost(row):
        cost_1h = row['cost_1h']
        if row["model_id"] in commercial_libs:
            return row['total_audio_seconds'] / 3600 * cost_1h
        else:
            processing_time = row["total_seconds_chunks"]/3600
            return processing_time * cost_1h

    #Calculate cost of transcribing 1h of audio for each model using processing_time_chunked
    eval_df["cost_1h"] = eval_df.apply(add_demand_cost, axis=1)
    eval_df["cost_1h_spot"] = eval_df.apply(add_spot_cost, axis=1)
    #Calculate cost of whole test set for each model using processing_time_chunked
    eval_df["cost_test"] = eval_df.apply(add_test_cost, axis=1)
    
    '''for col in ["no_repeats", "no_repeats_fuzzy", "no_repeats_asr_v2", "no_repeats_asr", "no_repeats_asr_fast"]:
        #Calc improvement over normalized
        eval_df[f"wer_{col}"] = eval_df["wer_normalized"] - eval_df[f"wer_{col}"]'''

    return eval_df

if __name__ == "__main__":
    results_dir = "results" if len(sys.argv) < 2 else sys.argv[1]
    
    print("Saving raw results")
    all_norm_path = f"{results_dir}/all_raw_results.csv"
    if os.path.exists(all_norm_path) and False:
        print(f"File {all_norm_path} already exists, skipping")
        all_normalized = pd.read_csv(all_norm_path)
    else:
        all_raw_results = load_all_raw_results(results_dir)
        all_normalized = apply_normalizations(all_raw_results)
        all_normalized.to_csv(all_norm_path, index=False)

    sampled_df = sample_normalizations(all_normalized)
    print("Saving sampled results")
    sampled_df.to_csv(f"{results_dir}/sampled_results.csv", index=False)

    eval_df = calc_wer_per_model(all_normalized)
    print("Saving eval results")
    eval_df.to_csv(f"{results_dir}/eval_results.csv", index=False)
    
    