import os
import polars as pl
from polars import exceptions as pl_exc
import sys
import time
from glob import glob
import json
import re

from tqdm import tqdm
from unidecode import unidecode
from rapidfuzz import fuzz
import numpy as np
from xlsxwriter import Workbook
import jiwer

from plotting import make_dashboard





'''def run_data_analysis(proj_dir, test_path, machine_name=None):
    prices_dataset_path = f"{proj_dir}/datasets/pricing_models.json"
    prices_dict = json.load(open(prices_dataset_path, "r"))
    test_configs = json.load(open(f"{test_path}/config.json", "r"))

    if machine_name == None:
        machine_name = get_hostname()

    machine_stats = prices_dict["instances"][machine_name]
    print(machine_stats)

    df_paths = glob(test_path + "/*.parquet")
    dfs = {os.path.basename(p): pl.read_parquet(p) for p in df_paths}

    emergencias_df = parse_emergencias(dfs["emergencias_df.parquet"])
    inicio_emergencias = min(emergencias_df["start_time"].to_list())
    final_emergencias = max(emergencias_df["end_time"].to_list())
    duracao_teste = final_emergencias - inicio_emergencias
    final_stats = {}
    duracao_total = emergencias_df["duracao"].sum()
    duracao_media = emergencias_df["duracao"].mean()

    comprimento_total_audios = dfs["audios_df.parquet"]["audio_length_seconds"].sum()
    audio_lens = {}
    for row in dfs["audios_df.parquet"].rows(named=True):
        # print(row)
        audio_lens[row["id"]] = row["audio_length_seconds"]

    carga = duracao_total / duracao_teste

    final_stats["duracao_total"] = duracao_total
    final_stats["duracao_media"] = duracao_media
    final_stats["duracao_teste"] = duracao_teste
    final_stats["carga_de_emergencias"] = carga
    final_stats["comprimento_total_audios"] = comprimento_total_audios

    valid_emergencies = emergencias_df["id"].to_list()
    for key in dfs.keys():
        if "emergencias_df" not in key:
            dfs[key] = drop_invalid_emergencies(dfs[key], valid_emergencies)
    n_calls = len(valid_emergencies)

    transcricao_df = parse_transcricao(dfs["transcricao_df.parquet"], audio_lens)
    transcricao_df = transcricao_df.with_columns(
        speed_up=transcricao_df["duracao_audio"]
        / transcricao_df["transcription_seconds"]
    )
    transcricao_tempo_total = transcricao_df["transcription_seconds"].sum()
    horas_transcritas = transcricao_df["duracao_audio"].sum() / 60 / 60
    final_stats["segundos_transcritos"] = transcricao_df["duracao_audio"].sum()
    final_stats["horas_transcritas"] = horas_transcritas
    speed_up_total = duracao_total / transcricao_tempo_total

    duracoes_t = transcricao_df["transcription_seconds"].to_list()
    final_stats["tempo_inferencia_medio_transcricao"] = np.mean(duracoes_t)
    final_stats["tempo_inferencia_std_transcricao"] = np.std(duracoes_t)
    final_stats["tempo_transcricao_total"] = transcricao_tempo_total
    final_stats["velocidade_de_transcricao"] = speed_up_total

    if "container_stats.parquet" in dfs:
        container_stats = parse_container_stats_df(
            dfs["container_stats.parquet"], machine_stats["cpus"]
        )
        hardware_stats_df = hardware_usage(container_stats)
    else:
        container_stats = None
        hardware_stats_df = None
    inferencia_df = parse_inferencia(dfs["inferencia_df.parquet"])
    inferencia_stats = estatisticas_sobre_inferencia(inferencia_df, n_calls)

    # print('\n'.join([json.dumps(la) for la in gastos_por_inferencia]))
    print(final_stats)

    final_stats_df = [
        {"Métrica": m.replace("_", " "), "Valor": v} for m, v in final_stats.items()
    ]
    final_stats_df = pl.DataFrame(final_stats_df)

    costs_df = parse_costs(
        n_calls,
        hardware_stats_df,
        inferencia_stats,
        prices_dict,
        machine_name,
        test_configs,
        horas_transcritas,
        duracao_teste,
    )

    excel_path = f"{test_path}/results.xlsx"
    print(excel_path)
    if os.path.exists(excel_path):
        os.remove(excel_path)

    dfs["emergencias_df.parquet"] = emergencias_df
    dfs["transcricao_df.parquet"] = transcricao_df
    dfs["inferencia_df.parquet"] = inferencia_df
    dfs["inferencia_stats"] = inferencia_stats
    if container_stats is not None:
        dfs["container_stats.parquet"] = container_stats
        dfs["uso_de_hardware"] = hardware_stats_df
    dfs["final_stats"] = final_stats_df
    dfs["costs_df"] = costs_df

    parsed_dfs_path = f"{test_path}/parsed_dfs"
    if not os.path.exists(parsed_dfs_path):
        os.mkdir(parsed_dfs_path)
    for df_path, df in dfs.items():
        name1 = os.path.basename(df_path).replace(".parquet", "")
        parquet_path = f"{parsed_dfs_path}/{name1}.parquet"
        df.write_parquet(parquet_path)

    with Workbook(excel_path, {"nan_inf_to_errors": True}) as wb:
        for df_path, df in dfs.items():
            name1 = os.path.basename(df_path).replace(".parquet", "")
            name2 = name1.replace("_", " ").title()
            df.write_excel(
                workbook=wb,
                worksheet=name2,
                autofit=True,
            )

    return dfs





if __name__ == "__main__":
    test_path = sys.argv[1]
    machine_name = sys.argv[2]
    # test_path = "/home/pita/docs/hermes/hermes-agents/results/azure_vm/2025-09-29_4/"
    dfs = run_data_analysis(test_path, machine_name)
    comparison_df = value_comparisons(test_path)
    make_dashboard(test_path)
'''