# %%
import os
from typing import final
import polars as pl
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
import numpy as np

from test_utils.data_processing import save_to_parquet
from test_utils.util import get_hostname
from test_utils.parsers import (
    parse_container_stats_df,
    parse_emergencias,
    parse_inferencia,
    parse_transcricao,
    estatisticas_sobre_inferencia,
    hardware_usage,
    drop_invalid_emergencies,
    parse_inference_server_readings,
)
from test_utils.info_costs import parse_costs, parse_prices_file
from test_utils.plotting import dashboard_stats, plot_delays, create_final_dashboard

# Get current python scripts last modified time
python_scripts = glob("*.py") + glob("test_utils/*.py")
mod_timestamps = [os.path.getmtime(script) for script in python_scripts]
last_mod_time = max(mod_timestamps)


def calcular_horas_de_ligacao_por_mes(duracao_teste_horas, duracao_ligacoes_horas):
    # Dado o número de horas que foram processadas em um espaço curto de tempo, projeta
    # quantas horas de ligação poderiam ser processadas em um mês, assumindo 730h/mês
    horas_totais_no_mes = 730 * (duracao_ligacoes_horas / duracao_teste_horas)
    return round(horas_totais_no_mes, 1)


def custo_mensal(
    custos_apis_total,
    duracao_ligacoes_horas,
    duracao_teste_horas,
    vms_hora_total,
    n_ligacoes,
):
    custo_api_por_hora = custos_apis_total / duracao_ligacoes_horas
    custo_vms_mensal = vms_hora_total * 730
    horas_totais_no_mes = calcular_horas_de_ligacao_por_mes(
        duracao_teste_horas, duracao_ligacoes_horas
    )
    custo_api_mensal = custo_api_por_hora * horas_totais_no_mes
    custo_total = custo_api_mensal + custo_vms_mensal

    # se em duracao_teste_horas foram atendidas n_ligacoes, quantas seriam atendidas em 730 horas?
    ligacoes_mes = (n_ligacoes / duracao_teste_horas) * 730
    custo_por_ligacao = custo_total / ligacoes_mes
    return custo_total, custo_por_ligacao, custo_vms_mensal


def list_tests(test_base_path, ignore_old=False):
    test_dirs = []
    test_paths = glob(f"{test_base_path}/*-n_*-cl_*-*")
    test_paths = [p for p in test_paths if os.path.exists(p + "/sqlite.db")]
    test_paths.sort(key=lambda x: os.path.getmtime(x + "/sqlite.db"), reverse=True)
    for test_path in test_paths:
        important_inputs = ["load_parameters.json", "config.json", "sqlite.db"]
        important_outputs = ["dashboard.html", "results.xlsx"]
        if all(os.path.exists(f"{test_path}/{file}") for file in important_inputs):
            if all(os.path.exists(f"{test_path}/{file}") for file in important_outputs):
                if (
                    os.path.getmtime(f"{test_path}/dashboard.html") > last_mod_time
                    and ignore_old
                ):
                    print(
                        f"Skipping {test_path} because it is older than the python scripts"
                    )
                else:
                    print("Has all outputs, but is old:", test_path)
                    test_dirs.append(test_path)
            else:
                test_dirs.append(test_path)
        else:
            print("Has missing inputs:", test_path)
    print("testing:")
    for p in test_dirs:
        print(p)
    return test_dirs


proj_dir = "../.."
network_names = {
    "0.0.0.0": "hermes-backend",
    "10.0.0.5": "hermes-asr",
    "10.0.0.4": "hermes-frontend",
    "10.0.0.10": "hermes-interpretation",
    "10.0.0.8": "hermes-ner",
    "10.221.112.6": "hermes-asr",
    "10.221.112.10": "hermes-ner",
    "10.221.112.12": "hermes-interpretation",
    "10.221.112.7": "hermes-backend",
    "10.221.112.3": "hermes-backend",
    "10.221.112.4": "hermes-frontend",
    # OLD IPS (not used anymore)
    "10.221.112.5": "hermes-asr",
    "10.221.112.9": "hermes-asr",
    "10.221.112.8": "hermes-ner",
}

prices_dataset_path = f"{proj_dir}/datasets/pricing_models.json"
prices_dict = parse_prices_file(prices_dataset_path)


test_base_path = sys.argv[1]
# test_base_path = "/home/pita/fs/hermes/hermes-agents/results/gcp.glinerx_ministral-n_6-cl_1.5-29-04-2026_20-21-29"
test_dirs = list_tests(test_base_path, ignore_old=False)
if len(test_dirs) == 0:
    test_dirs = [test_base_path]

current_i = 0
while current_i < len(test_dirs):
    test_path = test_dirs[current_i]
    test_configs = json.load(open(f"{test_path}/config.json", "r"))

    backend_name = network_names["0.0.0.0"]

    if backend_name is None:
        backend_name = get_hostname()

    backend_host_stats = prices_dict[backend_name]
    uses_interpret_vm = test_configs["interpretation"]["hardware_config"] in [
        "vllm-api",
        "triton-server",
    ]
    uses_ner_vm = test_configs["ner"]["hardware_config"] in ["triton-server"]
    uses_asr_vm = test_configs["asr"]["hardware_config"] in ["triton-server"]
    # print(backend_host_stats)
    try:

        save_to_parquet(test_path)

        df_paths = glob(test_path + "/*.parquet")
        dfs = {os.path.basename(p): pl.read_parquet(p) for p in df_paths}

        emergencias_df = parse_emergencias(dfs["emergencias_df.parquet"])
        # inicio_emergencias = min(emergencias_df["start_time"].to_list())
        # final_emergencias = max(emergencias_df["end_time"].to_list())
        n_calls = len(emergencias_df)
        comprimento_total_audios = dfs["audios_df.parquet"][
            "audio_length_seconds"
        ].sum()
        audio_lens = {}
        for row in dfs["audios_df.parquet"].rows(named=True):
            # print(row)
            audio_lens[row["id"]] = row["audio_length_seconds"]
        transcricao_df = parse_transcricao(dfs["transcricao_df.parquet"], audio_lens)
        transcricao_df = transcricao_df.with_columns(
            speed_up=transcricao_df["duracao_audio"]
            / transcricao_df["transcription_seconds"]
        )
        final_stats = {}
        transcricao_tempo_total = transcricao_df["transcription_seconds"].sum()
        horas_transcritas = transcricao_df["duracao_audio"].sum() / 60 / 60
        final_stats["segundos_transcritos"] = transcricao_df["duracao_audio"].sum()
        final_stats["horas_transcritas"] = horas_transcritas
        duracao_total = transcricao_df["duracao_audio"].sum()
        speed_up_total = duracao_total / transcricao_tempo_total

        duracoes_t = transcricao_df["transcription_seconds"].to_list()
        final_stats["tempo_inferencia_medio_transcricao"] = np.mean(duracoes_t)
        final_stats["tempo_inferencia_std_transcricao"] = np.std(duracoes_t)
        final_stats["tempo_transcricao_total"] = transcricao_tempo_total
        final_stats["velocidade_de_transcricao"] = speed_up_total

        inferencia_df = parse_inferencia(dfs["inferencia_df.parquet"], transcricao_df)
        inferencia_stats = estatisticas_sobre_inferencia(inferencia_df, n_calls)
        del dfs["inferencia_df.parquet"]

        dataset_indexes = {
            row["id"]: row["dataset_index"] for row in emergencias_df.rows(named=True)
        }

        emergency_stats = []
        for id_m, infers in inferencia_df.group_by("id_emergencia"):
            audios_len = infers["input_audio_seconds"].sum()
            tokens_in = infers["input_tokens"].sum()
            tokens_out = infers["output_tokens"].sum()
            start = float(infers["horario_contexto"].min())
            infer_hs_asr = [
                (row["horario_contexto"], row["input_audio_seconds"])
                for row in infers.rows(named=True)
                if row["tipo_de_inferencia"] == "asr"
            ]
            infer_hs_asr.sort()
            last_asr_context = infer_hs_asr[-1][0]

            end = last_asr_context + infer_hs_asr[-1][1]

            infer_hs_envolv = [
                (row["horario_contexto"], row["input_audio_seconds"])
                for row in infers.rows(named=True)
                if row["tipo_de_inferencia"] == "envolvimentos"
            ]
            infer_hs_envolv.sort()

            seconds_per_infer = {}
            for inf_type, inf_rows in infers.group_by("tipo_de_inferencia"):
                seconds_per_infer[inf_type[0]] = inf_rows["duracao_inferencia"].sum()
            if len(infer_hs_envolv) > 0:
                last_envolv_context = infer_hs_envolv[-1][0]
                if last_envolv_context < last_asr_context:
                    # Unfinished processing, timeout
                    seconds_per_infer["envolvimentos"] += (
                        last_asr_context - last_envolv_context + 180
                    )
            else:
                # Sem envolvimento algum calculado, adicionamos o tempo total da emergencia em 2x
                seconds_per_infer["envolvimentos"] = (end - start) * 3

            delay_sums = {}
            delay_sums["asr"] = seconds_per_infer.get("asr", 0.0)
            delay_sums["ner"] = delay_sums["asr"] + seconds_per_infer.get("ner", 0.0)

            delay_sums["descricao_e_observacao"] = delay_sums[
                "asr"
            ] + seconds_per_infer.get("descricao_e_observacao", 0.0)
            delay_sums["lista_de_naturezas"] = delay_sums[
                "descricao_e_observacao"
            ] + seconds_per_infer.get("lista_de_naturezas", 0.0)
            delay_sums["natureza_decisiva"] = delay_sums[
                "lista_de_naturezas"
            ] + seconds_per_infer.get("natureza_decisiva", 0.0)
            delay_sums["envolvimentos"] = delay_sums[
                "natureza_decisiva"
            ] + seconds_per_infer.get("envolvimentos", 0.0)

            real_time_factors = {
                inf_type: delay_sums[inf_type] / audios_len for inf_type in delay_sums
            }

            infer_rtf_1 = real_time_factors["ner"]
            infer_rtf_2 = real_time_factors["envolvimentos"]

            rtf_avg = (infer_rtf_1 + infer_rtf_2) / 2

            real_time_factors["total"] = max(delay_sums.values()) / audios_len
            new_line = {
                "id": id_m[0],
                "nome_no_dataset": dataset_indexes.get(id_m[0], None),
                "inicio": start,
                "fim": end,
                "duracao": end - start,
                "comprimento_total_audios": audios_len,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "rtf_avg": rtf_avg,
            }

            for inf_name, delay_sum in delay_sums.items():
                new_line[f"soma_delays_de_{inf_name}"] = delay_sum
                new_line[f"real_time_factor_{inf_name}"] = real_time_factors[inf_name]
            new_line["real_time_factor_total"] = real_time_factors["total"]

            emergency_stats.append(new_line)

        emergency_stats_df = pl.DataFrame(emergency_stats)

        rtfs = emergency_stats_df["rtf_avg"].to_list()

        final_stats["inicio"] = emergency_stats_df["inicio"].min()
        final_stats["fim"] = emergency_stats_df["fim"].max()
        duracao_teste = final_stats["fim"] - final_stats["inicio"]
        print(f"Inicio: {final_stats['inicio']}")
        print(f"Fim: {final_stats['fim']}")
        print(f"Duracao teste: {duracao_teste}")

        print(f"Dtypes: {type(final_stats['inicio'])}")
        print(f"Dtypes: {type(final_stats['fim'])}")
        print(f"Dtypes: {type(duracao_teste)}")

        mean_rtf = np.mean(rtfs)
        p95_rtf = np.percentile(rtfs, 95)

        final_stats["rtf_medio"] = mean_rtf
        final_stats["rtf_p95"] = p95_rtf
        final_stats["numero_de_chamadas"] = n_calls

        duracao_media = emergencias_df["duracao"].mean()

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

        api_uses = []
        for row in inferencia_stats.rows(named=True):
            if row["modelo utilizado"] in prices_dict:
                prices = prices_dict[row["modelo utilizado"]]
                has_audio = "dollars_1h_audio" in prices
                has_text = "dollars_1m_tokens_in" in prices
                cost = 0.0

                if has_text:
                    in_price = prices["dollars_1m_tokens_in"]
                    out_price = prices["dollars_1m_tokens_out"]
                    million_ins = row["input tokens total"] / 1000000
                    million_outs = row["output tokens total"] / 1000000
                    cost += million_ins * in_price + million_outs * out_price
                else:
                    million_ins = 0.0
                    million_outs = 0.0

                if has_audio:
                    # print(row)
                    h = row["input audio seconds total"] / 60 / 60
                    cost += h * prices["dollars_1h_audio"]
                else:
                    h = 0.0

                api_uses.append(
                    {
                        "model": row["modelo utilizado"],
                        "tipo": row["tipo"],
                        "input tokens (millions)": million_ins,
                        "output tokens (millions)": million_outs,
                        "Horas de Áudio": h,
                        "cost": cost,
                    }
                )

        if any([uses_asr_vm, uses_interpret_vm, uses_ner_vm]):
            raw_metrics_df = dfs["raw_metrics.parquet"]
            inference_server_stats = parse_inference_server_readings(
                raw_metrics_df,
                network_names,
                uses_asr_vm,
                uses_interpret_vm,
                uses_ner_vm,
            )
        else:
            inference_server_stats = None

        if "container_stats.parquet" in dfs:
            container_stats = parse_container_stats_df(
                dfs["container_stats.parquet"], backend_host_stats["cpus"]
            )
            container_stats = container_stats.with_columns(
                hostname=pl.lit(backend_name)
            )
            n_readings = container_stats["ReadingID"].max()
            interval = (final_stats["fim"] - final_stats["inicio"]) / n_readings
            container_stats = container_stats.with_columns(
                timestamp=pl.col("ReadingID") * interval + final_stats["inicio"]
            )
            if any([uses_asr_vm, uses_interpret_vm, uses_ner_vm]):
                container_stats = pl.concat(
                    [container_stats, inference_server_stats], how="align"
                )
            hardware_stats_df = hardware_usage(container_stats, uses_interpret_vm)

            hardware_timeline_rows = []
            for uniq_tp, rows in container_stats.group_by(["ReadingID", "hostname"]):
                read_id, hostname = uniq_tp
                mean_columns = ["timestamp", "CPUMax", "MaxRAM"]
                sum_columns = ["CPUPercRelative", "MemPerc"]
                new_row = {"ReadingID": read_id, "hostname": hostname}
                for c in mean_columns:
                    new_row[c] = np.mean(rows[c].to_list())
                for c in sum_columns:
                    new_row[c] = sum(rows[c].to_list())
                new_row["CPUPerc"] = new_row["CPUPercRelative"] * 100
                del new_row["CPUPercRelative"]
                hardware_timeline_rows.append(new_row)
            hardware_timeline_df = pl.DataFrame(hardware_timeline_rows)

        else:
            container_stats = None
            hardware_stats_df = None
            hardware_timeline_df = None

        # print('\n'.join([json.dumps(la) for la in gastos_por_inferencia]))
        # print(final_stats)

        costs_df = parse_costs(
            n_calls,
            hardware_stats_df,
            inferencia_stats,
            prices_dict,
            backend_name,
            test_configs,
            horas_transcritas,
            duracao_teste,
            api_uses,
        )
        # %%
        api_cost_values = costs_df.filter(pl.col("Tipo de Custo") == "API")[
            "Custo ($)"
        ].to_list()
        # A lista inclui o total
        api_total_cost = max(api_cost_values)
        final_stats["custo_api_por_hora"] = (
            api_total_cost / final_stats["horas_transcritas"]
        )
        custos_por_hora = costs_df["Custo por Hora ($)"].to_list()
        custos_por_hora = [v for v in custos_por_hora if v == v and v is not None]
        # A lista inclui o total
        vms_hora_total = max(custos_por_hora)

        por_mes, por_ligacao, custo_vms_total_mes = custo_mensal(
            api_total_cost,
            final_stats["horas_transcritas"],
            final_stats["duracao_teste"] / 60 / 60,
            vms_hora_total,
            final_stats["numero_de_chamadas"],
        )
        final_stats["cost_per_call"] = por_ligacao
        final_stats["cost_per_month"] = por_mes
        final_stats["cost_vms_per_month"] = custo_vms_total_mes

        final_stats_df = [
            {"Métrica": m.replace("_", " "), "Valor": v} for m, v in final_stats.items()
        ]
        timestamp_names = ["inicio", "fim"]
        from datetime import datetime

        for d in final_stats_df:
            if d["Métrica"] in timestamp_names:
                print(d["Valor"])
                # Convert d['Valor'] float absolute seconds to timestring
                d["Valor"] = datetime.fromtimestamp(d["Valor"]).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                print(d["Valor"])
        final_stats_df = pl.DataFrame(final_stats_df)

        costs_col_new_order = [
            "Recurso",
            "Sub-recurso",
            "Tipo de Custo",
            "Horas de Áudio",
            "Tokens de Input (milhões)",
            "Tokens de Output (milhões)",
            "Custo ($)",
            "Por Ligação ($)",
        ]

        costs_df = costs_df.select(costs_col_new_order)

        excel_path = f"{test_path}/results.xlsx"
        print(excel_path)
        if os.path.exists(excel_path):
            os.remove(excel_path)

        dfs["emergencias_df.parquet"] = emergency_stats_df
        # dfs["transcricao_df.parquet"] = transcricao_df
        del dfs["transcricao_df.parquet"]
        dfs["inferencia_df.parquet"] = inferencia_df
        dfs["inferencia_stats"] = inferencia_stats
        if container_stats is not None:
            dfs["container_stats.parquet"] = container_stats
            dfs["uso_de_hardware"] = hardware_stats_df
            dfs["hardware_timeline_df"] = hardware_timeline_df
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

        # return dfs

        """parsed_df_paths = glob(f"{test_path}/parsed_dfs/*.parquet")
        dfs2 = {}
        for df in parsed_df_paths:
            name = os.path.basename(df).replace(".parquet", "")
            dfs2[name] = pl.read_parquet(df)
            print(f"Re-loaded {name}")"""

        delays_plot_path = plot_delays(
            test_path,
            dfs["inferencia_df.parquet"],
            final_stats["inicio"],
            dfs["hardware_timeline_df"],
            dfs["raw_emergency_audios.parquet"],
        )

        # comparison_df = pl.read_csv(f'{test_path}/semantic_performance-complete.tsv', separator='\t')
        configs_path = f"{test_path}/config.json"
        dashboard_json = dashboard_stats(dfs)
        dashboard_json["final_stats_dict"] = final_stats
        dashboard_json["config"] = json.load(open(configs_path, "r"))
        # print(json.dumps(dashboard_json, ensure_ascii=False, indent=2))
        # %%
        dashboard = create_final_dashboard(dashboard_json, delays_plot_path, proj_dir)
        file_path = f"{test_path}/dashboard.html"
        dashboard.save(
            file_path,
            embed=True,  # Garante que todos os dados e JS sejam embutidos no arquivo
            title="Dashboard Hermes",
        )
        json.dump(
            final_stats,
            open(f"{test_path}/final_stats.json", "w"),
            indent=2,
            ensure_ascii=False,
        )
    except Exception as err:
        print(err)
        print("Test:", test_path)
        raise (err)

    current_i += 1
