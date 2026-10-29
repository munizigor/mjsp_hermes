import polars as pl
import numpy as np

from test_utils.util import datetime_to_int


def parse_emergencias(df: pl.DataFrame):
    df = df.with_columns(end_time=datetime_to_int(df["end_time"].to_list()))
    df = df.with_columns(start_time=datetime_to_int(df["start_time"].to_list()))
    df = df.filter(pl.col("end_time").is_not_nan())
    df = df.with_columns(duracao=df["end_time"] - df["start_time"])

    return df


def drop_invalid_emergencies(df: pl.DataFrame, valid_ids: list):
    if "id_emergencia" in df.columns:
        df = df.cast({"id_emergencia": pl.Int64})
        df = df.filter(pl.col("id_emergencia").is_in(valid_ids))
    return df


def parse_transcricao(df: pl.DataFrame, audio_lens: dict):
    # id_emergencias = emergencias_df['id'].to_list()
    # duracaos = emergencias_df['duracao'].to_list()
    # duracao_dict = {x: y for x,y in zip(id_emergencias, duracaos)}

    # print(duracao_dict)

    duracaos2 = np.array(
        [audio_lens[audio_id] for audio_id in df["audio_id"].to_list()]
    )
    print(duracaos2)
    df = df.with_columns(duracao_audio=duracaos2)
    return df


def parse_inference_server_readings(
    raw_metrics_df: pl.DataFrame,
    network_names,
    uses_asr_vm,
    uses_interpret_vm,
    uses_ner_vm,
):
    lines = []
    import json

    reading_id = 10000000
    n_cpus = 4
    for row in raw_metrics_df.rows(named=True):
        readings_dict = json.loads(row["metrics_dict"])
        valid_values = [x for x in readings_dict.values() if x != None and x != "None"]
        if len(valid_values) > 0:
            for k in readings_dict:
                if readings_dict[k] is None or readings_dict[k] == "None":
                    readings_dict[k] = "0.0"
                    if "total" in k:
                        readings_dict[k] = "1"
            hostname = row["hostname"]
            if hostname in network_names:
                hostname = network_names[hostname]
            lines.append(
                {
                    # "BlockIO": ,
                    "CPUPerc": float(readings_dict["cpu_utilization"]) * n_cpus * 100,
                    # "Container": np.nan,
                    "ID": str(row["id"] + reading_id),
                    "MemPerc": float(readings_dict["cpu_memory_used_bytes"])
                    / float(readings_dict["cpu_memory_total_bytes"])
                    * 100,
                    "MemUsage": float(readings_dict["cpu_memory_used_bytes"])
                    / 1024
                    / 1024
                    / 1024,
                    "Name": row["server_type"] + " " + row["inference_type"],
                    "ReadingID": row["id"] + reading_id,
                    # "NetIO": np.nan,
                    "MaxRAM": float(readings_dict["cpu_memory_total_bytes"])
                    / 1024
                    / 1024
                    / 1024,
                    "CPUMax": n_cpus * 100,
                    "CPUPercRelative": float(readings_dict["cpu_utilization"]),
                    # "n_processes": np.nan,
                    "hostname": hostname,
                    "timestamp": row["readings_time"],
                    "GPUPercRelative": float(readings_dict["gpu_utilization"]),
                    "VMemPerc": float(readings_dict["gpu_memory_used_bytes"])
                    / float(readings_dict["gpu_memory_total_bytes"])
                    * 100,
                    "MaxVRAM": float(readings_dict["gpu_memory_total_bytes"])
                    / 1024
                    / 1024
                    / 1024,
                }
            )
    return pl.DataFrame(lines)


def parse_container_stats_df(df, total_cpus):
    max_cpu_usage = total_cpus * 100
    mem_perc_strs = df["MemPerc"].to_list()
    mem_percs = []
    for s in mem_perc_strs:
        try:
            mem_percs.append(float(s.rstrip("%")))
        except ValueError as err:
            mem_percs.append(None)
    mem_percs = np.array(mem_percs)

    gib_strs = df["MemUsage"].to_list()
    max_gibs = []
    for s in gib_strs:
        try:
            max_gibs.append(
                float(s.split(" / ")[1].replace("GiB", "").replace("TiB", ""))
            )
        except ValueError as err:
            max_gibs.append(None)

    cpu_perc_strs = df["CPUPerc"].to_list()
    cpu_percs = []
    for s in cpu_perc_strs:
        try:
            cpu_percs.append(float(s.rstrip("%")))
        except ValueError as err:
            cpu_percs.append(0.0)

    n_containers = len(set(df["Name"].to_list()))
    reading_n = np.array([int(n / n_containers) for n, _ in enumerate(cpu_percs)])
    cpu_percs = np.array(cpu_percs)
    cpu_percs_relative = cpu_percs / max_cpu_usage
    gpu_percs = np.array([np.nan for _ in range(len(cpu_percs))])
    vmem_percs = np.array([np.nan for _ in range(len(cpu_percs))])

    df = df.with_columns(ReadingID=reading_n)
    df = df.with_columns(MemPerc=mem_percs)
    df = df.with_columns(MaxRAM=np.array(max_gibs))
    df = df.with_columns(MemUsage=df["MaxRAM"] * (df["MemPerc"] / 100))
    df = df.with_columns(CPUPerc=cpu_percs)
    df = df.with_columns(CPUMax=max_cpu_usage)
    df = df.with_columns(CPUPercRelative=cpu_percs_relative)
    df = df.with_columns(n_processes=np.array([int(x) for x in df["PIDs"].to_list()]))
    df = df.with_columns(GPUPercRelative=gpu_percs)
    df = df.with_columns(VMemPerc=vmem_percs)
    df.drop_in_place("PIDs")

    return df


def hardware_usage(container_stats: pl.DataFrame, uses_interpreter_vm) -> pl.DataFrame:

    lines = []
    for name, container_df in container_stats.group_by("Name"):
        hostname = container_df["hostname"].to_list()[0]
        if len(container_df) > 1:
            n_readings = len(container_df)
            total_relative_usage = (
                container_df["CPUPercRelative"].sum() / n_readings
            ) * 100
            total_relative_mem_usage = container_df["MemPerc"].sum() / n_readings
            total_relative_vmem_usage = container_df["VMemPerc"].sum() / n_readings
            total_relative_gpu_usage = (
                container_df["GPUPercRelative"].sum() / n_readings
            ) * 100
            mem_usage = container_df["MemUsage"]
            cpu_usage = container_df["CPUPerc"]

            mem_q1 = np.percentile(mem_usage, 25)
            mem_median = np.median(mem_usage)
            mem_q3 = np.percentile(mem_usage, 75)
            mem_max = np.percentile(mem_usage, 98)

            cpu_q1 = np.percentile(cpu_usage, 25)
            cpu_median = np.median(cpu_usage)
            cpu_q3 = np.percentile(cpu_usage, 75)
            cpu_max = np.percentile(cpu_usage, 98)

            try:
                tp = name[0].split("-")[-2]
            except Exception as err:
                tp = "Servidor de Inferência de " + name[0].split()[-1]

            lines.append(
                {
                    "Container": name[0].replace("hermes-agents-", ""),
                    "ContainerType": tp,
                    "hostname": hostname,
                    "Memoria Q1": mem_q1,
                    "Memoria Mediana": mem_median,
                    "Memoria Q3": mem_q3,
                    "Memoria Maxima": mem_max,
                    "CPU Q1": cpu_q1,
                    "CPU Mediana": cpu_median,
                    "CPU Q3": cpu_q3,
                    "CPU Máxima": cpu_max,
                    "Uso Relativo das CPUs": total_relative_usage,
                    "Uso Relativo da RAM": total_relative_mem_usage,
                    "Uso Relativo das GPUs": total_relative_gpu_usage,
                    "Uso Relativo da VRAM": total_relative_vmem_usage,
                }
            )

    if uses_interpreter_vm:
        "hermes-interpretation"
        lines.append(
            {
                "Container": "hermes-interpretation",
                "ContainerType": "Servidor de Inferência de Interpretation",
                "hostname": "hermes-interpretation",
                "Memoria Q1": np.nan,
                "Memoria Mediana": np.nan,
                "Memoria Q3": np.nan,
                "Memoria Maxima": np.nan,
                "CPU Q1": np.nan,
                "CPU Mediana": np.nan,
                "CPU Q3": np.nan,
                "CPU Máxima": np.nan,
                "Uso Relativo das CPUs": np.nan,
                "Uso Relativo da RAM": np.nan,
                "Uso Relativo das GPUs": np.nan,
                "Uso Relativo da VRAM": np.nan,
            }
        )

    return pl.DataFrame(lines)


def parse_inferencia(df: pl.DataFrame, asr_df: pl.DataFrame):
    # df = df.with_columns(end_time = datetime_to_int(df['end_time'].to_list()))
    # df = df.with_columns(start_time = datetime_to_int(df['start_time'].to_list()))
    # df = df.filter(pl.col('end_time').is_not_nan())
    new_lines = []
    for row in asr_df.rows(named=True):
        new_lines.append(
            {
                "id": None,
                "id_emergencia": row["id_emergencia"],
                "horario_contexto": row["start_time"],
                "horario_fim": row["start_time"] + row["transcription_seconds"],
                "tipo_de_inferencia": "asr",
                "resultado": row["part"],
                "duracao_inferencia": row["transcription_seconds"],
                "duracao_outros_processamentos": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "modelo_utilizado": row["transcription_model"],
                "input_audio_seconds": row["duracao_audio"],
            }
        )
    asr_lines = pl.DataFrame(new_lines)
    df = df.with_columns(input_audio_seconds=pl.lit(0.0))
    print(df.columns)
    print(asr_lines.columns)
    # df.drop_in_place("id")
    try:
        df = pl.concat([df, asr_lines])
    except pl.exceptions.SchemaError as err:
        df = df.with_columns(
            pl.col("horario_fim").cast(pl.Float64),
            pl.col("horario_contexto").cast(pl.Float64),
        )
        df = pl.concat([df, asr_lines])

    df = df.with_columns(delay=df["horario_fim"] - df["horario_contexto"])

    df = df.sort(["id_emergencia", "horario_contexto", "horario_fim"])

    return df


def estatisticas_sobre_inferencia(df: pl.DataFrame, n_calls):
    gastos_por_inferencia = []

    for tp, tp_df in df.group_by("tipo_de_inferencia"):
        line = {
            "tipo": tp[0],
            "inferencias": len(tp_df),
            "inferencias_por_chamada": len(tp_df) / n_calls,
            "delay_acumulado": tp_df["delay"].sum(),
            "tempo_inferencia": tp_df["duracao_inferencia"].sum(),
            "tempo_outros": tp_df["duracao_outros_processamentos"].sum(),
            "input_audio_seconds_total": tp_df["input_audio_seconds"].sum(),
            "input_tokens_total": tp_df["input_tokens"].sum(),
            "output_tokens_total": tp_df["output_tokens"].sum(),
            "modelo_utilizado": tp_df["modelo_utilizado"].unique()[0],
        }
        line["delay_medio"] = line["delay_acumulado"] / len(tp_df)
        line["input_tokens_per_sec"] = (
            line["input_tokens_total"] / line["tempo_inferencia"]
        )
        line["output_tokens_per_sec"] = (
            line["output_tokens_total"] / line["tempo_inferencia"]
        )
        line["input_tokens_per_call"] = line["input_tokens_total"] / n_calls
        line["output_tokens_per_call"] = line["output_tokens_total"] / n_calls
        line["tempo_inferencia_per_call"] = line["tempo_inferencia"] / n_calls
        line["tempo_outros_per_call"] = line["tempo_outros"] / n_calls

        gastos_por_inferencia.append(line)
    total_chaves = [
        "inferencias",
        "delay_acumulado",
        "tempo_inferencia",
        "tempo_outros",
        "input_audio_seconds_total",
        "input_tokens_total",
        "output_tokens_total",
        "input_tokens_per_call",
        "tempo_inferencia_per_call",
        "tempo_inferencia_per_call",
        "tempo_outros_per_call",
    ]
    totais = {"tipo": "Totais"}
    for k in total_chaves:
        totais[k] = sum([l[k] for l in gastos_por_inferencia])

    total_audio = float(totais["input_audio_seconds_total"])

    gastos_por_inferencia.append(totais)

    for line in gastos_por_inferencia:
        try:
            in_tokens_int = int(line["input_tokens_total"])
            out_tokens_int = int(line["output_tokens_total"])
            line["input_tokens_per_audio_second"] = in_tokens_int / total_audio
            line["output_tokens_per_audio_second"] = out_tokens_int / total_audio
        except Exception as err:
            line["input_tokens_per_audio_second"] = np.nan
            line["output_tokens_per_audio_second"] = np.nan

    for line in gastos_por_inferencia:
        for key in list(line.keys()):
            if "_" in key:
                line[key.replace("_", " ")] = line[key]
                del line[key]

    return pl.DataFrame(gastos_por_inferencia)
