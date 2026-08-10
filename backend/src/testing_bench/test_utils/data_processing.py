import glob
import json
import os
import sqlite3 as sqlite
import sys

import numpy as np
import polars as pl
from datasets import load_dataset, Audio, load_from_disk

from test_utils.scheduling import calcular_delays, read_audio_length

"""
Exemplo de metadata:
{
        "Emergencia": {
            "Endereco": {
                "descricao": "Rua Monjoli, Centro",
                "desc_tipo": "rua_bairro",
                "rua": "Rua Monjoli",
                "numero": "",
                "bairro": "",
                "cidade": "",
                "estado": "Paraná",
                "CEP": "85980-000",
                "coords": [
                    -24.082567,
                    -54.2531192
                ],
                "ref_name": "OdontoK",
                "ref_endereco_completo": "Rua Monjoli, 655 - Centro, Guaíra",
                "ref_tipos": [
                    "dentist",
                    "point_of_interest",
                    "health",
                    "establishment"
                ],
                "ref_distancia": 15.395682677529699
            },
            "Natureza": {
                "Prioridade": "Alta",
                "Natureza": "Ataque de Animal",
                "TiposAgencia": "CdBM",
                "Descrição": "Ataque de animal a um humano em área urbana"
            },
            "Duração da Ligacao (Minutos)": 3.03,
            "Hora": "13h27min"
        },
        "Perfil do Solicitante": {
            "Nome Solicitante": "",
            "Idade": 86,
            "Genero": "H",
            "Numero": "96477 6490",
            "Instrucao": "Analfabestismo Funcional",
            "Envolvimento": "Vizinho",
            "Nível de Desespero/Estresse/Medo (0 a 10)": 7
        },
        "roteiro": "<roteiro completo da chamada>",
        "index": 0
    }
"""


def load_audios(fake_calls_dataset_path, audio_model):
    # Verifica se os arquivos de áudio existem
    audio_paths = glob.glob(f"{fake_calls_dataset_path}/speech/{audio_model}/*.wav")
    audio_paths += glob.glob(f"{fake_calls_dataset_path}/speech/{audio_model}/*.WAV")
    audio_paths = glob.glob(f"{fake_calls_dataset_path}/{audio_model}/*.wav")
    audio_paths += glob.glob(f"{fake_calls_dataset_path}/{audio_model}/*.WAV")
    if not audio_paths:
        print(
            f"Nenhum arquivo de áudio encontrado em {fake_calls_dataset_path}/speech/{audio_model}/"
        )
        sys.exit(1)

    audios = {}
    for audio_path in audio_paths:
        if "_part" not in os.path.basename(audio_path):
            path_parts = audio_path.split("/")
            audio_id = path_parts[-1].replace(".wav", "").replace(".WAV", "")
            audios[audio_id] = audio_path

    print(f"Encontrados {len(audios)} arquivos de áudio:")
    for audio_id, audio_path in audios.items():
        print(f"  Audio ID: {audio_id} -> {audio_path}")

    return audios


def load_metadata(audios, fake_calls_dataset_path, carga_de_ligacoes):
    # Carrega o metadata
    metadata_file = f"{fake_calls_dataset_path}/chamadas_roteirizadas_final.json"
    if not os.path.exists(metadata_file):
        print(f"Arquivo de metadata não encontrado: {metadata_file}")
        sys.exit(1)

    metadata_json = json.load(open(metadata_file, "r"))

    # Filtra apenas entradas que têm áudio correspondente
    has_audio = []
    for n, entry in enumerate(metadata_json):
        entry["index"] = str(entry["index"])
        if entry["index"] in audios:
            has_audio.append(n)

    audio_info = [metadata_json[n] for n in has_audio]
    print(f"Metadata com áudio: {len(audio_info)} entradas")

    print("Lendo durações de áudios")
    for i, audio_data in enumerate(audio_info):
        audio_path = audios[audio_data["index"]]
        read_audio_length(audio_data, audio_path)

    print("Calculando delays de início para manter carga constante...")
    delays = calcular_delays(audio_info, carga_de_ligacoes)

    for i, (audio_data, delay) in enumerate(zip(audio_info, delays)):
        audio_data["Delay"] = delay
        print(
            f"Áudio {audio_data['index']} -> duração {audio_data['Duracao Real']:.1f}s, delay {delay:.1f}s"
        )

    return audio_info


def split_audio_by_silence(
    audio_data,
    audio_path,
    min_silence_milisec=500,
    silence_thresh_db=-20,
    keep_silence_milisec=200,
    min_audio_length_sec=10,
):
    from pydub import AudioSegment
    from pydub.silence import split_on_silence

    sound = AudioSegment.from_wav(audio_path)
    chunks = split_on_silence(
        sound,
        min_silence_len=min_silence_milisec,
        silence_thresh=silence_thresh_db,
        keep_silence=True,
    )

    chunks2 = []

    for part in chunks:
        if len(chunks2) == 0:
            chunks2.append(part)
        else:
            last_chunk_len = len(chunks2[-1])
            if last_chunk_len < min_audio_length_sec * 1000:
                chunks2[-1] = chunks2[-1] + part
            else:
                chunks2.append(part)

    segment_paths = []
    lengths = []
    for i, chunk in enumerate(chunks2):
        chunk_dir = os.path.dirname(audio_path)
        chunk_filename = f"{chunk_dir}/{audio_data['index']}_part{i}.wav"
        chunk.export(chunk_filename, format="wav")
        segment_paths.append(chunk_filename)
        lengths.append(len(chunk) / 1000)

    print(f"Áudio {audio_data['index']} dividido em {len(chunks2)} pedaços")
    return segment_paths, lengths


def split_audios(audio_info, audios):
    print("Dividindo áudios longos em pedaços menores...")
    for audio_data in audio_info:
        audio_path = audios[audio_data["index"]]
        print(audio_path)
        min_silence_milisec = 600
        silence_thresh_db = -20
        segment_paths, lengths = split_audio_by_silence(
            audio_data,
            audio_path,
            min_silence_milisec=min_silence_milisec,
            silence_thresh_db=silence_thresh_db,
        )
        print(f"Total de áudios após divisão: {len(segment_paths)}")

        while len(segment_paths) < 5:
            min_silence_milisec -= 10
            silence_thresh_db -= 0.5
            print(
                "Tentando novamente com min_silence_milisec=",
                min_silence_milisec,
                "silence_thresh_db=",
                silence_thresh_db,
            )
            segment_paths, lengths = split_audio_by_silence(
                audio_data,
                audio_path,
                min_silence_milisec=min_silence_milisec,
                silence_thresh_db=int(silence_thresh_db),
            )
            print(f"Total de áudios após divisão: {len(segment_paths)}")

        segmentos_dict = [
            {"path": p, "Duração Real": dr} for p, dr in zip(segment_paths, lengths)
        ]
        audio_data["Segmentos"] = segmentos_dict


def sql_to_df(lines, cols):
    new_lines = [{cols[n]: v for n, v in enumerate(line)} for line in lines]
    df = pl.DataFrame(new_lines)
    return df


def convert_sqlite_to_parquet(table_names, sqlite_path, output_dir):
    parquet_paths = {}
    for table_name in table_names:
        sqlite_conn = sqlite.connect(sqlite_path)
        cursor = sqlite_conn.cursor()
        output_parquet = f"{output_dir}/raw_{table_name}.parquet"
        emergency_rows = cursor.execute(f"SELECT * FROM {table_name}").fetchall()
        emergency_columns = [desc[0] for desc in cursor.description]
        sqlite_conn.close()
        emergencias_df = sql_to_df(emergency_rows, emergency_columns)
        emergencias_df.write_parquet(output_parquet)
        parquet_paths[table_name] = output_parquet
    return parquet_paths


def save_to_parquet(output_dir):
    sqlite_path = f"{output_dir}/sqlite.db"
    call_info_jsons = glob.glob(f"{output_dir}/run_tests-*.json")
    results_by_index = {}
    for p in call_info_jsons:
        d = json.load(open(p, "r"))
        results_by_index[str(d["id_emergencia"])] = {
            "dataset_index": d["dataset_index"],
            "delay_local": d["delay_local"],
            "all_results": d["results"],
            "call_start_delay": d["call_start_delay"] if "call_start_delay" in d else 0,
            "audio_lengths": d["audio_lengths"] if "audio_lengths" in d else [],
        }
    parquet_paths = convert_sqlite_to_parquet(
        [
            "emergencies",
            "emergency_transcripts",
            "emergency_audios",
            "resultados_inferencia",
            "metrics",
        ],
        sqlite_path,
        output_dir,
    )

    emergencias_df = pl.read_parquet(parquet_paths["emergencies"])

    id_emergencias = emergencias_df["id"].to_list()

    delays = []
    indexes = []
    call_start_delay = []
    audio_lengths = []
    for a in id_emergencias:
        if str(a) in results_by_index:
            indexes.append(results_by_index[str(a)]["dataset_index"])
            delays.append(results_by_index[str(a)]["delay_local"])
            audio_lengths.append(str(results_by_index[str(a)]["audio_lengths"]))
            if "call_start_delay" in results_by_index[str(a)]:
                call_start_delay.append(results_by_index[str(a)]["call_start_delay"])
            else:
                call_start_delay.append(0)
        else:
            indexes.append(None)
            delays.append(None)
            audio_lengths.append(None)
            call_start_delay.append(None)

    all_results_lines = []
    for id_emergencia, rbi in results_by_index.items():
        line = {"id_emergencia": id_emergencia, "dataset_index": rbi["dataset_index"]}
        for col, val in rbi["all_results"].items():
            line[col] = json.dumps(val, ensure_ascii=False, indent=2)
        all_results_lines.append(line)

    inference_summary = pl.DataFrame(all_results_lines)
    indexes = np.array(indexes)
    delays = np.array(delays)
    audio_lengths = np.array(audio_lengths)
    call_start_delay = np.array(call_start_delay)

    emergencias_df = emergencias_df.with_columns(dataset_index=indexes)
    emergencias_df = emergencias_df.with_columns(delay_local=delays)
    emergencias_df = emergencias_df.with_columns(segment_lengths=audio_lengths)
    emergencias_df = emergencias_df.with_columns(call_start_schedule=call_start_delay)

    transcricao_df = pl.read_parquet(parquet_paths["emergency_transcripts"])
    audios_df = pl.read_parquet(parquet_paths["emergency_audios"])
    inferencia_df = pl.read_parquet(parquet_paths["resultados_inferencia"])

    if str(emergencias_df.dtypes[-3]) == "Object":
        no_nones = [
            x if x is not None else 0 for x in emergencias_df["delay_local"].to_list()
        ]
        no_nones = np.array(no_nones)
        emergencias_df = emergencias_df.with_columns(delay_local=no_nones)
        print(emergencias_df["delay_local"])
    if str(emergencias_df.dtypes[-4]) == "Object":
        no_nones = [
            x if x is not None else np.nan
            for x in emergencias_df["dataset_index"].to_list()
        ]
        no_nones = np.array(no_nones)
        emergencias_df = emergencias_df.with_columns(dataset_index=no_nones)
    if str(emergencias_df.dtypes[-1]) == "Object":
        no_nones = [
            x if x is not None else 0
            for x in emergencias_df["call_start_schedule"].to_list()
        ]
        no_nones = np.array(no_nones)
        emergencias_df = emergencias_df.with_columns(call_start_schedule=no_nones)
        print(emergencias_df["call_start_schedule"])

    print(emergencias_df.dtypes)
    print(emergencias_df.columns)

    try:
        inference_summary.write_parquet(f"{output_dir}/inference_summary.parquet")
        transcricao_df.write_parquet(f"{output_dir}/transcricao_df.parquet")
        audios_df.write_parquet(f"{output_dir}/audios_df.parquet")
        inferencia_df.write_parquet(f"{output_dir}/inferencia_df.parquet")
        emergencias_df.write_parquet(f"{output_dir}/emergencias_df.parquet")
    except pl.exceptions.ComputeError as err:
        print(err)
        raise err
        return err, emergencias_df, transcricao_df, audios_df, inferencia_df


def get_inputs(n_samples: int, dataset_name: str):
    # Only load if a file with the correct sampling has not been sampled and saved yet
    df_path = f"inputs/df_{n_samples}.parquet"
    if os.path.exists(df_path):
        print(f"Loading from {df_path}")
        ds = load_from_disk(df_path)
        return ds, df_path
    else:
        print("Downloading dataset...")
        ds = load_dataset(
            dataset_name,
            split="train",
            streaming=False,
            # columns=colunas,
        )  # .take(max_n)
        ds = ds.filter(lambda x: x["modelo_texto"] != "cnmoro-gemma3-gaia-ptbr-4b_q8_0")
        ds = ds.shuffle(seed=1337)
        ds = ds.select(range(n_samples))
        print("Casting audio column...")
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        sample = next(iter(ds))["audio"]["array"]
        print("Sample:", sample.shape)
        os.makedirs("inputs", exist_ok=True)
        ds.save_to_disk(df_path)
        return ds, df_path
