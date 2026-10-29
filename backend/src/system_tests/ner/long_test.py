import os
import json
import multiprocessing as mp
from queue import Empty as QueueEmptyError
from time import time, sleep

import polars as pl

DATASETS_DIR = "/datasets"

def download_eval_dataset(total_rows=36, estado_especifico = 'Distrito Federal',
        estado_especifico_perc = 1/3):

    expected_path = f"{DATASETS_DIR}/df_emergencies_eval-n={total_rows}.parquet"
    if os.path.exists(expected_path):
        print(f"NER eval dataset already exists at {expected_path}")
        return pl.read_parquet(expected_path)

    n_estado = round(total_rows*estado_especifico_perc)
    n_other_states = total_rows - n_estado

    COLUMNS = [
        "ID",
        "roteiro_segmentado",
        "endereco_completo",
        "nome_solicitante",
        "rua",
        "numero",
        "complemento",
        "bairro",
        "cidade",
        "estado",
        "CEP",
        "ponto_de_referencia",
        "fatoOcorrendoNesteMomento",
        "autorDoFatoNoLocal",
        "autorDoFatoArmado",
        "feridosComRiscoDeMorte",
        "riscoDeTumulto",
        "leiMariaPenha",
    ]

    # Lazy scan across all Parquet shards on the Hub
    lf = pl.scan_parquet("hf://datasets/pitagoras-alves/fake-emergencies-br/data/train-*.parquet")

    df = (
        lf.select(COLUMNS)
        .filter(pl.col("estado") == estado_especifico)
        .head(n_estado)  # Adjust the number of rows you want to fetch
        .collect()
    )

    print(f"Downloaded {len(df)} rows.")
    print(df.head())

    df_other_states = (
        lf.select(COLUMNS)
        .filter(pl.col("estado") != estado_especifico)
        .head(n_other_states)  # Adjust the number of rows you want to fetch
        .collect()
    )

    print(f"Downloaded {len(df_other_states)} rows.")
    print(df_other_states.head())

    df_eval = pl.concat([df, df_other_states])

    # Save locally if needed
    df_eval.write_parquet(expected_path)
    df_eval.write_csv(expected_path.replace('.parquet', '.csv'))
    return df_eval


def eval_ner(transcript_queue: mp.Queue, result_queue: mp.Queue, total_rows=36):

    expected_result_path = f"{DATASETS_DIR}/ner_eval_results.json"

    df_eval = download_eval_dataset(total_rows=total_rows)

    existing_rows = 0
    if os.path.exists(expected_result_path):
        old_results = json.load(open(expected_result_path, "r"))
        existing_rows = len(old_results)
        if existing_rows >= total_rows:
            return expected_result_path

    test_id = -1
    results = []

    for row in df_eval.iter_rows(named=True):
        expected = {k: v for k, v in row.items()}
        transcript = row["roteiro"]
        print(transcript)

        transcript_queue.put((str(test_id), -1, transcript))
        wait_start = time()
        sleep(1)
        result_attempts = 3
        result_timeoff_secs = 3
        while result_attempts > 0:
            try:
                result = result_queue.get(timeout=result_timeoff_secs)
                inf_time = time() - wait_start
                result["inf_time"] = inf_time
                result["expected"] = expected
                results.append(result)
                break
            except QueueEmptyError:
                result_attempts -= 1
                result_timeoff_secs *= 2
                print("NER Timeout after", time() - wait_start, "; attempts left: ", result_attempts)

        test_id -= 1
    
        with open(expected_result_path, "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    return expected_result_path

    
    

    