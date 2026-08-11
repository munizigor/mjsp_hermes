import time
import os
import json
import sys
from typing import List
from typing import List
from multiprocessing import Pool
import gc
import re
from copy import copy
import numpy as np
from tqdm import tqdm
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance.JaroWinkler import (
    normalized_similarity as normalized_similarity1,
)
from rapidfuzz.distance.Levenshtein import (
    normalized_similarity as normalized_similarity2,
)
from unidecode import unidecode


entity_redundancies = {
    "rua": "rua_ou_logradouro",
    "municipio": "cidade",
    "street_number": "numero",
    "number": "numero",
    "endereço_complemento": "complemento",
}

SIM_TH = 0.71
NAME_SIM_TH = 0.7
NUMBER_TH = 0.95
USE_PERC = 0.96

HONORIFICS = [
    r"\bdra\b",
    r"\bdr\b",
    r"\bdr\.\b",
    r"\bsr\b",
    r"\bsr\.\b",
    r"\bsra\b",
    r"\bsra\.\b",
    r"\bsenhor\b",
    r"\bsenhora\b",
    r"\bprof\b",
    r"\bprof\.\b",
]

EXPANSIONS = {
    r"\bapto\b\.?": "apartamento",
    r"\bav\b\.?": "avenida",
    r"\bbl\b\.?": "bloco",
    r"\bqd\b\.?": "quadra",
    r"\bed\b\.?": "edificio",
    r"\bcj\b\.?": "conjunto",
    r"\bconj\b\.?": "conjunto",
    r"\br\b\.?": "rua",
    r"\bnum\b\.?": "numero",
}


def basic_norm(t: str, name=None):
    t2 = copy(t)
    t2 = t2.lower()
    t2 = unidecode(t2).replace("...", "").strip(".,;:-").strip()
    t3 = []
    for w in t2.split():
        repeated = False
        if len(t3) > 0:
            if w.strip(".,") == t3[-1].strip(".,"):
                repeated = True
        if not repeated:
            t3.append(w)
    t2 = " ".join(t3)

    if name != None:
        if "pessoa" in name or "nome" in name:
            for h in HONORIFICS:
                t2 = re.sub(h, "", t2)
        if "numero" in name:
            t2 = name + ": " + t2
            t2 = t2.replace("numero ", "")
    # t = t.replace('dr.', 'doutor')
    return t2


def basic_norm_with_abbrevs(t: str, name=None):
    t2 = copy(t)
    t2 = t2.lower()
    t2 = unidecode(t2).replace("...", "").strip(".,;:-").strip()

    # Pre-process abbreviations
    for pattern, replacement in EXPANSIONS.items():
        t2 = re.sub(pattern, replacement, t2)

    t3 = []
    for w in t2.split():
        repeated = False
        if len(t3) > 0:
            if w.strip(".,") == t3[-1].strip(".,"):
                repeated = True
        if not repeated:
            t3.append(w)
    t2 = " ".join(t3)

    if name != None:
        if "pessoa" in name or "nome" in name:
            for h in HONORIFICS:
                t2 = re.sub(h, "", t2)
        if "numero" in name:
            t2 = name + ": " + t2
            t2 = t2.replace("numero ", "")

    # Clean structural whitespace artifacts left by expansions
    t2 = re.sub(r"\s+", " ", t2).strip()

    return t2


diffs_printed = set()


def _process_single_column_fmax(args):
    """
    Worker function to process a single column.
    Must be top-level to be picklable by multiprocessing.
    """
    score_col, label_col, col_id = args

    # Optimization: Pre-calculate total positives for recall denominator
    # This avoids summing the label column 150 times
    total_positives = np.sum(label_col)

    # If no positive labels exist for this class, F1 is 0 by definition
    if total_positives == 0:
        return col_id, {
            "f1": 1.0,
            "th": 0.0,
            "recall": 1.0,
            "precision": 1.0,
            "recall_at_good_precision": 1.0,
        }

    thresholds = np.linspace(0, 1, 1000)
    best_f1 = 0.0
    best_th = 0.0
    best_recall = 0.0
    best_precision = 0.0

    best_recall_at_good_precision = 0.0
    good_precision = 0.8

    # Optimization: We can broadcast the threshold comparison if memory allows,
    # but iterating 150 times with raw numpy is fast enough and memory-safe.
    for th in thresholds:
        # Create binary predictions
        pred_bin = score_col > th

        # Fast Numpy Calculation of TP, FP, FN
        # True Positives: Where both pred and label are 1
        tp = np.sum(pred_bin & (label_col == 1))

        # Predicted Positives (TP + FP)
        pred_pos_count = np.sum(pred_bin)

        # Precision = TP / (TP + FP)
        # Recall = TP / (TP + FN) -> TP / total_positives

        if pred_pos_count == 0:
            prec = 0.0
        else:
            prec = tp / pred_pos_count

        rec = tp / total_positives

        if prec + rec > 0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            best_recall = rec
            best_precision = prec

        if prec >= good_precision and rec > best_recall_at_good_precision:
            best_recall_at_good_precision = rec

    result_dict = {
        "f1": best_f1,
        "th": best_th,
        "recall": best_recall,
        "precision": best_precision,
        "recall_at_good_precision": best_recall_at_good_precision,
    }

    return col_id, result_dict


def find_fmax_per_col_parallel(
    scores_matrix: np.ndarray,
    labels_matrix: np.ndarray,
    col_ids: list,
    n_jobs=12,
) -> dict:

    if scores_matrix.shape != labels_matrix.shape:
        raise ValueError("Scores and labels matrices must have the same shape.")

    n_samples, n_labels = scores_matrix.shape

    print("scores_matrix.shape", scores_matrix.shape)
    print("labels_matrix.shape", labels_matrix.shape)
    print("col_ids", col_ids)

    if len(col_ids) != n_labels:
        raise ValueError("Column IDs must match number of labels.")

    # print(f"Starting parallel processing on {n_jobs} cores...")

    # Prepare arguments as tuples for the worker function
    # We transpose (.T) to iterate over columns easily
    tasks = zip(scores_matrix.T, labels_matrix.T, col_ids)

    fmax_per_col = {}
    recalls_at_good_precisions = {}
    recalls = {}
    precisions = {}
    best_thresholds = {}

    # Use multiprocessing Pool
    with Pool(processes=n_jobs) as pool:
        # imap_unordered is often faster as it yields results as soon as they finish
        # We wrap it in tqdm for a progress bar
        results = list(
            tqdm(pool.imap(_process_single_column_fmax, tasks), total=n_labels)
        )

    # Unpack results
    print("Number of results", len(results))
    for r in results:
        col_id = r[0]
        result_dict = r[1]
        fmax_per_col[col_id] = result_dict["f1"]
        best_thresholds[col_id] = result_dict["th"]
        recalls_at_good_precisions[col_id] = result_dict["recall_at_good_precision"]
        recalls[col_id] = result_dict["recall"]
        precisions[col_id] = result_dict["precision"]

        # print(f"Best threshold for {col_id}: {th} (F1: {f1})")

    return (
        fmax_per_col,
        recalls_at_good_precisions,
        recalls,
        precisions,
        best_thresholds,
    )


def filter_entities_by_score(entities_with_scores: list, threshold: float) -> list:
    """
    Filters entities based on their score.

    Args:
        entities_with_scores: List of tuples (entity, score)
        threshold: Minimum score to keep the entity

    Returns:
        List of entities that meet the threshold
    """
    return [e for e, s in entities_with_scores if s >= threshold]


def normalized_similarity_simple(str1: str, str2: str, field_name: str = None):
    str1_norm = str1
    str2_norm = basic_norm(str2, field_name)
    if "nome" in field_name or "pessoa" in field_name:
        sim = fuzz.partial_ratio(str2_norm, str1_norm) / 100.0
        th = NAME_SIM_TH
    else:
        sim1 = normalized_similarity1(str1_norm, str2_norm)
        sim2 = normalized_similarity2(str1_norm, str2_norm)
        sim = (sim1 + sim2) / 2
        th = SIM_TH
    if "numero" in field_name:
        th = NUMBER_TH

    equals = sim >= th
    if abs(sim - th) < 0.04:
        if not (str1_norm, str2_norm, field_name) in diffs_printed:
            diffs_printed.add((str1_norm, str2_norm, field_name))
            print(
                f"{field_name} | {str1_norm} | {str2} -> {str2_norm} | {sim} | {equals}"
            )

    return sim, equals


def normalized_similarity_with_fixes(str1: str, str2: str, field_name: str = None):
    str1_norm = str1
    str2_norm = basic_norm_with_abbrevs(str2, field_name)
    field_name_safe = field_name.lower() if field_name else ""

    # 1. Base Similarity (Preserving your existing 0.82 F1 logic)
    if "nome" in field_name_safe or "pessoa" in field_name_safe:
        sim1 = normalized_similarity1(str1_norm, str2_norm)
        sim2 = normalized_similarity2(str1_norm, str2_norm)
        sim = (sim1 + sim2) / 2
        th = NAME_SIM_TH
    else:
        sim1 = normalized_similarity1(str1_norm, str2_norm)
        sim2 = normalized_similarity2(str1_norm, str2_norm)
        sim = (sim1 + sim2) / 2
        th = SIM_TH

    if "numero" in field_name_safe:
        th = NUMBER_TH

    # 2. DETERMINISTIC TOKEN GUARDRAILS
    # Strip common Portuguese stop words to prevent matching on prepositions alone
    stopwords = {
        "de",
        "da",
        "do",
        "das",
        "dos",
        "a",
        "o",
        "e",
        "em",
        "no",
        "na",
        "um",
        "uma",
        "com",
        "ao",
    }

    tokens1 = set(str1_norm.split())
    tokens2 = set(str2_norm.split())
    t1_clean = tokens1 - stopwords
    t2_clean = tokens2 - stopwords

    shared_tokens = t1_clean.intersection(t2_clean)

    # GUARD A: The Containment Fix (Locations, Addresses, Landmarks)
    if not ("nome" in field_name_safe or "pessoa" in field_name_safe):
        if len(t1_clean) > 0 and len(t2_clean) > 0:

            # If the core words of one are completely inside the other
            # Solves: 'sqsw 304' inside 'sqsw 304 bloco b', or 'restaurante salitre' inside 'perto do...'
            if t1_clean.issubset(t2_clean) or t2_clean.issubset(t1_clean):
                sim = max(sim, th + 0.15)  # Force True

            # If they share the vast majority of words, but aren't perfect subsets
            # Solves: 'av. b q' vs 'avenida b q' (Shares 'b' and 'q', which is 66% of the shortest string)
            elif len(shared_tokens) > 0:
                shortest_len = min(len(t1_clean), len(t2_clean))
                if (len(shared_tokens) / shortest_len) >= 0.65:
                    sim = max(sim, th + 0.05)  # Nudge slightly over threshold

    # GUARD B: The Hallucination Fix (Persons)
    if "nome" in field_name_safe or "pessoa" in field_name_safe:

        # Solves: 'homem caido' vs 'desconhecido', or 'menina' vs 'hadassa almeida'
        if len(shared_tokens) == 0:
            # Zero shared words. Squash the score to explicitly block Jaro-Winkler False Positives.
            sim = min(sim, th - 0.15)

        # Solves: 'claudio (vizinho)' vs 'claudio da silva', or 'rapaz (vitima)' vs 'rapaz estirado'
        elif len(shared_tokens) > 0:
            # They share at least one valid name/word. If base similarity is close, push it over.
            if sim > (th - 0.10):
                sim = max(sim, th + 0.05)

    equals = sim >= th

    # Logging logic preserved perfectly
    if abs(sim - th) < 0.06 or (abs(sim - th) < 0.3 and sim < th):
        if not (str1_norm, str2_norm, field_name) in diffs_printed:
            diffs_printed.add((str1_norm, str2_norm, field_name))
            print(
                f"{field_name} | {str1_norm} | {str2} -> {str2_norm} | {sim:.4f} | {equals}"
            )

    return sim, equals


def normalized_similarity_custom(
    str1: str, str2: str, field_name: str = None, use_simple=False
):
    sim_func = (
        normalized_similarity_simple if use_simple else normalized_similarity_with_fixes
    )
    return sim_func(str1, str2, field_name)


def compare_values(
    pred_entities: List[str],
    true_entities: List[str],
    field_name: str = None,
    use_simple=False,
) -> bool:
    """
    Compares predicted entities with true entities.

    Args:
        pred_entities: List of predicted entities
        true_entities: List of true entities

    Returns:
        True if predicted entities match true entities, False otherwise
    """

    pred_len = sum([len(x) for x in pred_entities])
    true_len = sum([len(x) for x in true_entities])

    if true_len == 0:
        verbosity_ratio = None
    else:
        verbosity_ratio = pred_len / true_len

    str_norm_func = basic_norm if use_simple else basic_norm_with_abbrevs

    true_entities_norm = [str_norm_func(t, field_name) for t in true_entities]
    true_entities_norm = [t for t in true_entities_norm if t != ""]

    if len(true_entities_norm) == 0 and len(pred_entities) == 0:
        return 1.0, 0, 0, 0, verbosity_ratio  # sim, fp, tp, fn
    elif len(true_entities_norm) == 0 and len(pred_entities) > 0:
        return 0.0, len(pred_entities), 0, 0, verbosity_ratio  # sim, fp, tp, fn
    elif len(true_entities_norm) > 0 and len(pred_entities) == 0:
        return 0.0, 0, 0, len(true_entities_norm), verbosity_ratio  # sim, fp, tp, fn
    else:
        best_matches = []
        true_with_equals = []
        true_without_equals = []

        preds_no_match = set(pred_entities)

        for true_entity in true_entities_norm:
            best_score = 0
            best_match = None
            equal_found = False
            for pred_entity in pred_entities:
                score, equals = normalized_similarity_custom(
                    true_entity, pred_entity, field_name, use_simple=use_simple
                )
                if score > best_score:
                    best_score = score
                    best_match = pred_entity
                    equal_found = equals
            if best_match in preds_no_match:
                preds_no_match.remove(best_match)
            best_matches.append(best_score)
            if equal_found:
                true_with_equals.append(true_entity)
            else:
                true_without_equals.append(true_entity)

        fp = len(preds_no_match)
        tp = len(true_with_equals)
        fn = len(true_without_equals)

        return np.mean(best_matches), fp, tp, fn, verbosity_ratio


def find_max_jw_sim(pred_values, true_values, field_name: str = None, use_simple=False):
    thresholds = np.linspace(0, 1, 150)
    best_threshold = 0
    best_sim = 0
    th_results = []
    for th in thresholds:
        filtered_pred_values = [
            filter_entities_by_score(pred_line, th) for pred_line in pred_values
        ]
        metrics = [
            compare_values(pred, t, field_name, use_simple=use_simple)
            for pred, t in zip(filtered_pred_values, true_values)
        ]
        sims = [m[0] for m in metrics]
        fp = sum([m[1] for m in metrics])
        tp = sum([m[2] for m in metrics])
        fn = sum([m[3] for m in metrics])
        vrs = [m[4] for m in metrics if m[4] is not None]

        # avoid float division by zero
        if tp + fp == 0:
            precision = 1.0
        else:
            precision = tp / (tp + fp)

        if tp + fn == 0:
            recall = 1.0
        else:
            recall = tp / (tp + fn)

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        sim = np.mean(sims)
        if len(vrs) > 0:
            vr = np.mean(vrs)
        else:
            vr = np.nan

        th_results.append((th, sim, fp, tp, fn, precision, recall, f1, vr))

    th_results_by_f1 = sorted(th_results, key=lambda x: x[7], reverse=True)
    fmax = th_results_by_f1[0][7]
    th_results_by_jw_sim = sorted(th_results, key=lambda x: x[1], reverse=True)
    jw_sim_max = th_results_by_jw_sim[0][1]
    verbosity_ratio = th_results_by_jw_sim[0][8]

    recall_at_fmax = th_results_by_f1[0][6]
    precision_at_fmax = th_results_by_f1[0][5]
    recall_at_jw_sim = th_results_by_jw_sim[0][6]
    precision_at_jw_sim = th_results_by_jw_sim[0][5]

    recall = max(recall_at_fmax, recall_at_jw_sim)
    precision = max(precision_at_fmax, precision_at_jw_sim)

    return fmax, jw_sim_max, recall, precision, verbosity_ratio


answer_to_value = {
    "Certamente Sim": 1.0,
    "Sim": 0.95,
    "Provavelmente Sim": 0.65,
    "Não sei": 0.5,
    "Provavelmente Não": 0.35,
    "Não": 0.05,
    "Certamente Não": 0.0,
}


def get_testing_inputs(usage_perc=USE_PERC):
    text1 = """Preciso de uma ambulância rápido, tem uma pessoa caída na Rua das Flores, número 123, perto da padaria. 
        Ela está inconsciente. Meu nome é Maria Oliveira. Meu telefone é 99999-8888, moro na cidade de São Paulo. """
    text2 = """Preciso de uma ambulância rápido, tem uma pessoa caída na Rua das Flores, número 123, 
        perto da padaria. Ela está inconsciente. Meu nome é Maria Oliveira. Meu telefone é 99999-8888, 
        moro na cidade de São Paulo. Ela não tem risco de morte! Nenhum!"""
    text3 = """Preciso de uma ambulância rápido, tem uma pessoa caída na Rua das Flores, número 123, perto da padaria. 
    Ela está inconsciente. Foi o marido que bateu nela! Aquele covarde! Meu nome é Maria Oliveira. 
    Meu telefone é 99999-8888, moro na cidade de São Paulo. Ela não tem risco de morte! Nenhum!"""
    text4 = """Preciso de uma ambulância rápido, tem uma pessoa caída na Rua das Flores, número 123, perto da padaria. 
    Foi o marido que bateu nela! Aquele covarde! Meu nome é Maria Oliveira. 
    Meu telefone é 99999-8888, moro na cidade de São Paulo. Sim, o marido violento espancou ela!."""
    text5 = """Socorro, preciso de uma viatura rápido! Estão atirando em mim! Quando? Agora mesmo! 
        Ele tem uma pistola! Meu nome é Maria Oliveira. Meu telefone é 99999-3888, moro na cidade de Natal. 
        Repito: o Claudio começou a atirar em mim com uma pistola, do nada"""
    text6 = """Quero avisar de um risco de tumulto. Meu nome é Maria Oliveira. Meu telefone é 99999-3888, 
    moro na cidade de Natal. Há uma manifestação em frente ao shopping. Há muitas pessoas e veículos.
    Parece que vai começar uma briga. """
    text7 = """Alô? É da ambulancia? Meu pai levou um tiro na rua. Ele chegou em casa sangrando e disse que foi tiro. 
    Não, ele não sabe quem foi. Aconteceu a mais ou menos 1 hora. Meu nome é Pedro. Meu telefone é 99999-3888, 
    moro na cidade de Natal. Meu pai é o Claudio. ele disse que foi na rua salgado filho, número 123, perto do shopping.
    Mas nós moramos em outro lugar. O endereço da nossa casa? É rua capitão mor golveia, número 450.
    """
    all_texts = [text1, text2, text3, text4, text5, text6, text7]

    fatoOcorrendoNesteMomento = "Fato Ocorrendo Neste Momento ?"
    autorDoFatoNoLocal = "Autor Do Fato No Local ?"
    autorDoFatoArmado = "Autor Do Fato Armado ?"
    feridosComRiscoDeMorte = "Feridos Com Risco de Morte ?"
    riscoDeTumulto = "Risco De Tumulto ?"
    leiMariaPenha = "Lei Maria da Penha ?"

    clfnames = [
        fatoOcorrendoNesteMomento,
        autorDoFatoNoLocal,
        autorDoFatoArmado,
        feridosComRiscoDeMorte,
        riscoDeTumulto,
        leiMariaPenha,
    ]
    clfnames_hf = [
        "fatoOcorrendoNesteMomento",
        "autorDoFatoNoLocal",
        "autorDoFatoArmado",
        "feridosComRiscoDeMorte",
        "riscoDeTumulto",
        "leiMariaPenha",
    ]

    correct_clfs = [
        {
            fatoOcorrendoNesteMomento: True,
            autorDoFatoNoLocal: False,
            autorDoFatoArmado: False,
            feridosComRiscoDeMorte: True,
            riscoDeTumulto: False,
            leiMariaPenha: False,
        },
        {
            fatoOcorrendoNesteMomento: True,
            autorDoFatoNoLocal: False,
            autorDoFatoArmado: False,
            feridosComRiscoDeMorte: False,
            riscoDeTumulto: False,
            leiMariaPenha: False,
        },
        {
            fatoOcorrendoNesteMomento: True,
            autorDoFatoNoLocal: False,
            autorDoFatoArmado: False,
            feridosComRiscoDeMorte: False,
            riscoDeTumulto: False,
            leiMariaPenha: True,
        },
        {
            fatoOcorrendoNesteMomento: True,
            autorDoFatoNoLocal: False,
            autorDoFatoArmado: False,
            feridosComRiscoDeMorte: True,
            riscoDeTumulto: False,
            leiMariaPenha: True,
        },
        {
            fatoOcorrendoNesteMomento: True,
            autorDoFatoNoLocal: True,
            autorDoFatoArmado: True,
            feridosComRiscoDeMorte: True,
            riscoDeTumulto: True,
            leiMariaPenha: False,
        },
        {
            fatoOcorrendoNesteMomento: True,
            autorDoFatoNoLocal: True,
            autorDoFatoArmado: False,
            feridosComRiscoDeMorte: False,
            riscoDeTumulto: True,
            leiMariaPenha: False,
        },
        {
            fatoOcorrendoNesteMomento: False,
            autorDoFatoNoLocal: False,
            autorDoFatoArmado: True,
            feridosComRiscoDeMorte: True,
            riscoDeTumulto: False,
            leiMariaPenha: False,
        },
    ]

    texts_hf = []
    correct_clfs_hf = []
    true_entities_hf = []

    hf_df = pl.read_parquet("input/dataset_filtrado.parquet")
    for row in hf_df.iter_rows(named=True):
        roteiro_str = "\n".join([" - ".join(p) for p in row["roteiro_segmentado"]])
        texts_hf.append(roteiro_str)
        clfs_line = np.array([int(row[c]) for c in clfnames_hf])
        correct_clfs_hf.append(clfs_line)

        pessoas = []
        for part in row["participacoes"]:
            new_person = part["pessoa"].replace("pessoa:", "").strip()
            if len(new_person) > 1:
                pessoas.append(new_person)
        true_entities = {"pessoa": pessoas}

        name_pairs = [
            ("nome_solicitante", "nome_do_solicitante"),
            ("rua", "rua_ou_logradouro"),
            ("complemento", "complemento"),
            ("numero", "numero"),
            ("bairro", "bairro"),
            ("cidade", "cidade"),
            ("estado", "estado"),
            ("ponto_de_referencia", "ponto_de_referencia"),
        ]

        for df_col, true_entity_name in name_pairs:
            true_entities[true_entity_name] = []
            if type(row[df_col]) == str:
                if len(row[df_col]) > 0:
                    true_entities[true_entity_name].append(row[df_col])
        # print(true_entities)
        true_entities_hf.append(true_entities)

    """for txt, clfs in zip(all_texts, correct_clfs):
        texts_hf.append(txt)
        correct_clfs_hf.append([int(clfs[c]) for c in clfnames])"""

    correct_clfs_hf = np.asarray(correct_clfs_hf)

    """"natureza_da_ocorrencia": {
        "Fato Ocorrendo Neste Momento": {"desc": "Se o fato relatado está ocorrendo neste momento", 
            "sim": "Fato Ocorrendo Neste Momento", "nao": "Não Ha Fato Ocorrendo Neste Momento"}, 
        "Autor Do Fato No Local": {"desc": "Se o autor (culpado/acusado) do fato está no local", 
            "sim": "Autor Do Fato No Local", "nao": "Não Ha Autor Do Fato No Local"}, 
        "Autor Do Fato Armado": {"desc": "Se o autor (culpado/acusado/suspeito) do fato estava armado", 
            "sim": "Autor Do Fato Armado", "nao": "Não Ha Autor Do Fato Armado"}, 
        "Feridos Com Risco de Morte": {"desc": "Se a ocorrência envolve feridos com risco de morte",
            "sim": "Feridos Com Risco de Morte", "nao": "Não Ha Risco de Morte"}, 
        "Risco De Tumulto": {"desc": "Se a ocorrência envolve um risco de tumulto", 
            "sim": "Risco De Tumulto", "nao": "Não Ha Risco De Tumulto"}, 
        "Lei Maria da Penha": {
            "desc": "Se a ocorrência se enquadra como um caso de lei maria da penha, a qual trata sobre a violência doméstica e conjugal", 
            "sim": "É um caso de Maria da Penha", "nao": "Não é um caso de Maria da Penha"
        },
    }"""

    clfs_bool = {
        fatoOcorrendoNesteMomento: {
            "desc": "Se o fato relatado está ocorrendo neste momento",
            "sim": "Sim",
            "nao": "Não",
        },
        autorDoFatoNoLocal: {
            "desc": "Se o autor (culpado/acusado) do fato está no local",
            "sim": "Sim",
            "nao": "Não",
        },
        autorDoFatoArmado: {
            "desc": "Se o autor (culpado/acusado/suspeito) do fato estava armado",
            "sim": "Sim",
            "nao": "Não",
        },
        feridosComRiscoDeMorte: {
            "desc": "Se a ocorrência envolve feridos com risco de morte",
            "sim": "Sim",
            "nao": "Não",
        },
        riscoDeTumulto: {
            "desc": "Se a ocorrência envolve um risco de tumulto",
            "sim": "Sim",
            "nao": "Não",
        },
        leiMariaPenha: {
            "desc": "Se a ocorrência se enquadra como um caso de lei maria da penha, a qual trata sobre a violência doméstica e conjugal",
            "sim": "Sim",
            "nao": "Não",
        },
    }

    clfs_multi = {
        label: {**details, "answers": list(answer_to_value.keys())}
        for label, details in clfs_bool.items()
    }

    clfs_named = {
        fatoOcorrendoNesteMomento: {
            "desc": "Se o fato relatado está ocorrendo neste momento",
            "sim": "Fato Ocorrendo Neste Momento",
            "nao": "Não Ha Fato Ocorrendo Neste Momento",
        },
        autorDoFatoNoLocal: {
            "desc": "Se o autor (culpado/acusado) do fato está no local",
            "sim": "Autor Do Fato No Local",
            "nao": "Não Ha Autor Do Fato No Local",
        },
        autorDoFatoArmado: {
            "desc": "Se o autor (culpado/acusado/suspeito) do fato estava armado",
            "sim": "Autor Do Fato Armado",
            "nao": "Não Ha Autor Do Fato Armado",
        },
        feridosComRiscoDeMorte: {
            "desc": "Se a ocorrência envolve feridos com risco de morte",
            "sim": "Feridos Com Risco de Morte",
            "nao": "Não Ha Risco de Morte",
        },
        riscoDeTumulto: {
            "desc": "Se a ocorrência envolve um risco de tumulto",
            "sim": "Risco De Tumulto",
            "nao": "Não Ha Risco De Tumulto",
        },
        leiMariaPenha: {
            "desc": "Se a ocorrência se enquadra como um caso de lei maria da penha, a qual trata sobre a violência doméstica e conjugal",
            "sim": "É um caso de Maria da Penha",
            "nao": "Não é um caso de Maria da Penha",
        },
    }

    full_schema_dict = {
        "entities": {
            "rua_ou_logradouro": "Nome de rua, logradouro ou avenida",
            "rua": "Nome de rua",
            "bairro": "Nome de bairro",
            "municipio": "Nome de município",
            "cidade": "Nome de cidade",
            "ponto_de_referencia": "Nome de ponto de referência do endereço",
            "nome_do_solicitante": "Nome do solicitante (pessoa que está fazendo o chamado)",
            "pessoa": "Nome de pessoa ou participante",
            "numero": "Número do endereço",
            "street_number": "Street Number",
            "number": "Número",
            "complemento": "Complemento do endereço (Exemplos: apt 301, casa A, etc)",
            "endereço_complemento": "Número ou código da casa, apartamento ou loja naquele endereço",
        },
        "boolean": {"natureza_da_ocorrencia": clfs_bool},
    }

    rac_perc = 1.0 - usage_perc
    # set default seed:
    np.random.seed(1337)
    rac_indexes = np.random.choice(
        len(texts_hf), int(len(texts_hf) * rac_perc), replace=False
    )

    rac_texts = [texts_hf[i] for i in rac_indexes]
    rac_clfs_true = np.asarray([correct_clfs_hf[i] for i in rac_indexes])
    print("rac_indexes", rac_indexes)

    pred_indexes = [i for i in range(len(texts_hf)) if i not in rac_indexes]

    print("pred_indexes", pred_indexes)

    texts_hf = [texts_hf[i] for i in pred_indexes]
    correct_clfs_hf = np.asarray([correct_clfs_hf[i] for i in pred_indexes])
    true_entities_hf = [true_entities_hf[i] for i in pred_indexes]

    examples = []
    for txt, clfs in zip(rac_texts, rac_clfs_true):
        all_clfs = [c.replace(" ?", "") for c in clfnames]
        true_examples = [c for n, c in enumerate(all_clfs) if int(clfs[n]) > 0]
        # print(true_examples)
        examples.append(
            {"text": txt, "true_labels": true_examples, "all_labels": all_clfs}
        )

    ml_categories = list(full_schema_dict["boolean"]["natureza_da_ocorrencia"].keys())

    return (
        ml_categories,
        texts_hf,
        clfnames,
        correct_clfs_hf,
        full_schema_dict,
        true_entities_hf,
    )


def calc_metadata(input_strs, output_strs, gpu_usage_secs):
    input_tokens = sum([len(s.split()) for s in input_strs])
    output_tokens = sum([len(s.split()) for s in output_strs])
    return input_tokens / gpu_usage_secs, output_tokens / gpu_usage_secs
