import argparse
import json
import os
import re
import unicodedata
from typing import List
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_EVAL_MODEL = "google/embeddinggemma-2"
TMP_CACHE_DIR = "/tmp"


def normalize_text(text: str) -> str:
    """
    Normaliza uma string convertendo para minúsculas, removendo acentos/diacríticos,
    substituindo barras e travessões/hífens por espaço e colapsando espaços múltiplos.
    """
    if not text or not isinstance(text, str):
        return ""
    # Converte para minúsculas
    val = text.strip().lower()
    # Remove acentos e marcas diacríticas (NFKD)
    val = "".join(
        ch for ch in unicodedata.normalize("NFKD", val)
        if not unicodedata.combining(ch)
    )
    # Substitui barras (/ e \) e travessões/hífens (-, –, —, ―, −) por espaço
    val = re.sub(r"[/\\\-–—―−]+", " ", val)
    # Remove caracteres de controle/pontuação redundante mantendo estrutura básica e colapsa espaços
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in val.splitlines()]
    return "\n".join(line for line in lines if line)


def add_field_context(field_name: str, value: str) -> str:
    """Adiciona o nome do campo como prefixo ('campo: valor') e normaliza a string resultante."""
    val = (value or "").strip()
    if not val:
        return ""
    prefix = f"{field_name}: "
    raw = val if val.lower().startswith(prefix.lower()) else f"{prefix}{val}"
    return normalize_text(raw)


def format_descricao(inferences: dict) -> str:
    """Combina descricao_breve e outras_observacoes já contextualizadas com os nomes dos campos."""
    desc = (inferences.get("descricao_breve") or "").strip()
    obs = (inferences.get("outras_observacoes") or "").strip()
    parts = []
    if desc:
        parts.append(add_field_context("descricao_breve", desc))
    if obs:
        parts.append(add_field_context("outras_observacoes", obs))
    return "\n".join(parts)


def format_participacao_item(item: dict) -> str:
    pessoa = (item.get("pessoa") or "").strip()
    envolvimento = (item.get("envolvimento") or "").strip()
    raw = f"{pessoa} ({envolvimento})".strip()
    return add_field_context("envolvimentos", raw) if raw and raw != "()" else ""


def format_envolvimento_item(item: dict) -> str:
    pessoa = (item.get("nome_pessoa") or "").strip()
    participacao = (item.get("participacao") or "").strip()
    raw = f"{pessoa} ({participacao})".strip()
    return add_field_context("envolvimentos", raw) if raw and raw != "()" else ""


def get_or_create_parquet(json_path: str) -> pd.DataFrame:
    """
    Carrega a versão tabulada (.parquet) do arquivo de avaliação caso já exista
    com o mesmo nome base; caso contrário, faz o parsing do JSON uma única vez
    e salva o arquivo .parquet correspondente.
    """
    base_path, ext = os.path.splitext(json_path)
    parquet_path = json_path if ext.lower() == ".parquet" else f"{base_path}.parquet"

    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)

    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    parsed_rows = []
    for idx, rec in enumerate(records):
        inf = rec.get("inferences") or {}
        pipeline_name = rec.get("pipeline_name") or "unknown_pipeline"
        emb_model_used = rec.get("embedding_model") or "unknown_embedding_model"

        roteiro_raw = (rec.get("roteiro") or "").strip()
        nat_inicial_raw = (rec.get("natureza_inicial") or "").strip()
        decisiva_raw = (inf.get("classificacao_decisiva") or "").strip()
        b25_w = inf.get("b25_w", 0.0)

        provaveis_raw = [
            item[1].strip()
            for item in (inf.get("classificacoes_provaveis") or [])
            if len(item) > 1 and item[1]
        ]
        provavel_top1_raw = provaveis_raw[0] if provaveis_raw else ""

        gt_part_items = [
            format_participacao_item(p) for p in (rec.get("participacoes") or []) if p
        ]
        gt_part_items = [x for x in gt_part_items if x]

        pred_env_items = [
            format_envolvimento_item(e) for e in (inf.get("envolvimentos") or []) if e
        ]
        pred_env_items = [x for x in pred_env_items if x]

        gt_part_concat_raw = "; ".join(
            f"{(p.get('pessoa') or '').strip()} ({(p.get('envolvimento') or '').strip()})".strip()
            for p in (rec.get("participacoes") or [])
            if p
        )
        pred_env_concat_raw = "; ".join(
            f"{(e.get('nome_pessoa') or '').strip()} ({(e.get('participacao') or '').strip()})".strip()
            for e in (inf.get("envolvimentos") or [])
            if e
        )
        decision_tokens = None
        decision_metadata = inf.get("decision_metadata") or {}

        if "meta" in decision_metadata:
            decision_tokens = decision_metadata["meta"]["input_tokens"] * decision_metadata["meta"]["output_tokens"]
        
        if "top" in pipeline_name:
            top_k = int(pipeline_name.split("top")[-1])
        else:
            top_k = 16

        model_name_parsed = emb_model_used.split('/')[-1].replace("sentence-encoder-ir", "")
        model_name_parsed = f"{model_name_parsed} - Top {top_k}"

        parsed_rows.append({
            "row_id": idx,
            "ID": rec.get("ID") or str(idx),
            "pipeline_name": pipeline_name,
            "model": rec.get("model") or "",
            "embedding_model": model_name_parsed,
            "pipeline_embedding": f"{pipeline_name} + {emb_model_used}",
            "filtering_seconds": inf.get("filtering_seconds", None),
            "decision_seconds": inf.get("decision_seconds", None),
            "decision_tokens": decision_tokens,
            "processing_seconds": inf.get("processing_seconds", None),
            # Strings contextualizadas ("campo: valor") e normalizadas
            "roteiro": add_field_context("roteiro", roteiro_raw),
            "descricao_pred": format_descricao(inf),
            "natureza_inicial_true": add_field_context("natureza_inicial", nat_inicial_raw),
            "natureza_inicial_pred_def": add_field_context("natureza_inicial", decisiva_raw),
            "natureza_inicial_pred_raw": add_field_context("natureza_inicial", provavel_top1_raw),
            "natureza_inicial_pred_all": [
                add_field_context("natureza_inicial", c) for c in provaveis_raw
            ],
            "envolvimentos_true_concat": add_field_context("envolvimentos", gt_part_concat_raw),
            "envolvimentos_pred_concat": add_field_context("envolvimentos", pred_env_concat_raw),
            "envolvimentos_true_items": gt_part_items,
            "envolvimentos_pred_items": pred_env_items,
            "b25_w": b25_w,
        })

    df_parsed = pd.DataFrame(parsed_rows)
    df_parsed.to_parquet(parquet_path, index=False)
    return df_parsed


def get_embedding_cache_path(eval_model_name: str, cache_dir: str = TMP_CACHE_DIR) -> str:
    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", eval_model_name.strip())
    return os.path.join(cache_dir, f"embeddings_cache_{safe_name}.parquet")


def load_and_update_embeddings(
    unique_texts: set,
    eval_model_name: str = DEFAULT_EVAL_MODEL,
    cache_dir: str = TMP_CACHE_DIR,
) -> dict:
    """
    Carrega o cache de embeddings de /tmp para o modelo avaliador, normaliza os textos
    antes do cálculo dos embeddings, calcula apenas as strings inéditas e persiste
    o cache atualizado no /tmp.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = get_embedding_cache_path(eval_model_name, cache_dir)

    emb_map = {}
    if os.path.exists(cache_path):
        try:
            df_cache = pd.read_parquet(cache_path)
            for text, emb in zip(df_cache["text"], df_cache["embedding"]):
                norm_key = normalize_text(text)
                if norm_key:
                    emb_map[norm_key] = np.asarray(emb, dtype=np.float32)
        except Exception as e:
            print(f"Aviso: falha ao ler cache {cache_path} ({e}). Recriando cache.")
            emb_map = {}

    normalized_unique = {normalize_text(t) for t in unique_texts if t}
    missing_texts = sorted(t for t in normalized_unique if t and t not in emb_map)
    if missing_texts:
        model = SentenceTransformer(eval_model_name)
        new_embeddings = model.encode(
            missing_texts,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        for text, emb in zip(missing_texts, new_embeddings):
            emb_map[text] = np.asarray(emb, dtype=np.float32)

        df_updated = pd.DataFrame({
            "text": list(emb_map.keys()),
            "embedding": [vec.tolist() for vec in emb_map.values()],
        })
        df_updated.to_parquet(cache_path, index=False)

    return emb_map


def text_similarity(text_a: str, text_b: str, emb_map: dict) -> float:
    norm_a = normalize_text(text_a)
    norm_b = normalize_text(text_b)
    if not norm_a or not norm_b:
        return 0.0
    vec_a = emb_map[norm_a].reshape(1, -1)
    vec_b = emb_map[norm_b].reshape(1, -1)
    return float(cosine_similarity(vec_a, vec_b)[0, 0])


def max_option_similarity(target_text: str, candidate_options: list, emb_map: dict) -> float:
    """
    Quando há várias opções contra as quais avaliar uma string, retorna a
    similaridade com a opção mais parecida (máxima similaridade).
    """
    norm_target = normalize_text(target_text)
    options = [normalize_text(c) for c in (candidate_options or []) if c]
    options = [c for c in options if c]
    if not norm_target or not options:
        return 0.0
    target_vec = emb_map[norm_target].reshape(1, -1)
    option_vecs = np.vstack([emb_map[c] for c in options])
    sims = cosine_similarity(target_vec, option_vecs)[0]
    return float(np.max(sims))


def compute_column_sim_single(gt_col: list, pred_col: list, emb_map: dict) -> list:
    scores = []
    for a, b in zip(gt_col, pred_col):
        score = text_similarity(a, b, emb_map)
        scores.append(score)
    return scores


def compute_column_sim_multi_options(gt_col: list, pred_options_col: list, emb_map: dict) -> list:
    scores = []
    for idx, gt_i in enumerate(gt_col):
        options_i = list(pred_options_col[idx]) if pred_options_col[idx] is not None else []
        score = max_option_similarity(gt_i, options_i, emb_map)
        scores.append(score)
    return scores


def compute_column_sim_sets(gt_sets_col: list, pred_sets_col: list, emb_map: dict) -> list:
    scores = []
    for i in range(len(gt_sets_col)):
        gt_items_i = list(gt_sets_col[i]) if gt_sets_col[i] is not None else []
        scores_local = []
        for g_item in gt_items_i:
            pred_items_j = list(pred_sets_col[i]) if pred_sets_col[i] is not None else []
            local_max = max_option_similarity(g_item, pred_items_j, emb_map)
            scores_local.append(local_max)
        scores.append(safe_mean(scores_local))
    return scores


def collect_unique_texts(df_parsed: pd.DataFrame) -> set:
    unique_texts = set()
    single_cols = [
        "roteiro",
        "descricao_pred",
        "natureza_inicial_true",
        "natureza_inicial_pred_def",
        "natureza_inicial_pred_raw",
        "envolvimentos_true_concat",
        "envolvimentos_pred_concat",
    ]
    for col in single_cols:
        for val in df_parsed[col]:
            if val:
                norm_val = normalize_text(val)
                if norm_val:
                    unique_texts.add(norm_val)

    list_cols = [
        "natureza_inicial_pred_all",
        "envolvimentos_true_items",
        "envolvimentos_pred_items",
    ]
    for col in list_cols:
        for seq in df_parsed[col]:
            if seq is not None:
                for item in seq:
                    if item:
                        norm_item = normalize_text(item)
                        if norm_item:
                            unique_texts.add(norm_item)

    return unique_texts


def safe_mean(values: list) -> float:
    valid = [float(v) for v in values if v is not None and not pd.isna(v)]
    return float(np.mean(valid)) if valid else float("nan")


def safe_std(values: list) -> float:
    valid = [float(v) for v in values if v is not None and not pd.isna(v)]
    return float(np.std(valid)) if valid else float("nan")


def find_verysimilar(original: str, options: List[str], emb_map: dict, threshold: float = 0.93) -> bool:
    norm_orig = normalize_text(original)
    norm_options = [normalize_text(x) for x in options if x]
    norm_options = [x for x in norm_options if x]

    if norm_orig in norm_options:
        return True, 0.0, norm_orig
    elif len(norm_options) > 0:
        sims = [text_similarity(norm_orig, x, emb_map) for x in norm_options]
        idx_max = int(np.argmax(sims))
        max_sim = sims[idx_max]
        diff = max_sim - threshold
        if diff > 0:
            return True, diff, norm_options[idx_max]
        else:
            return False, diff, norm_options[idx_max]
    else:
        return False, 1.0, None


def evaluate_file(json_path: str, eval_model_name: str = DEFAULT_EVAL_MODEL) -> pd.DataFrame:
    """
    Avalia o arquivo JSON (ou Parquet cacheado) usando um modelo SentenceTransformer
    com cache incremental de embeddings em /tmp e normalização prévia das strings.
    """
    df_parsed = get_or_create_parquet(json_path)
    unique_texts = collect_unique_texts(df_parsed)

    print("Embeddings unique texts")
    emb_map = load_and_update_embeddings(unique_texts, eval_model_name=eval_model_name)

    calcs = [
        ("roteiro", "descricao_pred", "descricao_sim", "single"),
        ("natureza_inicial_true", "natureza_inicial_pred_raw", "natureza_provavel_top1_sim", "single"),
        ("natureza_inicial_true", "natureza_inicial_pred_def", "natureza_decisiva_sim", "single"),
        ("envolvimentos_true_concat", "envolvimentos_pred_concat", "envolvimentos_concat_sim", "single"),
        ("natureza_inicial_true", "natureza_inicial_pred_all", "natureza_provavel_sim", "multi"),
        ("envolvimentos_true_items", "envolvimentos_pred_items", "envolvimentos_items_sim", "sets"),
    ]

    for col_a, col_b, new_col_name, sim_type_type in calcs:
        print("Calculating Semantic Similarity for column " + new_col_name)
        sim_func = compute_column_sim_single
        if sim_type_type == "multi":
            sim_func = compute_column_sim_multi_options
        elif sim_type_type == "sets":
            sim_func = compute_column_sim_sets
        col_a_list = df_parsed[col_a].tolist()
        col_b_list = df_parsed[col_b].tolist()
        sim_col = sim_func(col_a_list, col_b_list, emb_map)

        df_parsed[new_col_name] = sim_col

    nat_found_in_options = []
    iterator = zip(df_parsed["natureza_inicial_true"].tolist(), df_parsed["natureza_inicial_pred_all"].tolist())
    for natureza_inicial_true, options in iterator:
        found, diff, most_similar = find_verysimilar(natureza_inicial_true, options, emb_map)
        nat_found_in_options.append(found)

        #if not found and abs(diff) <= 0.02:
        #    print(f"Case close to threshold: {normalize_text(natureza_inicial_true)} | {most_similar} | {diff}")

    df_parsed["natureza_inicial_pred_has_true"] = nat_found_in_options

    grouped = df_parsed.groupby(
        ["pipeline_name", "embedding_model", "b25_w"],
        sort=False,
    )

    rows = []

    for (pipeline_name, emb_model_used, b25_weight), group_df in grouped:
        true_in_options = group_df["natureza_inicial_pred_has_true"].tolist()
        perc_found = sum(true_in_options) / len(true_in_options)
        rows.append({
            "pipeline_name": pipeline_name,
            "embedding_model": emb_model_used,
            "b25_weight": b25_weight,
            "n_amostras": len(group_df),
            "filtering_seconds": safe_mean(group_df["filtering_seconds"].tolist()),
            "filtering_seconds_std": safe_std(group_df["filtering_seconds"].tolist()),
            "decision_seconds": safe_mean(group_df["decision_seconds"].tolist()),
            "decision_tokens": safe_mean(group_df["decision_tokens"].tolist()),
            "descricao_sim": safe_mean(group_df["descricao_sim"].tolist()),
            "natureza_provavel_top1_sim": safe_mean(group_df["natureza_provavel_top1_sim"].tolist()),
            "natureza_provavel_sim": safe_mean(group_df["natureza_provavel_sim"].tolist()),
            "natureza_inicial_pred_has_true": perc_found,
            "natureza_decisiva_sim": safe_mean(group_df["natureza_decisiva_sim"].tolist()),
            "envolvimentos_concat_sim": safe_mean(group_df["envolvimentos_concat_sim"].tolist()),
            "envolvimentos_items_sim": safe_mean(group_df["envolvimentos_items_sim"].tolist()),
        })

    df = pd.DataFrame(rows)

    natureza_metrics = [
        "natureza_provavel_top1_sim",
        #"natureza_inicial_pred_has_true",
        "natureza_decisiva_sim",
    ]
    weights1 = [1, 3]
    df["Nota - Natureza"] = df[natureza_metrics].dot(weights1) / sum(weights1)

    interp_metrics = [
        "natureza_decisiva_sim",
        "envolvimentos_items_sim",
        "descricao_sim",
    ]
    weights2 = [1, 1, 1]
    df["Nota - Interpretação"] = df[interp_metrics].dot(weights2) / sum(weights2)

    df["Nota Media"] = (df["Nota - Natureza"] + df["Nota - Interpretação"]) / 2
    df.sort_values(by="Nota - Natureza", ascending=False, inplace=True)

    return df, df_parsed


def main():
    parser = argparse.ArgumentParser(
        description="Avalia semanticamente os campos inferidos em relação ao ground truth do JSON."
    )
    parser.add_argument(
        "json_path",
        type=str,
        help="Caminho para o arquivo JSON (ou Parquet) de avaliação.",
    )
    parser.add_argument(
        "--eval-model",
        type=str,
        default=DEFAULT_EVAL_MODEL,
        help=f"Nome do modelo SentenceTransformer avaliador (padrão: {DEFAULT_EVAL_MODEL}).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Caminho opcional para salvar a tabela resultante em CSV.",
    )
    args = parser.parse_args()

    df, df_all = evaluate_file(args.json_path, args.eval_model)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.max_colwidth", None)
    print("\n=== Tabela de Avaliação por (pipeline_name + embedding_model) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if not args.output_csv:
        output_path = args.json_path.replace(".json", "-eval.csv")
    else:
        output_path = args.output_csv

    df.to_csv(output_path, index=False)
    print(f"\nTabela salva em: {output_path}")
    all_outs_path = output_path.replace(".csv", "-all.csv")
    df_all.to_csv(all_outs_path, index=False)
    print(f"Tabela com todos os casos salvos em: {all_outs_path}")

    pretty_col_names = {"pipeline_name": "Versão da Pipeline",
    	"embedding_model": "Modelo de Embedding",
        "b25_weight": "Peso do B25",
        "n_amostras": "Número de Amostras",
        "Nota Media": "Nota Média",
        "Nota - Natureza": "Nota - Classificação da Natureza",
        "Nota - Interpretação": "Nota - Interpretação",
        "filtering_seconds": "Segundos de Filtragem",
        "decision_seconds": "Segundos de Decisão",
        "decision_tokens": "Tokens na Decisão",
        "descricao_sim": "Similaridade da Descrição",
        "natureza_provavel_top1_sim": "Similaridade da Natureza Provável",
        "natureza_inicial_pred_has_true": "Listas de Naturezas Prováveis Contendo Exatamente a Verdadeira (%)",
        "natureza_decisiva_sim": "Similaridade da Natureza Decisiva",
        "envolvimentos_items_sim": "Similaridade dos Envolvimentos"
    }   
    df_pretty = df.rename(columns=pretty_col_names)
    df_pretty = df_pretty[list(pretty_col_names.values())]

    pretty_time_cols = [
        "Segundos de Filtragem",
        "Segundos de Decisão",
    ]
    pretty_pct_cols = [
        "Nota Média",
        "Nota - Classificação da Natureza",
        "Nota - Interpretação",
        "Similaridade da Descrição",
        "Similaridade da Natureza Provável",
        "Similaridade da Natureza Decisiva",
        "Similaridade dos Envolvimentos",
        "Listas de Naturezas Prováveis Contendo Exatamente a Verdadeira (%)",
    ]

    df_pretty[pretty_time_cols] = df_pretty[pretty_time_cols].map(
        lambda x: f"{x:.3f}" if pd.notna(x) else ""
    )
    # 0.898271712 -> 89.83 (ou f"{x:.2%}" caso queira o símbolo '%')
    df_pretty[pretty_pct_cols] = df_pretty[pretty_pct_cols].map(
        lambda x: f"{x * 100:.3f}" if pd.notna(x) else ""
    )

    pretty_output_path = output_path.replace(".csv", "-pretty.csv")
    df_pretty.to_csv(pretty_output_path, index=False)
    print(f"Tabela formatada salva em: {pretty_output_path}")


if __name__ == "__main__":
    main()