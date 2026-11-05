"""
Avaliação do Laya nas seis classificações booleanas das ocorrências.

Usa o mesmo dataset, a mesma separação de amostras (get_testing_inputs) e as
mesmas métricas (find_fmax_per_col_parallel) dos demais testes desta pasta,
e grava o resultado no formato lido por make_df.py. O Laya não extrai
entidades, então só as métricas de classificação são preenchidas.

Além do Fmax (que escolhe o melhor limiar olhando o gabarito e por isso é
otimista), o script calcula as métricas no limiar fixo usado pelo runner do
backend (LAYA_THRESHOLD, padrão 0.5), que é o que o atendente veria.

Uso:
    pip install laya
    python dataset_loading.py            # gera input/dataset_filtrado.parquet
    python test_laya.py results/laya_results.json

    # comparar com resultados já gerados por outros testes
    python test_laya.py results/laya_results.json \
        --compare results/serpro_open_results.json results/gemini_results.json

    # rodada rápida em poucas amostras
    python test_laya.py results/laya_results.json --limit 20
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from testing import USE_PERC, find_fmax_per_col_parallel, get_testing_inputs

# Mesmo contexto usado em backend/src/ner/runners/laya_runner.py, para que o
# teste meça exatamente as perguntas que o runner faz em produção.
CONTEXTO_PERGUNTAS = (
    "O texto é a transcrição automática de uma ligação para um serviço de "
    "emergência (polícia, bombeiros ou atendimento médico). "
)


def build_questions(clfnames, full_schema_dict):
    """
    Uma pergunta "noul" (sim/não) por classificação, com a mesma descrição
    usada no schema dos LLMs. Retorna também o mapa id da pergunta -> nome
    da classificação no gabarito.
    """
    descs = full_schema_dict["boolean"]["natureza_da_ocorrencia"]
    questions = {}
    qid_to_clf = {}
    for i, clf in enumerate(clfnames):
        qid = f"q{i}"
        questions[qid] = {
            "type": "noul",
            "instructions": CONTEXTO_PERGUNTAS + descs[clf]["desc"] + ".",
        }
        qid_to_clf[qid] = clf
    return questions, qid_to_clf


def load_agent(model, subfolder, device, max_len):
    import laya

    kwargs = {}
    if subfolder:
        kwargs["subfolder"] = subfolder
    if device:
        kwargs["device"] = device
    print(f"Carregando {model} (subfolder={subfolder or '-'}, device={device or 'auto'})")
    agent = laya.load(model, **kwargs)
    if max_len:
        agent.cfg["max_len"] = int(max_len)
    return agent


def run_inference(agent, texts, questions, qid_to_clf, clfnames, batch_size):
    """
    Retorna a matriz de probabilidades (amostras x classificações), o total
    de tokens de entrada, o tempo de inferência e a lista de falhas.
    """
    from tqdm import tqdm

    scores = np.zeros((len(texts), len(clfnames)), dtype=float)
    clf_index = {c: i for i, c in enumerate(clfnames)}
    tokens_in = 0
    infer_time = 0.0
    failures = []
    ok = np.ones(len(texts), dtype=bool)

    # Aquecimento, para o tempo da primeira chamada não distorcer a latência.
    agent.predict(texts[0], questions)

    bar = tqdm(total=len(texts))
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        try:
            t0 = time.time()
            outputs = agent.predict_batch(chunk, questions, batch_size=batch_size)
            infer_time += time.time() - t0
        except Exception as err:
            for j in range(len(chunk)):
                ok[start + j] = False
                failures.append(f"amostra {start + j}: {err}")
            bar.update(len(chunk))
            continue

        for j, out in enumerate(outputs):
            answers = out.get("answers", {})
            for qid, clf in qid_to_clf.items():
                p = answers.get(qid, {}).get("noul", None)
                if p is None:
                    ok[start + j] = False
                    failures.append(f"amostra {start + j}: sem resposta para {clf}")
                    break
                scores[start + j, clf_index[clf]] = float(p)
            tokens_in += int((out.get("usage") or {}).get("input_tokens") or 0)
        bar.update(len(chunk))
    bar.close()
    return scores, ok, tokens_in, infer_time, failures


def fixed_threshold_metrics(scores, labels, clfnames, threshold):
    metrics = {}
    for i, clf in enumerate(clfnames):
        pred = scores[:, i] >= threshold
        true = labels[:, i] == 1
        tp = int(np.sum(pred & true))
        fp = int(np.sum(pred & ~true))
        fn = int(np.sum(~pred & true))
        tn = int(np.sum(~pred & ~true))
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        metrics[clf] = {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "accuracy": (tp + tn) / len(true) if len(true) > 0 else 0.0,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "positivos_no_gabarito": int(np.sum(true)),
        }
    return metrics


def print_comparison(own_entry, compare_paths, clfnames):
    rows = [(own_entry["model"], own_entry["fmax"])]
    for path in compare_paths:
        if not os.path.exists(path):
            print(f"Aviso: {path} não encontrado, ignorado.")
            continue
        with open(path, "r") as f:
            content = json.load(f)
        items = content if isinstance(content, list) else [content]
        for item in items:
            rows.append((item.get("model", path), item.get("fmax", {})))

    short = {c: c.replace(" ?", "")[:18] for c in clfnames}
    header = f"{'modelo':<40}" + "".join(f"{short[c]:>20}" for c in clfnames) + f"{'média':>10}"
    print("\nFmax por classificação")
    print(header)
    print("-" * len(header))
    for name, fmax in rows:
        vals = [fmax.get(c) for c in clfnames]
        present = [v for v in vals if v is not None]
        mean = sum(present) / len(present) if present else float("nan")
        line = f"{str(name)[:39]:<40}"
        line += "".join(f"{v:>20.3f}" if v is not None else f"{'-':>20}" for v in vals)
        line += f"{mean:>10.3f}"
        print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", nargs="?", default="results/laya_results.json")
    parser.add_argument("--model", default=os.getenv("LAYA_MODEL", "convaiinnovations/laya"))
    parser.add_argument("--subfolder", default=os.getenv("LAYA_SUBFOLDER", "multilingual"),
                        help="subpasta do checkpoint; vazio para o modelo em inglês")
    parser.add_argument("--device", default=os.getenv("LAYA_DEVICE", ""))
    parser.add_argument("--max-len", default=os.getenv("LAYA_MAX_LEN", ""),
                        help="limite de tokens da entrada; vazio usa o do checkpoint")
    parser.add_argument("--threshold", type=float, default=float(os.getenv("LAYA_THRESHOLD", "0.5")))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--testing-perc", type=float, default=USE_PERC)
    parser.add_argument("--limit", type=int, default=0, help="usar só as N primeiras amostras")
    parser.add_argument("--compare", nargs="*", default=[],
                        help="arquivos de resultado de outros testes para comparar o Fmax")
    args = parser.parse_args()

    (_, texts, clfnames, labels, full_schema_dict, _) = get_testing_inputs(
        usage_perc=args.testing_perc
    )
    if args.limit and args.limit > 0:
        texts = texts[: args.limit]
        labels = labels[: args.limit]
    labels = np.asarray(labels)
    print(f"Amostras: {len(texts)}")

    questions, qid_to_clf = build_questions(clfnames, full_schema_dict)
    agent = load_agent(args.model, args.subfolder, args.device, args.max_len)

    scores, ok, tokens_in, infer_time, failures = run_inference(
        agent, texts, questions, qid_to_clf, clfnames, max(1, args.batch_size)
    )

    n_requests = len(texts)
    n_failures = int(np.sum(~ok))
    if n_failures == n_requests:
        print("Nenhuma inferência bem-sucedida.")
        for f in failures[:10]:
            print(f)
        sys.exit(1)

    scores_ok = scores[ok]
    labels_ok = labels[ok]

    fmax, recalls_at_good_precision, recalls, precisions, best_thresholds = (
        find_fmax_per_col_parallel(scores_ok, labels_ok, clfnames, n_jobs=4)
    )
    fixed = fixed_threshold_metrics(scores_ok, labels_ok, clfnames, args.threshold)

    model_name = "laya:" + args.model + (f"/{args.subfolder}" if args.subfolder else "")
    samples = int(np.sum(ok))
    meta = {
        "tokens_per_second_in": tokens_in / infer_time if infer_time > 0 else 0,
        "tokens_per_second_out": 0,
        "latency_sum": infer_time,
        "gpu_seconds": infer_time,
        "seconds_per_transcript": infer_time / samples,
        "samples": samples,
        "tokens_total_in": tokens_in,
        "tokens_total_out": 0,
        "failures": n_failures,
        "failures_perc": n_failures / n_requests,
        "n_requests": n_requests,
        "failure_texts": failures,
        "device": args.device or "auto",
        "batch_size": args.batch_size,
        "threshold": args.threshold,
    }

    per_sample = []
    idx_ok = np.where(ok)[0]
    for k, i in enumerate(idx_ok):
        per_sample.append(
            {
                "indice": int(i),
                "probabilidades": {c: round(float(scores[i, j]), 4) for j, c in enumerate(clfnames)},
                "gabarito": {c: int(labels[i, j]) for j, c in enumerate(clfnames)},
            }
        )

    mean_fmax = float(np.mean(list(fmax.values())))
    mean_f1_fixed = float(np.mean([m["f1"] for m in fixed.values()]))
    entry = {
        "model": model_name,
        "mean_metrics": {
            "fmax": mean_fmax,
            "f1_limiar_fixo": mean_f1_fixed,
            "recall": float(np.mean(list(recalls.values()))),
            "precision": float(np.mean(list(precisions.values()))),
            # O Laya não extrai entidades.
            "jw_sim": None,
        },
        "meta": meta,
        "fmax": fmax,
        "recall": recalls,
        "precision": precisions,
        "recalls_at_good_precisions": recalls_at_good_precision,
        "best_thresholds": best_thresholds,
        "limiar_fixo": fixed,
        "jw_sim": {},
        "detailed_results": {"per_sample": per_sample},
    }

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump([entry], f, indent=4, ensure_ascii=False)

    print(f"\nModelo: {model_name}")
    print(f"Amostras válidas: {samples} de {n_requests} (falhas: {n_failures})")
    print(f"Tempo de inferência: {infer_time:.2f}s ({infer_time / samples * 1000:.1f} ms por transcrição)")
    print(f"\n{'classificação':<32}{'Fmax':>7}{'limiar*':>9}{'F1@' + str(args.threshold):>9}{'prec':>7}{'rec':>7}{'pos':>6}")
    for c in clfnames:
        m = fixed[c]
        print(
            f"{c:<32}{fmax[c]:>7.3f}{best_thresholds[c]:>9.3f}{m['f1']:>9.3f}"
            f"{m['precision']:>7.3f}{m['recall']:>7.3f}{m['positivos_no_gabarito']:>6}"
        )
    print(f"{'média':<32}{mean_fmax:>7.3f}{'':>9}{mean_f1_fixed:>9.3f}")
    print("* limiar que maximiza o F1 nestas amostras (otimista).")

    if args.compare:
        print_comparison(entry, args.compare, clfnames)

    print(f"\nResultado salvo em {args.output}")


if __name__ == "__main__":
    main()
