import unidecode
import re
from tqdm import tqdm
import jiwer
import json
import polars as pl
from rapidfuzz import fuzz

emb_cache = {}

def value_comparisons(test_path):
    from sentence_transformers import SentenceTransformer

    dataset_path = f"{test_path}/chamadas_roteirizadas_com_voz.json"
    inferences_path = f"{test_path}/inference_summary.parquet"
    inferences = pl.read_parquet(inferences_path)

    original_lines = {}
    for entry in json.load(open(dataset_path, "r")):
        roteiro2 = ""
        for fala in entry["Roteiro Segmentado"]:
            if len(fala) == 2:
                roteiro2 += " " + fala[1]
            elif len(fala) == 1:
                roteiro2 += " " + fala[0]
            else:
                roteiro2 += " " + str(fala)

        newline = {
            "rua_ou_logradouro": entry["Emergencia"]["Endereco"]["rua"],
            "numero": entry["Emergencia"]["Endereco"]["numero"],
            "bairro": entry["Emergencia"]["Endereco"]["bairro"],
            "municipio": entry["Emergencia"]["Endereco"]["cidade"],
            "ponto_de_referencia": entry["Emergencia"]["Endereco"]["ref_name"],
            "complemento": entry["Emergencia"]["Endereco"]["complemento"],
            "participacoes": entry["participacoes"],
            "nome_do_solicitante": entry["Perfil do Solicitante"]["Nome Solicitante"],
            "classificacao_decisiva": entry["Emergencia"]["Natureza"]["Natureza"],
            "transcricao": roteiro2,
        }
        original_lines[entry["index"]] = newline

    stm = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    def calc_emb(text):
        if not text in emb_cache:
            embs = stm.encode([text])
            emb = embs[0]
            emb_cache[text] = emb
        return emb_cache[text]

    HONORIFICS = [
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

    def norm_transc(t):
        t = t.strip("\"'")
        t = t.replace("\n", " ")
        t = t.lower()
        t = unidecode(t)
        t = t.replace("operador: ", "")
        t = t.replace("solicitante: ", "")
        return t

    def basic_norm(t, name=None):
        t = t.lower()
        t = unidecode(t)
        if name != None:
            t = name + ": " + t
            if "pessoa" in name or "nome" in name:
                for h in HONORIFICS:
                    t = re.sub(h, "", t)
            if "numero" in name:
                t = t.replace("numero ", "")
        # t = t.replace('dr.', 'doutor')
        return t

    results = []
    for row in tqdm(inferences.rows(named=True)):
        index = int(row["dataset_index"])
        original_data = original_lines[index]

        transcricao = norm_transc(row["transcricao"])
        original_text = norm_transc(original_data["transcricao"])
        embs = [calc_emb(t) for t in [transcricao, original_text]]
        # print(len(embs), len(embs[0]))
        similarities = stm.similarity(embs[0], embs[1])
        # print(similarities)
        wer = jiwer.wer(original_text, transcricao)
        # print(similarities, wer)
        newline = {
            "index": index,
            "roteiro": original_text,
            "transcricao": transcricao,
            "transcricao_sim": similarities.tolist()[0][0],
            "transcricao_wer": wer,
        }

        col = "pessoas"
        row["nome_do_solicitante"] = json.dumps(
            json.loads(row["nome_do_solicitante"]) + json.loads(row["pessoa"]),
            ensure_ascii=False,
        )
        pessoas_original = [p["pessoa"] for p in original_data["participacoes"]]
        if len(pessoas_original) > 0:
            pessoas_pred = [p["value"] for p in json.loads(row["pessoa"])]
            embs_original = [
                calc_emb(basic_norm(t, name=col)) for t in pessoas_original
            ]
            embs_pred = [calc_emb(basic_norm(t, name=col)) for t in pessoas_pred]

            pessoa_results = []
            for pessoa_orig, pessoa_orig_emb in zip(pessoas_original, embs_original):
                if len(pessoas_pred) > 0:
                    local_sims = []
                    for pessoa_pred, pessoa_pred_emb in zip(pessoas_pred, embs_pred):
                        new_sims = stm.similarity(pessoa_orig_emb, pessoa_pred_emb)
                        new_sim = new_sims.tolist()[0][0]
                        s_partial = fuzz.partial_ratio(pessoa_pred, pessoa_orig) / 100.0
                        final_score = (new_sim + s_partial) / 2
                        local_sims.append([final_score, pessoa_pred])
                    local_sims.sort()
                    highest_sim, most_similar_person = local_sims[-1]
                    pessoa_results.append([highest_sim, most_similar_person])
                else:
                    pessoa_results.append([0.0, None])
            found_th = 0.84
            not_nulls = [x for x, p in pessoa_results]
            mean_sim = sum(not_nulls) / len(pessoa_results)
            n_found = len([p for p in not_nulls if p >= found_th])
            newline[col + "_true"] = ", ".join(sorted(pessoas_original))
            newline[col + "_all"] = json.dumps(pessoa_results, ensure_ascii=False)
            newline[col + "_sim"] = mean_sim
            newline[col + "_perc_found"] = n_found / len(pessoas_original)
        else:
            newline[col + "_true"] = None
            newline[col + "_all"] = row["pessoa"]
            newline[col + "_sim"] = None
            newline[col + "_perc_found"] = None

        common_comparisons = [
            "classificacao_decisiva",
            "nome_do_solicitante",
            "rua_ou_logradouro",
            "numero",
            "bairro",
            "municipio",
            "ponto_de_referencia",
        ]
        compare_substrings = [
            "nome_do_solicitante",
            "rua_ou_logradouro",
            "numero",
            "bairro",
            "municipio",
        ]
        for col in common_comparisons:
            v_original = original_data[col]
            if v_original is None:
                v_original = ""
            if v_original != "":
                v_norm = basic_norm(v_original, name=col)
                newline[col + "_true"] = v_norm.replace(col + ": ", "")
                if row[col] != None:
                    v_inf_json = json.loads(row[col])
                else:
                    v_inf_json = []
                v_sims = []
                original_emb = calc_emb(v_norm)
                if len(v_inf_json) > 0:
                    for entry in v_inf_json:
                        value = entry["value"]
                        value_norm = basic_norm(value, name=col)
                        new_emb = calc_emb(value_norm)
                        new_sims = stm.similarity(original_emb, new_emb)
                        new_sim = new_sims.tolist()[0][0]
                        if col in compare_substrings:
                            s_partial = fuzz.partial_ratio(value_norm, v_norm) / 100.0
                            final_score = (new_sim + s_partial) / 2
                        else:
                            final_score = new_sim
                        v_sims.append(
                            [float(final_score), value_norm.replace(col + ": ", "")]
                        )
                    v_sims.sort()
                    v_inf_sim, v_inf_best = v_sims[-1]
                    newline[col + "_all"] = json.dumps(v_sims, ensure_ascii=False)
                    newline[col + "_inf"] = v_inf_best
                    newline[col + "_sim"] = v_inf_sim
                else:
                    newline[col + "_all"] = "[]"
                    newline[col + "_inf"] = None
                    newline[col + "_sim"] = 0.0
            else:
                newline[col + "_true"] = ""
                newline[col + "_all"] = "[]"
                newline[col + "_inf"] = None
                newline[col + "_sim"] = None

        results.append(newline)

    results_df = pl.DataFrame(results)

    cols_to_keep = [
        c for c in results_df.columns if "_sim" in c or "_wer" in c or c in ["index"]
    ]
    df2 = results_df[cols_to_keep]

    results_df.write_csv(
        f"{test_path}/semantic_performance-complete.tsv", separator="\t"
    )
    df2.write_csv(f"{test_path}/semantic_performance.tsv", separator="\t")

    return results_df