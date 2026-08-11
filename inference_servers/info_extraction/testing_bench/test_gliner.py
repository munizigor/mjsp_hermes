import os
import sys
import json
import gc
import numpy as np
from tqdm import tqdm

from testing import (
    get_testing_inputs,
    entity_redundancies,
    calc_metadata,
    find_max_jw_sim,
    find_fmax_per_col_parallel,
)


def test_gliner_model(model) -> dict:
    ml_categories, all_texts, clfnames, clfs_true, full_schema_dict, entities_true = (
        get_testing_inputs()
    )
    schema_str = json.dumps(full_schema_dict, ensure_ascii=False, indent=2)
    # print("classification schema:", schema_str)

    entity_names = list(full_schema_dict["entities"].keys())
    non_redundant_entity_names = [
        name for name in entity_names if name not in entity_redundancies.keys()
    ]

    clfs_scores = []
    entities_found = []
    gpu_usage_total = 0.0
    output_strs = []
    for transcript in tqdm(all_texts):
        # for transcript in [text5]:
        # print('\n\nTexto: ', transcript)
        new_entities, req_duration1 = model.single_inference(transcript, schema_str)
        output_strs.append(json.dumps(new_entities, ensure_ascii=False, indent=2))
        gpu_usage_total += req_duration1
        # print('\tLabels encontradas: ', new_entities)
        scores_line = [new_entities[c] if c in new_entities else 0.0 for c in clfnames]
        clfs_scores.append(scores_line)
        entities_line = {
            name: new_entities[name] for name in entity_names if name in new_entities
        }
        #print(scores_line)
        for redundant, correct_name in entity_redundancies.items():
            if redundant in entities_line:
                if not correct_name in entities_line:
                    entities_line[correct_name] = []
                entities_line[correct_name].extend(entities_line[redundant])
                del entities_line[redundant]
        entities_found.append(entities_line)
        # print(entities_found[-1])

    tokens_per_second_in, tokens_per_second_out = calc_metadata(
        all_texts, output_strs, gpu_usage_total
    )

    n_samples = len(clfs_scores)

    fmax_per_col, recalls_at_good_precisions, recalls, precisions, best_thresholds = (
        find_fmax_per_col_parallel(
            np.array(clfs_scores),
            clfs_true,
            clfnames,
            n_jobs=4,
        )
    )

    similarities_per_entity = {}
    for entity_name in non_redundant_entity_names:
        pred_values = [
            entities_found[i][entity_name] if entity_name in entities_found[i] else []
            for i in range(len(entities_found))
        ]
        true_values = [entities_true[i][entity_name] for i in range(len(entities_true))]

        """print(entity_name)
        print('\tpreds', pred_values)
        print('\ttrue', true_values)"""

        fmax, jw_sim_max, recall, precision = find_max_jw_sim(
            pred_values, true_values, field_name=entity_name
        )
        similarities_per_entity[entity_name] = jw_sim_max
        fmax_per_col[entity_name] = fmax
        recalls[entity_name] = recall
        precisions[entity_name] = precision

    result_dicts = {
        "fmax_per_col": fmax_per_col,
        "similarities_per_entity": similarities_per_entity,
        "recalls_at_good_precisions": recalls_at_good_precisions,
        "recalls": recalls,
        "precisions": precisions,
        "best_thresholds": best_thresholds,
        "meta": {
            "tokens_per_second_in": tokens_per_second_in,
            "tokens_per_second_out": tokens_per_second_out,
            "gpu_seconds": gpu_usage_total,
            "samples": n_samples,
        },
    }

    return result_dicts


if __name__ == "__main__":
    from gliner1_model import TritonPythonModel as TritonPythonModelGliner1

    results = []

    results_path = sys.argv[1]

    prev_res = None
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            prev_res = json.load(f)

    if prev_res is None:
        prev_res = []

    for prev_calc in prev_res:
        results.append(prev_calc)

    models_calculated_quality = [r["model"] for r in prev_res]

    gliclass_names = [
        #'knowledgator/gliclass-modern-base-v2.0',
        #'knowledgator/gliclass-modern-large-v2.0',
        #"knowledgator/gliclass-llama-1.3B-v1.0",
        #"knowledgator/gliclass-qwen-1.5B-v1.0",
        #"knowledgator/gliclass-x-base",
        #"BioMike/gliclass-large-reddit-1m-6k",
        #"knowledgator/gliclass_msmarco_merged",
        #"knowledgator/gliclass-base-v3.0",
        "knowledgator/gliclass-large-v3.0",
        #"knowledgator/gliclass-base-v2.0-rac-init",
        #'knowledgator/gliclass-modern-large-v3.0',
    ]

    non_gliclass_clfs = [
        'knowledgator/comprehend_it-base',
        'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7',
        'infly/inf-retriever-v1-1.5b',
        'voyageai/voyage-4-nano',
        'intfloat/multilingual-e5-large-instruct',
        'sergeyzh/BERTA',
        'ai-forever/FRIDA',
        'sergeyzh/rubert-mini-frida',
        'ai-sage/Giga-Embeddings-instruct',
        'Alibaba-NLP/gte-Qwen2-1.5B-instruct',
        'thenlper/gte-large',
        'google/embeddinggemma-300m',
        'sergeyzh/rubert-tiny-turbo',
        'sdadas/mmlw-e5-large',
        'BAAI/bge-large-en-v1.5',
        'Snowflake/snowflake-arctic-embed-l-v2.0',
        'google-t5/t5-large',
        'google-t5/t5-3b',
        'microsoft/mpnet-base',
        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        'deepvk/USER2-base',
        'jinaai/jina-embeddings-v5-text-small',
        'Qwen/Qwen3-Embedding-0.6B'
    ]

    glinerv1_names = [
        "knowledgator/gliner-x-large",
        #"numind/NuNER_Zero-span",
        #"numind/NuNER_Zero",
        #"nvidia/gliner-pii",
        #"knowledgator/gliner-pii-large-v1.0",
        #"gliner-community/gliner_xxl-v2.5",
        #"gliner-community/gliner_large-v2.5",
        #"gretelai/gretel-gliner-bi-large-v1.0",
        #"knowledgator/gliner-bi-large-v2.0",
        #"knowledgator/gliner-bi-large-v1.0",
        #"knowledgator/gliner-multitask-v1.0",
    ]

    glinerv2_names = [
        #"fastino/gliner2-multi-v1", 
        #"fastino/gliner2-large-v1"
    ]

    use_cpu_gliner1 = True

    for glinerv1_name in glinerv1_names:
        for gliclass_name in gliclass_names:
            comb_name = glinerv1_name + " + " + gliclass_name
            if use_cpu_gliner1:
                comb_name += " (CPU)"
            if comb_name in models_calculated_quality:
                continue
            try:
                model_gli1 = TritonPythonModelGliner1()
                model_gli1.initialize(
                    {"clf_model_id": gliclass_name, "model_id": glinerv1_name, 
                        "use_cuda": not use_cpu_gliner1}
                )

                result_dicts = test_gliner_model(model_gli1)

                for col_name, fmax in result_dicts["fmax_per_col"].items():
                    print(f"\tFmax para {col_name}: {fmax}")

                for col_name, sim in result_dicts["similarities_per_entity"].items():
                    print(f"\tJW Sim para {col_name}: {sim}")

                mean_fmax = np.mean(list(result_dicts["fmax_per_col"].values()))
                mean_jw_sim = np.mean(
                    list(result_dicts["similarities_per_entity"].values())
                )
                mean_good_recall = np.mean(
                    list(result_dicts["recalls_at_good_precisions"].values())
                )
                mean_recall = np.mean(list(result_dicts["recalls"].values()))
                mean_precision = np.mean(list(result_dicts["precisions"].values()))

                print(f"\tMedia Fmax: {mean_fmax}")
                print(f"\tMedia JW Sim: {mean_jw_sim}")
                print(f"\tMedia Good Recall: {mean_good_recall}")
                print(f"\tMedia Recall: {mean_recall}")
                print(f"\tMedia Precision: {mean_precision}")
                results.append(
                    {
                        "model": comb_name,
                        "mean_metrics": {
                            "fmax": mean_fmax,
                            "jw_sim": mean_jw_sim,
                            # "recall_at_95_precision": mean_good_recall,
                            "recall": mean_recall,
                            "precision": mean_precision,
                        },
                        "meta": result_dicts["meta"],
                        "fmax": result_dicts["fmax_per_col"],
                        "jw_sim": result_dicts["similarities_per_entity"],
                        # "recall_at_95_precision": result_dicts["recalls_at_good_precisions"],
                        "recall": result_dicts["recalls"],
                        "precision": result_dicts["precisions"],
                        "best_thresholds": result_dicts["best_thresholds"],
                    }
                )
                with open(results_path, "w") as f:
                    json.dump(results, f, indent=4)

                del model_gli1
                gc.collect()
                if not use_cpu_gliner1:
                    import torch
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error with model {gliclass_name}: {e}")
                del model_gli1
                gc.collect()
                if not use_cpu_gliner1:
                    import torch
                    torch.cuda.empty_cache()
                #raise(e)
                continue

    for m_name in glinerv2_names:
        from gliner2_model import TritonPythonModel as TritonPythonModelGliner2
        if m_name in models_calculated_quality:
            continue
        model_gli2 = TritonPythonModelGliner2()
        model_gli2.initialize({"model_id": m_name})

        result_dicts = test_gliner_model(model_gli2)

        del model_gli2
        gc.collect()
        if not use_cpu_gliner1:
            import torch
            torch.cuda.empty_cache()
        for col_name, fmax in result_dicts["fmax_per_col"].items():
            print(f"\tFmax para {col_name}: {fmax}")

        mean_fmax = np.mean(list(result_dicts["fmax_per_col"].values()))
        mean_jw_sim = np.mean(list(result_dicts["similarities_per_entity"].values()))
        mean_good_recall = np.mean(
            list(result_dicts["recalls_at_good_precisions"].values())
        )
        mean_recall = np.mean(list(result_dicts["recalls"].values()))
        mean_precision = np.mean(list(result_dicts["precisions"].values()))

        print(f"\tMedia Fmax: {mean_fmax}")
        print(f"\tMedia JW Sim: {mean_jw_sim}")
        # print(f"\tMedia Good Recall: {mean_good_recall}")
        print(f"\tMedia Recall: {mean_recall}")
        print(f"\tMedia Precision: {mean_precision}")

        results.append(
            {
                "model": m_name,
                "mean_metrics": {
                    "fmax": mean_fmax,
                    "jw_sim": mean_jw_sim,
                    # "recall_at_95_precision": mean_good_recall,
                    "recall": mean_recall,
                    "precision": mean_precision,
                },
                "meta": result_dicts["meta"],
                "fmax": result_dicts["fmax_per_col"],
                "jw_sim": result_dicts["similarities_per_entity"],
                # "recall_at_95_precision": result_dicts["recalls_at_good_precisions"],
                "recall": result_dicts["recalls"],
                "precision": result_dicts["precisions"],
                "best_thresholds": result_dicts["best_thresholds"],
            }
        )
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)

    # Save results to json:

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
