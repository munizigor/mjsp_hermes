import os
import sys
import json
import numpy as np

model_minimum_instances = {
    "default_small": "gcp_n1_t4",
    "llama-3.1-8B-instruct": "gcp_g2_l4",
    "gemma-3n-e4b-it": "gcp_g2_l4",
    "gemma-3-4b-it": "gcp_g2_l4",
    "deepseek-r1-distill-qwen-14b": "gcp_g2_2x-l4",
    "mistral-small-3.2-24b-instruct": "gcp_a2_a100",
    "magistral-small": "gcp_a2_a100",
    "qwen3.5-35b": "gcp_a2_a100",
    "gpt-oss-120b": "gcp_a2_a100",
}

if __name__ == "__main__":
    results_files = sys.argv[1:]

    # Load prices
    try:
        with open("input/test_prices.json", "r") as f:
            prices = json.load(f)
            # Default standard price for time-based cost
            standard_price_per_hour = prices.get(
                model_minimum_instances["default_small"], {}
            ).get("dollars_hour_standard", 0)
    except FileNotFoundError:
        print("Error: input/test_prices.json not found.")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing key in prices JSON: {e}")
        sys.exit(1)

    data = []

    specific_fmax_categories = [
        "Fato Ocorrendo Neste Momento ?",
        "Autor Do Fato No Local ?",
        "Autor Do Fato Armado ?",
        "Feridos Com Risco de Morte ?",
        "Risco De Tumulto ?",
        "Lei Maria da Penha ?",
    ]

    max_samples = 429

    for results_file in results_files:
        if not os.path.exists(results_file):
            print(f"Warning: File {results_file} not found. Skipping.")
            continue

        with open(results_file, "r") as f:
            try:
                file_content = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {results_file}. Skipping.")
                continue

        # Handle if the file contains a list of results or a single dict
        if isinstance(file_content, dict):
            items = [file_content]
        elif isinstance(file_content, list):
            items = file_content
        else:
            print(f"Warning: Unexpected content format in {results_file}. Skipping.")
            continue

        for item in items:
            row = {}

            # Model names
            full_model_name = item.get("model", "Unknown")
            row["full_model"] = full_model_name

            if " + " in full_model_name:
                parts = full_model_name.split(" + ")
                row["ner_model"] = parts[0]
                row["classification_model"] = parts[1]
            else:
                row["ner_model"] = full_model_name
                row["classification_model"] = full_model_name

            # Mean Metrics
            # mean_fmax (specific categories)
            fmax_dict = item.get("fmax", {})
            fmax_values = [
                fmax_dict.get(cat)
                for cat in specific_fmax_categories
                if cat in fmax_dict
            ]

            if fmax_values and len(fmax_values) > 0:
                row["mean_fmax"] = sum(fmax_values) / len(fmax_values)
            else:
                row["mean_fmax"] = None  # Or 0 depending on preference, None is safer

            # mean_jw_sim
            row["mean_jw_sim"] = item.get("mean_metrics", {}).get("jw_sim")

            # Categories (fmax for questions, jw_sim for labels)
            # Assumption: Questions end with "?" -> fmax. Others -> jw_sim.

            # Populate from fmax dict (for questions)
            for k, v in item.get("fmax", {}).items():
                if "?" in k:
                    row[k] = v

            # Populate from jw_sim dict (for labels - not questions)
            for k, v in item.get("jw_sim", {}).items():
                if "?" not in k:
                    row[k] = v

            # Metadata & Costs
            model_price_info = prices.get(full_model_name)
            meta = item.get("meta", {})
            samples = meta.get("samples", 446)
            if samples > max_samples:
                samples = max_samples
                row["samples"] = samples
            gpu_seconds = meta.get("gpu_seconds", 0)
            tokens_in_sec = meta.get("tokens_per_second_in", 0)
            tokens_out_sec = meta.get("tokens_per_second_out", 0)

            row["total_runtime"] = gpu_seconds
            row["tokens_per_second_in"] = tokens_in_sec
            row["tokens_per_second_out"] = tokens_out_sec
            row["samples"] = samples

            if "failures_perc" in meta:
                row["success_rate"] = 1 - meta["failures_perc"]
            else:
                success_rate = samples / max_samples
                row["success_rate"] = success_rate

            # Million tokens
            # Try to get raw tokens first if available (for exact calculation)
            # otherwise calculate from rate * time
            raw_tokens_in = meta.get("tokens_total_in", tokens_in_sec * gpu_seconds)
            raw_tokens_out = meta.get("tokens_total_out", tokens_out_sec * gpu_seconds)

            million_input_tokens = raw_tokens_in / 1_000_000.0
            million_output_tokens = raw_tokens_out / 1_000_000.0

            row["million_input_tokens"] = million_input_tokens
            row["million_output_tokens"] = million_output_tokens

            # Cost Calculation
            # Logic: Check if model has token-based pricing in test_prices.json
            # If so, use token pricing. Else, fallback to time-based (standard_price_per_hour * time)

            if model_price_info and "dollars_1m_tokens_in" in model_price_info:
                # Token-based pricing
                in_price = model_price_info["dollars_1m_tokens_in"]
                out_price = model_price_info["dollars_1m_tokens_out"]
                cost_in = million_input_tokens * in_price
                cost_out = million_output_tokens * out_price
                cost = cost_in + cost_out

                print(
                    f"Using API prices for {full_model_name}: $IN={in_price}/M, $OUT={out_price}/M, Total={cost}"
                )
            else:
                # Time-based pricing (fallback)
                # If no specific time-based price found for this model, fallback to gcp_n1_t4 standard
                # This assumes non-API models are running on our standard instance

                # We could look for model-specific time pricing if needed, e.g. prices.get(full_model_name, {}).get("dollars_hour_standard")
                # But currently prompt implies using gcp_n1_t4 for others.
                hourly_rate = standard_price_per_hour
                if full_model_name in model_minimum_instances:
                    instance_name = model_minimum_instances[full_model_name]
                    if instance_name in prices:
                        hourly_rate = prices[instance_name].get(
                            "dollars_hour_standard", standard_price_per_hour
                        )
                        print(
                            f"Using instance {instance_name} for model {full_model_name}"
                        )
                elif model_price_info and "dollars_hour_standard" in model_price_info:
                    hourly_rate = model_price_info["dollars_hour_standard"]

                cost = hourly_rate * (gpu_seconds / 3600.0)
                print(
                    f"Using time-based pricing for {full_model_name}: ${hourly_rate}/hour, Total=${cost}"
                )

            row["total_cost"] = cost
            if samples > 0:
                row["cost_per_transcript"] = cost / samples
                row["seconds_per_transcript"] = gpu_seconds / samples
            else:
                row["cost_per_transcript"] = np.inf
                row["seconds_per_transcript"] = np.inf

            row["mean_quality"] = (row["mean_fmax"] + row["mean_jw_sim"]) / 2

            data.append(row)

    # Deduplicate by full_model, keeping the one with the highest samples
    # If the same number of samples, keep the one with the highest mean_quality
    best_entries = {}
    for entry in data:
        model = entry["full_model"]
        curr_samples = entry.get("samples", 0)
        curr_quality = entry.get("mean_quality", 0)
        if (
            model not in best_entries
            or curr_samples > best_entries[model].get("samples", 0)
            or (
                curr_samples == best_entries[model].get("samples", 0)
                and curr_quality > best_entries[model].get("mean_quality", 0)
            )
        ):
            best_entries[model] = entry
    data = list(best_entries.values())

    # Sort if mean_quality can be calculated (requires mean_fmax and mean_jw_sim to be non-None)
    # We'll filter out rows where calculation fails for sorting, or treat Nonetype as 0
    data.sort(key=lambda x: x["mean_quality"], reverse=True)

    # Create DataFrame
    import pandas as pd

    df = pd.DataFrame(data)

    df["Qualidade Geral"] = df["mean_quality"] * df["success_rate"]
    df["Qualidade Classificação"] = df["mean_fmax"] * df["success_rate"]
    df["Qualidade Reconhecimento de Entidades"] = df["mean_jw_sim"] * df["success_rate"]

    # pdp Power–delay product
    df["log_avg_runtime"] = np.log(df["total_runtime"] + 1) / max_samples
    # df["log_cost"] = np.log(df["total_cost"])
    df["Power–delay product"] = df["seconds_per_transcript"] * df["cost_per_transcript"]

    df["Power–delay product norm."] = 1 - (
        df["Power–delay product"] / df["Power–delay product"].max()
    )

    """df["Custo-Benefício"] = (
        df["Qualidade Geral"] * 5 + df["Power–delay product norm."] * 5
    ) / 10"""

    # sort again
    df = df.sort_values(by="Qualidade Geral", ascending=False)

    # Ensure output directory exists
    output_path = "results/df.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Saved dataframe to {output_path}")

    excel_output = output_path.replace(".csv", ".xlsx")
    df.to_excel(excel_output, index=False)
    print(f"Saved dataframe to {excel_output}")

    df_simples = df[
        [
            "ner_model",
            "classification_model",
            # "Custo-Benefício",
            "Qualidade Geral",
            "Qualidade Classificação",
            "Qualidade Reconhecimento de Entidades",
            "Power–delay product",
            "total_cost",
            "cost_per_transcript",
            "success_rate",
            "seconds_per_transcript",
            "total_runtime",
        ]
    ]

    df_simples_path = output_path.replace(".csv", "_simples.xlsx")
    df_simples.to_excel(df_simples_path, index=False)
    print(f"Saved dataframe to {df_simples_path}")

    df_simples.to_csv(df_simples_path.replace(".xlsx", ".csv"), index=False)
    print(f"Saved dataframe to {df_simples_path.replace('.xlsx', '.csv')}")
