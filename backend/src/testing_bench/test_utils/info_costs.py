import polars as pl
import polars.exceptions as pl_exc
import json
import numpy as np


def parse_prices_file(prices_dataset_path):
    prices_dict = json.load(open(prices_dataset_path, "r"))
    all_prices = {}

    for name, prices in prices_dict["text_api"].items():
        all_prices[name] = prices

    for name, price in prices_dict["asr_api"].items():
        all_prices[name] = {"dollars_1h_audio": price}

    for instance_name, hardware_name in prices_dict["instance_names"].items():
        more_info = prices_dict["instances_types"][hardware_name]
        all_info = {k: v for k, v in more_info["hardware"].items()}

        demand_prices = [
            prices["on_demand"]
            for provider, prices in more_info.items()
            if "on_demand" in prices and provider == "gcp"
        ]
        spot_prices = [
            prices["spot"]
            for provider, prices in more_info.items()
            if "on_demand" in prices and provider == "gcp"
        ]
        all_info["demand_prices"] = demand_prices
        all_info["dollars_1h_demand"] = float(np.mean(demand_prices))
        all_info["spot_prices"] = spot_prices
        all_info["dollars_1h_spot"] = float(np.mean(spot_prices))

        all_prices[instance_name] = all_info

        if "-b" in instance_name:
            all_prices[instance_name.replace("-b", "")] = all_info

    return all_prices


def parse_costs(
    n_calls: int,
    hardware_usage_df: pl.DataFrame | None,
    inferencia_stats_df: pl.DataFrame,
    prices_dict: dict,
    machine_name: str,
    test_configs: dict,
    transcricao_tempo_total: float,
    duracao_teste,
    api_uses: list,
) -> pl.DataFrame:
    # TODO: calc api usage costs from inferences table
    # machine_stats = prices_dict[machine_name]
    # on_demand_prices = machine_stats["demand_prices"]
    # assert len(on_demand_prices) >= 1
    # on_demand_hour = machine_stats["dollars_1h_demand"]
    # print(machine_stats)
    """inferencia_por_container = {
        "ner": ["ner"],
        "interpretation": [
            "natureza_decisiva",
            "lista_de_naturezas",
            "descricao_e_observacao",
        ],
    }"""

    horas_de_teste = duracao_teste / 60 / 60

    cost_lines = []
    for api_usage_dict in api_uses:
        cost_lines.append(
            {
                "Recurso": "APIs Comerciais",
                "Sub-recurso": api_usage_dict["model"]
                + " - "
                + api_usage_dict["tipo"].upper(),
                "Tipo de Custo": "API",
                "Horas de Áudio": api_usage_dict["Horas de Áudio"],
                "Tokens de Input (milhões)": api_usage_dict["input tokens (millions)"],
                "Tokens de Output (milhões)": api_usage_dict[
                    "output tokens (millions)"
                ],
                "Custo ($)": api_usage_dict["cost"],
            }
        )
        """elif "Horas de Áudio" in api_usage_dict:
            cost_lines.append(
                {
                    "Recurso": "APIs Comerciais",
                    "Sub-recurso": api_usage_dict["model"] + " - " + api_usage_dict["tipo"].upper(),
                    "Tipo de Custo": "API",
                    "Horas de Áudio": api_usage_dict["Horas de Áudio"],
                    "Tokens de Input (milhões)": 0,
                    "Tokens de Output (milhões)": 0,
                    "Custo ($)": api_usage_dict["cost"]
                }
            )"""
    custos_apis = sum([r["Custo ($)"] for r in cost_lines])
    total_h = sum([r["Horas de Áudio"] for r in cost_lines])
    total_in = sum([r["Tokens de Input (milhões)"] for r in cost_lines])
    total_out = sum([r["Tokens de Output (milhões)"] for r in cost_lines])
    cost_lines.append(
        {
            "Recurso": "APIs Comerciais",
            "Sub-recurso": "Total Utilizado",
            "Tipo de Custo": "API",
            "Horas de Áudio": total_h,
            "Tokens de Input (milhões)": total_in,
            "Tokens de Output (milhões)": total_out,
            "Custo ($)": custos_apis,
        }
    )

    if hardware_usage_df is not None:
        for c_type, c_df in hardware_usage_df.group_by("hostname"):
            hostname = str(c_type[0])

            host_pricing = prices_dict[hostname]
            on_demand_hour = host_pricing["dollars_1h_demand"]

            vm_cost = on_demand_hour * horas_de_teste
            if "backend" not in hostname:
                hostname = "Servidor de Inferência " + hostname
            new_line = {
                "Recurso": "Maquina Virtual",
                "Sub-recurso": hostname,
                "Tipo de Custo": "Horas de Funcionamento",
                "Custo por Hora ($)": on_demand_hour,
                "Custo ($)": vm_cost,
            }
            cost_lines.append(new_line)

    """vms = set([line["Sub-recurso"] for line in cost_lines if line["Recurso"] == "Maquina Virtual"])

    for instance_name in vms:
        lines_group = [x for x in cost_lines 
            if x ["Sub-recurso"] == instance_name]"""

    new_line: dict = {
        "Recurso": "Maquina Virtual",
        "Sub-recurso": "Total Utilizado",
        "Tipo de Custo": "Horas de Funcionamento",
    }
    new_line["Custo ($)"] = sum(
        [row["Custo ($)"] for row in cost_lines if row["Recurso"] == "Maquina Virtual"]
    )
    new_line["Custo por Hora ($)"] = sum(
        [
            row["Custo por Hora ($)"]
            for row in cost_lines
            if row["Recurso"] == "Maquina Virtual"
        ]
    )

    cost_lines.append(new_line)

    new_line = {
        "Recurso": "Projeto Hermes",
        "Sub-recurso": None,
        "Tipo de Custo": "Projeto",
    }
    lines_group = [
        x
        for x in cost_lines
        if "Total" in x["Sub-recurso"] and "Utilizado" in x["Sub-recurso"]
    ]
    all_data_columns = [
        "Horas de Áudio",
        "Tokens de Input (milhões)",
        "Tokens de Output (milhões)",
        "Custo ($)",
    ]

    for col in all_data_columns:
        vals = [x[col] for x in lines_group if col in x]
        vals = [x for x in vals if x is not None]
        if len(vals) > 0:
            if "%" in col:
                mean_used = sum(vals) / len(vals)
                new_line[col] = mean_used
            else:
                new_line[col] = sum(vals)
        else:
            new_line[col] = None
    cost_lines.append(new_line)

    for cl in cost_lines:
        cl["Por Ligação ($)"] = cl["Custo ($)"] / n_calls
    tipo_de_custo_para_importancia = {
        "Horas de Funcionamento": 1,
        "API": 2,
        "Tokens": 3,
        "Projeto": 10,
    }
    cost_lines.sort(
        key=lambda l: (
            tipo_de_custo_para_importancia[l["Tipo de Custo"]],
            l["Custo ($)"],
            l["Recurso"],
            l["Sub-recurso"] if l["Sub-recurso"] is not None else "Z",
        )
    )
    df = pl.DataFrame(cost_lines)
    # df.drop_in_place("Tipo de Custo")
    return df
