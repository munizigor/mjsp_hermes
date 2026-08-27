import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors

df_path = "results/df_simples.csv"

# Markers for each model type (only filled formats)
model_type_to_format = {
    "Gliner v1": "o",
    "Gliner v2": "X",
    "Geração de Texto - Abertos": "^",
    "Geração de Texto - Comerciais": "*",
}

model_type_to_colors = {
    "Gliner v1": "red",
    "Gliner v2": "darkred",
    "Geração de Texto - Abertos": "blue",
    "Geração de Texto - Comerciais": "orange",
}

serpro_open_models = [
    "llama-3.1-8B-instruct",
    "qwen3.5-35b",
    "mistral-small-3.2-24b-instruct",
    "magistral-small",
    "gemma-3n-e4b-it",
    "gpt-oss-120b",  # 117B parameters with 5.1B active parameters
    "gemma-3-4b-it",
    "deepseek-r1-distill-qwen-14b",
]


def get_ner_model_type(ner_name):
    """
    Gliner v1
    Gliner v2
    Geração de Texto - Abertos
    Geração de Texto - Comerciais"""

    serpro_names_lower = [x.lower() for x in serpro_open_models]

    if ner_name.lower() in serpro_names_lower:
        return "Geração de Texto - Abertos"
    elif "gliner" in ner_name.lower() or "nuner" in ner_name.lower():
        if "gliner2" in ner_name.lower():
            return "Gliner v2"
        else:
            return "Gliner v1"
    elif "gpt" in ner_name.lower():
        return "Geração de Texto - Comerciais"
    elif ner_name.lower().startswith("o") and (
        ner_name.lower()[-1].isdigit() or ner_name.lower()[1].isdigit()
    ):
        # o3, o4, o4.1...
        return "Geração de Texto - Comerciais"
    elif "gemini" in ner_name.lower():
        return "Geração de Texto - Comerciais"
    elif "nuextract" in ner_name.lower():
        return "Geração de Texto - Abertos"
    else:
        print("Other is", ner_name)
        return "Geração de Texto - Abertos"


def sep_model_types(df):
    # Adds a column for type of LLM
    df["Tipo de Modelo"] = df["ner_model"].apply(get_ner_model_type)
    return df


def simple_hf_name(df):
    # If the model is a HF model, simplify the name to the last part
    df["simple_name"] = df["ner_model"].apply(
        lambda x: x.split("/")[-1] if "/" in x else x
    )
    df["simple_name_clf"] = df["classification_model"].apply(
        lambda x: x.split("/")[-1] if "/" in x else x
    )
    return df


def model_subtype(df):
    # Replace - and _ with ' ', identify the first word
    # Remove first word, remove _ / - from start of string
    def to_subtype(row):
        simple_name = row["simple_name"]
        model_type = row["Tipo de Modelo"]
        if model_type == "Geração de Texto - Abertos":
            simple_name = simple_name.replace("-Instruct-2507-FP8", "")
            simple_name = simple_name.replace("-Instruct-2512", "")
            if simple_name.endswith("-instruct"):
                simple_name = simple_name[:-9]
            if simple_name.endswith("-it"):
                simple_name = simple_name[:-3]
            return simple_name
        elif model_type == "Geração de Texto - Comerciais":
            return simple_name
        elif model_type == "Gliner v1":
            ner_simple = simple_name.replace("gliner", "").lstrip("-_").strip()
            clf_simple = row["simple_name_clf"]

            return f"{ner_simple} + {clf_simple}"
        else:
            words = simple_name.replace("-", " ").replace("_", " ").split(" ")
            first_w = words[0]
            if len(words) > 0:
                if first_w in ["o3", "o4", "o4.1"]:
                    return simple_name
                return simple_name.replace(first_w, "").strip(" -_")
            else:
                return simple_name

    df["subtype"] = df.apply(to_subtype, axis=1)
    return df


def plot_cost_vs_latency_vs_quality(df):
    """
    X axis is cost, Y axis is latency, color is quality
    Good quality should be blue, bad quality should be red
    """
    fig, ax = plt.subplots(figsize=(12, 7.5))

    # If the model is Gliner v1, Gliner v2 or Others, keep only the rows where the clf model is knowledgator/gliclass-large-v3.0
    df_gliners = df[
        (df["Tipo de Modelo"] == "Gliner v1")
        | (df["Tipo de Modelo"] == "Gliner v2")
        | (df["Tipo de Modelo"] == "Others")
    ]
    df_gliners = df_gliners[
        df_gliners["classification_model"] == "knowledgator/gliclass-large-v3.0"
    ]

    # df_not_gliners = df[(df["Tipo de Modelo"] != "Gliner v1") & (df["Tipo de Modelo"] != "Gliner v2") & (df["Tipo de Modelo"] != "Others")]
    # df_not_gliners = df[~df["Tipo de Modelo"].str.contains("Gliner")]
    # df = pd.concat([df_gliners, df_not_gliners])

    # keep onlythe top 18 results, using general quality

    tipos_bons = df["Tipo de Modelo"].unique()
    print(tipos_bons)

    # Predefine colors in RdBu scale, to be able to plot types separatly,
    # but only from smaller quality to larger quality
    low = df["Qualidade Geral"].min()
    high = df["Qualidade Geral"].max()

    # Make custom scale
    norm = plt.Normalize(low, high)
    cmap = plt.cm.RdBu

    df["color"] = df["Qualidade Geral"].apply(lambda x: cmap(norm(x)))
    df_to_name = df.nlargest(22, "Qualidade Geral")

    for model_type, type_df in df.groupby("Tipo de Modelo"):
        scatter = ax.scatter(
            type_df["total_cost"],
            type_df["total_runtime"],
            c=type_df["color"],
            marker=model_type_to_format[model_type],
            label=model_type,
            linewidths=0,
            s=190,
            alpha=0.9,
        )

    for model_type, type_df in df_to_name.groupby("Tipo de Modelo"):
        scatter = ax.scatter(
            type_df["total_cost"],
            type_df["total_runtime"],
            c=type_df["color"],
            marker=model_type_to_format[model_type],
            label=model_type,
            linewidths=1,
            edgecolors="black",
            s=230,
            alpha=0.95,
        )

    for index, row in df.iterrows():
        print(row["Tipo de Modelo"], row["subtype"], row["Qualidade Geral"])

    ax.set_xlabel("Custo do Teste ($)")
    ax.set_ylabel("Latência Total (minutos)")
    ax.set_title("Custo vs Latência")

    ax.set_xscale("log")
    ax.set_yscale("log")

    formatter = mticker.ScalarFormatter()
    formatter.set_scientific(False)
    # Optional: Set a limit for when scientific notation should be used (e.g., set to (0,0) to include all numbers)
    # formatter.set_powerlimits((0, 0))

    # 5. Apply the formatter to both major and minor ticks for the x and y axes
    ax.xaxis.set_major_formatter(formatter)
    # ax.xaxis.set_minor_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    # ax.yaxis.set_minor_formatter(formatter)

    # Add colorbar to figure, but using the custom scale
    # Cria um objeto mapeável explicitamente com as suas regras de cor
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Evita warnings em algumas versões do Matplotlib

    # Adiciona a barra de cores usando o ScalarMappable e os valores reais para os ticks
    cbar = fig.colorbar(sm, ax=ax, ticks=[low, (low + high) / 2, high])

    cbar.set_label("Qualidade Geral")
    cbar.ax.set_yticklabels([f"{low:.2f}", f"{(low+high)/2:.2f}", f"{high:.2f}"])

    # Legend outside plot of model types, with custom markers
    # Placed below graph to not overlap colorbar
    legend_elements = []
    for model_type, marker in model_type_to_format.items():
        if model_type not in tipos_bons:
            continue
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                label=model_type,
                markerfacecolor="k",
                markersize=8,
            )
        )
    ax.legend(
        handles=legend_elements, bbox_to_anchor=(0.5, -0.1), loc="upper center", ncol=5
    )

    # Add text labels for all results
    for i, row in df_to_name.iterrows():
        # Centered text, semi-transparent, small font
        annot_point = (row["total_cost"], row["total_runtime"])
        text_point = (60, 60)
        ax.annotate(
            row["subtype"],
            annot_point,
            xytext=text_point,
            textcoords="offset pixels",
            arrowprops={"arrowstyle": "simple"},
            ha="center",
            va="bottom",
            fontsize=8,
            alpha=0.85,
        )

    # Save plot
    plt.savefig("results/cost_vs_latency_vs_quality.png", dpi=300, bbox_inches="tight")
    plt.savefig("results/cost_vs_latency_vs_quality.svg", dpi=300, bbox_inches="tight")
    plt.close()


def plot_cost_vs_latency_vs_quality2(df):
    """
    X axis is quality, Y axis is cost, color is latency
    Fast should be green, slow should be red
    """
    fig, ax = plt.subplots(figsize=(12, 7.5))

    # df_not_gliners = df[(df["Tipo de Modelo"] != "Gliner v1") & (df["Tipo de Modelo"] != "Gliner v2") & (df["Tipo de Modelo"] != "Others")]
    # df_not_gliners = df[~df["Tipo de Modelo"].str.contains("Gliner")]
    # df = pd.concat([df_gliners, df_not_gliners])

    # keep onlythe top 18 results, using general quality

    tipos_bons = df["Tipo de Modelo"].unique()
    print(tipos_bons)

    # Predefine colors in RdBu scale, to be able to plot types separatly,
    # but only from smaller quality to larger quality
    low = df["seconds_per_transcript"].min()
    high = df["seconds_per_transcript"].max()

    # Make custom scale
    # norm = plt.Normalize(low, high)
    norm = mcolors.LogNorm(low, high)
    cmap = plt.cm.jet

    # Invert the colormap
    # cmap = cmap.reversed()

    df["color"] = df["seconds_per_transcript"].apply(lambda x: cmap(norm(x)))

    df_gliners = df[df["Tipo de Modelo"] == "Gliner v1"]
    df_not_gliner = df[df["Tipo de Modelo"] != "Gliner v1"]

    df_to_name = pd.concat([df_gliners.nlargest(5, "Qualidade Geral"), df_not_gliner])

    for model_type, type_df in df.groupby("Tipo de Modelo"):
        scatter = ax.scatter(
            type_df["Qualidade Geral"],
            type_df["cost_per_transcript"],
            c=type_df["color"],
            marker=model_type_to_format[model_type],
            label=model_type,
            linewidths=0,
            s=190,
            alpha=1,
        )

    for model_type, type_df in df_to_name.groupby("Tipo de Modelo"):
        scatter = ax.scatter(
            type_df["Qualidade Geral"],
            type_df["cost_per_transcript"],
            c=type_df["color"],
            marker=model_type_to_format[model_type],
            label=model_type,
            linewidths=1,
            edgecolors="black",
            s=230,
            alpha=1,
        )

    for index, row in df.iterrows():
        print(row["Tipo de Modelo"], row["subtype"], row["Qualidade Geral"])

    ax.set_xlabel("Qualidade Geral")
    ax.set_ylabel("Custo por Transcrição ($)")
    ax.set_title("Custo-Benefício de Modelos de Extração de Informação")

    # ax.set_xscale("log")
    ax.set_yscale("log")

    formatter = mticker.ScalarFormatter()
    formatter.set_scientific(False)
    # Optional: Set a limit for when scientific notation should be used (e.g., set to (0,0) to include all numbers)
    # formatter.set_powerlimits((0, 0))

    # 5. Apply the formatter to both major and minor ticks for the x and y axes
    # ax.xaxis.set_major_formatter(formatter)
    # ax.xaxis.set_minor_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    # ax.yaxis.set_minor_formatter(formatter)

    # Add colorbar to figure, but using the custom scale
    # Cria um objeto mapeável explicitamente com as suas regras de cor
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Evita warnings em algumas versões do Matplotlib

    # Adiciona a barra de cores usando o ScalarMappable e os valores reais para os ticks
    # mid = (low + high) / 2
    points_locs = [low, 0.5, 1, 2, 4, 8, 16, high]
    cbar = fig.colorbar(sm, ax=ax, ticks=points_locs)
    cbar.ax.minorticks_off()

    cb_ticklabels = [f"{x:.2f}" for x in points_locs]

    cbar.set_label("Tempo por Transcrição (s)")
    cbar.ax.set_yticklabels(cb_ticklabels)

    # Legend outside plot of model types, with custom markers
    # Placed below graph to not overlap colorbar
    legend_elements = []
    for model_type, marker in model_type_to_format.items():
        if model_type not in tipos_bons:
            continue
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                label=model_type,
                markerfacecolor="k",
                markersize=8,
            )
        )
    ax.legend(
        handles=legend_elements, bbox_to_anchor=(0.5, -0.1), loc="upper center", ncol=5
    )

    import matplotlib.patheffects as patheffects

    # Add text labels for all results
    for i, row in df_to_name.iterrows():
        # Centered text, semi-transparent, small font
        annot_point = (row["Qualidade Geral"], row["cost_per_transcript"])
        text_point = (0, 0)
        ax.annotate(
            row["subtype"],
            annot_point,
            # xytext=text_point,
            # textcoords="offset pixels",
            # arrowprops={"arrowstyle": "simple"},
            ha="center",
            va="center",
            fontsize=8,
            alpha=0.85,
            path_effects=[
                patheffects.withStroke(
                    linewidth=2, foreground="white", capstyle="round", alpha=0.75
                )
            ],
        )

    # Save plot
    plt.savefig("results/cost_vs_latency_vs_quality2.png", dpi=400, bbox_inches="tight")
    plt.savefig("results/cost_vs_latency_vs_quality2.svg", dpi=400, bbox_inches="tight")
    plt.close()


def ner_vs_clf_quality(df):
    # Scatter all the results, name all
    # With the exception of gliner models: name only the top 3 and gliner v2
    df_gliners = df[df["Tipo de Modelo"] == "Gliner v1"]
    df_gliners_top3 = df_gliners.nlargest(3, "Qualidade Geral")
    not_gliner_v1 = df[df["Tipo de Modelo"] != "Gliner v1"]
    df_named = pd.concat([df_gliners_top3, not_gliner_v1])

    tipos_encontrados = df["Tipo de Modelo"].unique()

    fig, ax = plt.subplots(figsize=(12, 7.5))

    ax.set_xlabel("Reconhecimento de Entidades")
    ax.set_ylabel("Classificação")
    ax.set_title("Qualidade de Extração de Informações")

    # Scatter all the results, but name only the 20 best and gliner v2
    for model_type, type_df in df.groupby("Tipo de Modelo"):
        color = model_type_to_colors[model_type]
        scatter = ax.scatter(
            type_df["Qualidade Reconhecimento de Entidades"],
            type_df["Qualidade Classificação"],
            c=color,
            marker=model_type_to_format[model_type],
            label=model_type,  # linewidths=1, edgecolors="black",
            s=170,
            alpha=0.8,
        )

    for model_type, type_df in df_named.groupby("Tipo de Modelo"):
        color = model_type_to_colors[model_type]
        scatter = ax.scatter(
            type_df["Qualidade Reconhecimento de Entidades"],
            type_df["Qualidade Classificação"],
            c=color,
            marker=model_type_to_format[model_type],
            linewidths=0.6,
            edgecolors="black",
            s=200,
            alpha=0.9,
        )

    # Legend outside plot of model types, with custom markers
    # Placed below graph to not overlap colorbar
    """legend_elements = []
    for model_type, marker in model_type_to_format.items():
        if model_type not in tipos_encontrados:
            continue
        legend_elements.append(plt.Line2D([0], [0], marker=marker, color='w', 
            label=model_type, markerfacecolor='k', markersize=8))"""
    ax.legend(bbox_to_anchor=(0.5, -0.1), loc="upper center", ncol=5)

    # Add text labels for all results
    for i, row in df_named.iterrows():
        # Centered text, semi-transparent, small font
        point_coords = (
            row["Qualidade Reconhecimento de Entidades"],
            row["Qualidade Classificação"],
        )
        xy_offset = (60, 60)
        arrow_props = {"arrowstyle": "simple"}
        ax.annotate(
            row["subtype"],
            point_coords,
            xytext=xy_offset,
            textcoords="offset pixels",
            arrowprops=arrow_props,
            ha="center",
            va="bottom",
            fontsize=8,
            alpha=0.85,
        )

    # Save plot
    plt.savefig("results/ner_vs_clf_quality.png", dpi=300, bbox_inches="tight")
    plt.savefig("results/ner_vs_clf_quality.svg", dpi=300, bbox_inches="tight")
    plt.close()


def cost_benefit(df):
    # Qualidade Geral VS Power-Delay Product
    df_gliners = df[df["Tipo de Modelo"] == "Gliner v1"]
    df_gliners_top3 = df_gliners.nlargest(3, "Qualidade Geral")
    not_gliner_v1 = df[df["Tipo de Modelo"] != "Gliner v1"]
    df_named = pd.concat([df_gliners_top3, not_gliner_v1])

    fig, ax = plt.subplots(figsize=(12, 7.5))

    ax.set_xlabel("Power-Delay Product")
    ax.set_ylabel("Qualidade Geral")
    ax.set_title("Custo-Benefício")

    # scatter by model type
    for model_type, type_df in df.groupby("Tipo de Modelo"):
        tp_color = model_type_to_colors[model_type]
        ax.scatter(
            type_df["Power–delay product"],
            type_df["Qualidade Geral"],
            c=tp_color,
            s=200,
            alpha=0.8,
        )

    # put x axis in log scale, exactly like in other plots
    ax.set_xscale("log")
    formatter = mticker.ScalarFormatter()
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)

    # Legend outside plot of model types, with custom markers
    # Placed below graph to not overlap colorbar
    legend_elements = []
    for model_type, marker in model_type_to_format.items():
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                color=model_type_to_colors[model_type],
                label=model_type,
                markerfacecolor=model_type_to_colors[model_type],
                markersize=8,
            )
        )
    ax.legend(
        handles=legend_elements, bbox_to_anchor=(0.5, -0.1), loc="upper center", ncol=5
    )

    for i, row in df_named.iterrows():
        ax.annotate(
            row["subtype"],
            (row["Power–delay product"], row["Qualidade Geral"]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=8,
        )

    plt.savefig("results/cost_benefit.png", dpi=300, bbox_inches="tight")
    plt.savefig("results/cost_benefit.svg", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    df = pd.read_csv(df_path)

    # Converter latência para minutos
    df["total_runtime"] = df["total_runtime"] / 60

    df = sep_model_types(df)
    df = simple_hf_name(df)
    df = model_subtype(df)

    # plot_cost_vs_latency_vs_quality(df)
    plot_cost_vs_latency_vs_quality2(df)
    # ner_vs_clf_quality(df)
    # cost_benefit(df)

    print("Modelos por tipo:", df["Tipo de Modelo"].unique())
    for model_type, type_df in df.groupby("Tipo de Modelo"):
        print(model_type)
        ner_names = type_df["simple_name"].unique()
        for ner_name in ner_names:
            print("\t", ner_name)

    print("Modelos de classificação:")
    for clf_name in df["classification_model"].unique():
        print(clf_name.split("/")[-1])

    df_apresentacao = df[
        [
            "subtype",
            "Tipo de Modelo",
            # "Custo-Benefício",
            "Qualidade Geral",
            "Power–delay product",
            "Qualidade Classificação",
            "Qualidade Reconhecimento de Entidades",
            "total_cost",
            "cost_per_transcript",
            "success_rate",
            "seconds_per_transcript",
            "total_runtime",
        ]
    ]

    df_apresentacao = df_apresentacao.rename(
        columns={
            "subtype": "Modelo",
            "Tipo de Modelo": "Tipo",
            "Qualidade Classificação": "Qualidade - Classificação",
            "Qualidade Reconhecimento de Entidades": "Qualidade - Reconhecimento de Entidades",
            "total_cost": "Custo Total",
            "cost_per_transcript": "Custo por Transcrição",
            "success_rate": "Taxa de Sucesso",
            "seconds_per_transcript": "Tempo por Transcrição",
            "total_runtime": "Tempo Total",
        }
    )

    df_apresentacao.to_csv("results/df_apresentacao.csv", index=False)
    # save xlsx
    df_apresentacao.to_excel("results/df_apresentacao.xlsx", index=False)
