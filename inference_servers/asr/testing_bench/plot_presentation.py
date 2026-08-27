import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Mapping model families to markers and colors for consistent visualization
MODEL_FAMILY_MARKERS = {
    "Google Chirp": "o",
    "OpenAI Whisper": "s",
    "Qwen": "h",
    "Meta MMS": "p",
    "Azure": "D"
}

MODEL_FAMILY_COLORS = {
    "Google Chirp": "#e74c3c",   # Red
    "OpenAI Whisper": "#c0392b", # Dark Red
    "Qwen": "#2980b9",           # Blue
    "Meta MMS": "#f39c12",       # Orange
    "Azure": "#7f8c8d"           # Gray
}

def plot_cost_vs_latency_vs_quality(df, base_dir):
    """
    X axis: Cost per 1h Audio ($)
    Y axis: Mean Latency (s)
    Color: Word Error Rate (WER %) - RdBu_r (Blue is better/lower error)
    """
    fig, ax = plt.subplots(figsize=(12, 7.5))

    # Calculate color normalization based on WER (lower is better)
    low_wer = df["Word Error Rate (WER)"].min()
    high_wer = df["Word Error Rate (WER)"].max()

    # RdBu_r: Red for high error, Blue for low error
    norm = plt.Normalize(low_wer, high_wer)
    cmap = plt.cm.RdBu_r
    df["color"] = df["Word Error Rate (WER)"].apply(lambda x: cmap(norm(x)))
    
    # Identify top 15 models with the lowest WER for highlighting
    df_top = df.nsmallest(15, "Word Error Rate (WER)")

    # Plot all points grouped by family for the legend
    for family, family_df in df.groupby("Família do Modelo"):
        marker = MODEL_FAMILY_MARKERS.get(family, "D")
        ax.scatter(
            family_df["Custo 1h Áudio ($)"], 
            family_df["Latência Média (s)"], 
            c=family_df["color"], 
            marker=marker, 
            label=family, 
            s=190, 
            alpha=0.75,
            linewidths=0
        )

    # Re-plot the best performing models with black edges
    for family, family_df in df_top.groupby("Família do Modelo"):
        marker = MODEL_FAMILY_MARKERS.get(family, "D")
        ax.scatter(
            family_df["Custo 1h Áudio ($)"], 
            family_df["Latência Média (s)"], 
            c=family_df["color"], 
            marker=marker, 
            s=230, 
            alpha=0.95, 
            edgecolors="black", 
            linewidths=1.2
        )

    ax.set_xlabel("Custo por 1h de Áudio ($)")
    ax.set_ylabel("Latência Média (s)")
    ax.set_title("ASR Benchmarking: Cost vs Latency (Colored by WER %)")

    ax.set_xscale("log")
    ax.set_yscale("log")

    # Use scalar formatter to avoid scientific notation on axes
    formatter = mticker.ScalarFormatter()
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    # Add Colorbar for WER
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Word Error Rate (%)")

    # Shape-based legend for model families
    legend_elements = []
    for family in df["Família do Modelo"].unique():
        marker = MODEL_FAMILY_MARKERS.get(family, "D")
        legend_elements.append(plt.Line2D([], [], marker=marker, color='w', 
                                          label=family, markerfacecolor='k', markersize=8))
    
    ax.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.15), 
              loc='upper center', ncol=len(legend_elements), frameon=False)

    # Label the top performing models
    for _, row in df_top.iterrows():
        ax.annotate(
            row["Nome do Modelo"], 
            (row["Custo 1h Áudio ($)"], row["Latência Média (s)"]),
            xytext=(5, 5), textcoords="offset points", fontsize=8, fontweight='bold'
        )
    
    os.makedirs(base_dir, exist_ok=True)
    plt.savefig(os.path.join(base_dir, "cost_vs_latency_vs_quality.png"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_cost_benefit(df, base_dir):
    """
    X axis: Power–delay product (Efficiency)
    Y axis: Word Error Rate (WER %)
    """
    fig, ax = plt.subplots(figsize=(12, 7.5))
    
    # Label the most efficient models with low error
    df_labeled = df.nsmallest(8, "Word Error Rate (WER)")

    for family, family_df in df.groupby("Família do Modelo"):
        color = MODEL_FAMILY_COLORS.get(family, "gray")
        marker = MODEL_FAMILY_MARKERS.get(family, "D")
        ax.scatter(
            family_df["Power–delay product"], 
            family_df["Word Error Rate (WER)"], 
            c=color, 
            marker=marker, 
            label=family, 
            s=200, 
            alpha=0.8
        )

    ax.set_xscale("log")
    ax.set_xlabel("Power–delay product")
    ax.set_ylabel("Word Error Rate (WER %)")
    ax.set_title("Análise de Custo-Benefício")

    formatter = mticker.ScalarFormatter()
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)

    ax.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=5, frameon=False)

    for _, row in df.iterrows():
        ax.annotate(
            row["Nome do Modelo"], 
            (row["Power–delay product"], row["Word Error Rate (WER)"]),
            xytext=(8, 8), textcoords="offset points", fontsize=8
        )
    
    os.makedirs(base_dir, exist_ok=True)
    plt.savefig(os.path.join(base_dir, "cost_benefit.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(base_dir, "cost_benefit.svg"), dpi=300, bbox_inches="tight")
    plt.close()

def get_pretty_name(model_id):
    corrections = {
        "hermes-chirp3-us": "Chirp3",
        "hermes-chirp2-us-central1": "Chirp2",
        "hermes-telephony-us": "Chirp Telephony",
    }
    return corrections.get(model_id, str(model_id))

def get_family(model_id):
    m_id = str(model_id).lower()
    if "chirp" in m_id or "telephony" in m_id:
        return "Google Chirp"
    if "whisper" in m_id:
        return "OpenAI Whisper"
    if "qwen" in m_id:
        return "Qwen"
    if "mms" in m_id:
        return "Meta MMS"
    return "Other"

if __name__ == "__main__":
    # Correctly parse the command line argument
    if len(sys.argv) < 2:
        print("Usage: python plot_presentation.py <results_path>")
        sys.exit(1)
    
    results_path = sys.argv[1]
    if not os.path.exists(results_path):
        print(f"Error: File '{results_path}' not found.")
        sys.exit(1)

    df = pd.read_csv(results_path)
    
    # wer_normalized -> Word Error Rate
    df["Word Error Rate (WER)"] = df["wer_normalized"] * 100
    df["Nome do Modelo"] = df["model_id"].apply(get_pretty_name)
    df["Família do Modelo"] = df["model_id"].apply(get_family)

    # turn wer_no_special, wer_no_punct, wer_no_repeats into percentage of wer_normalized
    df["Porcentagem do WER: Caracteres especiais incorretos (%)"] = (df["wer_normalized"] - df["wer_no_special"]) / df["wer_normalized"] * 100
    df["Porcentagem do WER: Pontuação incorreta (%)"] = (df["wer_normalized"] - df["wer_no_punct"]) / df["wer_normalized"] * 100
    df["Porcentagem do WER: Repetições (%)"] = (df["wer_normalized"] - df["wer_no_repeats"]) / df["wer_normalized"] * 100

    df["Horas de áudio testadas"] = df["total_audio_seconds"] / 3600
    df["Horas de processamento"] = df["total_seconds_chunks"] / 3600
    df["Velocidade"] = df["speed_chunks"]
    df["Custo 1h Áudio ($)"] = df["cost_1h"]
    df["Custo Total Teste ($)"] = df["cost_test"]

    df["Latência Média (s)"] = df["total_seconds_chunks"] / df["n_samples"]
    df["Custo Médio (s)"] = df["cost_test"] / df["n_samples"]

    df["log_avg_runtime"] = np.log(df["Latência Média (s)"] + 1)
    df["Power–delay product"] = (df["log_avg_runtime"] * df["Custo Médio (s)"])
    base_dir = os.path.dirname(results_path) or "."
    # 5. Generate Visualizations
    plot_cost_vs_latency_vs_quality(df, base_dir)
    plot_cost_benefit(df, base_dir)

    final_cols = ["Nome do Modelo", "Família do Modelo", "Word Error Rate (WER)",
        "Power–delay product", "Latência Média (s)", "Custo Médio (s)",
        "Porcentagem do WER: Caracteres especiais incorretos (%)", 
        "Porcentagem do WER: Pontuação incorreta (%)", 
        "Porcentagem do WER: Repetições (%)", 
        "Horas de áudio testadas", "Horas de processamento", "Velocidade", 
        "Custo 1h Áudio ($)", "Custo Total Teste ($)"]
    
    original_dir = os.path.dirname(results_path)
    if not original_dir:
        original_dir = "."
        
    df.to_csv(f"{original_dir}/eval_results_final.complete.csv", index=False)
    
    df_simple = df[final_cols]
    df_simple.to_csv(f"{original_dir}/eval_results_final.csv", index=False)
    # save excel file
    df_simple.to_excel(f"{original_dir}/eval_results_final.xlsx", index=False)