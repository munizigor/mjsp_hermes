import polars as pl
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_metainfo(metainfo_df, results_prefix):
    print(metainfo_df)

    # --- Novo código para boxplot comparativo ---

    # Filtra os tempos de resultado para cada modelo
    gliner_times = metainfo_df.filter(
        (pl.col("model") == "gliner")
    ).get_column("result_time").to_list()

    gaia_times = metainfo_df.filter(
        (pl.col("model") == "cnmoro/gemma3-gaia-ptbr-4b")
    ).get_column("result_time").to_list()

    # Cria o boxplot
    fig, ax = plt.subplots(1,1,figsize=(5, 8))
    ax.boxplot([gliner_times, gaia_times], labels=["gliner", "gaia"], patch_artist=True)
    ax.set_ylabel("Tempo de resposta (s)")
    ax.set_title("Comparação de tempo de resposta dos modelos")
    ax.grid(axis='y')
    fig.savefig(results_prefix + '.result_time.png', dpi=180)

    prompt_time = metainfo_df.filter(
        (pl.col("model") == "cnmoro/gemma3-gaia-ptbr-4b")
    ).get_column("prompt_time").to_list()
    result_time = metainfo_df.filter(
        (pl.col("model") == "cnmoro/gemma3-gaia-ptbr-4b")
    ).get_column("result_time").to_list()
    time_other_operations = metainfo_df.filter(
        (pl.col("model") == "cnmoro/gemma3-gaia-ptbr-4b")
    ).get_column("time_other_operations").to_list()
    result_tokens_per_second = metainfo_df.filter(
        (pl.col("model") == "cnmoro/gemma3-gaia-ptbr-4b")
    ).get_column("result_tokens_per_second").to_list()

    # Cria o boxplot
    fig, axes = plt.subplots(1,2,figsize=(12, 8))
    time_ax = axes[0]
    tokens_ax = axes[1]
    time_ax.boxplot([prompt_time, result_time, time_other_operations], 
        labels=["prompt_time", "result_time", "time_other_operations"], 
        patch_artist=True)
    time_ax.set_ylabel("Tempo Gasto (s)")
    time_ax.set_title("Tempo Gasto Pelo Modelo GAIA (RTX 3050 6GB)")
    time_ax.grid(axis='y')

    tokens_ax.boxplot([result_tokens_per_second], 
        labels=["result_tokens_per_second"], 
        patch_artist=True)
    tokens_ax.set_ylabel("Número de Tokens")
    tokens_ax.grid(axis='y')

    fig.savefig(results_prefix +'.gaia_time.png', dpi=180)