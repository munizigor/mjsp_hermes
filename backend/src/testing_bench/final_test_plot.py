import os
import json
import re
import matplotlib.pyplot as plt
import numpy as np
import sys
import requests
import numpy as np


def get_dolar_rate():
    url = "https://economia.awesomeapi.com.br/json/last/USD-BRL"
    response = requests.get(url).json()
    return float(response["USDBRL"]["bid"])


rate = get_dolar_rate()
print(f"Cotação Atual: R$ {rate:.2f}")

# 1. Configurações e Expressão Regular
# Expressão para capturar: <nome_do_teste>-n_<...>-cl_<carga>-<data>
# Exemplo: gcp.open_mid-n_72-cl_12-15-05-2026_12-15-28
padrao_pasta = re.compile(
    r"^(.+?)-n_\d+-cl_([\d\.]+)-\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2}$"
)

# Dicionário para armazenar os dados extraídos
dados_extraidos = {}

# Mapeamento de cores (Baseado nas APIs em azul e Open em vermelho)
cores = {
    "gcp.apis_cost": "#5DADE2",  # Azul claro
    "gcp.apis_mid": "#2874A6",  # Azul médio
    "gcp.apis_qual": "#1B4F72",  # Azul escuro
    "gcp.open_cost": "#8B0000",  # Vermelho escuro
    "gcp.open_mid": "#E74C3C",  # Vermelho
    "gcp.open_qual": "#F1948A",  # Salmão
}
cor_padrao = "#808080"  # Cinza caso apareça uma nova config

# 2. Varredura e Leitura dos Dados
for nome_pasta in os.listdir(sys.argv[1]):
    if not os.path.isdir(os.path.join(sys.argv[1], nome_pasta)):
        continue  # Pula arquivos (como duckdb, pngs, jsons isolados)

    match = padrao_pasta.match(nome_pasta)
    if match:
        nome_config = match.group(1)
        carga = float(match.group(2))

        caminho_json = os.path.join(sys.argv[1], nome_pasta, "final_stats.json")

        if os.path.exists(caminho_json):
            with open(caminho_json, "r") as f:
                try:
                    stats = json.load(f)
                    rtf_medio = stats.get("rtf_medio")
                    custo = stats.get("cost_per_call")
                    carga = round(stats.get("carga_de_emergencias"), 1)

                    if rtf_medio is not None and custo is not None:
                        if nome_config not in dados_extraidos:
                            dados_extraidos[nome_config] = []

                        dados_extraidos[nome_config].append(
                            {
                                "carga": carga,
                                "rtf_medio": rtf_medio,
                                "custo": custo * rate,
                            }
                        )
                except json.JSONDecodeError:
                    print(f"Erro ao ler JSON da pasta: {nome_pasta}")
print(json.dumps(dados_extraidos, indent=2))
# 3. Preparação e Plotagem do Gráfico
plt.figure(figsize=(12, 8), facecolor="white")
ax = plt.gca()

cargas_plotadas = set()  # Para a legenda de tamanhos
max_carga_geral = 0

for nome_config, testes in dados_extraidos.items():
    # Ordenar os testes dessa configuração pela carga (crescente) para traçar a linha corretamente
    testes_ordenados = sorted(testes, key=lambda x: x["carga"])

    rtfs = [t["rtf_medio"] for t in testes_ordenados]
    custos = [t["custo"] for t in testes_ordenados]
    cargas = [t["carga"] for t in testes_ordenados]

    cor = cores.get(nome_config, cor_padrao)

    # Plotagem da linha interligada
    plt.plot(custos, rtfs, color=cor, alpha=0.4, linestyle="--", linewidth=2, zorder=1)

    # Plotagem dos pontos (Scatter)
    for rtf, custo, carga in zip(rtfs, custos, cargas):
        area = carga * 40  # Multiplicador para o tamanho visual do ponto
        max_carga_geral = max(max_carga_geral, carga)
        cargas_plotadas.add(carga)

        plt.scatter(
            custo,
            rtf,
            s=area,
            color=cor,
            edgecolor="black",
            alpha=0.9,
            zorder=2,
            # label=nome_config if carga == cargas else "",
        )

        # Adicionar rótulo apenas no ponto de maior carga daquela linha para não poluir
        if carga == cargas[-1]:
            plt.annotate(
                f"{carga}",
                (custo, rtf),
                xytext=(-14, 16),
                textcoords="offset points",
                fontsize=12,
                fontweight="bold",
                color=cor,
            )

# 4. Estilização do Gráfico
plt.title(
    "Tempo Real vs Custo: Impacto da Carga de Ligações",
    fontsize=18,
    pad=20,
    fontweight="bold",
    color="#2c3e50",
)
plt.ylabel("RTF Médio (Fator de Tempo Real)", fontsize=13, labelpad=15)
plt.xlabel("Custo por Ligação (R$)", fontsize=13, labelpad=15)

# Formatação do eixo Y para muitas casas decimais (já que o custo é 0.003...)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.5f}".format(x)))

# Grid sutil
plt.grid(True, which="major", linestyle=":", color="black", alpha=0.4)

# 5. Legendas
# Legenda 1: Cores (Configurações)
conf_handles = []
conf_labels = []
for nome_config in dados_extraidos.keys():
    conf_handles.append(
        plt.scatter(
            [],
            [],
            s=80,
            color=cores.get(nome_config, cor_padrao),
            edgecolor="black",
        )
    )
    nome_bonito = ""
    if "gcp.apis" in nome_config:
        nome_bonito = "Modelo Gemini"
        if "cost" in nome_config:
            nome_bonito += " 2.5 Flash-lite"
        elif "mid" in nome_config:
            nome_bonito += " 2.5 Flash"
        elif "qual" in nome_config:
            nome_bonito += " 3.0 Flash"
    elif "gcp.open" in nome_config:
        nome_bonito = "Modelos Abertos"
        if "cost" in nome_config:
            nome_bonito += " - Gliner + Ministral 3B"
        elif "mid" in nome_config:
            nome_bonito += " - Ministral 3B"
        elif "qual" in nome_config:
            nome_bonito += " - Ministral 8B"

    conf_labels.append(nome_bonito)

legenda_configs = plt.legend(
    conf_handles,
    conf_labels,
    title="Configurações",
    loc="upper right",
    bbox_to_anchor=(1.01, 1.01),  # Posiciona no espaço em branco superior esquerdo
    fontsize=10,
)

# Legenda 2: Tamanhos (Carga)
ax.add_artist(legenda_configs)
carga_mediana = round(np.median(list(cargas_plotadas)), 2)
tamanhos_legenda = [min(cargas_plotadas), carga_mediana, max(cargas_plotadas)]
handles_tamanhos = [
    plt.scatter([], [], s=c * 40, c="gray", alpha=0.5, edgecolor="black")
    for c in tamanhos_legenda
]
labels_tamanhos = [f"{c:.1f}" for c in tamanhos_legenda]

# Posicionando a segunda legenda logo abaixo da primeira
plt.legend(
    handles_tamanhos,
    labels_tamanhos,
    title="Ligações\nSimultâneas",
    loc="upper right",
    bbox_to_anchor=(
        1.01,
        0.76,
    ),
    fontsize=11,
    labelspacing=1.4,
)

# Remover bordas
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Desenhar limiar do "tempo real" (rtf = 1)
plt.axhline(y=1, color="green", linestyle="--", linewidth=2, zorder=1)
plt.annotate(
    "RTF = 1 (Tempo Real)",
    (0.07, 1),
    xytext=(0.01, 7),
    textcoords="offset points",
    fontsize=12,
    fontweight="bold",
    color="green",
)

# Removido o tight_layout() pois o bbox_inches='tight' abaixo já faz o serviço corretamente
nome_arquivo_saida = f"{sys.argv[1]}/final_test_plot.png"
plt.savefig(nome_arquivo_saida, dpi=300, bbox_inches="tight")
print(f"Sucesso! Gráfico salvo como {nome_arquivo_saida}")
