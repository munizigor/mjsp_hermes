from glob import glob
import os
import json
import sys

import pandas as pd
import panel as pn
import plotly.express as px
from matplotlib import pyplot as plt
import numpy as np
import polars as pl

# Define uma extensão e um tema para o dashboard.
# 'bootstrap' é um tema limpo e profissional.
# 'plotly' carrega o backend necessário para renderizar os gráficos.
pn.extension("plotly", template="bootstrap")

pallete_brasil = [
    "#00d000ff",
    "#183fffff",
    "#ffd200ff",
    "#fe0000ff",
    "#373d41ff",
    "#168821ff",
]


def media_deslizante(x, y, n_avgs):
    x_start = min(x)
    x_end = max(x)
    x_diff = x_end - x_start
    x_step = x_diff / n_avgs
    x_range = x_step * 0.7

    current_x = x_start
    all_points = [(x, y) for x, y in zip(x, y)]

    new_x = []
    new_y = []
    while current_x <= x_end:
        window_start = current_x - x_range
        window_end = current_x + x_range
        y_values = [y for x, y in all_points if x >= window_start and x <= window_end]
        if len(y_values) > 0:
            y_mean = np.mean(y_values)
            new_x.append(current_x)
            new_y.append(y_mean)
        current_x += x_step

    return new_x, new_y


def plot_delays(
    test_path,
    inferencia_df: pl.DataFrame,
    hora_primeiro_audio,
    container_reads: pl.DataFrame,
    emergency_audios: pl.DataFrame,
):
    """palete = [
        "#FF6456",
        "#00EEFF",
        "#74FC8B",
        "#E5FF00",
        "#9B35EE",
        "#100c47"
    ]"""
    palete = pallete_brasil
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    ax = axes[0]
    starts = emergency_audios["start_time"].to_list()
    start2 = min(starts)
    end_abs = max(inferencia_df["horario_fim"].to_list()) - start2

    starts = [s - start2 for s in starts]
    # print("Audio starts:", starts)
    lens = emergency_audios["audio_length_seconds"].to_list()
    # print("Audio lens:", lens)
    start_ends = [(s, s + le) for s, le in zip(starts, lens)]
    last_end = max([e for s, e in start_ends])

    audio_load_vec = [0 for _ in range(round(last_end))]
    for s, e in start_ends:
        s2 = round(s)
        e2 = round(e)
        for t in range(s2, e2):
            audio_load_vec[t] += 1

    # 1. To get the HISTOGRAM (counts per bin):
    # counts = overlaps.sum(axis=0)
    ax.plot(range(len(audio_load_vec)), audio_load_vec, c="#373d41ff", linewidth=7)
    # ax.set_xlabel('Segundos Após Início do Teste')
    ax.set_ylabel("Chamadas Concorrentes")
    ax.set_title("Carga de Ligações")

    # print(inferencia_df.columns)
    horarios = inferencia_df["horario_fim"].to_list()
    duracoes = (
        inferencia_df["duracao_inferencia"].to_numpy()
        + inferencia_df["duracao_outros_processamentos"].to_numpy()
    )
    duracoes = list(duracoes)
    delays = inferencia_df["delay"].to_list()
    horarios = [h - start2 for h in horarios]
    lista_tipos = inferencia_df["tipo_de_inferencia"].to_list()

    tipos = {t: n for n, t in enumerate(sorted(set(lista_tipos)))}
    colors = [palete[tipos[t]] for t in lista_tipos]

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=palete[n],
            label=l,
            markerfacecolor=palete[n],
            markersize=12,
        )
        for l, n in tipos.items()
    ]

    duracao_por_tipo = {t: [] for t in tipos.keys()}
    delays_por_tipo = {t: [] for t in tipos.keys()}
    horarios_por_tipo = {t: [] for t in tipos.keys()}

    for h, d, d2, t in zip(horarios, duracoes, delays, lista_tipos):
        duracao_por_tipo[t].append(d)
        horarios_por_tipo[t].append(h)
        delays_por_tipo[t].append(d2)

    reg_por_tipo = {}
    reg_delay_por_tipo = {}
    for t in lista_tipos:
        horarios_t = horarios_por_tipo[t]
        duracao_t = duracao_por_tipo[t]
        delay_t = delays_por_tipo[t]
        res = np.polyfit(horarios_t, duracao_t, 2)
        reg_por_tipo[t] = np.poly1d(res)

        res2 = np.polyfit(horarios_t, delay_t, 1)
        reg_delay_por_tipo[t] = np.poly1d(res2)

    # ax[0].scatter(horarios, delays, c=colors, alpha=0.25, edgecolors='none')
    ax = axes[1]
    for t in tipos.keys():
        all_x = horarios_por_tipo[t]
        all_y = delays_por_tipo[t]
        x_points, y_means = media_deslizante(all_x, all_y, 24)
        ax.plot(x_points, y_means, c=palete[tipos[t]], alpha=0.8, linewidth=7)

        """all_y2 = duracao_por_tipo[t]
        x_points, y_means2 = media_deslizante(all_x, all_y2, 50)
        ax[1].plot(x_points, y_means2, c=palete[tipos[t]], alpha=0.8, linewidth=6)"""
    """for t, fn in reg_delay_por_tipo.items():
        #a,b = fn
        h_t = horarios_por_tipo[t]
        xs = np.linspace(min(h_t), max(h_t), 200)
        ax[0].plot(xs, fn(xs), c=palete[tipos[t]], alpha=0.9, linewidth=5)"""
    ax.legend(handles=legend_elements)
    ax.set_xlabel("Segundos Após Início do Teste")
    ax.set_ylabel("Delay da Inferência")
    ax.set_title("Delay Entre Fala e Informação Extraída")
    # ax[0].set_yscale('log')

    # ax[1].scatter(horarios, duracoes, c=colors, alpha=0.25, edgecolors='none')
    """for t, fn in reg_por_tipo.items():
        #a,b = fn
        h_t = horarios_por_tipo[t]
        xs = np.linspace(min(h_t), max(h_t), 200)
        ax[1].plot(xs, fn(xs), c=palete[tipos[t]], alpha=0.9, linewidth=5)"""
    # ax[1].set_xlabel('Segundos Após Início do Teste')
    # ax[1].set_ylabel('Tempo Inferindo')
    # ax[1].set_title('Tempo de Inferência por Tipo de Inferência')

    """ax = axes[2]
    for t in tipos.keys():
        all_x = horarios_por_tipo[t]
        all_y = duracao_por_tipo[t]
        x_points, y_means = media_deslizante(all_x, all_y, 24)
        ax.plot(x_points, y_means, c=palete[tipos[t]], alpha=0.8, linewidth=7)
    ax.legend(handles=legend_elements)
    ax.set_xlabel('Segundos Após Início do Teste')
    ax.set_ylabel('Latência da Inferência')
    ax.set_title('Latência das Requisições')"""
    print(container_reads)
    instance_names = set(container_reads["hostname"].to_list())
    instance_names = sorted(instance_names)

    cpu_by_instance = {i: [] for i in instance_names}
    ram_by_instance = {i: [] for i in instance_names}
    timestamps_by_instance = {i: [] for i in instance_names}
    for hostname in instance_names:
        instance_df = container_reads.filter(pl.col("hostname") == hostname)
        cpu_percs = instance_df["CPUPerc"].to_list()
        mem_percs = instance_df["MemPerc"].to_list()
        times = instance_df["timestamp"].to_list()
        min_t = min(times)
        times = [t - min_t for t in times]
        for c, m, ti in zip(cpu_percs, mem_percs, times):
            cpu_by_instance[hostname].append(c)
            ram_by_instance[hostname].append(m)
            timestamps_by_instance[hostname].append(ti)

    hostname_to_c = {
        instance_names[i]: pallete_brasil[i] for i in range(len(instance_names))
    }
    ax = axes[2]
    for hostname in instance_names:
        all_x = timestamps_by_instance[hostname]
        cpu_y = cpu_by_instance[hostname]
        mem_y = ram_by_instance[hostname]
        x_points, cpu_means = media_deslizante(all_x, cpu_y, 5)
        x_points2, mem_means = media_deslizante(all_x, mem_y, 5)
        ax.plot(
            x_points,
            cpu_means,
            c=hostname_to_c[hostname],
            alpha=0.8,
            linewidth=6,
            linestyle="-",
        )
        ax.plot(
            x_points2,
            mem_means,
            c=hostname_to_c[hostname],
            alpha=0.8,
            linewidth=6,
            linestyle="--",
        )

    legend_elements2 = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=c,
            label="RAM " + n,
            markerfacecolor=c,
            markersize=12,
            linestyle="--",
        )
        for n, c in hostname_to_c.items()
    ]
    legend_elements2 += [
        Line2D(
            [0],
            [0],
            marker="o",
            color=c,
            label="CPU " + n,
            markerfacecolor=c,
            markersize=12,
        )
        for n, c in hostname_to_c.items()
    ]
    ax.legend(handles=legend_elements2)
    ax.set_xlabel("Segundos Após Início do Teste")
    ax.set_ylabel("Uso do Hardware (%)")
    ax.set_title("Uso de CPU e RAM")

    fig.tight_layout()
    fig.savefig(test_path + "/delays.png", dpi=120)

    return test_path + "/delays.png"


def create_tab_qualidade(data):
    # Esta aba já era vertical e não precisa de alterações funcionais,
    # mas ajustei margens e fontes do gráfico para ficar mais estético.
    stats = data["comparison_stats"].copy()
    amostras = stats.pop("amostras", "N/A")
    wer = stats.get("transcricao_wer", 0)
    wer = round(wer * 100, 3)

    semantic_lines = []
    for k, v in stats.items():
        if "_sim" in k:
            k2 = k.replace("_sim", "").replace("_", " ").title()
            semantic_lines.append({"Label": k2, "Valor": v})
    df_qualidade = pd.DataFrame(semantic_lines)

    fig = px.bar(
        df_qualidade,
        x="Valor",
        y="Label",
        orientation="h",
        title="Similaridade Semântica",
        labels={"Valor": "Índice (%)", "Label": "Informações Extraídas"},
        template="plotly_white",
        text=df_qualidade["Valor"].apply(lambda x: f"{x:.2%}"),
        height=420,
    )
    # Layout mais compacto e fontes consistentes
    fig.update_layout(
        title_x=0.5,
        xaxis_range=[0, 1],
        yaxis={"categoryorder": "total ascending"},
        margin=dict(l=220, r=40, t=60, b=60),
        font=dict(family="Arial", size=12),
    )
    fig.update_traces(marker_color="#007bff", textposition="outside")
    fig.update_yaxes(tickfont=dict(size=11))

    titulo = pn.pane.Markdown("## Qualidade Semântica dos Resultados", align="center")
    alerta_amostras = pn.pane.Alert(
        f"Resultados baseados em **{amostras}** amostras.", alert_type="info"
    )
    alerta_wer = pn.pane.Alert(
        f"Word Error Rate (WER) da Transcrição: **{wer}**%.", alert_type="info"
    )
    grafico = pn.pane.Plotly(
        fig, config={"displayModeBar": False}, sizing_mode="stretch_width"
    )

    return pn.Column(
        titulo, alerta_amostras, alerta_wer, grafico, sizing_mode="stretch_width"
    )


def create_tab_desempenho(data, delays_img_path):

    image_pane = pn.pane.Image(
        delays_img_path, sizing_mode="stretch_width", align="center"
    )
    image_pane.servable()

    df_delays = pd.DataFrame(data["delays_medios"]).dropna()

    # Extrai o dado de 'final_stats'
    df_final_stats = pd.DataFrame(data["final_stats"])
    # Proteção caso a métrica não exista
    tempo_transcricao_valor = None
    mask = df_final_stats["Métrica"].str.lower() == "tempo inferencia medio transcricao"
    if mask.any():
        tempo_transcricao_valor = df_final_stats[mask]["Valor"].iloc[0]

    """if tempo_transcricao_valor is not None:
        nova_linha = pd.DataFrame([{'tipo': 'ASR', 'delay medio': tempo_transcricao_valor}])
        df_delays = pd.concat([df_delays, nova_linha], ignore_index=True)"""

    df_delays = df_delays.sort_values("delay medio")
    df_delays["tipo"] = (
        df_delays["tipo"].str.replace("_", " ").str.title().replace("Ner", "NER")
    )

    fig_delays = px.bar(
        df_delays,
        x="tipo",
        y="delay medio",
        title="Delay Médio por Tipo de Inferência",
        labels={"tipo": "Tipo de Inferência", "delay medio": "Delay Médio (segundos)"},
        template="plotly_white",
        text=df_delays["delay medio"].apply(lambda x: f"{x:.2f}s"),
        height=420,
    )
    fig_delays.update_layout(
        title_x=0.5,
        margin=dict(l=40, r=40, t=60, b=80),
        font=dict(family="Arial", size=12),
    )
    fig_delays.update_traces(marker_color="#28a745")

    df_stats = pd.DataFrame(data["final_stats"])
    df_stats["Métrica"] = df_stats["Métrica"].str.title()

    titulo = pn.pane.Markdown("## Delays e Desempenho Computacional", align="center")
    grafico = pn.pane.Plotly(
        fig_delays, config={"displayModeBar": False}, sizing_mode="stretch_width"
    )
    titulo_tabela = pn.pane.Markdown("#### Estatísticas Gerais de Desempenho")
    tabela = pn.widgets.DataFrame(
        df_stats,
        disabled=True,
        fit_columns=True,
        show_index=False,
        sizing_mode="stretch_width",
    )

    card_resumo2 = pn.Card(
        pn.pane.Markdown(
            f"""
        - ASR: Automatic Speech Recognition. Componente que transcreve áudios recebidos;
        - Interpretation - Pipeline interpretativa:
            - Descrição e Observação: Criação de um texto para resumo e outro para observações;
            - Lista de Naturezas: Listagem de naturezas similares;
            - Natureza Decisiva: Decisão de uma natureza adequada;
        - NER: Named Entity Recognition. Extração de informações-chave;
        """
        ),
        title="Tipos de Inferência",
        header_background="#007bff",
        header_color="white",
        sizing_mode="stretch_width",
    )

    return pn.Column(
        titulo,
        image_pane,
        grafico,
        card_resumo2,
        titulo_tabela,
        tabela,
        sizing_mode="stretch_width",
    )


def create_tab_custos(data, proj_dir):
    df_costs = pd.DataFrame(data["costs_df"])
    no_tokens = [c for c in df_costs.columns if "token" not in c.lower()]
    df_costs = df_costs[no_tokens]

    custo_total_valor = df_costs[df_costs["Recurso"] == "Projeto Hermes"][
        "Custo ($)"
    ].iloc[0]
    # custo_sistema_ocioso = df_costs[df_costs['Container'] == 'Sistema Ocioso']['Todos os\nCustos ($)'].iloc[0]
    # if custo_sistema_ocioso != custo_sistema_ocioso or custo_sistema_ocioso == None:
    #    custo_sistema_ocioso = 0.0
    # custo_eficiente = custo_total_valor - custo_sistema_ocioso
    horas_audio = data["final_stats_dict"]["segundos_transcritos"] / 60 / 60
    custo_por_hora = custo_total_valor / horas_audio
    # custo_eficiente_por_hora = custo_total_valor / horas_audio
    # print(df_costs)
    df_costs["Custo Por Hora ($)"] = df_costs["Custo ($)"] / horas_audio

    df_pie = df_costs[
        ~df_costs["Sub-recurso"].isin(["Total Utilizado"])
        & ~df_costs["Recurso"].isin(["Projeto Hermes"])
    ].copy()
    fig_pie = px.pie(
        df_pie,
        names="Sub-recurso",
        values="Custo ($)",
        title="Distribuição de Custos Por Recurso",
        hole=0.3,
        height=500,
    )
    fig_pie.update_layout(
        title_x=0.5,
        margin=dict(l=40, r=40, t=60, b=60),
        font=dict(family="Arial", size=12),
    )
    fig_pie.update_traces(textinfo="percent+label")

    """df_table = df_costs.rename(columns={"Custo de CPU\n($)": "Custo CPU ($)", 
                                        "Custo Total\nAPIs ($)": "Custo APIs ($)", 
                                        "Todos os\nCustos ($)": "Custo Total ($)"})"""
    df_table = df_costs
    for col in df_table.columns:
        if "$" in col:
            df_table[col] = df_table[col].apply(
                lambda x: round(x, 4) if pd.notna(x) else "N/A"
            )

    custo_por_ligacao_valor = df_costs[df_costs["Recurso"] == "Projeto Hermes"][
        "Por Ligação ($)"
    ].iloc[0]
    print(custo_por_ligacao_valor)
    # if custo_por_ligacao_vm_nao_usado != custo_por_ligacao_vm_nao_usado or custo_por_ligacao_vm_nao_usado == None:
    #    custo_por_ligacao_vm_nao_usado = 0.0
    custo_por_ligacao_valor_usado = custo_por_ligacao_valor

    centros_df = pd.read_csv(proj_dir + "/datasets/centros_atendimento.csv", sep=",")
    final_stats_df = pd.DataFrame(data["final_stats"])
    carga = float(
        final_stats_df[final_stats_df["Métrica"] == "carga de emergencias"][
            "Valor"
        ].iloc[0]
    )
    duracao_segundos = final_stats_df[final_stats_df["Métrica"] == "duracao teste"][
        "Valor"
    ].iloc[0]
    print(f"Duracao teste: {duracao_segundos} ({type(duracao_segundos)})")
    duracao_horas = float(duracao_segundos) / 60 / 60
    fracao_dia = 24 / duracao_horas

    horas_dia = horas_audio * fracao_dia
    horas_mes = horas_dia * 31

    ligacoes_teste = data["final_stats_dict"]["numero_de_chamadas"]
    ligacoes_mes = round(ligacoes_teste * fracao_dia * 31)

    new_line = {
        "Unidade": "Esta Simulação",
        "Ligações por Mês": ligacoes_mes,
        "Horas de Atendimento por Mês": horas_mes,
        "Carga de Ligações Média": round(carga, 3),
    }
    df2 = pd.DataFrame([new_line])
    centros_df = pd.concat([centros_df, df2])

    # centros_df['$ por Mês (100% eficiente)'] = centros_df['Horas de Atendimento por Mês'] * custo_eficiente_por_hora
    centros_df["$ por Mês"] = (
        centros_df["Ligações por Mês"] * custo_por_ligacao_valor_usado
    )

    for col in centros_df.columns:
        if "$" in col:
            centros_df[col] = centros_df[col].apply(
                lambda x: f"${x:,.3f}" if pd.notna(x) else "N/A"
            )

    titulo = pn.pane.Markdown("## Análise de Custos Operacionais", align="center")
    card_resumo = pn.Card(
        pn.pane.Markdown(
            f"""
        ## Custo da Simulação: ${custo_total_valor:,.2f}
        ### Custo por hora de ligação: ${custo_por_hora:,.4f}
        ### Custo médio por ligação: ${custo_por_ligacao_valor_usado}
        """
        ),
        title="Resumo dos Custos",
        header_background="#007bff",
        header_color="white",
        sizing_mode="stretch_width",
    )

    grafico = pn.pane.Plotly(
        fig_pie,
        config={"displayModeBar": False},
        height=450,
        sizing_mode="stretch_width",
    )
    card_resumo2 = pn.Card(
        pn.pane.Markdown(
            """
        - Instâncias: containers que compõe o backend do Sistema Hermes. O custo das instâncias é calculado com base no tempo de funcionamento delas;
            - Instância de API: Peça central do backend, responsável pela comunicação;
            - Instâncias de ASR, NER e Interpretation: containers que executam agentes especializados em tarefas específicas;
        - [Nome do Modelo] - [Tarefa]: Custos associados ao uso de uma API de LLM comercial para realizar uma tarefa específica;
            - Tarefa NER: Named Entity Recognition. Extração de informações-chave;
            - Tarefa ASR: Automatic Speech Recognition. Componente que transcreve áudios recebidos;
            - Tarefas DESCRICAO_E_OBSERVACAO, NATUREZA_DECISIVA E ENVOLVIMENTOS: Tarefas que fazem parte do ciclo de 'Interpretation'. Narração dos fatos, observações, determinação de naturezas, envolvidos e envolvimentos;
        """
        ),
        title="Componentes do Sistema",
        header_background="#007bff",
        header_color="white",
        sizing_mode="stretch_width",
    )
    titulo_tabela = pn.pane.Markdown("### Tabela Detalhada de Custos", align="center")
    tabela = pn.widgets.DataFrame(
        df_table,
        disabled=True,
        fit_columns=True,
        show_index=False,
        sizing_mode="stretch_width",
    )
    titulo_tabela2 = pn.pane.Markdown("### Projeções por Mês de Uso", align="center")
    tabela2 = pn.widgets.DataFrame(
        centros_df,
        disabled=True,
        fit_columns=True,
        show_index=False,
        sizing_mode="stretch_width",
    )

    return pn.Column(
        titulo,
        card_resumo,
        grafico,
        card_resumo2,
        titulo_tabela,
        tabela,
        titulo_tabela2,
        tabela2,
        sizing_mode="stretch_width",
    )


def create_tab_config(data):
    config_json = data["config"]
    titulo = pn.pane.Markdown("## Configuração da Simulação", align="center")
    card_api = pn.Card(
        pn.pane.Markdown(
            f"""
        - Processos: {config_json['hermes-api']['api_processes']};
        - Servidor: FastAPI;
        """
        ),
        title="API",
        header_background="#007bff",
        header_color="white",
        sizing_mode="stretch_width",
    )
    card_asr = pn.Card(
        pn.pane.Markdown(
            f"""
        - Processos: {config_json['asr']['n_asr_workers']};
        - Hardware: {config_json['asr']['hardware_config']};
        - Modelo: {config_json['asr']['model']} ({config_json['asr']['language']});
        """
        ),
        title="ASR",
        header_background="#007bff",
        header_color="white",
        sizing_mode="stretch_width",
    )
    n_gliner_procs = (
        config_json["ner"]["n_gliner_workers"] * config_json["ner"]["n_cpus"]
    )
    card_ner = pn.Card(
        pn.pane.Markdown(
            f"""
        - Processos: {n_gliner_procs};
        - Hardware: CPU;
        - Modelo: {config_json['ner']['gliner_mname']};
        """
        ),
        title="NER",
        header_background="#007bff",
        header_color="white",
        sizing_mode="stretch_width",
    )
    card_inter = pn.Card(
        pn.pane.Markdown(
            f"""
        - Hardware: {config_json['interpretation']['hardware_config']};
        - Modelo: {config_json['interpretation']['model']};
        - Interpretando a cada {config_json['interpretation']['min_transcricao']} caracteres;
        """
        ),
        title="Interpretação",
        header_background="#007bff",
        header_color="white",
        sizing_mode="stretch_width",
    )

    return pn.Column(
        titulo, card_api, card_asr, card_ner, card_inter, sizing_mode="stretch_width"
    )


def create_final_dashboard(data, delays_plot_path, proj_dir):
    """Cria e monta o dashboard completo com Panel, incluindo o cabeçalho."""
    # Header corrigido: uso de HTML com style inline garante centralização e responsividade.
    header_html = """
    <div style="background:#007bff;color:white;padding:16px 20px;text-align:center;">
      <div style="max-width:1200px;margin:0 auto;">
        <h2 style="margin:0;font-weight:700;">Dashboard de Análise do Projeto Hermes</h2>
        <div style="font-size:14px;opacity:0.95;margin-top:6px;">Ministério da Justiça e Segurança Pública (MJSP)</div>
      </div>
    </div>
    """
    header = pn.pane.HTML(header_html, sizing_mode="stretch_width")

    # tab_qualidade = create_tab_qualidade(data)
    tab_desempenho = create_tab_desempenho(data, delays_plot_path)
    tab_custos = create_tab_custos(data, proj_dir)
    tab_config = create_tab_config(data)

    dashboard_tabs = pn.Tabs(
        # ("Qualidade Semântica", tab_qualidade),
        ("Custos", tab_custos),
        ("Desempenho e Delays", tab_desempenho),
        ("Configuração", tab_config),
        tabs_location="above",
        sizing_mode="stretch_width",
    )

    final_layout = pn.Column(header, dashboard_tabs, sizing_mode="stretch_width")

    return final_layout


def dashboard_stats(dfs):
    delays_medios = dfs["inferencia_stats"]["tipo", "delay medio"].rows(named=True)
    """comparison_stats = {'amostras': len(comparison_df)}
    for col in comparison_df.columns:
        if '_sim' in col or 'wer' in col or '_perc' in col:
            vals = comparison_df[col].filter(comparison_df[col].is_not_nan())
            vals = [v for v in vals if v == v and v != None]
            comparison_stats[col] = np.mean(vals)"""

    dashboard = {
        "delays_medios": delays_medios,
        #'comparison_stats': comparison_stats,
        "final_stats": dfs["final_stats"].rows(named=True),
        "costs_df": dfs["costs_df"].rows(named=True),
    }

    return dashboard


def make_dashboard(test_path):
    parsed_df_paths = glob(f"{test_path}/parsed_dfs/*.parquet")
    dfs = {}
    for df in parsed_df_paths:
        name = os.path.basename(df).replace(".parquet", "")
        dfs[name] = pl.read_parquet(df)

    delays_plot_path = plot_delays(
        test_path, dfs["inferencia_df"], dfs["transcricao_df"]
    )

    # comparison_df = pl.read_csv(f'{test_path}/semantic_performance-complete.tsv', separator='\t')
    configs_path = f"{test_path}/config.json"
    dashboard_json = dashboard_stats(dfs)
    dashboard_json["config"] = json.load(open(configs_path, "r"))
    # print(json.dumps(dashboard_json, ensure_ascii=False, indent=2))
    # %%
    dashboard = create_final_dashboard(dashboard_json, delays_plot_path)
    file_path = f"{test_path}/dashboard.html"
    dashboard.save(
        file_path,
        embed=True,  # Garante que todos os dados e JS sejam embutidos no arquivo
        title="Dashboard Hermes",
    )
