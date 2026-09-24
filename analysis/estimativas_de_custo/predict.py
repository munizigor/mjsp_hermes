import sys
import os
from glob import glob
import json

import pandas as pd
import numpy as np

'''
Inputs:
- Environment token input price
- Environment token output price
- Environment ASR price per hour
- Monthly audio hours

Parameters:
- Input tokens per hour
- Output tokens per hour
- Standard deviation of input tokens per hour
- Standard deviation of output tokens per hour

From monthly audio hours and the constant of hours in a month, we can calculate average concurrent calls

# Avg Input Tokens: Input Tokens per hour * Monthly audio hours
# Standard Deviation Of input tokens = Standard deviation of input tokens per hour * Monthly audio hours

# Avg Output Tokens: Output Tokens per hour * Monthly audio hours
# Standard Deviation Of output tokens = Standard deviation of output tokens per hour * Monthly audio hours

ASR Cost = Monthly Audio Hours * ASR price per hour

Info. Extraction Cost = (Avg Input Tokens * Input token price) + (Avg Output Tokens * Output token price)

Cost Standard Variation = (Standard Deviation Of input tokens * Input token price) + (Standard Deviation Of output tokens * Output token price)

Total Cost = ASR Cost + Info. Extraction Cost

Expected Costs = Total Cost +- Cost Standard Variation

Model Json example:
{
    "in_tokens_per_hour": 883772.1839872964,
    "in_tokens_per_hour_std": 182163.10545722547,
    "out_tokens_per_hour": 240071.18467896548,
    "out_tokens_per_hour_std": 79755.9961990716,
    "db_size_mb_per_hour": 1.3512112620338215,
    "db_size_mb_per_hour_std": 0.35042260668473374
}
'''

'''
Não levados em consideração:
- Custos com hardware
- Como estimar custos de infraestrutura própria?
'''

cost_models_dir = os.path.join(os.path.dirname(__file__), 'costs_models')
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
train_data_dir = os.path.join(os.path.dirname(__file__), 'train_data')
dados_coletados = os.path.join(base_dir, "artefatos", "dados_coletados")
cenarios_excel_path = os.path.join(dados_coletados, 'cenarios.xlsx')
agencias_parsed_path = cenarios_excel_path.replace(".xlsx", "_agencias_parsed.csv")
pop_file = os.path.join(train_data_dir, 'populacao_brasil.txt')

meses_no_ano = 12
HORAS_POR_MES = 720
HORAS_POR_ANO = 8760
SEGUNDOS_EM_HORA = 3600

model_path = sys.argv[1]
results_path = sys.argv[2]
valor_dolar = 5.1 if len(sys.argv) <= 3 else float(sys.argv[3])
model = json.load(open(model_path, 'r'))

in_tokens_per_hour = model['in_tokens_per_hour']
in_tokens_per_hour_std = model['in_tokens_per_hour_std']
out_tokens_per_hour = model['out_tokens_per_hour']
out_tokens_per_hour_std = model['out_tokens_per_hour_std']
db_size_mb_per_hour = model['db_size_mb_per_hour']
db_size_mb_per_hour_std = model['db_size_mb_per_hour_std']

# Variancia do Input Tokens (por hora)
variancia_input = in_tokens_per_hour_std ** 2
# Variancia do Output Tokens (por hora)
variancia_output = out_tokens_per_hour_std ** 2

db_size_gb_per_hour = db_size_mb_per_hour / 1024
db_size_gb_per_hour_std = db_size_mb_per_hour_std / 1024

result_lines = []
#agencias = pd.read_excel(cenarios_excel_path, sheet_name='agencias')
agencias = pd.read_csv(agencias_parsed_path,sep=',')
deploys = pd.read_excel(cenarios_excel_path, sheet_name='deploy')

agencias["duracao_norm"] = agencias["Duração"].str.lower()
agencias["duracao_norm"] = agencias["duracao_norm"].apply(lambda x: x.split(" ")[0])
agencias["territorios_lista"] = agencias['Território'].apply(lambda x: x.split(", "))

agencias_brasil = []

mensal_names = set(['mensal', 'mes'])
anual_names = set(['anual', 'ano'])

brasil_data_row = agencias[agencias['Território'].str.lower() == 'brasil'].iloc[0]
taxa_urb_brasil = brasil_data_row['Taxa_Urbanizacao_IBGE']
POP_BRASIL = brasil_data_row['Pop_Total_IBGE']

for names_set, clf in [(mensal_names, 'mensal'), (anual_names, 'anual')]:
    agencias_clf = agencias[agencias['duracao_norm'].isin(names_set)]
    
    # Filtra os dados reais, excluindo o Brasil, para servir de base matemática
    agencias_com_horas = agencias_clf[(~agencias_clf['Horas de Áudio Totais'].isna()) & (agencias_clf['Território'].str.lower() != 'brasil')]

    # Caso a base do Excel tenha apenas dados "mensais", extrapolamos para o cálculo anual
    if agencias_com_horas.empty:
        agencias_com_horas = agencias[(agencias['duracao_norm'].isin(mensal_names)) & (~agencias['Horas de Áudio Totais'].isna()) & (agencias['Território'].str.lower() != 'brasil')].copy()
        agencias_com_horas['Horas de Áudio Totais'] *= meses_no_ano

    h_totais = agencias_com_horas['Horas de Áudio Totais'].values
    pop = agencias_com_horas['Pop_Total_IBGE'].values
    urb = agencias_com_horas['Taxa_Urbanizacao_IBGE'].values

    # =========================================================================
    # MÉTODO 1: Horas por habitante ponderada usando taxa de urbanização
    # =========================================================================
    h_per_capita = h_totais / pop
    h_avg_weighted = np.average(h_per_capita, weights=urb)
    horas_br_m1 = h_avg_weighted * POP_BRASIL

    # =========================================================================
    # MÉTODO 2: Regressão Simples Per Capita (Urbanização -> Horas Per Capita)
    # =========================================================================
    # Em vez de jogar a população absoluta no modelo, traçamos a reta apenas entre 
    # a urbanização e o consumo de horas por habitante.
    m, b = np.polyfit(urb, h_per_capita, 1)
    
    # Prevê o consumo per capita do Brasil baseado na sua taxa de urbanização (87,4%)
    h_per_capita_estimado_br = (m * taxa_urb_brasil) + b
    
    # Multiplica o per capita final pela população total do Brasil
    horas_br_m2 = h_per_capita_estimado_br * POP_BRASIL

    #Log all results:
    print(f"h_per_capita={h_per_capita}")
    print(f"urb={urb}")
    print(f"h_avg_weighted={h_avg_weighted}")
    print(f"h_per_capita_estimado_br={h_per_capita_estimado_br}")
    print(f"horas_br_m1={horas_br_m1}, horas_br_m2={horas_br_m2}")

    # Criação das novas linhas de cenário nacional
    agencias_brasil.append({
        'Território': 'Brasil (M1: Média Ponderada)',
        'Horas de Áudio Totais': horas_br_m1,
        'Duração': clf.capitalize(),
        'Pop_Total_IBGE': POP_BRASIL,
        'Taxa_Urbanizacao_IBGE': taxa_urb_brasil,
        'territorios_lista': ['Brasil'],
        'População da Região': POP_BRASIL,
        'duracao_norm': clf.lower(),
        'Pop_Urbana_IBGE': POP_BRASIL * taxa_urb_brasil,
    })

    '''agencias_brasil.append({
        'Território': 'Brasil (M2: Regressão Linear)',
        'Horas de Áudio Totais': horas_br_m2,
        'Duração': clf.capitalize(),
        'Pop_Total_IBGE': POP_BRASIL,
        'Taxa_Urbanizacao_IBGE': taxa_urb_brasil,
        'territorios_lista': ['Brasil'],
        'População da Região': POP_BRASIL,
        'duracao_norm': clf.lower(),
        'Pop_Urbana_IBGE': POP_BRASIL * taxa_urb_brasil,
    })'''

# Remove o "Brasil" original que não possuía horas mapeadas (evita falha na multiplicação cambial logo abaixo)
agencias = agencias[agencias['Território'].str.lower() != 'brasil'].copy()

# Adiciona as quatro novas predições processadas (Mensal M1, Mensal M2, Anual M1, Anual M2)
agencias_brasil_df = pd.DataFrame(agencias_brasil)
agencias = pd.concat([agencias, agencias_brasil_df], ignore_index=True)
#Salvar projeções para o brasil na pasta de resultados final
agencias.to_csv(results_path.replace('.csv', '_agencias_projections.csv'), index=False)

for _, agencia_row in agencias.iterrows():
    if agencia_row["duracao_norm"] in mensal_names:
        timespan_hours = HORAS_POR_MES
    elif agencia_row["duracao_norm"] in anual_names:
        timespan_hours = HORAS_POR_ANO
    else:
        timespan_hours = np.nan

    nome_agencia = agencia_row["Território"] + " (" + agencia_row["Duração"] + ")"
    horas_totais = agencia_row["Horas de Áudio Totais"]
    territorios = agencia_row["territorios_lista"] 
    print(nome_agencia, territorios)
    for _, deploy_row in deploys.iterrows():
        nome_deploy = deploy_row["Nome"]
        custo_output_1m = deploy_row["Custo por 1 milhão de Output Tokens"]
        custo_input_1m = deploy_row["Custo por 1 milhão de Input Tokens"]
        custo_asr_h = deploy_row["Custo por Hora de ASR"]

        custo_audio_avg = round(custo_asr_h * horas_totais, 2)
        db_gb_avg = round(horas_totais * db_size_gb_per_hour, 2)
        db_gb_std = round(horas_totais * db_size_gb_per_hour_std, 2)

        million_input_tokens_avg = (in_tokens_per_hour * horas_totais) / 1000000
        million_input_tokens_std = (in_tokens_per_hour_std * horas_totais) / 1000000

        custo_tokens_input_avg  = million_input_tokens_avg * custo_input_1m
        custo_tokens_input_std = million_input_tokens_std * custo_input_1m

        million_output_tokens_avg = (out_tokens_per_hour * horas_totais) / 1000000
        million_output_tokens_std = (out_tokens_per_hour_std * horas_totais) / 1000000

        custo_tokens_output_avg = million_output_tokens_avg * custo_output_1m
        custo_tokens_output_std = million_output_tokens_std * custo_output_1m
        
        variancia_input = custo_tokens_input_std ** 2
        variancia_output = custo_tokens_output_std ** 2
        custo_tokens_std = (variancia_input + variancia_output) ** 0.5

        custo_total_avg = round((custo_audio_avg + custo_tokens_input_avg + custo_tokens_output_avg) * valor_dolar, 2)
        custo_total_std = round(custo_tokens_std * valor_dolar, 2)

        custo_total_upper = round((custo_total_avg + custo_total_std), 2)
        custo_total_lower = round((custo_total_avg - custo_total_std), 2)

        custo_audio_avg_real = round(custo_audio_avg * valor_dolar, 2)
        custo_tokens_input_avg_reais = round(custo_tokens_input_avg * valor_dolar, 2)
        custo_tokens_input_std_real = round(custo_tokens_input_std * valor_dolar, 2)
        custo_tokens_output_avg_reais = round(custo_tokens_output_avg * valor_dolar, 2)
        custo_tokens_output_std_real = round(custo_tokens_output_std * valor_dolar, 2)

        million_input_tokens_per_sec = million_input_tokens_avg / timespan_hours / SEGUNDOS_EM_HORA
        million_output_tokens_per_sec = million_output_tokens_avg / timespan_hours / SEGUNDOS_EM_HORA
        million_input_tokens_per_sec_std = million_input_tokens_std / timespan_hours / SEGUNDOS_EM_HORA
        million_output_tokens_per_sec_std = million_output_tokens_std / timespan_hours / SEGUNDOS_EM_HORA

        million_input_tps_avg_r = round(million_input_tokens_per_sec, 4)
        million_input_tps_std_r = round(million_input_tokens_per_sec_std, 4)
        million_output_tps_avg_r = round(million_output_tokens_per_sec, 4)
        million_output_tps_std_r = round(million_output_tokens_per_sec_std, 4)

        if million_input_tps_std_r > 0.001:
            input_tokens_per_sec_value = f"{million_input_tps_avg_r} +/- {million_input_tps_std_r}"
        else:
            input_tokens_per_sec_value = str(million_input_tps_avg_r)
        if million_output_tps_std_r > 0.001:
            output_tokens_per_sec_value = f"{million_output_tps_avg_r} +/- {million_output_tps_std_r}"
        else:
            output_tokens_per_sec_value = str(million_output_tps_avg_r)

        custo_total_avg_mi = round(custo_total_avg / 1_000_000, 2)
        custo_total_std_mi = round(custo_total_std / 1_000_000, 2)
        

        result_lines.append({
            'Cenário de Agência': nome_agencia,
            'Opção de Deploy': nome_deploy,
            'Custo Total (milhões de R$)': f"{custo_total_avg_mi} +/- {custo_total_std_mi}",
            'Custo Total (R$)': f"{custo_total_avg} +/- {custo_total_std}",
            'Tamanho Banco de Dados Relacional (GB)': f"{db_gb_avg} +/- {db_gb_std}",
            'Custo Total Mínimo (R$)': custo_total_lower,
            'Custo Total Médio (R$)': custo_total_avg,
            'Custo Total Máximo (R$)': custo_total_upper,
            'Custo ASR (R$)': custo_audio_avg_real,
            'Custo Input Tokens (R$)': f"{custo_tokens_input_avg_reais} +/- {custo_tokens_input_std_real}",
            'Custo Output Tokens (R$)': f"{custo_tokens_output_avg_reais} +/- {custo_tokens_output_std_real}",
            'Milhões de Tokens por segundo (Input)': input_tokens_per_sec_value,
            'Milhões de Tokens por segundo (Output)': output_tokens_per_sec_value,
        })

result_df = pd.DataFrame(result_lines)
result_df.sort_values(['Cenário de Agência', 'Custo Total Médio (R$)'], inplace=True)
print(result_df)
result_df.to_csv(results_path, index=False)

        

        