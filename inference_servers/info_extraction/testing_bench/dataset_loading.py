from datasets import load_dataset
import pandas as pd
import polars as pl
import numpy as np
import os

# 1. Definição das colunas desejadas
colunas_flags = [
    "fatoOcorrendoNesteMomento",
    "autorDoFatoNoLocal",
    "autorDoFatoArmado",
    "feridosComRiscoDeMorte",
    "riscoDeTumulto",
    "leiMariaPenha",
]

colunas_metadados = [
    "ID",
    "rua",
    "numero",
    "complemento",
    "bairro",
    "cidade",
    "estado",
    "ponto_de_referencia",
    "natureza_inicial",
    "nome_solicitante",
    "roteiro_segmentado",
    "participacoes",
]

todas_colunas = colunas_flags + colunas_metadados

print("⏳ Carregando dataset (apenas metadados)...")

# 2. Carregar o dataset
# Usamos split="train" para pegar os dados diretamente.
# O HF datasets carrega tudo, mas vamos filtrar as colunas imediatamente para economizar memória.
ds = load_dataset(
    "pitagoras-alves/fake-emergencies-br",
    split="train",
    #streaming=True,
    columns=todas_colunas,
)

# Selecionar apenas as colunas necessárias
# ds_filtered = ds.select_columns(todas_colunas)

# Converter para Pandas para facilitar a amostragem estratificada
df_pandas = ds.to_pandas()

print(f"✅ Dataset carregado. Total inicial de linhas: {len(df_pandas)}")

# 3. Lógica de Filtragem (Garantir minímo de 10 True por flag)
indices_para_manter = set()

for col in colunas_flags:
    # Filtra linhas onde a coluna atual é True
    linhas_true = df_pandas[df_pandas[col] == True]

    # Se houver menos de 10, pega todas. Se houver mais, pega 10 aleatórias (ou as primeiras 10)
    # Aqui usamos sample para garantir variedade, ou head(10) para reprodutibilidade estática
    n_samples = min(len(linhas_true), 80)

    if n_samples > 0:
        amostra = linhas_true.sample(n_samples, random_state=42).index.tolist()
        indices_para_manter.update(amostra)
        print(
            f"  - Coluna '{col}': {len(linhas_true)} encontrados, {n_samples} selecionados."
        )
    else:
        print(f"  - ⚠️ Aviso: Coluna '{col}' não possui valores True.")

# 3.5. Pegar indices onde as colunas_flags são todas false ou indefinidas
indices_false = []
for i in range(len(df_pandas)):
    if all(df_pandas.iloc[i][col] == False or df_pandas.iloc[i][col] == None for col in colunas_flags):
        indices_false.append(i)
indices_false = np.random.choice(indices_false, 100, replace=False)
indices_para_manter.update(indices_false)
# 4. Criação do DataFrame Final
# Seleciona as linhas baseadas nos índices acumulados
df_final_pandas = df_pandas.loc[list(indices_para_manter)]

# 5. Asinalar false onde estiver indefinido
for col in colunas_flags:
    df_final_pandas[col] = df_final_pandas[col].fillna(False)

# Converte para Polars (já que você usou sintaxe Polars no prompt)
df_final_polars = pl.from_pandas(df_final_pandas)

print("-" * 30)
print(f"📊 DataFrame final pronto.")
print(f"Total de linhas filtradas: {df_final_polars.height}")
print(df_final_polars.head())

if not os.path.exists("input"):
    os.mkdir("input")

df_final_polars.write_parquet("input/dataset_filtrado.parquet")