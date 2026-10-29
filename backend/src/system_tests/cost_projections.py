import sys
import os
import csv
import json
import pandas as pd


def get_dolar_rate():
    import requests

    url = "https://economia.awesomeapi.com.br/json/last/USD-BRL"
    response = requests.get(url).json()
    return float(response["USDBRL"]["bid"])


rate = get_dolar_rate()
print(f"Cotação Atual: R$ {rate:.2f}")


def main():
    # Validação dos argumentos da linha de comandos
    if len(sys.argv) < 4:
        print("Uso incorreto.")
        print(
            "Sintaxe: python projecoes_de_custos.py <input.csv> <output.csv> <dir_cenario1> [dir_cenario2 ...]"
        )
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    cenarios_dirs = sys.argv[3:]

    cenarios_dados = {}

    # 1. Extração dos dados de cada cenário
    for cenario_dir in cenarios_dirs:
        # Usa o nome da pasta como identificador da coluna (ex: gcp.open_mid-n_72...)
        nome_cenario = os.path.basename(os.path.normpath(cenario_dir))
        json_path = os.path.join(cenario_dir, "final_stats.json")

        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                try:
                    stats = json.load(f)

                    # Evitar divisão por zero caso o valor esteja ausente
                    carga_emergencias = stats.get("carga_de_emergencias")
                    if not carga_emergencias or carga_emergencias == 0:
                        carga_emergencias = 1.0

                    cenarios_dados[nome_cenario] = {
                        "custo_api_por_hora": float(stats.get("custo_api_por_hora", 0))
                        * rate,
                        "cost_vms_per_month": float(stats.get("cost_vms_per_month", 0))
                        * rate,
                        "carga_de_emergencias": float(carga_emergencias),
                    }
                except json.JSONDecodeError:
                    print(f"Erro ao descodificar JSON no ficheiro: {json_path}")
        else:
            print(
                f"Aviso: Ficheiro não encontrado em {json_path}. A ignorar este cenário."
            )

    if not cenarios_dados:
        print("Nenhum dado de cenário foi carregado. Operação cancelada.")
        sys.exit(1)

    # 2. Leitura do CSV e cálculo das novas colunas
    df = pd.read_csv(input_csv)
    fieldnames = df.columns.copy()

    # Adicionar os novos cabeçalhos de forma dinâmica para cada cenário testado
    for cenario in cenarios_dados.keys():
        print(cenario, cenarios_dados[cenario])

    nome_cenario_total = f"Total - {len(df)} agências"
    colunas_soma = [
        "Ligações por Mês",
        "Horas de Atendimento por Mês",
        "Carga de Ligações Média",
        "População",
    ]
    new_row = {"Unidade": nome_cenario_total}

    for col in colunas_soma:
        new_row[col] = df[col].sum()

    old_rows = [row for _, row in df.iterrows()] + [new_row]
    rows = []
    for row in old_rows:
        # Converter os valores do CSV para float, substituindo eventuais vírgulas se o formato for PT-BR
        horas_mes = float(row["Horas de Atendimento por Mês"])
        carga_agencia = float(row["Carga de Ligações Média"])

        # Efetuar os cálculos para cada cenário
        for cenario, dados in cenarios_dados.items():
            custo_mensal = (horas_mes * dados["custo_api_por_hora"]) + dados[
                "cost_vms_per_month"
            ]
            capacidade_utilizada = carga_agencia / dados["carga_de_emergencias"]

            # Guardar os resultados formatados no dicionário da linha
            row[f"Custo Mensal - {cenario}"] = f"{custo_mensal:.2f}"
            # Formatar como percentagem para facilitar a leitura da capacidade
            row[f"Capacidade Hermes Utilizada - {cenario}"] = (
                f"{capacidade_utilizada:.2%}"
            )

        print(row)
        rows.append({k: v for k, v in row.items()})
    rows.sort(key=lambda x: x["Horas de Atendimento por Mês"])
    df2 = pd.DataFrame(rows)

    # 3. Escrita do novo ficheiro CSV
    df2.to_csv(output_csv, sep=",", index=False)

    print(f"Sucesso! Tabela gerada e guardada em: {output_csv}")


if __name__ == "__main__":
    main()
