# Estimativas de custo 

## Pacotes Necessários

- python >= 3.11
- requests >= 2.32
- pandas >= 3
- numpy >= 2.4.6

## Preparar Dados

Obter estimativas populacionais do IBGE:

```bash
python get_data.py
```

## Criar modelo de precificação

É necessário uma pasta com resultados de testes de carga realizados pelo módulo [backend/src/testing_bench](backend/src/testing_bench/README.md). Os modelos serão salvos em analysis/estimativas_de_custo/costs_models/.

```bash
python create_models.py <caminho_da_pasta_de_testes>
```

## Calcular custo de cenários

Os modelos serão usados para calcular o custo de cenários de uso do sistema.

```bash
python predict.py analysis/estimativas_de_custo/costs_models/model_simple.json artefatos/estimativas_de_custos/resultado.csv 5.1
```

Os argumentos são:
1. caminho do modelo json
2. caminho do arquivo de saída
3. valor do dólar (opcional, padrão 5.1)