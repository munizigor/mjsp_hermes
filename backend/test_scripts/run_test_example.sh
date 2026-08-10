#!/bin/bash

# Exemplo de execução do script de testes
# Este script demonstra como usar o run_tests.py

echo "=== Exemplo de Execução dos Testes ==="

# Configurações
DATASET_PATH="/datasets/fake_calls"  # Ajuste para o caminho correto
HERMES_HOST="localhost"
HERMES_PORT="8001"
AUDIO_MODEL="large"  # ou "medium" ou "turbo"

echo "Configurações:"
echo "  Dataset: $DATASET_PATH"
echo "  Hermes: $HERMES_HOST:$HERMES_PORT"
echo "  Modelo: $AUDIO_MODEL"
echo ""

# Verifica se o script existe
if [ ! -f "src/testing_bench/run_tests.py" ]; then
    echo "Erro: Script run_tests.py não encontrado!"
    echo "Execute este script a partir do diretório raiz do projeto"
    exit 1
fi

# Verifica se o docker-compose está rodando
if ! docker-compose ps | grep -q "hermes-api"; then
    echo "Iniciando serviços..."
    docker-compose up -d
    echo "Aguardando serviços inicializarem..."
    sleep 15
fi

# Executa os testes
echo "Executando testes..."
cd src/testing_bench
python run_tests.py "$DATASET_PATH" "$HERMES_HOST:$HERMES_PORT" "$AUDIO_MODEL"

echo ""
echo "Testes concluídos!" 