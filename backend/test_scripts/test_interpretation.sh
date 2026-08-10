#!/bin/bash

# Script para testar o sistema de interpretação
# Este script inicia os serviços necessários e executa os testes

echo "=== Teste do Sistema de Interpretação ==="

#Limpa cache e banco de dados
mkdir -p sqlite.db
rm -rf datasets/naturezas_cache_vllm.json sqlite.db/*

# Verifica se o docker compose está rodando
if ! docker compose ps | grep -q "hermes-api"; then
    docker compose build
    echo "Iniciando serviços com docker compose..."
    docker compose up -d
    echo "Aguardando serviços inicializarem..."
    sleep 5
else
    echo "Serviços já estão rodando"
fi

# Verifica se a API do Hermes está acessível
echo "Verificando se a API do Hermes está acessível..."
if curl -s http://localhost:8001/ > /dev/null; then
    echo "✓ API do Hermes está rodando"
else
    echo "✗ API do Hermes não está acessível"
    echo "Verifique se o docker compose está rodando corretamente"
    exit 1
fi
# Valida dataset
if [ -z "$1" ]; then
  echo "Uso: $0 <caminho_dataset> [HERMES_URL] [MODELO_AUDIO]"
  exit 1
fi

# Valida modelo de áudio
if [ -z "$3" ]; then
  echo "Aviso: MODELO_AUDIO não especificado. Usando 'gemini-2.5_multi_tts' por padrão."
  AUDIO_MODEL="gemini-2.5_multi_tts"
else
  AUDIO_MODEL="$3"
fi

echo "Iniciando testing_bench para executar os testes..."
conda run --live-stream -n testing_bench python src/testing_bench/run_tests.py $1 $2 ${AUDIO_MODEL}
echo "Testes concluídos."

docker compose down