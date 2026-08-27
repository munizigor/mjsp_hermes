#!/bin/bash
export TRITON_SERVER_URL=0.0.0.0:8000

# Definimos a lista de labels uma vez, como uma única string
# (Note que removemos as aspas extras do seu exemplo)
LABEL_STRING="rua_ou_logradouro, rua, bairro, municipio, cidade, ponto_de_referencia, nome_do_solicitante, pessoa, numero, street_number, number, complemento, endereço_complemento"

echo "--- Teste 1: Emergência na Rua das Flores ---"
transcript1="Preciso de uma ambulância rápido, tem uma pessoa caída na Rua das Flores, número 123, perto da padaria. Ela está inconsciente. Meu nome é Maria Oliveira. Meu telefone é 99999-8888, moro na cidade de São Paulo. "
echo $transcript1
time curl -X POST $TRITON_SERVER_URL/v2/models/gliner2/infer -d \
'{
  "inputs": [
    {
      "name": "PROMPT",
      "shape": [1, 1],
      "datatype": "BYTES",
      "data": ["Preciso de uma ambulância rápido, tem uma pessoa caída na Rua das Flores, número 123, perto da padaria. Ela está inconsciente. Meu nome é Maria Oliveira. Meu telefone é 99999-8888, moro na cidade de São Paulo."]
    },
    {
      "name": "LABEL_LIST",
      "shape": [1, 1],
      "datatype": "BYTES",
      "data": ["rua_ou_logradouro, rua, bairro, municipio, cidade, ponto_de_referencia, nome_do_solicitante, pessoa, numero, street_number, number, complemento, endereço_complemento"]
    },
    {
      "name": "CLASSIFICATION_SCHEMA",
      "shape": [1, 1],
      "datatype": "BYTES",
      "data": ["{\"natureza\": [\"leiMariaPenha::[casoDeMariaDaPenha|naoÉCasoDeMariaDaPenha]::str::Se a ocorrência se enquadra como um caso de violência doméstica\", \"feridosComRiscoDeMorte::[feridosComRiscoDeMorte|naoHaFeridosComRiscoDeMorte]::str::Se a ocorrência envolve feridos com risco de morte\"]}"]
    }
  ]
}'
