#!/bin/bash
export TRITON_SERVER_URL=0.0.0.0:8000

# Definimos a lista de labels uma vez, como uma única string
# (Note que removemos as aspas extras do seu exemplo)
LABEL_STRING="rua_ou_logradouro, rua, bairro, municipio, cidade, ponto_de_referencia, nome_do_solicitante, pessoa, numero, street_number, number, complemento, endereço_complemento"
SCHEMA_JSON='{"entities": {"rua_ou_logradouro": "Nome de rua, logradouro ou avenida", "rua": "Nome de rua", "bairro": "Nome de bairro", "municipio": "Nome de município", "cidade": "Nome de cidade", "ponto_de_referencia": "Nome de ponto de referência do endereço", "nome_do_solicitante": "Nome do solicitante (pessoa que está fazendo o chamado)", "pessoa": "Nome de pessoa ou participante", "numero": "Número do endereço", "street_number": "Street Number", "number": "Número", "complemento": "Complemento do endereço (Exemplos: apt 301, casa A, etc)", "endereço_complemento": "Número ou código da casa, apartamento ou loja naquele endereço"}, "boolean": {"natureza_da_ocorrencia": {"Fato Ocorrendo Neste Momento ?": {"desc": "Se o fato relatado está ocorrendo neste momento", "sim": "Sim", "nao": "Não"}, "Autor Do Fato No Local ?": {"desc": "Se o autor (culpado/acusado) do fato está no local", "sim": "Sim", "nao": "Não"}, "Autor Do Fato Armado ?": {"desc": "Se o autor (culpado/acusado/suspeito) do fato estava armado", "sim": "Sim", "nao": "Não"}, "Feridos Com Risco de Morte ?": {"desc": "Se a ocorrência envolve feridos com risco de morte", "sim": "Sim", "nao": "Não"}, "Risco De Tumulto ?": {"desc": "Se a ocorrência envolve um risco de tumulto", "sim": "Sim", "nao": "Não"}, "Lei Maria da Penha ?": {"desc": "Se a ocorrência se enquadra como um caso de lei maria da penha, a qual trata sobre a violência doméstica e conjugal", "sim": "Sim", "nao": "Não"}}}}'
echo "--- Teste 1: Emergência na Rua das Flores ---"
transcript1="Preciso de uma ambulância rápido, tem uma pessoa caída na Rua das Flores, número 123, perto da padaria. Ela está inconsciente. Meu nome é Maria Oliveira. Meu telefone é 99999-8888, moro na cidade de São Paulo. "
echo $transcript1

# 2. Usamos o jq para montar o payload do Triton
# O segredo está em passar o SCHEMA_JSON como uma string dentro do data
PAYLOAD=$(jq -n \
  --arg transcript "$TRANSCRIPT" \
  --arg schema "$SCHEMA_JSON" \
  '{
    "inputs": [
      {
        "name": "PROMPT",
        "shape": [1, 1],
        "datatype": "BYTES",
        "data": [$transcript]
      },
      {
        "name": "LABEL_LIST",
        "shape": [1, 1],
        "datatype": "BYTES",
        "data": [$schema]
      }
    ]
  }')

time curl -X POST $TRITON_SERVER_URL/v2/models/gliner_x_large/infer -H "Content-Type: application/json" \
  -d "$PAYLOAD"