#!/bin/bash
export TRITON_SERVER_URL=0.0.0.0:8000

# Definimos a lista de labels uma vez, como uma única string
# (Note que removemos as aspas extras do seu exemplo)
LABEL_STRING="rua_ou_logradouro, rua, bairro, municipio, cidade, ponto_de_referencia, nome_do_solicitante, pessoa, numero, street_number, number, complemento, endereço_complemento"

echo "--- Teste 1: Emergência na Rua das Flores ---"
transcript1="Preciso de uma ambulância rápido, tem uma pessoa caída na Rua das Flores, número 123, perto da padaria. Ela está inconsciente. Meu nome é Maria Oliveira. Meu telefone é 99999-8888, moro na cidade de São Paulo. "
echo $transcript1
time curl -X POST $TRITON_SERVER_URL/v2/models/gliner_x_large/infer -d \
'{
  "inputs": [
    {
      "name": "PROMPT",
      "shape": [1, 1],
      "datatype": "BYTES",
      "data": ["Ela está inconsciente. Meu nome é Maria Oliveira. Meu telefone é 99999-8888, moro na cidade de São Paulo. "]
    },
    {
      "name": "LABEL_LIST",
      "shape": [1, 1],
      "datatype": "BYTES",
      "data": ["rua_ou_logradouro, rua, bairro, municipio, cidade, ponto_de_referencia, nome_do_solicitante, pessoa, numero, street_number, number, complemento, endereço_complemento"]
    }
  ]
}'

echo -e "\n\n--- Teste 2: Incêndio na Avenida Principal ---"
transcript2="Meu nome é João Silva, estou ligando do bairro Centro, na Avenida Principal, 500. Tem um incêndio no apartamento 101. Sim, no condomínio Solar. Não, há pessoas presas lá dentro! Por favor, enviem ajuda urgente!"
echo $transcript2
time curl -X POST $TRITON_SERVER_URL/v2/models/gliner_x_large/infer -d \
'{
  "inputs": [
    {
      "name": "PROMPT",
      "shape": [1, 1],
      "datatype": "BYTES",
      "data": ["$transcript2"]
    },
    {
      "name": "LABEL_LIST",
      "shape": [1, 1],
      "datatype": "BYTES",
      "data": ["'"$LABEL_STRING"'"]
    }
  ]
}'

echo -e "\n\n--- Teste 3: Acidente em Petrópolis ---"
transcript3="Alô, é da polícia? Acabou de acontecer um acidente aqui na esquina da Sete de Setembro com a Rua Nova, em Petrópolis. Tem um carro capotado e várias pessoas machucadas. Meu nome é Carlos Mendes, estou ligando do bairro Alto da Serra. Por favor, mandem ajuda o mais rápido possível. "
echo $transcript3
time curl -X POST $TRITON_SERVER_URL/v2/models/gliner_x_large/infer -d \
'{
  "inputs": [
    {
      "name": "PROMPT",
      "shape": [1, 1],
      "datatype": "BYTES",
      "data": ["$transcript3"]
    },
    {
      "name": "LABEL_LIST",
      "shape": [1, 1],
      "datatype": "BYTES",
      "data": ["'"$LABEL_STRING"'"]
    }
  ]
}'