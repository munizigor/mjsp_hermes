# Make sure your my_model_repo is in your current directory
export TRITON_SERVER_URL=0.0.0.0:8000
export LLM_NAME=gaia
time curl -X POST $TRITON_SERVER_URL/v2/models/$LLM_NAME/generate_stream -d \
'{
  "text_input": "Dê uma descrição curta sobre a cidade de São Paulo.",
  "parameters": {
    "max_tokens": 128
  }
}'

time curl -X POST $TRITON_SERVER_URL/v2/models/$LLM_NAME/generate -d \
'{
  "text_input": "Quais são os benefícios da energia solar?",
  "parameters": {
    "max_tokens": 256,
    "stream": false
  }
}'

time curl -X POST $TRITON_SERVER_URL/v2/models/$LLM_NAME/generate -d \
'{
  "text_input": "Descreva o processo de fotossíntese nas plantas.",
  "parameters": {
    "max_tokens": 512,
    "stream": false
  }
}'

time curl -X POST $TRITON_SERVER_URL/v2/models/$LLM_NAME/generate -d \
'{
  "text_input": "Explique a teoria da relatividade de forma simples e rápida",
  "parameters": {
    "max_tokens": 128,
    "stream": false
  }
}'

time curl -X POST $TRITON_SERVER_URL/v2/models/$LLM_NAME/generate -d \
'{
  "text_input": "Qual seu tipo favorito de sorvete e por quê?",
  "parameters": {
    "max_tokens": 128,
    "stream": false
  }
}'