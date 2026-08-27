export TRITON_SERVER_URL=0.0.0.0:8000

time curl -X POST $TRITON_SERVER_URL/v2/models/text_llm/infer -d \
'{
  "inputs": [
    {
      "name": "PROMPT",
      "shape": [1, 1],
      "datatype": "BYTES",
      "data": ["O que é a computação em nuvem?"]
    }
  ]
}'

time curl -X POST $TRITON_SERVER_URL/v2/models/text_llm/infer -d \
'{
  "inputs": [
    {
      "name": "PROMPT",
      "shape": [1, 1],
      "datatype": "BYTES",
      "data": ["Quem foi Albert Einstein?"],
      "max_tokens": 128
    }
  ]
}'

time curl -X POST $TRITON_SERVER_URL/v2/models/text_llm/infer -d \
'{
  "inputs": [
    {
      "name": "PROMPT",
      "shape": [1, 1],
      "datatype": "BYTES",
      "data": ["Explique a teoria da relatividade."]
    }
  ]
}'
