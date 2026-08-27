#!/bin/bash
#Model name is the first argument
model_name=mistralai/Ministral-3-3B-Instruct-2512
hf_token_str=$1

sudo docker pull vllm/vllm-openai:v0.15.1 && \
    export HF_TOKEN=$hf_token_str && \
    sudo docker run --privileged --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e "HF_TOKEN=$HF_TOKEN" \
    -e "VLLM_LOGGING_LEVEL=DEBUG" \
    --network hermes-network \
    --ip 172.20.0.14 \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:v0.15.1 \
    --model $model_name \
    --max-model-len 8000
    