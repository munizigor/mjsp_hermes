#!/bin/bash
#Model name is the first argument
model_name=$1
hf_token_str=$2

sudo docker pull vllm/vllm-openai:v0.15.1 && \
    export HF_TOKEN=$hf_token_str && \
    sudo docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e "HF_TOKEN=$HF_TOKEN" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:v0.15.1 \
    --model $model_name