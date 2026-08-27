#!/bin/bash
#Model name is the first argument
#model_name=mistralai/Ministral-3-3B-Instruct-2512
#model_name=mistralai/Ministral-3-3B-Instruct-2512-GGUF
model_name=cyankiwi/Ministral-3-3B-Instruct-2512-AWQ-4bit
hf_token_str=$1

sudo docker pull vllm/vllm-openai:v0.15.1 && \
    export HF_TOKEN=$hf_token_str && \
    sudo docker run --rm --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e "LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH" \
    -e "HF_TOKEN=$HF_TOKEN" \
    -e "VLLM_LOGGING_LEVEL=DEBUG" \
    -p 8000:8000 \
    --network hermes-network \
    --ip 172.20.0.14 \
    --name vllm-inf-extraction \
    vllm/vllm-openai:v0.15.1 \
    --model $model_name \
    --max-model-len 4900  \
    --gpu-memory-utilization 0.82 \
    --max-num-seqs 3 \
    --tokenizer_mode mistral \
    --config_format mistral --load_format mistral \
    --enable-auto-tool-choice --tool-call-parser mistral

#--kv-cache-dtype fp8 \
#-e "VLLM_HOST_IP=127.0.0.1" \
#--ipc=host \
#-e "VLLM_USE_V1=0" \
#-e "NCCL_P2P_DISABLE=1" \
#--shm-size=8g \
#--ipc=host \
#--tensor-parallel-size 1 --disable-custom-all-reduce \