#!/bin/bash
#Model name is the first argument
#model_name=mistralai/Ministral-3-3B-Instruct-2512
#model_name=mistralai/Ministral-3-3B-Instruct-2512-GGUF
model_name=cyankiwi/gemma-4-E4B-it-AWQ-INT4
hf_token_str=$1

sudo docker pull vllm/vllm-openai && \
    export HF_TOKEN=$hf_token_str && \
    sudo docker run --rm --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e "LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH" \
    -e "HF_TOKEN=$HF_TOKEN" \
    -e "VLLM_LOGGING_LEVEL=DEBUG" \
    -p 8000:8000 \
    --name vllm-inf-extraction \
    vllm/vllm-openai \
    --model $model_name \
    --max-model-len 4900  \
    --gpu-memory-utilization 0.75 \
    --dtype auto \
    --max-num-seqs 3 \

#--network hermes-network \
#    --ip 172.20.0.14 \
#--kv-cache-dtype fp8 \
#-e "VLLM_HOST_IP=127.0.0.1" \
#--ipc=host \
#-e "VLLM_USE_V1=0" \
#-e "NCCL_P2P_DISABLE=1" \
#--shm-size=8g \
#--ipc=host \
#--tensor-parallel-size 1 --disable-custom-all-reduce \