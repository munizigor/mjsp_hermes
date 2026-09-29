#!/bin/bash
IMAGE_NAME="hermes-asr-qwen-fast"
CONTAINER_NAME="hermes-asr-qwen-fast"

mkdir hf_models

sudo docker build -t $IMAGE_NAME . \
    && sudo docker stop $CONTAINER_NAME || true \
    && sudo docker rm $CONTAINER_NAME || true \
    && sudo docker run \
        --privileged --runtime=nvidia --gpus all --shm-size 1G --rm \
        -p8000:8000 -p8011:8001 -p8012:8002 \
        -v ~/.cache/huggingface:/root/.cache/huggingface  \
        -v /tmp/vllm_cache:/root/.cache/vllm \
        --network hermes-network \
        --ip 172.20.0.15 \
        --name $CONTAINER_NAME \
        -e VLLM_USE_V1=1 \
        $IMAGE_NAME \
        qwen-asr-serve Qwen/Qwen3-ASR-0.6B \
        --host 0.0.0.0 --port 8000 \
        --gpu-memory-utilization 0.82 \
        --max-model-len 2048 \
        --max-num-seqs 32 \
        --max-num-batched-tokens 8192 \
        --enable-chunked-prefill \
        --enable-prefix-caching \
        --override-generation-config '{"temperature": 0.0}' \
        --compilation-config '{"compile_mm_encoder": true}' \
        --disable-log-requests

    #
    