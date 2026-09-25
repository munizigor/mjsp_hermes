#!/bin/bash
IMAGE_NAME="hermes-asr-qwen-fast"
CONTAINER_NAME="hermes-asr-qwen-fast"

mkdir hf_models

sudo docker build -t $IMAGE_NAME . \
    && sudo docker run \
        --privileged --runtime=nvidia --gpus all --shm-size 1G --rm \
        -p8000:8000 -p8001:8001 -p8012:8002 \
        -v ~/.cache/huggingface:/root/.cache/huggingface  \
        -v /tmp/vllm_cache:/root/.cache/vllm \
        --name $CONTAINER_NAME \
        $IMAGE_NAME \
        qwen-asr-serve Qwen/Qwen3-ASR-0.6B --gpu-memory-utilization 0.75 --host 0.0.0.0 --port 8000 \
        --max-model-len 12000 --max-num-seqs 20

    #--network hermes-network \
    #--ip 172.20.0.15 \
    #&& sudo docker stop $CONTAINER_NAME || true \
    #&& sudo docker rm $CONTAINER_NAME || true \