#!/bin/bash
IMAGE_NAME="hermes-asr-flask-qwen-fast"
CONTAINER_NAME="hermes-asr-flask-qwen-fast"

sudo docker build -t $IMAGE_NAME . \
    && sudo docker stop $CONTAINER_NAME || true \
    && sudo docker rm $CONTAINER_NAME || true \
    && sudo docker run \
        --privileged --runtime=nvidia --gpus all --shm-size 1G --rm \
        -p8000:8000 -p8001:8001 -p8012:8002 \
        -v /home/pita/hf_models:/root/.cache/huggingface  \
        -v /tmp/vllm_cache:/root/.cache/vllm \
        --name $CONTAINER_NAME \
        $IMAGE_NAME
