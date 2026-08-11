#!/bin/bash
IMAGE_NAME="hermes-asr-flask-qwen-fast"
CONTAINER_NAME="hermes-asr-flask-qwen-fast"

sudo docker build -t $IMAGE_NAME . > build.log 2>&1 \
    && sudo docker stop $CONTAINER_NAME || true \
    && sudo docker rm $CONTAINER_NAME || true \
    && sudo docker run \
        --gpus all \
        --privileged --runtime=nvidia --gpus all --shm-size 1G --rm \
        -p8010:8000 -p8011:8001 -p8012:8002 \
        -v /home/pita/hf_models:/root/.cache/huggingface  \
        --name $CONTAINER_NAME \
        $IMAGE_NAME
