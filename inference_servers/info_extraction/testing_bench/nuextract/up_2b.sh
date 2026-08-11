sudo docker run -it --rm --name nuextract --network=host \
    --runtime nvidia --gpus all \
    nuextract-2-2b --api-key putarealsecret --dtype half --max-model-len 8192 --enforce-eager