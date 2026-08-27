cd whisper/triton && sudo docker build -t triton-whisper ./ \
    && sudo docker run --privileged --runtime=nvidia --gpus all --shm-size 1G --rm \
        -p8000:8000 -p8001:8001 -p8002:8002 -v /home/pita/hf_models:/root/.cache/huggingface \
        --network hermes-network \
        --ip 172.20.0.15 \
        triton-whisper tritonserver --log-verbose 1 --model-repository=/models \ 
        --load-model=turbo_cuda --model-control-mode=explicit