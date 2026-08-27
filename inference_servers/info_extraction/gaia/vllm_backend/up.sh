# Set up paths for persistent, non-temporary storage
export MODEL_REPO_PATH=$(pwd)/model_repository
export HF_CACHE_PATH=$(pwd)/hf_cache
export VLLM_CACHE_PATH=$(pwd)/vllm_cache

# Create the directories on your host machine
mkdir -p $HF_CACHE_PATH
mkdir -p $VLLM_CACHE_PATH

sudo docker build -t triton-info . && sudo docker run --gpus all --rm \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -e CUDA_LAUNCH_BLOCKING=1 \
  -e HF_HOME=/hf-cache \
  -v $MODEL_REPO_PATH:/models \
  -v $HF_CACHE_PATH:/hf-cache \
  -v $VLLM_CACHE_PATH:/root/.cache \
  triton-info \
  tritonserver --model-repository=/models

#sudo docker run --gpus all --rm -it -p 8000:8000 -p 8001:8001 -p 8002:8002 -e CUDA_LAUNCH_BLOCKING=1 -e HF_HOME=/hf-cache -v $MODEL_REPO_PATH:/models -v $HF_CACHE_PATH:/hf-cache -v $VLLM_CACHE_PATH:/root/.cache triton-info /bin/bash