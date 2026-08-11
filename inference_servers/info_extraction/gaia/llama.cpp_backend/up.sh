sudo docker build -t triton-gaia:llamacpp .
mkdir -p /tmp/huggingface
sudo docker run --gpus all --rm -it \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -e HF_HOME=/tmp/huggingface \
  -e CUDA_LAUNCH_BLOCKING=1 \
  -v /tmp:/tmp \
  triton-gaia:llamacpp