# 1. Usamos a imagem oficial da NVIDIA com o ambiente de desenvolvimento (devel)
# vLLM funciona muito bem com CUDA 12.4. 
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# 2. Configurações de ambiente para evitar prompts durante o apt e configurar o CUDA
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# 3. Instalar Python 3.12 e ferramentas de download
RUN apt update && apt install -y \
    software-properties-common \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt update && apt install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4. Ajustar alias para o python3.12 ser o padrão
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --set python3 /usr/bin/python3.12

# 5. Instalar o pip nativo para Python 3.12 de forma correta
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3

RUN pip install --no-cache-dir numpy requests librosa psutil flask gunicorn --ignore-installed blinker --break-system-packages

# 6. Instalar o qwen-asr com suporte a vLLM
# O vLLM será compilado/configurado aqui. Ter o nvcc no path resolve o erro do flashinfer.
RUN pip install --no-cache-dir -U qwen-asr[vllm] --ignore-installed blinker --break-system-packages

EXPOSE 8000

#Configurações para NVIDIA 3050 6GB (Máquina Local)
#Qwen/Qwen3-ASR-0.6B -> 3844MiB de VRAM
#CMD ["qwen-asr-serve", "Qwen/Qwen3-ASR-0.6B", "--gpu-memory-utilization", "0.8", "--host", "0.0.0.0", "--port", "8000", "--max-model-len", "4000", "--dtype", "half", "--max-num-seqs", "6"]
#Qwen/Qwen3-ASR-1.7B -> 4894MiB de VRAM
#CMD ["qwen-asr-serve", "Qwen/Qwen3-ASR-1.7B", "--gpu-memory-utilization", "0.867", "--host", "0.0.0.0", "--port", "8000", "--max-model-len", "2400", "--dtype", "float16", "--max-num-seqs", "3", "--kv-cache-dtype", "fp8"]

#Configuração para NVIDIA T4 16GB (GCP)
CMD ["qwen-asr-serve", "Qwen/Qwen3-ASR-1.7B", "--gpu-memory-utilization", "0.97", "--host", "0.0.0.0", "--port", "8000", "--max-model-len", "8000", "--dtype", "half", "--max-num-seqs", "16"]
