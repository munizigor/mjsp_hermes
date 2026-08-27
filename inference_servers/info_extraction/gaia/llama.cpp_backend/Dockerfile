FROM nvcr.io/nvidia/tritonserver:25.10-pyt-python-py3

RUN apt update && apt install -y \
    python3 python3-pip \
    cmake build-essential make gcc sqlite3 libnuma-dev \
    build-essential libboost-system-dev libboost-thread-dev \
    libboost-program-options-dev libboost-test-dev libboost-filesystem-dev \
    libopenblas-dev libblas-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN CUDACXX=/usr/local/cuda-12/bin/nvcc CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES='75;89'" FORCE_CMAKE=1 \
    pip install --break-system-packages llama-cpp-python --upgrade --force-reinstall --no-cache-dir
RUN pip install --break-system-packages --no-cache-dir torch torchvision 
RUN pip install --break-system-packages --upgrade pydantic huggingface_hub datasets transformers accelerate bitsandbytes sentencepiece openai

# Cria o diretório de modelos dentro do container
RUN mkdir /models

# Copia seu repositório de modelo local para o container
COPY ./model_repository /models

# Expõe as portas padrão do Triton
EXPOSE 8000 
# HTTP
EXPOSE 8001 
# gRPC
EXPOSE 8002
# Métricas

# Comando padrão para iniciar o servidor Triton
CMD [ "tritonserver", "--model-repository=/models" ]