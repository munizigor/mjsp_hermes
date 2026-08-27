FROM nvcr.io/nvidia/tritonserver:24.08-py3

RUN pip install transformers torch numpy requests librosa tritonclient[http] \
    && pip cache purge
RUN pip install datasets[audio] accelerate \
    && pip cache purge
RUN apt update && apt install -y ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN pip install torchcodec \
    && pip cache purge
RUN pip install torchvision
RUN python3 -c "import torch; import torchvision; print(torchvision.ops.nms(torch.zeros(1,4), torch.ones(1), 0.5))"

RUN pip install -U qwen-asr  --ignore-installed blinker --break-system-packages
RUN python3 -c "import torch; import torchvision; print(torchvision.ops.nms(torch.zeros(1,4), torch.ones(1), 0.5))"

RUN pip install flask psutil gunicorn --break-system-packages

WORKDIR /app
# Copia o código para dentro do diretório /app
COPY server.py /app/server.py

EXPOSE 8000

# --- GUNICORN WORKERS ---
# Ajuste a seguir baseando-se na VRAM da sua GPU:
# - Se tiver GPU de 8GB: use --workers 1
# - Se tiver GPU de 16GB+: pode tentar --workers 2 (ou mais, se couber)
# Cada worker roda como um processo separado e carrega o modelo.
# O --threads é mantido em 16 para paralelismo dentro do processo.
CMD ["gunicorn", "--workers", "2", "--threads", "16", "--timeout", "300", "--bind", "0.0.0.0:8000", "server:app"]