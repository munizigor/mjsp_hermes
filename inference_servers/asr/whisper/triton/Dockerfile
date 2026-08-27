FROM nvcr.io/nvidia/tritonserver:24.08-py3

RUN pip install transformers torch numpy requests librosa tritonclient[http] \
    && pip cache purge
RUN pip install datasets[audio] accelerate \
    && pip cache purge
RUN apt update && apt install -y ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN pip install torchcodec \
    && pip cache purge
COPY ./model_repository /models
EXPOSE 8000
EXPOSE 8001
EXPOSE 8002

CMD ["tritonserver", "--log-verbose", "1", "--model-repository=/models", "--load-model=turbo_cuda", "--model-control-mode=explicit"]