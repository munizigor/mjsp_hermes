FROM nvcr.io/nvidia/tritonserver:25.10-pyt-python-py3

RUN apt update && apt install -y \
    cmake build-essential make gcc sqlite3 \
    build-essential libboost-system-dev libboost-thread-dev \
    libboost-program-options-dev libboost-test-dev libboost-filesystem-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --break-system-packages pydantic langdetect python-mecab-ko janome indic-nlp-library jieba
RUN pip install --break-system-packages camel-tools spacy
RUN pip install --break-system-packages gliclass==0.1.16 gliner[gpu,tokenizers,stanza]==0.2.25
RUN pip cache purge

RUN pip freeze > /requirements2.txt

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
CMD ["tritonserver", "--model-repository=/models"]