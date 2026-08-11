# Use a imagem base oficial do Triton com Python 3
FROM nvcr.io/nvidia/tritonserver:24.08-pyt-python-py3
# Você pode usar tags mais recentes, como 24.05, 24.06, etc.

# Instala as bibliotecas Python necessárias
# - transformers: Para carregar o modelo
# - accelerate: Para 'device_map="auto"'
# - bitsandbytes: Para carregar em 4-bit ('load_in_4bit=True')
# - sentencepiece: Requerido por muitos tokenizadores (incluindo Gemma)
RUN pip install --no-cache-dir transformers accelerate bitsandbytes sentencepiece

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