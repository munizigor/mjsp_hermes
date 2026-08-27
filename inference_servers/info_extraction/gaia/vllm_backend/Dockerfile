FROM nvcr.io/nvidia/tritonserver:25.10-vllm-python-py3

RUN pip install --no-cache-dir bitsandbytes>=0.46.1
RUN pip install --no-cache-dir --upgrade uv
RUN uv pip install --break-system-packages --system "vllm==0.10.2" --torch-backend=auto
# Expõe as portas padrão do Triton
EXPOSE 8000 
# HTTP
EXPOSE 8001 
# gRPC
EXPOSE 8002
# Métricas
#Valid backends are: ['FLASH_ATTN', 'FLASH_ATTN_VLLM_V1', 'TRITON_ATTN_VLLM_V1', 'XFORMERS', 'TORCH_SDPA', 'TORCH_SDPA_VLLM_V1', 'FLASHINFER', 'FLASHINFER_VLLM_V1', 'FLASHINFER_MLA', 'TRITON_MLA', 'TRITON_MLA_VLLM_V1', 'CUTLASS_MLA', 'FLASHMLA', 'FLASHMLA_VLLM_V1', 'FLASH_ATTN_MLA', 'PALLAS', 'PALLAS_VLLM_V1', 'IPEX', 'DUAL_CHUNK_FLASH_ATTN', 'DIFFERENTIAL_FLASH_ATTN', 'NO_ATTENTION', 'FLEX_ATTENTION', 'TREE_ATTN', 'XFORMERS_VLLM_V1']
#For old GPU compatibility: TRITON_ATTN_VLLM_V1
#ENV VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1

# Comando padrão para iniciar o servidor Triton
CMD [ "tritonserver", "--model-repository=/models" ]