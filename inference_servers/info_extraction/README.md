# Hermes Inference Servers: Information Extraction

Containers configured for information extraction based on NVIDIA Triton, with Gliner and Gliclass models.

## Create Hermes Network

By default, the containers defined here are created inside this network. It makes sure that the backend can communicate with inference servers even if they are on the same host.

```sh
sudo docker network create --subnet=172.20.0.0/16 hermes-network || true
```

## Run Ministral

This will run the Ministral 3B model as a service using vLLM on the GPU. It contains a OpenAI compatible API.
vLLM documentation: [https://docs.vllm.ai/en/latest/getting_started/quickstart.html](https://docs.vllm.ai/en/latest/getting_started/quickstart.html).

```sh
chmod +x vllm_server/*.sh
source vllm_server/start_ministral3b.sh
#Or _8b.sh
```

There is also a version compatible with 6GB GPUs:

```sh
source vllm_server/start_ministral3b_light.sh
```

## Build and Run Gliner+Gliclass

This will build the docker image (based on NVIDIA Triton) and run it with it's custom scripts for running encoder-only information extraction models of the Gliner and Gliclass families. This can be used by Hermes Backend for NER, but it is not capable of doing text generation tasks.

```sh
cd gliner_gliclass
chmod +x up.sh
./up.sh
```