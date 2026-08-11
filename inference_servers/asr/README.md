# Hermes Inference Servers - ASR

Repository which runs ASR models inside an inference server (NVIDIA Triton, vLLM or Flask). The client reads .wav files, send them to a inference backend and returns transcriptions in the response

## Create Hermes Network

By default, the containers defined here are created inside this network. It makes sure that the backend can communicate with inference servers even if they are on the same host. The default IP is 172.20.0.15.

```sh
sudo docker network create --subnet=172.20.0.0/16 hermes-network || true
```

## Quickstart scripts

These will download and start a ASR container. Check the respective directories for further configuration.

- ./run_qwen3_1.7b.sh - Runs Qwen3-ASR-1.7B on vLLM backend.
- ./run_whisper_cpu.sh - Runs Whisper Large v3 turbo on whisper.cpp backend.
- ./run_whisper_turbo.sh - Runs Whisper Large v3 turbo on NVIDIA Triton backend.

## Examples

### Whisper CPU

```bash
$ source run_whisper_cpu.sh 
[sudo] password for [username]: 
[+] Running 3/3
 ✔ Container whisper_cpp-model-benchmarker-1  R...                                             0.0s 
 ✔ Container whisper_cpp-model-downloader-1   Re...                                            0.0s 
 ✔ Container whisper_cpp-whisper-1            Recreated                                        0.0s 
Attaching to model-benchmarker-1, model-downloader-1, whisper-1
model-downloader-1   | O arquivo ggml-medium-q8_0.bin já existe. Pulando download.
model-downloader-1 exited with code 0
model-benchmarker-1  | Arquivo de benchmark /models/ggml-medium-q8_0.bin.bench.txt já existe. Pulando execução.
model-benchmarker-1 exited with code 0
whisper-1            | Iniciando servidor: Modelo=medium-q8_0 | Threads=8 | Idioma=auto
whisper-1            | whisper_init_from_file_with_params_no_state: loading model from '/models/ggml-medium-q8_0.bin'
whisper-1            | whisper_init_with_params_no_state: use gpu    = 1
whisper-1            | whisper_init_with_params_no_state: flash attn = 1
whisper-1            | whisper_init_with_params_no_state: gpu_device = 0
whisper-1            | whisper_init_with_params_no_state: dtw        = 0
whisper-1            | whisper_init_with_params_no_state: devices    = 1
whisper-1            | whisper_init_with_params_no_state: backends   = 1
...
```

On a different terminal:

```bash
$ time curl http://127.0.0.1:8080/inference   -H "Content-Type: multipart/form-data"   -F file="@test_audios/senhora_ataque_cachorro.wav"   -F response_format="json"
{"text":" Pode me dizer o que está acontecendo?\n Ai, Deus me livre. Estou com muito medo.\n Um cachorro me atacou.\n Calma, senhora. Tente me dizer o que está acontecendo.\n Estou na rua Sargento Sampaio, número 101, no centro de H\navaí.\n É aqui perto, eu acho.\n Ok, senhora. Você pode me descrever o que aconteceu?\n Ele veio correndo, todo furioso.\n Mordeu na minha perna. Eu estou tremendo.\n Sinto muito, senhora. Você está machucada?\n Sim, está doendo muito. Não sei o que fazer.\n Deixe-me confirmar o endereço.\n Rua Sargento Sampaio, número 101, centro de Havaí.\n Tudo correto. Há alguém com você?\n Não, estou sozinha.\n Só estou aqui na rua.\n Segure um momento, senhora. Preciso acionar a ajuda.\n Você consegue me dizer o que aconteceu com mais detalhes?\n É um cachorro, um grande, preto.\n Ele me mordeu. Estou com muito medo. Não sei o que fazer.\n Ok, senhora polícia e serviços de emergência já estão acamp\nados.\n"}
real    0m30.322s
user    0m0.002s
sys     0m0.006s
```