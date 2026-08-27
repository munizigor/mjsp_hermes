# hermes-backend

Repositório responsável pela API REST para o sistema Hermes (Ministério da Justiça - Governo Federal do Brasil) para transcrição e extração de informações de áudios de ligações de emergência em tempo real.

## Execução Simplificada em Desktop (para testes)

### Requisitos
- Docker com NVIDIA Container Toolkit instalado;
- GPU com 6GB ou mais de VRAM;
- Pelo menos 16GB de RAM;
- Conexão com a internet (para baixar modelos);

Iniciar servidores de inferência:
```sh
# Start inference servers
source inference_servers/desktop_inference.sh
```

Iniciar backend:
```sh
cd backend && source backend/build_and_deploy.sh
```

## Visão geral

O projeto sobe um conjunto de containers e serviços que compõem a pipeline principal:

- `hermes-api`: API principal do sistema.
- `hermes-asr`: serviço de transcrição automática de fala.
- `hermes-ner`: serviço de reconhecimento de entidades nomeadas.
- `hermes-interpretation`: serviço de interpretação e classificação da ocorrência.
- `mariadb`: banco auxiliar provisionado pelo `docker-compose`.

Os serviços leem configurações a partir de [config.json](./config.json) e dos arquivos de ambiente montados no container. Os presets em [envs/](./envs/) permitem alternar entre execução local, GPU, Triton, Azure, GCP e outros cenários já preparados.

## Regras de negócio da solução

O comportamento da solução segue estas regras principais:

1. O fluxo de processamento é sequencial: ASR transcreve o áudio, NER extrai entidades e a camada de interpretação consolida o contexto da ocorrência.
2. A etapa de interpretação só deve ser acionada quando a transcrição atinge o volume mínimo definido por `interpretation.min_transcricao`.
3. A escolha do backend de cada serviço é dirigida por `hardware_config` no `config.json`.
4. O modelo de ASR pode operar em CPU, GPU, Triton ou serviços externos, conforme o preset selecionado.
5. O NER e a interpretação usam parâmetros de lote e limite de tokens para controlar custo, latência e uso de memória.
6. Os resultados de testes e execuções são gravados em `results/` conforme `test_output_dir`.
7. A quantidade de processos e workers é configurável para permitir ajuste fino de throughput.

## Requisitos

### Software:

- Docker
- NVIDIA Container Toolkit
    - Seguir instruções oficiais no site da NVIDIA: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

- Git.
- Python instalado na máquina local para execução dos scripts de setup e hooks.
- `pre-commit` instalado pelo script de setup.

### GPU

Se a execução usar CUDA/NVIDIA, instale também o NVIDIA Container Toolkit.

## Configuração inicial

### 1. Arquivo `.env`

Crie um arquivo `.env` na raiz do projeto com as variáveis necessárias para o `docker-compose`.

Exemplo mínimo:

```env
LOCAL_SQL_DB_PATH=./sqlite.db
HERMES_ADDR=0.0.0.0
HERMES_PORT="8001"
MARIADB_ROOT_PASSWORD=value
MARIADB_DATABASE=value
MARIADB_USER=value
MARIADB_PASSWORD=value
DB_PORT=3306

```

O arquivo [docker/common.env](./docker/common.env) já é carregado pelos containers e define variáveis internas da aplicação, como caminho do banco SQLite e cache de modelos.

### 2. Escolha do preset de execução

Os arquivos em [envs/](./envs/) representam cenários prontos de configuração.

Exemplos:

- `envs/config.local.cpu_only.json`: execução local em CPU.
- `envs/config.azure.*.json`: cenários com serviços Azure.
- `envs/config.gcp.*.json`: cenários com serviços GCP.
- `envs/config.gcp.triton_and_gemini.json`: combinação de backend externo para ASR e interpretação.

Copie o preset desejado para `config.json` na raiz do projeto antes de subir a stack.

Windows PowerShell:

```powershell
Copy-Item .\envs\config.local.cpu_only.json .\config.json
```

Linux / macOS:

```bash
cp ./envs/config.local.cpu_only.json ./config.json
```

Para utilizar endpoints azure, chaves como essa são necessárias:

#### Asterisk

Esta API de back-end é capaz de receber arquivos de áudio diretamente do front-end. Porém, uma opção mais eficiente é sincronizar um diretório de áudios do servidor VoIP.
```shell
USE_ASTERISK=TRUE
ASTERISK_RECORDINGS_PATH=/path/to/audio_segments/dir
```
#### ASR

Para utilizar ASR com serviço GCP Speech-to-text:
```shell
GOOGLE_PROJECT_ID=[PROJECT_ID]
GOOGLE_RECOGNIZER_REGION=[RECOGNIZER_REGION]
```

Para utilizar ASR com serviço Azure
```shell
#Para usar API cognitiveservices.azure (Como 'asr' azure-api):
AZURE_AI_RESOURCE_KEY=[CHAVE]
AZURE_AI_RESOURCE_ENDPOINT=https://[nome].cognitiveservices.azure.com/
#Inserir a região do recurso:
AZURE_SERVICE_REGION=brazilsouth
```

Para utilizar ASR com servidor de inferência NVIDIA Triton:
```shell
TRITON_SERVER_URL=[TRITON_SERVER_URL]:8000
#Definir timeout:
TRITON_SERVER_TIMEOUT_SECS=120
TRITON_SERVER_NER_TIMEOUT_SECS=5
```

Para acessar o ASR de um servidor de inferência vLLM
```shell
ASR_VLLM_HOST=[IP_DO_SERVIDOR]
ASR_VLLM_PORT=[PORTA]
```

#### Interpretation

Para realizar a interpretação com vLLM:
```shell
INTERPRETATION_VLLM_HOST=[IP_DO_SERVIDOR]
INTERPRETATION_VLLM_PORT=[PORTA]
```

Para realizar interpretação com um modelo do AI Foundry, configure as seguintes chaves:
```shell
#Para usar endpoints do Azure AI Foundry (Como 'interpretation' azure-api):
AZURE_AI_RESOURCE_ENDPOINT_[NOME_DO_MODELO]=https://[recurso_ai_foundry].cognitiveservices.azure.com/
AZURE_AI_RESOURCE_KEY_[NOME_DO_MODELO]=[CHAVE]
AZURE_AI_RESOURCE_VERSION_[NOME_DO_MODELO]=2025-03-01-preview
```

Para realizar interpretação com um modelo Gemini, configure a seguinte chave:
```shell
GEMINI_API_KEY=[CHAVE]
```

#### NER

Para utilizar Named Entity Recognition com servidor NVIDIA Triton:
```shell
TRITON_SERVER_URL_NER=[TRITON_SERVER_URL]:8000
#Definir timeout:
TRITON_SERVER_NER_TIMEOUT_SECS=5
```

Você também pode realizar o NER com servidores de inferência vLLM. Pode inclusive ser o mesmo servidor utilizado para Interpretation:
```shell
TRITON_SERVER_URL_NER=[IP_DO_SERVIDOR]:[PORTA_DO_VLLM]
```

## Configurações do Projeto (config.json)

O arquivo `config.json` na raiz do projeto é utilizado para configurar diversos aspectos dos serviços Hermes, incluindo modelos de ASR (Reconhecimento Automático de Fala), NER (Reconhecimento de Entidades Nomeadas) e o serviço de Interpretação. 

Exemplos de configurações podem ser encontrados na pasta [env/](./envs/).

Abaixo, detalhamos as seções e parâmetros configuráveis:

```json
{
    "test_output_dir": "results",
    "hermes-api": {
        "api_processes": 1
    },
    "asr": {
        "model": "turbo_cuda",
        "language": "pt",
        "hardware_config": "triton-server",
        "n_asr_workers": 1,
        "n_cpus": 1
    },
    "ner":{
        "n_gliner_workers": 3,
        "hardware_config": "triton-server",
        "n_cpus": 1,
        "gliner_mname": "gliner_x_large",
        "batch_size": 1,
        "max_tokens": 256
    },
    "interpretation": {
        "hardware_config": "gcp-api",
        "embedding_hardware": "cpu",
        "model": "gemini-2.0-flash-lite",
        "n_cpus": 2,
        "min_transcricao": 16,
        "n_in_batch": 10,
        "max_similar_natures": 20
    }
}
```

### Descrição dos Parâmetros:

-   **`test_output_dir`**: Diretório onde os resultados dos testes de integração serão salvos (ex: `results`).

-   **`hermes-api`**:
    -   **`api_processes`**: Número de processos da API do Hermes a serem executados. Útil para balanceamento de carga.

-   **`asr`** (Automatic Speech Recognition - Reconhecimento Automático de Fala):
    -   **`model`**: Nome do modelo de ASR a ser utilizado.
        - Locais: `small`, `medium`, `large`, `turbo`;
        - API: `azure-fast`;
        - Triton: `turbo_cuda`;
        - GCP: Nome de um Recognizer criado para o projeto na GCP;
    -   **`language`**: Idioma do áudio a ser transcrito (ex: `pt` para Português).
    -   **`hardware_config`**: Configuração de hardware para o modelo ASR:
        - `cuda` para GPU local;
        - `cpu` para CPU local;
        - `azure-api` para endpoints azure;
        - `triton-server` pra endpoint de servidor Nvidia Triton;
        - `gcp-api`: Utilizar reconhecedores de fala da GCP;
    -   **`n_asr_workers`**: Número de workers do Whisper (serviço de ASR) a serem executados.

-   **`ner`** (Named Entity Recognition - Reconhecimento de Entidades Nomeadas):
    -   **`n_gliner_workers`**: Número de workers do GLiNER (serviço de NER) a serem executados.
    -   **`hardware_config`**: Configuração de hardware para o modelo NER (`cuda` para GPU, `cpu` para CPU).
        - `cuda` para GPU local;
        - `cpu` para CPU local;
        - `triton-server` pra endpoint de servidor Nvidia Triton;
    -   **`n_cpus`**: Número de CPUs a serem alocadas para os workers de NER (relevante apenas se `hardware_config` for `cpu`).
    -   **`gliner_mname`**: Nome do modelo GLiNER pré-treinado a ser utilizado:
        - Para execução local (cuda/cpu): Qualquer modelo Gliner no HuggingFace (ex: `knowledgator/gliner-x-large`);
        - triton-server: Uma das opções:
            - `gliner_x_large` (equivalente a `knowledgator/gliner-x-large`);
    -   **`batch_size`**: Tamanho do lote de processamento para inferência do GLiNER.
    -   **`max_tokens`**: Número máximo de tokens que o modelo GLiNER processará por vez.

-   **`interpretation`** (Serviço de Interpretação):
    -   **`hardware_config`**: Configuração de hardware para o serviço de interpretação:
        - `cuda` para uso de GPU local;
        - `cuda-low` para GPU local com menor uso de memória;
        - `cpu` para CPU local;
        - `azure-api` para endpoints azure;
        - `gcp-api` para utilizar modelos Gemini;
    -   **`model`**: Nome do modelo usado no endpoint. Caso seja da Azure, deve haver uma chave correspondente ao nome deste modelo (`gpt-5-nano`). No caso de API GCP, deve ser o nome de um modelo Gemini existente;
    -   **`n_cpus`**: Número de CPUs a serem alocadas para os workers de interpretação (relevante se `hardware_config` for `cpu`);
    -   **`min_transcricao`**: Número mínimo de novos caracteres pra que o agente execute a pipeline de classificação da emergência;
    -   **`n_in_batch`**: Número de inferências a serem processadas em lote (batch);
    -   **`max_similar_natures`**: Número máximo de naturezas similares (através de embedding) a serem consideradas no contexto do prompt de classificação;


## Configuração de negócio por serviço

### ASR

- `model`: nome do modelo de transcrição.
- `language`: idioma do áudio, normalmente `pt`.
- `hardware_config`: define se a inferência ocorre em `cpu`, `cuda`, `azure-api`, `triton-server` ou `gcp-api`.
- `n_asr_workers`: quantidade de workers de transcrição.

### NER

- `hardware_config`: define execução local ou via servidor externo.
- `gliner_mname`: nome do modelo de NER.
- `batch_size`: tamanho do lote para inferência.
- `max_tokens`: limite de tokens por processamento.
- `n_gliner_workers`: quantidade de workers.

### Interpretation

- `hardware_config`: backend usado para interpretação.
- `model`: modelo do provedor escolhido.
- `min_transcricao`: quantidade mínima de caracteres para acionar a interpretação.
- `n_in_batch`: volume processado em lote.
- `max_similar_natures`: número máximo de naturezas similares usadas como contexto.
- `n_cpus`: alocação de CPU quando o backend é local.

## Como executar

### 1. Subir os containers

Execute a partir da raiz do projeto:

```bash
docker compose up --build
```

Se preferir manter os logs em arquivo, crie a pasta `logs/` e redirecione a saída da execução.

### 2. Instalar dependências locais e hooks do Git

Depois de preparar o ambiente virtual, rode o script correspondente ao seu sistema.

Windows:

```powershell
./setup_dev.ps1
```

Linux / macOS:

```bash
bash ./setup_dev.sh
```

Esses scripts instalam as dependências do ambiente local e registram os hooks `pre-commit` e `pre-push` no repositório.

### 3. Validar a execução

Confira se os containers subiram corretamente:

```bash
docker compose ps
```

Para acompanhar logs de um serviço específico:

```bash
docker compose logs -f hermes-api
```

## Scripts auxiliares

- `build_and_deploy.sh`: limpa o ambiente, faz os builds e executa o projeto.
- `deploy.sh`: executa o projeto.
- `down.sh`: para os containers do projeto.

## Estrutura relevante

- [mariadb/init/001_schema.sql](./mariadb/init/001_schema.sql): scripts de inicialização do banco.
- [datasets/](./datasets/): bases de apoio para testes e cenários de inferência.
- [docs/openapi.html](./docs/openapi.html): documentação estática da API.
- [docs/openapi.json](./docs/openapi.json): especificação OpenAPI.
- [testing_notebooks/](./testing_notebooks/): notebooks de validação.

## Contribuição

1. Abra uma *issue* descrevendo o bug ou proposta.
2. Ou então submeta um *Fork* ➜ *feature branch* ➜ *pull request*.

## Licença

GNU AFFERO GENERAL PUBLIC LICENSE
Version 3, 19 November 2007

## Gerar uma API key segura

Se precisar gerar uma chave aleatória para testes ou integrações locais:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```
