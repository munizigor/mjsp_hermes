# Hermes

📞 Hermes - Sistema de Transcrição e Interpretação de Áudio
<div align="center">
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/Django-092E20?style=for-the-badge&logo=django&logoColor=white" alt="Django">
<img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black" alt="JavaScript">
</div>

## Sobre o Projeto
O Hermes é uma prova de conceito de um sistema fullstack para processamento de áudio. Ele simula o fluxo de uma chamada telefônica, com preencimento automático do formulário de atendimento em tempo real. A arquitetura é construída utilizando protocolos eficientes para cada tipo de comunicação:

**Frontend** : Interage com o usuário e exibe a transcrição em tempo real via Rest API.

**Geocoding** : Implementada Leaflet.js e Nominatim com dados do OpenStreetMap (OSM), como alternativa ao Google Maps para funcionalidade de Geocoding.

## O que existe no projeto

- `hermes_cad/`: aplicação principal com views, consumers, models e templates.
- `hermes_ecossistema/`: configuração do projeto Django.
- `tests/performance/`: cenários de carga com Locust.

## Requisitos

- Python 3
- Docker e Docker Compose
- Git

## Instalação local

Crie e ative um ambiente virtual:

```bash
python -m venv venv
```

Windows:

```bash
.\venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

Depois de ativar o ambiente virtual, use o script de setup correspondente ao seu sistema:

Windows:

```bash
bash setup_dev.ps1
```

Linux / macOS:

```bash
bash setup_dev.sh
```

Esses scripts instalam as dependências e registram os hooks do Git (`pre-commit` e `pre-push`).

Se preferir fazer manualmente, use:

```bash
pip install -r requirements.txt
pre-commit install --hook-type pre-commit
pre-commit install --hook-type pre-push
```

## Como executar o projeto

### Opção 1: Docker

1. Crie um arquivo `.env` na raiz do projeto com as variáveis necessárias.

```env
HERMES_BACKEND_URL=http://{URL_BACKEND:PORTA}
ALLOWED_HOSTS=0.0.0.0
HERMES_API_KEY={Obter esse valor com o time}
```

2. Suba os containers:

```bash
docker compose up --build
```

3. Abra a aplicação em:
http://127.0.0.1:8002/

4. Obs.: Para parar a aplicação:
```bash
docker compose down
```

### Opção 2: Execução local

Em terminais separados, execute:

Terminal 1 - mock de transcrição:

```bash
python mock_transcription_server.py
```

Terminal 2 - Execute a aplicação Django:

```bash
uvicorn hermes_ecossistema.asgi:application --reload --host 127.0.0.1 --port 8000
```

## Testes de performance com Locust

Os cenários estão disponibilizados no path `tests/performance/performance_tests.py`.

### Fluxos disponíveis

- `EmergenciaUser`: fluxo completo, com criação de emergência, envio de transcrição, consulta de inferência e encerramento para os flows 2, 3 e 4.
- `EmergenciaUserFlow1`: fluxo principal de consulta, usando IDs pré-existente para o flow (atendimento 100% com agente de IA a ser validado no host https://hermes-labs.samare.com.br).

### Como executar a interface do Locust

Abra o Locust e informe o host desejado na própria interface:

```bash
locust -f tests/performance/performance_tests.py
```

Na tela do Locust, preencha o campo **Host** com o ambiente que você quer testar:

- Host para execução local: http://127.0.0.1:8000
- Host para execução em cloud: https://hermes-labs.samare.com.br


Depois, selecione a classe desejada:

- `EmergenciaUser`
- `EmergenciaUserFlow1`

Você pode marcar uma ou as duas classes para executar os dois fluxos ao mesmo tempo.

### Execução em modo headless

Exemplo local:

```bash
locust -f tests/performance/performance_tests.py --host=http://127.0.0.1:8000 --headless -u 200 -r 20 -t 10m --csv=locust_results
```

### Filtrar tarefas por tag

As tasks possuem tags para facilitar testes pontuais:

- `transcricao`
- `inferencia`
- `encerrar`

Exemplo:

```bash
locust -f tests/performance/performance_tests.py --host=http://127.0.0.1:8000 --tags inferencia
```

## Geocoding

O projeto usa OpenStreetMap / Nominatim como alternativa ao Google Maps.

Use a API com cuidado, porque o Nominatim tem limite de requisição por segundo.

Exemplo de busca:

```text
https://nominatim.openstreetmap.org/search?q=Rua+das+Flores,+Aracaju&format=jsonv2&limit=1
```

## Observações

- Se o backend mudar de porta ou ambiente, ajuste apenas o host informado na interface do Locust.
- Para produção ou pré-produção, prefira testar em um ambiente espelho de homologação antes de executar carga mais alta.

## Repositorio

[![GitHub](https://img.shields.io/badge/GitHub-RafaelaOMarques%2Fhermes__django-181717?style=for-the-badge&logo=github)](https://github.com/RafaelaOMarques/hermes_django)