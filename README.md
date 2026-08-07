# Hermes

📞 Sistema de Transcrição e Interpretação de Áudio para Atendimento de Emergência

O Hermes é uma prova de conceito de um sistema fullstack que simula o fluxo de uma chamada telefônica de emergência, transcrevendo o áudio em tempo real e preenchendo automaticamente o formulário de atendimento.

Este repositório funciona como um monorepo que reúne as diferentes camadas do projeto, cada uma responsável por uma parte do fluxo de captura, transcrição e atendimento da chamada.

## Estrutura do repositório

```
mjsp_hermes/
├── frontend/     # Aplicação Django (Hermes CAD) - interface de atendimento e transcrição em tempo real
├── voip_server/  # Agente de integração com o Asterisk (sinalização e segmentação de áudio)
├── backend/      # Camada de backend/API do ecossistema Hermes
└── artefatos/    # Diagramas, documentação e demais artefatos do projeto
```

### `frontend/`

Aplicação Django (também conhecida como Hermes CAD) responsável por interagir com o atendente: exibe a transcrição da chamada em tempo real via API/websocket e geocodifica o endereço informado usando Leaflet.js + Nominatim (OpenStreetMap), como alternativa ao Google Maps.

Inclui também os cenários de teste de carga com Locust. Veja as instruções completas de instalação e execução em [`frontend/README.md`](frontend/README.md).

### `voip_server/`

Agente que integra um servidor **Asterisk** ao ambiente Hermes, cuidando da sinalização de chamadas (via AMI) e da segmentação de áudio em tempo real a partir das gravações do `MixMonitor`, permitindo transcrição local ou em nuvem. Veja detalhes em [`voip_server/asterisk/README.md`](voip_server/asterisk/README.md).

### `backend/`

Camada de backend/API do ecossistema Hermes, responsável por orquestrar a comunicação entre o frontend, o serviço de transcrição e o agente de VoIP (ver `frontend/hermes.proto` para o contrato gRPC utilizado entre os serviços).

### `artefatos/`

Diagramas, documentação de arquitetura e demais artefatos de apoio ao projeto.

## Visão geral da arquitetura

A comunicação entre as camadas utiliza protocolos distintos conforme a necessidade de cada fluxo:

- **REST/WebSocket**: entre o frontend e o usuário, para exibir a transcrição em tempo real.
- **AMI (Asterisk Manager Interface)**: entre o `voip_server` e o Asterisk, para sinalização de início/fim de chamada.
- **gRPC** (`hermes.proto`): para o streaming de áudio e texto entre CAD, backend e serviço de transcrição.

## Como começar

Cada camada possui seu próprio guia de instalação e execução:

1. [`frontend/README.md`](frontend/README.md) — instalação local ou via Docker, execução do mock de transcrição e testes de performance.
2. [`voip_server/asterisk/README.md`](voip_server/asterisk/README.md) — configuração do Asterisk (AMI e dialplan) e instalação dos serviços de sinalização/processamento de áudio.
