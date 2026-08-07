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

## Visão Geral da Arquitetura

A arquitetura do Hermes foi concebida para operar de forma desacoplada, permitindo sua integração com diferentes plataformas de Contact Center por meio da **API Hermes**. O frontend disponibilizado neste projeto possui caráter **exclusivamente demonstrativo**, sendo utilizado apenas para testes funcionais, acompanhamento da transcrição em tempo real e verificação do comportamento dos serviços durante o desenvolvimento.

Em um ambiente de produção, a interação dos operadores deve ocorrer diretamente por meio da plataforma de Contact Center adotada pelo órgão ou instituição, como Sinesp CAD (MJSP), Hefesto (SSP/DF) ou outra solução equivalente, que consumirá os serviços disponibilizados pela API Hermes para obtenção das transcrições, eventos e demais funcionalidades.

A comunicação entre os componentes da arquitetura utiliza diferentes protocolos, de acordo com a finalidade de cada fluxo:

* **REST**: utilizado pelo frontend de testes para consumir a API Hermes e exibir a transcrição e os eventos em tempo real. Em ambiente de produção, essa API deve ser consumida pela plataforma de Contact Center integrada.
* **AMI (Asterisk Manager Interface)**: utilizado pelo `voip_server` para receber do Asterisk os eventos de sinalização referentes ao início, término e demais mudanças de estado das chamadas telefônicas.
* **Webhook para envio das sinalizações SIP**: utilizado pelo `voip_server` para encaminhar ao backend do Hermes os eventos de sinalização das chamadas telefônicas processados a partir do AMI, permitindo o acompanhamento do ciclo de vida das chamadas e a sincronização do estado da comunicação.
* **Envio de chunks de aúdio do Asterisk ao backend do hermes**: realizado por meio de requisições HTTP, nas quais o `voip_server` transmite os chunks de áudio gerados durante a chamada para processamento assíncrono, transcrição, indexação e posterior disponibilização às aplicações consumidoras da API Hermes.


## Como começar

Cada camada possui seu próprio guia de instalação e execução:

1. [`frontend/README.md`](frontend/README.md) — instalação local ou via Docker, execução do mock de transcrição e testes de performance.
2. [`voip_server/asterisk/README.md`](voip_server/asterisk/README.md) — configuração do Asterisk (AMI e dialplan) e instalação dos serviços de sinalização/processamento de áudio.


> **IMPORTANTE:** A Prova de Conceito (PoC) do Projeto Hermes foi desenvolvida e validada utilizando o **Asterisk** como plataforma de telefonia, motivo pelo qual este documento apresenta procedimentos e exemplos específicos desse ambiente. Entretanto, a arquitetura da solução foi concebida para ser **independente da plataforma de PBX**, permitindo sua integração com outras soluções de telefonia, sejam elas **abertas ou proprietárias**, desde que seja possível atender aos requisitos de integração definidos pelo Hermes.
>
> Em particular, a plataforma de telefonia deverá ser capaz de fornecer ao backend do Hermes os **eventos de sinalização das chamadas** (início, término e demais mudanças de estado) e os **fragmentos de áudio (chunks)** produzidos durante a comunicação, observando os contratos e interfaces definidos pela API Hermes.
>
> Para integrações com plataformas diferentes do Asterisk, recomenda-se a leitura do [**Documento de Visão da Aplicação** ](https://www.gov.br/mj/pt-br), que descreve a arquitetura da solução, os requisitos funcionais e não funcionais, as premissas de integração entre o ambiente de telefonia e o backend do Hermes, bem como os mecanismos de comunicação esperados entre os componentes da solução.

