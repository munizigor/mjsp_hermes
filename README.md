# Hermes

📞 Sistema de Transcrição e Interpretação de Áudio para Atendimento de Emergência

O Hermes é um sistema de assitente para auxílio ao fluxo de atendimento de emergência, transcrevendo o áudio em tempo real e ajudando no preenchimento de dados do formulário de atendimento.

Este repositório funciona como um monorepo que reúne as diferentes camadas do projeto, cada uma responsável por uma parte do fluxo de captura, transcrição e atendimento da chamada.

## Execução Simplificada em Desktop (para testes)

### Requisitos
- Docker com NVIDIA Container Toolkit instalado;
- GPU com 6GB ou mais de VRAM;
- Pelo menos 16GB de RAM;
- Conexão com a internet (para baixar modelos);

Iniciar servidores de inferência:
```sh
# Start inference servers
./inference_servers/desktop_inference.sh
```

Iniciar backend:
```sh
cd backend && source backend/build_and_deploy.sh
```

## Estrutura do repositório

```
mjsp_hermes/
├── frontend/
├── voip_server/
├── backend/
└── artefatos/
```

### `frontend/`

Aplicação Django (também conhecida como Hermes CAD) que atua como um protótipo de front-end para o Hermes, permitindo que um atendente visualize as informações da chamada. Exibe a transcrição da chamada em tempo real via API/websocket, assim como as outras informações disponibilizadas pelo back-end, e geocodifica o endereço informado usando Leaflet.js + Nominatim (OpenStreetMap), como alternativa ao Google Maps. Inclui também os cenários de teste de carga com Locust. 

Veja as instruções completas de instalação e execução em [`frontend/README.md`](frontend/README.md).

### `voip_server/`

Agente que integra um servidor **Asterisk** ao ambiente Hermes, cuidando da sinalização de chamadas (via AMI) e da segmentação de áudio em tempo real a partir das gravações do `MixMonitor`, permitindo transcrição local ou em nuvem. 

Veja detalhes em [`voip_server/asterisk/README.md`](voip_server/asterisk/README.md).

### `backend/`

Core do sistema Hermes. Camada de backend com API REST, responsável por disponibilizaar as informações ao frontend, receber áudios, encaminhar áudios para transcrição e encontrar classificações ou informações chave. É o "cérebro" do sistema, responsável por orquestrar as requisições aos serviços de inferência que disponibilizam APIs de LLMs.

Mais detalhes e instruções de instalação em [`backend/README.md`](backend/README.md).

### `artefatos/`

Diagramas, documentação de arquitetura e demais artefatos de apoio ao projeto.

Veja mais detalhes em [`artefatos/README.md`](artefatos/README.md).

## Visão Geral da Arquitetura

A arquitetura do Hermes foi concebida para operar de forma desacoplada, permitindo sua integração com diferentes plataformas de Contact Center por meio da **API Hermes**. O frontend disponibilizado neste projeto possui caráter **exclusivamente demonstrativo**, sendo utilizado apenas para testes funcionais, acompanhamento da transcrição em tempo real e verificação do comportamento dos serviços durante o desenvolvimento.

Em um ambiente de produção, a interação dos operadores deve ocorrer diretamente por meio da plataforma de Contact Center adotada pelo órgão ou instituição, como Sinesp CAD (MJSP), Hefesto (SSP/DF) ou outra solução equivalente, que consumirá os serviços disponibilizados pela API Hermes para obtenção das transcrições, eventos e demais funcionalidades.

A comunicação entre os componentes da arquitetura utiliza diferentes protocolos, de acordo com a finalidade de cada fluxo:

* **REST**: utilizado pelo frontend de testes para consumir a API Hermes e exibir a transcrição e os eventos em tempo real. Em ambiente de produção, essa API deve ser consumida pela plataforma de Contact Center integrada;
* **AMI (Asterisk Manager Interface)**: utilizado pelo `voip_server` para receber do Asterisk os eventos de sinalização referentes ao início, término e demais mudanças de estado das chamadas telefônicas;
* **Webhook para envio das sinalizações SIP**: utilizado pelo `voip_server` para encaminhar ao backend do Hermes os eventos de sinalização das chamadas telefônicas processados a partir do AMI, permitindo o acompanhamento do ciclo de vida das chamadas e a sincronização do estado da comunicação;
* **Envio de chunks de aúdio do Asterisk ao backend do hermes**: realizado por meio de um diretório compartilhado e sincronizado, no qual o `voip_server` salva os chunks de áudio gerados durante a chamada como arquivos do tipo .wav. O back-end Hermes monitora o diretório e inclui novos áudios no banco de dados de áudios, a partir do qual são processados de forma assíncrona;

## Como começar

Cada camada possui seu próprio guia de instalação e execução:

1. [`frontend/README.md`](frontend/README.md) — instalação local ou via Docker, execução do mock de transcrição e testes de performance.
2. [`voip_server/asterisk/README.md`](voip_server/asterisk/README.md) — configuração do Asterisk (AMI e dialplan) e instalação dos serviços de sinalização/processamento de áudio.
3. [`backend/README.md`](backend/README.md) — instalação via Docker, com multiplos containers orquestrados através de docker-compose.

## Observações
> **IMPORTANTE:** A Prova de Conceito (PoC) do Projeto Hermes foi desenvolvida e validada utilizando o **Asterisk** como plataforma de telefonia, motivo pelo qual este documento apresenta procedimentos e exemplos específicos desse ambiente. Entretanto, a arquitetura da solução foi concebida para ser **independente da plataforma de PBX**, permitindo sua integração com outras soluções de telefonia, sejam elas **abertas ou proprietárias**, desde que seja possível atender aos requisitos de integração definidos pelo Hermes.
>
> Em particular, a plataforma de telefonia deverá ser capaz de fornecer ao backend do Hermes os **eventos de sinalização das chamadas** (início, término e demais mudanças de estado) e os **fragmentos de áudio (chunks)** produzidos durante a comunicação, observando os contratos e interfaces definidos pela API Hermes.
>
> Para integrações com plataformas diferentes do Asterisk, recomenda-se a leitura do [**Documento de Visão da Aplicação** ](https://www.gov.br/mj/pt-br), que descreve a arquitetura da solução, os requisitos funcionais e não funcionais, as premissas de integração entre o ambiente de telefonia e o backend do Hermes, bem como os mecanismos de comunicação esperados entre os componentes da solução.

