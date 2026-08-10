# HERMES ARTEFATOS

Este repositório tem como objetivo centralizar e organizar os artefatos documentais gerados ao longo do projeto Hermes.

A estrutura de diretórios foi pensada para facilitar a navegação e o acesso a documentos, diagramas, relatórios e demais registros produzidos.

Cada diretório contém um arquivo com links para os artefatos correspondentes, armazenados no OneDrive, garantindo que todo o time tenha acesso rápido, atualizado e padronizado às informações.

- [Repositórios](repositórios/repositorios_em_uso.md);
- Diagramas:
  - [Arquitetural](diagramas/diagrama_arquitetural);
  - [Sequencia](diagramas/diagrama_sequencia);
- Fluxogramas:
  - [Atendimento](fluxogramas/hermes_atendimento);
  - [Funcionalidades](fluxogramas/hermes_funcionalidades);
 
## Repositórios que integram o Projeto Hermes

### Dados Utilizados
- [pentalpha/hermes-chamadas-sinteticas](https://github.com/pentalpha/hermes-chamadas-sinteticas): Dataset de emergências geradas sintéticamente, usadas para testes e comparações estatísticas;

### Protótipos

#### Backend:
- [pentalpha/hermes-backend](https://github.com/pentalpha/hermes-agents): Servidor para receber informações de chamadas e retornar informações extraídas. Pode receber tanto transcrições quanto áudios de chamadas (.wav). Retorna informações como nomes, endereços, classificações de natureza da ocorrência e transcrição automática;

### Middleware:
- [michellsmr/hermes-asterisk-agent](https://github.com/michellsmr/hermes-asterisk-agent): Configuração de conexão do servidor VoIP Asterisk com o backend Hermes, através da sincronização de trechos de chamadas como arquivos .wav;

#### Frontend:
- [RafaelaOMarques/hermes_django](https://github.com/RafaelaOMarques/hermes_django): Projeto cujo objetivo é validar e provar o conceito da comunicação gRPC server-client-server e consumo de informações via Websocket;

#### Protótipos Fullstack:
- [munizigor/hermes_startupgov](https://github.com/munizigor/hermes_startupgov);
- [munizigor/protocolo-193](https://github.com/munizigor/protocolo-193);

### Servidores de Inferência
Configurações de servidores para criar endpoints de inferência. Estes repositórios não contém regras de negócio ou implementações de _features_ específicas ao projeto Hermes, são apenas adaptações de servidores existentes criadas para dar suporte aos modelos abertos escolhidos para o projeto:

- Transcrição (ASR): [pentalpha/hermes-inf_servers-asr](https://github.com/pentalpha/hermes-inf_servers-asr);
- Extração de Informações Chave (NER), Classificação e Geração de Texto: [pentalpha/hermes-inf_servers-info_extraction](https://github.com/pentalpha/hermes-inf_servers-info_extraction);

### Publicações

- "Evaluation of Transcription and Information Extraction Models for Emergency Calls in Brazil". Artigo científico, atualmente em processo de revisão e submissão. [pentalpha/hermes-paper](https://github.com/pentalpha/hermes-paper);
