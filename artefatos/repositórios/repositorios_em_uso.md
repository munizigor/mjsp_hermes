# Repositórios que integram o Projeto Hermes

## Dados Utilizados
- [pentalpha/hermes-chamadas-sinteticas](https://github.com/pentalpha/hermes-chamadas-sinteticas): Dataset de emergências geradas sintéticamente, usadas para testes e comparações estatísticas;

## Protótipos

### Backend:
- [pentalpha/hermes-agents](https://github.com/pentalpha/hermes-agents): Servidor para receber informações de chamadas e retornar informações extraídas. Pode receber tanto transcrições quanto áudios de chamadas (.wav). Retorna informações como nomes, endereços, classificações de natureza da ocorrência e transcrição automática;

### Middleware:
- [michellsmr/hermes-asterisk-agent](https://github.com/michellsmr/hermes-asterisk-agent): Configuração de conexão do servidor VoIP Asterisk com o backend Hermes, através da sincronização de trechos de chamadas como arquivos .wav;

### Frontend:
- [RafaelaOMarques/hermes_django](https://github.com/RafaelaOMarques/hermes_django): Projeto cujo objetivo é validar e provar o conceito da comunicação gRPC server-client-server e consumo de informações via Websocket;

### Protótipos Fullstack:
- [munizigor/hermes_startupgov](https://github.com/munizigor/hermes_startupgov);
- [munizigor/protocolo-193](https://github.com/munizigor/protocolo-193);

## Servidores de Inferência
Configurações de servidores para criar endpoints de inferência. Estes repositórios não contém regras de negócio ou implementações de _features_ específicas ao projeto Hermes, são apenas adaptações de servidores existentes criadas para dar suporte aos modelos abertos escolhidos para o projeto:

- Transcrição (ASR): [pentalpha/triton-whisperer](https://github.com/pentalpha/triton-whisperer);
- Extração de Informações Chave (NER): [pentalpha/triton-info-extraction](https://github.com/pentalpha/triton-info-extraction);
- Geração de Texto: [pentalpha/saxml-gemma-server](https://github.com/pentalpha/saxml-gemma-server);
