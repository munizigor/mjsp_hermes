# TODO: Roadmap de Evolução Arquitetural e Escala do `backend-hermes`

Este documento estrutura as tarefas técnicas de engenharia necessárias para evoluir a arquitetura do `backend/`, preparando-o para os objetivos do Plano de Trabalho de 2027 em diante.

---

## 1. SGBD e Camada de Persistência
Migrar de banco embarcado SQLite para SGBD cliente-servidor dedicado e viabilizar múltiplos containers/nós.

- [ ] **[SGBD-1] Ativar e validar container MariaDB no ambiente de staging (`docker-compose`)**
  - Configurar `healthcheck` no serviço `mariadb` do `docker-compose.yml`.
  - Substituir os loops de espera por arquivo (`while not os.path.exists(SQLITE_DB_PATH)`) nos serviços `staging`, `hermes`, `asr`, `ner` e `interpretation` por dependência de saúde do serviço (`depends_on: condition: service_healthy`) e tentativa de conexão com backoff.
  - Adaptar `src/init.sql` para sintaxe compatível com MariaDB (`AUTO_INCREMENT`, tipos `VARCHAR`/`TEXT`/`DATETIME(6)`, chaves estrangeiras) e criar índices explícitos para `id_emergencia`, `transcription_status`, `start_time`, `end_time` e `tipo_de_inferencia`.
- [ ] **[SGBD-2] Criar camada centralizada de acesso a dados (`db.py`) com Connection Pooling**
  - Substituir as chamadas diretas `sqlite3.connect(os.environ["SQLITE_DB_PATH"])` espalhadas pelos módulos por um conector centralizado (`PyMySQL` / `mariadb` com pool de conexões do `SQLAlchemy Core`).
  - Opcional: Manter suporte opcional a SQLite apenas para execução rápida de testes locais isolados.
- [ ] **[SGBD-3] Migrar queries SQL dos 5 serviços para o conector MariaDB**
  - Padronizar a passagem de parâmetros nas consultas SQL em `hermes/routers/`, `staging/monitors/`, `asr/`, `ner/` e `interpretation/` através da nova camada `db.py`.
  - Adaptar a query de captura de lote do ASR (`src/asr/asr_runner.py`) para transação compatível com MariaDB 10.6+ usando `SELECT ... FOR UPDATE SKIP LOCKED` seguido de `UPDATE`.

---

## 2. Aperfeiçoamento de Orquestração e Escalonamento Horizontal
Garantir que múltiplas instâncias do mesmo serviço de IA não processem a mesma tarefa simultaneamente e otimizar consultas de fila

- [ ] **[ORQ-1] Otimizar consultas de varredura nos loops de `ner` e `interpretation`**
  - Substituir as consultas iterativas por emergência em `emergencias_nao_analizadas()` (`src/ner/runners/ner_agent.py`) e `emergencias_nao_interpretadas()` (`src/interpretation/agent.py`) por **uma única query SQL consolidada com `JOIN`** que retorne diretamente as emergências com `ultimo_horario_de_transcricao > ultimo_horario_inferencia` (filtrando chamadas ativas ou recém-encerradas com pendências).
- [ ] **[ORQ-2] Implementar controle de concorrência distribuído (Claim / Lock atômico no banco)**
  - Substituir o controle de tarefas em memória local (`processando = set()`) por controle distribuído no banco para permitir rodar 2 ou mais containers de `ner` ou `interpretation` em paralelo.
  - Implementar tabela ou colunas de controle de fila/lock no MariaDB (`status_processamento`, `locked_by_worker`, `locked_at`) utilizando `SELECT ... FOR UPDATE SKIP LOCKED`.
  - Implementar mecanismo de *lease timeout* (re-enfileiramento automático caso um worker reinicie ou exceda o tempo limite durante a inferência).
- [ ] **[ORQ-3] Avaliar/Implementar serviço Orquestrador com fila dedicada (Redis / Broker)**
  - Após a conclusão de `[ORQ-1]` e `[ORQ-2]`, validar nos testes de carga do **Épico 6** (cenário multi-UF / "Hermes como Serviço" na STI/MJSP) se o controle via MariaDB `SKIP LOCKED` atende à demanda prevista ou se deve ser introduzido um serviço orquestrador dedicado baseado em **Redis** (Redis Streams / Celery / RQ) para distribuição de tarefas orientadas a eventos.

---

## 3. Melhoria de Envio de Áudio, Ciclo de Vida de Dados e Governança

- [ ] **[AUD-1] Ingestão de áudio via WebSocket (ou gRPC)**
  - Implementar endpoint de streaming/envio contínuo de chunks `.wav`/PCM via WebSocket (`/ws/audio/{id_emergencia}`) ou gRPC para reduzir a latência e o overhead de abertura de conexões HTTP a cada segmento, mantendo o endpoint REST `/send_audio/` por compatibilidade.
- [ ] **[AUD-2] Compactação de áudio e expurgo/transferência periódica do banco de staging (Archiver)**
  - Criar serviço/rotina agendada que processa chamadas encerradas há mais de $N$ dias:
    1. Concatena e comprime os segmentos `.wav` da chamada para formato compacto (ex.: `.opus` ou `.flac`, reduzindo o uso de disco).
    2. Transfere os registros consolidados da ocorrência (`emergencies`, `emergency_transcripts`, `resultados_inferencia`, `call_logs`) para o banco de armazenamento histórico / camada bruta da plataforma corporativa de dados em formatos abertos.
    3. Remove os arquivos `.wav` temporários e limpa os registros antigos do banco de staging operacional para manter alta performance em tempo real.
- [ ] **[GOV-1] Rastreabilidade de modelos/prompts e extensão para Qualidade e Despacho**
  - Registrar de forma padronizada em `resultados_inferencia` a versão do modelo e a versão/hash do template de prompt utilizado em cada inferência (suporte a auditoria, monitoramento de drift e reversão rápida de versão).
  - Preparar o esquema de `resultados_inferencia` e os endpoints de consulta para suportar os novos tipos de inferência previstos no roadmap do projeto:
    - `avaliacao_qualidade`: avaliação pós-atendimento frente aos protocolos.
    - `recomendacao_despacho`: recomendação assistida de recursos/unidades em modo silencioso ou assistido.
