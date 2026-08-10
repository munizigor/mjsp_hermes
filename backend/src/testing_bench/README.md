# Módulo de Testes - Hermes Agents

Este módulo contém scripts para testar o sistema Hermes Agents, simulando chamadas de emergência com áudios reais.

## Script Principal: `run_tests.py`

O script `run_tests.py` é responsável por:

1. **Iniciar emergências** na API do Hermes
2. **Enviar áudios** para cada emergência
3. **Aguardar a duração** de cada áudio
4. **Encerrar as emergências** automaticamente
5. **Processar múltiplas chamadas** em paralelo usando threads
6. **Iniciar e finalizar os containers docker** utilizando as configurações indicadas

## Pré-requisitos

- Python 3.11+
- Dependências instaladas
- Conda ou Mamba

```
conda env create -f env.yml
conda activate testing_bench
```

## Uso

### Parâmetros

- **`n_ligacoes`**: Número de ligações a serem realizadas
- **`carga_de_ligacoes`**: Carga de ligações a serem realizadas (minutos de áudio enviados a cada mínuto)
- **`config_path`**: Caminho completo para a pasta do dataset
- **`hermes_addr`**: Endereço da API do Hermes (opcional)

### Exemplos

```bash
# Teste com dataset local
cd src/testing_bench
python run_test.py <n_ligacoes> <carga_de_ligacoes> <config_path> [hermes_addr]

# Exemplo de uso
python run_test.py 10 1.0 .../../envs/config.gcp.apis_cost.json
```

## Funcionamento

### 1. Inicialização
- Verifica se os arquivos de áudio existem
- Carrega o metadata das chamadas
- Valida a conectividade com a API do Hermes

### 2. Processamento Paralelo
- Para cada áudio, cria uma thread separada
- Cada thread:
  - Inicia uma emergência única
  - Envia o áudio correspondente
  - Aguarda a duração especificada no metadata
  - Encerra a emergência

### 3. Controle de Concorrência
- Pequeno delay (0.2s) entre inícios de threads para evitar sobrecarga
- Todas as threads são aguardadas antes do script terminar
- Logs detalhados para cada operação

## Endpoints da API Utilizados

- `GET /start_call_interpretation/` - Inicia nova emergência
- `POST /send_audio/` - Envia áudio para emergência
- `GET /end_call/` - Encerra emergência

## Logs e Monitoramento

O script fornece logs detalhados para:
- Início e fim de cada emergência
- Status do envio de áudios
- Duração de espera para cada chamada
- Finalização de threads
- Resumo final do processamento

## Tratamento de Erros

- Validação de parâmetros de entrada
- Verificação de conectividade com API
- Tratamento de falhas na criação de emergências
- Logs de erro para debugging

## Troubleshooting

### API não acessível
- Verifique se o docker-compose está rodando
- Confirme se a porta 8001 está mapeada corretamente
- Teste com `curl http://localhost:8001/`

### Arquivos não encontrados
- Verifique o caminho do dataset
- Confirme a estrutura de diretórios
- Valide se os arquivos .wav existem

### Falhas de thread
- Verifique os logs de erro
- Confirme se a API está respondendo corretamente
- Valide o formato dos dados de entrada

## Integração com CI/CD

O script pode ser integrado em pipelines de CI/CD para:
- Testes automatizados de funcionalidade
- Validação de deployments
- Testes de carga e performance
- Verificação de integridade do sistema 