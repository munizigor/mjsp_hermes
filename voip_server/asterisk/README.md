# Agente de Integração Asterisk / Backend Hermes

## Visão Geral

Este projeto tem como objetivo integrar um servidor **Asterisk** ao ambiente de nuvem **Hermes**, possibilitando diferentes formas de transcrição de chamadas:

- **Transcrição Local**: processa o áudio no próprio servidor, enviando apenas o texto resultante (ideal em cenários com restrições de latência e/ou largura de banda).
- **Transcrição em Nuvem**: envia o áudio para o Hermes, garantindo maior precisão, desde que haja boa qualidade de rede.

O objetivo final é oferecer flexibilidade, permitindo escolher a melhor estratégia de transcrição de acordo com o ambiente e as necessidades do sistema.

---

## Status Atual do Projeto

Atualmente, estão implementadas as partes de **sinalização de chamadas** e **segmentação de áudio em tempo real**:

- O Asterisk sinaliza o início e o fim das chamadas via eventos AMI.
- O Asterisk grava chamadas via MixMonitor, gerando arquivos WAV em crescimento.
- O sistema acessa diretamente esses arquivos, ignorando o cabeçalho incompleto e tratando o áudio como um **fluxo contínuo de dados**.
- Assim, é possível detectar pontos de silêncio e segmentar o áudio **sem precisar esperar o término da chamada**.

As próximas etapas do desenvolvimento irão focar na integração completa com os módulos de **transcrição local** e **transcrição em nuvem**.

## Componentes

1. **sip-sign.py**: Conecta-se à AMI do Asterisk, monitora eventos de chamada (Newstate e Hangup) e envia notificações ao backend Hermes.
2. **direct_raw_processor.py**: Processa diretamente os dados brutos de um arquivo WAV em crescimento.
3. **direct_monitor.py**: Monitora o diretório de gravações e inicia o processamento de cada novo arquivo WAV.
4. **hermes-asterisk-agent.service**: Arquivo de configuração do serviço systemd para o processamento de áudio.
5. **hermes-sip-sign.service**: Arquivo de configuração do serviço systemd para a sinalização.

## Requisitos

- Linux com Asterisk (ex: Issabel, FreePBX)
- Python 3.6+
- ffmpeg
- inotify-tools

---

## Configuração Prévia do Asterisk

Antes de iniciar os serviços, é obrigatório preparar o Asterisk para permitir a comunicação via AMI e garantir que as gravações ocorram no formato e diretório corretos.

### 1. Criação do Usuário na AMI

O script de sinalização (`sip-sign.py`) requer acesso ao Asterisk Manager Interface (AMI) para ler eventos de chamada.

Edite o arquivo `/etc/asterisk/manager_custom.conf` (ou `manager.conf`, dependendo da sua distribuição) e adicione o seguinte bloco:

```ini
[hermes-sip-sign]
secret = sua_senha_segura_aqui
deny = 0.0.0.0/0.0.0.0
permit = 127.0.0.1/255.255.255.255
read = system,call,log,verbose,command,agent,user,config,dtmf,reporting,cdr,dialplan,originate
write = system,call,log,verbose,command,agent,user,config,command,reporting,originate
```

Após salvar, aplique as configurações executando no terminal:
```bash
asterisk -rx "manager reload"
```

### 2. Configuração do Dialplan (`extensions_custom.conf`)

O processador de áudio (`direct_raw_processor.py`) espera um formato cru de **WAV (PCM, 16-bit, 8000Hz, mono)**. Se o Asterisk gerar arquivos compactados (como `wav49` ou `.WAV` maiúsculo), o script falhará.

Para garantir que o MixMonitor grave corretamente, adicione as cláusulas abaixo no arquivo `/etc/asterisk/extensions_custom.conf`:

```ini
[custom-gravar-hermes]
exten => s,1,NoOp(--- Ativando MixMonitor para o Agente Hermes ---)
; Define a variável com o caminho completo e o nome do arquivo baseado no UniqueID
exten => s,n,Set(MONITOR_FILENAME=/var/spool/asterisk/monitor/${UNIQUEID})
; Inicia a gravação forçando a extensão minúscula (.wav) que corresponde ao formato PCM 16-bit
exten => s,n,MixMonitor(${MONITOR_FILENAME}.wav,b)
exten => s,n,Return()
```

*Nota:* Você deve garantir que suas rotas de entrada/saída (Inbound/Outbound Routes) ou ramais direcionem a chamada para este contexto via `GoSub(custom-gravar-hermes,s,1)` antes de efetuar o Dial, ou configurar a interface da sua PBX (ex: Issabel/FreePBX) para salvar gravações exclusivamente no formato `wav`.

---

## Instalação

1. **Copie os scripts para o servidor Asterisk/Issabel**:

```bash
mkdir -p /opt/hermes-asterisk-agent
cp direct_raw_processor.py direct_monitor.py sip-sign.py /opt/hermes-asterisk-agent/
chmod +x /opt/hermes-asterisk-agent/*.py
chown -R asterisk:asterisk /opt/hermes-asterisk-agent/
```

2. **Instale as dependências necessárias**:

```bash
yum install ffmpeg inotify-tools
pip3 install watchdog numpy audioop requests
```

3. **Configure as Credenciais no sip-sign.py**:

Edite o arquivo `/opt/hermes-asterisk-agent/sip-sign.py` e configure diretamente as variáveis no início do código, inserindo a senha do usuário AMI, além do IP, Porta e a Chave da API do Hermes.

```python
# Configurações AMI
AMI_HOST = "127.0.0.1"
AMI_PORT = 5038
AMI_USER = "hermes-sip-sign"
AMI_PASS = "sua_senha_segura_aqui"

# Configurações API Hermes
API_HOST = "api_ip_ou_hostname"
API_PORT = 8001
API_KEY = "sua_chave_de_api_aqui"
```

4. **Configure os serviços systemd**:

```bash
cp hermes-asterisk-agent.service /etc/systemd/system/
cp hermes-sip-sign.service /etc/systemd/system/

systemctl daemon-reload
systemctl enable hermes-asterisk-agent.service
systemctl enable hermes-sip-sign.service

systemctl start hermes-asterisk-agent.service
systemctl start hermes-sip-sign.service
```

5. **Verifique se os serviços estão funcionando**:

```bash
systemctl status hermes-asterisk-agent.service
systemctl status hermes-sip-sign.service
```

## Como Funciona

1. O script `sip-sign.py` detecta uma nova chamada via AMI e avisa a API do backend Hermes.
2. O script `direct_monitor.py` monitora o diretório de gravações do Asterisk para novos arquivos WAV iniciados pelo `MixMonitor`.
3. Quando um novo arquivo é detectado, o script inicia um processo `direct_raw_processor.py` para esse arquivo.
4. O `direct_raw_processor.py` acessa diretamente os dados brutos do arquivo, pulando o cabeçalho WAV.
5. O processador analisa o áudio em tempo real, detectando pontos de silêncio com base em um limiar dinâmico.
6. Quando um ponto de silêncio é detectado, o processador salva o segmento de áudio anterior como um arquivo WAV separado para posterior transcrição.
7. Ao término da chamada, o AMI avisa o `sip-sign.py`, que encerra a sinalização junto ao backend.

## Configuração Avançada de Áudio

Você pode ajustar os parâmetros de detecção de silêncio editando o arquivo de serviço:

```bash
vi /etc/systemd/system/hermes-asterisk-agent.service
```

Modifique a linha `ExecStart` para incluir os parâmetros desejados:

```ini
ExecStart=/usr/bin/python3 /opt/hermes-asterisk-agent/direct_monitor.py /var/spool/asterisk/monitor --folga-db 10 --min-silence 800 --keep-silence 250
```

Depois, recarregue e reinicie o serviço:

```bash
systemctl daemon-reload
systemctl restart hermes-asterisk-agent.service
```

## Parâmetros (Áudio)

- `--folga-db`: Valor em dB subtraído do volume médio para definir o limiar de silêncio (padrão: 12)
- `--min-silence`: Duração mínima do silêncio em ms para considerar um ponto de corte (padrão: 1000)
- `--keep-silence`: Quantidade de silêncio a manter no início e fim dos segmentos em ms (padrão: 250)

## Logs

Para verificar os logs dos serviços em tempo real:

```bash
# Logs do processamento de áudio
journalctl -u hermes-asterisk-agent.service -f

# Logs da sinalização AMI
journalctl -u hermes-sip-sign.service -f
```

## Observações

- O sistema assume que os arquivos WAV têm um formato específico (8000 Hz, 16 bits, mono). Se o seu Asterisk estiver configurado para usar um formato diferente, você precisará ajustar os parâmetros no script `direct_raw_processor.py` e forçar a adequação no `extensions_custom.conf`.
