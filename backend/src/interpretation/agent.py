import json
import sys
import time
from time import sleep
import sqlite3 as sqlite
import os
import multiprocessing as mp
from queue import Empty as QueueEmptyError

"""if not os.path.exists('/logs'):
    print("O diretório /logs não existe no container.")
else:
    print("O diretório /logs existe no container.")
import logging
logging.basicConfig(filename='/logs/main.log', level=logging.INFO)
logger = logging.getLogger(__name__)"""

from interpreters import EmergencyInterpreter

print("Iniciando modulo do InterpretationAgent")


class InterpretationAgent:
    """
    Agente que coordena a coleta de transcrições, execução de intérpretes e persistência.

    Contrato resumido:
    - Inicialização:
        InterpretationAgent(interpreter: EmergencyInterpreter, sqlite_path: str, ...)
        -> espera um `interpreter` já inicializado e caminho para o SQLite que contém
           tabelas 'emergencies', 'emergency_transcripts' e 'resultados_inferencia'.
    - Método principal:
        escutar(): loop que lê emergências não interpretadas e dispara processamentos.
    - Persistência:
        send_interpretations(interpretations): insere linhas na tabela resultados_inferencia.

    Entradas/saídas importantes:
    - transcriptions coletadas via `coletar_transcricao(em_id)` retornam (texto, horario)
    - interpretações geradas pelo interpreter têm formato:
        (resultado_dict_or_None, meta_dict) para create_interpretation_a
        (natureza_name_or_None, meta_dict) para create_interpretation_b
    - `meta_dict` deve incluir pelo menos: processing_time, no_gpu_time, input_tokens, output_tokens, model_name

    Erros e tempo limite:
    - O agente usa filas internas e `redo_queue` para reaplicar interpretações com falha.
    - Se o banco SQLite estiver inacessível, métodos que consultam o DB levantam as exceções correspondentes.
    """

    def __init__(
        self,
        interpreter: EmergencyInterpreter,
        sqlite_path: str,
        n_in_batch: int = 3,
        min_transcricao=150,
        max_similar_natures: int = 16,
    ):
        """
        Inicializa o agente.

        Args:
            interpreter (EmergencyInterpreter): instância responsável por executar os runners.
            sqlite_path (str): caminho para o arquivo SQLite usado para leitura/escrita.
            n_in_batch (int): tamanho de batch para processamento em lote.
            min_transcricao (int): tamanho mínimo da transcrição para considerar processar.
            max_similar_natures (int): número máximo de naturezas semelhantes a considerar.
        """

        self.interpreter = interpreter
        self.sqlite_path = sqlite_path
        self.n_in_batch = n_in_batch
        self.stop_flag = False
        self.running = False
        self.min_transcricao = min_transcricao
        self.max_similar_natures = max_similar_natures
        self.queue_log_path = "/tmp/hermes_queue-interpretation.tsv"
        no_queue = not os.path.exists(self.queue_log_path)
        self.queue_file = open(self.queue_log_path, "a", buffering=1, encoding="utf-8")
        if no_queue:
            self.queue_file.write(
                "\t".join(
                    [
                        "horario",
                        "fechadas",
                        "abertas_interpretadas",
                        "aguardando_conteudo",
                        "esperando_para_processar",
                        "batch_limpo",
                        "batch_com_backlog",
                    ]
                )
                + "\n"
            )
        print("Escrevendo fila em", self.queue_log_path)
        self.ultimos_contextos_processados = {}
        self.inicio_processamento = {}
        self.last_queue_line = ""
        print("InterpretationAgent initialized.")

    def listar_emergencias_abertas(self):
        """
        Retorna lista de IDs de emergências abertas (end_time IS NULL), ordenadas por start_time asc.
        Saída:
            List[int] - lista de ids (pode ser vazia).
        Erros:
            - Se o arquivo SQLite estiver inacessível, propaga exception.
        """
        sqlite_conn = sqlite.connect(self.sqlite_path)
        cursor = sqlite_conn.cursor()
        cursor.execute(
            """SELECT em.id FROM emergencies as em
                WHERE em.end_time IS NULL
            ORDER BY em.start_time ASC
            LIMIT 100"""
        )
        em_ids = cursor.fetchall()
        if em_ids == None:
            em_ids = []
        sqlite_conn.close()
        em_ids = [int(x[0]) for x in em_ids]
        return em_ids

    def listar_emergencias_fechadas(self):
        """
        Retorna lista de IDs de emergências fechadas (end_time IS NOT NULL).
        Saída:
            List[int]
        """
        sqlite_conn = sqlite.connect(self.sqlite_path)
        cursor = sqlite_conn.cursor()
        cursor.execute(
            """SELECT em.id FROM emergencies as em
                WHERE em.end_time IS NOT NULL"""
        )
        em_ids = cursor.fetchall()
        if em_ids == None:
            em_ids = []
        sqlite_conn.close()
        em_ids = [int(x[0]) for x in em_ids]
        return em_ids

    def ultimos_horarios_de_transcricao(self):
        """
        Consulta o banco e retorna para cada emergência o timestamp da última transcrição.
        Retorna:
            List[dict]: [{'id_emergencia': int, 'ultimo_horario_de_transcricao': timestamp_or_None}, ...]
        Uso:
            Usado para decidir quais emergências precisam de reinterpretação.
        """

        query = """SELECT 
            e.id AS emergency_id,
            MAX(et.horario_de_transcricao) AS ultimo_horario_de_transcricao
        FROM 
            emergencies e
        LEFT JOIN 
            emergency_transcripts et
        ON 
            e.id = et.id_emergencia
        GROUP BY 
            e.id;"""

        sqlite_conn = sqlite.connect(self.sqlite_path)
        cursor = sqlite_conn.cursor()
        cursor.execute(query)
        horarios_trans = cursor.fetchall()
        sqlite_conn.close()
        if horarios_trans == None:
            results = []
        else:
            results = [
                {"id_emergencia": int(x[0]), "ultimo_horario_de_transcricao": x[1]}
                for x in horarios_trans
            ]

        return results

    def horario_ultima_interpretacao(self, em_id: int):
        """
        Retorna o horário da última interpretação de 'natureza_decisiva' para a emergência.
        Se nenhuma interpretação registrada, retorna 0.
        """
        # listar linhas de resultados_inferencia associadas com a emergência
        sqlite_conn = sqlite.connect(self.sqlite_path)
        cursor = sqlite_conn.cursor()
        cursor.execute(
            """
            SELECT horario_contexto FROM resultados_inferencia
            WHERE tipo_de_inferencia = 'natureza_decisiva'
                AND id_emergencia = ?
        """,
            (em_id,),
        )
        horario_contexto = cursor.fetchall()
        sqlite_conn.close()
        if horario_contexto == None:
            horario_contexto = []
        if len(horario_contexto) > 0:
            ultimo_ponto = max([x[0] for x in horario_contexto])
            # desde_interpretacao = time.time() - ultimo_ponto
            return ultimo_ponto
        else:
            return 0

    def emergencias_nao_interpretadas(self):
        """
        Compara timestamps de transcrição com último horário de interpretação
        e retorna lista de emergências que precisam ser interpretadas.
        Retorna:
            List[int] - ids preparados para processamento.
        """
        horarios_ultimas_transcricoes = self.ultimos_horarios_de_transcricao()
        emergencias = []
        for h in horarios_ultimas_transcricoes:
            id_emergencia = int(h["id_emergencia"])
            if h["ultimo_horario_de_transcricao"] != None:
                ultimo_horario = float(h["ultimo_horario_de_transcricao"])
                horario_interpretacao = self.horario_ultima_interpretacao(id_emergencia)
                if horario_interpretacao < ultimo_horario:
                    emergencias.append([id_emergencia, horario_interpretacao])

        emergencias.sort(key=lambda x: x[1])
        id_emergencias = [x[0] for x in emergencias]
        return id_emergencias

    def coletar_transcricao(self, em_id: int):
        """
        Recupera e concatena todas as partes de transcrição para a emergência `em_id`.
        Retorna:
            (context_text: str, horario: int)
        Notas:
            - Se não houver transcrições, retorna ("Sem transcrição", 0).
        """
        sqlite_conn = sqlite.connect(self.sqlite_path)
        cursor = sqlite_conn.cursor()
        cursor.execute(
            """
            SELECT et.part, et.horario_de_transcricao 
            FROM emergency_transcripts et
            WHERE et.id_emergencia = ?
            ORDER BY et.start_time ASC
        """,
            (em_id,),
        )
        transcript_parts = cursor.fetchall()
        sqlite_conn.close()
        if transcript_parts == None:
            transcript_parts = []
        if len(transcript_parts) > 0:
            context = " ".join([part[0] for part in transcript_parts])
            horario = max([float(part[1]) for part in transcript_parts])
            # context = {'transcription': context, 'id_emergencia': em_id}
            return context, horario
        else:
            return "Sem transcrição", 0

    def parse_queue(
        self,
        tempo_atual_f,
        emergencias_fec: set,
        emergencias_abertas,
        nao_interpretadas,
        nao_interpretadas_sem_processamentos,
        prontas_para_processar,
        processando,
        batch_com_backlog,
    ):
        """
        Gera uma linha de log simplificada do estado da fila (usada para monitoramento/debug).
        Não altera estado do agente, apenas formata e escreve em `self.queue_file` quando houver mudança.
        """
        abertas_interpretadas = emergencias_abertas - set(nao_interpretadas)
        aguardando_conteudo = (
            set(nao_interpretadas_sem_processamentos) - prontas_para_processar
        )
        esperando_para_processar = (
            prontas_para_processar - processando - batch_com_backlog
        )
        batch_limpo = processando - batch_com_backlog
        # com_delay = emergencias_fec
        queue_line = [
            sorted(emergencias_fec),
            sorted(abertas_interpretadas),
            sorted(aguardando_conteudo),
            sorted(esperando_para_processar),
            sorted(batch_limpo),
            sorted(batch_com_backlog),
        ]
        queue_line_txt = "\t".join([str(x) for x in queue_line]) + "\n"
        if (
            queue_line_txt != self.last_queue_line
            and len(queue_line_txt.replace("\t", "")) > 1
        ):
            self.last_queue_line = queue_line_txt
            self.queue_file.write(str(tempo_atual_f) + "\t" + queue_line_txt)
            """time_logs = []
            for em_id, start_process in self.inicio_processamento.items():
                txt = str(em_id) + ': ' + str(round(tempo_atual_f-start_process, 1))
                time_logs.append(txt)
            if len(time_logs) > 0:
                self.queue_file.write('#Atrasos dos processamentos:\t' + '; '.join(time_logs) + '\n')"""

    def escutar(self):
        """
        Loop principal do agente que monitora emergências e dispara processamentos.

        Fluxo resumido:
        - Cria `redo_queue` para sinalizar reprocessamentos rejeitados.
        - Em cada iteração, lista emergências abertas/fechadas e determina quais precisam interpretar.
        - Agrupa contextos e envia para processamento (batch ou async) conforme `self.interpreter.concurrency_mode`.
        - Mantém logs simples em `self.queue_file`.

        Notas operacionais:
        - Método bloqueante; para parar, setar `self.stop_flag = True`.
        - Usa timeout curto para checar itens em `redo_queue` (não bloquear o loop).
        """
        self.running = True
        print("Listening for interpretations...")

        redo_queue = mp.Queue()
        self.interpreter.set_agent_can_redo_queue(redo_queue)
        processando = set()

        try:

            while not self.stop_flag:
                agent_cicly_start = time.time()
                redo_timeout_max = 0.03
                while time.time() - agent_cicly_start < redo_timeout_max:
                    try:
                        redo_item = redo_queue.get(timeout=0.01)
                        # Sinalizar que a emergência não está mais em interpretação
                        em_id, razao = redo_item
                        if em_id in processando:
                            processando.remove(em_id)
                        print(
                            f"InterpretationAgent.escutar: Re-enfileirando para Interpretação {em_id}"
                        )
                        print(
                            f"InterpretationAgent.escutar: Motivo do retorno de {em_id}:",
                            str(razao),
                        )
                    except QueueEmptyError as err:
                        break

                redo_queue_time = time.time() - agent_cicly_start

                listagens_inicio = time.time()
                interpretacao_feita = False
                emergencias_fec = set(self.listar_emergencias_fechadas())
                emergencias_abertas = set(self.listar_emergencias_abertas())
                prontas_para_processar = set()
                finalizadas_no_aguardo = set()
                nao_interpretadas = self.emergencias_nao_interpretadas()
                tempo_listagens = time.time() - listagens_inicio

                tempo_atual_f = time.time()
                tempo_atual = int(tempo_atual_f)
                batch_com_backlog = set(
                    [x for x in nao_interpretadas if x in processando]
                )
                nao_interpretadas_sem_processamentos = [
                    x for x in nao_interpretadas if x not in processando
                ]
                tempo_transcricoes = 0.0
                tempo_envio = 0.0
                if len(nao_interpretadas_sem_processamentos) > 0:
                    para_processar = []
                    transcricao_inicio = time.time()
                    for em_id in nao_interpretadas_sem_processamentos:
                        texto, horario = self.coletar_transcricao(em_id)
                        if not em_id in self.ultimos_contextos_processados:
                            self.ultimos_contextos_processados[em_id] = ""
                        new_context = {
                            "id_emergencia": em_id,
                            "horario_ultima": horario,
                            "delay": tempo_atual_f - horario,
                            "transcription": texto,
                        }
                        if (
                            len(texto)
                            > (
                                len(self.ultimos_contextos_processados[em_id])
                                + self.min_transcricao
                            )
                            or em_id in emergencias_fec
                        ):
                            self.ultimos_contextos_processados[em_id] = texto
                            para_processar.append(new_context)
                        if em_id in emergencias_fec:
                            finalizadas_no_aguardo.add(em_id)
                    tempo_transcricoes = time.time() - transcricao_inicio

                    inicio_envio = time.time()
                    if len(para_processar) > 0:
                        prontas_para_processar.update(
                            [a["id_emergencia"] for a in para_processar]
                        )
                        print("\nEmergencias com transcricao nao interpretada:")
                        for em in para_processar:
                            print(
                                "Emergencia: "
                                + json.dumps(em, indent=4, ensure_ascii=False)
                                + "\n"
                            )
                        space_available = self.n_in_batch - len(processando)
                        batch_size = min(space_available, len(para_processar))
                        if batch_size > 0:
                            batch = para_processar[:batch_size]
                            processando.update([a["id_emergencia"] for a in batch])
                            """for a in batch:
                                self.inicio_processamento[a['id_emergencia']] = tempo_atual_f"""
                            try:
                                self.interpreter.process_emergencies(
                                    batch, max_naturezas=self.max_similar_natures
                                )
                                # o interpretador salva as interpretacoes por conta propria
                                interpretacao_feita = True
                            except KeyError as err:
                                print(
                                    "Erro KeyError ao chamar self.interpreter.process_emergencies:",
                                    file=sys.stderr,
                                )
                                print(err, file=sys.stderr)
                                raise (err)
                            except Exception as err:
                                print(
                                    "Erro ao chamar self.interpreter.process_emergencies:",
                                    file=sys.stderr,
                                )
                                print(type(err), file=sys.stderr)
                                print(err, file=sys.stderr)
                                print(err.args, file=sys.stderr)
                                if "Error at extract_multiple_structured" in str(err):
                                    pass
                                else:
                                    print(err.__traceback__, file=sys.stderr)
                                    raise (err)
                    tempo_envio = time.time() - inicio_envio

                self.parse_queue(
                    tempo_atual_f,
                    emergencias_fec,
                    emergencias_abertas,
                    nao_interpretadas,
                    nao_interpretadas_sem_processamentos,
                    prontas_para_processar,
                    processando,
                    batch_com_backlog,
                )
                cycle_len = time.time() - agent_cicly_start
                if cycle_len > 0.5:
                    print(
                        "ciclo lento no agente:",
                        cycle_len,
                        redo_queue_time,
                        tempo_listagens,
                        tempo_transcricoes,
                        tempo_envio,
                        file=sys.stderr,
                    )
                if not interpretacao_feita:
                    sleep(0.01)  # waiting for input
            self.running = False
        except Exception as err:
            self.running = False
            raise (err)

    def escutar_async(self):
        """
        Loop principal do agente que monitora emergências e dispara processamentos.

        Fluxo resumido:
        - Cria `redo_queue` para sinalizar reprocessamentos rejeitados.
        - Em cada iteração, lista emergências abertas/fechadas e determina quais precisam interpretar.
        - Agrupa contextos e envia para processamento (batch ou async) conforme `self.interpreter.concurrency_mode`.
        - Mantém logs simples em `self.queue_file`.

        Notas operacionais:
        - Método bloqueante; para parar, setar `self.stop_flag = True`.
        - Usa timeout curto para checar itens em `redo_queue` (não bloquear o loop).
        """
        self.running = True
        print("Listening for interpretations...")

        redo_queue = mp.Queue()
        self.interpreter.set_agent_can_redo_queue(redo_queue)
        processando = set()

        try:

            while not self.stop_flag:
                agent_cicly_start = time.time()
                redo_timeout_max = 0.03
                while time.time() - agent_cicly_start < redo_timeout_max:
                    try:
                        redo_item = redo_queue.get(timeout=0.01)
                        # Sinalizar que a emergência não está mais em interpretação
                        em_id, razao = redo_item
                        if em_id in processando:
                            processando.remove(em_id)
                        print(
                            f"InterpretationAgent.escutar: Re-enfileirando para Interpretação {em_id}"
                        )
                        print(
                            f"InterpretationAgent.escutar: Motivo do retorno de {em_id}:",
                            str(razao),
                        )
                    except QueueEmptyError as err:
                        break

                redo_queue_time = time.time() - agent_cicly_start

                listagens_inicio = time.time()
                interpretacao_feita = False
                emergencias_fec = set(self.listar_emergencias_fechadas())
                emergencias_abertas = set(self.listar_emergencias_abertas())
                prontas_para_processar = set()
                finalizadas_no_aguardo = set()
                nao_interpretadas = self.emergencias_nao_interpretadas()
                tempo_listagens = time.time() - listagens_inicio

                tempo_atual_f = time.time()
                tempo_atual = int(tempo_atual_f)
                batch_com_backlog = set(
                    [x for x in nao_interpretadas if x in processando]
                )
                nao_interpretadas_sem_processamentos = [
                    x for x in nao_interpretadas if x not in processando
                ]
                tempo_transcricoes = 0.0
                tempo_envio = 0.0
                if len(nao_interpretadas_sem_processamentos) > 0:
                    para_processar = []
                    transcricao_inicio = time.time()
                    for em_id in nao_interpretadas_sem_processamentos:
                        texto, horario = self.coletar_transcricao(em_id)
                        if not em_id in self.ultimos_contextos_processados:
                            self.ultimos_contextos_processados[em_id] = ""
                        new_context = {
                            "id_emergencia": em_id,
                            "horario_ultima": horario,
                            "delay": tempo_atual_f - horario,
                            "transcription": texto,
                        }
                        # if (len(texto) > (len(self.ultimos_contextos_processados[em_id]) + self.min_transcricao)
                        #    or em_id in emergencias_fec):
                        self.ultimos_contextos_processados[em_id] = texto
                        para_processar.append(new_context)
                        if em_id in emergencias_fec:
                            finalizadas_no_aguardo.add(em_id)
                    tempo_transcricoes = time.time() - transcricao_inicio

                    inicio_envio = time.time()
                    prontas_para_processar.update(
                        [a["id_emergencia"] for a in para_processar]
                    )
                    print("\nEmergencias com transcricao nao interpretada:")
                    for em in para_processar:
                        print(
                            "Emergencia: "
                            + json.dumps(em, indent=4, ensure_ascii=False)
                            + "\n"
                        )
                    processando.update([a["id_emergencia"] for a in para_processar])
                    try:
                        self.interpreter.process_emergencies(
                            para_processar, max_naturezas=self.max_similar_natures
                        )
                        # o interpretador salva as interpretacoes por conta propria
                        interpretacao_feita = True
                    except KeyError as err:
                        print(
                            "Erro KeyError ao chamar self.interpreter.process_emergencies:",
                            file=sys.stderr,
                        )
                        print(err, file=sys.stderr)
                        raise (err)
                    except Exception as err:
                        print(
                            "Erro ao chamar self.interpreter.process_emergencies:",
                            file=sys.stderr,
                        )
                        print(type(err), file=sys.stderr)
                        print(err, file=sys.stderr)
                        print(err.args, file=sys.stderr)
                        if "Error at extract_multiple_structured" in str(err):
                            pass
                        else:
                            print(err.__traceback__, file=sys.stderr)
                            raise (err)
                    tempo_envio = time.time() - inicio_envio

                self.parse_queue(
                    tempo_atual_f,
                    emergencias_fec,
                    emergencias_abertas,
                    nao_interpretadas,
                    nao_interpretadas_sem_processamentos,
                    prontas_para_processar,
                    processando,
                    batch_com_backlog,
                )
                cycle_len = time.time() - agent_cicly_start
                if cycle_len > 0.5:
                    print(
                        "ciclo lento no agente:",
                        cycle_len,
                        redo_queue_time,
                        tempo_listagens,
                        tempo_transcricoes,
                        tempo_envio,
                        file=sys.stderr,
                    )
                if not interpretacao_feita:
                    sleep(0.01)  # waiting for input
            self.running = False
        except Exception as err:
            self.running = False
            raise (err)

    def stop(self, max_wait: float = 8.0):
        print("Stopping interpretation agent...")
        self.stop_flag = True
        while self.running and max_wait > 0:
            sleep(0.5)
            max_wait -= 0.5
        # self.
        self.queue_file.close()
        self.interpreter.close()
