from locust import HttpUser, between, task, tag
import random
import logging

logger = logging.getLogger("locust.performance")


class EmergenciaUser(HttpUser):
    """Flow 2, 3, 4: Teste end-to-end (criar -> enviar -> consultar -> encerrar)"""
    wait_time = between(1, 5)

    def on_start(self):
        """Executado quando tenta iniciar uma emergência e salva o ID."""
        self.id_emergencia = None
        with self.client.get(
            "/proxy/start_call_interpretation/?operator_unique_code=locust_test",
            catch_response=True,
            name="/start_call_interpretation/",
        ) as response:
            if response.status_code == 200:
                try:
                    self.id_emergencia = response.json().get("new_emergency_id")
                except Exception:
                    response.failure("Resposta JSON inválida ao iniciar emergência")
                    logger.exception("Falha ao parsear JSON de start_call_interpretation")
            else:
                response.failure("Não conseguiu iniciar a emergência")

    @tag("transcricao")
    @task(5)
    def enviar_texto(self):
        if not self.id_emergencia:
            self.on_start()
            if not self.id_emergencia:
                return

        frases = [
            "Preciso de uma ambulância na Rua Alagoas",
            "O paciente está desmaiado",
            "Tem muito sangue no local",
            "O nome do solicitante é João",
        ]

        payload_text = random.choice(frases)
        url = f"/proxy/send_transcript/?id_emergencia={self.id_emergencia}&transcript={payload_text}"
        with self.client.post(url, catch_response=True, name="/send_transcript/") as response:
            if response.status_code not in (200, 201):
                response.failure(f"Falha ao enviar transcript: {response.status_code}")
                logger.warning("send_transcript falhou com status %s", response.status_code)

    @tag("inferencia")
    @task(10)
    def consultar_inferencia(self):
        if not self.id_emergencia:
            self.on_start()
            if not self.id_emergencia:
                return

        url = f"/proxy/get_all_inference_results/?id_emergencia={self.id_emergencia}"
        with self.client.get(url, catch_response=True, name="/get_all_inference_results/") as response:
            if response.status_code != 200:
                response.failure(f"Falha ao consultar inferencia: {response.status_code}")
                logger.debug("get_all_inference_results retornou %s", response.status_code)

    @tag("encerrar")
    @task(1)
    def finalizar_chamada(self):
        if not self.id_emergencia:
            return

        url = f"/proxy/end_call/?id_emergencia={self.id_emergencia}"
        with self.client.post(url, catch_response=True, name="/end_call/") as response:
            if response.status_code not in (200, 204):
                response.failure(f"Falha ao encerrar chamada: {response.status_code}")
            else:
                # Reinicia o ciclo
                self.id_emergencia = None
                self.on_start()


class EmergenciaUserFlow1(HttpUser):
    """Flow 1: Testa performance de consulta em emergências pré-existentes (IDs 1-20)"""
    wait_time = between(1, 5)

    def on_start(self):
        """Seleciona um ID de emergência existente entre 1 e 20"""
        self.id_emergencia = random.randint(1, 20)
        logger.info(f"Usuário iniciado com ID de emergência: {self.id_emergencia}")

    @tag("inferencia")
    @task(10)
    def consultar_inferencia(self):
        """Testa a carga de consulta de resultados de inferência"""
        url = f"/proxy/get_all_inference_results/?id_emergencia={self.id_emergencia}"
        with self.client.get(url, catch_response=True, name="/get_all_inference_results/") as response:
            if response.status_code != 200:
                response.failure(f"Falha ao consultar inferencia: {response.status_code}")
                logger.debug("get_all_inference_results retornou %s", response.status_code)
            else:
                logger.debug(f"Consulta OK para ID: {self.id_emergencia}")
