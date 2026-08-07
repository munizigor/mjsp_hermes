import grpc
from concurrent import futures
import asyncio
import logging

import hermes_pb2 as hermes_pb2
import hermes_pb2_grpc as hermes_pb2_grpc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("MockTranscriptionServer")


class MockTranscriptionServicer(hermes_pb2_grpc.HermesServiceServicer):
    """
    Servidor gRPC mock para simular transcrição de áudio.
    """

    async def ProcessAndTranscribe(self, request_iterator, context):
        logger.info("Recebendo fluxo de áudio do Hermes...")

        try:
            async for request in request_iterator:
                logger.debug(f"Chunk de áudio recebido (id={request.id}, size={len(request.audioData)} bytes)")

            logger.info("Fim do fluxo de áudio. Iniciando transcrição simulada...")

            dummy_transcription = [
                "Soldado Mariana Bombeiros boa tarde \n",
                "Alô, estou ligando para reportar um acidente \n",
                "Qual seu nome. Paulo \n",
                "Paulo, em que posso ajudar. \n",
                "Esta tendo um incêndio aqui na Rua das Flores 32 \n",
                "bairro Atalaia \n",
                "tem algum ponto de referencia? \n",
                "Perto do Petrox \n",
                "seu telefone é 79-99991-2323 \n",
                "certo tá bem Paulo, já estou encaminhando uma equipe aqui para realizar o controle do incêndio Tá ok?",
                "Ok, aguardo a chegada dos bombeiros \n",
                "Obrigado \n",
                "De nada. Boa tarde \n",
            ]

            for text_chunk in dummy_transcription:
                logger.debug(f"Enviando transcrição parcial: {text_chunk.strip()}")
                yield hermes_pb2.TranscriptionResponse(text_chunk=text_chunk)
                await asyncio.sleep(1.5)

            logger.info("Transcrição simulada concluída. Enviando sinal de fim de stream.")
            yield hermes_pb2.TranscriptionResponse(text_chunk="", end_of_stream=True)

        except Exception as e:
            logger.error(f"Erro em ProcessAndTranscribe: {e}", exc_info=True)
            raise


async def serve(port_number: int):
    #50051
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    hermes_pb2_grpc.add_HermesServiceServicer_to_server(MockTranscriptionServicer(), server)
    server.add_insecure_port('[::]:'+str(port_number))

    await server.start()
    logger.info("Servidor gRPC mock rodando na porta "+str(port_number))
    await server.wait_for_termination()