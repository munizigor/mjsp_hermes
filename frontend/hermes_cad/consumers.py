import json
from channels.generic.websocket import AsyncWebsocketConsumer
import logging
import uuid

logger = logging.getLogger("TranscriptionConsumer")

class TranscriptionConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        await self.accept()
        self.session_id = str(uuid.uuid4())
        logger.info(f"WebSocket conectado (session={self.session_id})")

    async def disconnect(self, close_code):
        logger.info(f"WebSocket desconectado (session={self.session_id}, code={close_code})")

    async def receive(self, text_data=None, bytes_data=None):
        if text_data:
            try:
                payload = json.loads(text_data)
            except Exception:
                payload = {"raw": text_data}

            cmd = payload.get("command")
            if cmd == "stop":
                logger.info(f"Recebido comando STOP do front (session={self.session_id})")
                await self.send_json({"message": "Transcrição parada (simulada)."})

    async def send_json(self, data: dict):
        await self.send(text_data=json.dumps(data))
