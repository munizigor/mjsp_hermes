from django.urls import path
from . import consumers

# Lista de rotas WebSocket do app
websocket_urlpatterns = [
    path("ws/transcription/", consumers.TranscriptionConsumer.as_asgi()),
]
