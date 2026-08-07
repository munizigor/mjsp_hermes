import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from django.urls import path
#from hermes_cad.consumers import TranscriptionConsumer
from channels.auth import AuthMiddlewareStack
import hermes_cad.routing

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'hermes_ecossistema.settings')

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(
            hermes_cad.routing.websocket_urlpatterns
        )
    ),
})