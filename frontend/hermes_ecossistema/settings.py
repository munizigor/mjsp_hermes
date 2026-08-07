import sys
import os
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(os.path.join(BASE_DIR, '.env'))


#TODO: Ajustes para produção (template e key):
# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure--uxv7swvp0q_$&q0l&tn$dl*$hxyl&ie-f-lc3drs+)n5k+*sv'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True
#TODO: Ajuste a ser feito para PRD
#DEBUG = os.environ.get("DJANGO_DEBUG", "True") == "True"

## 🛠️ No seu settings.py

ALLOWED_HOSTS = [
    "localhost",
    "127.0.0.1",
    "hermes-labs.samare.com.br",
    "35.215.198.95", 
]

ALLOWED_HOSTS_str = os.environ["ALLOWED_HOSTS"] if "ALLOWED_HOSTS" in os.environ else None
if ALLOWED_HOSTS_str not in ["", None]:
    ALLOWED_HOSTS += ALLOWED_HOSTS_str.split(",")

X_FRAME_OPTIONS = 'ALLOWALL'

HERMES_BACKEND_URL = os.getenv("HERMES_BACKEND_URL")
HERMES_API_KEY = os.getenv("HERMES_API_KEY")
print("⚙️ HERMES_BACKEND_URL =", HERMES_BACKEND_URL)
print("⚙️ ALLOWED_HOSTS =", ALLOWED_HOSTS)


    

# Application definition

INSTALLED_APPS = [
    'channels', 
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'hermes_cad',
]

ASGI_APPLICATION = 'hermes_ecossistema.asgi.application'

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'corsheaders.middleware.CorsMiddleware',
]

CSRF_TRUSTED_ORIGINS = [
    "https://hermes-labs.samare.com.br",
]

CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels.layers.InMemoryChannelLayer",
        # Para produção usando Redis:
        # "BACKEND": "channels_redis.core.RedisChannelLayer",
        # "CONFIG": {"hosts": [("127.0.0.1", 6379)]},
    },
}

ROOT_URLCONF = 'hermes_ecossistema.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'hermes_ecossistema.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}


AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True

# STATIC_URL = 'static/'
# Usando para testar localmente WebSocket
STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

CORS_ORIGIN_ALLOW_ALL = True

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
    "loggers": {
        "django": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": True,
        },
        "channels": {
            "handlers": ["console"],
            "level": "DEBUG",   # WebSocket debug
            "propagate": False,
        },
        "grpc": {
            "handlers": ["console"],
            "level": "DEBUG",   # gRPC debug
            "propagate": False,
        },
    },
}
