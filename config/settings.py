from __future__ import annotations

import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
VOXHELM_OPERATOR_PRODUCER_LABEL = "__operator_ui__"
VOXHELM_RESERVED_BEARER_TOKEN_LABELS = {VOXHELM_OPERATOR_PRODUCER_LABEL}


def env_list(name: str, *, default: str = "") -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.replace("\n", ",").split(",") if item.strip()]


def env_bool(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def env_map(name: str) -> dict[str, str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return {}
    entries = (
        entry.split("=", 1) for entry in raw.replace("\n", ",").split(",") if entry.strip()
    )
    return {
        key.strip(): value.strip()
        for key, value in entries
    }


def env_tokens(name: str) -> dict[str, str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return {}
    if raw.startswith("{"):
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError(f"{name} must be a JSON object when JSON syntax is used.")
        return validate_bearer_token_labels(
            name,
            {str(key): str(value) for key, value in parsed.items()},
        )

    tokens: dict[str, str] = {}
    for entry in raw.replace("\n", ",").split(","):
        normalized = entry.strip()
        if not normalized:
            continue
        if "=" not in normalized:
            raise ValueError(f"Invalid {name} entry '{normalized}'. Use label=token pairs.")
        label, token = normalized.split("=", 1)
        tokens[label.strip()] = token.strip()
    return validate_bearer_token_labels(name, tokens)


def validate_bearer_token_labels(name: str, tokens: dict[str, str]) -> dict[str, str]:
    reserved = sorted(VOXHELM_RESERVED_BEARER_TOKEN_LABELS.intersection(tokens))
    if reserved:
        labels = ", ".join(reserved)
        raise ValueError(f"{name} contains reserved label(s): {labels}.")
    return tokens


def get_accepted_stt_models() -> set[str]:
    from django.conf import settings as django_settings

    models = {
        "gpt-4o-mini-transcribe",
        "whisper-1",
        django_settings.VOXHELM_MLX_MODEL,
        django_settings.VOXHELM_WHISPERCPP_MODEL,
    }
    if django_settings.VOXHELM_WHISPERKIT_ENABLED:
        models.update({"whisperkit", django_settings.VOXHELM_WHISPERKIT_MODEL})
    return models


def get_batch_accepted_stt_models() -> set[str]:
    return {"auto", *get_accepted_stt_models()}


SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "dev-only-secret-key")
DEBUG = os.getenv("DJANGO_DEBUG", "").lower() in {"1", "true", "yes", "on"}
ALLOWED_HOSTS = env_list("VOXHELM_ALLOWED_HOSTS", default="localhost,127.0.0.1")
ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

INSTALLED_APPS = [
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django_tasks",
    "django_tasks_db",
    "operators",
    "transcriptions",
    "jobs",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
]

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
            ]
        },
    }
]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

LOGIN_URL = "/"
LOGIN_REDIRECT_URL = "/"
LOGOUT_REDIRECT_URL = "/"

VOXHELM_BEARER_TOKENS = env_tokens("VOXHELM_BEARER_TOKENS")
VOXHELM_STT_BACKEND = os.getenv("VOXHELM_STT_BACKEND", "whispercpp").strip()
VOXHELM_STT_FALLBACK_BACKEND = os.getenv("VOXHELM_STT_FALLBACK_BACKEND", "mlx").strip()
VOXHELM_MLX_MODEL = os.getenv("VOXHELM_MLX_MODEL", "mlx-community/whisper-large-v3-mlx")
VOXHELM_WHISPERCPP_MODEL = os.getenv("VOXHELM_WHISPERCPP_MODEL", "ggml-large-v3.bin").strip()
VOXHELM_WHISPERCPP_BIN = os.getenv("VOXHELM_WHISPERCPP_BIN", "/opt/homebrew/bin/whisper-cli")
VOXHELM_WHISPERCPP_PROCESSORS = int(os.getenv("VOXHELM_WHISPERCPP_PROCESSORS", "4"))
VOXHELM_WHISPERKIT_ENABLED = env_bool("VOXHELM_WHISPERKIT_ENABLED", default=False)
VOXHELM_WHISPERKIT_HOST = os.getenv("VOXHELM_WHISPERKIT_HOST", "127.0.0.1").strip()
VOXHELM_WHISPERKIT_PORT = int(os.getenv("VOXHELM_WHISPERKIT_PORT", "50060"))
VOXHELM_WHISPERKIT_BASE_URL = os.getenv(
    "VOXHELM_WHISPERKIT_BASE_URL",
    f"http://127.0.0.1:{VOXHELM_WHISPERKIT_PORT}/v1",
).strip()
VOXHELM_WHISPERKIT_MODEL = os.getenv("VOXHELM_WHISPERKIT_MODEL", "large-v3-v20240930").strip()
VOXHELM_WHISPERKIT_AUDIO_ENCODER_COMPUTE_UNITS = os.getenv(
    "VOXHELM_WHISPERKIT_AUDIO_ENCODER_COMPUTE_UNITS",
    "cpuAndGPU",
).strip()
VOXHELM_WHISPERKIT_TEXT_DECODER_COMPUTE_UNITS = os.getenv(
    "VOXHELM_WHISPERKIT_TEXT_DECODER_COMPUTE_UNITS",
    "cpuAndGPU",
).strip()
VOXHELM_WHISPERKIT_CONCURRENT_WORKER_COUNT = int(
    os.getenv("VOXHELM_WHISPERKIT_CONCURRENT_WORKER_COUNT", "8")
)
VOXHELM_WHISPERKIT_CHUNKING_STRATEGY = os.getenv(
    "VOXHELM_WHISPERKIT_CHUNKING_STRATEGY",
    "vad",
).strip()
VOXHELM_WHISPERKIT_TIMEOUT_SECONDS = int(
    os.getenv("VOXHELM_WHISPERKIT_TIMEOUT_SECONDS", "900")
)
VOXHELM_STT_DEBUG_LOGGING = env_bool("VOXHELM_STT_DEBUG_LOGGING", default=False)
VOXHELM_MODEL_CACHE_DIR = Path(
    os.getenv("VOXHELM_MODEL_CACHE_DIR", str(BASE_DIR / "var" / "models"))
)
VOXHELM_WYOMING_STT_HOST = os.getenv("VOXHELM_WYOMING_STT_HOST", "0.0.0.0").strip()
VOXHELM_WYOMING_STT_PORT = int(os.getenv("VOXHELM_WYOMING_STT_PORT", "10300"))
VOXHELM_WYOMING_STT_BACKEND = os.getenv("VOXHELM_WYOMING_STT_BACKEND", "").strip()
VOXHELM_WYOMING_STT_MODEL = os.getenv("VOXHELM_WYOMING_STT_MODEL", "").strip()
VOXHELM_WYOMING_STT_LANGUAGE = os.getenv("VOXHELM_WYOMING_STT_LANGUAGE", "").strip()
VOXHELM_WYOMING_STT_LANGUAGES = env_list("VOXHELM_WYOMING_STT_LANGUAGES")
VOXHELM_WYOMING_STT_PROMPT = os.getenv("VOXHELM_WYOMING_STT_PROMPT", "").strip()
VOXHELM_WYOMING_STT_NORMALIZE_TRANSCRIPT = env_bool(
    "VOXHELM_WYOMING_STT_NORMALIZE_TRANSCRIPT",
    default=True,
)
VOXHELM_WYOMING_SAMPLES_PER_CHUNK = int(
    os.getenv("VOXHELM_WYOMING_SAMPLES_PER_CHUNK", "1024")
)
VOXHELM_LANE_SCHEDULER_ENABLED = env_bool("VOXHELM_LANE_SCHEDULER_ENABLED", default=False)
VOXHELM_LANE_SCHEDULER_DIR = Path(
    os.getenv("VOXHELM_LANE_SCHEDULER_DIR", str(BASE_DIR / "var" / "lane-scheduler"))
)
VOXHELM_LANE_SCHEDULER_STALE_SECONDS = int(
    os.getenv("VOXHELM_LANE_SCHEDULER_STALE_SECONDS", "1800")
)
VOXHELM_MAX_UPLOAD_BYTES = int(os.getenv("VOXHELM_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
VOXHELM_MAX_UPLOAD_MIB = VOXHELM_MAX_UPLOAD_BYTES // (1024 * 1024)
VOXHELM_MAX_URL_DOWNLOAD_BYTES = int(
    os.getenv("VOXHELM_MAX_URL_DOWNLOAD_BYTES", str(128 * 1024 * 1024))
)
VOXHELM_URL_FETCH_TIMEOUT_SECONDS = int(os.getenv("VOXHELM_URL_FETCH_TIMEOUT_SECONDS", "60"))
VOXHELM_BATCH_MAX_DOWNLOAD_BYTES = int(
    os.getenv("VOXHELM_BATCH_MAX_DOWNLOAD_BYTES", str(512 * 1024 * 1024))
)
VOXHELM_ALLOWED_URL_HOSTS = set(env_list("VOXHELM_ALLOWED_URL_HOSTS"))
VOXHELM_TRUSTED_HTTP_HOSTS = set(env_list("VOXHELM_TRUSTED_HTTP_HOSTS"))
VOXHELM_ACCEPTED_MODELS = get_accepted_stt_models()
VOXHELM_BATCH_ACCEPTED_MODELS = get_batch_accepted_stt_models()
VOXHELM_TTS_BACKEND = os.getenv("VOXHELM_TTS_BACKEND", "piper").strip()
VOXHELM_PIPER_VOICE_DIR = Path(
    os.getenv("VOXHELM_PIPER_VOICE_DIR", str(BASE_DIR / "var" / "piper"))
)
VOXHELM_PIPER_VOICES = env_list("VOXHELM_PIPER_VOICES")
VOXHELM_PIPER_DEFAULT_VOICE = os.getenv("VOXHELM_PIPER_DEFAULT_VOICE", "").strip()
VOXHELM_PIPER_LANGUAGE_VOICES = env_map("VOXHELM_PIPER_LANGUAGE_VOICES")
VOXHELM_TTS_MAX_INPUT_CHARS = int(os.getenv("VOXHELM_TTS_MAX_INPUT_CHARS", "5000"))
VOXHELM_ACCEPTED_SPEECH_MODELS = {
    "auto",
    "piper",
    "tts-1",
    "tts-1-hd",
}
VOXHELM_TASK_QUEUE = os.getenv("VOXHELM_TASK_QUEUE", "default")
VOXHELM_FFMPEG_BIN = os.getenv("VOXHELM_FFMPEG_BIN", "ffmpeg")

VOXHELM_ARTIFACT_BACKEND = os.getenv("VOXHELM_ARTIFACT_BACKEND", "filesystem")
VOXHELM_ARTIFACT_ROOT = Path(
    os.getenv("VOXHELM_ARTIFACT_ROOT", str(BASE_DIR / "var" / "artifacts"))
)
VOXHELM_ARTIFACT_BUCKET = os.getenv("VOXHELM_ARTIFACT_BUCKET", "voxhelm")
VOXHELM_ARTIFACT_PREFIX = os.getenv("VOXHELM_ARTIFACT_PREFIX", "voxhelm")
VOXHELM_ARTIFACT_S3_ENDPOINT_URL = os.getenv("VOXHELM_ARTIFACT_S3_ENDPOINT_URL", "").strip()
VOXHELM_ARTIFACT_S3_REGION = os.getenv("VOXHELM_ARTIFACT_S3_REGION", "us-east-1")
VOXHELM_ARTIFACT_S3_ACCESS_KEY_ID = os.getenv("VOXHELM_ARTIFACT_S3_ACCESS_KEY_ID", "").strip()
VOXHELM_ARTIFACT_S3_SECRET_ACCESS_KEY = os.getenv(
    "VOXHELM_ARTIFACT_S3_SECRET_ACCESS_KEY", ""
).strip()
VOXHELM_ARTIFACT_S3_FORCE_PATH_STYLE = env_bool(
    "VOXHELM_ARTIFACT_S3_FORCE_PATH_STYLE",
    default=True,
)

TASKS = {
    "default": {
        "BACKEND": os.getenv("VOXHELM_TASKS_BACKEND", "django_tasks_db.backend.DatabaseBackend")
    }
}
