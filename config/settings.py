from __future__ import annotations

import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def env_list(name: str, *, default: str = "") -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.replace("\n", ",").split(",") if item.strip()]


def env_bool(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def env_tokens(name: str) -> dict[str, str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return {}
    if raw.startswith("{"):
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError(f"{name} must be a JSON object when JSON syntax is used.")
        return {str(key): str(value) for key, value in parsed.items()}

    tokens: dict[str, str] = {}
    for entry in raw.replace("\n", ",").split(","):
        normalized = entry.strip()
        if not normalized:
            continue
        if "=" not in normalized:
            raise ValueError(f"Invalid {name} entry '{normalized}'. Use label=token pairs.")
        label, token = normalized.split("=", 1)
        tokens[label.strip()] = token.strip()
    return tokens


SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "dev-only-secret-key")
DEBUG = os.getenv("DJANGO_DEBUG", "").lower() in {"1", "true", "yes", "on"}
ALLOWED_HOSTS = env_list("VOXHELM_ALLOWED_HOSTS", default="localhost,127.0.0.1")
ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

INSTALLED_APPS = [
    "django_tasks",
    "django_tasks_db",
    "transcriptions",
    "jobs",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
]

TEMPLATES: list[dict[str, object]] = []

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

VOXHELM_BEARER_TOKENS = env_tokens("VOXHELM_BEARER_TOKENS")
VOXHELM_STT_BACKEND = os.getenv("VOXHELM_STT_BACKEND", "whispercpp").strip()
VOXHELM_STT_FALLBACK_BACKEND = os.getenv("VOXHELM_STT_FALLBACK_BACKEND", "mlx").strip()
VOXHELM_MLX_MODEL = os.getenv("VOXHELM_MLX_MODEL", "mlx-community/whisper-large-v3-mlx")
VOXHELM_WHISPERCPP_MODEL = os.getenv("VOXHELM_WHISPERCPP_MODEL", "ggml-large-v3.bin").strip()
VOXHELM_WHISPERCPP_BIN = os.getenv("VOXHELM_WHISPERCPP_BIN", "/opt/homebrew/bin/whisper-cli")
VOXHELM_WHISPERCPP_PROCESSORS = int(os.getenv("VOXHELM_WHISPERCPP_PROCESSORS", "4"))
VOXHELM_MODEL_CACHE_DIR = Path(
    os.getenv("VOXHELM_MODEL_CACHE_DIR", str(BASE_DIR / "var" / "models"))
)
VOXHELM_WYOMING_STT_HOST = os.getenv("VOXHELM_WYOMING_STT_HOST", "0.0.0.0").strip()
VOXHELM_WYOMING_STT_PORT = int(os.getenv("VOXHELM_WYOMING_STT_PORT", "10300"))
VOXHELM_WYOMING_STT_BACKEND = os.getenv("VOXHELM_WYOMING_STT_BACKEND", "").strip()
VOXHELM_WYOMING_STT_MODEL = os.getenv("VOXHELM_WYOMING_STT_MODEL", "").strip()
VOXHELM_WYOMING_STT_LANGUAGE = os.getenv("VOXHELM_WYOMING_STT_LANGUAGE", "").strip()
VOXHELM_WYOMING_STT_LANGUAGES = env_list("VOXHELM_WYOMING_STT_LANGUAGES")
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
VOXHELM_ACCEPTED_MODELS = {
    "gpt-4o-mini-transcribe",
    "whisper-1",
    VOXHELM_MLX_MODEL,
    VOXHELM_WHISPERCPP_MODEL,
}
VOXHELM_BATCH_ACCEPTED_MODELS = {
    "auto",
    *VOXHELM_ACCEPTED_MODELS,
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
