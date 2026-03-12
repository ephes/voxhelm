from __future__ import annotations

import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def env_list(name: str, *, default: str = "") -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.replace("\n", ",").split(",") if item.strip()]


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
ROOT_URLCONF = "voxhelm_project.urls"
WSGI_APPLICATION = "voxhelm_project.wsgi.application"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

INSTALLED_APPS = [
    "voxhelm_api",
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
VOXHELM_MLX_MODEL = os.getenv("VOXHELM_MLX_MODEL", "mlx-community/whisper-large-v3-turbo")
VOXHELM_MAX_UPLOAD_BYTES = int(os.getenv("VOXHELM_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
VOXHELM_MAX_UPLOAD_MIB = VOXHELM_MAX_UPLOAD_BYTES // (1024 * 1024)
VOXHELM_MAX_URL_DOWNLOAD_BYTES = int(
    os.getenv("VOXHELM_MAX_URL_DOWNLOAD_BYTES", str(128 * 1024 * 1024))
)
VOXHELM_URL_FETCH_TIMEOUT_SECONDS = int(os.getenv("VOXHELM_URL_FETCH_TIMEOUT_SECONDS", "60"))
VOXHELM_ALLOWED_URL_HOSTS = set(env_list("VOXHELM_ALLOWED_URL_HOSTS"))
VOXHELM_TRUSTED_HTTP_HOSTS = set(env_list("VOXHELM_TRUSTED_HTTP_HOSTS"))
VOXHELM_ACCEPTED_MODELS = {
    "gpt-4o-mini-transcribe",
    "whisper-1",
    VOXHELM_MLX_MODEL,
}

