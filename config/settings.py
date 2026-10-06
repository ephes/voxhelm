from __future__ import annotations

import importlib.util
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


def env_kokoro_models(name: str) -> dict[str, dict[str, str]]:
    """Parse the Kokoro voice map from a compact ``env_map``-style encoding.

    Each comma/newline-separated entry is ``voice_key=model:voicepack:key:lang``
    where the four colon-delimited fields are the ONNX model file, the voicepack
    file, the embedding key inside that pack (e.g. ``af_heart``/``martin``), and
    the espeak phoneme language (e.g. ``en-us``/``de``). Relative file paths are
    resolved against ``VOXHELM_KOKORO_MODEL_DIR`` by the backend.
    """
    models: dict[str, dict[str, str]] = {}
    for voice_key, value in env_map(name).items():
        fields = [field.strip() for field in value.split(":")]
        if len(fields) != 4 or not all(fields):
            raise ValueError(
                f"Invalid {name} entry for '{voice_key}'. Use "
                "voice_key=model:voicepack:voicepack_key:language."
            )
        model, voicepack, voicepack_key, language = fields
        models[voice_key] = {
            "model": model,
            "voicepack": voicepack,
            "voicepack_key": voicepack_key,
            "language": language,
        }
    return models


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
            {str(key).strip(): str(value).strip() for key, value in parsed.items()},
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


def validate_unique_token_values(name: str, tokens: dict[str, str]) -> dict[str, str]:
    seen: dict[str, str] = {}
    duplicates: list[str] = []
    for label, token in tokens.items():
        if token in seen:
            duplicates.append(f"{seen[token]} and {label}")
        seen[token] = label
    if duplicates:
        joined = ", ".join(duplicates)
        raise ValueError(f"{name} maps one bearer token to multiple labels: {joined}.")
    return tokens


def validate_disjoint_token_values(
    left_name: str,
    left_tokens: dict[str, str],
    right_name: str,
    right_tokens: dict[str, str],
) -> dict[str, str]:
    left_labels_by_token = {token: label for label, token in left_tokens.items()}
    overlaps = [
        f"{left_labels_by_token[token]} and {label}"
        for label, token in right_tokens.items()
        if token in left_labels_by_token
    ]
    if overlaps:
        joined = ", ".join(overlaps)
        raise ValueError(
            f"{left_name} and {right_name} must not share bearer token values: {joined}."
        )
    return right_tokens


def validate_bearer_token_labels(name: str, tokens: dict[str, str]) -> dict[str, str]:
    empty_labels = [label for label in tokens if not label]
    if empty_labels:
        raise ValueError(f"{name} contains an empty label.")
    empty_token_labels = sorted(label for label, token in tokens.items() if not token)
    if empty_token_labels:
        labels = ", ".join(empty_token_labels)
        raise ValueError(f"{name} contains empty token value(s) for label(s): {labels}.")
    reserved = sorted(VOXHELM_RESERVED_BEARER_TOKEN_LABELS.intersection(tokens))
    if reserved:
        labels = ", ".join(reserved)
        raise ValueError(f"{name} contains reserved label(s): {labels}.")
    return tokens


def validate_positive_int(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def validate_non_negative_int(name: str, value: int) -> int:
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


def validate_language_routing_dependencies(enabled: bool) -> None:
    """Fail fast at startup when routing is enabled without the ``routing`` extra.

    Automatic language routing needs ``lingua-language-detector``, shipped by the
    optional ``routing`` extra. Enabling ``VOXHELM_TTS_LANGUAGE_ROUTING`` without
    it installed is a deployment misconfiguration, so it is surfaced here at
    settings-import time (service startup) rather than letting every request
    silently skip routing. This startup check is the one place routing may raise;
    at request time routing never raises for routing-related reasons.
    """
    if not enabled:
        return
    if importlib.util.find_spec("lingua") is None:
        raise ValueError(
            "VOXHELM_TTS_LANGUAGE_ROUTING is enabled but the 'routing' extra is not "
            "installed. Install it with `uv sync --extra routing` "
            "(lingua-language-detector) or disable language routing."
        )


REMOTE_PULL_SHARED_ARTIFACT_BACKENDS = {"s3"}
TRANSCRIPTION_EXECUTION_MODES = {"django_tasks", "remote_pull"}


def validate_transcription_execution_mode(mode: str) -> str:
    if mode not in TRANSCRIPTION_EXECUTION_MODES:
        accepted = ", ".join(sorted(TRANSCRIPTION_EXECUTION_MODES))
        raise ValueError(f"VOXHELM_TRANSCRIPTION_EXECUTION_MODE must be one of: {accepted}.")
    return mode


def validate_remote_pull_worker_tokens(
    execution_mode: str,
    worker_tokens: dict[str, str],
) -> None:
    if execution_mode == "remote_pull" and not worker_tokens:
        raise ValueError(
            "VOXHELM_TRANSCRIPTION_EXECUTION_MODE=remote_pull requires "
            "VOXHELM_WORKER_TOKENS to configure at least one worker token."
        )


def validate_remote_pull_artifact_backend(execution_mode: str, artifact_backend: str) -> None:
    if execution_mode != "remote_pull":
        return
    if artifact_backend in REMOTE_PULL_SHARED_ARTIFACT_BACKENDS:
        return
    accepted = ", ".join(sorted(REMOTE_PULL_SHARED_ARTIFACT_BACKENDS))
    raise ValueError(
        "VOXHELM_TRANSCRIPTION_EXECUTION_MODE=remote_pull requires "
        f"VOXHELM_ARTIFACT_BACKEND to be one of: {accepted}."
    )


def validate_remote_pull_s3_configuration(
    execution_mode: str,
    artifact_backend: str,
    values: dict[str, str],
) -> None:
    if execution_mode != "remote_pull" or artifact_backend != "s3":
        return
    missing = sorted(name for name, value in values.items() if not value)
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            "VOXHELM_TRANSCRIPTION_EXECUTION_MODE=remote_pull requires complete "
            f"S3 artifact configuration: {joined}."
        )


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
CSRF_TRUSTED_ORIGINS = env_list("VOXHELM_CSRF_TRUSTED_ORIGINS")
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
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
VOXHELM_WORKER_TOKENS = validate_disjoint_token_values(
    "VOXHELM_BEARER_TOKENS",
    VOXHELM_BEARER_TOKENS,
    "VOXHELM_WORKER_TOKENS",
    validate_unique_token_values(
        "VOXHELM_WORKER_TOKENS",
        env_tokens("VOXHELM_WORKER_TOKENS"),
    ),
)
VOXHELM_STT_BACKEND = os.getenv("VOXHELM_STT_BACKEND", "whispercpp").strip()
VOXHELM_STT_FALLBACK_BACKEND = os.getenv("VOXHELM_STT_FALLBACK_BACKEND", "mlx").strip()
VOXHELM_MLX_MODEL = os.getenv("VOXHELM_MLX_MODEL", "mlx-community/whisper-large-v3-mlx")
# Conditioning the decoder on previously generated text is Whisper's main cause of
# runaway repetition loops on long audio (a hallucinated phrase feeds itself across
# 30s windows). Default off; flip to true to restore upstream Whisper behaviour.
VOXHELM_MLX_CONDITION_ON_PREVIOUS_TEXT = env_bool(
    "VOXHELM_MLX_CONDITION_ON_PREVIOUS_TEXT", default=False
)
VOXHELM_WHISPERCPP_MODEL = os.getenv("VOXHELM_WHISPERCPP_MODEL", "ggml-large-v3.bin").strip()
VOXHELM_WHISPERCPP_BIN = os.getenv("VOXHELM_WHISPERCPP_BIN", "/opt/homebrew/bin/whisper-cli")
VOXHELM_WHISPERCPP_PROCESSORS = int(os.getenv("VOXHELM_WHISPERCPP_PROCESSORS", "4"))
# whisper.cpp anti-hallucination controls. max-context 0 disables conditioning on
# previous text (the whisper.cpp equivalent of condition_on_previous_text=False);
# -1 restores the upstream unbounded default. suppress-nst drops non-speech tokens,
# which curbs hallucinated text over music/silence (e.g. outro-music loops).
VOXHELM_WHISPERCPP_MAX_CONTEXT = int(os.getenv("VOXHELM_WHISPERCPP_MAX_CONTEXT", "0"))
VOXHELM_WHISPERCPP_SUPPRESS_NST = env_bool("VOXHELM_WHISPERCPP_SUPPRESS_NST", default=True)
# Post-decode transcript sanitizer. A deterministic backstop that runs after
# every backend transcription (before format rendering) to collapse repeated-
# sentence loops and drop subtitle-credit / punctuation-only hallucinations the
# decode-level guards above reduce but cannot fully eliminate. Conservative by
# design; disable only for debugging raw decoder output.
VOXHELM_SANITIZE_TRANSCRIPT = env_bool("VOXHELM_SANITIZE_TRANSCRIPT", default=True)
# Minimum consecutive identical (normalized) segments before a run is collapsed
# to a single instance. 4 catches real loops (9-84x) without touching natural
# backchannels or rhetorical repetition.
VOXHELM_SANITIZE_REPEAT_THRESHOLD = validate_positive_int(
    "VOXHELM_SANITIZE_REPEAT_THRESHOLD",
    int(os.getenv("VOXHELM_SANITIZE_REPEAT_THRESHOLD", "4")),
)
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
VOXHELM_DIARIZATION_BACKEND = os.getenv("VOXHELM_DIARIZATION_BACKEND", "none").strip()
VOXHELM_PYANNOTE_MODEL = os.getenv(
    "VOXHELM_PYANNOTE_MODEL",
    "pyannote/speaker-diarization-3.1",
).strip()
VOXHELM_PYANNOTE_DEVICE = os.getenv("VOXHELM_PYANNOTE_DEVICE", "auto").strip()
VOXHELM_HUGGINGFACE_TOKEN = os.getenv(
    "VOXHELM_HUGGINGFACE_TOKEN",
    os.getenv("HF_TOKEN", ""),
).strip()
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
VOXHELM_LANE_SCHEDULER_INTERACTIVE_SLOTS = validate_non_negative_int(
    "VOXHELM_LANE_SCHEDULER_INTERACTIVE_SLOTS",
    int(os.getenv("VOXHELM_LANE_SCHEDULER_INTERACTIVE_SLOTS", "1")),
)
VOXHELM_LANE_SCHEDULER_NON_INTERACTIVE_SLOTS = validate_positive_int(
    "VOXHELM_LANE_SCHEDULER_NON_INTERACTIVE_SLOTS",
    int(os.getenv("VOXHELM_LANE_SCHEDULER_NON_INTERACTIVE_SLOTS", "1")),
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
VOXHELM_BATCH_MAX_STAGED_UPLOAD_BYTES = int(
    os.getenv("VOXHELM_BATCH_MAX_STAGED_UPLOAD_BYTES", str(512 * 1024 * 1024))
)
VOXHELM_BATCH_MAX_STAGED_UPLOAD_MIB = VOXHELM_BATCH_MAX_STAGED_UPLOAD_BYTES // (1024 * 1024)
VOXHELM_STAGED_INPUT_RETENTION_SECONDS = int(
    os.getenv("VOXHELM_STAGED_INPUT_RETENTION_SECONDS", str(24 * 60 * 60))
)
# D-09: non-exposed job source media is kept this long after the job finished and
# then removed by `manage.py prune_job_artifacts`; extracted audio goes once the job
# is terminal. Transcript and speech artifacts are never pruned.
VOXHELM_SOURCE_ARTIFACT_RETENTION_SECONDS = validate_non_negative_int(
    "VOXHELM_SOURCE_ARTIFACT_RETENTION_SECONDS",
    int(os.getenv("VOXHELM_SOURCE_ARTIFACT_RETENTION_SECONDS", str(24 * 60 * 60))),
)
# D-09: terminal job rows are deleted by `manage.py prune_job_artifacts` this long after the
# job finished (default 90 days), but only once the job owns no artifact row (transcript and
# speech artifacts keep their job indefinitely), queued deletion or claimed staged upload.
# 0 disables job metadata pruning.
VOXHELM_JOB_METADATA_RETENTION_SECONDS = validate_non_negative_int(
    "VOXHELM_JOB_METADATA_RETENTION_SECONDS",
    int(os.getenv("VOXHELM_JOB_METADATA_RETENTION_SECONDS", str(90 * 24 * 60 * 60))),
)
VOXHELM_ALLOWED_URL_HOSTS = set(env_list("VOXHELM_ALLOWED_URL_HOSTS"))
VOXHELM_TRUSTED_HTTP_HOSTS = set(env_list("VOXHELM_TRUSTED_HTTP_HOSTS"))
# Allowlisted hosts that may resolve to non-public IPs (RFC 1918, loopback,
# CGNAT/Tailscale, IPv6 ULA). Trusted HTTP hosts are implicitly included.
# Link-local/metadata, multicast and reserved addresses are always refused.
VOXHELM_PRIVATE_URL_HOSTS = set(env_list("VOXHELM_PRIVATE_URL_HOSTS"))
VOXHELM_ACCEPTED_MODELS = get_accepted_stt_models()
VOXHELM_BATCH_ACCEPTED_MODELS = get_batch_accepted_stt_models()
VOXHELM_TTS_BACKEND = os.getenv("VOXHELM_TTS_BACKEND", "piper").strip()
VOXHELM_PIPER_VOICE_DIR = Path(
    os.getenv("VOXHELM_PIPER_VOICE_DIR", str(BASE_DIR / "var" / "piper"))
)
VOXHELM_PIPER_VOICES = env_list("VOXHELM_PIPER_VOICES")
VOXHELM_PIPER_DEFAULT_VOICE = os.getenv("VOXHELM_PIPER_DEFAULT_VOICE", "").strip()
VOXHELM_PIPER_LANGUAGE_VOICES = env_map("VOXHELM_PIPER_LANGUAGE_VOICES")
# Kokoro ONNX TTS backend (optional; requires the `kokoro` extra + espeak-ng).
# VOXHELM_KOKORO_MODELS maps registry voice keys (convention: kokoro-<voicepack_key>)
# to their artifacts, e.g.:
#   kokoro-af_heart=kokoro-v1.0.onnx:voices-v1.0.bin:af_heart:en-us,
#   kokoro-martin=kokoro-martin.onnx:voices-martin.npz:martin:de
# Relative model/voicepack paths resolve against VOXHELM_KOKORO_MODEL_DIR. With
# no models configured the registry has no kokoro voices and Piper is untouched.
VOXHELM_KOKORO_MODEL_DIR = Path(
    os.getenv("VOXHELM_KOKORO_MODEL_DIR", str(BASE_DIR / "var" / "kokoro"))
)
VOXHELM_KOKORO_MODELS = env_kokoro_models("VOXHELM_KOKORO_MODELS")
# Default Kokoro voice used when a request reaches the Kokoro backend without an
# explicit voice. When unset the backend falls back to the first configured
# VOXHELM_KOKORO_MODELS entry in sorted-key order (mirrors VOXHELM_PIPER_DEFAULT_VOICE).
VOXHELM_KOKORO_DEFAULT_VOICE = os.getenv("VOXHELM_KOKORO_DEFAULT_VOICE", "").strip()
# Optional override for the libespeak-ng shared library (else espeakng-loader's
# bundled library is used).
VOXHELM_ESPEAK_LIBRARY = os.getenv("VOXHELM_ESPEAK_LIBRARY", "").strip()
# Automatic language routing (optional; requires the `routing` extra:
# lingua-language-detector). When enabled, outgoing TTS text is language-detected
# in synthesize_text (so Wyoming, HTTP, and batch all benefit) and, when the
# detection clears the routing floor, the pinned voice is replaced by the detected
# language's mapped voice and the detected language becomes effective end-to-end.
# Default off, in which case synthesis behavior is byte-for-byte unchanged.
# VOXHELM_TTS_LANGUAGE_VOICES maps a language code to a registry voice key across
# all backends (env_map style), e.g.:
#   VOXHELM_TTS_LANGUAGE_VOICES="de=kokoro-martin,en=kokoro-af_heart"
# (The Piper-only VOXHELM_PIPER_LANGUAGE_VOICES is separate and unchanged.)
VOXHELM_TTS_LANGUAGE_ROUTING = env_bool("VOXHELM_TTS_LANGUAGE_ROUTING", default=False)
VOXHELM_TTS_LANGUAGE_VOICES = env_map("VOXHELM_TTS_LANGUAGE_VOICES")
validate_language_routing_dependencies(VOXHELM_TTS_LANGUAGE_ROUTING)
VOXHELM_TTS_MAX_INPUT_CHARS = int(os.getenv("VOXHELM_TTS_MAX_INPUT_CHARS", "5000"))
# Accepted `model` field values for /v1/audio/speech. "auto"/"piper"/"tts-1"/
# "tts-1-hd" resolve to the default backend (see AUTO_BACKEND_MODEL_NAMES);
# "kokoro" explicitly forces the Kokoro backend for that request.
VOXHELM_ACCEPTED_SPEECH_MODELS = {
    "auto",
    "piper",
    "kokoro",
    "tts-1",
    "tts-1-hd",
}
VOXHELM_TASK_QUEUE = os.getenv("VOXHELM_TASK_QUEUE", "default")
VOXHELM_FFMPEG_BIN = os.getenv("VOXHELM_FFMPEG_BIN", "ffmpeg")
VOXHELM_ARTIFACT_BACKEND = os.getenv("VOXHELM_ARTIFACT_BACKEND", "filesystem").strip()
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
VOXHELM_TRANSCRIPTION_EXECUTION_MODE = validate_transcription_execution_mode(
    os.getenv(
        "VOXHELM_TRANSCRIPTION_EXECUTION_MODE",
        "django_tasks",
    ).strip()
)
VOXHELM_REMOTE_WORKER_LEASE_SECONDS = validate_positive_int(
    "VOXHELM_REMOTE_WORKER_LEASE_SECONDS",
    int(os.getenv("VOXHELM_REMOTE_WORKER_LEASE_SECONDS", str(30 * 60))),
)
VOXHELM_REMOTE_WORKER_POLL_SECONDS = validate_positive_int(
    "VOXHELM_REMOTE_WORKER_POLL_SECONDS",
    int(os.getenv("VOXHELM_REMOTE_WORKER_POLL_SECONDS", "5")),
)
VOXHELM_REMOTE_WORKER_MAX_ATTEMPTS = validate_positive_int(
    "VOXHELM_REMOTE_WORKER_MAX_ATTEMPTS",
    int(os.getenv("VOXHELM_REMOTE_WORKER_MAX_ATTEMPTS", "3")),
)
# Fair-share claim balancing. When multiple remote workers are active, the
# control plane defers a worker that is "ahead" on recent claims so a fresh,
# idle, less-loaded peer can take the next job — yielding a roughly even split
# instead of the lowest-latency worker winning every race. Self-heals to a
# single worker (no peer => never defer).
VOXHELM_REMOTE_WORKER_BALANCE_ENABLED = env_bool(
    "VOXHELM_REMOTE_WORKER_BALANCE_ENABLED", default=True
)
# Rolling window (seconds) over which recent per-worker claim counts are compared.
VOXHELM_REMOTE_WORKER_BALANCE_WINDOW_SECONDS = validate_positive_int(
    "VOXHELM_REMOTE_WORKER_BALANCE_WINDOW_SECONDS",
    int(os.getenv("VOXHELM_REMOTE_WORKER_BALANCE_WINDOW_SECONDS", "3600")),
)
# A peer only counts as a deferral target if it heartbeated within this many
# seconds (i.e. it is really online and polling). Bounds worst-case claim delay.
VOXHELM_REMOTE_WORKER_BALANCE_PEER_FRESH_SECONDS = validate_positive_int(
    "VOXHELM_REMOTE_WORKER_BALANCE_PEER_FRESH_SECONDS",
    int(os.getenv("VOXHELM_REMOTE_WORKER_BALANCE_PEER_FRESH_SECONDS", "30")),
)
validate_remote_pull_worker_tokens(
    VOXHELM_TRANSCRIPTION_EXECUTION_MODE,
    VOXHELM_WORKER_TOKENS,
)
validate_remote_pull_artifact_backend(
    VOXHELM_TRANSCRIPTION_EXECUTION_MODE,
    VOXHELM_ARTIFACT_BACKEND,
)
validate_remote_pull_s3_configuration(
    VOXHELM_TRANSCRIPTION_EXECUTION_MODE,
    VOXHELM_ARTIFACT_BACKEND,
    {
        "VOXHELM_ARTIFACT_S3_ENDPOINT_URL": VOXHELM_ARTIFACT_S3_ENDPOINT_URL,
        "VOXHELM_ARTIFACT_S3_ACCESS_KEY_ID": VOXHELM_ARTIFACT_S3_ACCESS_KEY_ID,
        "VOXHELM_ARTIFACT_S3_SECRET_ACCESS_KEY": VOXHELM_ARTIFACT_S3_SECRET_ACCESS_KEY,
        "VOXHELM_ARTIFACT_BUCKET": VOXHELM_ARTIFACT_BUCKET,
    },
)

TASKS = {
    "default": {
        "BACKEND": os.getenv("VOXHELM_TASKS_BACKEND", "django_tasks_db.backend.DatabaseBackend")
    }
}
