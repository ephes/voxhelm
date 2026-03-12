# Voxhelm

Voxhelm is the shared local media-processing service for homelab consumers.

Milestone 1a provides a synchronous, OpenAI-compatible transcription API for
Archive:

- `GET /v1/health`
- `POST /v1/audio/transcriptions`

## Local Development

```bash
uv sync
just test
uv run uvicorn config.asgi:application
```

## Required Environment

```bash
export DJANGO_SECRET_KEY="replace-me"
export VOXHELM_BEARER_TOKENS="archive=replace-me"
```

Optional settings:

```bash
export VOXHELM_ALLOWED_HOSTS="localhost,127.0.0.1"
export VOXHELM_STT_BACKEND="whispercpp"
export VOXHELM_STT_FALLBACK_BACKEND="mlx"
export VOXHELM_MLX_MODEL="mlx-community/whisper-large-v3-mlx"
export VOXHELM_WHISPERCPP_MODEL="ggml-large-v3.bin"
export VOXHELM_WHISPERCPP_BIN="/opt/homebrew/bin/whisper-cli"
export VOXHELM_WHISPERCPP_PROCESSORS="4"
export VOXHELM_MODEL_CACHE_DIR="$PWD/var/models"
export VOXHELM_WYOMING_STT_HOST="0.0.0.0"
export VOXHELM_WYOMING_STT_PORT="10300"
export VOXHELM_WYOMING_STT_BACKEND=""
export VOXHELM_WYOMING_STT_MODEL=""
export VOXHELM_WYOMING_STT_LANGUAGE=""
export VOXHELM_WYOMING_STT_LANGUAGES="de,en"
export VOXHELM_ALLOWED_URL_HOSTS="media.example.com"
export VOXHELM_TRUSTED_HTTP_HOSTS="internal.example.lan"
```

## OpenAI-Compatible API

Multipart upload:

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer replace-me" \
  -F "file=@sample.mp3" \
  -F "model=gpt-4o-mini-transcribe"
```

JSON URL input:

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer replace-me" \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/sample.mp3","model":"whisper-1"}'
```

## Wyoming STT

Milestone 2 adds a separate Wyoming STT sidecar process for Home Assistant:

```bash
uv run voxhelm-wyoming-stt
```

The sidecar reuses Voxhelm's existing STT backend layer. If
`VOXHELM_WYOMING_STT_BACKEND` and `VOXHELM_WYOMING_STT_MODEL` are unset, it
falls back to the service-wide backend defaults.

Current limitation: there is no cross-process lane scheduler yet. The Wyoming
sidecar runs in its own process, but it can still contend with the HTTP API and
batch worker for CPU, memory, and model cache usage on `studio`.
