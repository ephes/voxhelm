# Voxhelm

Voxhelm is the shared local media-processing service for homelab consumers.

Milestone 1a provides a synchronous, OpenAI-compatible transcription API for
Archive:

- `GET /v1/health`
- `POST /v1/audio/transcriptions`

The current slice also adds the first Voxhelm-owned operator UI:

- `/` browser login and operator transcript console
- sync routing for audio URLs and uploaded audio
- batch routing for video URLs
- transcript downloads for `text`, `json`, `vtt`, `dote`, and `podlove`
- staged batch uploads for oversized/private/local audio via `POST /v1/uploads`

`whisper.cpp` inputs are normalized through `ffmpeg` to 16 kHz mono PCM WAV before
inference so AAC/M4A and other container/codec quirks do not leak into the backend.

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
export VOXHELM_CSRF_TRUSTED_ORIGINS="https://voxhelm.example.com"
export VOXHELM_STT_BACKEND="whispercpp"
export VOXHELM_STT_FALLBACK_BACKEND="mlx"
export VOXHELM_MLX_MODEL="mlx-community/whisper-large-v3-mlx"
export VOXHELM_WHISPERCPP_MODEL="ggml-large-v3.bin"
export VOXHELM_WHISPERCPP_BIN="/opt/homebrew/bin/whisper-cli"
export VOXHELM_WHISPERCPP_PROCESSORS="4"
export VOXHELM_WHISPERKIT_ENABLED="false"
export VOXHELM_WHISPERKIT_HOST="127.0.0.1"
export VOXHELM_WHISPERKIT_PORT="50060"
export VOXHELM_WHISPERKIT_BASE_URL="http://127.0.0.1:50060/v1"
export VOXHELM_WHISPERKIT_MODEL="large-v3-v20240930"
export VOXHELM_WHISPERKIT_AUDIO_ENCODER_COMPUTE_UNITS="cpuAndGPU"
export VOXHELM_WHISPERKIT_TEXT_DECODER_COMPUTE_UNITS="cpuAndGPU"
export VOXHELM_WHISPERKIT_CONCURRENT_WORKER_COUNT="8"
export VOXHELM_WHISPERKIT_CHUNKING_STRATEGY="vad"
export VOXHELM_WHISPERKIT_TIMEOUT_SECONDS="900"
export VOXHELM_STT_DEBUG_LOGGING="false"
export VOXHELM_MODEL_CACHE_DIR="$PWD/var/models"
export VOXHELM_WYOMING_STT_HOST="0.0.0.0"
export VOXHELM_WYOMING_STT_PORT="10300"
export VOXHELM_WYOMING_STT_BACKEND="mlx"
export VOXHELM_WYOMING_STT_MODEL=""
export VOXHELM_WYOMING_STT_LANGUAGE=""
export VOXHELM_WYOMING_STT_LANGUAGES="de,en"
export VOXHELM_WYOMING_STT_PROMPT=""
export VOXHELM_ALLOWED_URL_HOSTS="media.example.com"
export VOXHELM_TRUSTED_HTTP_HOSTS="internal.example.lan"
export VOXHELM_BATCH_MAX_STAGED_UPLOAD_BYTES="536870912"
export VOXHELM_STAGED_INPUT_RETENTION_SECONDS="86400"
export VOXHELM_BOOTSTRAP_OPERATOR_USERNAME="jochen"
export VOXHELM_BOOTSTRAP_OPERATOR_EMAIL=""
export VOXHELM_BOOTSTRAP_OPERATOR_PASSWORD="replace-me"
```

Bootstrap the initial operator account after migrations:

```bash
uv run python manage.py bootstrap_operator --username jochen --password "replace-me"
```

Deploy-time note: the deployment layer should call the same in-app command with the real secret rather than creating the operator directly in a separate repo.

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

## Batch Large-Input Contract

Stage oversized/private/local audio into Voxhelm first:

```bash
curl -X POST http://127.0.0.1:8000/v1/uploads \
  -H "Authorization: Bearer replace-me" \
  -F "file=@large-private-episode.mp3"
```

Then submit the existing batch job with `input.kind=upload`:

```bash
curl -X POST http://127.0.0.1:8000/v1/jobs \
  -H "Authorization: Bearer replace-me" \
  -H "Content-Type: application/json" \
  -d '{
    "job_type": "transcribe",
    "priority": "normal",
    "lane": "batch",
    "backend": "auto",
    "model": "auto",
    "input": {"kind": "upload", "upload_id": "replace-me"},
    "output": {"formats": ["text", "json"]},
    "task_ref": "archive-item-123"
  }'
```

Staged uploads are stored in Voxhelm's configured artifact backend before worker
execution. The worker copies staged input into the normal job-owned source
artifact, then deletes the temporary staged object immediately after
materialization. Unclaimed staged uploads expire after
`VOXHELM_STAGED_INPUT_RETENTION_SECONDS` and are opportunistically cleaned on
later staging/submission requests.

Current scope note: batch staged uploads are audio-only in this slice. URL
audio and URL video keep working on the existing path. Uploaded video and true
service-owned chunk splitting/stitching are still explicitly deferred.

## Wyoming STT

Milestone 2 adds a separate Wyoming STT sidecar process for Home Assistant:

```bash
uv run voxhelm-wyoming-stt
```

The sidecar reuses Voxhelm's existing STT backend layer. If
`VOXHELM_WYOMING_STT_MODEL` is unset, the sidecar uses the default model for
the configured Wyoming backend. The recommended interactive default is
`VOXHELM_WYOMING_STT_BACKEND=mlx`, which avoids the short-command silence
hallucinations seen with the current `whisper.cpp` setup on `studio`.

Set `VOXHELM_STT_DEBUG_LOGGING=true` when tuning the HA path. Voxhelm will emit
one structured `stt_debug` log line per transcription with the input audio
shape, requested and resolved backend/model/language, transcript preview, and
latency.

## Experimental WhisperKit Backend

WhisperKit is now available as an experimental STT backend, but it is still
non-default. Enable it explicitly with `VOXHELM_WHISPERKIT_ENABLED=true`, run a
local `whisperkit-cli serve` instance, and request either the explicit
`whisperkit` model alias or the configured WhisperKit model name. `whisper-1`,
`gpt-4o-mini-transcribe`, `auto`, and the deployed default still resolve to
`whisper.cpp` unless you intentionally reconfigure the backend.

The intended `studio` shape is the local server mode rather than a direct CLI
wrapper. The tuned sidecar settings currently map to:

```bash
whisperkit-cli serve \
  --host 127.0.0.1 \
  --port 50060 \
  --model large-v3-v20240930 \
  --audio-encoder-compute-units cpuAndGPU \
  --text-decoder-compute-units cpuAndGPU \
  --concurrent-worker-count 8 \
  --chunking-strategy vad
```

Operational caveat: keep treating WhisperKit as experimental on `studio`. The
benchmark follow-on kept it competitive, but the tuned long-form run still
logged a Metal GPU recovery error, so the deployed default remains
`whispercpp`.

Current limitation: the first C13 lane scheduler slice is cross-process and
does gate Voxhelm's HTTP, batch, and Wyoming entry points, but it does not
reach inside the WhisperKit sidecar itself. Once Voxhelm has admitted a
WhisperKit request, the sidecar's internal inference concurrency remains
outside that scheduler's direct control.
