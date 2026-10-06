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

## Transcript Sanitization

Whisper occasionally emits artifacts that the decode-level guards
(`condition_on_previous_text=False`, `--max-context 0`, `--suppress-nst`) reduce
but cannot fully eliminate. Voxhelm applies a deterministic post-decode
sanitizer to every produced `TranscriptionResult` once at the segment level —
before any format is rendered — so `text`, `json`, `vtt`, `dote`, and `podlove`
all stay consistent regardless of which backend ran. It applies on both the
local transcription path and the `remote_pull` worker path, ahead of speaker
diarization in each. It removes two artifact classes:

- **Repeated-sentence loops** — a run of consecutive segments whose text is
  identical after light normalization (casefold, collapsed whitespace, stripped
  surrounding punctuation) is collapsed to a single segment. The first segment's
  start is kept and its end extended to the run's last end. The run must reach
  `VOXHELM_SANITIZE_REPEAT_THRESHOLD` (default `4`) consecutive repeats, which
  catches the real loops (9–84×) while leaving natural backchannels and
  rhetorical repetition inside a single cue untouched.
- **Non-speech / credit hallucinations** — subtitle-credit cues
  (`Untertitelung des ZDF, 2020`, `Untertitel im Auftrag des ZDF für funk, 2017`,
  `Untertitel von Amara.org`, and the related `Untertitel…`/`Amara.org` family)
  and punctuation-only noise such as long dot-runs are dropped. A long dot-run
  embedded in an otherwise-real segment is stripped while the real text and
  adjacent genuine segments survive.

The sanitizer is conservative by design: it biases toward false negatives over
removing genuine speech, only ever removes or collapses artifact segments, and
returns clean transcripts unchanged. Disable it with
`VOXHELM_SANITIZE_TRANSCRIPT=false` only to inspect raw decoder output.

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
# Anti-hallucination decoding (defaults shown). Conditioning the Whisper decoder on
# previously generated text is the main cause of runaway repetition loops on long
# audio, so it is disabled by default. To restore upstream Whisper behaviour set the
# mlx flag to "true" and whisper.cpp max-context to "-1".
export VOXHELM_MLX_CONDITION_ON_PREVIOUS_TEXT="false"
export VOXHELM_WHISPERCPP_MODEL="ggml-large-v3.bin"
export VOXHELM_WHISPERCPP_BIN="/opt/homebrew/bin/whisper-cli"
export VOXHELM_WHISPERCPP_PROCESSORS="4"
# max-context 0 disables conditioning on previous text (the loop trigger); -1 = upstream
# default. suppress-nst drops non-speech tokens to curb hallucinations over music/silence.
export VOXHELM_WHISPERCPP_MAX_CONTEXT="0"
export VOXHELM_WHISPERCPP_SUPPRESS_NST="true"
# Post-decode transcript sanitizer (defaults shown). Deterministic backstop that
# collapses repeated-sentence loops and drops subtitle-credit / punctuation-only
# hallucinations from every produced transcript. Disable only to inspect raw
# decoder output.
export VOXHELM_SANITIZE_TRANSCRIPT="true"
export VOXHELM_SANITIZE_REPEAT_THRESHOLD="4"
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
export VOXHELM_DIARIZATION_BACKEND="none"
export VOXHELM_PYANNOTE_MODEL="pyannote/speaker-diarization-3.1"
export VOXHELM_HUGGINGFACE_TOKEN=""
export VOXHELM_MODEL_CACHE_DIR="$PWD/var/models"
export VOXHELM_WYOMING_STT_HOST="0.0.0.0"
export VOXHELM_WYOMING_STT_PORT="10300"
export VOXHELM_WYOMING_STT_BACKEND="mlx"
export VOXHELM_WYOMING_STT_MODEL=""
export VOXHELM_WYOMING_STT_LANGUAGE=""
export VOXHELM_WYOMING_STT_LANGUAGES="de,en"
export VOXHELM_WYOMING_STT_PROMPT=""
# Host-wide lane scheduler for local inference (see "Lane Scheduler" below).
export VOXHELM_LANE_SCHEDULER_ENABLED="false"
export VOXHELM_LANE_SCHEDULER_DIR="$PWD/var/lane-scheduler"
export VOXHELM_LANE_SCHEDULER_STALE_SECONDS="1800"
export VOXHELM_LANE_SCHEDULER_INTERACTIVE_SLOTS="1"
export VOXHELM_LANE_SCHEDULER_NON_INTERACTIVE_SLOTS="1"
# Text-to-speech. Piper is the default backend. The optional Kokoro ONNX backend
# (English "af_heart" + the German "Martin" fine-tune) needs the `kokoro` extra
# (`uv sync --extra kokoro`) plus espeak-ng (`brew install espeak-ng`). Each
# VOXHELM_KOKORO_MODELS entry is voice_key=model:voicepack:voicepack_key:language;
# relative paths resolve against VOXHELM_KOKORO_MODEL_DIR. With no models
# configured the voice registry exposes no Kokoro voices and Piper is untouched.
# VOXHELM_KOKORO_DEFAULT_VOICE names the voice used when a request reaches the
# Kokoro backend without an explicit voice; unset falls back to the first
# VOXHELM_KOKORO_MODELS entry in sorted-key order.
# VOXHELM_ESPEAK_LIBRARY overrides the libespeak-ng shared library when set.
export VOXHELM_KOKORO_MODEL_DIR="$PWD/var/kokoro"
export VOXHELM_KOKORO_MODELS="kokoro-af_heart=kokoro-v1.0.onnx:voices-v1.0.bin:af_heart:en-us,kokoro-martin=kokoro-martin.onnx:voices-martin.npz:martin:de"
export VOXHELM_KOKORO_DEFAULT_VOICE=""
export VOXHELM_ESPEAK_LIBRARY=""
# Automatic language routing detects the language of outgoing TTS text and, when
# the detection clears the routing floor (>=20 chars, >=0.90 confidence, a mapped
# and registered voice), swaps the pinned voice for the language's mapped voice
# and makes the detected language effective end-to-end (backend, SynthesisResult,
# and the X-Voxhelm-Backend/Voice/Language response headers on /v1/audio/speech).
# Needs the `routing` extra (`uv sync --extra routing`); enabling it without the
# extra fails fast at startup. Default off (behavior byte-for-byte unchanged).
# VOXHELM_TTS_LANGUAGE_VOICES maps a language code to a registry voice key across
# all backends. Per HTTP request, `routing: false` bypasses routing (the Wyoming
# path always routes).
export VOXHELM_TTS_LANGUAGE_ROUTING="false"
export VOXHELM_TTS_LANGUAGE_VOICES="de=kokoro-martin,en=kokoro-af_heart"
export VOXHELM_ALLOWED_URL_HOSTS="media.example.com"
export VOXHELM_TRUSTED_HTTP_HOSTS="internal.example.lan"
export VOXHELM_PRIVATE_URL_HOSTS="s3.internal.example.lan"
export VOXHELM_BATCH_MAX_STAGED_UPLOAD_BYTES="536870912"
export VOXHELM_STAGED_INPUT_RETENTION_SECONDS="86400"
export VOXHELM_SOURCE_ARTIFACT_RETENTION_SECONDS="86400"
export VOXHELM_JOB_METADATA_RETENTION_SECONDS="7776000"
export VOXHELM_TRANSCRIPTION_EXECUTION_MODE="django_tasks"
export VOXHELM_WORKER_TOKENS="atlas=replace-worker-token"
export VOXHELM_REMOTE_WORKER_LEASE_SECONDS="1800"
export VOXHELM_REMOTE_WORKER_POLL_SECONDS="5"
export VOXHELM_REMOTE_WORKER_MAX_ATTEMPTS="3"
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

Staged uploads are stored in Voxhelm's configured artifact backend before
execution. Django Tasks jobs delete the temporary staged object immediately after
materialization; remote worker jobs delete it after a successful completion
records the worker-copied job-owned source artifact. Terminal remote failures
release the staged upload claim so the same `upload_id` can be retried until the
staged object expires. Unclaimed staged uploads expire after
`VOXHELM_STAGED_INPUT_RETENTION_SECONDS` and are opportunistically cleaned on
later staging/submission requests.
If the artifact backend, filesystem root, S3 endpoint, or bucket changes after
staging, submitters must stage the media again; Voxhelm rejects `upload_id`
values whose store identity no longer matches the active artifact store.

### Job artifact retention (D-09)

Every batch transcription job keeps a job-owned copy of its input (`source`
artifact, non-exposed) and, for video, the extracted WAV (`extracted_audio`).
Remote workers upload the same two kinds. These are intermediates, so they are
pruned by a periodic management command:

```bash
uv run python manage.py prune_job_artifacts --dry-run   # list only
uv run python manage.py prune_job_artifacts
```

- `extracted_audio` is deleted once the job is terminal (succeeded, failed,
  canceled or expired);
- `source` is deleted when the job finished more than
  `VOXHELM_SOURCE_ARTIFACT_RETENTION_SECONDS` ago (default `86400`, 24 h;
  `0` deletes it on the next run after the job finished);
- transcript, speaker-sidecar and speech artifacts, and anything marked exposed,
  are never touched; queued and running jobs are never touched.

Deletion is allowlist-based: only an object voxhelm generated for that job is
ever deleted, i.e. a key ending in exactly `jobs/<job_id>/<artifact name>` or
`jobs/<job_id>/attempt-<n>/<artifact name>` below the artifact prefix it was
written under (so a later `VOXHELM_ARTIFACT_PREFIX` change does not block
retention), with no empty, `.` or `..` segments. On the filesystem backend the resolved real path
must also equal the literal path under the root (no symlinks) and the file must
not be hard-linked. Anything else is refused: the row and object stay, and the
command (and `--dry-run`) lists it as refused; refused queued deletions stay
queued. An allowed object that another artifact row or
staged upload still references after resolution (store identity, real path
and inode, canonical S3 endpoint and exact key) is also kept; only its row
goes.

The command deletes the stored object (filesystem or S3/MinIO) first and then
the row; an already-missing object counts as deleted. If an object cannot be
deleted the row is kept for the next run and the command exits non-zero. When a
remote completion replaces a job's artifact rows, the replaced non-exposed
`source` / `extracted_audio` objects that pass the allowlist and that no kept
or final artifact shares are queued (`PendingArtifactDeletion`, in the same
transaction) and deleted right after the commit; a failed deletion stays queued
and the next `prune_job_artifacts` run retries it. A completion retry from the
same worker and lease still matches after pruning removed those intermediates.
Each deletion re-checks its candidate, the allowlist and the references under
the database write lock, so overlapping runs are safe.

After the artifacts (and the queued deletions), the same command deletes old
job metadata: the `Job` row of a terminal job that finished more than
`VOXHELM_JOB_METADATA_RETENTION_SECONDS` ago (default `7776000`, 90 days; the
job's `finished_at`, else its last update; `0` disables job metadata pruning).
A job is only deleted once it owns nothing anymore: no artifact row of any kind
(so intermediates must have been pruned first, and a job whose intermediate was
refused, failed to delete or is not yet past source retention keeps its row),
no queued `PendingArtifactDeletion` and no staged upload still claimed by it.
Transcript, speaker-sidecar and speech artifacts are kept indefinitely and are
served through their job, so a job that still has them keeps its row. Queued
and running jobs are never deleted. Each batch re-checks these conditions under
the database write lock. Deleting a job also frees its `task_ref`, so a later
submission with the same `task_ref` creates a new job. `--dry-run` lists the
job rows a real run would delete, counting the artifacts and queued deletions
that run would remove first as already gone.

Nothing schedules the command yet: run it from cron or a systemd timer (hourly
is enough) on the control plane, with the same environment as the service.

Current scope note: batch staged uploads are audio-only in this slice. URL
audio and URL video keep working on the existing path. Uploaded video and true
service-owned chunk splitting/stitching are still explicitly deferred.

## Batch Speaker Diarization

Batch `job_type=transcribe` requests can opt into speaker labels:

```bash
curl -X POST http://127.0.0.1:8000/v1/jobs \
  -H "Authorization: Bearer replace-me" \
  -H "Content-Type: application/json" \
  -d '{
    "job_type": "transcribe",
    "lane": "batch",
    "backend": "auto",
    "model": "auto",
    "input": {"kind": "url", "url": "https://example.com/episode.mp3"},
    "output": {"formats": ["json", "dote", "podlove", "vtt"]},
    "diarization": {"enabled": true, "num_speakers": 4}
  }'
```

When enabled, Voxhelm runs diarization after STT, aligns speaker turns to
transcript segments by largest timestamp overlap, and emits stable generic
labels such as `Speaker 1` and `Speaker 2`. If the expected speaker count is
known, pass `diarization.num_speakers`; alternatively pass
`diarization.min_speakers` and/or `diarization.max_speakers` as pyannote speaker
hints. Verbose JSON includes `speaker` only on labeled segments. DOTe fills
`speakerDesignation`; Podlove fills both `speaker` and `voice`. WebVTT
intentionally remains unchanged in this first slice.

The default `VOXHELM_DIARIZATION_BACKEND=none` makes requested diarization jobs
fail clearly instead of silently emitting unlabeled output. A guarded pyannote
adapter is available with `VOXHELM_DIARIZATION_BACKEND=pyannote`,
`VOXHELM_PYANNOTE_MODEL`, `VOXHELM_PYANNOTE_DEVICE`, and
`VOXHELM_HUGGINGFACE_TOKEN`. `VOXHELM_PYANNOTE_DEVICE=auto` uses Apple MPS when
available, then CUDA, then CPU. Install the optional model stack with:

```bash
uv sync --extra diarization
```

`VOXHELM_HUGGINGFACE_TOKEN` is required because the default pyannote pretrained
speaker-diarization pipeline is downloaded from Hugging Face and its model terms
must be accepted by the token-owning account before first use. With current
pyannote releases this may require access to the configured pipeline repository
and its gated component repositories, including
`pyannote/speaker-diarization-community-1`. If `VOXHELM_HUGGINGFACE_TOKEN` is
unset, Voxhelm also reads the common `HF_TOKEN` environment variable.

Production diarization requires all of the following:

- install dependencies with `uv sync --extra diarization`
- set `VOXHELM_DIARIZATION_BACKEND=pyannote`
- set `VOXHELM_HUGGINGFACE_TOKEN` or `HF_TOKEN`
- keep `VOXHELM_PYANNOTE_DEVICE=auto` or explicitly set `mps`, `cuda`, or `cpu`
- accept Hugging Face access for `pyannote/speaker-diarization-3.1`,
  `pyannote/speaker-diarization-community-1`, and any gated dependency reported
  by pyannote during model loading

The first successful run downloads model weights through the Hugging Face /
pyannote cache path and can take time. Long podcast episodes are CPU-heavy;
submit them as async batch jobs and inspect the worker logs rather than holding
an HTTP/admin request open.

Short smoke test:

```bash
curl -X POST http://127.0.0.1:8000/v1/jobs \
  -H "Authorization: Bearer replace-me" \
  -H "Content-Type: application/json" \
  -d '{
    "job_type": "transcribe",
    "lane": "batch",
    "backend": "auto",
    "model": "auto",
    "input": {"kind": "url", "url": "https://example.com/short-audio.mp3"},
    "output": {"formats": ["json", "dote", "podlove"]},
    "diarization": {"enabled": true}
  }'
```

After the job succeeds, verify the JSON, DOTe, and Podlove artifacts contain
`Speaker 1` / `Speaker 2` labels.

## Remote Pull Transcription Workers

Voxhelm can keep the producer-facing batch API unchanged while routing new
batch transcription jobs to trusted HTTP pull workers:

```bash
export VOXHELM_TRANSCRIPTION_EXECUTION_MODE="remote_pull"
export VOXHELM_WORKER_TOKENS="atlas=replace-worker-token"
export VOXHELM_ARTIFACT_BACKEND="s3"
export VOXHELM_ARTIFACT_S3_ENDPOINT_URL="https://minio.example"
export VOXHELM_ARTIFACT_S3_ACCESS_KEY_ID="replace-me"
export VOXHELM_ARTIFACT_S3_SECRET_ACCESS_KEY="replace-me"
export VOXHELM_ARTIFACT_BUCKET="voxhelm"
```

In `remote_pull` mode, `job_type=transcribe` submissions through `POST /v1/jobs`
are persisted as normal queued Voxhelm jobs but are not enqueued into Django
Tasks. `job_type=synthesize` continues to use Django Tasks. Switching
`VOXHELM_TRANSCRIPTION_EXECUTION_MODE` back to `django_tasks` restores the
studio-only local transcription path without a migration.
`remote_pull` requires a valid `VOXHELM_WORKER_TOKENS` entry plus the shared S3
artifact backend and complete S3 endpoint, credential, and bucket settings at
startup; the local filesystem artifact backend is valid for `django_tasks` mode
only. `VOXHELM_TRANSCRIPTION_EXECUTION_MODE` must be either `django_tasks` or
`remote_pull`. The remote lease, poll, and max-attempt settings must be positive
integers. URL inputs are validated against `VOXHELM_ALLOWED_URL_HOSTS` when the
job is submitted, before a remote job can remain queued.

URL fetches (sync transcription, batch jobs, the operator UI and remote
workers) share one SSRF-guarded fetcher:

- every redirect hop is re-checked against `VOXHELM_ALLOWED_URL_HOSTS` and the
  https-only / `VOXHELM_TRUSTED_HTTP_HOSTS` scheme rule, so an open redirect on
  an allowlisted host cannot reach another host or downgrade to plain HTTP;
  at most 5 redirects are followed;
- every resolved IP is checked before connecting, and the socket connects to
  exactly the checked address (no second DNS lookup, so DNS rebinding cannot
  swap it). Link-local (including cloud metadata `169.254.169.254`), the
  other known metadata endpoints (`fd00:ec2::254`, `100.100.100.200`,
  `192.0.0.192`), multicast, reserved, unspecified and `0.0.0.0/8` addresses are always
  refused. Other non-public addresses (RFC 1918, loopback, CGNAT/Tailscale
  `100.64.0.0/10`, IPv6 ULA) are only accepted for hosts listed in
  `VOXHELM_PRIVATE_URL_HOSTS` or `VOXHELM_TRUSTED_HTTP_HOSTS`. Allowlisted
  hosts that resolve to tailnet or LAN addresses must therefore be added to
  `VOXHELM_PRIVATE_URL_HOSTS` (on the control plane and on remote workers);
- `http_proxy`/`https_proxy` environment variables are ignored for media
  fetches.

When more than one remote worker is active, the control plane balances claims
fairly instead of letting the lowest-latency poller win every race: at claim
time a worker that is "ahead" on recent claims defers to a fresh, idle, enabled
peer that has handled strictly fewer recent claims, so two workers converge on a
roughly even split. With no eligible peer (the other worker is offline, busy, or
disabled) a worker never defers, so a lone worker still claims everything and the
fleet self-heals. Tunable via `VOXHELM_REMOTE_WORKER_BALANCE_ENABLED` (default
`true`), `VOXHELM_REMOTE_WORKER_BALANCE_WINDOW_SECONDS` (recent-claim comparison
window, default `3600`), and `VOXHELM_REMOTE_WORKER_BALANCE_PEER_FRESH_SECONDS`
(how recently a peer must have heartbeated to count, default `30`). This is
control-plane-only behavior; workers need no changes.

Worker endpoints are internal and use a separate bearer-token domain from
producer tokens. Startup configuration rejects any raw token value shared
between `VOXHELM_BEARER_TOKENS` and `VOXHELM_WORKER_TOKENS`:

- `POST /v1/internal/workers/heartbeat`
- `POST /v1/internal/work/claim`
- `POST /v1/internal/work/<job_id>/heartbeat`
- `POST /v1/internal/work/<job_id>/complete`
- `POST /v1/internal/work/<job_id>/fail`

Claims use studio server time, a bounded lease, and an atomic conditional
database update so concurrent workers cannot claim the same SQLite-backed job.
Workers at their advertised/requested active-claim capacity receive no new claim.
When a producer retries the same `task_ref`, Voxhelm reconciles any expired
remote lease first: attempts that remain move back to `queued`, while exhausted
attempts fail clearly.
Claim responses snapshot the attempt-scoped artifact prefix and non-secret
artifact-store identity on the job, and completion validates manifests against
that leased snapshot even if `VOXHELM_ARTIFACT_PREFIX`, the filesystem root, or
the S3 endpoint/bucket changes before the worker reports back. Voxhelm checks
artifact object existence and size before the short settlement transaction so a
slow object store does not hold SQLite's write lock. The committed artifact rows
keep the winning store identity so producer downloads continue to read from the
store that accepted the completion manifest.
Workers must advertise supported transcript `output_formats`, concrete STT
backend names, and concrete STT model names when claiming work. `auto`,
`whisper-1`, and `gpt-4o-mini-transcribe` match the configured default backend
and model; claim responses send that resolved concrete backend/model while
preserving the submitted aliases as `requested_backend`/`requested_model`.
Disabled workers are rejected before heartbeat state is updated.
Disabling a worker stops new claims but does not block completion, failure, or
lease heartbeat for a job the worker already owns. Worker completion accepts
only the currently assigned worker and lease token.
Completions must include a non-exposed job-owned source artifact plus the
requested transcript artifacts. Artifacts must be reported under the claimed
attempt prefix, for example `voxhelm/jobs/<job_id>/attempt-1/transcript.txt`;
Voxhelm verifies that each reported object exists in the configured artifact
store and that its stored size matches the manifest before marking the job
succeeded. The producer still downloads winning artifacts through
`GET /v1/jobs/<job_id>/artifacts/<name>`. Transcript and speaker-sidecar
artifacts must use the exact expected MIME types before Voxhelm exposes them.

Known-speaker jobs are claimable only by workers advertising the required
pyannote/wespeaker speaker-sidecar capability. Known-speaker reference URLs are
checked against `VOXHELM_ALLOWED_URL_HOSTS` at submission and again before a
remote claim is handed out. Uploaded known-speaker reference clips are rejected
before remote claim in this slice; use URL reference audio for remote-worker
jobs. Completion metadata stores only type-checked scalar worker fields plus
server-derived source, worker, attempt, and request metadata. URL completions
keep the server-owned job input URL as `source_url` and ignore worker-supplied
redirect URLs. URL-shaped strings, nested values, negative or non-finite
timings/summary metrics, and private known-speaker reference
URLs/ranges are not echoed through producer-visible job metadata. Speaker
sidecars are accepted only for known-speaker jobs.

Run a manual worker on a trusted macOS host with an env file containing the
worker token, Voxhelm URL, shared artifact credentials, local STT/model cache
settings, `VOXHELM_ALLOWED_URL_HOSTS`, and optional Hugging Face token. The
`caffeinate` wrapper keeps the machine awake only while the worker command is
running:

```bash
/usr/bin/caffeinate -i -m -s -- uvx --from ".[diarization]" voxhelm-remote-worker \
  --env-file "$HOME/Library/Application Support/voxhelm-remote-worker/worker.env" \
  --once
```

From this checkout, `just worker-once` runs the same one-job command. Override
`VOXHELM_WORKER_ENV_FILE` for a non-default env-file path. It defaults to
`$HOME/Library/Application Support/voxhelm-remote-worker/worker.env`.
Override `VOXHELM_WORKER_UVX_SOURCE` for a package or git source instead of
the local checkout, and include the `diarization` extra when processing
speaker-label jobs. Use `just worker-loop` only when you want to keep polling
manually until `Ctrl-C`.

The worker defaults to one active job, periodically heartbeats the leased job
while local inference runs, uploads the source, optional extracted audio,
requested transcript artifacts, and known-speaker `transcript.speakers.json`
sidecar under the claimed attempt prefix, then posts the completion manifest.

Operational note: the application endpoints still require worker auth, but the
macmini/Traefik edge must also block `/v1/internal/*` on public routes unless a
deliberately private worker route is configured.

Current implementation status: the studio control-plane endpoints and
`remote_pull` dispatch switch are implemented, `voxhelm` is published to PyPI
and installable as a `uv` tool (`uv tool install "voxhelm[diarization]"`), and a
`voxhelm-remote-worker` is deployed on `atlas`. Edge protection and the
production python-podcast proof remain follow-up work.

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

## Lane Scheduler

All local STT and TTS inference on one host (sync `POST /v1/audio/transcriptions`,
`POST /v1/audio/speech`, local batch jobs, and the Wyoming sidecar) passes through
one cross-process lane scheduler (`lane_scheduler.py`, state under
`VOXHELM_LANE_SCHEDULER_DIR`). Wyoming requests are the `interactive` lane;
everything else is `non-interactive`; callers cannot choose a lane.

Capacity is a small, bounded set of slots (decision D-24):

- `VOXHELM_LANE_SCHEDULER_INTERACTIVE_SLOTS` (default `1`) are reserved for the
  interactive lane. Non-interactive work never occupies them.
- `VOXHELM_LANE_SCHEDULER_NON_INTERACTIVE_SLOTS` (default `1`) cap how many
  non-interactive inferences run at once. Interactive requests may also use an
  idle non-interactive slot.
- Waiters are admitted by lane priority (interactive first), then FIFO. Running
  work is never preempted. With the defaults, one Wyoming request is admitted
  immediately while one long HTTP transcription is inside `whisper-cli`; both
  then share the GPU, so the interactive request is slower than on an idle host
  but not blocked.
- Each holder is recorded as its own file; a crashed holder is reclaimed when its
  process is gone, and a holder older than `VOXHELM_LANE_SCHEDULER_STALE_SECONDS`
  (default `1800`) is reclaimed even if the process is alive.
- `VOXHELM_LANE_SCHEDULER_ENABLED=false` disables all gating. Setting
  `0` interactive and `1` non-interactive slots restores the original
  single-slot behaviour without a code change.

Only backends that run outside the calling process may overlap: `whisper-cli`
subprocesses and the WhisperKit sidecar. In-process backends keep a lock per
backend inside their own process (`mlx-whisper` for STT; Piper and Kokoro for
TTS), so two requests that reach the same in-process backend in the same process
still serialize. The Wyoming sidecar is its own process, so its `mlx` backend
overlaps a `whisper-cli` run from the HTTP process without sharing state.

The sync transcription endpoint terminates its `whisper-cli` child when the HTTP
client disconnects (Django's ASGI handler cancels the request), so an abandoned
upload no longer burns a slot and GPU time to completion. In-process backends
cannot be interrupted mid-inference; a disconnect only prevents a backend that
has not started yet from running.

Deploy note: the holder layout changed from one `holder.json` to a `holders/`
directory. New code counts a leftover `holder.json` as a non-interactive holder
until it is released or reclaimed; old code does not see `holders/`, so during
the seconds in which the three launchd services restart one after another the
bound can be exceeded by one holder. Restart all three services in one deploy
run (the ops-library role does).

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

Current limitation: the lane scheduler gates Voxhelm's HTTP, batch, and Wyoming
entry points, but it does not reach inside the WhisperKit sidecar itself. Once
Voxhelm has admitted a WhisperKit request, the sidecar's internal inference
concurrency remains outside that scheduler's direct control.
