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
The normalizer runs ffmpeg at error log level in its own process group: a 2 KiB
stderr prefix is logged on failure (never part of the error), all further output is
drained and discarded so noisy but decodable media still succeeds, and on
every exit path the whole group is killed, so the decode deadline also covers
descendants holding its pipes.

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
  `VOXHELM_SANITIZE_REPEAT_THRESHOLD` (default `4`) consecutive repeats, with
  at most **one second** between the previous cue end and the next cue start.
  Longer gaps and backward-moving starts or ends split runs. Overlapping cues
  are eligible only when neither boundary moves backward. A speaker-label change
  also splits a run, including a change between a label and an absent label
  (`None`). Consecutive unlabeled cues may still collapse together. When a run
  collapses, its shared speaker label is retained. The repeat threshold
  catches the observed loops (9–84×) while leaving natural backchannels and
  rhetorical repetition inside a single cue untouched.
- **Non-speech / credit hallucinations** — subtitle-credit cues
  (`Untertitelung des ZDF, 2020`, `Untertitel im Auftrag des ZDF für funk, 2017`,
  `Untertitel von Amara.org`, `Untertitelung von Amara.org`, or standalone
  `Amara.org`, ignoring case, whitespace and surrounding punctuation)
  and punctuation-only noise such as long dot-runs are dropped. A long dot-run
  embedded in an otherwise-real segment is stripped while the real text and
  adjacent genuine segments survive.

Amara credits must match those complete attribution forms. Topic sentences such
as `Untertitel diskutieren wir heute auf Amara.org` and unknown attribution
phrasing remain intact, even when they end in that domain.

The sanitizer is conservative by design: it biases toward false negatives over
removing genuine speech, only ever removes or collapses artifact segments, and
returns clean transcripts unchanged. Disable it with
`VOXHELM_SANITIZE_TRANSCRIPT=false` only to inspect raw decoder output.

## Transcript numeric validation

Every cue must have finite start and end timestamps whose multiplication by
1000 remains finite, including empty-text cues.
NaN, positive/negative infinity and values too large to scale to milliseconds raise a controlled `TranscriptionError`
before sanitization (even when disabled), rendering or transcript artifact
publication. Local jobs and remote workers validate the entire cue list before
writing the first transcript artifact; source audio intermediates may already
exist. The same rejection policy applies to text, verbose JSON, VTT, DOTE and
Podlove. Finite negative/reversed timestamps retain the existing policy:
verbose JSON preserves raw boundaries; timed formats clamp reversed ends to
starts and render negative times as zero.

Known-speaker reference and segment embeddings must be nonempty and contain
only finite coordinates. Invalid evidence raises `KnownSpeakerError`, aborting
classification before a speaker sidecar can be published. Scaled averaging and
normalization keep very large or very small finite embeddings usable without
avoidable overflow/underflow. Zero vectors remain valid: their cosine score is
zero and they cannot clear the default auto-accept similarity threshold.

Known-speaker jobs re-cut transcript segments at speaker changes by default:
whisper.cpp then runs with per-word timings on one processor (slower than the
configured `VOXHELM_WHISPERCPP_PROCESSORS`), and the transcript artifacts and
the speaker sidecar carry the re-cut segments. `resegment_hop: 0` in
`diarization.known_speaker` turns this off. See
`specs/known-speaker-diarization.md`.

With `VOXHELM_KNOWN_SPEAKER_STT_BACKEND=parakeet`, those jobs (when their model
is `auto`, `whisper-1` or `gpt-4o-mini-transcribe`) are transcribed by Parakeet
TDT instead: sentence-shaped, punctuated segments with word timings that end at
speaker changes, and several times faster than whisper.cpp with one processor.
When Parakeet is unavailable (CLI or model missing, or the job names a language
Parakeet does not support) the job falls back to whisper.cpp, which also
returns word timings, and only then to the configured
`VOXHELM_STT_BACKEND`/`VOXHELM_STT_FALLBACK_BACKEND` chain, without the re-cut.
A Parakeet run that fails for any other reason fails the job. Other jobs, and
jobs naming a concrete model, are never rerouted. See
`specs/2026-10-15_parakeet_asr_backend.md`.

Pyannote diarization accepts at most two decoded hours of 16 kHz, 16-bit mono
PCM. Decode stops after one extra sample to detect oversized compressed inputs;
WAV geometry, frame count and complete PCM bytes are checked before waveform
allocation. Decoder stdout and diagnostics are discarded, with a 64 KiB stderr
allowance and the configured decode deadline; failures expose only safe categories.
Temporary decoded audio is removed on every exit. Pipeline initialization is
cached only after successful device placement, so failed placement can be retried.

## WebVTT Output

The `vtt` format (sync `response_format=vtt` and the batch `vtt` artifact)
writes one cue per segment, each with a single payload line. Like `dote` and
`podlove`, it skips segments with empty text and clamps an end time that lies
before the start time to the start time. Cue text is escaped for WebVTT: `&`,
`<`, and `>` become `&amp;`, `&lt;`, and `&gt;`, `-->` becomes `->`, and line
breaks inside a segment collapse to one space, so transcript text can neither
inject markup nor end or split a cue. Players decode the character references;
consumers that parse the raw file must decode them too. Speaker labels stay out
of WebVTT. Artifacts stored before this change are not rewritten.

## Kokoro inference and installed style validation

German preprocessing expands numeric units, EUR amounts and temperatures only
when the complete numeric token has at most 64 digits (integer and fractional
parts combined; sign and separator excluded). Longer numbers keep their literal
spelling, including leading zeros, without integer conversion or fractional
digit expansion. The independent `EUR` abbreviation still expands to `Euro`.
This limit is independent of request text length and Python's global digit limit.
Cardinal labels such as `Kapitel 2.` become `Kapitel zwei.` and keep their
sentence punctuation without reserving internal marker text; literal input such
as `__GODE_CARDINAL_DOT__` and its surrounding whitespace survives unchanged.

Kokoro accepts nonempty finite mono inference arrays, either a sample vector or
a single mono batch `(1, samples)`. Invalid output fails synthesis before WAV
publication. Valid samples keep the existing clipping and 16-bit PCM conversion.
Installed voicepacks must contain real numeric, finite float32-compatible styles
shaped `(rows, 1, 256)`, with 1–512 rows. Before NumPy allocation, the selected
NPY v1/v2 member must have a bounded header (10,000 bytes), a matching complete
payload and at most 2 MiB of expanded numeric data (up to 16 bytes per value). Token counts still clamp to the
last row. Invalid packs raise a controlled backend-unavailable error and are
never cached; replacing the file allows a subsequent load to recover. ONNX,
voicepack and Piper model/config loaders recheck all their artifact identities
after loading. A changed generation fails with a retryable backend-unavailable
error before cache publication; a mixed Piper model/config pair is discarded.

Kokoro refuses nonblank text that yields no supported phoneme tokens with an
actionable client error (HTTP 400 on the sync speech API). Supported one-character utterances and punctuation tokens
retain ordinary inference behavior. Audio chunks are validated and written
incrementally to an owned WAV, retaining at most one inference chunk and its PCM
conversion at a time. Total output is capped at 30 minutes (43,200,000 frames,
86,400,000 PCM bytes at 24 kHz mono). Duration comes from written frames. Late
inference errors, cancellation and cap overflow remove the partial WAV before
any artifact is published.

## Historical fingerprint upgrades

Migration 0008 backfills only the database selected by `migrate --database`.
It reads at most 100 jobs per keyset page, closes reads before SQLite writes,
and uses a temporary indexed lookup instead of retaining all duplicate keys.
The newest active matching job keeps its frozen fingerprint; failed jobs also
receive fingerprints. The entire upgrade rolls back on failure.

## Download and publication failure ownership

ASGI artifact downloads read at most 64 KiB at a time. On disconnect, cancellation
propagates immediately, while the active worker retains the reader until its read
settles and closes it exactly once. Response closure also handles streams that
never started. S3 reads use the transport's 60-second socket read timeout; local
regular-file reads settle through the filesystem. Closing never races a read.
ASGI responses keep the `Content-Length` (and any `Content-Disposition`) that a
filesystem download would send under WSGI.

If filesystem publication and temporary cleanup both fail, the original storage
error propagates and a fixed cleanup warning is logged without paths or payloads.
The previous destination and unrelated objects remain intact; the owned temporary
file may need repair. A cleanup failure on its own still propagates normally.

## Local Development

```bash
uv sync
just test
DJANGO_DEBUG=true uv run uvicorn config.asgi:application
```

Outside `DJANGO_DEBUG=true` the app refuses to start without `DJANGO_SECRET_KEY`
(see "Operator UI security" below). The test suite and mypy use
`tests/settings.py`, which supplies a fixed test-only key.

Install `ffmpeg` on PATH before `just check`; the synthetic WAV/MP3/AAC regressions
exercise real codec boundaries without downloading media or models. CI ensures it is
installed before running tests.

Remote ownership regressions use spawned processes with scratch SQLite databases
and synthetic filesystem artifacts (`tests/test_remote_process_contracts.py`),
including capacity, lock-wait lease timing, and completion/failure races.

`just check` runs Ruff lint, a Ruff format check (`just format-check`; fix with
`uv run ruff format .`), mypy and pytest. GitHub Actions
(`.github/workflows/ci.yml`) runs the same steps on macOS arm64 on every push and pull
request; tests that need the real Kokoro models are skipped there, and no
secrets or services are needed. A separate CI job runs `pip-audit` over the locked
dependencies, including the optional extras (`diarization`, `kokoro`, `routing`), and
fails on any known vulnerability. It runs on macOS arm64 because
pip-audit skips requirements whose environment markers do not match the machine it runs
on, and the Darwin/arm64-only dependencies (mlx-whisper, piper-tts, torch and what they
pull in) would otherwise go unaudited. Run the same audit locally on a Mac with:

```bash
uv export --locked --all-extras --no-hashes --no-emit-project -o /tmp/voxhelm-requirements.txt
uvx pip-audit==2.10.1 -r /tmp/voxhelm-requirements.txt --disable-pip --no-deps
```

## Required Environment

```bash
export DJANGO_SECRET_KEY="replace-me"
export VOXHELM_BEARER_TOKENS="archive=replace-me"
```

`DJANGO_SECRET_KEY` is mandatory unless `DJANGO_DEBUG=true`: a missing or blank
value raises `ImproperlyConfigured` at startup instead of falling back to a
public default. Only `DJANGO_DEBUG=true` uses a fixed development key.

Optional settings:

```bash
export VOXHELM_ALLOWED_HOSTS="localhost,127.0.0.1"
export VOXHELM_CSRF_TRUSTED_ORIGINS="https://voxhelm.example.com"
# Operator UI hardening (defaults shown; see "Operator UI security").
# VOXHELM_SECURE_COOKIES defaults to true unless DJANGO_DEBUG=true.
export VOXHELM_SECURE_COOKIES="true"
export VOXHELM_LOGIN_MAX_FAILURES="5"
export VOXHELM_LOGIN_LOCKOUT_SECONDS="900"
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
# Parakeet TDT through the parakeet-mlx CLI (defaults shown). The CLI runs as a
# child process from its own Python 3.13 tool environment, offline, against a
# model directory deployed at a pinned commit (config.json + model.safetensors);
# an empty model directory leaves the backend unavailable. Supported languages:
# bg cs da de el en es et fi fr hr hu it lt lv mt nl pl pt ro ru sk sl sv uk.
export VOXHELM_PARAKEET_BIN="parakeet-mlx"
export VOXHELM_PARAKEET_MODEL_DIR=""
export VOXHELM_PARAKEET_MODEL_NAME="parakeet-tdt-0.6b-v3"
export VOXHELM_PARAKEET_CHUNK_SECONDS="600"
export VOXHELM_PARAKEET_OVERLAP_SECONDS="30"
export VOXHELM_PARAKEET_MAX_SENTENCE_SECONDS=""  # unset: no --max-duration
export VOXHELM_PARAKEET_TIMEOUT_SECONDS="1800"
# Window transcription for the ad-analysis drift check (D-27): whisper
# large-v3-turbo through VOXHELM_WHISPERCPP_BIN. A file name in
# VOXHELM_MODEL_CACHE_DIR or a path; the file must have the pinned SHA-256
# (transcriptions/windows.py), any other file leaves window transcription, and so
# the ad-analysis-v1 profile, unavailable.
export VOXHELM_WINDOW_MODEL="ggml-large-v3-turbo.bin"
export VOXHELM_WINDOW_TIMEOUT_SECONDS="120"
# ffprobe measures the media duration the ad-analysis-v1 profile checks a job's
# chapter marks against (D-29); without it the profile is unavailable.
export VOXHELM_FFPROBE_BIN="ffprobe"
# "parakeet" routes known-speaker jobs with the re-cut to Parakeet first (see
# the known-speaker section); empty keeps them on the configured chain.
export VOXHELM_KNOWN_SPEAKER_STT_BACKEND=""
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
export VOXHELM_WYOMING_STT_MAX_AUDIO_SECONDS="120"
export VOXHELM_WYOMING_MAX_EVENT_DATA_BYTES="262144"
export VOXHELM_WYOMING_MAX_EVENT_PAYLOAD_BYTES=""
export VOXHELM_WYOMING_MAX_CONNECTIONS="16"
export VOXHELM_WYOMING_MAX_INFERENCE_THREADS="4"
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
export VOXHELM_REMOTE_WORKER_POLL_SECONDS="5"  # integer 1..3600; server polling hint
export VOXHELM_REMOTE_WORKER_MAX_ATTEMPTS="3"
export VOXHELM_BOOTSTRAP_OPERATOR_USERNAME="jochen"
export VOXHELM_BOOTSTRAP_OPERATOR_EMAIL=""
export VOXHELM_BOOTSTRAP_OPERATOR_PASSWORD="replace-me"
```

Bootstrap the initial operator account after migrations:

```bash
uv run python manage.py bootstrap_operator --username jochen --password "replace-me"
```

Passwords supplied through `--password` or `VOXHELM_BOOTSTRAP_OPERATOR_PASSWORD`
are preserved exactly, including leading and trailing whitespace, when creating
or updating an account. An empty password is rejected; whitespace-only passwords
are preserved. Use shell quoting for intentional spaces, for example
`--password ' passphrase '`. Usernames and email addresses still have surrounding
whitespace removed.

The operator login form also preserves password whitespace. Earlier versions
trimmed bootstrap passwords: rerunning bootstrap now replaces that trimmed
credential with the exact supplied value. Remove unintended whitespace (such as
a trailing newline from a secret file) before bootstrapping. For browser sign-in,
avoid passwords containing line breaks, which password inputs remove.

Deploy-time note: the deployment layer should call the same in-app command with the real secret rather than creating the operator directly in a separate repo.

### Operator UI security

The operator UI at `/` is meant to be reached through the private Traefik HTTPS
ingress, which forwards `X-Forwarded-Proto: https`.

- **Secret key:** `DJANGO_SECRET_KEY` must be set unless `DJANGO_DEBUG=true`;
  startup fails closed otherwise. The ops-library `voxhelm_deploy` role writes it
  to `voxhelm.env`. The remote worker defaults it to a fixed local value because
  it serves no sessions.
- **Secure cookies:** the session and CSRF cookies carry `Secure` (and the
  session cookie `HttpOnly`) unless `DJANGO_DEBUG=true`. Browsers do not store
  `Secure` cookies from a plain-HTTP origin other than `localhost`, so signing in
  directly at `http://<host>:8787` stops working. Use the HTTPS ingress, or set
  `VOXHELM_SECURE_COOKIES=false` deliberately if plain-HTTP LAN login is needed.
- **Framing:** every response sends `X-Frame-Options: DENY`, including on the
  direct backend port. The ops-library Traefik headers middleware sets
  `customFrameOptionsValue: SAMEORIGIN`, which may replace it at the edge.
- **Login throttle:** `VOXHELM_LOGIN_MAX_FAILURES` (default 5) bounds completed
  failures plus in-flight password checks for each client address and username.
  Admission reserves both budgets before authentication; exhausted budgets return
  HTTP 429 without checking a password. Failures expire after
  `VOXHELM_LOGIN_LOCKOUT_SECONDS` (default 900, starting at the first completed
  failure). Pending checks keep their slots across expiry until they finish.
  Success clears username failures, preserves other pending checks and leaves
  address failures intact. Denials for inactive or non-staff accounts (including
  correct passwords) and authentication exceptions settle as failures too.
  The client address is `REMOTE_ADDR`, so behind Traefik all browsers share the
  proxy's address bucket; `X-Forwarded-For` is deliberately not trusted because
  the backend port is also reachable directly. State lives in the default Django
  cache and admission is serialized within one process, matching the single
  uvicorn process. Restart or cache eviction clears state; a shared cache alone
  does not extend this admission guarantee to multiple server processes.
- **Private responses:** producer job submission/status/results, staging and
  artifact responses send `Cache-Control: private, no-store` (plus Django's
  never-cache directives) and `Vary: Authorization`, including errors. Operator
  pages and transcript downloads use the same privacy policy with `Vary: Cookie`,
  including access denials and login redirects. Downloads remain subject to
  producer ownership, staff/operator ownership and artifact exposure checks;
  these are private authenticated resources, not public media URLs. Cache
  middleware must wrap the application in Django's documented order (update
  first, fetch last). Previously cached copies should be purged when adopting
  this policy if caching was enabled; headers cannot erase existing copies.

## OpenAI-Compatible API

The synchronous speech and transcription JSON endpoints require UTF-8 JSON.
Invalid UTF-8 returns HTTP 400 with the same `invalid_request_error` envelope
as malformed JSON, before any download or inference. Authentication runs first.

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
An upload that is interrupted or fails while Voxhelm copies it to local disk
(client disconnect, disk write error) removes its partial temporary file, so a
failed staging request leaves no local audio behind.
Staging registers the upload in the database before it writes the object to the
artifact backend, so a stored staged object always has a row that expiry cleanup
can find. If the object upload or the final row update fails, Voxhelm deletes
the object and the row again and returns the error; if that deletion fails as
well, the row is kept (or written again) as already expired, so the next expiry
cleanup retries the object deletion. While the object is still uploading the
registration carries a 24-hour in-progress expiry, so a concurrent cleanup never
removes it; `VOXHELM_STAGED_INPUT_RETENTION_SECONDS` counts from the finished
upload.
The staging byte ceiling is checked against both the declared size and the
actual chunks before they are written to the temporary file. An over-limit
stream removes its partial file and publishes no object or staging row. An
exact-limit stream is accepted; staging and job metadata use the copied size.
Expiry cleanup re-fetches each candidate and rechecks expiry and its job claim
inside the same database write/row lock used by submission, holding it through
object deletion and row removal. A claim committed after candidate enumeration
keeps the input while its job is active. Expired unclaimed inputs and inputs
whose owning job has finished can be removed. Storage deletion failure keeps
the row for a later cleanup retry.
If the artifact backend, filesystem root, S3 endpoint, or bucket changes after
staging, submitters must stage the media again; Voxhelm rejects `upload_id`
values whose store identity no longer matches the active artifact store.

Job numeric fields reject booleans and numbers that cannot be represented as
finite floats with field-specific HTTP 400 errors before any job is saved.
Ordinary finite numbers and existing synthesis speed bounds remain supported.

Producer and operator artifact downloads stream bounded reads from the artifact's
recorded filesystem/S3 store, including asynchronous reads under ASGI. Closing or
cancelling the response closes the underlying reader once its active read settles. Authorization, private
cache headers and recorded content types still apply. A missing stored object
returns HTTP 404; other supported storage/configuration failures before response
headers return HTTP 503 with a bounded JSON error and no storage details. Rows
are kept for diagnosis and retry. A read failure after headers have been sent
terminates the stream; it cannot change the HTTP status or become a JSON error.

Local Django Tasks execution atomically claims a queued job for its expected
task ID. Replayed, already running, terminal, or stale-task invocations return
`execution: not_claimed` without consuming input or starting inference. The
executor alone sets running/success states; backend polling can resolve failed
unfinished tasks, but cannot revive terminal jobs or take the execution claim.
Success, failure and artifact publication check the same task ownership.

Local artifacts use fresh `jobs/<job-id>/allocation-<uuidhex>/<filename>` keys.
The artifact row switches references in a database transaction, preserving the
previous bytes on metadata failure. Previous objects are queued for deletion
only after commit; failed allocations are compensated through the same recorded
store identity and reference checks. Deletion failures stay queued for the
existing prune command to retry. Other references preserve shared objects.
Retention accepts these allocation keys alongside existing local and remote
attempt keys; download URLs and artifact names stay stable. Empty allocation
directories may remain after object cleanup; retention removes stored objects.

### Private filesystem artifacts

Filesystem keys must be nonempty relative paths without empty, `.` or `..`
segments. All I/O rejects symlinks in keys (including aliases inside the root),
using directory descriptors and no-follow opens. A configured root symlink is
resolved once when the store is created. The configured root and its
ancestors must be trusted administrator-owned paths; untrusted users must not
be able to rename these directories. Retention still lists symlink entries so
its existing ownership checks can refuse unsafe deletion.

New artifact directories use mode 0700 and published objects use mode 0600.
Writes use private sibling temporary files, flush them, and atomically replace
the final object only after success. Readers see the previous complete object
until replacement; failed writes preserve old bytes and attempt temporary cleanup.
If cleanup also fails, a fixed warning is logged and the original write error survives.
Existing directory modes are not changed. These guarantees apply to the
filesystem store; S3 behavior is unchanged. Adoption requires an application or
worker update; no deployment is performed by the change itself.

### Bounded cleanup progress

Upload admission and upload-backed job submission try at most four expired staged
uploads. Each candidate has its own locked transaction and expiry/claim recheck.
A storage or row deletion failure rolls back that candidate, retains its retry row,
and continues the bounded pass so a healthy new upload can proceed. Warnings log
only the upload ID, without storage keys or raw exception details. Expiry removal
is eventual; maintenance retries failures. A failed candidate's `expires_at` moves to
the time of the failure: it stays expired, but later passes reach older healthy
uploads first, so repeatedly failing rows cannot starve the bounded pass.
The maintenance command below also drains expired unclaimed/finished-job uploads.

`prune_job_artifacts --batch-size 100` examines at most 100 rows in each of five
phases (artifacts, pending deletions, attempt prefixes, staged uploads, job metadata),
plus one lookahead row per phase. Results are bounded to this batch. The command saves the cursor in the database automatically, including on reported
storage failures, so existing scheduled invocations resume without new arguments.
An exhausted sweep resets to `{}` to retry failures/refusals and pick up rows inserted
behind the cursor. `--cursor '<json>'` overrides the saved starting point; `{}` restarts
a sweep. Run the command serially; overlapping runs may repeat work or rewind progress.
A dry run starts at `{}`, uses the same limits and never updates saved progress; its cursor is for preview only. Do not pass a dry-run
cursor to a deleting run. Each prefix still lists its own objects for the existing
all-objects allowlist check. A sweep can leave newly eligible job rows for the next sweep.

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
  are never touched, except for jobs of a transcription profile with a backstop
  (below); queued and running jobs are never touched.
- every artifact of an `ad-analysis-v1` job (any kind, exposed or not) is deleted
  once the job finished more than 7 days ago, and its job row with it (the row
  holds the transcript text), whether or not the producer deleted the job itself
  (see [Transcription Profiles](#transcription-profiles)). The same run deletes
  what the job's remote attempts stored below their artifact prefix without a
  completion registering it (a rejected, failed, retried or abandoned attempt;
  the prefix of every profile attempt is recorded at claim time), a staged
  upload the job still holds (a completion whose staged-object deletion failed),
  and the job's Django task result. This backstop is owned by the profile and
  applies even with `VOXHELM_JOB_METADATA_RETENTION_SECONDS=0`.

Deletion is allowlist-based: only an object voxhelm generated for that job is
ever deleted, i.e. a key ending in exactly `jobs/<job_id>/<artifact name>` or
`jobs/<job_id>/attempt-<n>/<artifact name>` or
`jobs/<job_id>/allocation-<uuidhex>/<artifact name>` below the artifact prefix it was
written under (so a later `VOXHELM_ARTIFACT_PREFIX` change does not block
retention), with no empty, `.` or `..` segments; a staged upload's key must end in exactly
`staged-inputs/<upload_id>/<filename>`. On the filesystem backend the resolved real path
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
the database write lock, so overlapping runs are safe. If the cleanup process dies
after object deletion but before commit, the ownership row rolls back; the next
prune treats the missing object as deleted and removes that row. Independently
referenced objects remain protected across this retry.

Remote success is committed before staged-input cleanup. If that cleanup fails,
the committed result and staged ownership remain available: an identical completion
retry from the same worker and lease retries cleanup without republishing the
manifest or changing the finish time. Changed result metadata is refused with a
conflict. `tests/test_remote_recovery_contracts.py` covers these boundaries with
process death, reopened connections and a portable synthetic object store.

After the artifacts (and the queued deletions), the same command deletes old
job metadata: the `Job` row of a terminal job that finished more than
`VOXHELM_JOB_METADATA_RETENTION_SECONDS` ago (default `7776000`, 90 days; the
job's `finished_at`, else its last update; `0` disables job metadata pruning).
A job is only deleted once it owns nothing anymore: no artifact row of any kind
(so intermediates must have been pruned first, and a job whose intermediate was
refused, failed to delete or is not yet past source retention keeps its row),
no queued `PendingArtifactDeletion`, no recorded remote attempt prefix still to
clean and no staged upload still claimed by it. Transcript, speaker-sidecar and speech artifacts are kept indefinitely and are
served through their job, so a job that still has them keeps its row. A
profile job past its backstop loses those artifacts in the same run and its row
then goes regardless of the metadata retention. Queued and running jobs are
never deleted. Each batch re-checks these conditions under
the database write lock. Deleting a job also frees its `task_ref`, so a later
submission with the same `task_ref` creates a new job. `--dry-run` lists the
job rows a real run would delete, counting the artifacts and queued deletions
that run would remove first as already gone.

Nothing schedules the command yet: run it from cron or a systemd timer (hourly
is enough) on the control plane, with the same environment as the service.

Current scope note: batch staged uploads are audio-only in this slice. URL
audio and URL video keep working on the existing path. Uploaded video and true
service-owned chunk splitting/stitching are still explicitly deferred.

## Transcription Profiles

A batch transcription job can name a profile instead of a backend and model. A
profile pins every setting that changes its output, never falls back to another
engine, and is versioned: any change to its settings is a new profile name.
`ad-analysis-v1` (the podcast server-side ad analysis) runs Parakeet TDT through
`parakeet-mlx` with 30 s chunks, 5 s overlap, full attention and no
`--max-duration`, whatever the deployment's `VOXHELM_PARAKEET_CHUNK_SECONDS`,
`VOXHELM_PARAKEET_OVERLAP_SECONDS` and `VOXHELM_PARAKEET_MAX_SENTENCE_SECONDS`
say (it uses `VOXHELM_PARAKEET_BIN`, `VOXHELM_PARAKEET_MODEL_DIR`,
`VOXHELM_PARAKEET_MODEL_NAME` and `VOXHELM_PARAKEET_TIMEOUT_SECONDS`). The
transcript is not sanitized. For a German job (`language_hint` `de`) the
profile then runs the German drift check (D-28): it counts English and German
function words in every 30 s grid window `[30k, 30k + 30)` of the Parakeet
word starts, flags a window with at least 4 English ones that are at least
10 % of both, re-transcribes each flagged window with whisper large-v3-turbo
(window transcription, below), and keeps the retry only when whisper detects
German with p >= 0.9; then every Parakeet word starting in the window is
replaced by the retry's words. A job may also carry the edges of the episode's
sponsor chapters (`chapter_marks_ms`, D-29): the profile then transcribes the
`turbo-window-v1` word window around each mark with the same turbo model
(chapter windows, below). The profile needs its runtimes on every host
(Parakeet, `whisper-cli` with the pinned turbo model, and ffprobe), whatever the
job's language.

```bash
curl -X POST http://127.0.0.1:8000/v1/jobs \
  -H "Authorization: Bearer replace-me" \
  -H "Content-Type: application/json" \
  -d '{
    "job_type": "transcribe",
    "profile": "ad-analysis-v1",
    "language_hint": "de",
    "input": {"kind": "upload", "upload_id": "replace-me"},
    "output": {"formats": ["json"]},
    "chapter_marks_ms": [1591000, 1701741],
    "task_ref": "smart-skip-<sha256>"
  }'
```

- `language_hint` is the feed language. Parakeet takes no language argument;
  the hint is normalized (`de-DE` -> `de`), recorded with the job, and a hint
  outside Parakeet's 25 languages is rejected with 400. A profile job takes no
  `language`, no `model` other than `auto`, and no diarization.
- Besides the requested transcript formats the job always writes
  `words.json` (`result.artifacts.words`):
  `{"schema_version": 1, "profile": ..., "identity": {...}, "words": [{"w", "s", "e", "c"}, ...], "drift": {...}}`
  with one entry per word (a token starting with a space starts a word; other
  tokens join it), start and end in seconds and `c` the mean token confidence,
  all rounded to 3 decimals. Words of an accepted turbo retry have `c: null`.
  The words carry the drift check's splice; the transcript formats stay
  Parakeet's sentences.
- `words.json` also holds the drift check's records, `"drift"`:
  `{"rule": "fw-en4-share10-v1", "checked": true, "language": "de", "reason": null,
  "summary": {"windows", "flagged", "accepted", "rejected", "retry_seconds"},
  "windows": [{"t", "en", "de", "flagged", "words_before", "words_after"}, ...]}`,
  where a flagged window adds `language`, `p`, `accepted`, `retry_words` and
  `retry_seconds`. For other languages `checked` is `false`, `reason` says why
  and `windows` is empty. They are part of the words artifact, so `DELETE` and
  the backstop remove them with it.
- `chapter_marks_ms` (optional, only with a profile): whole milliseconds of file
  time, 1 to 256 of them, each in `[0, 24 h]`, strictly increasing; anything
  else is rejected with 400, and the marks are part of the `task_ref` identity.
  A mark after the media's end (ffprobe's duration, rounded to milliseconds)
  fails the job before Parakeet runs. The job then also writes
  `chapter_windows.json` (`result.artifacts.chapter_windows`):
  `{"schema_version": 1, "profile", "rule": "turbo-window-v1", "identity": {...}, "duration_ms",
  "summary": {"windows", "ok", "seconds"}, "windows": [{"mark_ms", "ok", "clip_start", "shift",
  "exit_codes", "seconds", "words": [{"w", "s", "e"}, ...]}, ...]}`, one window
  per mark in order, words in file seconds; `ok: false` (no words) when
  whisper.cpp failed on all five clips. `result.metadata.chapter_windows` holds
  its identity and summary. Without marks nothing changes. A remote worker gets
  the marks with the claim, and its completion must include an exposed
  `chapter_windows.json` and the pinned chapter windows identity (only its
  whisper.cpp version is taken), or it is refused.
- `result.metadata.profile` and the artifact's `identity` record the engine:
  profile name, model name, the model's repo, commit and weights checksum (from
  the deployment's `VOXHELM_MODEL_REVISION` marker in the model directory, or a
  Hugging Face snapshot directory name), the `parakeet-mlx` and `mlx` versions
  of the CLI's environment, the profile settings, the drift check (`drift`:
  rule, thresholds, function-word list hash, acceptance, and `retry` with the
  whisper.cpp version, the turbo model name, its SHA-256 and the decoding
  settings) and the language hint.
- No fallback and no silent skip: a local job without the Parakeet CLI or
  model, or without `whisper-cli` or the pinned turbo model, fails before
  Parakeet runs, and a window retry that fails fails the job; in `remote_pull`
  mode a profile job is only claimed by a worker that advertises the profile
  (`capabilities.profiles`), so it waits until one does. A worker that loses
  either model after claiming fails the attempt as retryable and the job goes
  back to the queue.
- `DELETE /v1/jobs/{id}` (producer token): removes a finished profile job's
  artifacts, their stored objects, queued object deletions, every object its
  remote attempts stored below their artifact prefix (also those no completion
  registered), a staged upload it still holds, its Django task result, and
  finally the job row, through the same allowlist, reference checks and write
  lock as `prune_job_artifacts`. `204` when everything is gone; `404` for an
  unknown job or another producer's job (so also for a repeated delete); `409`
  for a job without a profile, a job that has not finished, or one holding an
  artifact, attempt object or staged upload outside the allowlist (nothing is
  deleted); `503` when a stored object could not be deleted or listed (what was
  deleted stays deleted, retry the request).
- A profile job's task result (`django_tasks_db`) records only
  `{"job_id", "profile"}`, never the transcript.
- Backstop: `prune_job_artifacts` deletes every artifact and the row of an
  `ad-analysis-v1` job 7 days after it finished (see the retention section).

See `specs/2026-10-16_ad_analysis_profile.md`.

### Chapter windows (whisper turbo, `turbo-window-v1`)

For a profile job with `chapter_marks_ms`,
`transcriptions.chapter_windows.run_chapter_windows` transcribes, after the drift
check and inside the same admission, the 120 s clip
`[max(0, t - 60 + shift), +120 s)` around each mark `t` exactly as the ad
research's `asrwin120.py` did: ffmpeg to 16 kHz mono WAV, `whisper-cli` with the
pinned turbo model (`VOXHELM_WINDOW_MODEL`, the drift check's file), English,
DTW token timestamps (`-dtw large.v3.turbo`), one word per segment
(`-ml 1 -sow`), the prompt `Hello, and welcome back. Thanks!`, plain JSON, and
whisper.cpp's default decoding passed explicitly (`-t 4 -p 1 -bs 5 -bo 5
-tp 0.00 -tpi 0.20`). When whisper.cpp exits with an error (a DTW abort) the
clip is shifted by 0.7, -0.7, 1.9 and -1.9 s in turn; the pinned model is
checked again after every failed clip, so a model that went away fails the job
instead of producing untranscribed windows. Words are every non-blank segment's
stripped text with the clip start plus its offsets, rounded to 3 decimals.
Offsets of kept words must be integers with `0 <= from <= to <= 121 s`: the
120 s clip plus a 1 s tolerance, because whisper.cpp's DTW timestamps can end
slightly past the last decoded sample (the G6 research windows end at most
100 ms past the clip). Impossible offsets, like other malformed output, fail
the run, as do whisper.cpp JSON files over 8 MiB. The JSON is opened without
following a symlink or blocking and must be a regular file: for the drift-check
windows anything else fails the run; for chapter windows a FIFO or directory at
the output path counts as a failed clip (shifted and retried, then an untranscribed
window) and a symlink fails the run. On
the 26 episodes of the podcast gate G6 the 102 windows are identical, word for
word and to the millisecond, to the research's. See
`specs/2026-10-17_chapter_windows.md`.

### Window transcription (whisper turbo)

`transcriptions.windows.transcribe_window(audio_path, t)` re-transcribes the
30 s window `[t, t + 30)` of a local source file with whisper large-v3-turbo,
for the profile's German drift check (not an HTTP route, D-27; called by
`ad-analysis-v1`, D-28). It decodes the clip
`[max(0, t - 1), +32 s)` with ffmpeg to 16 kHz mono, runs `whisper-cli` with
the language detected on the clip (`-l auto`, greedy best-of 2, temperature 0
with 0.2 increments, 4 threads) and returns the words starting in the window
(`[{"w", "s", "e"}]`, file time, 3 decimals), the detected language and its
probability, and the identity (whisper.cpp version, model name and SHA-256,
settings). The model is `VOXHELM_WINDOW_MODEL`; a file whose SHA-256 is not
the pinned `1fc70f77...bc69` is refused (`BackendUnavailableError`), and no
other model is ever used. The clip lives in a temporary directory of the call;
nothing is stored. `evals/window_retry_compare.py` compares it with the
research retries (identical on 40 windows, see
`specs/2026-10-16_window_transcription.md`); `evals/drift_check_g2.py` runs
gate G2 of the whole drift check on the 22 German research episodes
(`specs/2026-10-16_drift_check.md`).

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
labels such as `Speaker 1` and `Speaker 2`. Turns with non-finite start or end
times are dropped. Alignment uses an interval index, so its work grows with the
number of overlapping turns per segment rather than segments × turns. If the expected speaker count is
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
- for known-speaker jobs (`diarization.strategy=pyannote_known_speaker`), also
  make sure the token can download the embedding model, by default
  `pyannote/wespeaker-voxceleb-resnet34-LM`; accept its terms if Hugging Face
  asks. Embedding extraction loads it with the same token and fails when no
  token is set

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
peer that can run a current candidate and has handled strictly fewer recent claims,
so two compatible workers converge on a
roughly even split. With no eligible peer (the other worker is offline, busy,
incompatible, or disabled) a worker never defers, so a lone worker still claims everything and the
fleet self-heals. Tunable via `VOXHELM_REMOTE_WORKER_BALANCE_ENABLED` (default
`true`), `VOXHELM_REMOTE_WORKER_BALANCE_WINDOW_SECONDS` (recent-claim comparison
window, default `3600`), and `VOXHELM_REMOTE_WORKER_BALANCE_PEER_FRESH_SECONDS`
(how recently a peer must have heartbeated to count, default `30`). This is
control-plane-only behavior; workers need no changes. Compatible fresh peers
share one grouped active/recent Job count query instead of per-peer counts. Capacity checks and new
lease expiry use the admission time sampled after acquiring the worker write lock,
so lock contention does not consume the issued lease.

Claim scans are bounded and priority-correct. Each scan hydrates and validates at
most 100 queued/expired jobs in priority/creation-time/job-ID order; a poll runs
a second scan only after losing a conditional-claim race. A worker-owned cursor
(`Worker.claim_scan_cursor`) lets a sweep continue across polls and API processes
past a large backlog the worker cannot run; it moves only across jobs the sweep
rejected before its first eligible one, so a job that was found but not leased
(capacity, fairness, a lost race) is found again. Before resuming, one existence
query checks whether any claimable job at or before the cursor was created,
requeued or otherwise updated, or had its lease expire, since the sweep started
(with a 30-second grace for writers that stamp `updated_at` before waiting for
the SQLite write lock and commit later); if so the scan restarts from the top,
so a new high-priority job is never hidden behind the cursor. A change of the
worker's advertised capabilities or a finished sweep also restarts from the top;
sweeps have no age limit, so a quiet queue of any size is swept to its end, and
changes no row timestamp records (server configuration) are seen after one full
sweep. Each claimable-job change at or before the cursor restarts the sweep, so
while such changes keep arriving faster than a sweep takes, scans favour the head
of the queue (priority first). Only the Python-side hydration and validation is
bounded; the database still evaluates and orders the claimable set (served by the
`(execution_mode, state, priority, created_at)` index). An unchanged cursor is
not rewritten, so idle polls do not write.

Configure `VOXHELM_WORKER_BASE_URL` (or `VOXHELM_BASE_URL`) with the canonical
worker API origin (the final
HTTPS URL when using ingress). Worker control-plane requests refuse all HTTP
redirects, including redirects on the same origin, and report only the status
and a configuration hint. A redirect never forwards the worker bearer token or
prints the redirect target or response body. Media URL redirects still use the
separate allowlisted media fetcher.

Worker control requests use `--http-timeout-seconds` (default `60`) as one
elapsed budget for connection, response headers and streamed body consumption;
slow progress does not reset it. Successful responses are limited to 1 MiB,
independent of Content-Length. HTTP error bodies are never read; their status is
preserved for retry decisions. Responses and clients close
on success, failure or expiry; an OS DNS lookup may finish later without opening
a connection. Workers request identity encoding and read raw bytes, so compressed
responses cannot bypass the byte cap. Error bodies and remote reason phrases are
never echoed in exceptions/logs; HTTP errors report only the status and its
standard description; transport failures include only the exception class name.
TLS verification retains the standard-library default trust context.
Oversized, timed-out or invalid claim responses cannot start inference. The success
cap applies to every control endpoint, including completion/failure responses that
echo transcript text. A completion may already be recorded by the server before
an oversized acknowledgement is rejected; check the server job status before
retrying that job. Update/restart remote workers to adopt these limits.

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

Persisted transcription request fingerprints are SHA-256 of compact JSON with
recursively sorted keys and literal UTF-8 (no ASCII escaping). Identity covers
job type, backend, model, language (empty becomes null), input, sorted output
formats and diarization (absent becomes `{"enabled": false}`). Upload identity
uses only `kind` and `upload_id`; filename, size and staging/storage metadata do
not affect reuse. Priority, context and `task_ref` are outside the digest (reuse
is already scoped to the producer and `task_ref`). Absent/empty `profile` and
`chapter_marks_ms` keys are omitted for compatibility with historical digests;
present profiles and ordered chapter marks are identity. Synthesis retains its
looser rule: only job type enters the digest, so a non-failed job of that type
with the same producer and `task_ref` can be reused despite payload changes.
`uv run pytest tests/test_fingerprints.py` protects these persisted bytes with
handwritten JSON/golden digests and a synthetic matcher-equivalence matrix.

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
Known-speaker jobs routed to Parakeet (`VOXHELM_KNOWN_SPEAKER_STT_BACKEND` on the
control plane) are the exception: the target is chosen per claiming worker as
the first of Parakeet, whisper.cpp and the configured chain that the worker
advertises, so such a job never waits for a Parakeet worker, and the claim adds
`fallbacks` (`[{"backend": ..., "model": ...}]`, the remaining advertised
entries in order) that the worker tries when the claimed backend turns out to
be unavailable. Workers advertise `parakeet` only while `VOXHELM_PARAKEET_BIN`
resolves and `VOXHELM_PARAKEET_MODEL_DIR` holds `config.json` and
`model.safetensors`, re-checked on every heartbeat, and log the CLI version at
startup; they also advertise `whispercpp` whenever its binary and model are
present, even outside their configured chain, so the fallback keeps word
timings. Routed jobs with a language Parakeet does not support skip it. Remote STT runs are admitted through the non-interactive lane of the
lane scheduler, like local ones.
Workers advertise transcription profiles (`"profiles": ["ad-analysis-v1"]`) by
the same per-heartbeat Parakeet check, and only while `whisper-cli` and the
pinned turbo model (`VOXHELM_WINDOW_MODEL`) also resolve; a profile job is claimed only by a worker
that lists its profile, its claim carries `profile` and no `fallbacks`, and the
completion must include `words.json` (kind `words`, exposed, `application/json`)
and `result_metadata.profile` with the engine identity, which the control plane
reduces to the known fields; its `drift` part must equal the control plane's
own drift rule and pinned turbo model (only the worker's whisper.cpp version
is taken from it), otherwise the completion is refused.
Heartbeat storage contract: `hostname` is optional and trimmed, with at most
255 Unicode characters. `concurrency` is an integer from 1 through 2147483647
(booleans are rejected); omitted or null values default to 1. `capabilities` is
an object containing finite JSON values with at most 100 nested object/array
levels (the outer object counts as level 1); `running_job_ids` is a list of strings
whose entries are trimmed and empty entries omitted. Omitted or null objects
and lists default to empty. Hostnames, JSON string values and object keys, and
running-job IDs must be valid Unicode without NUL or unpaired surrogates.
Invalid reports return HTTP 400 (`invalid_request_error`) before creating or
updating a worker, leaving liveness, capabilities and all other stored fields
unchanged. Valid multilingual text and emoji are preserved.

Disabled workers are rejected before heartbeat state is updated.
Disabling a worker stops new claims but does not block completion, failure, or
lease heartbeat for a job the worker already owns. Worker completion accepts
only the currently assigned worker and lease token.
Completions must include a non-exposed job-owned source artifact plus the
requested transcript artifacts. Artifacts must be reported under the claimed
attempt prefix, for example `voxhelm/jobs/<job_id>/attempt-1/transcript.txt`;
Voxhelm verifies that each reported object exists in the configured artifact
store and that its stored size matches the manifest before marking the job
succeeded. A manifest lists at most 32 artifacts, checked before any store
access, and the per-object store checks share a 30-second cumulative budget
(checked before each request; one request is bounded by the store client's own
timeouts), after which the completion is refused without settling the job; names, storage keys and content types must fit their columns (255, 512
and 255 characters, no NUL), `size_bytes` must fit a signed 64-bit integer, and no
two artifacts may name the same object (keys are compared after collapsing `//`
and `.` segments). The producer still downloads winning artifacts through
`GET /v1/jobs/<job_id>/artifacts/<name>`. Transcript and speaker-sidecar
artifacts must use the exact expected MIME types before Voxhelm exposes them.

Known-speaker jobs are claimable only by workers advertising the required
pyannote/wespeaker speaker-sidecar capability. Known-speaker reference URLs are
checked against `VOXHELM_ALLOWED_URL_HOSTS` at submission and again before a
remote claim is handed out. Uploaded known-speaker reference clips
(`audio.kind=upload`) are rejected with 400 at submission in every execution
mode; use URL reference audio. Jobs stored before that check still fail before
remote claim. Completion metadata stores only type-checked scalar worker fields plus
server-derived source, worker, attempt, and request metadata. URL completions
keep the server-owned job input URL as `source_url` and ignore worker-supplied
redirect URLs. URL-shaped strings, nested values, negative or non-finite
timings/summary metrics, and private known-speaker reference
URLs/ranges are not echoed through producer-visible job metadata. Speaker
sidecars are accepted only for known-speaker jobs.

Run a manual worker on a trusted macOS host with an env file containing the
worker token (`VOXHELM_WORKER_TOKEN`), Voxhelm URL, shared artifact credentials, local STT/model cache
settings, `VOXHELM_ALLOWED_URL_HOSTS`, and optional Hugging Face token. The
`caffeinate` wrapper keeps the machine awake only while the worker command is
running:

```bash
/usr/bin/caffeinate -i -m -s -- uvx --from ".[diarization]" voxhelm-remote-worker \
  --env-file "$HOME/Library/Application Support/voxhelm-remote-worker/worker.env" \
  --once
```

The env file holds the worker token and the artifact credentials, so keep it
readable only by the account that runs the worker:

```bash
chmod 600 "$HOME/Library/Application Support/voxhelm-remote-worker/worker.env"
```

On a new machine without a checkout, run the published PyPI package instead
of the local source; pin the release the control plane expects:

```bash
/usr/bin/caffeinate -i -m -s -- uvx --from "voxhelm[diarization]==<version>" \
  voxhelm-remote-worker \
  --env-file "$HOME/Library/Application Support/voxhelm-remote-worker/worker.env" \
  --once
```

The legacy `--token` flag still works and takes precedence over the environment,
but is deprecated and prints a warning to stderr because command-line arguments
can expose credentials in process listings. Use `VOXHELM_WORKER_TOKEN` in the
environment or the env file shown above instead.

From this checkout, `just worker-once` runs the same one-job command. Override
`VOXHELM_WORKER_ENV_FILE` for a non-default env-file path. It defaults to
`$HOME/Library/Application Support/voxhelm-remote-worker/worker.env`.
Override `VOXHELM_WORKER_UVX_SOURCE` for a package or git source instead of
the local checkout, and include the `diarization` extra when processing
speaker-label jobs. Use `just worker-loop` only when you want to keep polling
manually until `Ctrl-C`.

The server orders same-lease heartbeat publication by sampled time: delayed or
equal-time beats return the current lease without replacing newer progress, and
extensions never shorten a stored expiry. Rotated or expired leases still fail.

Lease heartbeats own their active HTTP request: leaving a claim cancels the
request, closes its streams and joins the heartbeat thread within five seconds.
Heartbeats are serialized: a phase update waits for an already active beat
(up to the configured HTTP budget). No heartbeat starts after shutdown. A definitive HTTP 401, 403, 404 or 409
revokes local ownership; subsequent phases and publication stop cooperatively,
without completion or failure reporting for that revoked attempt. Transient
errors (including 5xx) leave ownership intact. An in-progress model call cannot
be interrupted; ownership is checked again before the next phase.

`--once` exits zero for no work or a completed job, nonzero for failed
processing (even when failure reporting succeeds), a failed control request,
or a failed report. Continuous mode keeps polling after reported job failures.
A failed poll waits before the next attempt. The first wait is the poll
interval, each further consecutive failure doubles it up to 60 seconds (or the
poll interval when that is longer), and the first successful poll resets it.

An HTTP 403 on the worker heartbeat or claim usually means the worker was
disabled, for example during a deploy drain. It logs
`control plane rejected worker (HTTP 403, worker disabled?); retrying in Ns`
without a traceback. Once the worker is re-enabled, it logs
`control plane accepted worker again`. Every other failure logs its traceback,
as before.

`--poll-seconds`, `--http-timeout-seconds`, and `--heartbeat-interval-seconds`
accept integers from 1 to 3600 seconds and are checked before runtime probes or
client construction. Server `poll_after_seconds` hints use the same integer
range; invalid hints fail the poll before a claim. Direct Python configuration
also permits finite positive fractional seconds up to 3600, excluding booleans.

The worker defaults to one active job, periodically heartbeats the leased job
while local inference runs, uploads the source, optional extracted audio,
requested transcript artifacts, and known-speaker `transcript.speakers.json`
sidecar under the claimed attempt prefix, then posts the completion manifest.

Operational note: the application endpoints still require worker auth, and the
macmini/Traefik edge also blocks `/v1/internal/*` on public routes unless a
deliberately private worker route is configured (ops-library
`voxhelm_ingress_deploy`, `voxhelm_ingress_internal_api_enabled`).

Current implementation status: the studio control plane runs in `remote_pull`
mode, and `voxhelm` is published to PyPI and installable as a `uv` tool
(`uv tool install "voxhelm[diarization]"`). Two `voxhelm-remote-worker`
instances, on `atlas` and on `studio`, run a pinned PyPI release as the system
LaunchDaemon `de.wersdoerfer.voxhelm-remote-worker` (`RunAtLoad`, `KeepAlive`),
installed by the ops-library `voxhelm_remote_worker_deploy` role. The
macmini/Traefik edge blocks `/v1/internal/*`. The production python-podcast
proof ([BACKLOG.md](BACKLOG.md)) remains follow-up work.

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

The Wyoming protocol has no authentication, and the sidecar binds
`0.0.0.0:10300` by default. Expose the port only to the LAN or to Home
Assistant, never to the internet. To bound memory use on the inference host,
the sidecar enforces these input limits per connection:

- A complete event (header, data and payload together) has a fixed 30-second
  elapsed read deadline, including idle waits and trickled input. Expiry returns
  `code="event_timeout"` and closes the connection.
- Consumed synthesis, transcription and audio fields are type-checked before
  state changes or inference. Missing required text/audio fields, malformed voice
  objects and invalid scalar types return one `code="invalid_event"` error and
  close the connection. Optional transcription model/language fields may be omitted.
- Audio must start with `AudioStart`; each chunk must match its rate, sample
  width and channel count and contain whole frames. Supported PCM geometry is
  8000–192000 Hz, 1, 2 or 4 bytes per sample (the bundled
  converter has no 24-bit support), mono or stereo. Unsupported geometry,
  mismatches and chunks before a start return `code="invalid_event"` and close
  the connection before conversion or inference. Each `Transcribe` or
  `AudioStart` discards prior audio and resets resampling history; history is
  preserved between chunks of the same utterance.
- Every event is checked before its body is read. The sidecar reads events
  through its own bounded reader instead of the `wyoming` library's, which
  would buffer whatever `data_length` / `payload_length` a client declares.
  - `VOXHELM_WYOMING_MAX_EVENT_DATA_BYTES` (default `262144`, 256 KiB, must be
    positive) caps the declared `data_length` of one event. The default leaves
    room for a `Synthesize` request at the default `VOXHELM_TTS_MAX_INPUT_CHARS`;
    raise it together with that limit if needed.
  - `VOXHELM_WYOMING_MAX_EVENT_PAYLOAD_BYTES` (must be positive when set) caps
    the declared `payload_length` of one event. Unset or empty, it equals the
    whole STT audio cap in converted bytes
    (`VOXHELM_WYOMING_STT_MAX_AUDIO_SECONDS` × 32000, so 3840000 bytes for
    120 s).
  - An event over either limit, or with a header line over 64 KiB, is answered
    with an `error` event (`code="event_too_large"`). A malformed event (a
    negative or non-integer length, a header that is not a JSON object with a
    `type`, or event data that is not a JSON object) is answered with
    `code="invalid_event"`. In both cases the connection is closed. The length
    and header checks run before any body byte is read; event data is read
    (within the data limit) before it is parsed and checked.

- `VOXHELM_WYOMING_STT_MAX_AUDIO_SECONDS` (default `120`, must be positive) caps
  the audio buffered for one transcription, measured after conversion to
  16 kHz, 16-bit, mono PCM. Before conversion, cumulative whole input frames
  bound the converted output using the sample-rate ratio rounded up to a whole
  output frame. This admits the exact duration limit while conservatively
  rejecting over-budget resampling before allocation. A chunk over the cap is
  answered with a Wyoming `error` event (`code="audio_too_long"`), the buffer is
  discarded, and the connection is closed without transcribing.
- `Synthesize` requests use the same `VOXHELM_TTS_MAX_INPUT_CHARS` cap (default
  `5000`, counted after trimming surrounding whitespace) as
  `POST /v1/audio/speech`. Longer text is answered with an `error` event
  (`code="text_too_long"`) and the connection is closed without synthesizing.
- `VOXHELM_WYOMING_MAX_CONNECTIONS` (default `16`, must be positive) caps open
  client connections. A connection over the cap gets one `error` event
  (`code="server_busy"`) and is closed before any event is read.
- `VOXHELM_WYOMING_MAX_INFERENCE_THREADS` (default `4`, must be positive) caps
  synthesis and transcription worker threads, checked before a thread starts.
  A thread keeps its slot until it actually finishes, even after its client
  disconnected or the request was cancelled, so abandoned inference cannot pile
  up. A request over the cap gets `code="busy"` and the connection is closed.
- Each outbound event must be accepted by the client within 30 seconds; a client
  that stops reading has its connection aborted.
- A synthesis result produced after its request was cancelled (client gone or
  server shutdown) is deleted by the worker thread when it finishes.
- Failure events (`synthesis_failed`, `transcription_failed`) carry a fixed
  public text; exception details (paths, decoder output) are only logged.
- On stop or cancellation the sidecar aborts, cancels and awaits every connection
  handler and stops listening; each handler releases its transport even if
  cleanup fails. Everything after a successful listener start (including the
  startup log line) runs under that shutdown. The sidecar installs SIGINT/SIGTERM
  handlers on its event loop and removes them when it returns (unless a handler
  registered after its own, such as a second sidecar on the same loop, has taken
  over), so an embedding loop can start it again; asyncio keeps one handler per
  signal, so a handler the embedding application registered for those signals
  before is replaced while the sidecar runs and is not restored.
- Inference admission refuses rather than queues: a request over
  `VOXHELM_WYOMING_MAX_INFERENCE_THREADS` (default `4`) fails immediately with
  `code="busy"`. The lane scheduler still serializes admitted backend work.

Set `VOXHELM_STT_DEBUG_LOGGING=true` when tuning the HA path. Voxhelm will emit
one structured `stt_debug` log line per transcription with the input audio
shape, requested and resolved backend/model/language, transcript preview, and
latency. Debug mode intentionally includes content: trimmed prompt previews of at
most 160 characters and transcript previews of at most 300 characters, followed
by `...` when truncated. A nonblank raw transcript is included only when it
differs from the normalized transcript, with the same 300-character preview limit.
The default (`false`) emits no `stt_debug` record and skips HTTP audio metadata
inspection. Unreadable, unsupported or truncated WAV metadata is diagnostic-only
and does not discard a completed transcription; temporary input cleanup still runs.
Synthetic coverage of these boundaries: `uv run pytest tests/test_observability.py`.

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
- Holder publication and waiter removal run under the control lock. A failed
  publication removes only that attempt's holder; admission logging cannot revoke
  a returned ticket. An operational state read error refuses admission and leaves
  existing holders untouched. Missing or malformed historical records retain the
  recovery policy below; fix filesystem access before retrying an I/O failure.
- Queued cancellation is checked every 50 ms while waiting for the control lock,
  including initial waiter registration. Cancellation removes only the attempt's
  waiter without waiting for that lock; no holder is published for it.
- Each new ticket holds an OS file lock on its holder record until release.
  Process death releases that lock, so a later admission reclaims the abandoned
  holder even if its PID has been reused. Keep the returned ticket alive until
  inference ends and call `release` (normally via `admit_local_inference`).
  Live holders retain capacity regardless of age. `VOXHELM_LANE_SCHEDULER_STALE_SECONDS`
  (default `1800`) is a warning threshold only: each scheduler instance logs an
  old live holder once while its file remains present (normally once per admission
  attempt, since callers create a scheduler for each attempt). A stuck live process
  must be stopped by the operator before its capacity can be reclaimed; deleting its
  holder file while inference runs can exceed the capacity bound.
  Older records without the lock marker, including legacy `holder.json`, still
  use PID liveness to protect in-flight work during upgrades; PID reuse can keep
  those records reserved. A leaked ticket inside a live daemon also retains its
  lock. To recover such an orphan, stop all local inference consumers
  (HTTP, Tasks workers and Wyoming), verify their inference subprocesses have also
  exited, and then remove the orphan holder files from the configured scheduler
  directory before restarting. Do not clear holder files on a running service.
  Restart services/workers when adopting this recovery policy.
- `VOXHELM_LANE_SCHEDULER_ENABLED=false` disables all gating. Setting
  `0` interactive and `1` non-interactive slots restores the original
  single-slot behaviour without a code change.
- Remote pull workers (`voxhelm-remote-worker`, including the one on `studio`)
  run with the scheduler disabled and do not take slots (decision D-25): each
  processes one job at a time, and sharing the single non-interactive slot would
  make sync requests wait behind whole podcast jobs.

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

### Media processing budgets

`VOXHELM_URL_FETCH_TIMEOUT_SECONDS` (default `60`, positive integer) is one
elapsed budget for DNS, connection attempts, redirects, response headers and
body download of a sync URL input (`POST /v1/audio/transcriptions` and the
operator UI's sync route). Slow progress does not reset it. Expiry interrupts
network sockets and removes partial temporary media; an OS DNS lookup may
finish later in a daemon thread, which cannot open a connection or create a
file. Sync URL inputs are limited to `VOXHELM_MAX_URL_DOWNLOAD_BYTES`
(default 128 MiB) and are fetched once. Any URL fetch whose body ends before
its announced `Content-Length` fails instead of passing a truncated file on.

Batch media, meaning the URL input of a batch job and known-speaker reference
audio, whether fetched by the Django Tasks worker or by a remote worker,
uses `VOXHELM_BATCH_URL_FETCH_TIMEOUT_SECONDS` (default `600`, positive
integer) as the same kind of elapsed budget per try, and is limited to
`VOXHELM_BATCH_MAX_DOWNLOAD_BYTES` (default 512 MiB). The sync size limit does
not apply to it. A try that fails transiently (the elapsed budget, a DNS,
connection or read error, or HTTP 408, 425, 429, 500, 502, 503 or 504) is
retried (as is a body shorter than its `Content-Length`) inside the same
job attempt, up to `VOXHELM_BATCH_URL_FETCH_ATTEMPTS`
tries in total (default `3`, positive integer), after 5 s, then 10 s, doubling
up to 30 s. Each retry is logged at warning level with the host, never the
full URL. Allowlist and resolved-address rejections, too many redirects,
an unsupported media type, the size limit, a certificate verification failure
and other HTTP statuses fail at once. Only a download that fails on every try
fails the job attempt; a remote worker reports it as retryable as before. A
remote worker keeps heartbeating its lease while it fetches and retries, so a
batch fetch can hold a claim for up to about
`attempts × budget` plus the backoff (about 30 minutes with the defaults).

`VOXHELM_AUDIO_DECODE_TIMEOUT_SECONDS` (default `900`, positive integer) bounds
each ffmpeg decode for whisper.cpp normalization, pyannote diarization and
known-speaker reference/segment audio on the server and remote workers. Expiry
kills and reaps ffmpeg and reports a decode-deadline message using the existing
normalization or speaker error type.
Increase this budget for long episodes on slower hosts. These decode calls and speech
export commands use `-nostdin` to avoid interactive input.

`VOXHELM_VIDEO_EXTRACTION_TIMEOUT_SECONDS` (default `900`, positive integer)
bounds ffmpeg video-to-audio processing on the server and remote workers.
Expiry kills and reaps ffmpeg. Every failed extraction removes its reserved WAV;
a successful extraction leaves the WAV for its caller to consume and remove.

Piper models are cached per worker process by resolved model/config paths and
file metadata (device, inode, size, nanosecond modification and change times),
with at most eight least-recently-used entries. Unchanged artifacts reuse the
loaded voice regardless of its registry key; replacing or modifying either file
reloads it on the next request. Publish complete model/config pairs while the
worker is stopped to avoid requests observing a partially updated pair. Restart
workers after changes on filesystems that do not reliably report metadata changes.

Kokoro ONNX sessions and voicepack styles use the same canonical path/file metadata
identity as Piper, with separate caches of at most eight entries each. Replacing a
model or voicepack at the same path reloads it on the next synthesis; unchanged files
reuse the loaded objects. Cache loading/publication is serialized separately from
inference, and NumPy voicepack archives close after loading, including on errors.
Publish complete model/voicepack pairs while workers are stopped; restart workers
when filesystem metadata changes cannot be relied upon.

Known-speaker references share one downloaded source per exact URL within a job
attempt, including the original job URL. Source ranges are decoded by ffmpeg's seek
and duration options before PCM allocation; only interval samples reach reference
window extraction. Output seeking preserves codec priming/resampling and may decode
and discard preceding frames; this bounds PCM memory rather than decode CPU time.
See [reference ownership and decoding](specs/known-speaker-diarization.md#reference-source-lifetime-and-interval-decoding).

MP3/OGG speech export runs ffmpeg in its own process group with a positive
integer deadline configured by `VOXHELM_AUDIO_EXPORT_TIMEOUT_SECONDS` (default
`300` seconds). The deadline covers the whole run, including descendants that
keep its output pipes open. stdout is discarded and stderr is drained with a
64 KiB ceiling; overflow fails the export. On every exit path the whole process
group is killed and the converter reaped, so no descendant outlives the export.
Timeout, startup, diagnostic-overflow and nonzero-exit failures remove the owned
output and raise a synthesis error with a fixed message (HTTP 500 `server_error`
on the sync API, failed batch job); at most 2 KiB of converter output is logged,
never returned. Callers still own and clean up the source WAV. WAV passthrough
does not invoke ffmpeg or use this deadline.

`POST /v1/audio/speech` admission and responses:

- The JSON body is refused as invalid JSON (HTTP 400) for `NaN`/`Infinity`,
  integers beyond Python's digit limit and nesting beyond the parser's limit.
  `speed` must be a finite JSON number; booleans and values that overflow a float
  are refused with the `speed` field error.
- Backend, converter and filesystem failures return HTTP 500 with the fixed
  message `Speech synthesis failed.`; details are logged only. Input errors stay
  client-facing as HTTP 400 `invalid_request_error`: a requested voice that is not
  installed (with the available voices) and Kokoro text without supported tokens.
  The Wyoming sidecar passes the same messages through in `synthesis_failed`.
- Audio is streamed in bounded reads from an open file handle with
  `Content-Length`, using worker-thread reads under ASGI (uvicorn) so Django never
  materializes the file in memory. The temporary WAV/export files are removed
  before the response is returned; the handle is closed with the response.
- A Piper voice whose `.onnx.json` config cannot be read or parsed is skipped
  with a logged warning; other installed voices stay available.

Known-speaker reference clips and source intervals are limited to ten minutes of
decoded 16 kHz mono audio (9,600,000 frames, 19.2 MB PCM16 / 38.4 MB float32).
Longer references fail before PCM reads or float conversion; ordinary full job
audio has no reference frame ceiling. Decoder diagnostics are discarded in chunks; references
have a combined 64 KiB stdout/stderr budget; overflow kills and reaps ffmpeg just
like a decode timeout, without exposing tool output. Worker-control JSON integer
and nesting limit failures produce the fixed invalid-JSON worker error. Nesting
limits vary between Python builds; regressions inject that parser failure rather
than depending on a particular recursion threshold.

Known-speaker execution validates stored and claimed settings as strictly as API
submission, rejects ambiguous normalized display names, cooperates with remote
lease cancellation between embeddings, and caps the whole-file track at 32,768
frames/model calls; very long audio gets a wider track step (reported as
`speaker_track_hop`) instead of a failed job. See [known-speaker admission and work budgets](specs/known-speaker-diarization.md#configuration-and-display-name-admission).

Automatic voice routing accepts only finite de/en confidence in `[0.90, 1]`;
malformed detector evidence keeps the pinned voice and language. Piper discovery
reads each `.onnx.json` through a nonblocking, no-follow descriptor, requires a
regular file, and limits declared/actual config bytes to 1 MiB. FIFO, symlink,
oversized and malformed configs are skipped independently, leaving other voices
available; a model without a config is skipped quietly. Discovery never loads
model data.
