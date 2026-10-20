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
  `VOXHELM_SANITIZE_REPEAT_THRESHOLD` (default `4`) consecutive repeats, with
  at most **one second** between the previous cue end and the next cue start.
  Longer gaps and backward-moving starts or ends split runs. Overlapping cues
  are eligible only when neither boundary moves backward. The repeat threshold
  catches the observed loops (9–84×) while leaving natural backchannels and
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

## Local Development

```bash
uv sync
just test
DJANGO_DEBUG=true uv run uvicorn config.asgi:application
```

Outside `DJANGO_DEBUG=true` the app refuses to start without `DJANGO_SECRET_KEY`
(see "Operator UI security" below). The test suite and mypy use
`tests/settings.py`, which supplies a fixed test-only key.

`just check` runs Ruff lint, a Ruff format check (`just format-check`; fix with
`uv run ruff format .`), mypy and pytest. GitHub Actions
(`.github/workflows/ci.yml`) runs the same steps on macOS arm64 on every push and pull
request; tests that need the real Kokoro models are skipped there, and no
secrets or services are needed.

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
- **Login throttle:** after `VOXHELM_LOGIN_MAX_FAILURES` failed sign-ins (default
  5) from one client address or for one username, further attempts are refused
  with HTTP 429 without checking the password until the window of
  `VOXHELM_LOGIN_LOCKOUT_SECONDS` (default 900, starting at the first failure)
  expires. A successful login resets the username counter. The client address is
  `REMOTE_ADDR`, so behind Traefik all browsers share the proxy's address
  bucket; `X-Forwarded-For` is not trusted because the backend port is also
  reachable directly. Counters live in the default Django cache, which is
  per-process local memory and matches the single uvicorn process; a restart
  clears them.

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
`jobs/<job_id>/attempt-<n>/<artifact name>` below the artifact prefix it was
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
the database write lock, so overlapping runs are safe.

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
stripped text with the clip start plus its offsets, rounded to 3 decimals. On
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
succeeded. The producer still downloads winning artifacts through
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

The worker defaults to one active job, periodically heartbeats the leased job
while local inference runs, uploads the source, optional extracted audio,
requested transcript artifacts, and known-speaker `transcript.speakers.json`
sidecar under the claimed attempt prefix, then posts the completion manifest.

Operational note: the application endpoints still require worker auth, but the
macmini/Traefik edge must also block `/v1/internal/*` on public routes unless a
deliberately private worker route is configured.

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
the sidecar enforces two input limits per connection:

- `VOXHELM_WYOMING_STT_MAX_AUDIO_SECONDS` (default `120`, must be positive) caps
  the audio buffered for one transcription, measured after conversion to
  16 kHz, 16-bit, mono PCM. A chunk that would push the buffer past the cap is
  answered with a Wyoming `error` event (`code="audio_too_long"`), the buffer is
  discarded, and the connection is closed without transcribing.
- `Synthesize` requests use the same `VOXHELM_TTS_MAX_INPUT_CHARS` cap (default
  `5000`, counted after trimming surrounding whitespace) as
  `POST /v1/audio/speech`. Longer text is answered with an `error` event
  (`code="text_too_long"`) and the connection is closed without synthesizing.

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
  process is gone. Live holders retain capacity until release, regardless of age,
  including legacy `holder.json` records. `VOXHELM_LANE_SCHEDULER_STALE_SECONDS`
  (default `1800`) is a warning threshold only: each scheduler instance logs an
  old live holder once while its file remains present (normally once per admission
  attempt, since callers create a scheduler for each attempt). A stuck live process
  must be stopped by the operator before its capacity can be reclaimed; deleting its
  holder file while inference runs can exceed the capacity bound.
  PID liveness cannot detect PID reuse after a crash/reboot, or a leaked holder
  inside a live daemon. To recover an orphan, stop all local inference consumers
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

MP3/OGG speech export runs ffmpeg with a positive integer deadline configured by
`VOXHELM_AUDIO_EXPORT_TIMEOUT_SECONDS` (default `300` seconds). Timeout kills
and reaps the converter; timeout, startup and nonzero-exit failures remove the
owned output and raise a synthesis error (HTTP 500 `server_error` on the sync
API, failed batch job). Callers still own and clean up the source WAV. WAV
passthrough does not invoke ffmpeg or use this deadline.
