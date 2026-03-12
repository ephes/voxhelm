# Voxhelm Milestones

**Date:** 2026-03-11
**Status:** M1a implemented on 2026-03-12; later milestones still draft
**Input:** `specs/2026-03-11_voxhelm_service.md`

## Current Implementation Snapshot

Implemented today:

- M1a only
- single Django + `uvicorn` process on `studio`
- private HTTPS ingress on `macmini` at `https://voxhelm.home.xn--wersdrfer-47a.de`
- `GET /v1/health`
- `POST /v1/audio/transcriptions`
- bearer auth
- multipart upload and JSON URL mode
- response formats `json`, `text`, `verbose_json`, and `vtt`
- accepted model aliases `gpt-4o-mini-transcribe` and `whisper-1`
- Archive validation with env vars only

Not implemented yet:

- Django Tasks runtime
- batch jobs
- MinIO artifacts
- video preprocessing
- Wyoming
- TTS
- additional consumers beyond Archive validation

## Design Decisions

These decisions shape the milestone structure and must be stated up front.

### 1. What is the smallest credible v1?

A working OpenAI-compatible `POST /v1/audio/transcriptions` endpoint on `studio` that Archive can call by changing three env vars (`ARCHIVE_TRANSCRIPTION_API_KEY`, `ARCHIVE_TRANSCRIPTION_API_BASE`, `ARCHIVE_TRANSCRIPTION_MODEL`). This proves the core value (local transcription, no hosted API cost) with the least integration work on any consumer.

The full batch-job model is M1b, not M1a. Archive's current code is purely synchronous -- it downloads media, posts to an OpenAI-compatible endpoint, and reads back JSON with a `text` field. Forcing Archive through the batch path before basic transcription works would delay the first useful deployment by weeks.

### 2. Is Home Assistant voice M1 or M2?

**M2.** Wyoming is an independent protocol with no dependency on the batch job system. It can be developed in parallel with M1b. It is M2 because it does not share integration surface with any other consumer and has its own operational concerns (latency, streaming, Piper for TTS).

### 3. Is TTS part of v1 or follows batch transcription?

**Follows.** TTS is M3. The only v1 consumer that might want TTS is Home Assistant, and that is already M2. Batch TTS for Archive article audio has no current consumer code at all.

### 4. Is OpenClaw in v1?

**No.** OpenClaw is M4. There is no existing OpenClaw integration code and no concrete API contract to target. It belongs in v1 only as an architectural placeholder: keep the HTTP API generic enough that a future OpenClaw plugin can call it.

### 5. Is diarization a v1 feature, spike, or later?

**Later.** Diarization is not required by any current consumer. django-cast's Transcript model has a `speakerDesignation` field, but it is currently always empty. podcast-transcript does not produce speaker labels. Diarization should be a spike at earliest in M2, with implementation deferred until a consumer needs it.

---

## Milestone 0: Technical Spikes

**Purpose:** Reduce uncertainty around backend expansion and later milestones without blocking the Archive-first path.

**What ships:** Decision records, not production code.

### Spike 0a: STT Backend Benchmark

- **Question:** Which backend (WhisperKit, mlx-whisper, whisper.cpp) should be the default for batch STT on `studio`? What are the real latency/quality/memory tradeoffs?
- **Method:** Transcribe the same 3 audio files (short <1min, medium ~10min, long ~60min) with each backend on `studio`. Measure wall-clock time, peak memory, and compare transcript quality by manual inspection.
- **Duration:** 1-2 days.
- **Depends on:** `studio` access, model downloads.
- **Blocks:** Does not block M1a. Validates or revisits the initial backend choice later.

### Spike 0b: WhisperKit Server Mode

- **Question:** Can WhisperKit serve an OpenAI-compatible HTTP endpoint natively, or does Voxhelm need to wrap it as a subprocess? What is the startup/teardown cost?
- **Method:** Install WhisperKit on `studio`, attempt to run its server mode, test with a curl POST to `/audio/transcriptions`.
- **Duration:** 0.5-1 day.
- **Depends on:** `studio` access.
- **Blocks:** Does not block M1a. Only matters if WhisperKit becomes a target backend.

### Spike 0c: Wyoming Protocol Feasibility

- **Question:** What is the minimum code needed to expose a Wyoming-compatible STT/TTS service backed by a local whisper model and Piper? Can existing `wyoming-faster-whisper` or similar be reused as-is, or does Voxhelm need a custom Wyoming server?
- **Method:** Run `wyoming-faster-whisper` (or equivalent) on `studio`, connect Home Assistant to it, test a voice command end-to-end.
- **Duration:** 1 day.
- **Depends on:** Home Assistant instance access, `studio`.
- **Blocks:** M2 scope definition. Does NOT block M1.

---

## Milestone 1a: Synchronous STT Endpoint

**Title:** OpenAI-compatible transcription API on `studio`

### What ships

- Django project skeleton on `studio` (project name: `voxhelm`)
- `POST /v1/audio/transcriptions` endpoint
  - Accepts either multipart upload (`file`) or JSON URL input (`url`) plus `model`, `language`, `prompt`, `response_format`
  - Returns `json`, `text`, `verbose_json`, or `vtt`
  - Validates upload size (25 MiB limit for file uploads), MIME type, supported formats
- One working STT backend adapter (`mlx-whisper` as the accepted starting backend)
- Backend abstraction layer (Python protocol class, same shape as podcast-transcript's `TranscriptionBackend`)
- Token-based authentication (bearer token, per-consumer producer tokens)
- Health endpoint (`GET /v1/health`)
- Deployment via ops-library role to `studio` (launchd service, not systemd — `studio` runs macOS)
- Private HTTPS ingress via Traefik on `macmini`
- Archive can switch to Voxhelm by setting three env vars

### What is deferred

- Batch/async job model (M1b)
- MinIO artifact storage (M1b)
- Separate worker execution loop / Django Tasks runtime (M1b)
- Multiple STT backends (M1c)
- Video input / audio extraction (M1b)
- Time-aligned/structured transcript output (M1b)
- Wyoming (M2)
- TTS (M3)
- podcast-transcript integration (M1c)
- python-podcast / django-cast integration (M1c)

### Dependencies

- `studio` deployment infrastructure (ops-library role)

### Success criteria

- Archive's existing `transcriptions.py` works against Voxhelm with zero code changes to Archive (only env var changes)
- The same endpoint also supports URL-based transcription requests for near-term Archive evolution and other producers
- Representative Archive audio items in the M1a scope complete within Archive's existing 300-second timeout budget
- Service survives restart and resumes accepting requests
- Private HTTPS ingress on `macmini` works for trusted local/Tailscale clients

**Note:** Video input handling (audio extraction from video) is deferred to M1b. M1a handles audio-only inputs and has no root `/` route; `/v1/health` is the health surface.

### Key risks

- `studio` model loading time may cause HTTP timeouts for the first request (mitigation: preload model on service start)
- Large files (near 25 MiB) may OOM or timeout (mitigation: enforce size limits, test with real podcast audio)

---

## Milestone 1b: Batch Job System and Artifact Storage

**Title:** Async job model, MinIO artifacts, and video support

### What ships

- Task/job tracking in Django/SQLite tied to Django Tasks:
  - Job states: `queued`, `running`, `succeeded`, `failed`, `canceled`, `expired`
  - Job types: `transcribe`
  - Producer identity and idempotency key (`task_ref`)
  - Internal linkage from Voxhelm job UUID to the Django task/result id returned by `enqueue()`
- `POST /v1/jobs` endpoint for batch job submission
- `GET /v1/jobs/{id}` for status polling
- Django Tasks worker processes on `studio`
  - Backed initially by `django_tasks.backends.database.DatabaseBackend`
- MinIO artifact storage:
  - Input media stored to MinIO before processing
  - Transcript outputs stored to MinIO with stable references
  - Artifact reference returned in job result
- Video-to-audio extraction (ffmpeg preprocessing)
- Structured transcript output: plain text + Whisper-format JSON
- Producer auth tokens plus operator/admin auth; task worker runtime is internal

### What is deferred

- Multiple STT backends beyond the default (M1c)
- podcast-transcript backend class (M1c)
- python-podcast / django-cast integration (M1c)
- podcast-pipeline integration (M1c)
- Interactive/batch lane separation (M2)
- TTS (M3)
- Diarization (later)
- Admin UI for job management (later)

### Dependencies

- M1a (working STT endpoint and backend adapter)

### Success criteria

- A batch job submitted via `POST /v1/jobs` with a URL input completes end-to-end: submit, queue, process, store to MinIO, return result
- A video URL input produces a transcript (audio extracted automatically)
- Duplicate submissions with the same `task_ref` do not create duplicate jobs
- A worker restart resumes or safely requeues unfinished jobs
- Artifacts are retrievable from MinIO after job completion

### Key risks

- SQLite WAL mode contention if the sync endpoint (M1a) and Django Tasks workers both write simultaneously (mitigation: keep writes minimal, use `PRAGMA busy_timeout`)
- MinIO connectivity from `studio` (mitigation: test MinIO access early, during M1a deployment)
- Unnecessary custom worker orchestration could add complexity without solving a real v1 problem (mitigation: start with stock Django Tasks database backend plus launchd supervision)

---

## Milestone 1c: Consumer Integrations for Transcription

**Title:** podcast-transcript backend, python-podcast integration, podcast-pipeline support

### What ships

- **podcast-transcript backend:** New `Voxhelm` class implementing the `TranscriptionBackend` protocol in podcast-transcript. This class calls `POST /v1/audio/transcriptions` (sync) or submits a batch job, downloads the result, and writes it to the expected transcript path. Added as a `--backend voxhelm` option to the podcast-transcript CLI.
- **podcast-pipeline support:** With the podcast-transcript backend in place, podcast-pipeline's `transcribe` entrypoint (which shells out to `podcast-transcript`) works automatically. No changes needed to podcast-pipeline itself.
- **python-podcast / django-cast integration:** Management command or signal-based trigger in python-podcast/django-cast that submits a transcription job to Voxhelm for an episode's audio, polls for completion, and creates/updates the `Transcript` model with Podlove JSON, WebVTT, and DOTe outputs. Voxhelm performs the server-side conversion from canonical Whisper JSON into these consumer-facing formats.
- Structured output format negotiation: job submission can request `["text", "json", "dote", "podlove", "vtt"]` output formats
- Second and third STT backend adapters (all three of whisperkit, mlx-whisper, whisper.cpp available, selectable via `model` or `backend` parameter)

### What is deferred

- Wyoming / Home Assistant (M2)
- TTS (M3)
- OpenClaw (M4)
- Diarization
- Automatic transcript generation triggers (consumers poll or call explicitly)

### Dependencies

- M1b (batch job system, MinIO artifacts, structured outputs)

### Success criteria

- `podcast-transcript --backend voxhelm <url>` produces the same output formats (DOTe, Podlove, WebVTT, plain text) as the existing whisper-cpp backend
- podcast-pipeline's `transcribe` command works with `--backend voxhelm` in the podcast-transcript config
- python-podcast can trigger transcript generation for an episode and populate the django-cast Transcript model
- All three STT backends are selectable and produce valid transcripts

### Key risks

- Output format differences between backends (whisper.cpp produces different JSON than mlx-whisper) -- mitigation: normalize in Voxhelm before returning, do not push format conversion to consumers
- podcast-transcript's `TranscriptionBackend` protocol expects `(audio_file: Path, transcript_path: Path)` -- the Voxhelm backend either uploads the file or passes a URL, which is a different flow than local execution. Mitigation: the Voxhelm backend class handles this internally.

---

## Milestone 2: Home Assistant Voice (Wyoming)

**Title:** Wyoming STT/TTS for Home Assistant

### What ships

- Wyoming-compatible STT server process on `studio`
- Wyoming-compatible TTS server process on `studio` (Piper) — **requires C15 (TTS backend adapter) to be complete; if M2 ships before M3, TTS follows separately**
- Interactive execution lane: separate from batch queue, priority scheduling
- Low-latency STT backend configuration (possibly a smaller/faster model for interactive use)
- Piper TTS engine deployment on `studio` (tied to C15 availability)
- Deployment runbook for connecting Home Assistant to Voxhelm Wyoming endpoints
- ops-library role for Wyoming companion processes (launchd services on `studio`)

### What is deferred

- Batch TTS for article audio (M3)
- Wake-word detection (out of scope per PRD)
- OpenClaw voice turns (M4)
- Diarization

### Dependencies

- Spike 0c (Wyoming feasibility)
- M1a (STT backend adapter layer, deployment infrastructure)
- **Wyoming STT** does not depend on M1b or M1c (separate protocol path, can develop in parallel)
- **Wyoming TTS** depends on C15 (Piper TTS backend adapter, which is an M3 chunk). If M2 ships before M3, it ships with STT only and TTS follows when C15 is ready

### Success criteria

- Home Assistant Assist pipeline uses `studio` for both STT and TTS
- Voice commands through a HA voice device or mobile app work end-to-end
- Interactive voice requests are not blocked by concurrent batch transcription jobs
- Piper produces intelligible speech for response text

### Key risks

- Latency budget: Home Assistant voice expects sub-second STT response for interactive use. If the chosen STT model is too large, interactive use will feel sluggish. Mitigation: use a smaller/turbo model for interactive lane, benchmark in Spike 0c.
- Resource contention: running a batch transcription and an interactive voice request simultaneously on `studio` may cause memory pressure. Mitigation: batch workers pause or reduce concurrency when interactive requests are active.

---

## Milestone 3: TTS Batch Generation

**Title:** Batch text-to-speech for article audio

### What ships

- `synthesize` job type in the batch system
- Piper TTS backend adapter for batch use (reuse from M2 deployment)
- `POST /v1/audio/speech` synchronous endpoint
- `POST /v1/jobs` with `job_type: synthesize` for batch use
- Configurable voice/model presets per producer
- Audio output stored to MinIO as artifacts
- Archive integration: Archive can submit text for TTS and receive a stable audio URL for feed enclosures

### What is deferred

- Additional TTS engines (Kokoro, etc.)
- OpenClaw integration (M4)
- Diarization

### Dependencies

- M1b (batch job system, MinIO)
- M2 (Piper deployed and operational)

### Success criteria

- Archive can submit article text and receive generated speech audio
- Generated audio is a valid podcast-quality MP3/WAV usable as a feed enclosure
- Batch TTS jobs do not starve interactive voice traffic

### Key risks

- Piper audio quality may not be sufficient for article-length content. Mitigation: evaluate Piper output quality during M2; if inadequate, Kokoro or another engine becomes an M3 requirement.

---

## Milestone 4: OpenClaw Integration

**Title:** Stable API contract for OpenClaw plugins

### What ships

- Documented HTTP API contract (versioned, stable)
- Example OpenClaw plugin/tool that calls Voxhelm for STT and TTS
- Optional interactive low-latency path for voice turns via HTTP API
- Rate limiting and per-producer usage tracking

### What is deferred

- Deep OpenClaw-specific features
- Diarization (unless a concrete use case emerges)

### Dependencies

- M1a (HTTP API), M3 (TTS)

### Success criteria

- An OpenClaw tool/plugin can transcribe audio and generate speech through the Voxhelm HTTP API
- OpenClaw does not need its own speech engine stack

### Key risks

- OpenClaw's API needs are not yet concrete. Mitigation: keep the HTTP API generic and OpenAI-compatible so it can serve OpenClaw without OpenClaw-specific code in Voxhelm.

---

## Milestone Summary Table

| Milestone | Title | Key Consumer | Depends On | Estimated Effort |
|-----------|-------|-------------|------------|-----------------|
| M0 | Technical Spikes | (internal) | `studio` access | 3-4 days |
| M1a | Sync STT Endpoint | Archive | `studio` access | 1-2 weeks |
| M1b | Batch Jobs + MinIO | python-podcast, future consumers | M1a | 1-2 weeks |
| M1c | Consumer Integrations | podcast-transcript, python-podcast, podcast-pipeline | M1b | 1-2 weeks |
| M2 | Wyoming Voice | Home Assistant | M0 Spike 0c, M1a | 1-2 weeks |
| M3 | TTS Batch | Archive (article audio) | M1b, M2 | 1 week |
| M4 | OpenClaw | OpenClaw | M1a, M3 | 1 week |

**Critical observation:** M1a is the fastest path to production value. A single developer should target M1a as the first deliverable, which could be usable within 2 weeks of starting (including spikes). The full PRD Milestone 1 has been split into M1a/M1b/M1c to keep each increment deployable and testable.

**Parallelization note:** M2 (Wyoming) can proceed in parallel with M1b/M1c after M1a is complete, since it shares the backend adapter layer but not the job system.
