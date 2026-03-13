# Voxhelm Milestones

**Date:** 2026-03-11
**Status:** M1a, M1b, the current M1c consumer slices, and the core M2/M3 runtime work are implemented as of 2026-03-13; remaining planned work is C13 lane scheduling, further backend expansion, Archive article-audio follow-on, and M4/OpenClaw
**Input:** `specs/2026-03-11_voxhelm_service.md`

## Current Implementation Snapshot

Implemented today:

- M1a and M1b
- M1c consumer work: `podcast-transcript --backend voxhelm`, `podcast-pipeline` compatibility, and `python-podcast` / `django-cast` Wagtail-admin transcript workflow
- M2 Home Assistant voice wiring: Wyoming STT/TTS sidecar on `studio`, Home Assistant integration, declarative Assist pipelines, and area-registry alias updates
- core M3 runtime: Piper-backed TTS, `POST /v1/audio/speech`, and batch `synthesize` jobs
- Django + `uvicorn` HTTP process plus Django Tasks worker on `studio`
- Wyoming STT/TTS sidecar on `studio`
- private HTTPS ingress on `macmini` at `https://voxhelm.home.xn--wersdrfer-47a.de`
- `GET /v1/health`
- `POST /v1/audio/transcriptions`
- `POST /v1/audio/speech`
- `POST /v1/jobs`
- `GET /v1/jobs/{id}`
- `GET /v1/jobs/{id}/artifacts/{name}`
- bearer auth
- multipart upload and JSON URL mode
- response formats `json`, `text`, `verbose_json`, and `vtt`
- accepted model aliases `gpt-4o-mini-transcribe` and `whisper-1`
- Django Tasks database-backed job execution
- MinIO-backed artifact storage in bucket `voxhelm`
- video-to-audio extraction for batch jobs
- Archive validation with env vars only
- live batch-job submission, completion, artifact fetch, and MinIO object verification
- live validation of `podcast-transcript` against the deployed Voxhelm edge service
- live validation of direct Home Assistant STT against the deployed Wyoming path

Not implemented yet:

- interactive lane scheduling / capacity reservation between batch and Wyoming traffic
- additional STT backends beyond the current `whisper.cpp` and `mlx-whisper` set
- Archive article-to-audio consumer integration
- OpenClaw integration

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
- remaining consumer integrations (M1c)

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
- `GET /v1/jobs/{id}/artifacts/{name}` for artifact download through Voxhelm
- Django Tasks worker processes on `studio`
  - Backed by `django_tasks_db.backend.DatabaseBackend`
- MinIO artifact storage:
  - Input media stored to MinIO before processing
  - Transcript outputs stored to MinIO with stable references
  - Artifact reference returned in job result
- Video-to-audio extraction (ffmpeg preprocessing)
- Structured transcript output: plain text + Whisper-format JSON
- Producer auth tokens plus operator/admin auth; task worker runtime is internal

### What is deferred

- Multiple STT backends beyond the default (M1c)
- python-podcast / django-cast integration (M1c)
- podcast-pipeline integration (M1c)
- Interactive/batch lane separation (M2)
- TTS (M3)
- Diarization (later)
- Admin UI for job management (later)

### Dependencies

- M1a (working STT endpoint and backend adapter)

### Success criteria

Completed on 2026-03-12:

- [x] A batch job submitted via `POST /v1/jobs` with a URL input completes end-to-end: submit, queue, process, store to MinIO, return result
- [x] A video URL input produces a transcript (audio extracted automatically)
- [x] Duplicate submissions with the same `task_ref` do not create duplicate jobs
- [x] The deployed worker starts successfully under launchd after migrations run during deploy
- [x] Artifacts are retrievable through Voxhelm and present in MinIO after job completion

### Key risks

- SQLite WAL mode contention if the sync endpoint (M1a) and Django Tasks workers both write simultaneously (mitigation: keep writes minimal, use `PRAGMA busy_timeout`)
- MinIO connectivity from `studio` (mitigation: test MinIO access early, during M1a deployment)
- Unnecessary custom worker orchestration could add complexity without solving a real v1 problem (mitigation: start with stock Django Tasks database backend plus launchd supervision)

---

## Milestone 1c: Consumer Integrations for Transcription

**Title:** podcast-transcript backend, python-podcast integration, podcast-pipeline support

**Implementation note (2026-03-12):** `podcast-transcript` now has a `Voxhelm` backend and `--backend voxhelm`, and that path has been validated against the deployed edge service. `podcast-pipeline` has the required compatibility follow-on. `django-cast` now also ships the required Wagtail-admin workflow: privileged editors can trigger transcript generation from Episode and Audio edit views, site admins can manage Voxhelm connection settings in Wagtail admin, and the existing `generate_transcripts` management command remains as fallback operator tooling.

### What ships

- **podcast-transcript backend:** New `Voxhelm` class implementing the `TranscriptionBackend` protocol in podcast-transcript. This class currently calls `POST /v1/audio/transcriptions` and writes Whisper-compatible JSON into the existing output pipeline. Added as a `--backend voxhelm` option to the podcast-transcript CLI.
- **podcast-pipeline follow-on:** Delivered. The pipeline now provides the required audio-input compatibility for the Voxhelm-backed `podcast-transcript` path.
- **python-podcast / django-cast integration:** Wagtail-admin workflow in `django-cast` / `python-podcast` that lets privileged editors trigger transcript generation from the Wagtail admin UI for an episode or audio object, persists the existing `Transcript` artifacts, and does not require shell access.
- **python-podcast / django-cast configuration:** Voxhelm API base URL, API token, and optional model/language preferences are manageable through Wagtail admin (for example via Wagtail settings or a protected snippet), not only through Django settings or environment variables.
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

Completed on 2026-03-12:

- [x] `podcast-transcript --backend voxhelm <url>` produces the expected output formats (DOTe, Podlove, WebVTT, plain text) through its existing pipeline
- [x] podcast-pipeline's `transcribe` command works with the Voxhelm-backed podcast-transcript path
- [x] Wagtail editors can trigger transcript generation for an episode or audio object from Wagtail admin
- [x] Voxhelm connection settings for python-podcast are manageable through Wagtail admin
- [x] The Wagtail-admin flow populates the django-cast Transcript model without shell access

Still pending:

- [ ] All three STT backends are selectable and produce valid transcripts

### Key risks

- Output format differences between backends (whisper.cpp produces different JSON than mlx-whisper) -- mitigation: normalize in Voxhelm before returning canonical JSON/VTT outputs, with consumer-local Podlove/DOTe conversion where needed
- podcast-transcript's `TranscriptionBackend` protocol expects `(audio_file: Path, transcript_path: Path)` -- the Voxhelm backend either uploads the file or passes a URL, which is a different flow than local execution. Mitigation: the Voxhelm backend class handles this internally.
- podcast-pipeline's current transcriber command contract was narrower than assumed in the initial plan. Mitigation: treat it as a small follow-on integration step instead of assuming zero-change compatibility.
- Storing a Voxhelm API token in Wagtail-admin-managed configuration requires careful permissions and auditability. Mitigation: restrict editing to privileged Wagtail admins and use the Wagtail admin surface rather than Django admin.

---

## Milestone 2: Home Assistant Voice (Wyoming)

**Title:** Wyoming voice for Home Assistant

**Implementation note (2026-03-13):** Delivered. Voxhelm now runs a Wyoming STT/TTS sidecar on `studio`, and the Home Assistant deploy/config path provisions Wyoming plus declarative Assist pipelines. The remaining gap from the original M2 plan is C13 lane scheduling: the current deployment works, but it still shares host resources with the HTTP API and batch worker.

### What ships

- Wyoming-compatible STT/TTS sidecar process on `studio`
- Home Assistant integration that provisions the Wyoming provider and selects it in the Assist pipeline
- Declarative multi-pipeline Assist configuration and preferred-pipeline selection
- Area-registry alias and canonical-sensor updates for Assist-friendly room resolution
- Interactive execution lane only if real HA use proves that batch work causes unacceptable contention
- Low-latency STT backend configuration (possibly a smaller/faster model for interactive use)
- Deployment runbook for connecting Home Assistant to Voxhelm Wyoming endpoints
- ops-library role for Wyoming companion processes (launchd services on `studio`)

### What is deferred

- Interactive lane scheduling / resource reservation (C13)
- Archive article-audio consumer follow-on for the shared TTS runtime
- Wake-word detection (out of scope per PRD)
- OpenClaw voice turns (M4)
- Diarization

### Dependencies

- Spike 0c (Wyoming feasibility)
- M1a (STT backend adapter layer, deployment infrastructure)
- **Wyoming STT** does not depend on M1b or M1c (separate protocol path, can develop in parallel)
- The shared Piper/TTS runtime from M3 now backs the Wyoming TTS path as well. The still-open follow-on is C13 scheduling rather than missing protocol support.

### Success criteria

- Home Assistant Assist pipeline uses `studio` for STT and TTS
- At least one Assist turn succeeds end-to-end through the real Home Assistant Assist pipeline using Voxhelm
- C13 remains the explicit next step because the current deployment still allows batch and Wyoming contention on `studio`
- Operator docs accurately describe the live voice setup and the default-off debug-logging guidance

### Key risks

- Latency budget: Home Assistant voice expects sub-second STT response for interactive use. If the chosen STT model is too large, interactive use will feel sluggish. Mitigation: use a smaller/turbo model for interactive lane, benchmark in Spike 0c.
- Resource contention: running a batch transcription and an interactive voice request simultaneously on `studio` may cause memory pressure. Mitigation: batch workers pause or reduce concurrency when interactive requests are active.

---

## Milestone 3: Shared TTS Runtime And Batch Generation

**Title:** Piper-backed TTS for Home Assistant and batch consumers

**Implementation note (2026-03-13):** Delivered at the Voxhelm service/runtime layer. Piper-backed TTS is live in Voxhelm, Home Assistant can use the Wyoming TTS path, and Voxhelm exposes both synchronous speech generation and batch `synthesize` jobs. Archive article-to-audio consumer integration remains a future follow-on.

### What ships

- Piper TTS backend adapter and deployment on `studio`
- Wyoming-compatible TTS server process on `studio` for Home Assistant Assist
- `POST /v1/audio/speech` synchronous endpoint
- `synthesize` job type in the batch system
- `POST /v1/jobs` with `job_type: synthesize` for batch use
- Configurable voice/model presets per producer
- Audio output stored to MinIO as artifacts
- Shared runtime that Home Assistant already consumes via Wyoming TTS

### What is deferred

- Archive article-to-audio consumer integration
- Additional TTS engines (Kokoro, etc.)
- OpenClaw integration (M4)
- Diarization

### Dependencies

- M1b (batch job system, MinIO)
- M2 (Home Assistant STT deployment/runbook and Wyoming integration path)

### Success criteria

- Home Assistant Assist pipeline can use `studio` for TTS
- A Home Assistant TTS request produces intelligible speech
- Voxhelm exposes valid audio output through both `POST /v1/audio/speech` and batch `synthesize` jobs
- Archive article-audio remains a future consumer slice rather than a missing service/runtime capability
- Batch TTS jobs still need C13-style protection so they do not starve interactive voice traffic

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
| M2 | Wyoming Voice | Home Assistant | M0 Spike 0c, M1a | delivered |
| M3 | TTS Runtime + Batch | Home Assistant, future Archive article audio | M1b, M2 | delivered at service/runtime layer |
| M4 | OpenClaw | OpenClaw | M1a, M3 | 1 week |

**Critical observation:** M1a is the fastest path to production value. A single developer should target M1a as the first deliverable, which could be usable within 2 weeks of starting (including spikes). The full PRD Milestone 1 has been split into M1a/M1b/M1c to keep each increment deployable and testable.

**Parallelization note:** The original M2 and M3 plan was executed as intended: Wyoming voice and then shared Piper/TTS were added without changing the producer-facing HTTP contracts. The next concrete implementation step is C13 lane scheduling so the live voice path remains responsive under mixed load.
