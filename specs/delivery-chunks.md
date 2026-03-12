# Voxhelm Delivery Chunks

**Date:** 2026-03-11
**Input:** `2026-03-11_voxhelm_service.md` (PRD), consumer repo exploration
**Status:** M1a and M1b chunks implemented on 2026-03-12; C9 implemented on 2026-03-12; later chunks still draft

Current completion state:

- Implemented: C1, C2, C3, C4, C5, C6, C7, C8, C9, and C11
- Not implemented yet: C10, C12-C17

---

## Chunk Overview

| ID | Title | Milestone Slice | Dependencies |
|----|-------|-----------------|--------------|
| S1 | STT backend spike | M0 | None |
| S2 | Wyoming feasibility spike | M0 | None |
| C1 | Django project skeleton and auth | M1a | None |
| C2 | Task tracking and submission API | M1b | C1 |
| C3 | Django Tasks runtime integration | M1b | C2 |
| C4 | MinIO artifact persistence | M1b | C1 |
| C5 | STT backend adapter layer | M1a+c | C1 |
| C6 | Batch transcription pipeline | M1b | C3, C4, C5 |
| C7 | OpenAI-compatible STT endpoint | M1a | C5 |
| C8 | Archive integration | M1a | C7, C11 |
| C9 | podcast-transcript Voxhelm backend | M1c | C7, C6 |
| C10 | python-podcast / django-cast integration | M1c | C6 |
| C11 | Deployment role (ops-library) | M1a+b | C1 |
| C12 | Wyoming STT/TTS adapter | M2 | S2, C5 |
| C13 | Interactive lane scheduling | M2 | C12, C3 |
| C14 | Home Assistant integration | M2 | C12, C13, C11 |
| C15 | TTS backend adapter layer (Piper) | M3 | C1 |
| C16 | Batch TTS jobs | M3 | C15, C3, C4 |
| C17 | OpenClaw integration | M4 | C7 |

---

## Spikes

### S1 -- STT Backend Benchmark Spike

**Purpose:** Benchmark backend expansion options on `studio` without blocking the accepted `mlx-whisper` starting point.

**Included scope:**

- Install and run WhisperKit, mlx-whisper, and whisper.cpp on `studio`
- Benchmark each on 3-5 representative audio files (short voice clip, long podcast episode, video-extracted audio, German-language content)
- Measure: wall-clock time, peak memory, transcript quality (subjective comparison)
- Test model variants: large-v3 and large-v3-turbo where available
- Document any installation or runtime issues on macOS

**Explicitly excluded scope:**

- Writing production adapter code (that is C5)
- TTS benchmarking
- Diarization testing

**Dependencies:** None

**Consumer(s):** All downstream chunks (C5, C6, C7)

**Primary interfaces:** None (produces a written report only)

**Acceptance criteria:**

- A short written comparison document with timing and quality data
- A recommendation on whether the accepted `mlx-whisper` default should change later
- A recommendation on whether interactive STT should use a different backend
- Known installation issues documented

**Main risks:**

- WhisperKit may not have a stable Python/CLI interface on macOS yet -- if so, document and defer it
- mlx-whisper performance characteristics may differ significantly from published benchmarks

**Suggested implementation order:** Optional. Can run in parallel with implementation.

---

### S2 -- Wyoming Adapter Feasibility Spike

**Purpose:** Determine the practical strategy for exposing Voxhelm STT and TTS via the Wyoming protocol for Home Assistant.

**Included scope:**

- Study the Wyoming protocol specification and existing implementations (`wyoming-faster-whisper`, `wyoming-piper`)
- Determine whether to run Wyoming as a sidecar process that proxies to Voxhelm's HTTP API, or embed Wyoming protocol handling inside the Django service
- Prototype one approach (STT only) to validate feasibility
- Document HA configuration requirements for adding a custom Wyoming provider

**Explicitly excluded scope:**

- Production-quality Wyoming implementation (that is C12)
- Wake-word detection
- TTS via Wyoming (deferred to C12 full implementation)

**Dependencies:** None (can run any time, but should complete before C12 starts)

**Consumer(s):** C12, C14

**Primary interfaces:** None (produces a written recommendation and prototype)

**Acceptance criteria:**

- Written recommendation: sidecar vs. embedded, with justification
- Working prototype that routes one STT request from HA to a local whisper backend
- List of HA configuration steps required

**Main risks:**

- Wyoming protocol may have undocumented requirements beyond the public spec
- Sidecar approach may introduce operational complexity that is undesirable for a homelab

**Suggested implementation order:** Can run in parallel with M1 work. Must complete before C12.

---

## M1 Chunks -- Core Batch Transcription Service

### C1 -- Django Project Skeleton and Auth

**Implementation note (2026-03-12):** Delivered. Auth is environment-configured bearer tokens; there is no token CRUD model/admin UI yet.

**Purpose:** Establish the runnable Django project with database, settings, auth tokens, and health endpoint so all subsequent chunks have a foundation to build on.

**Included scope:**

- Django project scaffolding (settings, urls, wsgi/asgi)
- SQLite database
- Producer bearer-token auth configured from environment
- Health endpoint (`GET /v1/health`) returning service status
- Basic input validation middleware (content-type checks, size limits)
- Project tooling: justfile, pre-commit, ruff, mypy, pytest skeleton

**Explicitly excluded scope:**

- Job model (C2)
- MinIO integration (C4)
- Any STT/TTS logic
- Deployment (C11)

**Dependencies:** None

**Consumer(s):** All subsequent chunks

**Primary interfaces:**

- `GET /v1/health` -- health check
- Django admin for token CRUD
- Bearer token authentication on all `/v1/` endpoints

**Acceptance criteria:**

- `just test` passes with at least one integration test for health endpoint
- `just lint` and `just typecheck` pass
- Producer tokens configured via environment can be used in API requests
- Unauthenticated requests are rejected with 401

**Main risks:**

- Low risk. Standard Django setup.

**Suggested implementation order:** First chunk. No dependencies.

---

### C2 -- Task Tracking and Submission API

**Implementation note (2026-03-12):** Delivered in a narrower M1b form. Implemented endpoints are `POST /v1/jobs`, `GET /v1/jobs/{id}`, and `GET /v1/jobs/{id}/artifacts/{name}`. `GET /v1/jobs` and `DELETE /v1/jobs/{id}` remain deferred.

**Purpose:** Allow producers to submit jobs and query their status. This is the write side of the control plane.

**Included scope:**

- Producer-facing task/job tracking model: id (UUID), task name, state, lane (batch/interactive), backend preference, model preference, language hint, input descriptor (JSON), output descriptor (JSON), context (JSON), task_ref (idempotency key), producer identity, timestamps (created, started, completed, expired), error detail, retry count, linked Django task/result id returned by `enqueue()`
- Job states: `queued`, `running`, `succeeded`, `failed`, `canceled`, `expired`
- State transitions aligned with Django Tasks execution lifecycle
- `POST /v1/jobs` -- submit a new job (producer token required)
- `GET /v1/jobs/{id}` -- get job status and result metadata (producer token required, scoped to own jobs)
- `GET /v1/jobs` -- list jobs with filtering by state, producer, job_type (producer token required, scoped to own jobs)
- `DELETE /v1/jobs/{id}` -- cancel a pending job (producer token required)
- Idempotency: if `task_ref` already exists for the same producer and is not in a terminal failed state, return the existing job
- Input validation: validate job_type against allowed types, validate input descriptor shape, enforce size limits

**Explicitly excluded scope:**

- Django Tasks runtime configuration (C3)
- Artifact storage (C4)
- Actual job execution
- Synchronous transcription endpoint (C7)

**Dependencies:** C1

**Consumer(s):** Archive, python-podcast, podcast-pipeline (via their future integration code)

**Primary interfaces:**

- `POST /v1/jobs` -- job submission
- `GET /v1/jobs/{id}` -- job status
- `GET /v1/jobs/{id}/artifacts/{name}` -- artifact download
- `GET /v1/jobs` -- job listing (deferred)
- `DELETE /v1/jobs/{id}` -- job cancellation (deferred)

**Acceptance criteria:**

- Jobs can be created, listed, and retrieved via the API
- Idempotency works: submitting the same `task_ref` returns the existing job
- Invalid job types and malformed input descriptors are rejected with 400
- Producer tokens can only see their own jobs
- State transitions are enforced (cannot move backward)
- `just test` passes with tests covering submission, retrieval, idempotency, and validation

**Main risks:**

- Getting the input descriptor schema flexible enough for future job types without making it too loose. Mitigate by starting with `transcribe` only and adding types as needed.

**Suggested implementation order:** After C1.

---

### C3 -- Django Tasks Runtime Integration

**Implementation note (2026-03-12):** Delivered. The current runtime uses `django_tasks_db.backend.DatabaseBackend`, stores the linked Django task/result id on each job, and runs a dedicated launchd worker on `studio`.

**Purpose:** Execute queued async work through Django Tasks on `studio` and connect task execution to the producer-facing job/task records.

**Included scope:**

- Configure the chosen Django Tasks backend for production use
- Initial backend choice: `django_tasks_db.backend.DatabaseBackend`
- Define task entrypoints for transcription and extraction work
- Launch Django Tasks worker processes on `studio`
- Map queued/running/completed task execution back into producer-facing job/task records
- Retry policy for transient failures where appropriate
- Restart recovery using Django Tasks plus persisted task/job tracking data
- Concurrency configuration for batch work

**Explicitly excluded scope:**

- Actual transcription pipeline logic (C6)
- MinIO artifact upload (C4)
- Interactive lane policy (C13)

**Dependencies:** C2

**Consumer(s):** Internal

**Primary interfaces:**

- Django Tasks task definitions
- Django Tasks worker processes on `studio`
- Database-backed task result records linked from the producer-facing job model

**Acceptance criteria:**

- A queued task is executed by a Django Tasks worker and the producer-facing record transitions to `running`
- Successful task completion updates result metadata and terminal state
- Failed task execution records structured failure detail
- Restart recovery preserves or safely requeues unfinished work according to the configured backend behavior
- `just test` passes for task submission, execution, and recovery behavior

**Main risks:**

- SQLite concurrent write performance under contention. Mitigate with WAL mode and short transactions. This is acceptable for the expected low-concurrency workload on `studio`.
- Re-implementing custom worker orchestration too early. Mitigate by starting with the stock database backend and launchd supervision before adding handshake/heartbeat logic.

**Suggested implementation order:** After C2.

---

### C4 -- MinIO Artifact Persistence

**Implementation note (2026-03-12):** Delivered. Production uses the S3-compatible artifact backend against MinIO bucket `voxhelm`, while the filesystem backend remains the local default.

**Purpose:** Store and retrieve job input and output artifacts in MinIO so they are durable and accessible by both the control plane and workers.

**Included scope:**

- MinIO client integration (boto3 or minio-py)
- Artifact model: id, job reference, kind (input/output), content type, storage key, size, checksum, created timestamp
- Storage key convention: `{job_type}/{job_id}/{artifact_kind}/{filename}`
- Upload helper: store bytes or file to MinIO, create artifact record
- Download helper: retrieve artifact by storage key
- HTTP proxy endpoint for artifact delivery (`GET /v1/jobs/{id}/artifacts/{name}`) — consumers access artifacts through Voxhelm, not directly from MinIO
- Cleanup: configurable retention policy (delete artifacts after N days for completed jobs)
- Settings: S3-compatible endpoint URL, bucket name, access key ID, secret access key, prefix, and path-style toggle (from environment)

**Explicitly excluded scope:**

- Local filesystem scratch/cache (workers use local temp storage during processing; only final artifacts go to MinIO)
- Archive-specific artifact formats (C8)
- TTS artifacts (C16)

**Dependencies:** C1

**Consumer(s):** C6, C7, C8, C9, C10 (all chunks that produce or consume artifacts)

**Primary interfaces:**

- Internal Python API: artifact store read/write helpers
- `GET /v1/jobs/{id}/artifacts/{name}` -- proxy download from MinIO

**Acceptance criteria:**

- Artifacts can be uploaded to MinIO and retrieved by key
- Artifact records are linked to jobs in the database
- Artifact proxy endpoint serves files correctly to consumers
- Retention cleanup deletes old artifacts (tested with a short retention window)
- Missing MinIO connectivity produces clear error messages, not crashes
- `just test` passes (MinIO tests may require a test container or mock)

**Main risks:**

- MinIO availability on `studio`. Mitigate by ensuring MinIO is already deployed or deploying it as part of C11.
- Test isolation: MinIO tests need either a real instance or a well-behaved mock. Recommend moto or a dedicated test bucket.

**Suggested implementation order:** After C1. Can be developed in parallel with C2 and C3.

---

### C5 -- STT Backend Adapter Layer

**Purpose:** Provide a pluggable backend abstraction so the rest of the system can request transcription without knowing which engine runs underneath.

**Included scope:**

- `TranscriptionBackend` protocol/interface: `transcribe(audio_path: Path, options: TranscribeOptions) -> TranscriptionResult`
- `TranscribeOptions`: language, model, prompt, output formats requested, word timestamps flag
- `TranscriptionResult`: segments (list of timed text segments), full text, metadata (backend, model, language, duration, processing time)
- Backend implementations:
  - `mlx-whisper` (accepted starting backend)
  - `whisper.cpp` (subprocess-based, proven in podcast-transcript)
  - `WhisperKit` (if spike S1 confirms viability)
- Backend registry: lookup by identifier string (`auto`, `mlx`, `whispercpp`, `whisperkit`)
- `auto` selection logic: configurable default per lane (batch vs interactive)
- Audio preprocessing: resampling to 16kHz mono WAV via ffmpeg (reuse approach from podcast-transcript's WhisperCpp backend)
- Output format: Whisper-native JSON with segments as canonical internal representation. Voxhelm can convert this server-side into plain text, DOTe, Podlove JSON, and WebVTT artifacts.

**Explicitly excluded scope:**

- TTS backends (C15)
- Diarization (S3)
- Job orchestration (C6)
- HTTP endpoint wiring (C7)

**Dependencies:** C1

**Consumer(s):** C6, C7, C12

**Primary interfaces:**

- Python API: `get_backend(identifier: str) -> TranscriptionBackend`
- Python API: `transcribe(audio_path, options) -> TranscriptionResult`
- Python API: Whisper-native JSON as canonical internal output; Voxhelm-side converters can derive requested artifact formats

**Acceptance criteria:**

- At least two backends (mlx-whisper and whisper.cpp) pass integration tests with a short audio sample
- `auto` resolves to the configured default
- Unsupported backend identifiers raise a clear error
- Audio preprocessing produces valid 16kHz mono WAV
- Whisper-native JSON output includes valid segments with `start`, `end`, `text` fields
- `just test` passes (backend integration tests may be marked as requiring `studio` hardware)

**Main risks:**

- Backend installation complexity. mlx-whisper requires Apple Silicon. whisper.cpp requires a compiled binary. Mitigate by documenting prerequisites and providing installation scripts.
- Model download size and time. First run will download large models. Document this.

**Suggested implementation order:** After C1. Can be developed in parallel with C3 and C4.

---

### C6 -- Batch Transcription Pipeline

**Purpose:** Wire together Django Tasks, STT backends, and artifact storage so that submitted transcription jobs are actually executed end-to-end.

**Included scope:**

- Task executor for `transcribe` job type:
  1. Run queued task via Django Tasks
  2. Download input (URL fetch, or MinIO reference)
  3. If video input, extract audio via ffmpeg
  4. Preprocess audio (resample via C5)
  5. Transcribe via configured backend (C5)
  6. Convert results to requested output formats (C5)
  7. Upload output artifacts to MinIO (C4)
  8. Persist success with artifact references in the control plane
- Input handling: URL download with size/duration limits, MinIO object reference resolution
- Video-to-audio extraction: ffmpeg subprocess, extract best audio track
- Error handling: download failures, backend failures, timeout, size exceeded -- all reported as structured failure detail
- Progress reporting: stage-aware task/job status updates (downloading, extracting, transcribing, uploading)
- `extract_audio` job type: same pipeline but stops after extraction, uploads extracted audio as artifact

**Explicitly excluded scope:**

- Synchronous HTTP transcription (C7)
- TTS execution (C16)
- Interactive lane priority (C13)
- Consumer-specific result formatting (C8, C9, C10)

**Dependencies:** C3, C4, C5

**Consumer(s):** C8, C9, C10 (consumers submit jobs and retrieve artifacts)

**Primary interfaces:**

- Django Tasks workers execute queued batch tasks
- Jobs are submitted via C2 API, results retrieved via C2 + C4 APIs

**Acceptance criteria:**

- Submit a transcription job with an audio URL. A Django Tasks worker processes it, stores artifacts in MinIO, and reports success.
- Submit a transcription job with a video URL. Worker extracts audio first, then transcribes.
- Submit an extract_audio job. Worker extracts and uploads audio only.
- Failed downloads produce a structured error in job status.
- Backend failures produce a structured error and the job is retried.
- Progress stages are visible in job status during execution.
- End-to-end test: submit job, poll until complete, download transcript artifact, verify content.

**Main risks:**

- Large media files may cause memory pressure. Mitigate with streaming download to disk and size limits.
- Long transcription times may hold scarce worker capacity. Mitigate with conservative concurrency and clear task timeout configuration.

**Suggested implementation order:** After C3, C4, C5 are complete. This is the first full vertical slice.

---

### C7 -- OpenAI-Compatible Synchronous STT Endpoint

**Purpose:** Provide a synchronous `POST /v1/audio/transcriptions` endpoint that is compatible with the OpenAI Audio API, enabling Archive to switch to Voxhelm with only an env var change.

**Included scope:**

- `POST /v1/audio/transcriptions` -- multipart form upload, synchronous response
  - Required fields: `file` (audio file), `model` (string, mapped to backend+model)
  - Optional fields: `language`, `prompt`, `response_format` (json, text, verbose_json)
  - Response: JSON with `text` field (and segments for verbose_json)
- Model mapping: map OpenAI-style model names to Voxhelm backend+model combinations. For Archive compatibility, v1 must accept `gpt-4o-mini-transcribe` and map it to the configured default STT backend/model; `whisper-1` should be accepted as an OpenAI-compatible alias.
- Input validation: max file size (25 MiB to match Archive's limit), allowed MIME types
- Timeout: target Archive's existing 300-second timeout budget for the M1a compatibility path; reject or redirect callers to the batch API if the request is expected to exceed the synchronous budget
- URL fetch policy for JSON URL mode: allow `https://` by default; allow `http://` only for explicitly configured trusted internal hosts; reject URLs outside the configured host allowlist
- Authentication: producer token via Bearer header (matching Archive's existing `Authorization: Bearer <key>` pattern)
- When used, transcription runs in-process (not via the job queue) for lower latency
- Artifacts optionally stored in MinIO for audit (configurable)

**Explicitly excluded scope:**

- Batch job submission (that goes through C2)
- TTS endpoint (C16)
- Wyoming (C12)
- Streaming responses

**Dependencies:** C5 (backend adapter layer); C4 is optional if sync artifacts are persisted for audit

**Consumer(s):** Archive (primary), podcast-transcript (as a new backend), operator tools

**Primary interfaces:**

- `POST /v1/audio/transcriptions` -- OpenAI-compatible multipart POST

**Acceptance criteria:**

- Archive's existing transcription code works against this endpoint by changing only `ARCHIVE_TRANSCRIPTION_API_BASE` and `ARCHIVE_TRANSCRIPTION_API_KEY` env vars
- Multipart upload with `file`, `model`, `prompt` fields returns JSON with `text` field
- `response_format=verbose_json` returns segments with timestamps
- Files exceeding 25 MiB are rejected with 413
- Unsupported file types are rejected with 400
- Invalid/missing auth returns 401
- `model=gpt-4o-mini-transcribe` is accepted for zero-code-change Archive compatibility
- End-to-end test: upload a short audio file, receive transcript text

**Main risks:**

- Synchronous transcription of long files may exceed Archive's 300-second timeout budget. Mitigate with a configurable max duration and clear error message suggesting batch API for large files.
- Subtle OpenAI API compatibility differences. Mitigate by testing with Archive's actual request code.

**Suggested implementation order:** Can start after C5. Prioritize this alongside C6 because it unlocks Archive integration (C8).

---

### C8 -- Archive Integration

**Purpose:** Connect Archive to Voxhelm so that Archive's metadata worker uses Voxhelm for transcription instead of a hosted API.

**Included scope:**

- Verify Archive works with C7 endpoint (env var change only): set `ARCHIVE_TRANSCRIPTION_API_BASE` to Voxhelm URL, `ARCHIVE_TRANSCRIPTION_API_KEY` to a Voxhelm producer token, `ARCHIVE_TRANSCRIPTION_MODEL` to a supported model name
- Document the configuration change
- Test with representative Archive items: podcast episodes, video items
- Verify that Archive's 25 MiB size limit, prompt construction, and response parsing all work correctly
- If any incompatibilities are found, fix them in C7 or document workarounds

**Explicitly excluded scope:**

- Changing Archive's transcription code (goal is zero-change integration via env vars)
- Batch job API integration for Archive (future enhancement -- Archive's current model is synchronous)
- TTS/article-to-audio for Archive (C16, M3)

**Dependencies:** C7, C11 (Voxhelm must be deployed and reachable from the macmini where Archive runs)

**Consumer(s):** Archive

**Primary interfaces:**

- Same as C7 (`POST /v1/audio/transcriptions`)

**Acceptance criteria:**

- Archive's `run_metadata_worker` successfully transcribes a podcast episode (audio) via Voxhelm
- No changes to Archive source code required (env var only)
- Transcription quality is at least comparable to previous hosted API results

**Note:** Video item transcription requires audio extraction (C6/M1b). Archive video items that fit within the 25 MiB sync limit may work if the backend handles video input, but full video support is validated in M1b, not here.

**Main risks:**

- Network latency between macmini (Archive) and `studio` (Voxhelm) could cause timeouts for large files. Mitigate by tuning Archive's `--transcription-timeout` and Voxhelm's request timeout.
- Video items may exceed 25 MiB. Archive currently only downloads 25 MiB. Voxhelm's synchronous endpoint should handle this gracefully. If Archive needs to transcribe larger files, that would require switching to the batch API (future work).

**Suggested implementation order:** After C7 and C11 are deployed. This is the first production consumer validation.

---

### C9 -- podcast-transcript Voxhelm Backend

**Implementation note (2026-03-12):** Delivered as the first M1c slice. `podcast-transcript` now has a `Voxhelm` backend, `--backend voxhelm`, README/docs updates, and test coverage. It was validated against the deployed Voxhelm edge service. The follow-on assumption that podcast-pipeline would work without changes was not borne out by repo inspection; its transcribe command contract still needs a small compatibility step or wrapper.

**Purpose:** Add Voxhelm as a fourth backend to podcast-transcript, so it can use the `studio` service instead of local mlx-whisper, local whisper.cpp, or Groq API.

**Included scope:**

- New `Voxhelm` class in podcast-transcript's `backends.py` implementing the `TranscriptionBackend` protocol
- The backend sends audio to Voxhelm's `POST /v1/audio/transcriptions` endpoint (C7) for shorter files and can fall back to the batch API (C6) for longer files
- Uses `response_format=verbose_json` to get timestamped segments when the synchronous path is used
- Configuration: Voxhelm API base URL, API key, model preference (via podcast-transcript's existing settings mechanism)
- podcast-transcript can continue using its existing output pipeline, even though Voxhelm also exposes server-generated transcript artifacts for consumers that want them

**Explicitly excluded scope:**

- Changes to podcast-transcript's chunking, resampling, or output format logic
- Changes to podcast-pipeline (C10 or separate integration)

**Dependencies:** C7 (OpenAI-compatible endpoint), C6 (batch fallback for long episodes)

**Consumer(s):** podcast-transcript users (with future indirect podcast-pipeline usage once its transcribe command contract is adjusted)

**Primary interfaces:**

- `TranscriptionBackend.transcribe(audio_file, transcript_path)` -- writes Whisper-format JSON to transcript_path

**Acceptance criteria:**

- [x] `podcast-transcript` can transcribe an audio file using the Voxhelm backend
- [x] Output is valid Whisper-format JSON that the existing DOTe/Podlove/WebVTT conversion pipeline handles correctly
- [x] All four output formats (DOTe, Podlove, WebVTT, plaintext) are produced correctly from Voxhelm output
- [x] Configuration is documented
- [ ] podcast-pipeline can use podcast-transcript with the Voxhelm backend without modification

**Main risks:**

- Whisper JSON format differences between Voxhelm's response and what podcast-transcript expects (segment structure, field names). Mitigate by ensuring C7's `verbose_json` format matches the structure that mlx-whisper and whisper.cpp produce.

**Suggested implementation order:** After C7, with C6 available before enabling batch fallback. Can be developed in parallel with C8.

---

### C10 -- python-podcast / django-cast Integration

**Purpose:** Enable python-podcast / django-cast to trigger transcript generation via Voxhelm and store the results as Transcript model instances.

**Included scope:**

- New management command or service function in python-podcast/django-cast: `generate_transcript(audio_id)` that:
  1. Gets the Audio instance and its file URL
  2. Submits a batch transcription job to Voxhelm (`POST /v1/jobs`) with the required output formats
  3. Polls for completion and retrieves the job result
  4. Downloads the server-converted Podlove JSON, WebVTT, DOTe JSON, and plain-text artifacts from Voxhelm
  5. Creates or updates the `Transcript` model instance linked to the Audio
- Configuration: Voxhelm API base URL, API key, model preference (via django settings or env vars)
- Admin action: "Generate transcript" bulk action on Audio admin

**Explicitly excluded scope:**

- Automatic triggering (e.g., on Audio upload) -- that can be added later
- Speaker diarization (S3)
- Changes to django-cast's Transcript model schema
- TTS generation for episodes

**Dependencies:** C6 (batch transcription and server-side artifact generation)

**Consumer(s):** python-podcast / django-cast operators

**Primary interfaces:**

- Management command: `manage.py generate_transcripts [--audio-id ID]`
- Admin action on Audio model
- Voxhelm HTTP API (C2 or C7)

**Acceptance criteria:**

- A podcast episode audio can be transcribed via Voxhelm and the resulting Transcript model has valid Podlove, WebVTT, and DOTe files
- The admin bulk action works for selected Audio instances
- Long episodes (> 25 MiB) use the batch API and poll for results
- Existing manually-uploaded transcripts are not overwritten unless explicitly requested

**Main risks:**

- This is the most greenfield integration -- python-podcast/django-cast has no auto-transcription today. The scope needs to be kept narrow to avoid scope creep.
- Audio file accessibility: django-cast Audio files may be stored in Django's default storage (local or S3). Voxhelm needs a URL to fetch from, which may require presigned URLs or direct file upload.

**Suggested implementation order:** After C6 and C7 are working. This is a lower-priority M1 chunk that can slide to early M2 if needed.

---

### C11 -- Deployment Role (ops-library)

**Implementation note (2026-03-12):** Delivered. The role now deploys the HTTP app and worker launchd units, runs Django migrations before restarting services, renders the S3-compatible artifact env vars, and includes post-deploy API and worker health checks.

**Purpose:** Deploy Voxhelm to `studio` using the existing ops-library/ops-control patterns so it can be operated as a standard homelab service.

**Included scope:**

- Ansible role in ops-library: `voxhelm`
  - Sync/update Voxhelm repo on `studio`
  - Python venv setup, dependency installation
  - Django migrations
  - `uvicorn` launchd service for the control plane (`studio` runs macOS, not Linux — use launchd, not systemd)
  - Traefik ingress on `macmini` for private HTTPS
  - Launchd service(s) for Django Tasks workers in the M1b expansion
  - Environment file with MinIO credentials, auth tokens, backend configuration
  - Log rotation
  - Health check integration
- ops-control inventory and variables for `studio`
- FastDeploy registration (if applicable)
- Runbook: how to deploy, check status, view logs, restart, create tokens

**Explicitly excluded scope:**

- Wyoming sidecar deployment (C14)
- MinIO installation (assumed already available or deployed separately)
- Monitoring dashboards

**Dependencies:** C1 for initial service deployment; expand alongside C6 for worker and batch wiring

**Consumer(s):** Operator (Jochen)

**Primary interfaces:**

- `just deploy-one voxhelm` from ops-control
- launchd service on `studio`
- Traefik dynamic config on `macmini`

**Acceptance criteria:**

- `just deploy-one voxhelm` deploys the M1a service to `studio`
- `just deploy-one voxhelm` also refreshes the private HTTPS ingress on `macmini`
- The health endpoint is accessible from the Tailscale network
- Django Tasks worker processes are running and execute queued jobs once M1b is enabled
- Logs are rotatable and inspectable
- Service survives `studio` reboot

**Main risks:**

- `studio` runs macOS, not Ubuntu like macmini. The ops-library role must use launchd plists, not systemd units. Existing ops-library macOS patterns should be followed.
- Python dependency installation on macOS (especially mlx-whisper) may have build issues.

**Suggested implementation order:** Start after C1 so M1a can be deployed early. Expand it alongside C6 for M1b worker and MinIO wiring.

---

## M2 Chunks -- Home Assistant Voice Support

### C12 -- Wyoming STT/TTS Adapter

**Purpose:** Expose Voxhelm's STT (and later TTS) capabilities via the Wyoming protocol so Home Assistant can use them as local voice providers.

**Included scope:**

- Implementation based on S2 spike findings (sidecar process or embedded)
- Wyoming STT provider: receives audio stream, invokes Voxhelm STT backend, returns transcript text
- Wyoming TTS provider: receives text, invokes Voxhelm TTS backend (Piper or configured engine), returns audio stream
- Configuration: backend selection, model selection, language defaults
- If sidecar: standalone script to run the Wyoming server, managed via launchd on `studio`
- If embedded: Django management command that runs the Wyoming event loop

**Explicitly excluded scope:**

- Wake-word detection
- Satellite/device management (that stays in HA)
- Batch job integration (Wyoming is interactive only)

**Dependencies:**
- S2 (feasibility spike)
- C5 (STT backends) — required for Wyoming STT
- C15 (TTS backend adapter / Piper) — required for Wyoming TTS. **If C15 is not yet complete, C12 ships with STT only and TTS is added when C15 is ready.**

**Consumer(s):** Home Assistant

**Primary interfaces:**

- Wyoming protocol (TCP, typically port 10300 for STT, 10200 for TTS)

**Acceptance criteria:**

- Home Assistant can discover and configure the Wyoming STT provider
- A voice command spoken to an HA device is transcribed by Voxhelm
- Latency is acceptable for interactive voice (< 3 seconds for short utterances)
- (When C15 is available) Home Assistant can discover and configure the Wyoming TTS provider
- (When C15 is available) A TTS request from HA produces audible speech output

**Main risks:**

- Wyoming protocol compatibility issues. Mitigate with S2 spike.
- Latency requirements may be hard to meet with large models. Mitigate by using smaller/faster models for interactive.

**Suggested implementation order:** After S2 completes and C5 is available. TTS portion requires C15.

---

### C13 -- Interactive Lane Scheduling

**Purpose:** Ensure that interactive voice requests (from Wyoming/HA) are not blocked by long-running batch transcription jobs.

**Included scope:**

- Lane-aware worker scheduling: interactive jobs preempt or run on a separate execution path from batch jobs
- Option A: separate worker process for interactive lane
- Option B: priority queue with interactive jobs jumping ahead of batch
- Option C: resource reservation (e.g., interactive lane always has one processing slot reserved)
- Configurable concurrency limits per lane
- Monitoring: queue depth per lane visible via health/status endpoint

**Explicitly excluded scope:**

- GPU/memory isolation (out of scope for v1)
- Preempting running batch jobs (too complex for v1)

**Dependencies:** C3 (worker model), C12 (Wyoming adapter generates interactive workload)

**Consumer(s):** Home Assistant (indirect -- ensures responsive voice)

**Primary interfaces:**

- Configuration: `VOXHELM_INTERACTIVE_SLOTS`, `VOXHELM_BATCH_SLOTS`
- `GET /v1/status` -- includes queue depth per lane

**Acceptance criteria:**

- A running batch transcription job does not block an interactive STT request from completing within the latency target
- Queue depth per lane is visible in the status endpoint
- Configuration controls how many slots are reserved for each lane

**Main risks:**

- On a single-GPU host, two simultaneous transcriptions may OOM. Mitigate by limiting total concurrency to 1 for the GPU and using separate CPU-based backends for interactive if needed.

**Suggested implementation order:** After C3 and C12 are in progress. Can be a thin layer added to the existing worker model.

---

### C14 -- Home Assistant Integration

**Purpose:** Complete the Home Assistant integration: deploy Wyoming adapters, configure HA, and validate the end-to-end voice pipeline.

**Included scope:**

- Deploy Wyoming adapter(s) on `studio` (extend C11 deployment role)
- Configure Home Assistant to use Voxhelm Wyoming providers
- Test with Nabu Casa voice device and HA mobile app
- Document HA configuration (which Wyoming server to add, how to select it in Assist pipeline)
- Validate latency and reliability under normal conditions

**Explicitly excluded scope:**

- Wake-word infrastructure
- Custom satellite hardware setup
- OpenClaw voice integration

**Dependencies:** C12, C13, C11 (deployed service)

**Consumer(s):** Home Assistant users (Jochen)

**Primary interfaces:**

- Wyoming protocol endpoints on `studio`
- Home Assistant Assist pipeline configuration

**Acceptance criteria:**

- "Hey Nabu" or equivalent wake triggers STT on Voxhelm and TTS response is audible
- HA mobile app voice input works through Voxhelm
- Voice interactions complete within acceptable latency (< 5 seconds total round trip)
- Service is stable over 24 hours of normal use

**Main risks:**

- HA pipeline quirks or version-specific requirements. Mitigate by testing against current HA version.
- Network latency between HA (macmini) and Voxhelm (`studio`) over Tailscale.

**Suggested implementation order:** After C12 and C13. This is primarily integration testing and configuration, not heavy development.

---

## M3 Chunks -- TTS Batch Generation

### C15 -- TTS Backend Adapter Layer (Piper)

**Purpose:** Provide a pluggable TTS backend abstraction, starting with Piper, so the system can generate speech audio.

**Included scope:**

- `SynthesisBackend` protocol/interface: `synthesize(text: str, options: SynthesizeOptions) -> SynthesisResult`
- `SynthesizeOptions`: voice/model identifier, language, speed, output format (wav, mp3, ogg)
- `SynthesisResult`: audio data (bytes or path), metadata (backend, model, language, duration, processing time)
- Piper backend implementation: invoke Piper CLI or library
- Backend registry: lookup by identifier (`auto`, `piper`)
- Voice/model management: document how to install Piper voices, configurable default voice per language
- Audio format conversion: output in requested format via ffmpeg

**Explicitly excluded scope:**

- Batch job wiring (C16)
- Wyoming TTS (C12 will use this layer)
- Kokoro or other future TTS engines

**Dependencies:** C1

**Consumer(s):** C12 (Wyoming TTS), C16 (batch TTS)

**Primary interfaces:**

- Python API: `get_tts_backend(identifier: str) -> SynthesisBackend`
- Python API: `synthesize(text, options) -> SynthesisResult`

**Acceptance criteria:**

- Piper backend can synthesize a short text string to WAV audio
- Output audio is valid and playable
- Voice selection works (at least one English and one German voice)
- `auto` resolves to configured default
- `just test` passes with at least one integration test

**Main risks:**

- Piper installation on macOS. Piper is primarily Linux-focused. May need to test building from source on ARM macOS or use a container.

**Suggested implementation order:** Can start any time after C1. Not blocking M1.

---

### C16 -- Batch TTS Jobs

**Purpose:** Enable producers (primarily Archive) to submit text-to-speech jobs that generate audio artifacts.

**Included scope:**

- Task executor for `synthesize` job type:
  1. Run queued task
  2. Extract text from input (direct text field, or URL to fetch article text)
  3. Synthesize speech via C15
  4. Upload output audio artifact to MinIO
  5. Report success with artifact reference
- Synchronous TTS endpoint: `POST /v1/audio/speech` -- for short text, returns audio directly
- Input validation: max text length, allowed voice identifiers
- Configurable voice presets per producer

**Explicitly excluded scope:**

- HTML/article text extraction from URLs (manual text input only in v1)
- SSML support
- Multi-voice narration

**Dependencies:** C15, C3, C4

**Consumer(s):** Archive (article-to-audio), operator tools

**Primary interfaces:**

- `POST /v1/jobs` with `job_type: synthesize`
- `POST /v1/audio/speech` -- synchronous TTS

**Acceptance criteria:**

- A TTS job produces an audio file stored in MinIO
- The synchronous endpoint returns audio for short text
- Archive can submit a synthesis job and retrieve the resulting audio
- Output audio is valid and playable

**Main risks:**

- Long article synthesis may produce very large audio files. Mitigate with text length limits and chunked synthesis.

**Suggested implementation order:** After C15 and the M1 Django Tasks infrastructure are stable.

---

## M4 Chunks -- OpenClaw and Later Work

### C17 -- OpenClaw Integration

**Purpose:** Provide a documented, stable HTTP API surface that OpenClaw can use for voice and media processing.

**Included scope:**

- Document the stable API contract for OpenClaw (subset of existing endpoints)
- Example OpenClaw tool/plugin that calls `POST /v1/audio/transcriptions` and `POST /v1/audio/speech`
- Authentication: dedicated producer token for OpenClaw
- Rate limiting for interactive endpoints

**Explicitly excluded scope:**

- Deep OpenClaw integration
- Streaming voice responses
- Real-time conversation turn handling

**Dependencies:** C7 (STT endpoint), C16 (TTS endpoint)

**Consumer(s):** OpenClaw

**Primary interfaces:**

- `POST /v1/audio/transcriptions`
- `POST /v1/audio/speech`

**Acceptance criteria:**

- An OpenClaw tool can transcribe audio via Voxhelm
- An OpenClaw tool can generate speech via Voxhelm
- API documentation is published

**Main risks:**

- OpenClaw requirements may evolve. Mitigate by keeping the integration thin.

**Suggested implementation order:** After M3 is complete. Low priority.

---

## Implementation Sequence

### M0: Optional Spikes (can run in parallel with implementation)

```
S1 (STT Backend Spike) ──────────────────> informs later backend expansion
S2 (Wyoming Spike) ───────────────────────> feeds into C12
```

### M1a-M1b: Foundation

```
C1 (Skeleton + Auth) ─────┬──> C2 (Task Tracking) ──> C3 (Django Tasks)
                           │
                           ├──> C4 (MinIO Artifacts)
                           │
                           └──> C5 (STT Adapters) ─────────────────────┐
                                                                       │
                                C3 + C4 + C5 ──> C6 (Batch Pipeline) ──┤
                                                                       │
                                C5 ─────────────> C7 (OpenAI Endpoint) ┘
```

### M1c: Consumer Integration

```
C7 ──> C8 (Archive Integration)
C7 + C6 ──> C9 (podcast-transcript Backend)
C6 ──> C10 (python-podcast Integration)
C1/C6 ──> C11 (Deployment)
```

### M2: Home Assistant

```
S2 + C5 ──> C12 (Wyoming Adapter)
C12 + C3 ──> C13 (Interactive Scheduling)
C12 + C13 + C11 ──> C14 (HA Integration)
```

### M3-M4: TTS and Later

```
C1 ──> C15 (TTS Backends)
C15 + C3 + C4 ──> C16 (Batch TTS)
C7 + C16 ──> C17 (OpenClaw)
```

### Parallelism Notes

Within M1a-M1b, C2, C4, and C5 can all be developed in parallel once C1 is done. C3 depends on C2. C6 needs C3, C4, and C5. C7 only needs C5.

Within M1c, C8, C9, and C10 are independent consumer integrations once their upstream APIs exist. C11 starts in M1a and expands in M1b rather than waiting for consumer work.

M2 is largely sequential. S2 should be completed during M0 or early M1 so it does not block M2.

M3-M4 has no dependency on M1c, but M3 still depends on both M1b and M2.

---

## Key Design Decisions Embedded in This Plan

1. **Archive integration is via OpenAI-compatible endpoint (C7), not batch API.** Archive's current code is synchronous and uses the OpenAI multipart POST pattern. The fastest path to integration is matching that API exactly. Batch API integration for Archive is future work.

2. **podcast-transcript gets a Voxhelm backend (C9), not a replacement.** podcast-transcript already handles chunking, resampling, and library/CLI output workflows. Voxhelm is another transcription backend alongside mlx-whisper, whisper.cpp, and Groq. This keeps the remaining podcast-pipeline integration step small, but it does not eliminate it entirely because podcast-pipeline's current transcribe command contract is narrower than originally assumed.

3. **python-podcast/django-cast gets new integration code (C10).** This is the only consumer with no existing transcription path, so it needs the most new code. Its planned integration path is the batch API so it can rely on server-generated transcript artifacts.

4. **Wyoming is a sidecar or companion process, not embedded in Django.** Django's synchronous request-response model is a poor fit for Wyoming's long-lived TCP connections. The spike (S2) will confirm, but the expected answer is a separate process.

5. **MinIO from day one for async artifacts (C4).** Artifacts go to MinIO immediately for the batch path rather than starting with local filesystem and migrating later. This avoids a migration step and ensures artifact URLs are stable.

6. **Interactive/batch lane separation is an M2 concern (C13).** Until Home Assistant is integrated, all workloads are batch. Adding scheduling complexity before there is interactive traffic is premature.
