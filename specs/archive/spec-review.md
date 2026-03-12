# Spec Review: Voxhelm PRD

**Date:** 2026-03-11
**Reviewed by:** Product Architect, Systems Architect, Security Reviewer, Integrations Reviewer
**Input:** `2026-03-11_voxhelm_service.md`

---

## 1. Scope Discipline

### 1.1 v1 is too large

The PRD's Milestone 1 bundles three distinct consumer integrations (Archive, python-podcast, podcast-pipeline), three STT backends (WhisperKit, mlx-whisper, whisper.cpp), video-to-audio extraction, MinIO artifact storage, a full pull-worker model, and a Django control plane. This is at least six independently testable capabilities treated as one milestone.

The realistic smallest v1 is: one STT backend, batch job submission, worker claim/execute, artifact return, and one consumer integration (Archive, because it has the most concrete existing code). python-podcast and podcast-pipeline can follow immediately after but should not gate the first deployable version.

### 1.2 Interactive and batch are correctly separated -- but v1 should not include interactive at all

The PRD correctly identifies interactive (Wyoming/HA) and batch as different workloads. However, the Milestone 1 scope does not include interactive, and Milestone 2 is correctly sequenced. This is good. The concern is that the architecture description intermingles both throughout the document, making it unclear which parts of the "Core responsibilities" section are v1 and which are later.

**Recommendation:** Strip interactive-path language from the Milestone 1 scope description entirely. Interactive scheduling, lane separation, and low-latency tuning should only appear in the Milestone 2 spec.

### 1.3 Diarization appears multiple times but is correctly marked optional

The PRD lists `diarize` as an optional job type and mentions "optional speaker-labeled transcript output later." This is fine as written, but diarization should be explicitly excluded from all milestones and treated as a spike. pyannote.audio requires a HuggingFace token and separate model downloads -- it is operationally distinct from Whisper-family backends.

### 1.4 TTS is correctly deferred to Milestone 3

No issue here. The milestone ordering (batch STT -> Wyoming -> TTS -> OpenClaw) is sound.

---

## 2. Architecture Clarity

### 2.1 Control plane vs. worker ownership is underspecified

The PRD says the control plane should "validate requests, schedule work, isolate interactive and batch queues, fetch/download inputs, extract audio from video, normalize/transcode media, invoke STT/TTS backend, persist artifacts." This conflates control-plane responsibilities (validate, schedule, persist metadata) with worker responsibilities (download, extract, transcode, invoke backend). The OpsGate model that the PRD cites as inspiration is clear about this: the control plane stores tickets and state; the runner does all execution.

**Recommendation:** Make explicit:
- **Control plane owns:** job acceptance, validation, queue state, auth, artifact metadata, job status, MinIO references
- **Worker owns:** media download, audio extraction, transcoding, backend invocation, artifact upload to MinIO
- **Consumer apps own:** triggering jobs, polling for results, using returned artifact references

### 2.2 OpsGate parallel is useful but differences are not stated

The PRD says Voxhelm's pull-worker model is "inspired by OpsGate." Having reviewed OpsGate's actual code, the key differences must be acknowledged:

- OpsGate tickets carry an `execution_plan` with steps executed by AI agents via tmux sessions. Voxhelm jobs carry a declarative `job_type` + backend/model config executed by deterministic media processing code. There is no execution plan, no multi-step workflow, no agent delegation.
- OpsGate's runner uses tmux for interactive agent sessions. Voxhelm's worker will run Python/subprocess calls directly.
- OpsGate has an approval gate (pending_approval -> approved -> running). Voxhelm jobs presumably go directly from submitted to claimable.

The parallel is useful for the claim/status/heartbeat pattern and the auth separation (submitter tokens, runner token, operator session). The PRD should state what it borrows and what it does not.

### 2.3 Where does the Django service run?

The PRD says the service runs on the macstudio. The control plane is Django + SQLite. The workers also run on the macstudio. This means the control plane and workers are co-located. But Archive runs on the macmini. This raises the question: does Archive submit jobs to an HTTP API on the macstudio, or is there a control plane instance on the macmini?

**Recommendation:** State explicitly: the Django control plane runs on the macstudio. All remote consumers (Archive on macmini, HA, etc.) reach it over Tailscale HTTP. There is no control plane replica.

### 2.4 SQLite concurrency is adequate but constraints must be stated

SQLite with WAL mode (as OpsGate uses) handles modest concurrent writes. With one control plane and one worker process on the same host, contention is minimal. But if multiple worker processes or threads claim jobs concurrently, `SELECT ... FOR UPDATE` semantics differ from PostgreSQL. Django's SQLite backend does not support `select_for_update()` with row-level locking; it acquires a database-level write lock.

**Recommendation:** State explicitly: single worker process (or serialized claim) is required with SQLite. If parallel workers are desired later, PostgreSQL migration or an external queue (Redis, etc.) becomes necessary.

---

## 3. Integration Realism

### 3.1 Archive: synchronous OpenAI-compatible API is the path of least resistance

**Actual code behavior:** Archive's `generate_item_transcript()` (in `transcriptions.py`) downloads media itself (up to 25 MiB), then POSTs multipart form-data to `{ARCHIVE_TRANSCRIPTION_API_BASE}/audio/transcriptions` with fields `file`, `model`, and `prompt`. It expects a JSON response with a `text` field. This is a synchronous blocking call with a 300-second timeout.

Archive's enrichment worker (`run_metadata_worker.py`) is a management command that polls for pending items and processes them serially. It already handles the asynchronous dimension internally -- items are queued via `transcript_status=PENDING`, claimed atomically via `select_for_update()`, and processed one at a time.

**PRD assumption mismatch:** The PRD says Archive needs a "batch transcription jobs" API (async job submission, claim, status polling). But Archive's actual code does not have any job-tracking infrastructure for external async jobs. Archive's worker calls the transcription API synchronously and stores the result directly in the Item model's `transcript` TextField.

**Integration options for Archive:**
1. **OpenAI-compatible endpoint on Voxhelm** (simplest): Expose `POST /v1/audio/transcriptions` that accepts multipart form-data and returns `{"text": "..."}`. Archive changes only the env vars `ARCHIVE_TRANSCRIPTION_API_BASE` and `ARCHIVE_TRANSCRIPTION_API_KEY`. Zero code changes in Archive.
2. **Full async job API** (PRD's stated approach): Archive must be refactored to submit jobs, poll for completion, and retrieve artifacts. This requires significant Archive-side code changes.

**Recommendation:** Option 1 for Milestone 1. It unblocks Archive integration immediately. The async batch API can be built alongside it for python-podcast and podcast-pipeline, which do not have existing API contracts to preserve.

**25 MiB limit concern:** Archive downloads media and enforces a 25 MiB limit *before* sending to the API. Many podcast episodes exceed 25 MiB. This means Archive currently cannot transcribe long-form content. If Voxhelm accepts a URL reference instead of an uploaded file, this limit goes away on the Voxhelm side. But Archive's code would need modification to pass a URL instead of uploading bytes.

### 3.2 python-podcast / django-cast: greenfield integration, artifact format compatibility is the real requirement

**Actual code behavior:** django-cast's `Transcript` model stores three file fields: `podlove` (JSON), `vtt` (WebVTT), and `dote` (DOTe JSON). These are uploaded as static files. There is no auto-transcription trigger, no STT integration, and no job model.

podcast-transcript already produces all four output formats (DOTe, Podlove, WebVTT, plain text) from Whisper-family JSON output via its `single_track.py` module.

**Key insight:** The output format conversion logic already exists in podcast-transcript. Voxhelm does not need to implement DOTe/Podlove/WebVTT conversion itself. Voxhelm should return raw Whisper-format JSON (with segments, timestamps, text). Consumers or a thin client library can convert to their preferred format.

**Alternatively:** Voxhelm could accept `output.formats` in the job request and perform the conversion server-side. This is cleaner for consumers but means Voxhelm absorbs podcast-transcript's format conversion logic.

**Recommendation:** Voxhelm should return structured Whisper-format JSON as the canonical output. Format conversion should be a client-side concern initially, possibly with a Voxhelm-side option later. This keeps the service focused and avoids baking podcast-specific format knowledge into the media service.

### 3.3 podcast-pipeline: CLI compatibility is the integration surface

**Actual code behavior:** podcast-pipeline's `transcribe.py` runs an external command (`podcast-transcript` by default) via `subprocess.run()`. It expects `transcript.txt` and optional `chapters.txt` to appear in a mode-specific output directory.

**Integration path:** The simplest Voxhelm integration is a new podcast-transcript backend that calls Voxhelm's HTTP API instead of invoking local Whisper directly. podcast-pipeline does not need to know about Voxhelm at all -- it continues to invoke `podcast-transcript`, which gains a `--backend voxhelm` option.

**PRD assumption mismatch:** The PRD lists podcast-pipeline as a direct Voxhelm consumer needing "API-driven transcript generation for episode workspaces." In practice, podcast-pipeline delegates to podcast-transcript, which is the actual integration point. The PRD should acknowledge this indirection.

### 3.4 Home Assistant: Wyoming is the correct and only interface

No Wyoming code exists in the codebase yet. The PRD correctly identifies Wyoming as the integration protocol. The existing wyoming-faster-whisper and Piper ecosystem provides reference implementations. This is Milestone 2 and correctly deferred.

**Note:** wyoming-whisper-cpp is archived (as the PRD notes). If the Wyoming adapter wraps whisper.cpp, it needs a custom Wyoming server implementation, not the archived one.

### 3.5 OpenClaw: correctly deferred, no action needed in v1

The PRD places OpenClaw in Milestone 4. Given that OpenClaw's integration needs are unspecified and the HTTP API will be stable by then, this is correct.

---

## 4. Security Clarity

### 4.1 Auth model is sound but must be made concrete

The PRD specifies three auth levels: producer submission, worker claim/status, operator/admin. This mirrors OpsGate's proven model (submitter tokens, runner token, session auth). Good.

**Missing details:**
- How are producer tokens provisioned? Environment variables (like OpsGate) or a Django admin interface?
- Is there one producer token per consumer, or shared tokens?
- Does the OpenAI-compatible sync endpoint use the same token scheme?

**Recommendation:** Per-consumer producer tokens (one for Archive, one for python-podcast, etc.) via environment variables. The OpenAI-compatible endpoint should accept the producer token as a Bearer token in the Authorization header, matching Archive's existing `ARCHIVE_TRANSCRIPTION_API_KEY` usage.

### 4.2 URL fetch policy is a real security surface

The PRD lists "direct media URL" as an input type but does not define which URLs the worker may fetch. A URL input with no restrictions is an SSRF vector (Server-Side Request Forgery), even on a trusted network.

**Recommendation:** Define an allowlist of URL schemes (`https://`, `http://` for local services) and optionally an allowlist of hosts/domains. For v1, restricting to known hosts (Archive's domain, podcast feed domains) is sufficient.

### 4.3 MinIO access patterns need explicit scoping

The PRD says MinIO is used for artifact storage. But it does not specify who writes and who reads:

- **Worker writes** transcription results to MinIO
- **Control plane reads** MinIO references to serve results
- **Consumers read** artifacts from MinIO (directly? via Voxhelm proxy?)

If consumers read directly from MinIO, they need MinIO credentials or pre-signed URLs. If Voxhelm proxies artifact delivery, only Voxhelm needs MinIO credentials.

**Recommendation:** Voxhelm proxies artifact delivery via its HTTP API (e.g., `GET /v1/jobs/{id}/artifacts/{name}`). Consumers never access MinIO directly. This keeps the security boundary narrow.

### 4.4 No arbitrary execution paths exist -- good

The declarative job schema (job_type, backend, model, input, output) does not allow shell commands, arbitrary paths, or executable references. This is correct and should be maintained strictly. Backend IDs must be validated against a fixed allowlist, not treated as executable names.

---

## 5. Delivery Realism

### 5.1 Milestone 1 can be implemented without later decisions -- if scoped correctly

If Milestone 1 is reduced to:
1. Django control plane with job model and auth
2. One STT backend (mlx-whisper, since it is pure Python and easiest to integrate)
3. Worker claim/execute loop
4. MinIO artifact persistence
5. OpenAI-compatible sync endpoint (for Archive)
6. Async job API (for future consumers)

...then it can be built and deployed without decisions about Wyoming, TTS, diarization, or OpenClaw.

### 5.2 Three STT backends in M1 is ambitious

The PRD requires WhisperKit, mlx-whisper, and whisper.cpp all in Milestone 1. Each has different installation requirements, different output formats, and different invocation patterns. podcast-transcript's code shows that whisper.cpp requires WAV conversion and output format transformation, while mlx-whisper works directly with Python.

**Recommendation:** Ship Milestone 1 with one backend (mlx-whisper). Add whisper.cpp and WhisperKit as follow-up work within the same milestone or immediately after, once the backend abstraction is proven.

### 5.3 Spikes are needed before implementation

Three spikes should precede implementation:
1. **Backend benchmark:** Compare mlx-whisper, whisper.cpp, and WhisperKit on the macstudio for speed, quality, and memory usage on representative podcast audio.
2. **WhisperKit server evaluation:** WhisperKit offers a local server with an OpenAI Audio API. Evaluate whether Voxhelm should wrap WhisperKit's server or invoke it as a library.
3. **MinIO integration pattern:** Validate the MinIO deployment on macstudio and confirm the bucket/access pattern before building the artifact layer.

---

## 6. Contradictions and Ambiguities

### 6.1 "Synchronous endpoints for small/interactive requests" vs. batch-only v1

The PRD lists synchronous HTTP endpoints (`POST /v1/audio/transcriptions`) alongside async job endpoints (`POST /v1/jobs`). But the milestone plan puts interactive low-latency in Milestone 2. Are the synchronous endpoints part of Milestone 1 or Milestone 2?

**Resolution needed:** The synchronous transcription endpoint is needed in Milestone 1 for Archive compatibility (Archive's current code is synchronous). The "interactive" label in the PRD is misleading -- Archive's synchronous use is not low-latency voice; it is a blocking HTTP call with a 5-minute timeout. Clarify that `POST /v1/audio/transcriptions` is a synchronous batch convenience endpoint, not an interactive voice endpoint.

### 6.2 "MinIO-backed artifacts from the start" vs. plain text transcript return

Archive expects a JSON response with `{"text": "transcript content"}` from the sync endpoint. It does not fetch artifacts from MinIO. The PRD says "MinIO-backed artifacts from the start" but does not clarify whether the sync endpoint bypasses MinIO or stores-then-returns.

**Recommendation:** The sync endpoint should return the transcript directly in the HTTP response (for Archive compatibility). It may also store the result in MinIO for auditability. The async job API should use MinIO for artifact storage and return MinIO references in the job result.

### 6.3 "Pull-worker" model but control plane and worker are co-located

The PRD describes a pull-worker model where "a macstudio worker claims the job." But the control plane also runs on the macstudio. In OpsGate, the pull model makes sense because the control plane runs on the macmini and the runner runs on the macstudio -- different hosts. In Voxhelm, both are on the macstudio.

This is not a contradiction but a design choice that should be acknowledged. The pull model still provides clean separation of concerns and crash recovery, even when co-located. However, it also means the worker could use a simpler in-process queue (like Django management commands with a work loop, which is how Archive's own enrichment worker operates) without the HTTP claim/status overhead.

**Recommendation:** Keep the pull-worker model for architectural cleanliness and future flexibility (e.g., running workers on additional Apple Silicon hosts). But acknowledge that in v1, both processes run on the same machine.

### 6.4 Multiple references to "later" without milestone assignment

The PRD mentions several features as "later" or "optional later" without assigning them to a specific milestone:
- `analyze_media` job type
- `kokoro` TTS backend
- HTML/article source reference for text extraction
- Speaker-labeled transcript output (diarization)

**Recommendation:** Create an explicit "Future / Unscheduled" section rather than scattering "later" throughout the document.

---

## 7. Missing Decisions

### 7.1 Which STT backend is the default?

The PRD lists `auto` as a backend option but does not define what `auto` resolves to. This must be decided before implementation. See decision log D-04.

### 7.2 Should Archive upload files or pass URL references?

Archive currently downloads media itself and uploads bytes. If Voxhelm accepts URL references, the 25 MiB limit is bypassed. But this requires Archive code changes. See decision log D-05.

### 7.3 What is the canonical transcript output format?

The PRD lists "plain transcript text, structured transcript JSON, subtitle/time-aligned formats." But which JSON schema? Whisper's native format? OpenAI's response format? A Voxhelm-specific schema? See decision log D-06.

### 7.4 How does the worker invoke backends?

Does the worker import Python libraries (mlx-whisper), shell out to CLI tools (whisper-cli), or proxy to running server processes (WhisperKit server)? Each has different deployment, concurrency, and resource implications. See decision log D-07.

### 7.5 How are results delivered to async job consumers?

The PRD mentions "producer callback/update submission or producer polling." Which one? Callbacks require producers to expose an HTTP endpoint. Polling is simpler but higher latency. Webhooks are a middle ground. See decision log D-08.

### 7.6 Retention policy is unspecified

The PRD's Open Question 6 asks about retention for source media, intermediate audio, transcript artifacts, and generated speech audio. No default is proposed. See decision log D-09.

### 7.7 Service process topology

The PRD's Open Question 1 asks whether this is one binary or multiple processes. The answer affects deployment, monitoring, and the ops-library role. See decision log D-10.
