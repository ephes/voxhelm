# Decision Log: Voxhelm

**Date:** 2026-03-11
**Status:** Accepted defaults; the M1-M3 core runtime slices, including the first C13 lane-scheduling slice, the operator transcript UI, the shared transcript-output follow-on, and the first C21 speaker-output slice are implemented. Remaining planned work is later backend expansion, C21 diarization backend validation/named-speaker follow-ons, Archive article-audio consumer work, and M4/OpenClaw.

---

## D-01: What is the smallest credible v1?

**Context:** The PRD's Milestone 1 includes three consumer integrations, three STT backends, video extraction, MinIO, pull-worker model, and Django control plane. This is too broad to ship as one atomic milestone and risks the project stalling on scope.

**Options:**

| Option | Scope | Tradeoff |
|--------|-------|----------|
| A. Full M1 as written | All three backends, all three consumers, full async + sync API | High risk of delay; many moving parts before anything is deployable |
| B. Archive-first v1 | One backend (mlx-whisper), OpenAI-compatible sync endpoint, MinIO storage, Archive integration only | Ships fast; other consumers follow immediately; validates the architecture with real traffic |
| C. Skeleton-first v1 | Django control plane + job model + worker loop, no consumer integration | Proves architecture but delivers no user value |

**Recommended default:** Option B. Archive has the most concrete existing code, the clearest API contract (OpenAI-compatible), and can switch to Voxhelm with zero code changes if the sync endpoint is compatible. python-podcast and podcast-pipeline follow as the next increments.

**Blocks implementation:** Yes -- scope definition gates all planning.

---

## D-02: Is Home Assistant voice support part of v1, or Milestone 2?

**Context:** The PRD places Wyoming/HA in Milestone 2. The coordination spec asks whether it should be pulled into v1.

**Options:**

| Option | Tradeoff |
|--------|----------|
| A. v1 (Milestone 1) | Adds Wyoming protocol implementation, interactive scheduling, and low-latency tuning to an already large milestone |
| B. Milestone 2 (as written) | Keeps v1 focused on batch; HA voice waits until batch is proven and deployed |

**Recommended default:** Option B. Wyoming is a separate protocol with different latency requirements. It adds no value to Archive, python-podcast, or podcast-pipeline. The batch infrastructure (job model, worker, backend abstraction) built in M1 provides the foundation that M2 builds on.

**Blocks implementation:** No -- Milestone 2 can begin as soon as M1's backend abstraction is stable.

---

## D-03: Is TTS part of v1?

**Context:** The PRD places TTS in Milestone 3. No consumer currently needs TTS. Archive's article-to-audio feature is aspirational.

**Options:**

| Option | Tradeoff |
|--------|----------|
| A. Include in v1 | Adds Piper integration and synthesis job type; no consumer ready to use it |
| B. Milestone 3 (as written) | TTS follows after Wyoming; Archive article-to-audio ships later |

**Recommended default:** Option B. No consumer has TTS code or integration points today. Building TTS before anyone can consume it wastes effort.

**Blocks implementation:** No.

---

## D-04: Which STT backend is the initial default?

**Context:** The PRD lists WhisperKit, mlx-whisper, and whisper.cpp. The `auto` backend must resolve to a concrete default. Each backend has different characteristics on Apple Silicon.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| A. mlx-whisper | Pure Python, easiest to integrate and deploy; podcast-transcript already has a working MLX backend; native Apple Silicon optimization via MLX framework | Less battle-tested than whisper.cpp for long audio |
| B. whisper.cpp | Most mature; strong Apple Silicon support via Metal; proven in podcast-transcript | Requires WAV conversion, CLI invocation, output format transformation; C++ dependency |
| C. WhisperKit | Best Apple Silicon optimization; offers local server with OpenAI API | Least familiar; Swift-based; server mode evaluation needed |

**Recommended default:** Option A (mlx-whisper) for initial implementation. It is the lowest-friction path: Python-native, already proven in podcast-transcript, matches the current local Apple Silicon workflow, and runs well on the planned worker host. WhisperKit stays optional until there is a concrete reason to support it.

**Implementation note (2026-03-14):** The initial bootstrap choice was later overtaken by real `studio` benchmarking and implementation work. Voxhelm now ships `whisper.cpp`, `mlx-whisper`, and an experimental WhisperKit backend. The deployed default STT backend is still `whispercpp` with `mlx` as the configured fallback for `auto` requests. The current benchmark source of truth is `specs/2026-03-13_whisperkit_re_evaluation_studio.md`. That rerun keeps `whisper.cpp` as the deployed default, corrects the MLX baseline upward under Python 3.14, and justifies WhisperKit only as an explicit opt-in path on `studio` because the tuned long-form run still logged a GPU recovery error.

**Blocks implementation:** No. The initial default was enough to unblock implementation, and later benchmarks were free to overturn it without changing the producer-facing API.

---

## D-05: Should Archive upload files or pass URL references?

**Context:** Archive currently downloads media (up to 25 MiB) and uploads bytes via multipart form-data to the transcription API. Many podcast episodes exceed 25 MiB. The PRD lists both "uploaded audio file" and "direct media URL" as input types.

**Options:**

| Option | Tradeoff |
|--------|----------|
| A. File upload (status quo) | Zero Archive code changes; 25 MiB limit persists; large podcast episodes cannot be transcribed |
| B. URL reference | Archive passes the media URL; Voxhelm downloads directly; no size limit on Voxhelm side; requires minor Archive code change |
| C. Both | Sync endpoint accepts either file upload or URL; most flexible; slightly more complex endpoint |

**Recommended default:** Option C. The sync endpoint should accept both `file` (multipart) and `url` (JSON field) inputs from the start. Archive can switch over immediately with file-upload compatibility, then move to URL-based submission to avoid redundant downloading and the upload-size limit.

**Implementation note (2026-03-14):** The sync endpoint now accepts both multipart upload and JSON URL input as planned, but the large-media problem is not solved by that alone. Oversized/private/local media still needs an explicit batch large-input path so consumers do not have to invent chunking/transcoding around the 25 MiB sync ceiling.

**Blocks implementation:** No.

---

## D-06: What is the canonical transcript output format?

**Context:** Archive expects `{"text": "plain transcript"}`. podcast-transcript produces Whisper-format JSON with segments (start, end, text, id, seek). django-cast needs DOTe JSON, Podlove JSON, and WebVTT. podcast-pipeline expects `transcript.txt` (plain text).

**Options:**

| Option | Tradeoff |
|--------|----------|
| A. Whisper-native JSON as canonical | Segments with timestamps; consumers convert to their preferred format; matches what all backends produce natively |
| B. Voxhelm-specific schema | Normalize all backend outputs into one Voxhelm schema; backends differ in what they return |
| C. Consumer-requested format | `output.formats` field in job request; Voxhelm performs conversion server-side |

**Recommended default:** Option C with Whisper-native JSON as the internal canonical transcript representation. The sync OpenAI-compatible endpoint still returns `{"text": "..."}` (or `verbose_json` with segments) to match Archive's and podcast-transcript's expectations, but the batch/job layer should be the place where Voxhelm owns shared transcript conversions. Producer-requested output formats should converge on plain text, Whisper JSON, WebVTT, DOTe, and Podlove JSON so multiple consumers can rely on one service contract instead of duplicating conversion logic.

**Implementation note (2026-03-14):** This decision is now implemented on the batch path. Voxhelm exposes/stores `json`, `text`, `vtt`, `dote`, and `podlove` batch artifacts, while the sync OpenAI-compatible contract remains unchanged. `django-cast` now consumes the server-owned `dote` / `podlove` artifacts directly instead of rendering them locally from the JSON artifact.

**Blocks implementation:** No — the sync endpoint format is clear (OpenAI-compatible), and the async artifact/output model is now explicit.

---

## D-07: How does the worker invoke backends?

**Context:** Different backends have different invocation models. mlx-whisper is a Python library (`import mlx_whisper`). whisper.cpp is a CLI tool (`whisper-cli`). WhisperKit offers both a Swift library and a local server.

**Options:**

| Option | Tradeoff |
|--------|----------|
| A. In-process Python calls | Worker imports backends directly; simplest for mlx-whisper; requires Python bindings or subprocess for whisper.cpp |
| B. Subprocess/CLI invocation | Worker shells out to backend CLIs; uniform interface; extra process overhead; output parsing needed |
| C. HTTP proxy to backend servers | Worker calls backend HTTP servers (WhisperKit server, whisper.cpp server); adds operational complexity but clean separation |
| D. Mixed | mlx-whisper via Python import; whisper.cpp via subprocess; WhisperKit via HTTP; pragmatic but inconsistent |

**Recommended default:** Option D (mixed, pragmatic). This matches podcast-transcript's existing approach: MLX backend uses Python import, WhisperCpp backend uses subprocess. The backend abstraction layer (a Python protocol/interface class) hides the invocation difference from the worker. This is the pattern already proven in podcast-transcript's `TranscriptionBackend` protocol.

**Blocks implementation:** No -- the backend abstraction interface can accommodate any invocation style.

---

## D-08: How are results delivered to async job consumers?

**Context:** The PRD mentions "producer callback/update submission or producer polling" without choosing.

**Options:**

| Option | Tradeoff |
|--------|----------|
| A. Polling only | Consumer calls `GET /v1/jobs/{id}` periodically; simple; higher latency; no consumer-side HTTP endpoint needed |
| B. Webhook callback | Voxhelm POSTs to a consumer-provided URL on completion; lower latency; consumers must expose an endpoint |
| C. Polling with optional webhook | Default is polling; consumers can register a callback URL per job; flexible; slightly more complex |

**Recommended default:** Option A (polling only) for v1. None of the current consumers have webhook endpoints. python-podcast has no async infrastructure. Archive currently uses the synchronous OpenAI-compatible endpoint (not the batch API), so polling is not relevant to Archive's v1 path. The batch API with polling is primarily for python-podcast/django-cast. Adding webhooks later is a non-breaking extension.

**Blocks implementation:** No.

---

## D-09: What retention policy applies?

**Context:** PRD Open Question 6 asks about retention for source media, intermediate audio, transcript artifacts, and generated speech audio. No defaults proposed.

**Options:**

| Artifact type | Recommended default | Rationale |
|---------------|-------------------|-----------|
| Downloaded source media | 24 hours, then delete | Source is retrievable from origin; local copy is only for processing |
| Extracted intermediate audio (e.g., WAV from video) | Delete after job completion | Only needed during processing |
| Transcript artifacts (in MinIO) | Indefinite (until explicit deletion or policy sweep) | Artifacts are the value; consumers reference them |
| Generated speech audio (TTS, future) | Indefinite | Same as transcript artifacts |
| Job metadata (in SQLite) | 90 days, then archive or delete | Audit trail; not unbounded growth |

**Recommended default:** The above table. Implement cleanup as a periodic management command, not inline with job processing.

**Blocks implementation:** No -- defaults can be set at deployment time. But the cleanup mechanism should be designed into the job model from the start (e.g., `created_at` timestamps on all records).

**Implementation note (2026-10-06):** Source media and extracted intermediate audio are now pruned by `manage.py prune_job_artifacts` (non-exposed `source` artifacts of terminal jobs after `VOXHELM_SOURCE_ARTIFACT_RETENTION_SECONDS`, default 24 h; `extracted_audio` once the job is terminal). Transcript and speech artifacts stay indefinite. Scheduling the command (systemd timer) is still open.

**Implementation note (2026-10-06, job metadata):** `manage.py prune_job_artifacts` now also deletes the `Job` row of terminal jobs that finished more than `VOXHELM_JOB_METADATA_RETENTION_SECONDS` ago (default 90 days, `0` disables), after the artifact pass and only once the job owns no artifact row, queued object deletion or claimed staged upload. Delete, not archive. Jobs that still own indefinite transcript or speech artifacts keep their row, because those artifacts are served through the job; an archive/sweep policy for them remains open.

**Implementation note (2026-10-16, transcription profiles):** Profile jobs (D-26, `ad-analysis-v1`) are the first exception to "transcripts indefinite": their producer deletes them with `DELETE /v1/jobs/{id}` (artifacts, objects, queued deletions, a claimed staged upload, then the row; same allowlist and reference checks as the prune), and `prune_job_artifacts` applies the profile's 7-day backstop, deleting every artifact of any kind and then the row of a terminal profile job that finished more than 7 days ago, also with job metadata retention disabled. The row goes too because it carries the transcript text. Both paths also delete what the job's remote attempts stored below their artifact prefix (recorded at claim time as `RemoteAttemptPrefix`, listed in the store, so objects of a rejected or failed completion are found), a staged upload the job still holds (same allowlist, reference check and write lock as artifacts), and the job's `DBTaskResult`; a profile task returns only its job id and profile.

---

## D-10: What is the service process topology?

**Context:** PRD Open Question 1. Affects deployment, monitoring, and the ops-library Ansible role.

**Options:**

| Option | Processes | Tradeoff |
|--------|-----------|----------|
| A. Single Django process | Django serves HTTP + runs worker in a management command thread | Simplest ops; worker crash affects API; GIL contention during transcription |
| B. Django + Django Tasks workers | Django serves HTTP; Django Tasks workers run queued async work as separate processes | Clean separation; standard Django async pattern; fits existing production usage |
| C. Django + Django Tasks workers + Wyoming sidecar | Option B plus a separate Wyoming-protocol process for HA (Milestone 2) | Most flexible; slightly more complex launchd setup |

**Recommended default:** Split by milestone:

- **M1a:** single Django + `uvicorn` process for the synchronous Archive-first path
- **M1b:** Django + Django Tasks workers for batch execution
- **M2:** add a separate Wyoming sidecar/listener

Async transcription work must run outside the HTTP-serving process once the batch API exists. Django Tasks is the preferred mechanism for that separation, but that worker runtime is intentionally deferred out of M1a.

**Initial M1b runtime shape:** start with `django_tasks.backends.database.DatabaseBackend` against the same local SQLite database on `studio`. The initial `TASKS` configuration should stay conservative and close to the proven Steel Model setup:
- `database_alias: default`
- `poll_interval: 1.0`
- `max_attempts: 3`

Voxhelm should persist its own producer-facing job record and store the returned Django task/result identifier as an internal linkage field. Do not start with a separate custom worker registry, launch-token handshake, or heartbeat subsystem unless the stock database-backed runtime proves insufficient.

**Blocks implementation:** No -- but the Ansible role design depends on this.

**Clarification:** The producer-facing contract is still the common Voxhelm HTTP API (`/v1/audio/transcriptions`, `/v1/jobs`, `/v1/jobs/{id}`). Django Tasks workers are an internal execution detail, not a separate producer integration model.

---

## D-11: Is OpenClaw integration part of v1?

**Context:** The PRD places OpenClaw in Milestone 4. The coordination spec asks whether it belongs in v1.

**Options:**

| Option | Tradeoff |
|--------|----------|
| A. Include in v1 | No known OpenClaw requirements; would add speculative scope |
| B. Milestone 4 (as written) | OpenClaw uses the stable HTTP API when ready; no v1 work |

**Recommended default:** Option B. OpenClaw's integration needs are unspecified. The HTTP API built for other consumers will be sufficient when OpenClaw is ready.

**Blocks implementation:** No.

---

## D-12: Is diarization a v1 feature, a spike, or a later milestone?

**Context:** The PRD lists `diarize` as an optional job type and mentions pyannote.audio. Diarization adds significant complexity (separate model, HuggingFace token, post-processing to merge diarization with transcription).

**Options:**

| Option | Tradeoff |
|--------|----------|
| A. v1 feature | Adds pyannote dependency, speaker labeling logic, and new output format to an already large scope |
| B. Spike before Milestone 2 | Evaluate feasibility and quality on representative audio; inform whether it is worth building |
| C. Later milestone (post-M4) | Defer entirely; no v1 or near-term work |

**Recommended default:** Option C. Defer diarization from v1 and the near-term core runtime work.

**Update (2026-05-18):** django-cast now has a concrete backlog item for speaker diarization in generated podcast transcripts. That does not change the v1 decision, but it does move diarization from "purely hypothetical later" to a later-work spike/output follow-on. Track it as C21: first prove a diarization backend on representative podcast audio, then decide whether to add speaker labels as an explicit option on `job_type=transcribe`, a separate `diarize` job type, or both.

**Update (2026-05-19):** The first C21 output slice chose an explicit top-level option on batch `job_type=transcribe`: `diarization: {"enabled": true}`. Omitted or `false` keeps existing behavior. Voxhelm normalizes backend speaker IDs to generic `Speaker N` labels, aligns diarization turns to STT segments by largest timestamp overlap, and propagates labels into verbose JSON, DOTe `speakerDesignation`, and Podlove `speaker` / `voice`. WebVTT remains unchanged for this slice. A separate `job_type=diarize`, named-speaker mapping, contributor mapping, and mandatory diarization remain deferred. The pyannote path is guarded and optional; the heavyweight diarization stack is isolated behind the `diarization` optional dependency extra and still requires backend quality/runtime validation before production use.

**Update (2026-05-27):** Representative full-episode testing on Python Podcast audio showed that anonymous pyannote diarization with an exact speaker-count hint is not sufficient for known-speaker podcast workflows. The request and metadata path worked, but pyannote merged one real speaker into another cluster and produced a tiny fourth label. Tuning can improve this but is not a durable production answer. Known-speaker embedding experiments using clean contributor reference material were much stronger, so C21 should split into: anonymous diarization as fallback/debug metadata, quality warnings for pathological clusters, and a later explicit known-speaker postprocessor that classifies transcript segment windows against private contributor references and returns candidates, margins, confidence, raw diarization labels, and uncertainty flags. django-cast should store private contributor voice references and reviewable speaker suggestions; Voxhelm should own model-versioned embedding extraction and centroid caching.

**Blocks implementation:** No.

---

## D-13: Should python-podcast and podcast-pipeline target a common HTTP API, or should one get a compatibility shim?

**Context:** python-podcast (django-cast) has no existing transcription API -- it is greenfield. podcast-pipeline delegates to podcast-transcript, which has a `TranscriptionBackend` protocol.

**Options:**

| Option | Tradeoff |
|--------|----------|
| A. Common HTTP API for both | Both consume Voxhelm's async job API directly; clean but requires python-podcast to build job-submission code |
| B. podcast-transcript gets a Voxhelm backend; python-podcast uses HTTP API | podcast-pipeline continues using podcast-transcript (with a new `--backend voxhelm` option); python-podcast builds direct HTTP integration |
| C. Thin client library wrapping the HTTP API | Both python-podcast and podcast-pipeline use a shared `voxhelm-client` Python package |

**Recommended default:** Option B for initial delivery. podcast-pipeline's integration is naturally through podcast-transcript because that is its current workflow boundary, and adding a Voxhelm backend to podcast-transcript is minimal work. python-podcast builds direct HTTP integration since it has no existing CLI dependency.

**Clarification:** `podcast-transcript` is not just a CLI; it is also an importable Python package today. That still does not require Voxhelm to depend on it as a runtime library. Reusable code can be shared later if a stable extraction point emerges.

**Blocks implementation:** No -- but affects how consumer integration chunks are scoped.

---

## D-14: Is Django + SQLite good enough for the control plane?

**Context:** The PRD recommends Django + SQLite. The coordination spec asks for explicit constraints.

**Assessment:** Yes, with stated constraints:
- **Local-process coordination:** Django HTTP handlers and Django Tasks workers share one SQLite database on the same host. Write contention must stay low and task/job state transitions must be serialized carefully.
- **No row-level locking:** Django's SQLite backend does not support true `select_for_update()`. Any producer-facing tracking around queued/running work must respect SQLite's coarse write-lock behavior.
- **No remote DB access:** All SQLite access is local to `studio`. Consumers interact only via HTTP.
- **Scale ceiling:** If multiple workers or high task throughput is needed later, migration to PostgreSQL is the natural path. Design the models and task integration to be database-agnostic where practical.

**Recommended default:** Django + SQLite for v1. Document the low-concurrency constraint and keep the door open for PostgreSQL later.

**Operational note:** The accepted v1 task backend is also SQLite-backed (`django_tasks.backends.database.DatabaseBackend`), so Voxhelm must keep transactions short, enable WAL mode, and treat worker count on `studio` as intentionally low.

**Blocks implementation:** No.

---

## D-15: Which interfaces should be native vs. wrappers?

**Context:** The PRD lists HTTP API, Wyoming, and pull-worker as external interfaces. The active planning package now treats the producer-facing HTTP API as the stable contract and treats worker execution as internal service plumbing unless later scaling needs force a separate interface.

**Assessment:**

| Interface | Native or Wrapper | Rationale |
|-----------|-------------------|-----------|
| HTTP API (sync + async) | Native Django | Core control plane functionality; Django REST framework or plain Django views |
| Async task execution | Native Django Tasks workers | Default v1 path. Tasks run on `studio` as part of the Django deployment model |
| Optional worker claim/status API | Deferred / optional internal interface | Only needed if Voxhelm later adds remote workers or wants stricter process isolation beyond Django Tasks; not part of the producer-facing contract |
| Wyoming STT | Wrapper/sidecar | Wyoming protocol is TCP-based, not HTTP. Implement as a separate Python process that accepts Wyoming connections and proxies to Voxhelm's backend abstraction |
| Wyoming TTS | Wrapper/sidecar | Same as Wyoming STT |
| Backend invocation | Mixed (see D-07) | mlx-whisper: Python import. whisper.cpp: subprocess. WhisperKit: evaluate in spike |

**Blocks implementation:** No.

---

## D-16: Which parts must use MinIO from day one?

**Context:** The PRD says "MinIO-backed artifacts from the start." The coordination spec asks whether this is truly necessary from day one.

**Options:**

| Option | Tradeoff |
|--------|----------|
| A. MinIO from day one | All artifacts stored in MinIO; clean architecture; requires MinIO to be operational on `studio` before first job |
| B. Local filesystem first, MinIO later | Artifacts stored in a local directory initially; simpler bootstrap; migration to MinIO adds work later |
| C. MinIO for async job artifacts; local filesystem for sync endpoint results | Sync endpoint returns results inline (no storage needed); async jobs store in MinIO; balanced approach |

**Recommended default:** Option C. The sync endpoint (for Archive) returns results directly in the HTTP response -- no artifact storage needed. Async jobs (for python-podcast, podcast-pipeline) store artifacts in MinIO. This lets the sync endpoint work even if MinIO has an issue, while establishing the MinIO pattern for the async path from the start.

**Blocks implementation:** Partially -- MinIO must be deployed on `studio` before async jobs work. The sync endpoint can work without MinIO.

---

## D-17: Is `studio` the only worker host?

**Context:** The PRD assumes `studio` is the primary execution host. But the batch-worker model could later support multiple workers on different hosts.

**Assessment:** For v1, `studio` is the only worker host. The internal task-worker design should avoid hardcoding host-specific paths so additional Apple Silicon workers remain possible later if the control plane or task backend evolves.

**Blocks implementation:** No.

---

## D-18: Top three technical spikes before implementation

**Context:** The coordination spec requires identifying spikes.

**Recommended spikes (non-blocking):**

### Spike 1: STT backend benchmarks on `studio`
- **Goal:** Compare mlx-whisper, whisper.cpp, and WhisperKit on representative podcast/video audio (German and English, short and long).
- **Outputs:** Speed (real-time factor), peak memory, transcript quality comparison, Apple Silicon GPU utilization.
- **Duration:** 1-2 days.
- **Implementation note (2026-03-13):** Delivered and re-run. The current record is `specs/2026-03-13_whisperkit_re_evaluation_studio.md`. It still supports the current `whisper.cpp` default on `studio`, but only after correcting two earlier blind spots: MLX under Python 3.14 and WhisperKit under the newer `large-v3-v20240930` GPU-tuned path.
- **Blocks:** No longer blocks implementation; now serves as recorded evidence for the current default and for the next WhisperKit decision.

### Spike 2: WhisperKit server evaluation
- **Goal:** Determine whether WhisperKit's local OpenAI-compatible server is suitable as a Voxhelm backend (vs. wrapping WhisperKit as a library/CLI).
- **Outputs:** API compatibility assessment, performance comparison with direct invocation, deployment complexity.
- **Duration:** 0.5-1 day.
- **Implementation note (2026-03-13):** Re-evaluated. WhisperKit's local server is current and viable enough for a smoke-tested OpenAI-compatible transcription flow on `studio`; see `specs/2026-03-13_whisperkit_re_evaluation_studio.md`. This spike is no longer blocked on basic feasibility. The remaining blocker is operational confidence, especially the long-run Metal GPU recovery error observed during the tuned long-form run.
- **Blocks:** No; the open question is now stability/operability, not whether the server mode exists.

### Spike 3: MinIO deployment and integration pattern on `studio`
- **Goal:** Deploy MinIO on `studio`, validate bucket creation, confirm Python client (boto3 or minio) works, test artifact upload/download round-trip.
- **Outputs:** Working MinIO instance, verified access pattern, deployment notes for ops-library role.
- **Duration:** 0.5 day.
- **Blocks:** Async artifact work, but not the Archive-first sync endpoint.

---

## D-19: What is the first accepted C13 lane-scheduling design on `studio`?

**Context:** As of 2026-03-13, Voxhelm is live on one Apple Silicon host (`studio`) with three long-lived processes: the Django HTTP API, one Django Tasks worker, and the Wyoming STT/TTS sidecar. STT and TTS are each serialized today behind separate in-process Python locks, but those locks are modality-local rather than host-wide. STT and TTS can still overlap within one process, and there is no coordination across processes. That means a long transcription or synthesize run can still contend with the Wyoming sidecar for CPU, RAM, and model cache state.

**Options:**

| Option | Tradeoff |
|--------|----------|
| A. Separate interactive worker/runtime | Stronger isolation, but adds new process topology, queueing, and deployment complexity immediately |
| B. Reserved parallel slots per lane | Implies useful concurrent inference on one host; risky for memory pressure and not clearly needed on `studio` |
| C. Host-wide admission control plus cooperative serialization | Minimal change: one shared runtime gate on `studio`, Wyoming gets priority when competing with HTTP/batch work, no preemption of running work |

**Recommended default:** Option C.

**Implementation note (2026-03-14):** Delivered. Voxhelm now ships the session-authenticated operator UI at `/`, mixed sync/batch input routing, and server-owned `dote` / `podlove` batch outputs. The same-epic `django-cast` cleanup that consumes those outputs is also complete.

The first C13 slice should use a thin host-wide scheduler around local inference calls, not a new queueing system. Concretely:

- keep the existing three-process deployment shape on `studio`
- add one shared scheduler state/lock directory on local disk
- classify **interactive lane** as Wyoming STT/TTS requests only
- classify **non-interactive lane** as all non-Wyoming local inference on `studio`: batch `transcribe`, batch `synthesize`, `POST /v1/audio/transcriptions`, and `POST /v1/audio/speech`
- allow only one local inference holder at a time across processes
- use that same single admission slot for both STT and TTS inference
- give Wyoming requests admission priority over queued non-interactive work
- do **not** preempt a batch or HTTP request that is already running
- do **not** expose `lane=interactive` on the public batch API in this slice; `Job.lane` remains `batch` for producer-submitted jobs, and the scheduler's `non-interactive` lane is an internal runtime concept only

**Feasible guarantee on one host:** Voxhelm can prevent new batch/HTTP inference from starting ahead of a waiting Wyoming request, but it cannot guarantee a hard latency bound if a non-interactive inference is already in progress when the interactive request arrives.

**Operator API decision:** `GET /v1/status` remains deferred. The first slice should rely on log-based verification and black-box live tests rather than broadening into a new status surface.

**Config shape:** Add only the minimum knobs needed to operate the shared scheduler, for example:

- enable/disable flag
- shared state directory path
- stale-lock timeout / recovery window (historical; see the 2026-10-13 D-24 amendment)

**Recommended initial default:** `VOXHELM_LANE_SCHEDULER_STALE_SECONDS=1800`.

That default is intentionally conservative because the first slice does not yet define a lease-heartbeat mechanism, and long non-interactive inference on `studio` can legitimately run for many minutes. Lower values are safer for crash recovery but raise the risk of falsely reclaiming a live holder.

Do not add slot-count tuning in the first slice because the accepted design is single-slot cooperative serialization, not reserved parallel capacity.

**Blocks implementation:** Yes -- this is the reviewed C13 design baseline.

---

## D-20: What is the smallest sensible operator transcript slice after M3?

**Context:** At the time of this decision, Voxhelm already had the sync and batch STT primitives, but no human HTML UI. A second transcript consumer wanted the same DOTe/Podlove shapes that `django-cast` then rendered locally, which changed the boundary: a consumer-local conversion decision would create duplication instead of preserving a clean service contract.

**Options:**

| Option | Tradeoff |
|--------|----------|
| A. Separate operator app; keep DOTe/Podlove consumer-local | Avoids touching Voxhelm's HTML/auth surface, but duplicates transcript conversion logic across the operator UI and `django-cast` |
| B. Voxhelm-owned operator UI plus immediate async upload support for all file types | One product surface, but broadens the slice into a new upload pipeline before the current batch contract is ready |
| C. Voxhelm-owned operator UI with mixed sync/batch routing, server-owned DOTe/Podlove batch outputs, and async uploads deferred | Smallest coherent slice that reuses the shipped APIs, fixes the shared-output boundary, and avoids inventing a new large-upload path prematurely |

**Recommended default:** Option C.

The follow-on slice should be defined as:

- the operator web UI lives inside Voxhelm's existing Django app, not as a separate service
- auth for that UI uses Django session auth on the existing private ingress; browsers do not use producer bearer tokens directly
- the root route `/` should be the operator login/landing flow; after login it becomes the place where operators submit work and see their recent transcripts
- the first UI slice is **mixed**, not sync-only or batch-only:
  - URL audio -> sync `POST /v1/audio/transcriptions`
  - uploaded audio -> sync `POST /v1/audio/transcriptions`
  - URL video -> batch `POST /v1/jobs`
  - uploaded video -> deferred until batch upload support exists
- the current 25 MiB sync upload limit remains in place for uploaded-audio convenience because it matches the shipped sync API contract; this does **not** make sense as the long-form podcast path, so long audio/video should use URL input and the batch path instead
- large async file uploads are explicitly **deferred**; the first slice should tell operators to use a fetchable URL when work needs the batch path
- Voxhelm becomes the canonical producer of DOTe and Podlove for batch transcription outputs, using Whisper-native JSON as the internal canonical representation
- the sync API contract stays OpenAI-compatible (`json`, `text`, `verbose_json`, `vtt`) rather than growing new sync response formats for DOTe/Podlove
- the first UI result surface should show the transcript inline plus download links for the available output formats instead of only returning raw files
- the first UI should also show a recent-transcripts list scoped to the authenticated operator's submissions
- the Homelab tile is part of the same follow-on slice as a tiny discoverability step once the UI route exists, but Homelab stores only service metadata (name, URL, description), not Voxhelm product behavior
- deployment should provision one initial operator account for Jochen so the UI is usable immediately after rollout
- Archive article-audio remains a separate TTS consumer follow-on and is intentionally not grouped into this transcript/UI slice

**Recommended Homelab tile target:** name `Voxhelm`, URL `https://voxhelm.home.xn--wersdrfer-47a.de/`, description `Operator UI for local audio/video transcription and transcript exports`.

**django-cast recommendation:** keep the local `render_dote(...)` and `render_podlove(...)` code until the Voxhelm web flow and server-owned outputs are working, then do the `django-cast` simplification as the next follow-up in the same epic.

**Blocks implementation:** No -- the runtime foundation already exists. This decision only gates the next follow-on planning slice.

---

## D-21: How should python-podcast / django-cast make Wagtail transcript generation non-blocking?

**Implementation note (2026-03-14):** Delivered. The consumer chose option A in a narrow form: `django-cast` / `python-podcast` now use `django-tasks` with the database backend for transcript completion only. Wagtail Episode and Audio transcript actions submit to Voxhelm, persist a local one-to-one `TranscriptGeneration` status record on `Audio`, enqueue background completion, and return editors immediately to a usable admin surface. The background task polls Voxhelm, downloads `podlove` / `dote` / `vtt`, persists them onto the existing `Transcript` model, and updates local queued/running/succeeded/failed state. The Audio edit-view hardening follow-on is part of the same delivered slice: transcript generation now uses a dedicated POST form rather than piggybacking on the main audio edit form.

**Context:** The current Wagtail `Generate transcript` action in `python-podcast` / `django-cast` still waits inside one admin POST while Voxhelm transcribes, polls, downloads artifacts, and persists the `Transcript`. That is tolerable for short jobs, but it is poor editor UX for longer episodes and can lose to request timeouts even when Voxhelm itself finishes correctly. Staging validation also showed that the Audio edit-view action/refresh path needs hardening: after transcript generation, the Wagtail admin UI can render in a broken state instead of returning cleanly to a usable editor surface.

**Options:**

| Option | Tradeoff |
|--------|----------|
| A. Add `django-tasks` (or an equivalent app-level task runner) to `django-cast` / `python-podcast` | Reuses a known async pattern and gives explicit task records, but adds a new consumer-side dependency/runtime surface |
| B. Use Django's newer built-in task primitives once the consumer stack/runtime fit is clear | Avoids committing to a third-party task dependency too early, but depends on framework/version fit and still needs concrete persistence/execution design |
| C. Keep the current synchronous Wagtail action | No new consumer runtime, but editor UX remains poor and long transcripts stay exposed to request timeouts |

**Recommended default:** treat this as the next consumer follow-on and keep the execution substrate open until implementation. The spec should require bounded background completion plus editor-visible status/results, but should not prematurely force either `django-tasks` or Django's built-in task mechanism before the consumer repo chooses the better fit.

**Chosen implementation shape:** a narrow `django-tasks` database-backed worker path plus a consumer-local `TranscriptGeneration` record holding per-audio queued/running/succeeded/failed status, Voxhelm job linkage, and last error for editor-visible feedback. This stayed inside the transcript workflow and did not broaden into a generic job framework.

**Minimum shape for the follow-on:**

- Wagtail transcript generation returns quickly after enqueue/submission instead of holding the browser open for the full transcription duration
- the consumer persists enough local state to show queued/running/succeeded/failed status back to editors
- the background path is responsible for polling Voxhelm, downloading artifacts, and updating the existing `Transcript` model
- the Wagtail admin flow remains usable on both Episode and Audio edit views after submission/completion instead of leaving the editor on a broken or partially rendered form state
- retries and duplicate submissions remain bounded and coherent with Voxhelm `task_ref`
- the management command remains optional operator tooling, not the editor-facing solution

**Blocks implementation:** No -- delivered on 2026-03-14.

---

## D-22: How should Voxhelm handle oversized or private/local media without pushing chunking into consumers?

**Context:** The sync OpenAI-compatible endpoint intentionally keeps its 25 MiB upload ceiling and remains the convenience path for smaller files. Batch transcription already owns media download, normalization, extraction, and execution once media is local, but the producer-facing contract is still URL-only. That leaves consumers with oversized private/local media tempted to implement their own chunking/transcoding or to expose public URLs just to reach Voxhelm.

**Options:**

| Option | Tradeoff |
|--------|----------|
| A. Raise the sync upload limit | Smallest code change, but breaks the intended sync-small/batch-large boundary and still keeps long jobs inside one request |
| B. Let each consumer chunk/transcode locally and keep Voxhelm URL-only | Avoids changing Voxhelm's API, but duplicates media-processing logic and creates long-term contract drift across consumers |
| C. Add an explicit large-media batch input path in Voxhelm | Preserves the small sync API, keeps large/private/local media on a deliberate batch path, and centralizes chunking/transcoding/stitching once media is local |

**Recommended default:** Option C.

**Implementation note (2026-03-15):** Delivered in a narrow first pass. Voxhelm now exposes `POST /v1/uploads` for producer-authenticated staged batch audio, and `POST /v1/jobs` accepts `input.kind=upload` plus `input.upload_id`. Staged media lives in the configured artifact backend until the worker materializes it, copies it into the normal job-owned source artifact, and deletes the temporary staged object. Unclaimed staged uploads expire after `VOXHELM_STAGED_INPUT_RETENTION_SECONDS` and are opportunistically cleaned on later staging/submission requests. URL audio and URL video remain unchanged. Uploaded video and true chunk splitting/stitching are still explicitly deferred, so this closes the reusable large/private/local audio handoff gap without pretending the full chunking follow-on is done.

**Implementation note (2026-10-12, staging failure cleanup):** `POST /v1/uploads` no longer leaves residue when staging fails. A failed copy of the upload stream removes its partial local temp file. The `StagedMedia` row is now committed (durable transaction, so it must not run inside a caller's transaction) before the object is uploaded; a failed upload or final row update deletes the object and row again, and if that compensation fails the row stays (or is written again, already expired) as the record that expiry cleanup uses to retry the object deletion. An in-progress registration carries a 24-hour expiry so concurrent expiry cleanup cannot remove it mid-upload; the configured retention counts from the finished upload. Previously a failed row save after a successful upload left an object no row pointed at, invisible to expiry cleanup.

**Recommended shape:**

- keep `POST /v1/audio/transcriptions` unchanged as the small-file convenience path
- keep the current sync upload limit in place rather than silently broadening it
- extend the batch contract with an explicit large-input path, most likely `input.kind=upload` and/or a staged-upload handle that resolves to local object storage before work starts
- once media is local inside Voxhelm, the worker owns normalization, existing video-to-audio extraction where applicable, oversized-media chunking/transcoding, per-chunk transcription, and final transcript stitching
- whole-job failure is preferred over silent partial success when a chunk fails unless a stronger resumability design is introduced deliberately later
- the consumer contract should be reusable across Archive and future consumers, not Archive-specific

**Blocks implementation:** No -- this is a backlog and contract decision for the next Voxhelm media-input follow-on.

---

## D-23: How should Voxhelm add `atlas.local` as a second transcription worker?

**Status:** Accepted. The server-side control-plane slice is implemented as of
2026-05-30; the installable worker command, public PyPI publication, deployment,
and production `atlas.local` validation remain pending. Detailed implementation
concept: `specs/remote-transcription-workers.md`.

**Context:** Voxhelm's producer-facing batch API is already stable, but batch execution currently assumes local Django Tasks workers on `studio`. A second Apple Silicon machine, `atlas.local`, can reduce the transcription backlog if it connects to Voxhelm, claims work, runs STT and the required diarization/known-speaker path, uploads artifacts, and reports results without changing consumer APIs.

This reopens a complexity boundary that D-10 and `interface-map.md` intentionally deferred: custom worker registry, worker auth, heartbeat, and leases. Remote workers are the concrete trigger that makes that previously deferred worker coordination worthwhile. Any future local `studio` pull-worker path must still preserve the C13 host-wide lane scheduler for STT/TTS inference; that scheduler is host-local and does not coordinate `atlas.local`.

**Options:**

| Option | Tradeoff |
|--------|----------|
| A. Shared database workers | Smallest app-level change if Voxhelm first migrates to PostgreSQL, but gives remote workers broad DB access and couples them to full Django internals |
| B. Internal HTTP pull-worker API | Clean worker trust boundary; `studio` keeps DB ownership; workers claim leased jobs over HTTP; requires a new worker API and daemon |
| C. HTTP pull-worker API for both `studio` and remote workers | Clean steady state with one transcription execution model, but larger than the first `atlas.local` slice because it changes local `studio` worker deployment too |
| D. Ad-hoc SSH/NFS offload | Quick for experiments, but brittle and contrary to Voxhelm's goal of avoiding generic remote shell execution |

**Accepted decision:** Option B.

First-slice details:

- `studio` remains the only database owner.
- Worker auth is separate from producer tokens.
- New worker machines should be easy to add with a package command such as `voxhelm-remote-worker`, installed/run via `uv tool install` or `uvx`, plus Voxhelm URL and credentials; worker hosts must not need Django server setup or database credentials.
- `job_type=transcribe` can run in `remote_pull` mode; `job_type=synthesize` stays on Django Tasks.
- Remote-pull transcription jobs are not enqueued to Django Tasks, avoiding a mixed scheduler race.
- Trusted workers such as `atlas.local` are included in the internal MinIO credential domain for the first slice and post artifact manifests back to Voxhelm.
- Worker artifacts are written under attempt-scoped prefixes so stale/zombie workers cannot overwrite the producer-visible winning attempt after lease reclaim.
- `/v1/internal/*` worker endpoints must be blocked at public edge routes unless explicitly exposed through a private worker-only route.
- Claiming uses SQLite-safe conditional updates and affected-row checks; all lease time is based on `studio` server time.
- Goal completion requires a real production python-podcast `pyannote_known_speaker` transcript to execute on `atlas.local`, populate the django-cast `Transcript.speakers` sidecar, and meet the known-speaker runbook quality bar.
- The implementation must choose and document whether Atlas runs pyannote/wespeaker known-speaker processing itself, or whether a hybrid path has `studio` perform known-speaker classification after Atlas STT/artifact upload.

**Blocks implementation:** No for the architecture choice; implementation should follow `specs/remote-transcription-workers.md` and keep `interface-map.md` in sync as details evolve.

---

## D-24: Should the C13 lane scheduler admit more than one local inference at a time?

**Status:** Accepted and delivered 2026-09-06: three planning review rounds, five code review rounds, deployed to `studio` and live-verified (interactive Wyoming latency 0.57 to 0.85 s while a 20-minute `whisper-cli` job runs, versus 0.39 s idle; disconnect terminates `whisper-cli` within 2 s). Slice description: `specs/2026-09-06_bounded_inference_slots.md`. Reopens the Option B branch of D-19 with new evidence; D-19 itself stays the record of the first slice.

**Context:** D-19 accepted single-slot cooperative serialization and explicitly deferred "reserved parallel slots per lane" as risky for memory pressure and not clearly needed on `studio`. Since then two things changed. First, the interactive path proved to be starved in practice: the scheduler cannot preempt a running inference, so a 40-minute voice memo through the sync endpoint (15 to 20 minutes of `whisper-cli`) blocks the Home Assistant voice pipeline for that long, and the Daybook Voice Memo importer is about to add a long-memo lane while podcast-length batch jobs already exist. Second, measurements on 2026-09-05 removed the memory concern: `studio` (M4 Max, 128 GiB) runs two or three concurrent `whisper-cli` processes with the 2.9 GiB `ggml-large-v3.bin` without memory pressure; the shared resource is Metal GPU time, and concurrent runs slow each other roughly proportionally. Production STT is `whispercpp` (one subprocess per request) with `mlx` as fallback, and the Wyoming sidecar runs `mlx` in its own process.

**Options:**

| Option | Tradeoff |
|--------|----------|
| A. Keep single slot; schedule long jobs on the importer side | No Voxhelm change, but every producer must learn Voxhelm's load, and the voice assistant still stalls whenever any producer gets it wrong |
| B. Bounded slots in the existing scheduler: one reserved interactive slot plus one non-interactive slot (D-19 Option B, now justified) | Interactive requests are admitted immediately while a long job runs, at the cost of a proportional GPU slowdown of the long job; no new queue or process topology |
| C. Preempt or chunk long inferences | Hard latency bound, but requires interrupting or re-architecting backends and adds far more machinery than the problem needs |
| D. Separate interactive runtime (D-19 Option A) | Strong isolation, but new process topology and deployment complexity for a problem that two slots already solve |

**Accepted decision:** Option B.

- Capacity is `VOXHELM_LANE_SCHEDULER_INTERACTIVE_SLOTS` (default `1`) plus `VOXHELM_LANE_SCHEDULER_NON_INTERACTIVE_SLOTS` (default `1`); production pins `1 + 1` through ops-control. Non-interactive work can never hold more than its own slot count; interactive work may use any free slot, so the interactive slots are reserved.
- Priority semantics from D-19 stay: interactive waiters win the next free slot, running work is never preempted, callers still cannot pick a lane.
- Only backends that run outside the calling process may overlap: `whisper-cli` subprocesses and the WhisperKit sidecar. In-process backends keep a per-backend lock in their own process (`mlx-whisper` for STT; Piper and Kokoro already have one for TTS). The process-wide `_TRANSCRIPTION_LOCK` goes away because its safety concern was in-process state, not separate processes.
- The deployed Wyoming backend (`mlx`, in the Wyoming sidecar process) already overlaps a `whisper-cli` subprocess from the HTTP process without sharing in-process state, so no Wyoming configuration change is required.
- Dead-holder recovery applies per holder file (`holders/<token>.json` replaces `holder.json`). The 1800 s age threshold is warning-only under the 2026-10-13 amendment below.
- The sync endpoint terminates its `whisper-cli` child when the HTTP client disconnects (Django 5 ASGI request cancellation), so an abandoned upload no longer burns a slot and GPU time to completion.
- `VOXHELM_LANE_SCHEDULER_ENABLED=false` still means no gating; `0` interactive plus `1` non-interactive slot is the config-only rollback to D-19 behaviour. A legacy `holder.json` is counted as a non-interactive holder until it is released or reclaimed, so new-code admissions respect a legacy holder that is still running. The reverse does not hold: during the mixed-version window of a deploy or rollback an old-code process can exceed the bound by one holder. That window is seconds (sequential launchd restarts, all three services in one playbook run) or at worst one in-flight legacy inference, and only separate-process backends can overlap, so it is accepted rather than bridged.

**Feasible guarantee on one host:** with `1 + 1` slots an interactive request is admitted immediately whenever total capacity is not exhausted, which on `studio` means: at most one non-interactive inference is running and no other interactive request holds the reserved slot. Admission is not execution: an in-process backend still waits for its own per-process lock (relevant only when two interactive requests overlap inside the Wyoming sidecar). Interactive latency is then bounded by GPU contention (roughly proportional slowdown), not by the remaining runtime of the long job. Live holders remain counted until release, so age alone cannot admit work beyond the capacity bound (2026-10-13 amendment).

**Recovery amendment (2026-10-13):** Claims have no renewal mechanism. Reclaiming
an old live PID lets inference continue while its capacity is reused, violating
the bound. `VOXHELM_LANE_SCHEDULER_STALE_SECONDS` therefore becomes only an
operational warning threshold (default still 1800 s); dead-PID and invalid-state
cleanup remain. This supersedes age-based recovery in D-19, C13 and the dated
bounded-slots spec, for modern and legacy holder files. PID reuse and leaked
holders within a live daemon require operator recovery: stop all local inference
consumers and verify their inference children have exited before removing orphan
holder files, then restart. Stronger process identity/leases remain future work;
capacity preservation takes precedence over automatic age-based recovery.

**Not decided here:** slot counts above `1 + 1`, a status API, importer-side scheduling, preemption, and any local `studio` pull worker (which must keep using this scheduler, see BACKLOG).

**Blocks implementation:** No; the design review is complete and implementation follows `specs/2026-09-06_bounded_inference_slots.md`.

---

## D-25: Should the remote transcription workers take lane-scheduler slots?

**Status:** Accepted 2026-10-16. No code or deployment change: the worker already wraps its STT and embedding steps in `admit_local_inference(LANE_NON_INTERACTIVE)`, and the worker env (ops-library `voxhelm_remote_worker_deploy`) sets no `VOXHELM_LANE_SCHEDULER_*`, so `VOXHELM_LANE_SCHEDULER_ENABLED` keeps its default `false` there and admission is a no-op. This decision keeps it that way. It narrows the "any local `studio` pull worker must keep using this scheduler" notes in D-23 and D-24 for the remote-pull worker that already runs on `studio`.

**Context:** Two `voxhelm-remote-worker` processes claim batch transcription jobs: `atlas` and `studio`. Each processes one job at a time. On `atlas` nothing else runs Voxhelm inference. On `studio` the control plane (uvicorn and the Wyoming sidecar) runs as `root` with the scheduler enabled at `1 + 1` slots (D-24) in `/opt/apps/voxhelm/site/var/lane-scheduler`, and the worker runs as the service user. The one non-interactive slot serves the sync endpoint, which Daybook's voice-memo long lane depends on; ops-control ties that lane to the `1 + 1` pins. A podcast job on a worker takes 2 to 7 minutes of Parakeet (measured on the 2026-10-15 pilot re-runs), longer with whisper.cpp, and the bulk run covers about 190 episodes.

**Options:**

| Option | Tradeoff |
|--------|----------|
| A. Enable the scheduler on `atlas` only | Changes nothing: one worker process, one job at a time, no other consumer to coordinate with |
| B. Share the control plane's scheduler on `studio` | Counts the worker's GPU use, but needs a group-writable scheduler directory and lock file across `root` and the service user, and puts the worker in the single non-interactive slot: during a bulk run every voice memo waits up to one whole podcast job, which risks Daybook client timeouts |
| C. Raise `studio` to `1 + 2` and share the scheduler | Keeps a slot free for the sync endpoint while a worker job runs, but the bound is then the same as today's unscheduled concurrency (Wyoming, one sync request, one worker job) plus cross-user directory permissions and a change to the pinned Daybook contract |
| D. Leave the workers unscheduled | At most three GPU users on `studio` at once (Wyoming, one sync request, one worker job); each is slowed proportionally, none is blocked; 128 GiB leaves ample memory |

**Accepted decision:** Option D. Revisit if a second worker process per host, a concurrent local LLM job, or measured interactive latency during a bulk run makes the unscheduled overlap a problem.

**Blocks implementation:** No.

---

## D-26: How does a producer get reproducible, engine-pinned transcripts it can delete?

**Status:** Implemented 2026-10-16 (not yet deployed). Slice 1 of the podcast server-side ad analysis (`~/projects/podcast/docs/player/specs/server-ad-analysis-design.md`); slice description: `specs/2026-10-16_ad_analysis_profile.md`.

**Context:** The podcast ad detector was tuned on Parakeet words produced with 30 s chunks, 5 s overlap and full attention. Voxhelm routes Parakeet only for known-speaker jobs, with the deployment's 600 s / 30 s / local attention / `--max-duration 30` settings and a whisper.cpp fallback, and `verbose_json` drops word timings. A fallback or a settings change would silently change the detector's input. Transcript artifacts are kept forever, while the podcast backend's retention promises that voxhelm keeps no copy for long.

**Options:**

| Option | Tradeoff |
|--------|----------|
| A. Let producers pass engine settings per job | Flexible, but every producer must know Parakeet's flags and results cannot be cached by name |
| B. Named, versioned profiles owned by Voxhelm | One name pins engine, settings and output shape; a change is a new name; adds a small registry |
| C. Reuse the known-speaker routing with a flag | No new concept, but inherits the fallback chain and the deployment's chunk settings |

**Accepted decision:** Option B.

- `POST /v1/jobs` accepts `profile` (only `ad-analysis-v1` today) and `language_hint`; such a job takes no `language`, no concrete `model` and no diarization, and the profile is part of the `task_ref` fingerprint (only when set, so existing fingerprints are unchanged).
- A profile never falls back: local jobs fail when the CLI or model is missing; remote workers advertise `capabilities.profiles` from the same per-heartbeat check as Parakeet, and only such workers claim profile jobs, so they wait otherwise.
- The profile writes a `words` artifact (`words.json`) with per-word times and mean token confidence built from all tokens exactly as the research runner did, and records engine identity (model commit and checksum from the deployment marker, `parakeet-mlx` and `mlx` versions, settings, language hint) in the job result and the artifact.
- Retention: producer-scoped `DELETE /v1/jobs/{id}` for finished profile jobs and a profile-owned 7-day backstop in `prune_job_artifacts` (D-09 note of 2026-10-16).

**Not decided here:** the drift check and turbo window endpoint (slices 2-3 of the podcast spec, which will extend the profile's identity and artifact), scheduling `prune_job_artifacts` (slice 16), profiles for other producers. (The window endpoint: D-27; the drift check, amending this profile before its first deploy: D-28.)

**Blocks implementation:** No.

---

## D-27: How does the ad-analysis profile re-transcribe one window with whisper turbo?

**Status:** Implemented 2026-10-16 (not yet deployed, not yet called by a profile). Slice 2 of the podcast server-side ad analysis (`~/projects/podcast/docs/player/specs/server-ad-analysis-design.md`, "window endpoint"); slice description: `specs/2026-10-16_window_transcription.md`.

**Context:** The German drift check (slice 3) re-transcribes flagged 30 s windows with whisper large-v3-turbo, language detected on the clip, and needs the research runner's words, detected language and probability. The podcast spec calls this a "window endpoint" and requires that nothing (clip, words, job row) remains after a response. Its only caller is the `ad-analysis-v1` profile, which runs where the episode audio is: the Django Tasks worker or a remote worker. Delve never calls it.

**Options:**

| Option | Tradeoff |
|--------|----------|
| A. In-process function `transcribe_window(audio_path, t)` used by the profile | Runs on the already materialised source on any host that runs the profile; no request, job or artifact exists, so nothing can persist; no new producer surface |
| B. Producer-authenticated synchronous HTTP route over a staged upload or job source | Callable by producers, but no producer needs it; a remote worker would send its audio back to the control plane per window, or the control plane would keep a source copy it otherwise deletes; adds auth, limits, admission and retention duties |
| C. A batch job type per window | Reuses job plumbing, but creates job rows and artifacts per window that then have to be deleted, the opposite of the retention requirement |

**Accepted decision:** Option A; the spec's "window endpoint" is this function.

- whisper.cpp `whisper-cli` (the binary voxhelm already runs) with the decoding the research `whisper-server` used: `-l auto -t 4 -p 1 -bs 1 -bo 2 -tp 0.00 -tpi 0.20 -ml 60 -ojf`; clip `[max(0, t - 1), +32 s)` decoded by ffmpeg exactly as `retry.py`; words assembled from tokens and kept for starts in `[t, t + 30)` as `retry.py`; the probability is read from whisper.cpp's log line and cross-checked with the JSON result's language.
- Pinned model: the ggml file named by `VOXHELM_WINDOW_MODEL` must have SHA-256 `1fc70f77...bc69` (in code); otherwise `BackendUnavailableError`, never another model or backend. Identity: engine, CLI version, model name and hash, settings.
- No residue: temporary directory per call, no ORM access, error messages without transcript text. Verified identical to the research output on 40 dev windows (3,342 words).
- A producer route can later wrap the same function if a producer needs windows; that is a new decision.

**Not decided here:** which windows are retried, the p >= 0.9 acceptance, the splice and per-window records (slice 3), the profile version that adds the retry model hash, worker capability for the turbo model. (All decided in D-28.)

**Blocks implementation:** No.

---

## D-28: How does the ad-analysis profile run the German drift check, and does it stay `ad-analysis-v1`?

**Status:** Implemented 2026-10-16 (not yet deployed). Slice 3 of the podcast server-side ad analysis (`~/projects/podcast/docs/player/specs/server-ad-analysis-design.md`, "Analysis Stages" 3, adapter contract row "Drift check with turbo retry" (b), gate G2); slice description: `specs/2026-10-16_drift_check.md`.

**Context:** Parakeet drifts into English inside German speech in about 9 % of German 30 s windows. The frozen research rule (`round2-german-drift`, FREEZE.txt 2026-10-15) flags grid windows by English and German function words and re-transcribes them with whisper large-v3-turbo; the spec adopts the post-hoc acceptance (episode language with p >= 0.9). The podcast spec wants the per-window records to "come back with the words and be kept", the retry model hash in the identity, workers that advertise the profile only with the turbo model, and calls the profile versioned ("any change to its settings is a new profile name"). Slice 1 (D-26) is committed but has never been deployed or served to a producer.

**Options:**

| Question | Options | Tradeoff |
|----------|---------|----------|
| Where the records live | A. Inside `words.json` (`drift`) / B. a sibling `drift.json` artifact / C. job result metadata | A: one artifact that the producer already fetches, covered unchanged by `DELETE`, the backstop, the remote completion check and its exposure; the records are a few KB. B: a second required remote artifact, a second fetch, and words and records could get out of step. C: metadata outlives nothing but is not where Delve reads, and remote metadata is sanitized to scalars |
| Profile name | A. Amend `ad-analysis-v1` before the first deploy / B. `ad-analysis-v2` | A: no producer, cache or deployment ever saw the slice 1 output; the podcast spec's profile JSON already pairs `voxhelmProfile: ad-analysis-v1` with the drift block. B: a name whose predecessor never existed outside this Mac, and a `v1` that would have to stay accepted or be removed |
| Runtime requirement | A. The profile needs the turbo runtime for every job / B. only for German jobs | A: capability is per profile, not per job language, so a worker that advertises the profile can run any of its jobs; the identity (retry hash) is the same for every job. B: an English job could run where German ones cannot, but advertising and identity would depend on the job |
| Retry process | A. One `whisper-cli` per window (D-27) / B. a resident `whisper-server` | A: no extra service, nothing to keep alive or secure; measured cost is within G2's budget. B: saves the model load per window but adds a long-lived process per host |

**Accepted decision:** A in every row.

- `transcriptions/drift.py` reproduces the frozen check verbatim: research word lists and tokenizer (`lib.py`, list hash recorded), grid `[30k, 30k + 30)` over Parakeet word starts, flag `en >= 4 and en / max(1, en + de) >= 0.10`, retry through `transcribe_window`, accept when the detected language equals the job's `language_hint` and p >= 0.9, splice as `splice.py` (drop words starting in the window, add the retry's, stable sort by start). The word list keeps "these" (REPORT.md's later suggestion to drop it is not part of the frozen rule).
- It runs only for the profile's languages (`de`); other jobs carry `drift.checked: false` with a reason. Every window gets a record (t, counts, flag, and for flagged windows language, p, accepted, word counts, retry seconds); `words.json` keeps `schema_version` 1 (nothing was ever served with the old shape).
- `ad-analysis-v1` is amended in place, recorded here; from its first deploy on, any change to the profile (including the rule, the word lists or the turbo model) is a new name.
- Both runtimes are resolved before Parakeet runs; a missing or unverified turbo model fails a local job and makes a claimed remote attempt retryable; workers advertise the profile only while Parakeet and the window runtime resolve; a window retry that fails fails the job. Nothing skips the check silently.
- The identity gains `drift` (rule record plus `retry`: whisper.cpp version, model, SHA-256, settings); the control plane refuses a remote completion whose drift identity differs from its own and keeps only the worker's whisper.cpp version.
- The splice changes only `words.json`; the transcript formats stay Parakeet's sentences.
- Cancellation (`cancel_event`) is checked before Parakeet, before and inside each window retry, and once more after the last window.

**Not decided here:** dropping "these" from the English list or restricting whisper's language choice (would be a new profile), drift checks for other languages, splicing the transcript formats, holdout G2 (the gate is dev-only).

**Blocks implementation:** No.

---

## D-29: How does Delve get the turbo word windows of its sponsor-chapter edges?

**Status:** Implemented 2026-10-17 (not yet deployed). Podcast Server-Side Ad Analysis v1, slice 10b (Delve slice 10, "Chapter stage and result assembly" of `~/projects/podcast/docs/backend/specs/smart-skip-coordination.md`, gate G6); slice description: `specs/2026-10-17_chapter_windows.md`. Amends `ad-analysis-v1` in place (D-28's rule: nothing of the profile has been deployed or served yet).

**Context:** Delve's chapter text check moves each sponsor-chapter edge onto the advertisement's real start or end. G6 fails with the Parakeet words Delve already has and passes with the research's turbo windows (`round2-chapter-textcheck/asrwin120.py`, FREEZE.txt 2026-10-15T00:58:02Z): whisper large-v3-turbo (the drift check's pinned file), English, DTW token timestamps, one word per segment, a punctuated prompt, 120 s around each edge, shifted on a DTW abort. Voxhelm's window function (D-27) is the drift check's: 32 s, language detected, no prompt, no DTW. Delve knows the edges before it submits (they come from the file's ID3 chapters and its duration), but it holds the audio only in its own media directory; the voxhelm job already has it.

**Options:**

| Question | Options | Tradeoff |
|----------|---------|----------|
| Where the pass runs | A. Inside `ad-analysis-v1`: the job carries the marks, the profile writes a `chapter_windows` artifact / B. A separate profile `chapter-windows-v1`, a second job per episode / C. A producer route over a staged upload | A: one upload, one job, one DELETE and the 7-day backstop cover it; the audio is already materialised where Parakeet runs; Delve stores words and windows in one step. B: a second staged upload of the same episode, a second job lifecycle in Delve's state machine (upload, submit, poll, fetch, delete) for a few seconds of whisper. C: a new producer surface with its own auth, limits, admission and retention (D-27's reasons) |
| Artifact | A. A sibling `chapter_windows.json` / B. Inside `words.json` (as D-28's drift records) | A: the windows are another engine's words with their own identity, a job without marks has none, and `words.json` (verified in G1/G2) stays byte for byte what it is without marks; costs one more required remote artifact and one more fetch. B: one fetch, but the words document and its identity would change shape with the request |
| Which marks | A. Strictly increasing whole milliseconds, 1-256, checked against ffprobe's duration on the materialised source / B. Seconds as floats, any order | A: one canonical request per set of edges (it is part of the `task_ref` identity), exactly the research's `t` (`round(t * 1000)` is whole for every research edge), and the same duration measure Delve takes; B: duplicates and float formatting would make equal requests differ |
| Profile name | A. Amend `ad-analysis-v1` / B. `ad-analysis-v2` | A: never deployed or served; a job without marks is unchanged (same words, identity and artifacts); B: a name whose predecessor never existed outside this Mac |

**Accepted decision:** A in every row.

- `POST /v1/jobs` takes `chapter_marks_ms` with a profile only (400 otherwise, and for an empty list, more than 256 marks, a non-integer, a mark outside `[0, 24 h]` or marks that are not strictly increasing); they are stored in `output_data`, part of the `task_ref` match and fingerprint (only when present, so other fingerprints are unchanged), and carried by a remote claim.
- Before Parakeet runs, the profile resolves the window runtime and ffprobe (now part of the profile's runtime and of `capabilities.profiles`), measures the source's duration and refuses a mark after its end (`ChapterMarkError`: the job fails; on a remote worker the failure is not retryable). After the drift check it runs `transcriptions.chapter_windows.run_chapter_windows`: for each mark the research clip, ffmpeg arguments, `whisper-cli -l en -t 4 -p 1 -bs 5 -bo 5 -tp 0.00 -tpi 0.20 -dtw large.v3.turbo -ml 1 -sow --prompt 'Hello, and welcome back. Thanks!' -oj -np` (whisper.cpp 1.9.5's default decoding passed explicitly), the research's shift sequence on a whisper.cpp error and its word assembly; a window none of the five clips transcribes is `ok: false` with its exit statuses. The pinned model is checked again after every failed clip, so a model that went away fails the run instead of passing as untranscribed windows; ffmpeg failures, timeouts and malformed output fail the run.
- `chapter_windows.json` (kind `chapter_windows`, exposed): schema 1, profile, rule `turbo-window-v1`, identity (rule, engine, whisper.cpp version, model and SHA-256, every clip and decoding setting including the prompt and its hash and the shifts), duration, summary and one window per mark (`mark_ms`, `ok`, `clip_start`, `shift`, `exit_codes`, `seconds`, `words` `{w, s, e}` in file seconds). `result.metadata.chapter_windows` keeps identity and summary (no words). A remote completion of a job with marks must include it exposed and a chapter windows identity equal to the control plane's (only the whisper.cpp version is taken); one without marks must not include it.
- Cancellation is checked before each window and inside each clip's processes; one temporary directory per pass, removed on every path; errors carry logs, never words. `DELETE` and the backstop remove the artifact like any other of a profile job.
- Verified on the 26 G6 episodes: all 102 windows identical to the research's (36,313 words, text, start and end, clip start and shift; the same model hash), and G6 through Delve's real route with a local voxhelm passes (6 fixed, 0 worse, 22 of 22 kit-2 cuts inside; the 102 windows Delve stored identical to the research's).

**Not decided here:** windows for non-English episodes (the rule is the research's English-forced recipe; Delve sends the marks of every file with sponsor chapters), a resident whisper process for the windows, a producer route.

**Blocks implementation:** No.

---

## D-30: How long may a batch or remote worker fetch large URL media, and does a slow fetch cost a job attempt?

**Status:** Implemented 2026-10-17 (not yet deployed).

**Context:** All URL fetches shared one elapsed budget, `VOXHELM_URL_FETCH_TIMEOUT_SECONDS` (60 s, set to 60 in production too). A 128 MB python-podcast source (`pp_48.mp3`, job `4fe55e2d-4691-4681-b5d8-7f3fe482f4ef`) needs about 2.1 MB/s from CloudFront to fit; on `studio` the fetch ran out of budget on attempts 1 and 2 and succeeded on attempt 3 of 3, in under 35 s. Every budget expiry failed the whole remote attempt, so one more slow fetch would have failed the job for good. Of python-podcast's 68 sources, 18 are at least 100 MB and the largest is 160 MB. The Django Chat bulk run (2026-10-16) also lost two attempts to URL fetches.

**Options:**

| Question | Options | Tradeoff |
|----------|---------|----------|
| Budget | A. A separate per-try budget for batch media, default 600 s / B. Raise the shared budget / C. A budget proportional to `Content-Length` | A: sync requests keep their short bound, batch media (at most 512 MiB) needs only about 0.9 MB/s, no env change. B: a sync request could hold its HTTP request for minutes. C: the length is known only after the headers, a missing header needs a fallback anyway, and a minimum-throughput rule is harder to state and test than a fixed bound |
| Retry | A. Retry transient fetch failures inside the attempt / B. Server-side: fetch failures do not count against `max_attempts` / C. None | A: worker-local, no protocol or model change; a cold CDN edge or a dropped connection is usually fine a few seconds later. B: needs a failure category in the fail API and attempt accounting, and an endless loop guard; C: keeps losing attempts |

**Accepted decision:** A and A.

- `download_allowed_media` (batch job inputs and known-speaker reference audio, on the server's Django Tasks worker and on remote workers) uses `VOXHELM_BATCH_URL_FETCH_TIMEOUT_SECONDS` (default 600, positive) per try and retries a `TransientFetchError` up to `VOXHELM_BATCH_URL_FETCH_ATTEMPTS` tries in total (default 3, positive), after 5 s, 10 s, doubling up to 30 s.
- Transient: the elapsed budget, DNS resolution errors, connection and read errors, a body shorter than its `Content-Length` (before this change `http.client` passed such a truncated file on as complete, for sync inputs too) and HTTP 408, 425, 429, 500, 502, 503 and 504. Everything else stays a plain `ApiError` and fails at once: allowlist, scheme and resolved-address rejections, more than 5 redirects, an unsupported media type, the size limit, a certificate verification failure and other HTTP statuses.
- Sync URL inputs (`transcriptions.input_media`) keep `VOXHELM_URL_FETCH_TIMEOUT_SECONDS` and one try. Size limits are unchanged: sync 128 MiB (`VOXHELM_MAX_URL_DOWNLOAD_BYTES`), batch and remote 512 MiB (`VOXHELM_BATCH_MAX_DOWNLOAD_BYTES`).
- A download that fails every try fails the attempt as before (retryable on a remote worker). Retries log the host and the error, never the URL.

**Not decided here:** a separate failure category so the server does not count a fetch failure against `max_attempts` (option B; see BACKLOG), fetch throughput logging.

**Blocks implementation:** No.

---

## PRD Open Questions (cross-referenced)

The PRD's 8 open questions are addressed by the following decisions:

| PRD Open Question | Decision |
|-------------------|----------|
| 1. One binary or multiple processes? | D-10 |
| 2. SQLite queue or file-backed? | D-14 (SQLite via Django ORM) |
| 3. Sync HTTP endpoints: direct or proxy? | D-15 (native Django for HTTP; wrappers for Wyoming) |
| 4. Default backends for interactive/batch STT/TTS? | D-04 (STT: mlx-whisper); TTS deferred to M3 |
| 5. Archive upload/fetch model? | D-05 |
| 6. Retention policies? | D-09 |
| 7. Wake-word infrastructure? | Out of scope for all milestones (PRD non-goal) |
| 8. OpenClaw direct or via HA? | D-11 (deferred to M4; decision not needed now) |
