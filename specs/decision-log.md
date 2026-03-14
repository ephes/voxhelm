# Decision Log: Voxhelm

**Date:** 2026-03-11
**Status:** Accepted defaults; the M1-M3 core runtime slices, including the first C13 lane-scheduling slice, are implemented as of 2026-03-13. Later backend expansion, Archive article-audio follow-on work, and M4/OpenClaw remain planned.

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

**Recommended default:** Option C with Whisper-native JSON as the internal canonical transcript representation. The sync OpenAI-compatible endpoint still returns `{"text": "..."}` (or `verbose_json` with segments) to match Archive's and podcast-transcript's expectations, but the async job API should support producer-requested output formats such as plain text, Whisper JSON, DOTe, Podlove JSON, and WebVTT. Voxhelm should normalize backend output internally and perform the required format conversions server-side so different consumers can rely on one service contract.

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

**Recommended default:** Option C. Defer diarization entirely for now. django-cast needs contributor/speaker support before speaker labels have meaningful product value, so diarization should not drive near-term Voxhelm scope.

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
- stale-lock timeout / recovery window

**Recommended initial default:** `VOXHELM_LANE_SCHEDULER_STALE_SECONDS=1800`.

That default is intentionally conservative because the first slice does not yet define a lease-heartbeat mechanism, and long non-interactive inference on `studio` can legitimately run for many minutes. Lower values are safer for crash recovery but raise the risk of falsely reclaiming a live holder.

Do not add slot-count tuning in the first slice because the accepted design is single-slot cooperative serialization, not reserved parallel capacity.

**Blocks implementation:** Yes -- this is the reviewed C13 design baseline.

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
