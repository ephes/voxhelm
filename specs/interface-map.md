# Voxhelm Interface Map

**Date:** 2026-03-11
**Status:** Active architecture doc; M1-M3 core runtime slices are implemented as of 2026-03-13

This document is the active source of truth for Voxhelm's architecture boundaries, interface contracts, artifact access model, and auth domains.

---

## Topology and Responsibility Split

Current implemented topology:

- One Django + `uvicorn` HTTP process on `studio`
- One Django Tasks worker process on `studio`
- One Wyoming STT/TTS sidecar process on `studio`
- Private HTTPS ingress on `macmini` via Traefik at `https://voxhelm.home.xn--wersdrfer-47a.de`
- MinIO-backed artifact handling for batch work, with artifacts served back through Voxhelm HTTP endpoints

Remaining planned topology work:

- Add the reviewed C13 first slice: one host-wide cooperative inference gate on `studio` so Wyoming traffic gets admission priority over queued HTTP/batch work without adding a new worker tier
- Add future OpenClaw-facing consumers without changing the core service boundaries

```
Archive / podcast-transcript / python-podcast / OpenClaw (future)
                        |
                        v
             Traefik on macmini
                        |
                        v
              Django HTTP API on studio
      (sync STT, auth, validation, health endpoint)

Home Assistant now reaches a separate Wyoming listener on `studio`.
```

| Component | Responsibilities |
|-----------|------------------|
| Django HTTP process (implemented in M1a) | Authenticate producers, validate requests, fetch allowed URL inputs, invoke the configured sync STT backend, render OpenAI-compatible responses, expose health, accept batch jobs, proxy artifact downloads |
| Django Tasks workers (implemented in M1b) | Execute queued transcription work, fetch media, extract audio from video, invoke STT backends, upload artifacts, and report status/results back into the control plane |
| Consumers | Submit sync or batch requests, poll for status where needed, consume returned text or artifact references, and apply only consumer-local post-processing beyond Voxhelm's published artifact formats |
| Artifact store (implemented in M1b) | Hold source inputs, intermediates, and final artifacts; never exposed directly to consumers |

---

## 1. Interface Inventory

Voxhelm exposes five producer/operator-facing interface surfaces:

| # | Interface | Protocol | Direction | v1 | Consumers |
|---|-----------|----------|-----------|-----|-----------|
| 1 | OpenAI-compatible STT API | HTTP | inbound | yes | Archive, podcast-transcript |
| 2 | Batch job API | HTTP | inbound + poll | yes | python-podcast |
| 3 | Wyoming STT/TTS | TCP (Wyoming) | inbound | yes | Home Assistant |
| 4 | Artifact storage API | S3 (MinIO) | bidirectional | yes | Voxhelm workers and control plane only (consumers access artifacts via Voxhelm HTTP proxy) |
| 5 | Health / operator API | HTTP | inbound | yes | ops tooling, monitoring |

The worker execution path is intentionally not part of the producer-facing contract. In the current production deployment, the HTTP process and the Django Tasks worker run as separate launchd services on `studio`.

OpenClaw is an architectural placeholder for v1. It will consume interface 1 and/or 2 when it integrates (Milestone 4).

---

## 2. Interface Details

### 2.1 OpenAI-Compatible Synchronous STT API

**Purpose:** Drop-in replacement for the hosted OpenAI Audio Transcriptions endpoint so that Archive (and later other consumers) can point their existing `API_BASE` at Voxhelm with no code changes.

**Implemented in M1a:** yes

**Protocol:** HTTP/1.1, POST

**Endpoint:** `POST /v1/audio/transcriptions`

**Auth:** Bearer token in `Authorization` header.

**Request format:** either `multipart/form-data` with an uploaded file or JSON with a source URL.

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `file` | binary | yes if using multipart upload | Audio file. Field name must be `file`; filename is informational. |
| `url` | string | yes if using URL mode | Remote media URL for Voxhelm to fetch directly. |
| `model` | string | yes | Model identifier. Voxhelm maps this to a backend+model pair internally. `gpt-4o-mini-transcribe` must be accepted for Archive compatibility; `whisper-1` is also accepted as an OpenAI-style alias. |
| `prompt` | string | no | Transcription guidance/context string. |
| `language` | string | no | ISO 639-1 hint (e.g. `de`, `en`). |
| `response_format` | string | no | `json` (default), `verbose_json`, `text`, `vtt`. DOTe, Podlove, and other multi-artifact transcript formats are batch-only outputs. |

**Response format:**

Default (`json`):
```json
{"text": "transcribed content..."}
```

`verbose_json` (used by podcast-transcript's Groq backend today):
```json
{
  "text": "full transcript...",
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 5.12,
      "text": "segment text..."
    }
  ]
}
```

**Size limit:** 25 MiB when using direct file upload. URL mode avoids that upload cap but still follows service-side processing limits.

**Supported M1a suffixes:**
`.flac`, `.m4a`, `.mp3`, `.mpeg`, `.mpga`, `.oga`, `.ogg`, `.wav`

Larger URL-driven inputs can be handled through this interface or through the batch API in interface 2.2, depending on latency and timeout needs. For zero-code-change Archive compatibility, M1a targets inputs that can complete within Archive's existing 300-second timeout budget. Video containers still belong to the batch preprocessing path.

**Primary consumers:** Archive, podcast-transcript-style backends, and future OpenClaw integrations.

**Contract constraints:**
- `response_format=json` returns `Content-Type: application/json` with `{"text": "..."}`.
- `response_format=verbose_json` returns Whisper-style segments with `id`, `seek`, `start`, `end`, and `text`.
- Clients may provide either an uploaded file or a source URL; the producer-facing contract stays the same regardless of how Voxhelm fetches media.
- URL fetch policy for v1: allow `https://` by default; allow `http://` only for explicitly configured trusted internal hosts; reject URLs outside the configured host allowlist.
- This is a synchronous HTTP transcription contract for blocking clients; it is not the Wyoming interactive voice path.

**Workload lane:** Synchronous (bounded-size, blocking request/response). This is not the interactive voice lane — it serves Archive and similar consumers that make blocking HTTP calls.

---

### 2.2 Batch Job API

**Implemented in M1b:** yes

**Purpose:** Asynchronous job execution for long-running transcription, TTS, and media processing through one common producer-facing API.

**Protocol:** HTTP/1.1, JSON

**Endpoints:**

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| POST | `/v1/jobs` | Producer token | Submit a new job |
| GET | `/v1/jobs/<job_id>` | Producer token | Poll job status |
| GET | `/v1/jobs/<job_id>/artifacts/<name>` | Producer token | Download an exposed artifact |

#### 2.2.1 Job Submission (Producer Side)

**Request:** `POST /v1/jobs`

```json
{
  "job_type": "transcribe",
  "priority": "normal",
  "lane": "batch",
  "backend": "auto",
  "model": "auto",
  "language": "de",
  "input": {
    "kind": "url",
    "url": "https://example.com/episode.mp3"
  },
  "output": {
    "formats": ["text", "json"]
  },
  "context": {
    "producer": "archive",
    "item_id": 123
  },
  "task_ref": "archive-item-123-transcript-v1"
}
```

**Job types (current v1):** `transcribe`
**Job types (later):** `synthesize`, `extract_audio`, `analyze_media`, `diarize`

**Input kinds (current M1b):** `url`
**Input kinds (later):** `upload`, `minio_ref`

**Output formats for `transcribe`:**

| Format | Shape | Consumer |
|--------|-------|----------|
| `text` | Plain text string | Archive (via sync endpoint), podcast-pipeline (indirect) |
| `json` | Whisper-style `{"segments": [{"id", "start", "end", "text", ...}]}` | podcast-transcript, python-podcast |
| `vtt` / `webvtt` | WebVTT text | subtitle and transcript consumers |
| `dote` | DOTe JSON | django-cast, podcast-oriented consumers |
| `podlove` | Podlove transcript JSON | django-cast / Podlove consumers |

Voxhelm uses Whisper-native JSON as its internal canonical structured transcript representation. The current M1b implementation stores `text`, `json`, and `vtt` artifacts. Additional server-side conversions remain planned.

**Response:** `201 Created`

```json
{
  "id": "uuid",
  "state": "queued",
  "created_at": "2026-03-11T10:00:00Z",
  "task_ref": "archive-item-123-transcript-v1"
}
```

The producer-facing `id` is Voxhelm's stable job identifier. Voxhelm also stores the internal Django Tasks result/task identifier for execution tracking, but that internal identifier is not part of the producer contract.

#### 2.2.2 Job Status (Producer Polling)

**Request:** `GET /v1/jobs/<job_id>`

**Response:**

```json
{
  "id": "uuid",
  "state": "succeeded",
  "job_type": "transcribe",
  "created_at": "...",
  "started_at": "...",
  "finished_at": "...",
  "result": {
    "text": "full transcript...",
    "artifacts": {
      "text": "/v1/jobs/<id>/artifacts/transcript.txt",
      "json": "/v1/jobs/<id>/artifacts/transcript.json"
    },
    "metadata": {
      "backend": "whisperkit",
      "model": "large-v3",
      "language": "de",
      "duration_seconds": 3612.5,
      "processing_seconds": 45.2
    }
  }
}
```

**States:** `queued` -> `running` -> `succeeded` | `failed` | `canceled` | `expired`

#### 2.2.3 Artifact Download

**Request:** `GET /v1/jobs/<job_id>/artifacts/<name>`

Returns the named exposed artifact for the caller's job. Consumers do not need direct MinIO access.

#### 2.2.3 Internal Execution Model

The batch API is the stable producer contract. How jobs move from `queued` to `running` is an internal implementation choice.

**Current default:** the control plane enqueues work through Django Tasks, and worker processes on `studio` execute queued tasks.

**Concrete runtime choice:** use `django_tasks_db.backend.DatabaseBackend` with the same local SQLite database as the control plane.

The current implementation also persists the linked Django task/result identifier in the producer-facing job record and reconciles terminal state from the Django Tasks result backend when jobs are queried.

**Reviewed C13 runtime rule:** Batch jobs continue to use Django Tasks, but any task step that enters local STT/TTS inference on `studio` must participate in the same host-wide lane scheduler as the HTTP API and Wyoming sidecar. C13 does not introduce a second task queue or a separate interactive worker host.

**Future-compatible option:** if Voxhelm later needs remote workers or stricter process isolation beyond the current Django Tasks setup, it can add more explicit worker coordination without changing the producer-facing batch API.

**Primary direct consumers:** python-podcast / django-cast and future integrations that need long-running or URL-based processing.

**Indirect or non-batch consumers:** Archive uses the synchronous STT endpoint; podcast-pipeline reaches Voxhelm indirectly through podcast-transcript.

---

### 2.3 Wyoming STT/TTS Interface

**Purpose:** Home Assistant local voice pipeline integration.

**Protocol:** Wyoming (TCP, event-driven, asyncio-based).

**Direction:** Home Assistant -> Voxhelm.

**Wyoming STT service:**
- Accepts chunked audio stream (typically 16kHz, 16-bit, mono PCM).
- Returns `Transcript` event with text.
- Must respond within interactive latency budget (< 2-3 seconds for short utterances).

**Wyoming TTS service:**
- Accepts `Synthesize` event with text and optional voice ID.
- Returns chunked audio stream.

**Auth:** None at protocol level. Security via network binding (Tailscale, local interface only).

**Workload lane:** Interactive. Must not be blocked by batch jobs.

**Reviewed C13 lane definition:**

| Lane | Traffic in scope now | Notes |
|------|----------------------|-------|
| `interactive` | Wyoming STT and Wyoming TTS only | Internal runtime classification; not exposed through `POST /v1/jobs` |
| `non-interactive` | `POST /v1/audio/transcriptions`, `POST /v1/audio/speech`, batch `transcribe`, batch `synthesize` | Internal scheduler lane. This intentionally includes sync HTTP inference even though the producer-facing job model still only exposes `lane=batch` |

**Reviewed C13 mechanism:** Host-wide admission control plus cooperative serialization. All local inference on `studio` shares one scheduler gate and one admission slot for both STT and TTS. A waiting Wyoming request jumps ahead of queued non-interactive work, but Voxhelm does not interrupt work that already holds the gate.

**Feasible guarantee on one host:** Voxhelm can prevent new HTTP/batch inference from starting ahead of a waiting Wyoming request and can avoid simultaneous heavy inference across processes. It cannot guarantee sub-second latency if an earlier HTTP/batch inference is already running when the Wyoming request arrives.

**Consumers:**
- Home Assistant Assist pipeline.

---

### 2.4 Artifact Storage Interface (MinIO / S3)

**Purpose:** Durable storage for transcripts, generated audio, intermediate files.

**Protocol:** S3-compatible API (MinIO).

**Direction:** Bidirectional -- Voxhelm workers write artifacts; the control plane reads them to serve to consumers. Consumers do not access MinIO directly.

**Bucket structure (current):**

```
voxhelm/
  jobs/<job_id>/
    sample.wav
    extracted.wav
    transcript.json
    transcript.txt
    transcript.vtt
```

**Auth:** S3-compatible credentials. Used only by Voxhelm workers and the control plane — never exposed to consumers.

**Consumers:** Consumers retrieve artifacts through the Voxhelm HTTP API (`GET /v1/jobs/{id}/artifacts/{name}`), which proxies the download from MinIO. This keeps the security boundary narrow: only Voxhelm holds MinIO credentials. Django Tasks workers on `studio` read/write directly using S3 credentials.

---

### 2.5 Health / Operator API

**Purpose:** Liveness checks, basic observability, administrative controls.

**Protocol:** HTTP/1.1, JSON

**Endpoints:**

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| GET | `/v1/health` | None | Liveness probe |
| GET | `/v1/status` | Operator session or admin token | Deferred. Queue depth, active jobs, worker liveness remain a later operator surface |

**Response (`/v1/health`):**
```json
{
  "status": "ok",
  "service": "voxhelm",
  "version": "0.1.0"
}
```

**Current scope note:** `GET /v1/health` is live. `GET /v1/status` is explicitly deferred out of the first C13 slice so the implementation can stay focused on runtime coordination.

**Consumers:** Monitoring (nyxmon), operator tooling.

---

## 3. Producer/Consumer Matrix

| Consumer | Sync STT (2.1) | Batch Jobs (2.2) | Wyoming (2.3) | Artifacts (via HTTP proxy) | Health (2.5) |
|----------|:-:|:-:|:-:|:-:|:-:|
| **Archive** | yes (primary) | -- | -- | -- | -- |
| **python-podcast / django-cast** | -- | yes (primary) | -- | read (via HTTP) | -- |
| **podcast-transcript** | yes (as backend) | fallback (long files) | -- | -- | -- |
| **podcast-pipeline** | -- (indirect via podcast-transcript) | -- | -- | -- | -- |
| **Home Assistant** | -- | -- | yes (M2) | -- | -- |
| **OpenClaw** | future (M4) | future (M4) | -- | -- | -- |
| **ops tooling / nyxmon** | -- | -- | -- | -- | yes |

---

## 4. Auth Boundary Map

Voxhelm maintains three required credential domains in v1, plus an optional worker credential domain if a private worker HTTP interface is added later.

### 4.1 Producer Tokens (Job Submission)

Each producer system gets its own bearer token for submitting jobs and polling status.

| Producer | Token env var (proposed) | Scope |
|----------|--------------------------|-------|
| Archive | `VOXHELM_PRODUCER_TOKEN_ARCHIVE` | Call sync STT; future batch access for its own jobs |
| python-podcast | `VOXHELM_PRODUCER_TOKEN_PODCAST` | Submit `transcribe` jobs; poll own job status |
| podcast-pipeline | (no direct token -- accesses Voxhelm indirectly via podcast-transcript) | N/A |
| OpenClaw | `VOXHELM_PRODUCER_TOKEN_OPENCLAW` | Future; submit jobs, call sync API |
| Operator | `VOXHELM_PRODUCER_TOKEN_OPERATOR` | Submit any job type; full status visibility |

Each token maps to a producer identity and its allowed interface surface.

Archive's existing settings would be reconfigured:
- `ARCHIVE_TRANSCRIPTION_API_KEY` -> set to `VOXHELM_PRODUCER_TOKEN_ARCHIVE` value
- `ARCHIVE_TRANSCRIPTION_API_BASE` -> set to `https://voxhelm.home.xn--wersdrfer-47a.de/v1` for the current private edge URL, or the direct `studio` URL when debugging

### 4.2 Django Tasks Runtime

Queued task execution is internal to the service. Producer auth and operator auth remain the relevant external boundaries; worker runtime credentials are an implementation detail of the Django Tasks backend and deployment.

**v1 backend:** `django_tasks_db.backend.DatabaseBackend`

**v1 settings shape:**
- `TASKS["default"]["BACKEND"] = "django_tasks_db.backend.DatabaseBackend"`

**Model boundary:** Voxhelm keeps a producer-facing job record keyed by its own UUID and stores the linked Django task/result id internally for execution and recovery. Consumers only see the Voxhelm job UUID.

**Deferred complexity:** no custom worker registry, launch-token handshake, or heartbeat table in v1. Add those only if the stock database backend and normal process supervision prove insufficient on `studio`.

### 4.3 Operator / Admin Session

Session-based authentication for the operator web UI and admin endpoints (if Voxhelm exposes one via Django admin or a custom dashboard).

| Setting | Proposed env var |
|---------|------------------|
| UI username | `VOXHELM_UI_USERNAME` |
| UI password | `VOXHELM_UI_PASSWORD` (or bcrypt hash) |
| Session secret | `VOXHELM_SESSION_SECRET` |

### 4.4 MinIO Credentials

S3-compatible credentials for artifact storage. Used only by Voxhelm internally (workers and the control plane).

| Setting | Current env var |
|---------|------------------|
| Artifact backend | `VOXHELM_ARTIFACT_BACKEND` |
| Artifact root (filesystem backend only) | `VOXHELM_ARTIFACT_ROOT` |
| S3 endpoint URL | `VOXHELM_ARTIFACT_S3_ENDPOINT_URL` |
| S3 region | `VOXHELM_ARTIFACT_S3_REGION` |
| S3 access key ID | `VOXHELM_ARTIFACT_S3_ACCESS_KEY_ID` |
| S3 secret access key | `VOXHELM_ARTIFACT_S3_SECRET_ACCESS_KEY` |
| S3 bucket | `VOXHELM_ARTIFACT_BUCKET` |
| Object prefix | `VOXHELM_ARTIFACT_PREFIX` |
| Force path-style addressing | `VOXHELM_ARTIFACT_S3_FORCE_PATH_STYLE` |

Consumers do not access MinIO directly. They retrieve artifacts through the Voxhelm HTTP API (`GET /v1/jobs/{id}/artifacts/{name}`), which proxies the download from MinIO. The sync endpoint returns results inline in the HTTP response body.

### 4.5 Network Boundary

All interfaces assume Tailscale or local-network-only access:
- Bind to Tailscale IP or `127.0.0.1` by default.
- Optionally enforce allowed CIDRs (same pattern as OpsGate's `OPSGATE_ALLOWED_CIDRS`).
- Wyoming binds to a separate port, also on Tailscale/local interface only.

### 4.6 Credential Domain Isolation

```
                  +-----------------------+
                  |     Voxhelm Service    |
                  |   (Django on studio)   |
                  +-----------+-----------+
                              |
         +--------------------+------------------+
         |                    |                  |
   Producer Tokens      Operator Session   MinIO credentials
   (per-consumer)       (web UI / admin)   (internal only)
         |                    |                  |
   Archive token        Django session      used by Django +
   Podcast token        + CSRF              task workers only
   OpenClaw token
   Operator token
```

---
