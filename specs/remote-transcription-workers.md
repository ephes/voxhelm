# Remote Transcription Workers

**Date:** 2026-05-30  
**Status:** Accepted implementation concept for the next remote-worker slice; no code implementation started.  
**Chosen architecture:** Option B -- internal HTTP pull-worker API.  
**First worker target:** `atlas.local`.  
**Goal-complete validation:** real production python-podcast known-speaker diarized transcript executed on `atlas.local`.

## Goal

Let additional machines take long-running Voxhelm batch transcription work while `studio` remains the stable control plane and public/private HTTP entrypoint.

The first target is a two-machine setup:

- `studio`: Voxhelm API, operator UI, producer auth, job metadata, MinIO/S3 artifact storage, existing local worker fallback, Wyoming sidecar.
- `atlas.local`: remote batch transcription worker that connects outbound to Voxhelm on `studio`, claims eligible work, runs local STT plus the required diarization/known-speaker path, writes artifacts, and reports results.

Producer-facing APIs must not change. Consumers continue using:

- `POST /v1/jobs`
- `GET /v1/jobs/{id}`
- `GET /v1/jobs/{id}/artifacts/{name}`

## Non-goals for the first slice

- No remote Wyoming/interactive voice work.
- No remote synchronous `/v1/audio/transcriptions` execution.
- No generic remote shell/job runner.
- No public worker registration by untrusted machines.
- No automatic cross-host preemption of work already running.
- No mandatory remote synthesis support in the first slice.
- No broad scheduler dashboard requirement.
- No PostgreSQL migration requirement for the first slice.
- No producer-facing API changes in python-podcast/django-cast/Voxhelm.

## Accepted first-slice decisions

### Architecture

Use an internal HTTP pull-worker API:

- `studio` owns the database, producer API, job state, and authorization decisions.
- Remote workers authenticate to `studio`, heartbeat capabilities, claim leased jobs, execute work locally, and report completion/failure.
- Remote workers do not receive database credentials.
- Producer-facing job IDs, polling, and artifact download URLs remain unchanged.

### Dispatch mode

Use a remote-only queue for batch transcription when remote workers are enabled:

- `job_type=transcribe` jobs are created in Voxhelm as normal producer-visible jobs.
- When configured for remote execution, those jobs are **not** enqueued into Django Tasks.
- Remote workers claim jobs directly from the Voxhelm job table through the worker API.
- `job_type=synthesize` continues using the existing Django Tasks path.
- Existing local transcription execution can be restored by configuration, without a migration, by switching transcription dispatch back to Django Tasks.

Suggested setting shape:

```bash
VOXHELM_TRANSCRIPTION_EXECUTION_MODE="django_tasks"   # current/default safe mode
VOXHELM_TRANSCRIPTION_EXECUTION_MODE="remote_pull"   # remote-worker mode
```

Avoid mixing Django Tasks transcription and remote-pull transcription for the same job pool in the first slice. The goal is to prevent double execution, not to build a multi-scheduler fallback system immediately.

### Artifact handoff

Use B1 for the first remote-worker implementation: trusted remote workers receive MinIO/S3 credentials and post artifact manifests back to Voxhelm.

This deliberately widens the current artifact credential boundary from "only the local Voxhelm deployment on `studio`" to "Voxhelm control plane plus trusted Voxhelm worker hosts". It is acceptable for the first `atlas.local` slice because it minimizes new media proxy/upload code and lets the remote worker reuse the existing artifact-store abstraction.

Implications:

- `interface-map.md` must describe trusted remote workers as part of the internal artifact credential domain.
- MinIO credentials must be stored in protected worker environment files, not command lines or logs.
- Remote workers must only write under attempt-scoped job prefixes such as `jobs/<job_id>/attempt-<attempt>/...` or the configured `VOXHELM_ARTIFACT_PREFIX` equivalent. Do **not** write final artifacts directly to stable names like `jobs/<job_id>/transcript.json` from the worker.
- Attempt-scoped prefixes prevent a zombie worker from overwriting the winning worker's S3 objects after a lease reclaim. The database manifest chooses which attempt is producer-visible.
- Completion must validate artifact names/kinds/content types before exposing them through producer artifact URLs.

B2/B3 remain later alternatives if the credential boundary should be tightened after the first worker proves useful.

### C13 lane scheduling

The existing C13 host-wide lane scheduler is local to `studio`:

- `atlas.local` inference is outside the `studio` scheduler and does not need to acquire the `studio` lane gate.
- Any future local pull worker running on `studio` must still use the C13 admission path before STT/TTS inference.
- The first Option B slice does not migrate the local `studio` transcription worker to pull-worker execution, so this remains a follow-up constraint rather than a first-slice implementation task.

### Diarization and known-speaker work

The implementation goal is broader than a plain-transcription worker: goal completion requires a production python-podcast known-speaker diarization job to run on `atlas.local` and produce the same private `speakers` sidecar quality expected from the existing `studio` path.

Implementation may be sliced so early infrastructure initially claims only `diarization.enabled=false` jobs, but the remote-worker effort is **not complete** until `pyannote_known_speaker` jobs are eligible and validated end-to-end.

Before implementing the known-speaker slice, choose and document one execution placement:

1. **Atlas-runs-known-speaker (preferred default).** `atlas.local` receives the Hugging Face/pyannote/wespeaker configuration and fetches contributor references, computes centroids, runs pyannote + known-speaker classification locally, uploads `transcript.speakers.json`, and posts a completion manifest. This keeps `studio` as the control plane and avoids a second hybrid job phase.
2. **Hybrid classification.** `atlas.local` runs STT and uploads canonical transcript/audio artifacts; `studio` runs the known-speaker postprocessor against the uploaded job audio and private references before final completion. This is **design-on-selection, not first-slice ready**: choosing it requires adding an explicit awaiting-postprocess state/trigger before implementation. Choose this only if private-reference access or model deployment makes Atlas-runs-known-speaker impractical.

Either approach must preserve the producer-facing `POST /v1/jobs` contract and must populate the django-cast `Transcript.speakers` sidecar through the normal production python-podcast flow.

Known-speaker remote execution widens a privacy boundary: the normalized claim payload sends contributor names/ids plus private reference descriptors from `studio` to the trusted worker. Worker logs, progress messages, and result metadata must not expose private reference URLs, ranges, candidate details beyond the intended speakers sidecar, tokens, or credentials.

Known-speaker remote eligibility requires worker capabilities for:

- `diarization.strategy=pyannote_known_speaker`;
- pyannote diarization backend and Hugging Face token access;
- `pyannote/wespeaker-voxceleb-resnet34-LM` or the requested embedding model;
- fetching source-range reference audio from hosts allowed by `VOXHELM_ALLOWED_URL_HOSTS`;
- producing the `transcript_speakers` artifact kind.

## Current constraints

Voxhelm already has a clean producer-facing batch API and a separated worker execution path, but the current runtime is local:

- `create_job_from_payload(...)` enqueues Django Tasks work immediately today.
- `django_tasks_db.backend.DatabaseBackend` currently uses the Django database as the task queue.
- The production service historically assumes the HTTP process and Django Tasks worker are on `studio`.
- SQLite is acceptable for local low-concurrency coordination, but should not become a remotely shared worker database.
- SQLite has no row-level `select_for_update()` semantics, so remote claiming must use short atomic updates and affected-row checks rather than long row locks.
- There is a known open idempotency race for concurrent same-`task_ref` submissions. Remote worker claiming must not introduce an equivalent double-claim race.

## Worker packaging and onboarding

Adding another worker machine should not require cloning private deployment logic or configuring database access. The worker should be installable and runnable as a normal Python command with only Voxhelm URL, worker credentials, artifact credentials, backend/model settings, and optional diarization secrets.

Preferred operator experience after the first implementation:

```bash
uv tool install "voxhelm[diarization] @ git+ssh://git.example/voxhelm.git"
voxhelm-remote-worker \
  --base-url https://voxhelm.home.xn--wersdrfer-47a.de \
  --worker-id atlas \
  --env-file /etc/voxhelm-worker/worker.env
```

A one-shot/smoke-test mode should also work through `uvx` once packaging is available:

```bash
uvx --from "voxhelm[diarization] @ git+ssh://git.example/voxhelm.git" \
  voxhelm-remote-worker --once --base-url https://voxhelm.home.xn--wersdrfer-47a.de
```

Repository-checkout mode remains acceptable for development:

```bash
uv run voxhelm-remote-worker --once --base-url http://studio.local:8000
```

The worker command should not require Django server setup, database settings, migrations, or access to `studio`'s SQLite database. It may import Voxhelm's shared transcription, diarization, artifact-store, and format-rendering code, but it should run as a worker-only process.

Minimum worker environment:

```bash
VOXHELM_WORKER_ID="atlas"
VOXHELM_WORKER_TOKEN="replace-me"
VOXHELM_BASE_URL="https://voxhelm.home.xn--wersdrfer-47a.de"
VOXHELM_ARTIFACT_BACKEND="s3"
VOXHELM_ARTIFACT_S3_ENDPOINT_URL="https://minio.example"
VOXHELM_ARTIFACT_S3_ACCESS_KEY_ID="replace-me"
VOXHELM_ARTIFACT_S3_SECRET_ACCESS_KEY="replace-me"
VOXHELM_ARTIFACT_BUCKET="voxhelm"
VOXHELM_MODEL_CACHE_DIR="/var/lib/voxhelm-worker/models"
VOXHELM_STT_BACKEND="whispercpp"
VOXHELM_WHISPERCPP_MODEL="ggml-large-v3.bin"
VOXHELM_HUGGINGFACE_TOKEN="replace-me"  # required for pyannote/wespeaker jobs
```

Adding a future worker should be a repeatable ops task:

1. Install `ffmpeg` and the chosen STT backend/model.
2. Install the Voxhelm worker package with `uv tool install` or run it with `uvx`.
3. Create a protected worker env file containing worker token, Voxhelm base URL, MinIO credentials, model cache, and optional Hugging Face token.
4. Start `voxhelm-remote-worker --env-file ...` under launchd or another supervisor.
5. Confirm the worker appears through heartbeat and can claim only jobs matching its advertised capabilities.

## Worker auth

Add a separate worker credential domain. Producer bearer tokens must not authorize worker endpoints.

Suggested environment shape:

```bash
VOXHELM_WORKER_TOKENS="atlas=replace-me"
```

Token behavior:

- Worker requests use `Authorization: Bearer <token>`.
- A token maps to exactly one configured `worker_id`, for example `atlas`.
- The request body may include `worker_id`, but the authenticated token identity wins.
- Unknown, disabled, or mismatched worker IDs return `401` or `403`.
- Tokens are never returned by the API and must not appear in logs.

## Worker API

All worker endpoints are internal and should be exposed only on the private Voxhelm network/ingress. The public/private edge on `macmini`/Traefik must not route `/v1/internal/*` to Voxhelm unless it is a deliberately private worker route; block those paths at the edge in addition to requiring worker-token auth.

### `POST /v1/internal/workers/heartbeat`

Purpose: register liveness and current capabilities.

Request:

```json
{
  "worker_id": "atlas",
  "hostname": "atlas.local",
  "version": "0.1.0",
  "concurrency": 1,
  "capabilities": {
    "job_types": ["transcribe"],
    "backends": ["whispercpp", "mlx"],
    "models": ["ggml-large-v3.bin", "mlx-community/whisper-large-v3-mlx"],
    "output_formats": ["text", "json", "vtt", "dote", "podlove", "speakers"],
    "diarization": {
      "anonymous": true,
      "known_speaker": true,
      "embedding_models": ["pyannote/wespeaker-voxceleb-resnet34-LM"]
    }
  },
  "running_job_ids": []
}
```

Response:

```json
{
  "worker_id": "atlas",
  "enabled": true,
  "server_time": "2026-05-30T12:00:00Z",
  "poll_after_seconds": 5
}
```

### `POST /v1/internal/work/claim`

Purpose: claim one eligible job and create/extend the execution lease.

Request:

```json
{
  "worker_id": "atlas",
  "max_jobs": 1,
  "capabilities": {
    "job_types": ["transcribe"],
    "backends": ["whispercpp", "mlx"],
    "models": ["ggml-large-v3.bin", "mlx-community/whisper-large-v3-mlx"],
    "diarization": {
      "anonymous": true,
      "known_speaker": true,
      "embedding_models": ["pyannote/wespeaker-voxceleb-resnet34-LM"]
    }
  }
}
```

No work response: `204 No Content`.

Claim response:

```json
{
  "job": {
    "id": "7d9b9f0b-4c0d-4d0d-9e1a-4c3f4d2f3f1a",
    "job_type": "transcribe",
    "attempt": 1,
    "lease_token": "opaque-random-token",
    "lease_expires_at": "2026-05-30T12:30:00Z",
    "backend": "auto",
    "model": "auto",
    "language": "de",
    "input": {
      "kind": "url",
      "url": "https://media.example.com/episode.mp3"
    },
    "output": {
      "formats": ["text", "json", "vtt", "dote", "podlove"],
      "diarization": {"enabled": false}
    },
    "artifact_prefix": "jobs/7d9b9f0b-4c0d-4d0d-9e1a-4c3f4d2f3f1a/attempt-1/"
  }
}
```

For staged uploads, the claim response should include enough object-store metadata for the worker to copy the staged object into the job-owned source artifact before inference:

```json
{
  "input": {
    "kind": "upload",
    "filename": "episode.mp3",
    "content_type": "audio/mpeg",
    "size_bytes": 123456789,
    "staged_artifact": {
      "storage_backend": "s3",
      "storage_key": "staged/<upload-id>/episode.mp3"
    }
  }
}
```

For known-speaker jobs, the claim response includes the normalized `diarization` payload from the producer request, including `enabled`, `strategy`, speaker-count hints, `known_speakers`, reference descriptors, and `known_speaker` thresholds. The worker must consume the normalized object; it must not re-derive `strategy` from `enabled`. Worker logs must not print private reference URLs or ranges. If the chosen placement is Atlas-runs-known-speaker, the worker fetches reference audio with the same allow-list and private-media rules as the current `studio` path. If the chosen placement is hybrid, the claim/complete contract must first define an awaiting-postprocess state and studio-side trigger rather than marking the job succeeded at Atlas completion.

The worker should not need database access to resolve staged uploads or known-speaker references.

### `POST /v1/internal/work/{job_id}/heartbeat`

Purpose: extend a running job lease while the worker is still making progress.

Request:

```json
{
  "worker_id": "atlas",
  "lease_token": "opaque-random-token",
  "progress": {
    "phase": "transcribing",
    "message": "running whisper.cpp"
  }
}
```

Response:

```json
{
  "job_id": "7d9b9f0b-4c0d-4d0d-9e1a-4c3f4d2f3f1a",
  "lease_expires_at": "2026-05-30T12:35:00Z",
  "server_time": "2026-05-30T12:05:00Z"
}
```

### `POST /v1/internal/work/{job_id}/complete`

Purpose: atomically mark a leased job as succeeded and persist the worker-produced artifact records.

Request:

```json
{
  "worker_id": "atlas",
  "lease_token": "opaque-random-token",
  "result_text": "Plain transcript text...",
  "result_metadata": {
    "backend_name": "whisper.cpp",
    "model_name": "ggml-large-v3.bin",
    "language": "de",
    "processing_seconds": 1234.5,
    "worker_id": "atlas",
    "attempt": 1
  },
  "artifacts": [
    {
      "name": "source.mp3",
      "kind": "source",
      "format": "mp3",
      "storage_backend": "s3",
      "storage_key": "jobs/7d9b9f0b-4c0d-4d0d-9e1a-4c3f4d2f3f1a/attempt-1/source.mp3",
      "content_type": "audio/mpeg",
      "size_bytes": 123456789,
      "exposed": false
    },
    {
      "name": "transcript.txt",
      "kind": "transcript_text",
      "format": "text",
      "storage_backend": "s3",
      "storage_key": "jobs/7d9b9f0b-4c0d-4d0d-9e1a-4c3f4d2f3f1a/attempt-1/transcript.txt",
      "content_type": "text/plain; charset=utf-8",
      "size_bytes": 12345,
      "exposed": true
    }
  ]
}
```

Completion rules:

- The server accepts completion only from the currently assigned worker with the current lease token.
- Completion is idempotent for the same `(job_id, lease_token)` after a successful commit: if a worker retries the same completion because the HTTP response was lost, return the already-succeeded job/manifest. If the retry supplies a different manifest or the token no longer matches the recorded successful attempt, return a conflict.
- Artifact `storage_key` values must be under the claimed attempt-scoped artifact prefix.
- Artifact names must match existing Voxhelm artifact naming rules and be unique per job.
- Transcript output formats must match the job's requested `output.formats`.
- Known-speaker jobs must include the `transcript_speakers` artifact (`transcript.speakers.json`) and `result_metadata.diarization.known_speaker_summary` when the request strategy is `pyannote_known_speaker`.
- The `speakers` sidecar remains private/reviewable consumer state; public DOTe/Podlove/VTT labeling behavior must stay consistent with `specs/known-speaker-diarization.md` and django-cast's review/apply policy.
- On success, the server sets `state=succeeded`, `finished_at`, `result_text`, `result_metadata`, and creates `JobArtifact` rows.
- For staged uploads, after a successful source artifact copy is recorded, `studio` may delete the staged object/row as part of completion cleanup.
- Losing attempts may leave orphaned attempt-scoped objects; cleanup can be a later retention/sweep task and must not affect the manifest-selected winning attempt.

### `POST /v1/internal/work/{job_id}/fail`

Purpose: report a worker failure before the lease expires.

Request:

```json
{
  "worker_id": "atlas",
  "lease_token": "opaque-random-token",
  "retryable": true,
  "error_detail": "whisper.cpp exited with status 1"
}
```

Failure rules:

- The server accepts heartbeat/fail only from the currently assigned worker with the current lease token hash, same as completion. A stale worker must not be able to extend or fail a reassigned job.
- If `retryable=true` and attempts remain, reset the job to `queued`, clear assignment, and keep error detail in metadata/logs.
- If `retryable=false` or attempts are exhausted, mark the job `failed` and set `error_detail`.
- The first slice can use a conservative maximum such as `VOXHELM_REMOTE_WORKER_MAX_ATTEMPTS=3`.

## Data model additions

Names are implementation guidance, not final migration names.

### `Worker`

A small runtime/liveness model:

- `id` / `worker_id` string, for example `atlas`
- `hostname`
- `enabled`
- `capabilities` JSON
- `concurrency` integer
- `last_seen_at`
- `running_job_ids` JSON/list
- `created_at`, `updated_at`

Token storage can remain environment-backed for the first slice; no token needs to be stored in this model.

### `Job` additions

- `execution_mode`: `django_tasks` or `remote_pull`
- `assigned_worker_id`: nullable string or FK to `Worker`
- `lease_token_hash`: nullable string
- `lease_expires_at`: nullable datetime
- `attempt_count`: integer default `0`
- `max_attempts`: integer default from settings
- `last_worker_heartbeat_at`: nullable datetime
- optional `worker_progress`: JSON for last phase/message

Indexes to consider:

- `(execution_mode, state, priority, created_at)` for queued claim scans
- `(assigned_worker_id, state)` for worker visibility
- `(lease_expires_at)` for stale lease cleanup/reclaim

## Claim semantics

Claiming must be SQLite-safe and short-lived.

Server-side algorithm shape:

1. Find a small ordered candidate set with `execution_mode=remote_pull`, eligible type/capabilities, and either:
   - `state=queued`, or
   - `state=running` with `lease_expires_at < now` and `attempt_count < max_attempts`.
2. For each candidate, attempt a single conditional `UPDATE` that sets:
   - `state=running`
   - `assigned_worker_id=<authenticated worker>`
   - `lease_token_hash=hash(new_token)`
   - `lease_expires_at=now + lease_seconds`
   - `attempt_count=attempt_count + 1`
   - attempt-scoped artifact prefix derived from the new attempt number
   - `started_at=COALESCE(started_at, now)`
   - `last_worker_heartbeat_at=now`
3. The `WHERE` clause must restate the claim conditions so a concurrent claimant loses cleanly.
4. Treat affected row count `0` as a lost race and try the next candidate.
5. Return the new opaque lease token only once, in the successful claim response.

All lease and heartbeat comparisons use `studio` server time. Worker-supplied timestamps are informational only.

## Worker execution loop

The first worker command should be a simple long-running loop:

1. Send worker heartbeat.
2. Claim at most one job.
3. If no job is available, sleep `poll_after_seconds` and repeat.
4. Materialize input:
   - URL input: `studio` validates allowed hosts at submission/claim time, and the worker re-validates the claim-provided URL against the same allow-list before downloading. A worker must reject URLs that fail its local allow-list rather than trusting arbitrary claim payloads.
   - Upload input: download/copy the staged object from MinIO using claim-provided storage metadata.
5. Store the job-owned source artifact under the attempt-scoped job artifact prefix.
6. If source is video, extract audio locally and store an `extracted_audio` artifact.
7. Run STT locally using the configured backend/model.
8. If the job requests anonymous diarization or `pyannote_known_speaker`, run the chosen diarization/known-speaker placement:
   - Atlas-runs-known-speaker: run pyannote, fetch references, compute embeddings/centroids, classify segments, and build the `speakers` sidecar locally.
   - Hybrid: upload required transcript/audio artifacts and let `studio` run the known-speaker postprocessor before producer-visible completion.
9. Render requested transcript artifacts locally using the same Voxhelm format functions.
10. Upload artifacts to MinIO/S3 under the attempt-scoped prefix returned by the claim response.
11. POST completion manifest.
12. Clean local temporary files.
13. Repeat.

The worker should heartbeat the leased job periodically while downloading, extracting, transcribing, rendering, and uploading. A reasonable first default is every 60 seconds with a lease duration of 30 minutes; long jobs rely on heartbeat extension rather than an enormous static lease.

## Capability matching

First-slice claim eligibility:

- `job_type` must be `transcribe`.
- `execution_mode` must be `remote_pull`.
- `state` must be `queued` or stale `running` with attempts remaining.
- requested `model`/`backend` must be compatible with worker capabilities:
  - `auto` can match any worker that advertises a configured default backend/model;
  - explicit models require exact advertised support or a server-owned alias map;
  - `whisperkit` should not match unless the worker explicitly advertises it.
- `diarization.enabled=false` or omitted can match a plain transcription-capable worker.
- `diarization.strategy=pyannote` requires anonymous diarization capability.
- `diarization.strategy=pyannote_known_speaker` requires known-speaker capability, the requested embedding model, reference-fetch support, and `transcript_speakers` artifact support.
- If no worker advertises the required diarization/known-speaker capability, the job must remain queued or fail clearly according to the chosen operator policy; it must not silently run as plain transcription.

Priority order should follow current job priority semantics: high before normal before low, then oldest first inside each priority.

## Status and serialization

Producer-visible `GET /v1/jobs/{id}` should not expose lease tokens or worker credentials.

It may include minimal worker metadata in `result.metadata` after completion, for example:

```json
{
  "worker_id": "atlas",
  "attempt": 1,
  "execution_mode": "remote_pull"
}
```

Queued/running/succeeded/failed semantics should remain compatible with existing consumers.

## Operational starter plan for `atlas.local`

Before implementation/deployment, validate manually:

- `atlas.local` can install/run the worker through the documented `uv tool install`, `uvx`, or checkout-based `uv run` path.
- `atlas.local` can reach Voxhelm on `studio` over the private network.
- `atlas.local` can reach MinIO/S3.
- `ffmpeg` is installed and matches Voxhelm expectations.
- at least one STT backend works locally on `atlas.local`:
  - `whisper.cpp` with the configured model, or
  - `mlx-whisper` with the configured model.
- pyannote and `pyannote/wespeaker-voxceleb-resnet34-LM` work locally on `atlas.local`, or the hybrid postprocessor path is explicitly chosen and implemented.
- `VOXHELM_ALLOWED_URL_HOSTS` / worker equivalent includes the CloudFront media host used for production source-range references.
- model cache paths are explicit and not assumed to match `studio`.
- protected worker env file contains worker token, MinIO credentials, and any required Hugging Face token/model-cache settings.
- worker process supervision is clear: launchd or another existing ops pattern.
- logs are local to `atlas.local` but contain Voxhelm job IDs and worker ID for correlation.

## Backlog chunks

### RW-1: Spec normalization and migration plan

- Update `interface-map.md` and `decision-log.md` with Option B as accepted.
- Document trusted remote workers as part of the internal MinIO credential domain.
- Add implementation ticket boundaries for model migrations, API endpoints, worker command, and deployment.

### RW-2: `atlas.local` capability smoke test

- Verify network reachability to `studio` and MinIO.
- Verify local STT backend and model availability.
- Record speed/quality baseline on one representative short and one representative long file.

### RW-3: Worker auth, heartbeat, and claim contract

- Add worker token parsing/auth.
- Add `Worker` persistence/liveness.
- Add claim, lease heartbeat, complete, and fail endpoints.
- Add SQLite-safe atomic job claim with lease and retry semantics.
- Keep endpoints private and non-producer-facing.

### RW-4: Remote transcription worker command and packaging

- Add a worker command/process, e.g. `voxhelm-remote-worker`, that polls/claims work, runs transcription, uploads artifacts, and reports result.
- Make it runnable from a repository checkout with `uv run` and installable/runnable on a new machine with `uv tool install` or `uvx` once packaging is available.
- Keep worker startup configuration to env file / CLI options for Voxhelm base URL, worker id/token, artifact credentials, model cache, backend/model, and optional Hugging Face token.
- Default to one concurrent job per worker.
- Include structured logs with worker ID and Voxhelm job ID.

### RW-5: Remote known-speaker diarization

- Choose Atlas-runs-known-speaker or hybrid classification and document the choice.
- Support `pyannote_known_speaker` job payloads from django-cast/python-podcast without producer API changes.
- Ensure reference audio fetching, pyannote/wespeaker model access, `transcript.speakers.json`, and `result_metadata.diarization` match the existing `studio` behavior.
- Validate quality against `python-podcast/docs/known-speaker-runbook.rst` and `evals/known_speaker_results.md`.

### RW-6: Remote dispatch mode

- Add `VOXHELM_TRANSCRIPTION_EXECUTION_MODE` or equivalent.
- In `remote_pull` mode, create transcribe jobs without enqueuing Django Tasks.
- Keep synthesis and local fallback behavior unchanged.
- Ensure `serialize_job`/state reconciliation does not require `django_task_id` for remote-pull jobs.

### RW-7: Deployment and operations

- Add deployment variables for worker token, Voxhelm base URL, artifact access, model cache, backend selection, lease durations, max attempts, concurrency, Hugging Face token, and allowed reference URL hosts.
- Add a repeatable new-worker onboarding recipe using `uv tool install`/`uvx`, a protected env file, and launchd.
- Add launchd service for `atlas.local`.
- Add minimal operator/log visibility for queued/running/worker-stale states if logs are not sufficient.

## Acceptance criteria for the first deployed goal

- A production python-podcast episode transcript is generated through the normal Wagtail/django-cast "Generate transcript" flow, which submits to Voxhelm on `studio` using the unchanged producer-facing `POST /v1/jobs` API.
- The Voxhelm job is claimed and executed by `atlas.local` (`worker_id=atlas` in job metadata/log evidence), not by the local `studio` Django Tasks transcription worker.
- The job requests `pyannote_known_speaker` and produces proper known-speaker diarization:
  - private `transcript.speakers.json` / django-cast `Transcript.speakers` sidecar populated;
  - known Johannes passage attributed;
  - quality meets `python-podcast/docs/known-speaker-runbook.rst` bar (`>=90%` segment top-1 on the gold-set passages, with uncertain segments treated according to the runbook policy).
- Result artifacts remain available through Voxhelm's existing `GET /v1/jobs/{id}/artifacts/{name}` endpoint.
- Evidence is captured: Voxhelm job id, worker log/metadata proving Atlas execution, diarized result/quality score, and proof the submission came through python-podcast production rather than a hand-crafted job or dev harness.
- Remote-pull transcription jobs are not also executed by the Django Tasks transcription worker.
- If `atlas.local` dies mid-job, the job becomes claimable again after the lease expires or fails clearly after bounded attempts.
- `studio` and `atlas.local` do not execute the same job concurrently.
- The worker never needs database credentials.
- Adding another machine is documented and repeatable with `uv tool install`/`uvx` plus Voxhelm URL and credentials; no Django server/database setup is required on the worker host.
- Existing local `studio` transcription operation can be restored by configuration if the remote worker is disabled.
