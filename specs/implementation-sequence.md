# Voxhelm Implementation Sequence

**Date:** 2026-03-11
**Status:** M1a and M1b completed on 2026-03-12; first M1c slice completed on 2026-03-12; later phases still draft
**Input:** `specs/2026-03-11_voxhelm_service.md`, `specs/milestones.md`

Current implementation checkpoint:

- Phases 1a and 1b are complete.
- The first Phase 1c slice is complete: `podcast-transcript` now supports `--backend voxhelm` and has been validated against the deployed edge service.
- The deployed runtime is a Django + `uvicorn` HTTP process plus a Django Tasks worker on `studio`.
- Private HTTPS ingress is live on `macmini` at `https://voxhelm.home.xn--wersdrfer-47a.de`.
- Batch jobs, MinIO-backed artifacts, video extraction, and artifact proxy download are live.
- `podcast-pipeline` is not yet config-only compatible with that backend because its current transcribe command contract does not pass or resolve an audio input for the external transcriber.
- Wyoming, TTS, and the remaining consumer integrations remain future phases.

## Execution Order Overview

```
Completed on 2026-03-12                              Remaining draft phases
──────────────────────────────────────────────────── ──────────────────────────────────
[M1a: sync API] [M1b: jobs+minio] [M1c: podcast-transcript]
[deploy + live verification]                         [M1c: remaining] [M2: Wyoming]
                                                    [M3: TTS batch]  [M4: OpenClaw]
```

---

## Phase 0: Optional Technical Spikes (Days 1-4 or later)

### Spike 0a: STT Backend Benchmark

**Question:** Which STT backend should be the default for batch transcription on `studio`?

**How to conduct:**

1. Install all three backends on `studio`:
   - whisper.cpp (via `whisper-cli` -- already known to work from podcast-transcript)
   - mlx-whisper (`pip install mlx-whisper` -- already a podcast-transcript backend)
   - WhisperKit (install Swift package or server binary)
2. Select 3 test audio files: a 45-second clip, a 10-minute podcast segment, and a 60-minute full episode.
3. For each backend with `large-v3` model:
   - Measure wall-clock transcription time
   - Measure peak RSS memory
   - Capture the transcript text for quality comparison
4. Record results in a markdown table.

**Expected duration:** 1-2 days (model download time dominates).

**Expected outcome:** A ranked recommendation. Based on podcast-transcript's existing code, whisper.cpp and mlx-whisper are known to work. WhisperKit is the unknown. If WhisperKit is significantly faster on Apple Silicon, it becomes the default; otherwise mlx-whisper or whisper.cpp (both proven) are safe choices.

**What depends on the answer:**
- M1a backend adapter selection
- Whether to invest in WhisperKit integration or defer it

### Spike 0b: WhisperKit Server Mode

**Question:** Does WhisperKit provide a stable HTTP server mode, or must Voxhelm wrap it as a subprocess?

**How to conduct:**

1. Check WhisperKit documentation and releases for server/API mode.
2. If a server mode exists: start it on `studio`, send a test transcription request via curl.
3. If no server mode: test CLI invocation from Python subprocess, measure startup overhead.

**Expected duration:** 0.5-1 day (part of Spike 0a).

**Expected outcome:** Either "WhisperKit has a server mode that speaks OpenAI-compatible API" (in which case Voxhelm can proxy to it) or "WhisperKit is CLI/library only" (in which case Voxhelm wraps it like podcast-transcript wraps whisper.cpp).

**What depends on the answer:**
- Backend adapter design: proxy vs. subprocess vs. library call
- Whether the sync endpoint in M1a can delegate to WhisperKit's own HTTP server

### Spike 0c: Wyoming Protocol Feasibility

**Question:** What is the minimum viable Wyoming integration for Home Assistant STT/TTS?

**How to conduct:**

1. Install `wyoming-faster-whisper` (or `wyoming-whisper`) on `studio`.
2. Install Piper and `wyoming-piper` on `studio`.
3. Configure Home Assistant to use these as STT and TTS providers.
4. Test a voice command end-to-end.
5. Document: what worked, what configuration was needed, what the latency felt like.

**Expected duration:** 1 day.

**Expected outcome:** Either "existing Wyoming packages work out of the box on `studio` with Apple Silicon" (M2 is primarily deployment + ops work) or "we need a custom Wyoming server that delegates to Voxhelm backends" (M2 requires more code).

**What depends on the answer:**
- M2 scope and effort estimate
- Whether M2 reuses Voxhelm's backend layer or runs independently

**Parallelization:** These spikes can run in parallel with M1a once the Archive-first path starts. They reduce uncertainty for backend expansion and M2, but they do not block the initial service build.

---

## Phase 1a: Synchronous STT Endpoint (Days 5-12)

### Critical path items

These must be completed in order:

1. **Django project skeleton** (day 5)
   - Create the `voxhelm` Django project
   - SQLite database with WAL mode
   - Basic settings, ALLOWED_HOSTS, CSRF, auth middleware
   - Service user and launchd plist for `studio` deployment (`studio` runs macOS, not Linux)

2. **Backend adapter layer** (day 5-6)
   - Python protocol: `class STTBackend(Protocol): def transcribe(self, audio: Path, params: TranscribeParams) -> TranscribeResult`
   - First adapter for `mlx-whisper`
   - `TranscribeResult` contains at minimum: `text: str`, `duration_seconds: float`

3. **OpenAI-compatible sync endpoint** (day 6-7)
   - `POST /v1/audio/transcriptions`
   - Request parsing for either multipart upload (`file`) or JSON URL input (`url`), plus `model`, `language`, `prompt`, `response_format`
   - Input validation: upload size (25 MiB for file uploads), MIME type, supported formats
   - Accept `gpt-4o-mini-transcribe` for zero-code-change Archive compatibility; accept `whisper-1` as an OpenAI-style alias
   - URL fetch policy: `https://` by default, `http://` only for explicitly configured trusted internal hosts
   - Bearer token authentication
   - Response: `{"text": "transcribed content..."}`
   - Error responses matching OpenAI error format (so Archive's error handling works)
   - Synchronous timeout budget aligned with Archive's current 300-second default; reject or redirect larger work to the batch API

4. **ops-library deployment role** (day 8-9)
   - Ansible role for Voxhelm on `studio`
   - Python venv, launchd service plists
   - Private Traefik ingress on `macmini`
   - MinIO bucket creation is deferred to M1b

5. **Archive switchover test** (day 9-10)
   - Set Archive env vars to point at Voxhelm
   - Run Archive's `run_metadata_worker` command
   - Verify transcripts appear for podcast episodes (audio items)
   - Measure latency and quality
   - Video item transcription is deferred to M1b (requires audio extraction support)

### Decision gate: Archive acceptance

Before proceeding to M1b, verify:
- [ ] Archive produces transcripts through Voxhelm without code changes (audio items only; video deferred to M1b)
- [ ] Transcription quality is acceptable (manual review of 5+ transcripts)
- [ ] No timeout failures for representative Archive audio items that fit the M1a sync scope and Archive's 300-second timeout budget
- [ ] Service is stable under the Archive metadata worker's sequential processing pattern

---

## Phase 1b: Batch Job System + MinIO (Completed on 2026-03-12)

### Implementation sequence

1. **Django Tasks + task tracking** (day 11-12)
   - Configure Django Tasks for production use on `studio` with `django_tasks_db.backend.DatabaseBackend`
   - Initial `TASKS` settings: `database_alias=default`, `poll_interval=1.0`, `max_attempts=3`
   - Define transcription tasks and task payload schema
   - Persist producer-facing task/job tracking with `task_ref`, requested formats, timestamps, result metadata, terminal state, and linked Django task/result id
   - Ensure idempotency for repeated producer submissions
   - Admin visibility for queued/running/completed work

2. **Job submission API** (day 12-13)
   - `POST /v1/jobs` -- accepts JSON job descriptor
   - Producer authentication (bearer token, producer identity)
   - Input validation: known job types, valid backend/model, input descriptor shape
   - Returns job ID and status

3. **Django Tasks worker runtime** (day 13-14)
   - Launch worker processes on `studio`
   - Execute queued transcription tasks using the backend adapter layer from M1a
   - Persist progress and terminal state back into the control plane
   - Recover unfinished work after restart using the configured task backend and task/job tracking data
   - Do not add a custom worker registry / handshake / heartbeat layer unless the stock runtime proves insufficient during deployment

4. **MinIO integration** (day 14-15)
   - `boto3` / `django-storages` for S3-compatible storage
   - Store input media to MinIO when submitted as upload
   - Store transcript artifacts to MinIO on job completion
   - Artifact references in job result: `{"artifacts": {"text": "/v1/jobs/{id}/artifacts/transcript.txt", ...}}`
   - HTTP proxy endpoint for artifact delivery (consumers access via Voxhelm, not directly from MinIO)

5. **Video preprocessing** (day 15-16)
   - ffmpeg audio extraction from video inputs
   - Automatic detection: if input MIME type is `video/*`, extract audio first
   - Store extracted audio as intermediate artifact

6. **Structured output formats** (day 16-17)
   - Whisper-format JSON (segments with timestamps)
   - Plain text
   - VTT
   - Artifact naming and HTTP proxy delivery for those canonical outputs

### Decision gate: Job system validation

Completed on 2026-03-12:
- [x] A batch job submitted via API completes end-to-end
- [x] A video URL input produces a transcript
- [x] MinIO artifacts are retrievable via Voxhelm's HTTP proxy endpoint
- [x] Duplicate `task_ref` submissions are rejected or return existing job
- [x] The deployed worker starts successfully under launchd after migrations run during deploy

---

## Phase 1c: Consumer Integrations (Days 17-24; first slice completed on 2026-03-12)

### Parallelizable work

The remaining consumer integrations are largely independent and could be done in any order. The recommended order still optimizes for risk (podcast-transcript was lowest risk and is now done; python-podcast remains the highest-risk stream).

#### Stream A: podcast-transcript backend (days 17-19)

**Implementation note (2026-03-12):** Delivered. The current implementation uses Voxhelm's synchronous `POST /v1/audio/transcriptions` path, requests `verbose_json`, keeps `podcast-transcript`'s existing output conversions local, and accepts `VOXHELM_API_BASE` values pointing at either the service root or `/v1`.

1. **New `Voxhelm` backend class** in podcast-transcript
   - Implements existing `TranscriptionBackend` protocol: `def transcribe(self, audio_file: Path, transcript_path: Path) -> None`
   - Internally: upload audio file to Voxhelm's `POST /v1/audio/transcriptions` (sync mode for small files) or submit batch job and poll (for large files)
   - Download result and write to `transcript_path` in whisper-compatible JSON format
   - Configuration via env vars: `VOXHELM_API_BASE`, `VOXHELM_API_KEY`

2. **CLI integration**
   - Add `--backend voxhelm` to podcast-transcript's argument parser
   - Factory function `voxhelm_from_settings` following existing pattern

3. **podcast-pipeline: follow-on still required**
   - podcast-pipeline shells out to an external transcriber command, but its current contract only passes `{mode}`, `{output_dir}`, and `{workspace}`
   - repo inspection after Stream A showed that no audio input is passed or resolved for the transcriber, so config-only switching is not currently possible
   - the remaining work is a small compatibility step or wrapper, not a new Voxhelm backend

#### Stream B: Additional STT backends (days 17-19)

1. **Second backend adapter** (whichever of whisperkit/mlx/whisper.cpp was not the M1a default)
2. **Third backend adapter**
3. **Backend selection logic:** `auto` mode selects based on availability and job lane (interactive vs. batch)
4. **Backend health checks:** verify each backend is functional on startup

**Can run in parallel with Stream A.**

#### Stream C: python-podcast / django-cast integration (days 19-23)

1. **Voxhelm client library** (or inline HTTP client in django-cast)
   - Submit batch transcription job with audio URL
   - Poll for job completion
   - Download artifacts

2. **Transcript format conversion** (server-side + consumer integration)
   - Voxhelm normalizes backend output into Whisper-native JSON internally
   - Voxhelm produces requested DOTe, Podlove, WebVTT, and plain-text artifacts server-side
   - Create/update django-cast `Transcript` model with the returned artifacts

3. **Management command: `generate_transcripts`**
   - Find Audio objects without Transcript
   - Submit Voxhelm job for each
   - Poll and populate Transcript models

4. **Optional: signal-based trigger**
   - On Audio save, schedule Voxhelm job if no transcript exists

### Decision gate: Consumer acceptance

Before declaring M1 complete:
- [x] `podcast-transcript --backend voxhelm <url>` produces the expected transcript artifacts through its existing output flow
- [ ] podcast-pipeline's `transcribe` command works with voxhelm backend configured
- [ ] python-podcast can generate transcripts for episodes through Voxhelm
- [ ] All three STT backends produce valid output
- [ ] Archive continues to work (regression check)

---

## Phase 2: Wyoming / Home Assistant (Days 20-30)

**Wyoming STT can start as soon as M1a is complete. Wyoming TTS follows only when Piper/TTS backend work from M3 is available.**

### Implementation sequence

1. **Wyoming STT server** (day 20-22)
   - Based on Spike 0c results, either:
     - **Option A:** Deploy `wyoming-faster-whisper` as companion service (if spike showed it works on `studio`)
     - **Option B:** Write thin Wyoming STT server in Voxhelm that delegates to the backend adapter layer
   - Launchd plist for Wyoming STT service
   - Configure port binding

2. **Optional Piper deployment + Wyoming TTS** (day 22-24)
   - This step depends on the TTS backend adapter work in M3. If that work is not ready, M2 ships STT only.
   - Install Piper on `studio`
   - Download voice models (at least one English, one German)
   - Deploy Wyoming TTS service via launchd
   - Configure separate port binding

3. **Interactive lane scheduling** (day 24-26)
   - If Option B: Voxhelm's backend adapter must prioritize interactive requests
   - Mechanism: batch worker pauses/yields when interactive request arrives
   - Simpler alternative if Option A: Wyoming processes have their own model instances, no contention with batch

4. **Home Assistant configuration** (day 26-27)
   - Add Wyoming STT provider in HA config
   - Add Wyoming TTS provider if the optional TTS step shipped
   - Test voice pipeline end-to-end
   - Document setup in ops-control runbook

5. **Deployment role** (day 27-28)
   - ops-library role for Wyoming companion processes
   - Model download automation
   - Health monitoring

### Decision gate: HA voice acceptance

- [ ] Voice command through HA device works end-to-end for STT
- [ ] STT latency is acceptable for interactive use (< 2 seconds for a typical command)
- [ ] If Wyoming TTS shipped, Piper produces intelligible responses
- [ ] Batch transcription jobs are not noticeably degraded during voice use

---

## Phase 3: TTS Batch Generation (Days 28-35)

### Implementation sequence

1. **TTS backend adapter** (day 28-29)
   - `class TTSBackend(Protocol): def synthesize(self, text: str, params: SynthesizeParams) -> SynthesizeResult`
   - Piper adapter (reuse M2 deployment)
   - Voice/model preset configuration

2. **Synthesize job type** (day 29-30)
   - Add `synthesize` to job type enum
   - Job submission with text input, voice preset, output format
   - Worker processes synthesize jobs

3. **Sync TTS endpoint** (day 30-31)
   - `POST /v1/audio/speech`
   - Text input, voice selection, response as audio stream
   - Size limits on input text

4. **Archive integration** (day 31-33)
   - Archive submits article text for TTS
   - Receives a stable Voxhelm artifact URL backed by MinIO
   - Attaches as feed enclosure

---

## Phase 4: OpenClaw Integration (Days 35+)

Deferred until OpenClaw's API needs are concrete. Implementation is expected to be minimal: OpenClaw calls the existing HTTP API. The main work is documentation and an example plugin.

---

## Critical Path

```
M1a (sync endpoint) -> M1b (jobs) -> M1c (consumers)
          |                    |
          ├-> optional backend spikes / expansion
          └-> M2 (Wyoming) ----┴----> M3 (TTS batch)
```

The critical path to first production value is:

**M1a (1-2 weeks) -> Archive switchover**

Everything else can follow incrementally.

---

## Parallelization Matrix

| Work Item | Can Start After | Independent Of |
|-----------|----------------|---------------|
| Spike 0a+0b | Day 1 | M1a |
| Spike 0c | Day 1 | M1a, M1b |
| M1a: Django skeleton | Day 1 | Spike 0c |
| M1a: Backend adapter | Day 1 | M1b, M2 |
| M1a: Sync endpoint | Backend adapter | M1b, M2, M1c |
| M1a: ops-library role | Day 1 (infra) | M1a code |
| M1b: Django Tasks setup | M1a | M2, M1c |
| M1b: MinIO integration | M1a (deployment) | Django Tasks setup (can prototype early) |
| M1c: podcast-transcript backend | M1a (sync endpoint) | M1b (can use sync mode only) |
| M1c: Additional backends | M1a | M1b, M1c streams |
| M1c: python-podcast integration | M1b (needs job API) | podcast-transcript backend |
| M2: Wyoming | M1a + Spike 0c | M1b, M1c |
| M3: TTS | M1b + M2 (Piper) | M1c |
| M4: OpenClaw | M1a + M3 | M1c |

---

## Deployment And Verification Order

1. Deploy M1a to `studio` and verify `GET /v1/health`, then run the Archive switchover test against audio inputs only.
2. Deploy M1b on the same host with Django Tasks workers and MinIO connectivity, then verify URL input, artifact proxying, restart recovery, and video preprocessing.
3. Roll consumer integrations in M1c one stream at a time: podcast-transcript first, then indirect podcast-pipeline validation, then python-podcast / django-cast.
4. Start M2 once M1a is stable; verify Wyoming STT first, then add Wyoming TTS only if the Piper/TTS dependency is ready.
5. Start M3 after M1b and Piper readiness; verify batch synthesize jobs before wiring Archive article-audio usage.
