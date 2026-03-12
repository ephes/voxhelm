# PRD: Media Service

**Date:** 2026-03-11
**Status:** PRD reference; M1a and M1b implemented on 2026-03-12, later milestones still planned

## Summary

Implementation note as of 2026-03-12:

- Milestones 1a and 1b are implemented today.
- The deployed service includes the Archive-first synchronous STT slice plus the first batch-job slice:
  - `GET /v1/health`
  - `POST /v1/audio/transcriptions`
  - `POST /v1/jobs`
  - `GET /v1/jobs/{id}`
  - `GET /v1/jobs/{id}/artifacts/{name}`
  - `mlx-whisper`
  - bearer-token auth
  - multipart upload and JSON URL input
  - Django Tasks worker execution
  - MinIO-backed artifact storage via the `VOXHELM_ARTIFACT_*` S3-compatible env vars
  - video-to-audio extraction for batch work
  - private HTTPS ingress via Traefik on `macmini`
- Wyoming, TTS, and broader consumer integrations remain deferred to later milestones.

Create a small self-hosted media service that provides shared speech and media processing capabilities for:

- Archive
- python-podcast / django-cast
- podcast-pipeline
- Home Assistant
- OpenClaw
- future operator tooling or automations

The service should run on `studio` and act as the central place for:

- speech-to-text (STT)
- text-to-speech (TTS)
- audio extraction from video
- audio normalization/transcoding
- asynchronous batch transcription/synthesis jobs
- low-latency interactive speech services for Home Assistant voice devices and similar clients

The service should not require Archive, Home Assistant, or OpenClaw to remotely execute arbitrary commands on `studio`. Instead, it should expose narrow, purpose-built interfaces and use declarative job contracts.

## Problem

Several current and planned products need overlapping media capabilities:

- Archive needs asynchronous transcription for podcast and video items.
- python-podcast / django-cast needs podcast transcript generation and should be able to use the same shared API instead of a one-off transcript path.
- podcast-pipeline already has transcript-first production workflows and should be able to use the shared service for transcript generation while continuing to use `summarize.sh` or other downstream summarization tooling where appropriate.
- Archive may later need article-to-audio generation for a spoken feed.
- Home Assistant with the Nabu Casa voice device needs STT and TTS for local/private voice interactions.
- OpenClaw may later benefit from speech input/output and shared media processing.

Today, these concerns are fragmented and risk growing into multiple one-off integrations:

- Archive currently assumes an OpenAI-compatible transcription API and has a hosted-API-oriented implementation.
- local transcription on the Intel macmini is likely too slow or operationally unattractive.
- `studio` is the best compute host, but direct remote execution from web apps is not acceptable.
- Home Assistant and OpenClaw should not each grow separate, incompatible speech stacks if one shared local service can support them safely.

The result is a clear need for one secure local service that centralizes media processing without over-coupling callers to a single backend or model family.

## Goals

- Provide a shared media-processing service on `studio`.
- Support interchangeable STT backends:
  - WhisperKit
  - `mlx-whisper`
  - `whisper.cpp`
- Support interchangeable TTS backends over time.
- Support both:
  - low-latency interactive requests
  - asynchronous long-running batch jobs
- Support audio extraction from video inputs.
- Support Home Assistant integration using native local voice-friendly protocols.
- Support Archive integration without requiring dedicated OpenAI/Anthropic API keys.
- Support python-podcast / django-cast integration through an API-first transcript workflow.
- Support podcast-pipeline as another transcript-producing consumer.
- Support future OpenClaw integration through a narrow API/plugin surface.
- Avoid unsafe remote execution from Archive, Home Assistant, or OpenClaw to `studio`.
- Keep operational complexity reasonable for a homelab deployment.

## Non-goals

- Building a generic public SaaS media API.
- Replacing Home Assistant as the orchestration layer for voice devices.
- Implementing wake-word detection in the first version unless required by a concrete device integration.
- Building a multi-tenant untrusted upload platform.
- Supporting every possible speech or media model in v1.
- Providing arbitrary shell execution or a generic job runner.

## Primary Use Cases

### 1. Archive podcast/video transcription

1. Archive captures an item and publishes it immediately.
2. Archive creates a transcription job with media metadata.
3. Voxhelm accepts the request through its shared API and queues asynchronous work when needed.
4. Django Tasks workers on `studio` download media or fetch a referenced local object.
5. If the input is video, audio is extracted first.
6. The workers transcribe audio using a configured backend.
7. Transcript text and optional time-aligned outputs are returned to Archive.
8. Archive updates summaries/tags asynchronously using the transcript.

### 2. Archive article-to-audio generation

1. Archive identifies an article/text item eligible for TTS.
2. Archive creates a synthesis job with text content and voice settings.
3. Voxhelm queues synthesis work when needed.
4. Django Tasks workers on `studio` generate speech audio and store the output.
5. Archive attaches the result as a stable enclosure for a spoken feed.

### 3. Home Assistant voice interaction

1. A user speaks to a Home Assistant voice device, phone, or future satellite.
2. Home Assistant captures audio and handles pipeline orchestration.
3. Home Assistant sends STT requests to the `studio` service.
4. Home Assistant sends TTS requests to the `studio` service for spoken responses.
5. The response audio is played back on the requesting device.

### 4. python-podcast transcript generation

1. python-podcast identifies an episode that needs a transcript.
2. python-podcast submits a transcription job or synchronous request to the media service.
3. The media service fetches episode audio, transcribes it, and stores transcript artifacts.
4. python-podcast updates django-cast-backed transcript content from the returned result.

### 5. podcast-pipeline transcript production

1. podcast-pipeline starts a transcript generation step for an episode workspace.
2. Instead of invoking a local one-off transcriber directly, it can submit a job to the media service.
3. The media service returns transcript artifacts in the expected formats.
4. podcast-pipeline continues with chunking, summarization, and review workflows.

`summarize.sh` remains a downstream summarization/tagging tool, not the media service control plane.

### 6. OpenClaw voice/media integration

1. OpenClaw receives a voice or media-related request.
2. OpenClaw uses a plugin/tool/handler to call the media service.
3. The media service returns transcript text, generated speech audio, or extracted audio.
4. OpenClaw uses the result without needing its own dedicated speech engine stack.

## Product Principles

1. Narrow interfaces, not remote execution.
   Clients submit declarative requests. They do not pass shell commands.
2. Backend flexibility matters.
   The service must avoid coupling callers to a single STT/TTS implementation.
3. Interactive and batch workloads are different.
   Low-latency voice paths must not be blocked by long podcast jobs.
4. Home Assistant should remain the device-facing voice orchestrator.
   The media service should provide speech capabilities, not replace HA pipelines.
5. Security boundaries should be explicit.
   Separate producer auth, worker auth, and operator/admin controls.
6. Operational simplicity matters.
   The service should be deployable with current ops patterns and understandable to operate.

## Users and Systems

Primary operator:

- Jochen

Primary systems:

- Archive at `/Users/jochen/projects/archive`
- python-podcast at `/Users/jochen/projects/python-podcast`
- django-cast at `/Users/jochen/projects/django-cast`
- podcast-pipeline at `/Users/jochen/projects/podcast-pipeline`
- podcast-transcript at `/Users/jochen/projects/podcast-transcript`
- OpsGate at `/Users/jochen/projects/opsgate`
- ops-control at `/Users/jochen/projects/ops-control`
- ops-library at `/Users/jochen/projects/ops-library`
- Home Assistant installation
- OpenClaw installation
- `studio` as media execution host

Secondary systems:

- MinIO object storage from the start
- light health/availability monitoring

### Deployment assumptions

- most relevant services live under `~/projects`
- deployment should follow the existing `~/projects/ops-control` and `~/projects/ops-library` patterns
- the service should fit the existing homelab deployment style rather than introduce a separate deployment stack

## High-Level Architecture

The media service should have one logical core and multiple adapters.

### Core responsibilities

- validate requests
- schedule work
- isolate interactive and batch queues
- fetch/download inputs when allowed
- extract audio from video
- normalize/transcode media
- invoke configured STT or TTS backend
- persist artifacts and metadata
- return synchronous results or publish asynchronous job completion

### External interfaces

#### 1. Wyoming interface

Purpose:

- Home Assistant local STT/TTS integration

Requirements:

- expose STT and TTS over Wyoming-compatible services
- support low-latency interactive requests
- allow independent configuration of STT and TTS backends

#### 2. HTTP API

Purpose:

- Archive
- OpenClaw plugins
- operator tools
- future lightweight clients

Requirements:

- narrow, documented endpoints only
- synchronous endpoints for small/interactive requests
- asynchronous job endpoints for long-running work

#### 3. Async execution runtime

Purpose:

- secure execution on `studio` for expensive batch jobs

Requirements:

- job submission by producers
- execution by local worker processes
- producer-visible job status and result retrieval
- no generic remote shell execution

## Functional Requirements

### 1. Job types

The service should support at least these job types:

- `transcribe`
- `synthesize`
- `extract_audio`
- `analyze_media` optional later
- `diarize` optional later

`transcribe` may internally perform `extract_audio` first when the input is video.

### 2. Input types

The service should accept:

- direct media URL
- uploaded audio file
- uploaded video file
- MinIO object reference
- plain text input for TTS

Later:

- HTML/article source reference for integrated text extraction

### 3. Output types

The service should support:

- plain transcript text
- structured transcript JSON
- subtitle/time-aligned formats when available
- optional speaker-labeled transcript output later
- extracted audio file
- synthesized speech audio
- metadata about engine, model, language, duration, and processing time

### 4. STT backend abstraction

The service must support selecting a backend independently of callers.

Required initial backend identifiers:

- `auto`
- `whisperkit`
- `mlx`
- `whispercpp`

Selection should also support a model identifier or model preset.

Examples:

- backend `auto`, model `auto`
- backend `whisperkit`, model `large-v3`
- backend `mlx`, model `mlx-community/whisper-large-v3-mlx`
- backend `whispercpp`, model `ggml-large-v3.bin`

### 5. TTS backend abstraction

The service must support selecting a TTS backend independently of callers.

Required initial backend identifiers:

- `auto`
- `piper`

Optional later:

- `kokoro`
- other local engines if operationally justified

### 6. Interactive vs batch workload separation

The service must maintain distinct execution lanes:

- `interactive`
  - low-latency
  - Home Assistant STT/TTS
  - future OpenClaw voice turn requests
- `batch`
  - podcast transcription
  - long video processing
  - article TTS generation
  - python-podcast transcript generation
  - podcast-pipeline transcript generation

The scheduler must prevent long batch work from starving interactive voice traffic.

### 7. Audio extraction and preprocessing

The service must support:

- extracting audio from video
- resampling/normalization
- format conversion for backend compatibility
- configurable retention of extracted intermediate files

### 8. Result delivery

For synchronous/interactive requests:

- return the result directly

For batch jobs:

- store artifacts locally or in object storage
- expose job status APIs
- support producer polling in v1, with callbacks/webhooks deferred

### 9. Observability

The service should record:

- health/availability
- basic queue depth
- active jobs
- worker liveness

Detailed per-stage monitoring is explicitly not a v1 requirement.

### 10. Administrative controls

The service should support:

- enabling/disabling backends
- configuring default backend/model by workload class
- concurrency limits
- per-job retention policies
- input size/duration limits
- maintenance/drain mode

## API and Protocol Requirements

### 1. Batch job submission API

Producers should be able to submit a job with:

- producer identity
- job type
- requested outputs
- input descriptor
- backend preference
- model preference
- language hint
- metadata/context
- idempotency key / task reference

Example conceptual schema:

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
    "url": "https://example.com/audio.mp3"
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

### 2. Synchronous HTTP API

The service should expose narrow endpoints such as:

- `POST /v1/audio/transcriptions`
- `POST /v1/audio/speech`
- `POST /v1/media/extract-audio`
- `POST /v1/jobs`

These endpoints should be intended for:

- low-latency interactive use
- bounded-size inputs
- trusted internal clients only

### 3. Wyoming support

The service should expose Wyoming-compatible STT and TTS endpoints or companion processes so Home Assistant can consume them directly.

### 4. Artifact storage interface

The service should support:

- MinIO/S3-style object storage in v1
- local filesystem scratch/cache storage on `studio`

Artifact references should be stable and machine-readable.

## Security Requirements

### 1. No arbitrary remote execution

Archive, Home Assistant, OpenClaw, and other producers must not be able to request arbitrary commands or scripts to run on `studio`.

### 2. Separate identities

The system should use separate credentials for:

- producer submission
- operator/admin access

### 3. Narrow job schema

Jobs must be declarative and validated.

Allowed fields:

- backend id
- model id
- language hint
- input descriptor
- output descriptor
- metadata/context

Disallowed:

- shell snippets
- arbitrary executable paths
- free-form command arguments outside validated backend config

### 4. Trusted-network assumptions

Initial deployment may assume Tailscale and local/private network access only.

Even so:

- bind private endpoints conservatively
- require authentication
- validate input sizes and content types
- rate-limit interactive endpoints where appropriate

### 5. Input validation and limits

The service must validate:

- MIME type or file suffix when relevant
- max upload size
- max media duration if enforced
- max text size for TTS
- allowed URL schemes and hosts if a fetch policy is introduced

### 6. Sandboxed execution model

Backends should run under a dedicated service user on `studio`.

The service should avoid unnecessary broad privileges and should not require root for routine processing.

### 7. Auditability

The system should retain:

- job submission metadata
- task execution history
- result/failure state
- relevant timestamps
- backend/model used

## Reliability Requirements

### 1. Async safety

Producer systems must remain functional if the media service is unavailable.

For Archive specifically:

- publication must remain immediate
- transcription/TTS must remain asynchronous and optional

### 2. Retry behavior

The service should support bounded retries for:

- transient download failures
- transient backend invocation failures
- storage write failures

### 3. Restart recovery

Task execution should recover safely from restarts:

- queued or running jobs should be re-runnable or resumable according to backend/task semantics
- partial artifacts should be detectable and cleaned up or resumed

### 4. Idempotency

Producers should be able to submit the same logical job without accidental duplication when an idempotency key or task reference is present.

## Deployment and Operations

### 1. Host placement

Primary execution host:

- `studio`

Reasoning:

- Apple Silicon performance
- enough RAM for local model execution
- suitable for both STT and TTS workloads

### 2. Deployment model

Preferred:

- service deployed through existing `ops-control` / `ops-library` patterns

Possible runtime forms:

- one Python service with helper processes
- one service plus dedicated worker process
- companion backend daemons for Wyoming or model-serving integration

### 3. Data locations

The service will need:

- state database
- work queue state
- local cache for downloads and normalized media
- MinIO-backed artifact/output storage
- logs

### 4. Monitoring

The service should expose a simple health endpoint and minimal queue/worker visibility. Rich monitoring is not required in v1.

## Integration Requirements by Consumer

### Archive

Required in v1:

- OpenAI-compatible synchronous STT endpoint with file upload and URL input
- Archive-first transcription workflow
- video-to-audio extraction as part of transcription flow
- later batch transcription support for longer-running jobs
- later batch TTS jobs for article audio

Nice to have:

- richer async integration once Archive is ready to use the batch API

### python-podcast / django-cast

Deferred beyond the Archive-first v1 slice:

- API-driven transcript generation for podcast episodes
- artifact formats compatible with current transcript publishing needs
- ability to choose backend/model without changing python-podcast application logic

Nice to have:

- transcript update callbacks or polling-friendly job status
- optional speaker diarization later for multi-speaker episodes

### podcast-pipeline

Deferred beyond the Archive-first v1 slice:

- API-driven transcript generation for episode workspaces
- transcript outputs compatible with existing pipeline expectations
- ability to keep current downstream chunking/summarization workflows intact

Nice to have:

- a compatibility mode that mimics current `podcast-transcript` output layouts closely enough to reduce migration work
- optional chapter/timestamp support when the backend can provide it

### Home Assistant

Deferred to Milestone 2:

- local/private STT and TTS for Assist pipelines
- support for Home Assistant voice devices and mobile clients via HA pipeline orchestration

Preferred integration:

- Wyoming

### OpenClaw

Initial target:

- no deep coupling required in v1
- provide a stable HTTP API that an OpenClaw plugin/tool can call later

Longer term:

- OpenClaw may use the service for voice turn STT/TTS or other media workflows

## Milestones

### Milestone 1: Core batch transcription service

- service skeleton
- producer-facing sync and batch submission APIs
- Django Tasks worker execution on `studio`
- MinIO-backed artifacts from the start
- `studio` execution host
- backend abstraction for:
  - `mlx-whisper`
- video audio extraction via preprocessing
- Archive-compatible synchronous transcription path with file upload and URL input
- canonical Whisper JSON plus server-side transcript format conversion
- polling-based batch path for longer-running follow-up work

Success criteria:

- Archive can submit synchronous transcription requests safely
- `studio` executes jobs without remote shell access from Archive
- transcripts are returned reliably for normal podcast/video workloads

### Milestone 2: Home Assistant voice support

- Wyoming STT
- Wyoming TTS
- interactive lane scheduling
- low-latency tuning and limits
- deployment/runbook for Home Assistant integration

Success criteria:

- Home Assistant can use the `studio` service for local STT/TTS
- the Nabu Casa voice device and HA mobile voice use cases are supported through HA pipelines

### Milestone 3: TTS batch generation

- synthesis job type
- initial local TTS backend, preferably Piper
- artifact generation for Archive audio outputs
- configurable voice/model presets

Success criteria:

- Archive can generate audio for article/text items asynchronously
- generated audio is usable as a feed enclosure

### Milestone 4: OpenClaw integration

- stable HTTP contract for plugins/tools
- example OpenClaw integration path
- optional interactive low-latency path for voice turns

Success criteria:

- OpenClaw can consume the service without embedding its own speech engine stack

## Resolved Since Draft

The original draft left several implementation questions open. They are now resolved in the active planning docs:

- architecture, topology, interface contracts, auth boundaries, and artifact access model:
  `specs/interface-map.md`
- accepted and deferred decisions:
  `specs/decision-log.md`
- milestone scope and sequencing:
  `specs/milestones.md`
- execution order and deployment/verification flow:
  `specs/implementation-sequence.md`

At the time of this PRD, the resolved direction is:

- one Django-based service on `studio`
- Django Tasks for asynchronous execution
- shared HTTP APIs for producers
- Wyoming sidecars/adapters for Home Assistant integration
- `mlx-whisper` as the initial STT backend
- file upload and URL input support
- MinIO-backed async artifact storage
- wake word, diarization, and OpenClaw integration deferred beyond the initial Archive-first path

This PRD remains the product-intent document. If the active planning docs intentionally narrow or refine the initial draft, the active docs win.

## Existing OSS Landscape

The current open-source landscape suggests that reuse is possible, but no single project appears to cover the whole target shape out of the box.

### 1. Speaches

Closest fit as an existing general-purpose speech server.

Strengths:

- OpenAI-compatible API
- STT and TTS in one server
- supports faster-whisper for STT
- supports Piper and Kokoro for TTS
- supports streaming and realtime APIs

Limitations for this project:

- not Wyoming-native
- not specifically shaped around Voxhelm's planned async batch execution model
- no obvious built-in Archive/python-podcast/OpenClaw integration model

Conclusion:

- strong candidate for evaluation as a backend or reference implementation
- not an obvious complete drop-in for the full service as envisioned here

### 2. Home Assistant / Wyoming ecosystem

Strongest fit for Home Assistant-facing voice integration.

Relevant components:

- Wyoming protocol
- `wyoming-faster-whisper`
- `wyoming-satellite`
- Piper

Strengths:

- directly aligned with Home Assistant local voice architecture
- proven path for local STT/TTS and satellites

Limitations for this project:

- focused on HA voice integration, not a generic media platform
- not designed as the batch job system Archive and python-podcast need
- `wyoming-whisper-cpp` is archived and should not be treated as a strategic core dependency

Conclusion:

- use Wyoming as an integration adapter for Home Assistant
- do not treat the Wyoming stack alone as the whole service

### 3. WhisperKit

Strongest Apple-Silicon-native STT backend candidate.

Strengths:

- optimized for Apple Silicon
- local server implementing OpenAI Audio API

Limitations:

- STT-focused, not a full shared media platform
- does not solve TTS, object storage, queueing, or multi-consumer orchestration

Conclusion:

- excellent STT backend candidate
- likely part of the service rather than the whole service

### 4. whisper.cpp

Strong portable STT backend and reference server.

Strengths:

- mature, widely used, portable
- server example with OAI-like API
- strong Apple Silicon support

Limitations:

- STT-focused only
- does not provide the wider service shape by itself

Conclusion:

- good backend option
- good fallback/reference implementation

### 5. LocalAI

Broad OpenAI-compatible local inference platform.

Strengths:

- supports audio transcription and text-to-speech
- supports realtime API and composed voice pipelines
- large ecosystem and broad model support

Limitations:

- broader and heavier than the service needed here
- not Home Assistant Wyoming-native
- would still need product-specific orchestration around jobs, MinIO artifacts, and multi-consumer integration

Conclusion:

- plausible evaluation candidate if a general local AI gateway is desired
- likely more platform than necessary for a focused media service

### 6. pyannote.audio

Most relevant open-source diarization toolkit.

Strengths:

- strong diarization toolkit and pretrained pipelines
- local open-source path exists for diarization

Limitations:

- separate integration complexity
- not itself a full transcription service

Conclusion:

- best candidate for optional later diarization support

### Overall conclusion

There are good reusable building blocks, but there does not appear to be a single mature project that cleanly combines:

- Home Assistant Wyoming compatibility
- OpenAI-compatible HTTP APIs
- async batch execution
- MinIO artifact management
- Archive integration
- python-podcast / django-cast integration
- podcast-pipeline integration
- future OpenClaw integration

The most realistic path is therefore:

- build a thin project-specific control plane
- reuse existing engines and protocols underneath it
