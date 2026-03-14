# Voxhelm Specs Map

Current implementation snapshot as of 2026-03-14:

- M1a, M1b, and the current M1c consumer slices are implemented and deployed.
- M2 Home Assistant voice wiring is implemented: Voxhelm now runs a Wyoming sidecar on `studio`, Home Assistant can use Voxhelm STT/TTS through Assist pipelines, and area-registry aliases can be managed from deploy config.
- The core M3 service/runtime slice is implemented: Piper-backed TTS, `POST /v1/audio/speech`, and batch `synthesize` jobs are live in Voxhelm.
- The post-M3 operator transcript follow-on is also implemented: Voxhelm now ships the session-authenticated operator UI at `/`, mixed sync/batch operator routing, and server-owned `dote` / `podlove` transcript artifacts.
- The live production shape is a Django HTTP process plus a Django Tasks worker and a Wyoming STT/TTS sidecar on `studio`, with private HTTPS ingress on `macmini` at `https://voxhelm.home.xn--wersdrfer-47a.de`.
- Implemented endpoints: `GET /v1/health`, `POST /v1/audio/transcriptions`, `POST /v1/audio/speech`, `POST /v1/jobs`, `GET /v1/jobs/{id}`, and `GET /v1/jobs/{id}/artifacts/{name}`.
- Implemented sync STT contract: bearer auth, multipart upload, JSON URL mode, accepted models `gpt-4o-mini-transcribe` and `whisper-1`, plus the explicit `whisperkit` opt-in path when that backend is enabled, and response formats `json`, `text`, `verbose_json`, and `vtt`.
- Implemented TTS contract: Piper-backed synchronous speech generation plus batch `synthesize` jobs with artifact storage.
- Implemented batch contract: persisted jobs and artifacts, Django Tasks internal execution, idempotent `task_ref` handling, video-to-audio extraction, artifact download through the Voxhelm HTTP proxy, and canonical batch transcript artifacts for `json`, `text`, `vtt`, `dote`, and `podlove`.
- The same-epic `django-cast` consumer cleanup is now complete: it requests and persists Voxhelm-owned `podlove`, `dote`, and `vtt` artifacts directly instead of converting `dote` / `podlove` locally.
- Production artifact storage is MinIO-backed via the S3-compatible `VOXHELM_ARTIFACT_*` env vars, using bucket `voxhelm`.
- Archive-compatible sync transcription, live batch jobs, direct Home Assistant STT, and the restored production debug-logging default have all been validated against the deployed service.
- Remaining planned work is narrower now: operational work beyond the current `whisper.cpp` + `mlx-whisper` STT set plus the experimental non-default WhisperKit path, Archive article-audio consumer follow-on, and M4/OpenClaw.
- The STT benchmark spike was re-run on `studio`, and the current source of truth is [`2026-03-13_whisperkit_re_evaluation_studio.md`](./2026-03-13_whisperkit_re_evaluation_studio.md). The revised evidence keeps `whisper.cpp` as the deployed default for now, but WhisperKit is no longer merely provisional: on the tuned `studio` path it is now a real follow-on candidate, with GPU stability caveats.

This directory currently contains both the original PRD and the planning package derived from it. The goal of this file is to make the document stack explicit, so readers know:

- which document answers which class of question
- which document is the source of truth when two documents overlap
- what order to read the package from high level to low level
- where the current package should be tightened to reduce duplication

## Recommended Reading Order

Read the documents in this order:

1. [`2026-03-11_voxhelm_service.md`](./2026-03-11_voxhelm_service.md)
   The product definition: problem, goals, users, use cases, non-goals, milestone intent, and open questions.
2. [`interface-map.md`](./interface-map.md)
   The v1 technical shape: service boundaries, auth boundaries, interface contracts, producer/consumer relationships, and storage/access model.
3. [`decision-log.md`](./decision-log.md)
   The decision register: open or recently resolved choices, available options, recommended default, and blocker status.
4. [`milestones.md`](./milestones.md)
   The delivery slices: what ships in each milestone, what is deferred, milestone dependencies, and milestone success criteria.
5. [`delivery-chunks.md`](./delivery-chunks.md)
   The implementation work packages: chunk scope, exclusions, dependencies, interfaces, and acceptance criteria.
6. [`implementation-sequence.md`](./implementation-sequence.md)
   The execution plan: ordering, parallelism, spikes, gates, and implementation timing.
7. [`archive/spec-review.md`](./archive/spec-review.md)
   The research review against the original PRD and consumer repos.
8. [`archive/planning-review.md`](./archive/planning-review.md)
   The review of the planning package itself.

## Source Of Truth Hierarchy

When documents overlap, use this order of authority:

1. `interface-map.md`
   Source of truth for interface shape, auth boundaries, artifact access model, and producer/consumer mapping.
2. `decision-log.md`
   Source of truth for decisions that are still open, recently resolved, or explicitly conditional.
3. `milestones.md`
   Source of truth for what belongs in each milestone.
4. `delivery-chunks.md`
   Source of truth for chunk boundaries, dependencies, and implementation-ready scope.
5. `implementation-sequence.md`
   Source of truth for ordering, parallelism, and pre-implementation spikes.
6. `archive/spec-review.md`
   Source of truth for evidence-backed review findings against the original PRD and consumer repos.
7. `2026-03-11_voxhelm_service.md`
   Historical product source. If the planning package intentionally narrows or corrects the PRD, the planning package wins.

Rule: if a lower document wants to contradict a higher one, the contradiction should first be resolved in the higher document.

## What Belongs Where

### 1. PRD

File:
- [`2026-03-11_voxhelm_service.md`](./2026-03-11_voxhelm_service.md)

Keep here:
- problem statement
- goals and non-goals
- user and consumer overview
- use cases
- milestone intent
- open questions
- high-level product direction

Do not keep here long-term:
- exact endpoint payloads
- exact auth token layout
- exact job schema details
- implementation ordering

### 2. Architecture / Interface Spec

Current file:
- [`interface-map.md`](./interface-map.md)

Keep here:
- v1 topology
- control-plane vs worker responsibilities
- native vs wrapper boundaries
- interface contracts
- auth domains
- artifact access model
- producer/consumer matrix

Move out over time:
- implementation priority
- repo-specific compatibility research and consumer-code evidence that belong in `archive/spec-review.md`
- milestone sequencing commentary

### 3. Decision Register

Current file:
- [`decision-log.md`](./decision-log.md)

Keep here:
- decisions with options
- recommended defaults
- blocker status
- unresolved questions

Rule:
- once a decision is stable, its outcome should also be reflected in `interface-map.md` or `milestones.md`, and the decision log should become the audit trail rather than the only place the decision exists.

### 4. Milestone Plan

Current file:
- [`milestones.md`](./milestones.md)

Keep here:
- milestone names
- what ships
- what is deferred
- milestone dependencies
- milestone success criteria

Do not keep here:
- chunk-level implementation mechanics
- duplicate API contract detail unless needed to explain milestone boundaries

### 5. Work Package Plan

Current file:
- [`delivery-chunks.md`](./delivery-chunks.md)

Keep here:
- chunk IDs
- chunk scope and exclusions
- chunk dependencies
- acceptance criteria
- risks
- implementation order within a milestone

Do not keep here:
- competing architectural alternatives that belong in `decision-log.md`
- milestone-level narrative that belongs in `milestones.md`

### 6. Execution Plan

Current file:
- [`implementation-sequence.md`](./implementation-sequence.md)

Keep here:
- what starts first
- what can run in parallel
- spikes and gates
- critical path
- deployment/verification order

Move out over time:
- consumer-code walkthroughs and repo findings already captured in `archive/spec-review.md`
- duplicated risk register content already captured in milestone/chunk docs

### 7. Review Artifacts

Files:
- [`archive/spec-review.md`](./archive/spec-review.md)
- [`archive/planning-review.md`](./archive/planning-review.md)
- [`archive/2026-03-11_voxhelm_spec_review_and_chunking.md`](./archive/2026-03-11_voxhelm_spec_review_and_chunking.md)
- [`archive/review-prompt.md`](./archive/review-prompt.md)

Keep here:
- evidence-backed critiques
- repo findings
- ambiguity and contradiction callouts

Rule:
- review docs should not become the canonical architecture spec. Once a review finding is accepted, normalize the resulting decision into the core docs.

Active top-level spec set:
- [`2026-03-11_voxhelm_service.md`](./2026-03-11_voxhelm_service.md)
- [`interface-map.md`](./interface-map.md)
- [`decision-log.md`](./decision-log.md)
- [`milestones.md`](./milestones.md)
- [`delivery-chunks.md`](./delivery-chunks.md)
- [`implementation-sequence.md`](./implementation-sequence.md)
- [`README.md`](./README.md)

## Current File Mapping

The current planning package maps reasonably well to a high-level-to-low-level stack:

- Product definition:
  [`2026-03-11_voxhelm_service.md`](./2026-03-11_voxhelm_service.md)
- Architecture and contracts:
  [`interface-map.md`](./interface-map.md)
- Decision register:
  [`decision-log.md`](./decision-log.md)
- Delivery slices:
  [`milestones.md`](./milestones.md)
- Work packages:
  [`delivery-chunks.md`](./delivery-chunks.md)
- Execution order:
  [`implementation-sequence.md`](./implementation-sequence.md)
- Review artifacts:
  [`archive/spec-review.md`](./archive/spec-review.md), [`archive/planning-review.md`](./archive/planning-review.md)

The active docs now map cleanly to a high-level-to-low-level chain. The main maintenance rule is to keep repo evidence and review reasoning in `archive/`, not in the active architecture and sequencing docs.

## Recommended Cleanup Pass

To make the stack cleaner without a full rewrite, do one focused cleanup pass in this order:

1. Trim `interface-map.md` so it is only architecture and contract truth.
   Move or delete:
   - implementation priority sections
   - compatibility notes that duplicate `spec-review.md`
   - milestone commentary that belongs in `milestones.md`

2. Trim `implementation-sequence.md` so it is only execution planning.
   Move or delete:
   - consumer code analysis already present in `spec-review.md`
   - duplicated risk discussion already captured elsewhere

3. Normalize milestone vs chunk ownership.
   - `milestones.md` should answer "what ships when"
   - `delivery-chunks.md` should answer "what work package implements that"

4. Keep `decision-log.md` short and active.
   - if a decision is fully settled and no longer controversial, reflect it in the architecture or milestone docs and keep only the decision record summary

## Target End State

At steady state, the stack should look like this:

1. PRD
   Why the project exists and what success looks like.
2. Architecture / Interface Spec
   What the v1 system is.
3. Decision Log
   What is still being chosen or was recently chosen.
4. Milestones
   What ships in each release slice.
5. Delivery Chunks
   What engineers implement as work packages.
6. Implementation Sequence
   In what order the work happens.
7. Reviews / ADRs / Validation Notes
   Evidence, critique, and historical reasoning.

## Practical Rule For Future Updates

Before editing, ask:

- "Am I changing product intent?"
  Edit the PRD.
- "Am I changing the technical contract or system boundary?"
  Edit `interface-map.md`.
- "Am I resolving or reopening a choice?"
  Edit `decision-log.md`.
- "Am I moving scope between releases?"
  Edit `milestones.md`.
- "Am I changing engineer work-package scope?"
  Edit `delivery-chunks.md`.
- "Am I changing execution order or spike sequencing?"
  Edit `implementation-sequence.md`.

If the answer is "more than one", update the highest-level affected document first, then propagate downward.
