# Spec: Multi-Agent Review And Chunking For Voxhelm

**Date:** 2026-03-11
**Status:** Draft
**Primary input:** `/Users/jochen/projects/voxhelm/specs/2026-03-11_voxhelm_service.md`

## Summary

Run a structured review of the Voxhelm PRD with a small team of agents and produce a revised planning package that:

- identifies ambiguities, contradictions, and missing decisions
- validates the architecture against the real consumer systems
- separates v1 from later work
- restructures the project into implementable chunks with clear dependencies
- produces a practical execution order for implementation

This is not an implementation spec. It is a coordination spec for reviewing and partitioning the product spec into an execution-ready plan.

## Goal

Turn the current PRD into a planning set that a coding team can execute with limited ambiguity.

The desired output is not one giant rewritten PRD. The desired output is:

- a reviewed PRD with issues called out
- a decision log for unresolved choices
- a chunked delivery plan
- a milestone breakdown
- implementation-ready work packages

## Non-goals

- writing production code
- resolving every architectural choice by implementation
- creating tickets in external systems
- benchmarking models
- deploying any services

## Inputs

Primary spec:

- `/Users/jochen/projects/voxhelm/specs/2026-03-11_voxhelm_service.md`

Reference repos and systems:

- `/Users/jochen/projects/archive`
- `/Users/jochen/projects/python-podcast`
- `/Users/jochen/projects/django-cast`
- `/Users/jochen/projects/podcast-pipeline`
- `/Users/jochen/projects/podcast-transcript`
- `/Users/jochen/projects/opsgate`
- `/Users/jochen/projects/ops-control`
- `/Users/jochen/projects/ops-library`

## Required Outputs

The agent team must produce these artifacts:

1. `spec-review.md`
   - findings against the current PRD
   - inconsistencies
   - missing decisions
   - places where the current spec is too broad or too vague

2. `decision-log.md`
   - explicit unresolved decisions
   - available options
   - tradeoffs
   - recommended default
   - whether the item blocks implementation

3. `delivery-chunks.md`
   - the project split into concrete, manageable chunks
   - each chunk small enough for one implementation stream
   - each chunk with scope, dependencies, acceptance criteria, and risks

4. `milestones.md`
   - ordered milestones
   - what ships in each milestone
   - what is deferred

5. `interface-map.md`
   - external interfaces to implement first
   - producers and consumers
   - protocols and auth boundaries

6. `implementation-sequence.md`
   - recommended execution order
   - parallelizable work
   - spikes required before real implementation

## Agent Team

Use a small team with explicit responsibilities.

### 1. Product Architect

Focus:

- overall scope shape
- v1 versus later scope
- milestone clarity
- consumer fit

Questions to answer:

- is the spec trying to do too much in v1?
- what is the smallest coherent first release?
- what should be delayed even if it is desirable?

### 2. Systems Architect

Focus:

- service boundaries
- worker/control-plane split
- protocol choices
- runtime topology
- Django/SQLite fit

Questions to answer:

- what belongs in the control plane versus workers?
- what must be synchronous versus asynchronous?
- where are the hidden complexity traps?

### 3. Security Reviewer

Focus:

- producer/worker trust boundaries
- remote execution risks
- MinIO access patterns
- Home Assistant/OpenClaw integration risks

Questions to answer:

- where can clients smuggle arbitrary work into the service?
- which interfaces must be private-only?
- what auth separation is mandatory in v1?

### 4. Integrations Reviewer

Focus:

- Archive integration
- python-podcast / django-cast integration
- podcast-pipeline integration
- Home Assistant integration
- OpenClaw integration

Questions to answer:

- what does each consumer actually need first?
- where can one consumer force needless complexity on all others?
- what common API surface is genuinely shared?

### 5. Delivery Planner

Focus:

- implementation chunks
- dependency order
- spike identification
- avoidance of oversized milestones

Questions to answer:

- how should this be partitioned into work packages?
- what can be developed independently?
- which decisions must be made before coding starts?

## Review Method

The agent team should work in this order.

### Phase 1: Read and annotate

All agents read the primary spec and identify:

- unclear language
- contradictions
- open questions hidden inside requirements
- scope that is too broad for a first implementation

### Phase 2: Consumer-by-consumer validation

Validate the spec against each real consumer separately:

- Archive
- python-podcast / django-cast
- podcast-pipeline
- Home Assistant
- OpenClaw

For each consumer, capture:

- required v1 functionality
- optional functionality
- wrong assumptions in the current PRD
- integration points that should be deferred

### Phase 3: Architecture normalization

Reduce the architecture to a clear v1 shape.

The agent team must explicitly decide whether the v1 system is:

- one Django control-plane service plus local workers
- one Django service plus sidecar adapters
- one thin control plane over existing backend servers

If the answer is mixed, the team must say exactly which parts are native and which parts are wrappers.

### Phase 4: Chunking

Break the project into chunks that satisfy all of:

- implementable in isolation
- testable in isolation
- low ambiguity
- clear dependency boundaries
- produces user-visible or architecture-visible progress

### Phase 5: Final planning package

Produce the required output documents in a consistent format.

## Review Criteria

Every agent must evaluate the spec against these criteria.

### 1. Scope discipline

- Is v1 still small enough to finish?
- Are interactive voice and batch media too tightly coupled?
- Is diarization wrongly included too early?

### 2. Architecture clarity

- Is it clear what the control plane owns?
- Is it clear what workers own?
- Is it clear what MinIO owns?
- Is it clear what stays in consumer apps?

### 3. Integration realism

- Does Archive need synchronous STT in v1 or only batch?
- Does python-podcast need API compatibility or artifact compatibility?
- Does podcast-pipeline need full migration or only an optional backend path?
- Does Home Assistant require only Wyoming in v1?
- Does OpenClaw belong in v1 at all, or only as a later consumer?

### 4. Security clarity

- Are all interfaces narrow and declarative?
- Are producer tokens separate from worker tokens?
- Are upload/fetch permissions explicit?
- Is there any accidental path to arbitrary code execution?

### 5. Delivery realism

- Can one team implement the first milestone without waiting on every later decision?
- Are the milestones vertically useful?
- Are there spikes identified where uncertainty is high?

## Required Chunking Rules

When creating `delivery-chunks.md`, follow these rules.

### Rule 1: Chunk by stable boundary, not by documentation section

Bad:

- "Implement Functional Requirements"
- "Implement Security"

Good:

- "Control-plane job model and auth"
- "Batch transcription worker"
- "MinIO artifact persistence"
- "Archive producer integration"
- "Wyoming adapter for Home Assistant"

### Rule 2: Keep v1 chunks narrow

If a chunk contains more than one major protocol or more than one major consumer integration, split it unless the coupling is unavoidable.

### Rule 3: Separate spikes from implementation chunks

Examples of spike-worthy work:

- backend benchmark comparison
- WhisperKit versus `mlx-whisper` default choice
- diarization feasibility
- Wyoming adapter implementation strategy

### Rule 4: Defer speculative integrations

If OpenClaw or diarization materially complicate v1, place them in later milestones unless there is a strong architectural reason to include them early.

### Rule 5: Prefer capability-first chunks

Chunks should represent a coherent system capability, such as:

- submit and store jobs
- claim and execute jobs
- persist artifacts to MinIO
- return transcripts

not:

- models
- serializers
- Django apps

## Expected Output Shape For Delivery Chunks

Each chunk in `delivery-chunks.md` must include:

- `Chunk ID`
- `Title`
- `Purpose`
- `Included scope`
- `Explicitly excluded scope`
- `Dependencies`
- `Consumer(s)`
- `Primary interfaces`
- `Acceptance criteria`
- `Main risks`
- `Suggested implementation order`

## Suggested Chunk Model

The team should at least test whether the project naturally decomposes into something like:

1. Control plane foundation
2. Worker claim/execution model
3. MinIO artifact model
4. STT backend adapter layer
5. Batch transcription v1
6. Archive integration
7. python-podcast / django-cast integration
8. podcast-pipeline integration
9. Wyoming adapter and Home Assistant integration
10. TTS support
11. OpenClaw integration
12. Diarization spike or later extension

This list is only a starting hypothesis. The team may restructure it if they can justify a better decomposition.

## Explicit Questions The Team Must Resolve

The outputs must explicitly answer these questions.

1. What is the smallest credible v1 for Voxhelm?
2. Is Home Assistant voice support part of v1, or should it be Milestone 2?
3. Is TTS part of v1, or should it follow after batch transcription?
4. Is OpenClaw integration part of v1, or should it remain only an architectural placeholder?
5. Is diarization a v1 feature, a spike, or a later milestone?
6. Should python-podcast and podcast-pipeline both target a common HTTP API, or should one get a compatibility shim first?
7. Is Django + SQLite good enough for the control plane, and what constraints must be stated explicitly?
8. Which interfaces should be implemented natively, and which should wrap existing tools or servers?
9. Which parts must use MinIO from day one?
10. What are the top three technical spikes that should happen before implementation starts?

## Quality Bar

The planning outputs are only acceptable if they:

- reduce ambiguity rather than restating the PRD
- produce a realistic implementation sequence
- separate v1 from later ambitions clearly
- identify which areas are still unknown
- make the project easier to execute for a coding team

If the agent team cannot answer a question conclusively, it must:

- say that explicitly
- give options
- recommend a default
- mark whether the issue blocks implementation

## Final Deliverable

The final deliverable from the agent team should be a review package under `specs/` that is sufficient for a follow-up implementation planning session without rereading the entire original PRD from scratch.
