# Review Prompt: Voxhelm Planning Package

You are reviewing a planning package produced by an agent team for the Voxhelm project. Your job is to evaluate the quality, consistency, and usefulness of the planning documents — not to rewrite them.

## Context

The primary PRD is at `/Users/jochen/projects/voxhelm/specs/2026-03-11_voxhelm_service.md`.

The coordination spec that defined the review process is at `/Users/jochen/workspaces/ws-voxhelm/specs/2026-03-11_voxhelm_spec_review_and_chunking.md`.

The planning package to review consists of these 6 documents in `/Users/jochen/projects/voxhelm/specs/`:

1. `spec-review.md`
2. `decision-log.md`
3. `delivery-chunks.md`
4. `milestones.md`
5. `implementation-sequence.md`
6. `interface-map.md`

## Reference Repos

These are the real consumer systems. Use them to validate claims made in the planning documents:

- `/Users/jochen/projects/archive` — self-hosted capture/enrichment service
- `/Users/jochen/projects/python-podcast` — Django/Wagtail podcast platform
- `/Users/jochen/projects/django-cast` — podcast Django package used by python-podcast
- `/Users/jochen/projects/podcast-pipeline` — multi-stage podcast production pipeline
- `/Users/jochen/projects/podcast-transcript` — CLI transcription tool with pluggable backends
- `/Users/jochen/projects/opsgate` — pull-worker execution system (architectural template)

## What To Evaluate

### 1. Accuracy

- Do the documents correctly describe how each consumer system works today?
- Are the claimed integration paths (e.g., "Archive switches with env vars only") actually true when you read the consumer code?
- Are interface descriptions (endpoints, field names, formats) correct?

### 2. Completeness

- Did the planning package address all 10 explicit questions from the coordination spec?
- Did it address all 8 open questions from the PRD?
- Are there important concerns or integration gaps that were missed?
- Does every delivery chunk have all required fields (ID, title, purpose, scope, exclusions, dependencies, consumers, interfaces, acceptance criteria, risks, order)?

### 3. Internal Consistency

- Do the milestones in `milestones.md` match the chunks in `delivery-chunks.md`?
- Does `implementation-sequence.md` reference the same chunks and milestones?
- Do the decisions in `decision-log.md` align with the recommendations in the other documents?
- Are there places where two documents contradict each other?

### 4. Quality Bar (from coordination spec)

The coordination spec defined this quality bar. Evaluate against it:

- Do the documents reduce ambiguity rather than restating the PRD?
- Do they produce a realistic implementation sequence?
- Do they separate v1 from later ambitions clearly?
- Do they identify which areas are still unknown?
- Do they make the project easier to execute for a coding team?

### 5. Actionability

- Could a developer pick up a chunk from `delivery-chunks.md` and start implementing without significant further clarification?
- Are acceptance criteria testable?
- Are dependencies between chunks clear enough to avoid blocking?
- Are the spikes well-defined enough to actually conduct?

### 6. Risk and Blind Spots

- Are there architectural risks the documents don't mention?
- Are there consumer integration complexities that were oversimplified?
- Is the scope of M1a actually as small as claimed?
- Are there deployment or operational concerns that were overlooked?

## What To Produce

Write your review to `/Users/jochen/projects/voxhelm/specs/planning-review.md` with these sections:

1. **Overall Assessment** — 2-3 sentence verdict on the planning package quality
2. **Accuracy Issues** — specific factual errors found (with evidence from the repos)
3. **Consistency Issues** — contradictions between documents
4. **Completeness Gaps** — missing items or unanswered questions
5. **Strength Highlights** — what the planning package got right
6. **Recommended Fixes** — prioritized list of corrections or additions, with severity (blocking / important / minor)

Be specific. Cite file paths, line numbers, and actual code when pointing out accuracy issues. Don't rewrite the documents — just identify what needs fixing.
