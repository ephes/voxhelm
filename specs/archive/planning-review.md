# Planning Review: Voxhelm Planning Package

## Overall Assessment

The package is strong on consumer-aware framing, especially for Archive and podcast-transcript, and it does a good job breaking the work into implementation-sized chunks. It is not yet execution-ready, though, because several documents disagree on the core v1 integration path, the deployment substrate on macstudio, and the artifact/auth model; those contradictions are large enough to cause rework if coding starts from the current set.

## Accuracy Issues

### 1. Archive is still described as a batch-job consumer in parts of the package, but the current Archive code is synchronous only

- `interface-map.md:17`, `interface-map.md:225-228`, and `interface-map.md:315` describe Archive as a v1 batch-job consumer that submits jobs and polls for results.
- `decision-log.md:147` justifies polling by saying "Archive's enrichment worker already uses a poll loop."

That is not how Archive integrates today. Archive downloads media locally and makes one blocking multipart POST to `{ARCHIVE_TRANSCRIPTION_API_BASE}/audio/transcriptions`, then reads `response.json()["text"]`; there is no external job submission or external job-status polling in the Archive transcription path today.

Evidence:
- `/Users/jochen/projects/archive/src/archive/transcriptions.py:58-70`
- `/Users/jochen/projects/archive/src/archive/transcriptions.py:166-214`
- `/Users/jochen/projects/archive/src/archive/services.py:264-317`

Impact: this overstates v1 scope and mis-prioritizes the interface inventory. The package's own stronger analysis in `spec-review.md:70-84` is the version that matches the code.

### 2. podcast-pipeline is described as a direct Voxhelm batch consumer, but it currently shells out to podcast-transcript

- `interface-map.md:17`, `interface-map.md:228`, and `interface-map.md:318` describe podcast-pipeline as a direct batch-job consumer.
- `interface-map.md:228` also says it expects `text` and optional `chapters` from the job API.

The current code does not call any transcription API directly. `podcast-pipeline` runs an external transcriber command, then checks for workspace files like `transcript.txt` and `chapters.txt`.

Evidence:
- `/Users/jochen/projects/podcast-pipeline/src/podcast_pipeline/entrypoints/transcribe.py:53-79`
- `/Users/jochen/projects/podcast-pipeline/src/podcast_pipeline/entrypoints/transcribe.py:146-171`

Impact: the interface map currently models the wrong integration boundary. The actual boundary is `podcast-transcript`, not podcast-pipeline.

### 3. The deployment plan assumes systemd on macstudio, but the current macstudio ops patterns are launchd-based

- `milestones.md:73-82` and `implementation-sequence.md:98` describe systemd units on macstudio.
- `delivery-chunks.md:623-627` defines control-plane and worker systemd services for the macstudio deployment role.

The current ops repos already treat macstudio as a macOS/launchd target:

Evidence:
- `/Users/jochen/projects/ops-control/inventories/prod/host_vars/macstudio.yml:13-17`
- `/Users/jochen/projects/ops-library/README.md:114`

Impact: C11 and the M1a deployment notes are not just underspecified; they point at the wrong service manager, which will mis-shape the first deployment role.

### 4. "Zero-code-change Archive video switchover" is overstated given Archive's 25 MiB pre-upload limit

- `implementation-sequence.md:119-123` makes video verification part of the M1a Archive switchover.
- `delivery-chunks.md:508-516` expects Archive video items to work through the sync endpoint with no Archive code changes.

Archive downloads the media itself and rejects payloads over 25 MiB before any request reaches Voxhelm.

Evidence:
- `/Users/jochen/projects/archive/src/archive/transcriptions.py:98-124`

Impact: the package is correct that a sync OpenAI-compatible endpoint is the easiest Archive path, but it over-claims what that path can cover for larger podcast/video inputs without either Archive changes or a URL-based/batch path.

## Consistency Issues

### 1. M1a defers video handling, but M1a/C8 acceptance still requires video success

- `milestones.md:91` defers "Video input / audio extraction" to M1b.
- `implementation-sequence.md:119-123` makes video verification part of Phase 1a.
- `delivery-chunks.md:508-509` says Archive video transcription succeeds through C7/C8.
- `interface-map.md:76` says the synchronous STT API must extract audio from video.

These cannot all be true at once.

### 2. The package disagrees on whether artifacts are proxied by Voxhelm or exposed via MinIO presigned URLs

- `spec-review.md:145-147` recommends proxying artifact delivery through Voxhelm HTTP endpoints so consumers never access MinIO directly.
- `delivery-chunks.md:281`, `delivery-chunks.md:299-300`, `interface-map.md:19`, `interface-map.md:266`, `interface-map.md:279`, and `interface-map.md:422` all assume presigned-URL-based consumer access.

This is a real API and security-boundary decision, not a wording difference.

### 3. The package disagrees on where transcript format conversion belongs

- `decision-log.md:110` recommends Whisper-native JSON as canonical output and says existing client-side conversion code should handle DOTe/Podlove/WebVTT.
- `delivery-chunks.md:336`, `delivery-chunks.md:385-386`, and `interface-map.md:125-149` push DOTe/Podlove/WebVTT generation into Voxhelm itself.

That changes both service scope and consumer responsibilities.

### 4. The token model is inconsistent

- `milestones.md:80` says M1a ships with a "single producer token."
- `spec-review.md:129` recommends per-consumer tokens.
- `interface-map.md:332-346` defines per-producer tokens.

This should be one decision, not three variants.

### 5. M2 is described as independent of M1b/M1c, but its TTS half depends on later work

- `milestones.md:231` says M2 "Does NOT depend on M1b or M1c."
- `delivery-chunks.md:675-676` and `delivery-chunks.md:686-707` make the Wyoming TTS path depend on C15, which is an M3 chunk.
- `implementation-sequence.md:260-269` also brings Piper/TTS work into Phase 2.

The STT half may be parallelizable after M1a, but the current "M2" definition is not actually independent as written.

## Completeness Gaps

### 1. The required input repos `ops-control` and `ops-library` are barely validated, even though the package makes concrete deployment claims about them

The coordination spec listed `/Users/jochen/projects/ops-control` and `/Users/jochen/projects/ops-library` as required inputs, but the package mostly treats deployment as a generic future chunk. That is what allowed the systemd/launchd mistake to survive into `milestones.md`, `implementation-sequence.md`, and `delivery-chunks.md`.

Evidence:
- `/Users/jochen/workspaces/ws-voxhelm/specs/2026-03-11_voxhelm_spec_review_and_chunking.md:45-54`

### 2. Home Assistant and OpenClaw are not validated consumer-by-consumer with the same rigor as Archive and podcast-transcript

`spec-review.md` gives source-backed detail for Archive, django-cast, podcast-transcript, and podcast-pipeline, but Home Assistant and OpenClaw are mostly treated as architectural placeholders with little evidence or explicit "required v1 / optional / defer" structure.

Evidence:
- `/Users/jochen/workspaces/ws-voxhelm/specs/2026-03-11_voxhelm_spec_review_and_chunking.md:187-202`
- `/Users/jochen/projects/voxhelm/specs/spec-review.md:106-114`

The current treatment is probably acceptable for OpenClaw because it is intentionally deferred, but it is weaker than the review method required by the coordination spec.

## Strength Highlights

- `spec-review.md` is the strongest document in the package. The Archive and podcast-transcript analysis is grounded in real code and materially reduces ambiguity for v1.
- The package clearly separates an Archive-first path from later consumer work, which is a better execution shape than the original PRD.
- The chunk documents are detailed enough to be useful working material once the cross-document contradictions above are resolved.
- The spike list is practical and tied to real implementation decisions instead of vague "research more" placeholders.

## Recommended Fixes

### Blocking

1. Normalize the Archive story across all documents. Archive should be treated as a synchronous OpenAI-compatible consumer for the first release, not as a primary batch-job client. Update `interface-map.md`, `decision-log.md`, and the producer matrix to match the actual Archive code.
2. Fix the deployment substrate in `milestones.md`, `delivery-chunks.md`, and `implementation-sequence.md` to reflect macstudio's launchd/macOS reality, then re-check C11 against `ops-library`/`ops-control`.
3. Reconcile the M1a/M1b boundary for video handling. Either move video support into M1a everywhere, or stop using video success as an M1a/C8 acceptance gate.

### Important

4. Pick one artifact-delivery model: presigned MinIO URLs or Voxhelm-proxied artifact delivery. Then update `spec-review.md`, `delivery-chunks.md`, and `interface-map.md` so auth and API shape line up.
5. Pick one transcript-conversion model: server-side conversion in Voxhelm or client-side conversion in consumers. The current package assigns the work to both sides.
6. Split M2 into "Wyoming STT" and "Wyoming TTS" if needed, or explicitly move the TTS half behind C15/M3 dependencies. The current milestone wording overstates parallelism.

### Minor

7. Standardize the auth language to per-consumer producer tokens throughout.
8. Stop labeling the synchronous transcription endpoint as an "interactive" lane in the interface map; for Archive it is a synchronous batch convenience endpoint, not the HA voice path.
9. Tighten podcast-pipeline wording so it is consistently described as an indirect consumer through podcast-transcript, not a direct API client.
