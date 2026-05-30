# Voxhelm Backlog

## Remote transcription workers / `atlas.local`

Concept added on 2026-05-30: [`specs/remote-transcription-workers.md`](specs/remote-transcription-workers.md).
Option B is accepted for the next slice: `studio` remains the Voxhelm control plane, and `atlas.local` connects
outbound over an internal HTTP pull-worker API to claim leased batch transcription jobs, run local STT, upload
artifacts, and report results without changing producer-facing APIs. The implementation uses a remote-only
transcription queue and MinIO-backed artifact manifests from trusted workers. Goal completion requires a real
production python-podcast known-speaker diarized transcript to execute on `atlas.local`.
No implementation has started.

### Implementation backlog

- [x] Normalize the accepted Option B design into `specs/interface-map.md` and keep D-23 in sync.
- [ ] Add worker auth and liveness:
  - worker token map, e.g. `VOXHELM_WORKER_TOKENS="atlas=..."`;
  - `Worker` runtime/liveness model;
  - `POST /v1/internal/workers/heartbeat`.
- [ ] Add remote-pull job state:
  - `execution_mode`, `assigned_worker_id`, `lease_token_hash`, `lease_expires_at`, `attempt_count`,
    `max_attempts`, `last_worker_heartbeat_at`;
  - SQLite-safe conditional claim update with affected-row check;
  - attempt-scoped artifact prefixes such as `jobs/<job_id>/attempt-<n>/` to prevent zombie-worker S3 overwrites;
  - all lease decisions use `studio` server time.
- [ ] Add worker endpoints:
  - `POST /v1/internal/work/claim`;
  - `POST /v1/internal/work/{job_id}/heartbeat`;
  - `POST /v1/internal/work/{job_id}/complete` with same-token idempotent retry after successful completion;
  - `POST /v1/internal/work/{job_id}/fail`;
  - heartbeat/fail/complete all verify assigned worker plus lease token hash.
- [ ] Add `VOXHELM_TRANSCRIPTION_EXECUTION_MODE=django_tasks|remote_pull`; in `remote_pull` mode, transcribe jobs
  are not enqueued to Django Tasks.
- [ ] Add deployment/edge protection: block `/v1/internal/*` on public Traefik/macmini routes unless explicitly exposed on a private worker route.
- [ ] Add easy worker packaging/onboarding:
  - expose a worker command such as `voxhelm-remote-worker`;
  - support checkout-based `uv run` plus install/run paths with `uv tool install` or `uvx`;
  - require only Voxhelm base URL, worker id/token, artifact credentials, model/cache settings, and optional HF token;
  - document a repeatable new-machine setup with protected env file and launchd.
- [ ] Add the `atlas.local` worker command/process:
  - claim one job at a time;
  - materialize URL or staged-upload input;
  - store source/extracted artifacts in MinIO under attempt-scoped prefixes;
  - run local STT;
  - render requested transcript formats;
  - upload artifacts and post completion manifest.
- [ ] Add remote known-speaker diarization support:
  - choose Atlas-runs-known-speaker or hybrid classification and document the choice;
  - support `pyannote_known_speaker` payloads from django-cast/python-podcast;
  - treat private known-speaker reference descriptors as trusted worker-only data and keep them out of logs;
  - ensure `transcript.speakers.json` and `result_metadata.diarization` match existing studio behavior;
  - validate against `python-podcast/docs/known-speaker-runbook.rst` and `evals/known_speaker_results.md`.
- [ ] Keep claim eligibility bounded by advertised capabilities: `job_type=transcribe`, batch only, and
  known-speaker jobs only when the worker/hybrid path advertises the required pyannote/wespeaker/reference support.
- [ ] Preserve the C13 lane scheduler for any future local `studio` pull worker; `atlas.local` is outside that
  host-local gate.
- [ ] Prove the goal with a real production python-podcast Generate Transcript run that is claimed by `atlas.local`,
  populates the django-cast `Transcript.speakers` sidecar, clears the known-speaker quality bar, and leaves artifacts
  retrievable through normal Voxhelm job artifact URLs.

### `atlas.local` readiness checks

Summary of RW-2 in `specs/remote-transcription-workers.md`:

- [ ] Verify `atlas.local` can reach Voxhelm on `studio` over the private network.
- [ ] Verify `atlas.local` can reach MinIO/S3 for trusted worker artifact upload.
- [ ] Verify `ffmpeg` and at least one STT backend/model locally on `atlas.local`.
- [ ] Verify pyannote/wespeaker or the chosen hybrid postprocessor path for known-speaker jobs.
- [ ] Record a short and long transcription smoke-test baseline on `atlas.local` before implementing the worker API.
- [ ] Decide launchd/service supervision and log location for the future remote worker process.

## Speaker diarization deployment and consumer follow-through

Status as of 2026-05-27: first Voxhelm diarization output slice is implemented and locally smoke-tested. A
3-minute clip from `pp_67` completed successfully with generic `Speaker 1` / `Speaker 2` labels in verbose JSON,
DOTe, and Podlove. Full-episode research on representative Python Podcast audio showed that anonymous pyannote
diarization with speaker-count hints can still merge a real speaker into another cluster, so generic diarization
should be treated as a fallback/debug signal for known-speaker podcasts. Known-speaker voiceprint experiments with
clean contributor reference material are the preferred follow-up direction.

### Voxhelm repo

- [ ] Clean up and commit the diarization implementation, including pyannote 4 `DiarizeOutput` unwrapping.
- [ ] Resolve the pending `jobs` migration check:
  - `uv run python manage.py makemigrations --check --dry-run` currently reports an index-rename migration for `StagedMedia`.
- [ ] Close the pre-existing batch job idempotency race: concurrent submissions with the same `task_ref` and normalized payload can both miss the dedup query and create duplicate jobs. Consider a uniqueness strategy such as a normalized payload hash or locking the dedup path inside the create transaction.
- [ ] Keep `pyannote.audio` behind the optional `diarization` extra and verify fresh install with:
  - `uv sync --extra diarization`
- [ ] Decide whether the current pyannote/torchcodec warning is acceptable in production logs. Voxhelm avoids torchcodec decoding by passing ffmpeg-decoded waveform data to pyannote, but pyannote still emits the import-time warning.
- [ ] Add diarization quality metadata and warnings for pathological label distributions, including tiny clusters and
  distributions that contradict the requested speaker count.
- [ ] Prefer `exclusive_speaker_diarization` for transcript alignment when pyannote returns it.
- [ ] Add a non-default known-speaker postprocessor behind an explicit request flag. It should classify mastered
  transcript segment windows against contributor reference embeddings, preserve anonymous pyannote labels as
  fallback/debug metadata, and emit candidates, margins, confidence, and uncertainty flags.
- [ ] Keep `pyannote/wespeaker-voxceleb-resnet34-LM` as the first known-speaker embedding model candidate based on
  the 2026-05-27 research spike; keep `pyannote/embedding` as a possible high-precision/lower-coverage alternative.
- [ ] Document required Hugging Face access for the configured pyannote model and gated dependencies:
  - `pyannote/speaker-diarization-3.1`
  - `pyannote/speaker-diarization-community-1`
  - `pyannote/wespeaker-voxceleb-resnet34-LM`
  - any additional gated repo pyannote reports at model load time.

### ops-library

- [ ] Extend `roles/voxhelm_deploy` defaults with diarization settings:
  - `voxhelm_diarization_backend: "none"`
  - `voxhelm_pyannote_model: "pyannote/speaker-diarization-3.1"`
  - `voxhelm_huggingface_token: ""`
- [ ] Render these into `/etc/voxhelm/voxhelm.env`:
  - `VOXHELM_DIARIZATION_BACKEND`
  - `VOXHELM_PYANNOTE_MODEL`
  - `VOXHELM_HUGGINGFACE_TOKEN`
  - optionally `HF_TOKEN` with the same value for library compatibility.
- [ ] Make the role install optional dependencies when pyannote is enabled:
  - current: `uv sync --frozen --no-dev`
  - needed: `uv sync --frozen --no-dev --extra diarization`
- [ ] Ensure the token is only present in the protected env file, not inline in launchd plists or task logs.
- [ ] Ensure app and worker restart when the env file or dependency set changes.

### ops-control

- [ ] Add `huggingface_token` to `secrets/prod/voxhelm.yml.example`.
- [ ] Store the real token only in SOPS-encrypted `secrets/prod/voxhelm.yml`.
- [ ] Pass the secret to `local.ops_library.voxhelm_deploy` in `playbooks/deploy-voxhelm.yml`:
  - `voxhelm_diarization_backend: "pyannote"`
  - `voxhelm_pyannote_model: "pyannote/speaker-diarization-3.1"`
  - `voxhelm_huggingface_token: "{{ service_secrets.huggingface_token | default('') }}"`
- [ ] Add deploy validation: if pyannote is enabled, fail clearly when the token is missing or still `CHANGEME`.
- [ ] Add/keep a short diarization smoke-test command in docs or runbook.

### django-cast

- [ ] Add private contributor voice references for reviewed clips or source ranges. These are private admin/editor
  data, not public contributor profile metadata.
- [ ] Once Voxhelm has a known-speaker contract, send approved references for expected episode contributors using
  private job artifacts, signed private URLs, or source ranges. Do not send public profile URLs.
- [ ] Store returned candidates, margins, confidence, raw diarization labels, and uncertainty flags as reviewable
  transcript speaker state.
- [ ] Keep generic labels as fallback/debug metadata; do not auto-map `Speaker 1` to the first contributor.
- [ ] Decide how the current destructive contributor mapping workflow should evolve into non-destructive mapping or
  reviewed suggestion state.

### python-podcast

- [ ] Bump/update the `django-cast` dependency after django-cast diarization support lands.
- [ ] Refresh `uv.lock` and apply django-cast migrations.
- [ ] Enable diarization through django-cast site settings or environment.
- [ ] Confirm the `cast_transcripts` worker remains the path for completion; full diarization must not block admin HTTP requests.
- [ ] Update deployment docs with expected long-running CPU-heavy diarization behavior.

### podcast-transcript CLI

- [ ] Optional: add a Voxhelm batch mode if CLI-generated transcripts should include speaker labels.
- [ ] Current `podcast-transcript` Voxhelm backend calls sync `/audio/transcriptions` and performs local DOTe/Podlove conversion, so it drops speaker labels.
