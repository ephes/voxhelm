# Voxhelm Backlog

## Speaker diarization deployment and consumer follow-through

Status as of 2026-05-20: first Voxhelm diarization output slice is implemented and locally smoke-tested. A 3-minute clip from `pp_67` completed successfully with generic `Speaker 1` / `Speaker 2` labels in verbose JSON, DOTe, and Podlove. Full-episode runtime and deployment wiring are still open.

### Voxhelm repo

- [ ] Clean up and commit the diarization implementation, including pyannote 4 `DiarizeOutput` unwrapping.
- [ ] Resolve the pending `jobs` migration check:
  - `uv run python manage.py makemigrations --check --dry-run` currently reports an index-rename migration for `StagedMedia`.
- [ ] Close the pre-existing batch job idempotency race: concurrent submissions with the same `task_ref` and normalized payload can both miss the dedup query and create duplicate jobs. Consider a uniqueness strategy such as a normalized payload hash or locking the dedup path inside the create transaction.
- [ ] Keep `pyannote.audio` behind the optional `diarization` extra and verify fresh install with:
  - `uv sync --extra diarization`
- [ ] Decide whether the current pyannote/torchcodec warning is acceptable in production logs. Voxhelm avoids torchcodec decoding by passing ffmpeg-decoded waveform data to pyannote, but pyannote still emits the import-time warning.
- [ ] Validate runtime on representative full podcast episodes, not only short clips.
- [ ] Document required Hugging Face access for the configured pyannote model and gated dependencies:
  - `pyannote/speaker-diarization-3.1`
  - `pyannote/speaker-diarization-community-1`
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

- [ ] Add a configurable diarization switch:
  - global/env setting: `CAST_VOXHELM_DIARIZATION_ENABLED`, default false;
  - site-level `VoxhelmSettings.diarization_enabled` boolean.
- [ ] Include top-level `"diarization": {"enabled": true}` in Voxhelm batch job payloads only when enabled.
- [ ] Add tests for payload shape, settings precedence, and unchanged behavior when disabled.
- [ ] Add migration and docs/release notes.
- [ ] Keep generic labels as the first slice; do not auto-map `Speaker 1` to the first contributor.
- [ ] Design/implement a later contributor mapping workflow from generic speaker labels to `EpisodeContributor` records.

### python-podcast

- [ ] Bump/update the `django-cast` dependency after django-cast diarization support lands.
- [ ] Refresh `uv.lock` and apply django-cast migrations.
- [ ] Enable diarization through django-cast site settings or environment.
- [ ] Confirm the `cast_transcripts` worker remains the path for completion; full diarization must not block admin HTTP requests.
- [ ] Update deployment docs with expected long-running CPU-heavy diarization behavior.

### podcast-transcript CLI

- [ ] Optional: add a Voxhelm batch mode if CLI-generated transcripts should include speaker labels.
- [ ] Current `podcast-transcript` Voxhelm backend calls sync `/audio/transcriptions` and performs local DOTe/Podlove conversion, so it drops speaker labels.
