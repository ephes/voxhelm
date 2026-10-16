# `ad-analysis-v1` transcription profile

Status: 2026-10-16, implemented and verified locally (see "Verification");
not deployed. Decision: D-26 (and the D-09 note of the same day). Source:
slice 1 of the podcast server-side ad analysis,
`~/projects/podcast/docs/player/specs/server-ad-analysis-design.md`
("Voxhelm Adapter Contract", "Retention in voxhelm", gate G1).

## Why

The podcast backend (Delve) sends each analysed episode to Voxhelm and feeds
the words to an ad detector that was tuned on research transcripts: Parakeet
TDT 0.6B v3 through `parakeet-mlx` 0.5.3, model commit
`ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15`, 30 s chunks, 5 s overlap, full
attention, no sentence cap. Voxhelm so far routes Parakeet only for
known-speaker jobs, with the deployment's 600 s / 30 s / local attention /
`--max-duration 30` settings, a whisper.cpp fallback and no word output in
`verbose_json`. Any of these would change the detector's input without
anyone noticing. Delve also promises that Voxhelm keeps no transcript copy
for long, but transcript artifacts are kept forever today.

## Scope

In scope: a named profile `ad-analysis-v1` on `POST /v1/jobs`, locally and
on remote workers; the `words` artifact; recorded engine identity;
`language_hint`; worker capability advertising; producer
`DELETE /v1/jobs/{id}`; the profile's 7-day backstop in
`prune_job_artifacts`.

Out of scope (later slices of the podcast spec): the whisper turbo window
endpoint and the German drift check, scheduling `prune_job_artifacts` (slice
16). The window transcription (slice 2) is now in
`2026-10-16_window_transcription.md` (D-27). Amended 2026-10-16 by slice 3
(`2026-10-16_drift_check.md`, D-28): the drift check runs inside this profile,
adds per-window records to the words artifact (`drift`) and the drift rule and
retry model hash to the identity, and the profile needs the turbo window
runtime as well. Because nothing of this profile had been deployed or served,
`ad-analysis-v1` was amended in place instead of becoming a new name (D-28);
from its first deploy on any change is a new name.

## Design

### Profile registry

`transcriptions/profiles.py` holds `PROFILES`, one frozen entry per name. A
profile owns every result-affecting setting; it is never configured by
environment. `ad-analysis-v1`: backend `parakeet`, `--chunk-duration 30
--overlap-duration 5 --full-attention`, no `--max-duration`, greedy decoding
and bf16 (the CLI defaults), artifact backstop 7 days. It uses only the
deployment's `VOXHELM_PARAKEET_BIN`, `VOXHELM_PARAKEET_MODEL_DIR`,
`VOXHELM_PARAKEET_MODEL_NAME` and `VOXHELM_PARAKEET_TIMEOUT_SECONDS`. A change
to the settings is a new profile name.

`ParakeetBackend` gained `run()` (one CLI run, returns the JSON payload; the
known-speaker path's `transcribe()` builds on it) and a `local_attention`
option (default on, so the known-speaker invocation is unchanged).

### Submission

`POST /v1/jobs` with `job_type=transcribe` accepts `profile` and
`language_hint`:

- an unknown profile, `language` together with a profile, a `model` other
  than `auto`, diarization, a `language_hint` without a profile, and a hint
  outside Parakeet's 25 languages are rejected with 400;
- the hint is normalized (`de-DE` -> `de`) and stored as the job's language;
  Parakeet takes no language argument, so it only selects support and is
  recorded;
- `Job.profile` is a new column (migration `0010`, indexed with `state`); the
  profile is part of the `task_ref` match and fingerprint (only when set, so
  fingerprints of existing jobs are unchanged);
- the job payload carries `"profile"`.

### Execution without fallback

Local (`django_tasks`): `execute_transcription_job` runs
`run_profile_transcription` under the non-interactive lane instead of
`transcribe_audio`. It builds exactly one `ParakeetBackend` with the profile's
settings; a missing CLI or model (`BackendUnavailableError`) fails the job.
No other backend is ever built. The sanitizer does not run: the words and
the transcript artifacts are the engine's output. The task returns only
`{"job_id", "profile"}`: `django_tasks_db` keeps a task's return value in
`DBTaskResult`, outside the job, so it must not carry the transcript.

Remote (`remote_pull`): workers advertise `capabilities.profiles`, computed by
the same per-heartbeat check that advertises `parakeet` (CLI resolves, model
directory holds `config.json` and `model.safetensors`). The control plane lets
only such a worker claim a profile job; its STT target is the profile's
backend and model, the claim carries `profile` and never `fallbacks`. The
claim transaction records the attempt's artifact prefix and store
(`RemoteAttemptPrefix`, migration `0010`) before the worker uploads anything,
so objects of an attempt whose completion never registered them stay
findable. A job
whose only worker lacks the profile waits. A worker that loses the model
between heartbeat and run raises `BackendUnavailableError`, reports the
attempt as retryable, and the job returns to the queue (bounded by
`VOXHELM_REMOTE_WORKER_MAX_ATTEMPTS`).

### Words artifact

`words.json`, kind `words`, format `words`, `application/json`, exposed:

```json
{"schema_version": 1, "profile": "ad-analysis-v1", "identity": {...},
 "words": [{"w": "Hi,", "s": 5.84, "e": 6.24, "c": 0.973}, ...],
 "drift": {...}}
```

The `words` are Parakeet's with the drift check's accepted retries spliced in
(retry words have `c: null`); `drift` holds its per-window records
(`2026-10-16_drift_check.md`).

Words are built from all tokens of the CLI output in order, across sentence
boundaries, with the research runner's rule: a token whose text starts with
a space starts a word, any other token is appended and moves the end; `c` is
the mean of the word's token confidences; times and `c` rounded to 3
decimals. The CLI writes token confidences already rounded to 3 decimals, so
`c` can differ from the research value (mean of unrounded confidences) by
0.001; words and times are unaffected. Malformed output (missing tokens,
times or confidences) fails the run.

Remote completions of a profile job must include `words.json`; other jobs
must not.

### Identity

`result.metadata.profile` and the artifact's `identity`:

| Field | Source |
| --- | --- |
| `profile`, `backend`, `settings` | the profile entry |
| `model` | `VOXHELM_PARAKEET_MODEL_NAME` |
| `model_repo`, `model_revision`, `model_sha256` | the `VOXHELM_MODEL_REVISION` marker the ops-library role writes into the model directory; for a Hugging Face snapshot directory the commit is its name; empty when unknown |
| `cli`, `cli_version`, `mlx_version` | the CLI script's own interpreter (`importlib.metadata` of `parakeet-mlx` and `mlx`), `--version` as fallback; cached per CLI file and mtime |
| `drift` | the drift rule (`fw-en4-share10-v1`, thresholds, function-word list hash, acceptance) and `retry`: whisper.cpp version, turbo model name and SHA-256, decoding settings (D-28) |
| `language_hint` | the job's hint |

The control plane keeps only these fields from a remote worker and sets
`language_hint` from the job; the worker's `drift` must equal the control
plane's own (only its whisper.cpp version is kept from it).

### Retention

`DELETE /v1/jobs/{id}` with the producer token: for a finished profile job of
that producer, refuse up front if any artifact key, any object below a
recorded attempt prefix (`jobs/<id>/attempt-<n>/<file>`, listed in the store)
or a claimed staged upload's key (`staged-inputs/<upload_id>/<filename>`) is
outside the allowlist (409, nothing deleted); then delete each artifact
(object, then row, own transaction under the write lock, an object still
referenced elsewhere is kept), drain the job's queued object deletions, delete
whatever each attempt prefix still holds (objects of rejected, failed, retried
or abandoned attempts) and then its record, delete a staged upload it still
holds (same lock, allowlist and reference check), and finally the row together
with the job's `DBTaskResult` (older results still carry the transcript). 204
on success; 404 for an unknown or another producer's job (also on a repeat);
409 for a job without a profile or not yet finished; 503 when an object could
not be deleted or listed (progress is kept, the request can be repeated).

`prune_job_artifacts`: every artifact of a terminal profile job that finished
more than the profile's backstop ago (7 days) is a candidate, whatever its
kind or exposure, through the same allowlisted per-artifact deletion; then
its attempt prefixes are cleaned and a staged upload it still holds is
deleted (a remote completion keeps the staged row when the object deletion
fails), as in DELETE; the row and its task result follow in the same run once
nothing is left, independent of
`VOXHELM_JOB_METADATA_RETENTION_SECONDS` (also when that is `0`). Within the
7 days only the usual D-09 rules apply (the `source` copy after 24 h). Other
jobs are unchanged.

## Tests

`tests/test_ad_analysis_profile.py`:

- submission: stored profile and normalized hint; every rejected combination;
  the profile in the `task_ref` identity;
- local job with a fake CLI: profile arguments (30 / 5 / full attention, no
  `--max-duration`) although the deployment's Parakeet settings say
  600 / 30 / 30 s; identity; the served words artifact; with the model
  missing the job fails and no other backend is built;
- words builder (sub-word joins across a sentence boundary, mean confidence,
  rounding) and malformed output; snapshot-directory revision fallback;
- remote: a worker advertising `parakeet` and `whispercpp` but not the
  profile gets no claim and the job stays queued; a profile worker gets a
  claim with `profile` and no `fallbacks`; completion records the sanitized
  identity and serves the words; a completion without words, with unexposed
  words or with a wrong profile identity is rejected; capability advertising
  follows the model directory; a profile claim with `fallbacks` and a missing
  model raises without building another backend and is retryable; a worker
  run through `process_claim` uploads `words.json` and the identity;
- DELETE: artifacts, objects, queued deletions, staged upload and row gone,
  then 404; the objects of a failed final attempt whose completion was
  rejected are gone; a staged object another job's artifact references is
  kept; a staged upload or an attempt object outside its layout gives 409
  with nothing deleted; the task result of a job run through `db_worker`
  holds no transcript and goes with the job, as does an older one that does; another producer gets 404 and nothing changes; non-profile,
  queued and running jobs get 409; an artifact outside the allowlist gets
  409 with nothing deleted; a failed object deletion gives 503 and the retry
  finishes;
- backstop: a profile job finished 7 days and 1 minute ago loses every
  artifact and its row on the first run (dry run lists the same), one
  finished 6 days 23 hours ago keeps transcript and words, an ordinary
  30-day-old job keeps its final artifacts and row; the backstop also applies
  with job metadata retention disabled; it removes the objects of a retried
  attempt, a staged upload left claimed (but not one of a recent job) and an
  older task result holding the transcript.

## Verification

### G1: words against the research transcripts (2026-10-16)

All 44 development episodes of the ad research (22 German, 22 English,
44.9 h; `ws-podcast-ad-reassess-20261012/audio`), each staged with
`POST /v1/uploads` and run as an `ad-analysis-v1` job through the local
`django_tasks` code path (Django test client, immediate task backend, one
process), compared with the research Parakeet words
(`work/night-2026-10-14/parakeet-chain/pk-dev`, produced by `prep_run_pk.py`
through the Python API with `chunk_duration=30, overlap_duration=5`).

| | Research | This profile |
| --- | --- | --- |
| `parakeet-mlx` | 0.5.3 (Python API) | 0.5.3 (CLI; the package sources are byte-identical) |
| `mlx` | 0.32.3 | 0.32.4 |
| model | `mlx-community/parakeet-tdt-0.6b-v3` from the HF cache, commit `ed2b7e8c...` | the same snapshot directory, recorded as `model_revision` |
| settings | 30 s / 5 s, full attention, bf16, greedy | the same |

Result: on every one of the 44 episodes the word sequence is identical
(487,470 words, 100 % agreement, none missing or extra) and every word's
start and end time is identical (median and maximum difference 0.000 s). The
only difference is the confidence: 24,078 words (4.9 %) differ by 0.001
because the CLI writes token confidences rounded to 3 decimals before the
mean is taken; no word differs by more. The `mlx` patch release makes no
difference to words or times. G1 (≥ 99 % same words, median start difference
≤ 0.05 s) passes on all 44 files.

ASR time 1,891 s for 44.9 h (85x real time, on a shared, loaded M4 Max; the
research runs on the same machine took 0.8-7.6x as long per episode, under
other load, so the timings are noise-dominated). Peak resident size of the CLI child per episode 0.96-3.1 GB (median
2.2 GB); it grows with episode length, not with the number of episodes.

### Memory over consecutive episodes in one worker

The 10 longest development episodes (1.5-2.4 h each, 21.5 h) submitted at
once and run back to back by one Django Tasks `db_worker` (`--max-tasks 10`,
the local production execution path), sampling the worker's and its
children's resident size every second:

| Episode (in run order) | CLI child peak RSS | Worker RSS when the job started |
| --- | --- | --- |
| talkpython-0 | 2.4 GB | 73 MB |
| lwai-0 | 2.5 GB | 98 MB |
| engkiosk-2 | 2.4 GB | 103 MB |
| lwai-2 | 2.6 GB | 113 MB |
| lwai-1 | 2.7 GB | 123 MB |
| lwaidev-1 | 2.9 GB | 129 MB |
| atp-1 | 2.7 GB | 127 MB |
| bitsundso-0 | 3.0 GB | 143 MB |
| atp-0 | 2.2 GB | 141 MB |
| bitsundso-1 | 3.1 GB | 163 MB |

All 10 succeeded (53-114 s each). The child's peak follows the episode's
length and does not grow with the number of jobs; the worker process itself
stays below 170 MB (the research worker, which kept the model in-process,
grew to 8.9 GB over 46 files). The 10 words files are identical to the G1
run's, so the profile is deterministic across runs. Resident size, not the
macOS memory footprint, was sampled, as in the research numbers.
