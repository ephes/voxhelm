# Bounded Inference Slots for the C13 Lane Scheduler

Status: PLANNED (design reviewed 2026-09-06, implementation in progress). Decision: D-24 in
`decision-log.md` (reopens D-19 Option B with new evidence).

## Problem

Every local STT inference on `studio` (sync `POST /v1/audio/transcriptions`,
local batch jobs, Wyoming STT) is serialized twice: by the cross-process C13
lane scheduler (`lane_scheduler.py`, one `holder.json`) and, inside each
process, by a process-wide `_TRANSCRIPTION_LOCK` in `transcriptions/service.py`.
The scheduler gives the interactive lane priority for the *next* admission but
cannot preempt a running inference. A 40-minute memo through the sync endpoint
(15 to 20 minutes of `whisper-cli`) therefore makes the Home Assistant voice
pipeline unusable for that long.

Motivation: the Daybook Voice Memo importer is about to gain a long-memo lane
(ops-meta spec `2026-09-05_voice_memo_long_lane.md`) and podcast-length batch
jobs already exist. The fix belongs in Voxhelm, not in importer-side scheduling.

## Established facts (measured 2026-09-05)

- `studio`: Apple M4 Max, 128 GiB RAM. `ggml-large-v3.bin` is 2.9 GiB; two or
  three concurrent `whisper-cli` processes are not a memory problem. The shared
  resource is the Metal GPU: concurrent runs slow each other roughly
  proportionally.
- Production (`/private/etc/voxhelm/voxhelm.env`): `VOXHELM_STT_BACKEND=
  whispercpp` (subprocess per request), `VOXHELM_STT_FALLBACK_BACKEND=mlx`,
  `VOXHELM_WYOMING_STT_BACKEND=mlx` (in-process, inside the Wyoming sidecar
  process), `VOXHELM_LANE_SCHEDULER_ENABLED=true`, stale window 1800 s,
  `VOXHELM_TRANSCRIPTION_EXECUTION_MODE=remote_pull` (batch transcribe jobs go
  to remote workers; only the sync endpoint and Wyoming run STT locally).
- Process topology on `studio` (unchanged by this slice): one uvicorn ASGI
  process for the HTTP API (Django 5.2, sync views run in per-request threads),
  one Django Tasks worker (TTS batch jobs only in production), one Wyoming
  sidecar (asyncio, inference via `asyncio.to_thread`).
- TTS already uses per-backend in-process locks (`_PIPER_LOCK`, `_KOKORO_LOCK`)
  under the same scheduler and has no process-wide lock.

## Design

### Scheduler: bounded slots with a reserved interactive slot

`LaneScheduler` keeps its state directory, `control.lock`, waiter files,
lane priority (interactive first, then FIFO by `created_at`), and the
stale-holder recovery. `holder.json` becomes a `holders/` directory with one
`<token>.json` per admitted holder (`token`, `lane`, `pid`, `claimed_at`).

Admission rule, evaluated under the control lock for the head-of-queue waiter
in priority order:

- Let `I` = `VOXHELM_LANE_SCHEDULER_INTERACTIVE_SLOTS` (default 1) and
  `N` = `VOXHELM_LANE_SCHEDULER_NON_INTERACTIVE_SLOTS` (default 1).
  Total capacity is `I + N` (default 2).
- A non-interactive waiter is admitted only while fewer than `N` non-interactive
  holders exist **and** fewer than `I + N` holders exist in total.
- An interactive waiter is admitted while fewer than `I + N` holders exist in
  total. The `I` slots are therefore reserved for the interactive lane: a
  non-interactive job can never occupy them, while an interactive request may
  also use an idle non-interactive slot (it has priority anyway).
- Waiters are still examined in priority order (lane, then FIFO). A waiter is
  admitted only when it is the first admissible waiter, so FIFO holds within a
  lane and interactive waiters keep winning the next free slot.
- No preemption: running holders are never interrupted by the scheduler.
- Per-holder stale recovery: each holder file is reclaimed independently when
  its pid is dead or its age exceeds `VOXHELM_LANE_SCHEDULER_STALE_SECONDS`
  (unchanged default 1800 s). Dead-waiter cleanup is unchanged.
- `VOXHELM_LANE_SCHEDULER_ENABLED=false` keeps yielding a `disabled` ticket
  without any gating.
- `I=0, N=1` reproduces the old single-slot behaviour exactly; it is the
  config-only rollback and is covered by a test.
- Release removes only the holder file that belongs to the ticket's token,
  under the control lock; releasing a ticket whose file was already reclaimed
  (stale window elapsed, or a repeated release) is a harmless no-op and never
  touches another holder's file.
- Holder records are validated like waiter records (`token`, `lane` through
  `normalize_lane`, `pid`, `claimed_at`); a malformed or unreadable holder file
  is removed with a warning, because an unparseable record cannot take part in
  the admission count.
- Legacy `holder.json` (written by the previous single-slot code) is treated as
  a non-interactive holder in the count, read under the control lock and
  subject to the same dead-pid and stale reclamation, and never deleted merely
  because the layout is new. This protects new-code admissions from a legacy
  holder that is still running. It is one-directional: old code does not see
  `holders/`, so during a mixed-version window (the seconds between the
  sequential launchd restarts of the three services, or at worst the length of
  one in-flight legacy inference) an old-code process can still acquire
  `holder.json` next to a new-code holder and exceed the bound by one. The
  same applies to a code rollback until all three services run the old code.
  Accepted as a transient GPU-share effect with no safety impact: the only
  backends that can overlap are separate processes. A deploy or rollback must
  restart all three services in one run (the playbook already does); an
  operator who wants a strict bound during the switch stops all three services
  first. No bidirectional compatibility layer is added.

Validation at settings import: `N >= 1`, `I >= 0`, both integers; anything
else fails fast like the other `validate_positive_int` knobs.

Optional cooperative abort while waiting: `admit_local_inference(lane,
cancel_event=None)` passes the event to `LaneScheduler.acquire(lane,
cancel_event=None)`. When it is set while the waiter is still queued, the
waiter file is removed and `InferenceCancelled` (defined in `lane_scheduler`
and re-exported by the service) is raised, so a client that disconnected while
queued never starts an inference. With the scheduler disabled there is no wait,
and the service-level checkpoints below still apply.

### Service: locks only where a backend is in-process

- Remove the process-wide `_TRANSCRIPTION_LOCK` from `transcribe_audio`.
- `MlxWhisperBackend.transcribe` takes a module-level `_MLX_LOCK` around
  `mlx_whisper.transcribe`: mlx-whisper keeps model state in the process and is
  not safe to overlap inside one process.
- `WhisperCppBackend` (separate `whisper-cli` process per request) and
  `WhisperKitBackend` (sidecar over HTTP) run under scheduler slots only. The
  README already documents that the WhisperKit sidecar's internal concurrency
  is outside the scheduler.
- TTS (`synthesis/service.py`) already matches this model; no change beyond
  documentation.
- Diarization and known-speaker locks are untouched (batch worker only).

Why this is safe for the deployed interactive path: the Wyoming sidecar is its
own process, so its MLX lock is process-local and its inference overlaps a
`whisper-cli` subprocess started by the HTTP process without sharing any
in-process state. No configuration change to the Wyoming backend is required.
The shared cost is Metal GPU time: an interactive request that overlaps a
long `whisper-cli` run is slower than on an idle host (measured live, see
verification), not blocked.

### Subprocess cancellation on client disconnect (sync endpoint)

- `TranscribeParams` gains `cancel_event: threading.Event | None =
  field(default=None, compare=False, repr=False)`; every existing constructor
  call (views, Wyoming, `jobs/services.py`, `jobs/remote_worker_cli.py`) and
  parameter equality stay unchanged.
- `WhisperCppBackend` runs `whisper-cli` with `subprocess.Popen` and polls
  `communicate(timeout=...)`; when `cancel_event` is set it terminates the
  child (SIGTERM, then SIGKILL after a short grace period) and raises
  `InferenceCancelled`. Without a cancel event the behaviour is identical to
  today. The short ffmpeg normalization step is left as is.
- Cancellation checkpoints: `transcribe_audio` raises `InferenceCancelled`
  when the event is set after admission and again before every backend attempt
  in the primary/fallback loop; `MlxWhisperBackend` checks once more after it
  has acquired `_MLX_LOCK`; `WhisperCppBackend` checks immediately before
  `Popen` (after ffmpeg normalization). `InferenceCancelled` is not a
  `BackendUnavailableError`, so cancellation never triggers backend fallback.
  These checkpoints also hold with the scheduler disabled.
- Subprocess lifecycle on cancel: SIGTERM, wait up to 5 s, SIGKILL if still
  alive, then wait again so the child is reaped and its pipes drained before
  `InferenceCancelled` propagates. Slot release (scheduler `finally`) and temp
  file deletion (view `finally`) therefore always happen after the child is
  gone. `communicate(timeout=...)` only bounds the wait; it never kills.
- `audio_transcriptions` becomes an `async def` view with two phases so the
  request body is never read after Django's ASGI handler may close it:
  1. Parse phase (auth, multipart/JSON parsing, copying the upload or download
     to a temp file) runs in a worker thread wrapped in `asyncio.shield`. If
     the task is cancelled during this phase, the view still awaits the
     shielded parse to finish, deletes the temp file it produced, and only then
     re-raises. Nothing was admitted yet, so there is nothing to terminate.
  2. Inference phase runs `transcribe_audio` in a worker thread with the
     cancel event, as a retained task awaited through `asyncio.shield`. On
     `asyncio.CancelledError` the view sets the event, attaches a done-callback
     that consumes the task's result or exception (logging anything other than
     `InferenceCancelled`), and re-raises at once. Because the inner task is
     shielded it is never cancelled itself: if its thread has not started yet
     (executor saturated) it still runs later, hits the first checkpoint, and
     raises `InferenceCancelled`; if it is running, it terminates the child and
     releases the slot. In both cases the thread function's own `finally`
     deletes the temp file (ownership of the temp file moves to the inference
     thread for this phase).
  Django 5.2's ASGI handler cancels the request task on `http.disconnect`
  ([docs](https://docs.djangoproject.com/en/5.2/topics/async/#handling-disconnects)).
  Under WSGI/`runserver` and the Django test client the view runs through
  `async_to_sync` with no cancellation signal; responses are unchanged.
- In-process backends (MLX) cannot be cancelled mid-inference; a disconnect
  only prevents a not-yet-started backend from running.
- Wyoming: not applicable. Requests are a few seconds long and the deployed
  backend is in-process; a Wyoming client disconnect keeps today's behaviour.
- Production note: the sync endpoint is reached through Traefik on `macmini`.
  Traefik cancels the upstream request when the client disconnects, so the
  signal reaches uvicorn; the live verification includes one deliberate
  disconnect to confirm this on the deployed stack.

### Configuration

Voxhelm (`config/settings.py`, README, env examples):

| Variable | Default | Notes |
|---|---|---|
| `VOXHELM_LANE_SCHEDULER_INTERACTIVE_SLOTS` | `1` | reserved for the interactive lane; `>= 0` (`0` = D-19 single-slot rollback) |
| `VOXHELM_LANE_SCHEDULER_NON_INTERACTIVE_SLOTS` | `1` | cap for non-interactive holders; `>= 1` |

Existing knobs (`ENABLED`, `DIR`, `STALE_SECONDS`) keep their meaning.

ops-library `voxhelm_deploy`: `voxhelm_lane_scheduler_interactive_slots: 1`,
`voxhelm_lane_scheduler_non_interactive_slots: 1` in defaults, rendered into
`voxhelm.env.j2`, documented in the role README.

ops-control `deploy-voxhelm.yml`: pin both values explicitly (`1` / `1`) next
to the existing Wyoming/scheduler vars so production concurrency is
deliberate, not inherited.

### Documentation checklist

- README: replace the "Current limitation: the first C13 lane scheduler
  slice ..." paragraph with the slot model, keep the WhisperKit
  internal-concurrency caveat, document the two knobs and the disconnect
  behaviour of the sync endpoint, and the Wyoming note (MLX in its own process
  overlaps safely).
- `specs/decision-log.md`: D-24 (this decision); D-19 untouched.
- BACKLOG: keep the line "Preserve the C13 lane scheduler for any future local
  `studio` pull worker" (still true: a local pull worker would take
  non-interactive slots); add/close the C13 follow-up for this slice.
- `specs/implementation-sequence.md` and `specs/delivery-chunks.md`: one-line
  status note under the C13 entries pointing at D-24.
- ops-library role README: the two new vars and the updated cooperative/
  non-preemptive paragraph.
- ops-control: pinned values with a comment pointing at this spec.

### Out of scope (unchanged)

Public HTTP/batch API, job model, remote-pull worker protocol, artifact
storage, `GET /v1/status`, client-facing lane selection, preemption, any new
queue or process topology, slot counts above `1 + 1` in production.

## Test plan

`tests/test_lane_scheduler.py`

- interactive waiter is admitted while a non-interactive holder runs (reserved
  slot), and a second non-interactive waiter is not;
- non-interactive cap: with `N=1`, two non-interactive requests never overlap;
  interactive waiter still wins the next free slot over an older
  non-interactive waiter (priority + FIFO retained);
- total bound: with `I=1, N=1`, never more than two holders;
- same-lane FIFO: three non-interactive waiters queued behind a holder (with
  `N=1`) are admitted in creation order;
- priority with capacity full: `I=1, N=1`, one interactive and one
  non-interactive holder, an older non-interactive waiter and a younger
  interactive waiter; the test waits until both waiter files exist, then
  releases the **non-interactive** holder so both waiters become eligible;
  the interactive waiter must be admitted first;
- `I=0, N=1` reproduces single-slot serialization;
- per-holder stale recovery: one stale and one live holder file (live pid =
  the test process), the stale one is reclaimed and the live one is kept;
  dead-pid holder reclaimed immediately; malformed holder JSON is removed;
- release semantics: releasing a ticket whose holder file was already reclaimed
  (and replaced by a newer holder) is a no-op that leaves the newer holder
  intact; releasing twice is harmless;
- legacy `holder.json` counts as a non-interactive holder until it is released
  by its stale/dead-pid path, and is not deleted on layout creation;
- `cancel_event` set while queued removes the waiter and raises
  `InferenceCancelled`;
- all contention tests use events/barriers (holder signals "admitted" before
  waiters start; waiters signal admission) rather than sleeps for ordering;
- real subprocesses: two or three Python child processes acquire against the
  same state directory and report their overlap; the cross-process bound and
  the reservation hold (the existing suite uses threads; this adds the
  cross-process case the design depends on).

`tests/test_service.py`

- `transcribe_audio` no longer serializes a subprocess-style backend: with
  the scheduler disabled (or `N=2`), two concurrent calls overlap (barrier
  inside the stub backend proves both are active at once), while two
  concurrent `MlxWhisperBackend` calls (stubbed `mlx_whisper`) never overlap;
- `WhisperCppBackend` terminates the child and raises `InferenceCancelled`
  when the cancel event is set, for a child that sleeps and for a child that
  ignores SIGTERM (escalation to SIGKILL); afterwards the child pid is gone
  and `Popen.returncode` is set;
- cancel set before the backend starts skips every backend and does not fall
  back; cancel set between primary and fallback attempts stops the loop;
- cancel set while a second caller waits for `_MLX_LOCK` makes that caller
  raise after acquiring the lock without invoking `mlx_whisper`;
- cancel set during ffmpeg normalization (stubbed) raises before `Popen`;
- cancel racing a child that exits on its own returns the child's result or
  `InferenceCancelled` but never hangs or leaves a zombie;
- every cancellation path releases the scheduler slot (assert `holders/` is
  empty afterwards with the scheduler enabled);
- checkpoints hold with the scheduler disabled.

`tests/test_api.py`

- ASGI disconnect during inference: drive `get_asgi_application()` directly
  with a scripted `receive` (multipart body, then `http.disconnect` once the
  stub `transcribe_audio` reports it is running); assert the stub's cancel
  event was set, the stub returned, and the upload temp file is gone;
- ASGI disconnect during parsing: the stub upload-to-tempfile blocks until
  the disconnect has been sent; assert the request body stream was still
  readable until parsing finished, no backend was invoked, and the temp file
  is gone;
- ASGI disconnect with a saturated executor: the loop's default executor is a
  one-worker pool held busy by the test when the disconnect arrives, so the
  inference thread starts only afterwards; assert it raises at the first
  checkpoint, no backend was invoked, and the temp file is gone (worker
  cleanup is awaited separately from ASGI completion);
- existing sync-client tests keep passing unchanged.

`tests/test_settings*.py` (or the existing settings tests): validation of the
two new knobs.

`tests/test_wyoming.py`: unchanged behaviour; the scheduler-disabled path is
already covered.

## Live verification plan (studio, content-free)

1. Deploy via `just deploy-one voxhelm` from `ops-control` (canonical
   checkout), confirm the three launchd services restarted and
   `voxhelm.env` carries the two new knobs.
2. Generate synthetic audio locally: `say` a repeated neutral sentence to
   AIFF, `ffmpeg` it to a 20-minute mono Opus/MP3 file under the 25 MiB upload
   limit, and a separate 3-second clip for the interactive probe.
3. Baseline: time one Wyoming STT request (3-second clip, via the `wyoming`
   client library against `studio:10300`) on an idle host.
4. Start the 20-minute file through `POST /v1/audio/transcriptions`
   (`model=whisper-1`) and, while `whisper-cli` is running, time the same
   Wyoming request three times. Record both latencies, the scheduler log
   lines (`lane_scheduler admitted lane=... wait_ms=...`), peak RSS of
   `whisper-cli` and the Wyoming process (`ps -o rss`), and that the long
   job returned HTTP 200.
5. Disconnect check: start a second long request through the Traefik host and
   abort the client after ~20 s; confirm the `whisper-cli` child disappears
   within a few seconds and the slot is released (log line, `holders/` empty).
   Repeat once directly against the uvicorn port to separate proxy behaviour
   from application behaviour if the Traefik run does not cancel.
6. Remote-pull batch path: confirm the worker heartbeat/claim log lines
   continue and `just check` covers the untouched job code; no batch job is
   submitted for this slice.

## Rollback

Redeploy with `voxhelm_lane_scheduler_interactive_slots: 0` and
`voxhelm_lane_scheduler_non_interactive_slots: 1` to return to single-slot
serialization without a code revert; or revert the commit range on `main`.
