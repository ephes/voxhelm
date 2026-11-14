# Changelog

## Unreleased

- German synthesis: preserve numeric tokens longer than 64 digits literally
  across units, EUR and temperatures, avoiding integer conversion failures and
  unbounded fractional expansion; ordinary pronunciation remains unchanged.
- Preserve literal `__GODE_CARDINAL_DOT__` input by rendering cardinal-label
  punctuation directly instead of reserving and globally restoring a marker.
- Transcript repetition runs now stop at every speaker-label change, including
  labeled/unlabeled boundaries; same-speaker and unlabeled artifact loops still
  collapse with their shared identity and timing preserved.
- Restrict Amara credit removal to whole subtitle attributions with `von` and
  the standalone domain, preserving genuine topic sentences and unknown phrasing
  that merely end in Amara.org.
- Keep pinned synthesis parameters for nonfinite, out-of-range or unsupported
  language detections; admit Piper voice configs through bounded regular-file
  descriptors, skipping unsafe configs without hiding healthy voices and without
  leaking the descriptor; a model without its config is still skipped quietly.
- Bound known-speaker interval work: fold tiny pieces without repeated word
  copying, resolve overlapping turns with a heap, and index unmapped overlays.
- Stop known-speaker embedding work cooperatively on cancellation/remote lease
  loss; cap the speaker track at 32,768 frames/model calls by widening the track
  step for very long audio (recorded as `speaker_track_hop` in the summary)
  instead of failing the job after transcription.
- Validate known-speaker configuration at API, stored/claimed and direct execution
  boundaries; cap durations/padding and reject ambiguous speaker display names
  before embeddings while preserving multiple clips for one identity.

## 0.1.13 (2026-11-12)

Published to PyPI for the remote transcription workers (`atlas`, `studio`),
which run 0.1.12 from PyPI until now. It carries the remote-worker backoff
below (`202ffe9`): a worker rejected by the control plane, for example while it
is disabled during a deploy drain, logs one line per attempt and backs off
instead of logging a traceback on every poll.

- Remote worker: a worker that the control plane rejects no longer logs a
  traceback on every poll. HTTP 403 on the worker heartbeat or claim happens,
  for example, while the worker is disabled during a deploy drain. The worker
  then logs one line per attempt:
  `control plane rejected worker (HTTP 403, worker disabled?); retrying in Ns`.
  Consecutive failed polls back off exponentially from the poll interval,
  doubling up to 60 s, or to the poll interval when that is longer. The first
  successful poll resets the backoff and, after a rejection, logs that the
  control plane accepted the worker again. Other failures use the same backoff
  and keep their traceback, as before. That covers 5xx, connection errors, and
  errors after a claim, such as a lost lease or a 403 on a job report.
  `Ctrl-C` during a wait now exits with status 130.

## 0.1.12 (2026-11-12)

Published to PyPI for the remote transcription workers (`atlas`, `studio`),
which install voxhelm from PyPI and ran 0.1.11 until now. It carries every
entry below that was written after 0.1.11 (`6594d0f`), in particular the
worker-side bounds: lease heartbeats that stop work when the lease is lost,
bounded reference and Pyannote decoding, cancellable lane waits, the 8 MiB
regular-file window JSON, chapter offsets within the clip length plus 1 s and
the normalizer in its own bounded process group. The claim-scan and Wyoming
changes only affect the control plane on `studio`, deployed from source.

- Remote claims: bound every claim scan to 100 hydrated and validated jobs with a
  resumable per-worker cursor (`Worker.claim_scan_cursor`, migration `0013`) so a
  large incompatible backlog is no longer loaded into Python on each poll, and keep
  the scan priority-correct: any claimable job created, requeued, updated or with
  an expired lease at or before the cursor since the sweep began restarts the scan
  from the top. Capability changes and finished sweeps also restart; a poll runs at
  most two scans and writes the cursor only when it changed.
- Remote completions: the per-object store checks of one manifest share a
  30-second cumulative budget and fail with HTTP 400 without settling the job.
- Window transcription: whisper.cpp JSON output must be a regular file (opened
  without following symlinks or blocking on a FIFO); chapter-window word offsets
  beyond the clip plus a 1 s tolerance are refused (was twice the clip length).
- Wyoming: remove the SIGINT/SIGTERM handlers the sidecar installed when it
  returns, and run all post-start code (including the startup log line) under the
  shutdown `finally`.

- Drop non-finite diarization turns and align speakers through an interval index
  instead of scanning every turn per segment; refuse negative, inverted or absurd
  chapter-window word offsets and cap whisper.cpp window JSON at 8 MiB.
- Remote completions: cap manifests at 32 artifacts before any store access,
  enforce artifact column limits and 64-bit sizes, and refuse two artifacts that
  name the same storage object.
- Run whisper.cpp audio normalization at error log level in its own process group,
  logging only a 2 KiB stderr prefix and discarding the rest (never returned); rotate failing expired staged uploads
  behind healthy ones during bounded admission cleanup.

- Speech API: refuse NaN/Infinity, over-long integers, parser nesting limits,
  boolean and overflowing `speed` values as HTTP 400; return a fixed public
  message for synthesis/export failures (missing voices and Kokoro text without
  supported tokens become HTTP 400); stream
  audio from an open handle after removing temporary files; skip unreadable Piper
  voice configs instead of failing discovery; run ffmpeg export in its own process
  group with a 64 KiB stderr ceiling and a deadline covering descendants.

- Wyoming: delete synthesis output that finishes after its request was cancelled,
  abort clients that stop reading after a 30-second write deadline, cap open
  connections (`VOXHELM_WYOMING_MAX_CONNECTIONS`, default 16) and inference
  threads before they start (`VOXHELM_WYOMING_MAX_INFERENCE_THREADS`, default 4),
  send fixed public failure texts instead of exception details (missing-voice
  messages stay client-facing), and abort and cancel all connection handlers on
  server stop or cancellation.

- Bound Pyannote decoding to two hours of 16 kHz mono PCM, reject invalid/truncated
  WAVs before waveform allocation, and discard bounded decoder diagnostics.
- Cache Pyannote pipelines only after successful device placement, allowing retries.

- Observe queued cancellation while waiting for the lane control lock, including
  registration and cleanup. New admission tickets hold an OS file lock so abandoned
  holders are reclaimed after process death even if their PID is reused; preserve
  live PID-only records from older workers and never reclaim by age alone.

- Refuse Kokoro requests with no supported speech tokens instead of returning
  empty WAVs; write PCM chunks incrementally with a 30-minute total frame cap
  and remove partial output after inference, cancellation or budget failure.

- Reject empty, nonfinite or malformed Kokoro inference audio before publishing
  WAVs; validate installed style shape and finite values before caching voicepacks,
  and report empty, truncated or corrupt voicepack archives as backend unavailable.

- Bind the historical fingerprint upgrade to the selected database and bound
  payload reads to 100-job keyset pages with indexed duplicate tracking.

- Retain ASGI download reader ownership through cancelled worker reads and close
  once after settlement, including streams closed before their first read, while
  keeping the filesystem `Content-Length` header.
- Preserve the primary artifact publication error when temporary cleanup also
  fails, reporting cleanup separately without exposing private diagnostics.

- Bind Wyoming chunks to supported `AudioStart` PCM geometry (8–192 kHz, 1/2/4-byte
  samples, mono/stereo) and whole frames,
  check the cumulative resampled frame budget before conversion, and reset
  converter history at each utterance boundary.

- Compensate failed lane holder publication without touching other holders, and
  keep admission logging non-authoritative. Refuse admission on state read I/O
  errors instead of reclaiming unreadable live holders.

- Reject malformed Wyoming typed events before state changes or inference, and
  close stalled/trickled event reads after a single 30-second elapsed deadline.

- Cover deletion-before-commit process death and completed staged-upload cleanup
  retry convergence, preserving referenced objects and committed manifests.

- Cover remote worker capacity, durable attempt ownership and complete/fail
  settlement with real process races, plus lease timing after writer contention.

- Group remote fair-share active/recent load counts in one query regardless of fleet size.

- Restrict remote fair sharing to peers capable of running current candidates;
  sample admission time after worker serialization for capacity and full leases.

- Pin historical request fingerprint bytes and optional-key omission with independent
  golden tests; cover transcription matcher equivalence, upload metadata normalization
  and synthesis's existing looser identity without changing persisted behavior.

- Isolate truncated WAV diagnostics from completed HTTP transcriptions and skip
  audio metadata inspection when STT debug logging is disabled.
- Protect real debug-log privacy, Unicode preview bounds and raw-preview gating
  with synthetic logging tests.

- Make the worker JSON recursion regression independent of Python build nesting limits.

- Isolate each expired staged-upload cleanup failure so healthy admissions continue;
  preserve failed rows for retry and log only upload IDs without private error details.

- Normalize worker-control JSON integer/nesting limit errors to safe WorkerError diagnostics.
- Cap local/remote known-speaker references at ten decoded minutes, checking WAV
  headers before PCM reads and float conversion without limiting ordinary job audio.
- Bound and discard reference decoder diagnostics at 64 KiB; kill/reap on overflow and retain
  deadline and temporary-file cleanup without exposing tool output.

- Decode known-speaker source ranges before PCM materialization, preserving 16 kHz
  sample rounding and EOF behavior. Reuse exact-URL source snapshots within local
  and remote jobs, including the main input, and clean reference-owned files once.
- Reload Kokoro sessions/voicepacks when artifact metadata changes, serialize cache
  publication, evict old entries and close NumPy archives on success and failure.

- Publish same-lease heartbeats monotonically, retaining the greater expiry and
  returning current lease data for delayed requests.
- Bound retention selection/results to five finite per-phase batches, with a JSON
  automatically saved database resume cursor that advances past failures/refusals; drain expired staged inputs
  during maintenance. Upload admission tries at most four expired inputs.

- Reject unrepresentable job numeric values and boolean synthesis speeds with
  controlled field-specific 400 responses before persistence.
- Stream producer/operator artifacts in bounded chunks on WSGI and ASGI and
  close readers with responses. Return bounded 404/503 storage errors before
  headers without leaking storage details; preserve artifact metadata for retry.

- Claim local tasks once for their expected task ID and fence terminal/artifact
  publication against stale execution. Keep committed terminal job outcomes
  authoritative during task backend reconciliation.
- Publish local artifact replacements under fresh allocation keys, preserving
  previous objects on metadata failures and reference-checking queued cleanup
  against each object's recorded store identity.

- Return a failing one-shot worker status after processing errors even when the
  failure was reported; validate worker timings and server polling hints before use.
  The server now requires VOXHELM_REMOTE_WORKER_POLL_SECONDS to be an integer
  from 1 to 3600 and refuses startup outside that range.

- Cancel active lease-heartbeat HTTP requests on claim exit and stop subsequent
  worker phases/publication after definitive lease or authorization loss.

- Reserve operator login capacity before password authentication. Failure windows
  and other in-flight guesses survive concurrent successful logins; pending
  checks retain their slots across expiry in the supported single-process scope.
  Inactive/non-staff denials and authentication exceptions consume the failure
  budget, including correct passwords for accounts without operator access.
- Mark private producer job/result/staging/artifact and operator transcript/page
  responses private/no-store, including errors and redirects, varying by bearer
  authorization or session cookies so cached successes cannot bypass revocation.

- Reject unsafe filesystem artifact keys and symlink aliases across store I/O,
  retaining fail-closed retention listing. Publish owner-only artifacts through
  flushed sibling temporary files and atomic replacement, preserving existing
  bytes on failure; create new artifact directories with mode 0700.

- Serialize staged-input expiry deletion with job submission and recheck expiry
  and claim state under the database lock, so stale cleanup candidates cannot
  delete a newly active job's input. Failed object deletions remain retryable.

- Enforce the batch staging byte ceiling while copying upload chunks, before
  object publication, and record the actual copied size even if the upload
  declares a smaller size. Over-limit copies remove their temporary file.

- Docs: `BACKLOG.md` ticks the items that are already done and cites the voxhelm,
  ops-library or ops-control commit for each (edge block of `/v1/internal/*`, PyPI
  packaging, the LaunchDaemon worker on `atlas`, the ops-library diarization, private
  URL host, prune job and Wyoming settings, and the ops-control Hugging Face token wiring
  and deploy check). It adds an open check: remote workers allowlist
  `macmini.tailde2ec.ts.net` but do not list it as a private URL host.
  `specs/remote-transcription-workers.md` no longer calls PyPI publication, deployment
  and edge protection pending. The README's Hugging Face access notes now name the
  known-speaker embedding model `pyannote/wespeaker-voxceleb-resnet34-LM`; its remote
  worker section says the edge blocks `/v1/internal/*` (it said the edge "must" block it),
  tells you to keep the worker env file at `chmod 600`, and shows the manual `uvx` run
  from the published PyPI package next to the checkout one.

- Security: the Wyoming sidecar now bounds the size of a single event. It reads events
  through its own reader, which checks the client-declared `data_length` and
  `payload_length` before reading the body; the `wyoming` library's reader buffered
  any declared size, so any client on the unauthenticated port could make the process
  allocate arbitrary memory. New settings `VOXHELM_WYOMING_MAX_EVENT_DATA_BYTES`
  (default 256 KiB) and `VOXHELM_WYOMING_MAX_EVENT_PAYLOAD_BYTES` (default: the STT
  audio cap in converted bytes, 3840000 for 120 s). Oversized events are answered with
  an `error` event `event_too_large`, malformed ones (negative or non-integer lengths,
  invalid header or data JSON) with `invalid_event`, and the connection is closed. The
  handler now also closes the connection when its loop ends. Because the reader
  replaces library internals, the `wyoming` requirement is narrowed to `>=1.8.0,<1.9`
  and a test asserts the locked version (1.8.0).

- Update the locked dependencies of the optional extras (mainly `diarization`) to fix
  their pip-audit findings: aiohttp 3.14.4 (with yarl 1.25.1), multidict 6.9.1,
  lightning 2.6.6, pytorch-lightning 2.6.6 and opentelemetry-api 1.45.1 (with the
  matching opentelemetry SDK, exporter and semantic-conventions packages), so the CI
  audit can include the extras. Lock-based installs (`uv sync --extra diarization`)
  get these versions. Remote workers started with `uvx --from ".[diarization]"` (as
  `just worker-once` and `just worker-loop` do) resolve without the lock and may reuse
  a cached tool environment; run them once with `uvx --refresh --from ".[diarization]"
  voxhelm-remote-worker ...` to pick up the fixed releases.
- Update locked dependencies to fix all pip-audit findings in the default lock export
  (76 known vulnerabilities in 12 packages): Django 5.2.17 (stays on the 5.2 LTS line),
  urllib3 2.8.0, requests 2.33.0, idna 3.15, sqlparse 0.6.0, setuptools 84.1.0,
  fsspec 2026.6.0, anyio 4.14.2, Pygments 2.20.0, pytest 9.0.3, and the major bumps
  filelock 4.2.1 and torch 2.13.0 (with triton 3.7.1 and the Linux-only NVIDIA wheels
  that torch pulls in). Raise the `Django` floor to `>=5.2.17` and the dev `pytest`
  floor to `>=9.0.3` so the declared ranges no longer admit the vulnerable versions.
  Running instances pick up the fixes with their next deployment.
- CI: add a `pip-audit` job that audits the locked dependencies including all optional
  extras (`uv export --locked --all-extras --no-hashes --no-emit-project`, checked with
  `pip-audit` 2.10.1)
  on every push and pull request, without ignores. The job runs on macOS arm64 like
  production, because pip-audit skips requirements whose environment markers do not
  match the runner and the Darwin/arm64-only dependencies (torch, mlx-whisper and
  their dependencies) would otherwise be left out.
