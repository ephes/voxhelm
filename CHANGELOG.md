# Changelog

## Unreleased

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
