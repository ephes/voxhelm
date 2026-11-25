# Changelog

## Unreleased

- Return a Kokoro input error when a sentence with letters or digits produces
  no supported tokens, so mixed requests cannot silently omit it.

- Reject Kokoro synthesis when the model or voicepack changes while the request
  binds the pair, before running inference.

- Optional recording of Wyoming STT turns for model comparisons:
  `VOXHELM_WYOMING_STT_RECORD_DIR` (empty = off, the default) keeps the
  newest `VOXHELM_WYOMING_STT_RECORD_MAX` (40) turns in its own namespace as 16 kHz WAV plus a
  JSON with transcript, backend and language, owner-only. Meant for short
  test windows; household speech, delete afterwards.

- Track Wyoming warm-up work through bounded listener shutdown.
- Warm the Wyoming Parakeet fallback even when the primary model fails to load.
- Keep the first Parakeet transcript when the Wyoming language retry backend is unavailable.
- Reload resident Parakeet when the installed model generation changes.
- Measure Wyoming TTS loudness off the listener loop and stream with a single gain.
- Validate Wyoming TTS RMS targets as finite values from -96 to 0 dBFS.
- Move optional Wyoming recordings to a bounded background writer.
- Record raw and served Wyoming STT text and requested and result languages for comparisons.
- Keep Wyoming transcripts available when the recorder or its failure logging fails.
- Cap retained Wyoming recording audio and metadata at 160 MiB by default.
- Publish Wyoming recording audio and metadata together and recover interrupted staging.

- Continue draining queued artifact deletions when diagnostic logging fails
  and keep unresolved entries eligible for retry.
- Index protected filesystem objects by device and inode once per artifact
  replacement, preserving shared objects while queueing obsolete intermediates.

- Recheck inference cancellation after the lane control lock exits and retire
  the pending holder if the request was cancelled before ticket handoff.
- Preserve a replacement lane holder receipt on ticket release, close the
  original handle, and report the identity mismatch.

- Wyoming STT with `parakeet-mlx`: when the transcript is not in the
  requested language (de/en, judged with the TTS routing detector), the
  turn is transcribed again with mlx-whisper forced to that language.
  Parakeet detects the language itself and turned short German follow-ups
  into English ("Uh, that's linking.", "Yep."), and Home Assistant's agent
  then answered in English. The warm-up now also loads mlx-whisper.

- Wyoming TTS answers are scaled to an RMS level of
  `VOXHELM_WYOMING_TTS_TARGET_DBFS` (default -19 dBFS, at most +/-8 dB,
  peaks below about -1 dBFS; empty disables it). Piper Thorsten, Kokoro
  Martin and Kokoro af_heart differed by about 6 dB.

- Add the in-process Parakeet backend `parakeet-mlx` for the Wyoming
  sidecar (`VOXHELM_WYOMING_STT_BACKEND=parakeet-mlx`, new `parakeet` extra
  with `parakeet-mlx==0.5.3`, model from `VOXHELM_PARAKEET_MODEL_DIR`). The
  model loads once and stays resident, so a voice command takes about 0.1 s
  on the Studio instead of 0.4 s with mlx-whisper large-v3. If the package,
  the model directory or the language is not available, the turn falls back
  to mlx-whisper (`VOXHELM_MLX_MODEL`). The batch `parakeet` CLI backend is
  unchanged.

- Warm up the Wyoming STT model right after the listener starts
  (`VOXHELM_WYOMING_STT_WARMUP`, default on): one second of silence goes
  through the configured backend on the interactive lane and the transcript
  is discarded. The first Home Assistant voice command after a restart took
  about 12 s with mlx-whisper large-v3 because the model loaded lazily. A
  failed warm-up is logged and the listener keeps serving.

- Retire a pending inference holder if the control lock fails to exit before
  the caller receives its ticket.

- Reject nonregular inference control locks and refuse stale lock generations
  before accessing the lane queue.

- Select the next admissible inference waiter in one pass and skip queue
  reads while all inference slots are occupied.

- Bound lane scheduler receipt reads to 64 KiB, reject symlinks and nonregular
  files, and preserve holders whose state or ownership cannot be verified.

- Close URL fetch HTTP error responses without reading their bodies and retain
  the original status and retry classification if close fails.

- Preserve the original fetch failure or cancellation when temporary media
  removal fails and log a fixed cleanup warning.

- Reject unsupported remote transcript output formats before downloading or
  processing the input. Claims that omit formats still use text and JSON.

- Recheck remote lease ownership after rendering each transcript format and
  stop before upload if the lease was lost.

- Skip Piper configs that change during discovery and refuse synthesis with
  stale config metadata before loading or reusing a model.

- Skip nonregular Piper models during discovery and reject them before native
  loading. Model symlinks to regular files still work.

- Close speaker evaluation fingerprint descriptors when a file is rejected or
  the read stream cannot be opened.

- Require a nonempty regular output after ffmpeg finishes an MP3 or Ogg export
  and clean up the owned output on rejection.

- Complete remote jobs and omit optional numeric diagnostics that exceed the
  finite double-precision range, including oversized JSON integers.

- Reject lone Unicode surrogate escapes in batch and worker-control JSON before
  fingerprinting or state changes, while accepting valid supplementary characters.

- Invalidate the window transcription CLI version cache when the executable is
  replaced, even if its modification time is preserved. Discard versions probed
  while the executable changes.

- Retry failed window transcription CLI version probes on the next call and
  cache only successful versions.

- Keep IPv6 endpoint addresses and ports distinct when retention compares S3
  object references.

- Resolve filesystem retention roots afresh for each deletion and reuse that
  root for admission, reference checks and object removal.

- Read operator bootstrap passwords from stdin with `--password-stdin`,
  preserving whitespace apart from the line terminator.

- Validate operator bootstrap usernames and nonblank emails before hashing
  credentials or looking up the account. Blank emails preserve the existing address.

- Remove incomplete synchronous transcription upload copies after copy or close
  failures and preserve the original error if cleanup also fails.

- Preserve native transcription errors and add a fixed diagnostic when writable
  scratch removal also fails after confirmed process retirement.

- Limit expired final attempts reconciled per worker claim request to 16, then
  scan for runnable work even when the expired backlog has rows left.

- Commit exhausted remote lease failure and staged upload claim release together
  so a failed claim release can be retried.

- Persist retry order for default queued artifact deletion drains so refused or
  failed entries cannot prevent later entries from being attempted.

- Apply retention key ownership and shared-reference checks during staged upload
  admission and submission cleanup.

- Speak only valid 24-hour clock values in German synthesis, with `00:00` as
  midnight. Preserve out-of-range and multi-component colon notation.

- Validate German synthesis dates against the Gregorian calendar and keep
  invalid date candidates literal. Spoken years stay limited to 2000–2099.
  Named-month dates now end at any non-word character, so `3. Mai)` or
  `3. Mai;` is read as a date.

- Return retryable HTTP 503 for malformed S3 errors during artifact downloads.

- List each exposed operator transcript artifact with a name-specific download
  link. Keep format-only links when they identify a single exposed artifact.

- Require matching S3 `ContentLength` metadata for recorded artifact downloads.
  Reject missing or invalid lengths with HTTP 503 and close the response body.

- Close the filesystem artifact reader if closing its parent directory fails
  before handoff, preserving the directory error.

- Preserve local transcription outcomes when cleanup fails and continue
  cleaning the remaining resources.

- Return `not_claimed` when a local batch task loses its claim before the
  final success update.

- Count speaker track support in one pass without copying frames.

- Normalize the probe once per known-speaker classification.

- Give a single ASGI stream traversal exclusive ownership of the reader and its
  cleanup.
- Validate maintenance cursor overrides and batch sizes before creating shared
  artifact-retention progress.

- Preserve the primary streaming reader error when close and cleanup logging
  both fail.

- Normalize VTT, DOTe and Podlove cues as they are rendered after full timestamp
  validation, avoiding an intermediate list of cue wrappers.
- Check cancellation during drift word grouping and unflagged window scans.
- Reuse ordered drift words when no retry is accepted, retaining stable
  start-time order.

- Keep in-flight operator login reservations when unrelated cache writes evict
  cached failure counts.
- Refuse native transcript output when its pathname no longer names the opened
  file generation after the bounded read.
- Measure WAV size and format from the same open regular file in opt-in
  transcription diagnostics.
- Report fixed categories in audio diagnostic error fields to keep local
  paths and parser messages out of the logs.
- Restrict the larger ASGI body ceilings and upload pools to POSTs. Small
  requests using other methods continue to reach Django's method checks.
- Reserve a pool slot when ASGI first delivers unannounced upload bytes.
  Staged uploads still require a usable length for a body.
- Compare declared lengths with the ASGI route ceiling before converting them
  to integers, so oversized lengths get a 413 and leading zeros are accepted
  by the admission check.
- Match Django's ASGI script prefix when choosing request body limits and pools,
  including deployments that set `FORCE_SCRIPT_NAME`.
- Apply the same single-credential bearer policy during ASGI upload admission
  and producer authentication. Repeated or comma-joined Authorization fields
  keep the default body ceiling.
- Check the opened filesystem or S3 artifact size against the recorded size
  before streaming. Close mismatched readers and return HTTP 503.

## 0.1.15 (2026-11-18)

Published to PyPI for the remote transcription workers (`atlas`, `studio`),
which run 0.1.14 at most until now. It carries the changes merged since 0.1.14,
above all the long-recording defaults: a batch URL download may now be up to
16 GiB (`VOXHELM_BATCH_MAX_DOWNLOAD_BYTES`, was 512 MiB) and each try may take
3600 s (`VOXHELM_BATCH_URL_FETCH_TIMEOUT_SECONDS`, was 600). The workers pick
it up only after the ops-control worker pin moves to 0.1.15 and they are
redeployed; a host or env override of either setting still wins.

- Let long recordings through. Staged uploads
  (`VOXHELM_BATCH_MAX_STAGED_UPLOAD_BYTES`) and batch URL downloads
  (`VOXHELM_BATCH_MAX_DOWNLOAD_BYTES`) now default to 16 GiB instead of
  512 MiB, and a batch URL download try may take
  `VOXHELM_BATCH_URL_FETCH_TIMEOUT_SECONDS` = 3600 seconds by default instead
  of 600 (remote workers pick both up with their next release).

- Time ASGI request-body receipt before routing. Each chunk must arrive within
  `VOXHELM_ASGI_BODY_IDLE_TIMEOUT_SECONDS` (default 120), and the whole body
  within the larger of `VOXHELM_ASGI_BODY_TIMEOUT_SECONDS` (default 600) and its
  declared `Content-Length` at `VOXHELM_ASGI_BODY_MIN_BYTES_PER_SECOND`
  (default 128 KiB/s), so a multi-gigabyte producer upload over a slow link
  completes while anonymous and JSON routes keep the 600-second total. Stalled
  and drip-fed uploads get a JSON 408 after spool cleanup. Only a
  `Content-Length` that frames the body counts: beside `Transfer-Encoding` it
  does not extend the budget (an oversized one still gets the early 413). A
  producer's staged upload that announces a body without a usable length gets
  a JSON 411 before any body is read, since Django cannot parse its multipart
  form. Inference after a completed upload has its own lifetime.

- Bound concurrent large-upload spools per ASGI process, including cleanup
  after disconnect or cancellation. Body-carrying requests to the operator form,
  to synchronous transcription and to staged uploads each use their own pool of
  `VOXHELM_ASGI_MAX_ACTIVE_UPLOADS` slots (default four per route) for the full
  Django request lifetime, and get a JSON 503 without a body read when their
  pool is full. Health, the job API and the remote worker control plane are
  never counted.

- Bound request bodies at the ASGI entry point before Django spools them: Django
  copied every body to a temporary file before routing or authentication, so any
  client reaching the port could stream an unbounded body to disk without
  credentials. Large bodies are now admitted only on the upload routes (staged
  uploads and sync transcription with a valid producer token, the operator form);
  everything else is capped at 3 MiB. Oversized bodies get a JSON 413 with
  `Connection: close` (and the app's `X-Frame-Options: DENY` and `nosniff`), a
  declared oversized `Content-Length` before any byte is read.

- Docs: the README no longer claims the Traefik edge replaces `X-Frame-Options`
  with `SAMEORIGIN`; the ops-library edge sends `DENY` since 2.31.18.

## 0.1.14 (2026-11-17)

Published to PyPI for the remote transcription workers (`atlas`, `studio`),
which run 0.1.13 from PyPI until now. It carries the changes merged since
0.1.13, including WebVTT cue whitespace normalization for `remote_pull` jobs,
short-segment speaker naming from episode voiceprints and bounded native
process cleanup. The workers pick it up only after the ops-control worker pin
moves to 0.1.14 and they are redeployed.

- Gate migration drift locally: `just check` now runs `just check-migrations`
  (`makemigrations --check --dry-run` under `tests.settings`), so a model change
  without its committed migration fails before merge; a test keeps the recipe
  wired into `just check` and asserts no migrations are pending.

- Preserve speech responses and primary errors when temporary-file cleanup fails;
  attempt both removals independently and close readers on response setup failure.
  Fixed warnings identify cleanup recovery without exposing file/error details.

- Normalize WebVTT cue whitespace in linear work, preserving ordinary interior
  spacing and existing escaping while collapsing CR/LF whitespace runs.

- Atomically publish standalone speaker reports through owned sibling temporary
  files; failed writes/replaces preserve the previous complete report. Reports
  keep an existing file's mode (new ones use the umask default), and only
  refused admission becomes an argparse error; runtime failures keep tracebacks.

- Protect standalone speaker-evaluation source/control files from report output,
  including resolved-path, symlink and hardlink aliases; recheck at publication.

- Report explicit target-passage coverage in known-speaker evaluations; absent
  or partial target evidence is unassessed (`null`), never an automatic pass.

- Admit bounded stable known-speaker gold JSON, exact finite segment records
  and nonnegative limits before backend construction or reference inference.

- Preflight mixed-speaker subwindow expansion with a 10,000-inference budget
  and representable integer-index progress before acquiring a model.

- Validate mixed-speaker probe intervals and decoded source extents before
  reference embedding; retain exact endpoints and short valid segment padding.

- Attempt every URL-fetch watchdog socket close and always restore the caller's
  deadline context. Preserve primary failures and cancellation; otherwise refuse
  the downloaded media with a fixed cleanup error and value-free diagnostics.

- Require complete, nonempty HTTP 200 URL media for sync and batch inputs;
  reject unsolicited ranges and empty responses before allocating media files.
  Preserve redirect guards and retries for bodies shorter than Content-Length.

- Reuse unchanged offline speaker source/model/code fingerprints within each
  suite run. Shared model roles and episodes hash once; preflight and pre/post
  replay identity checks still run, but for an unchanged file generation they
  now compare device, inode, size, mtime and ctime of the opened file and its
  name, and rehash only when that generation changes.

- Keep offline speaker publisher locks bound to owned, single-link named files
  and held parent directories; refuse detected replacement before bundle or
  suite publication, preserving the previous completed tag. A bundle
  destination's parent or a suite `results/` directory that is itself a symlink
  is now refused; point the paths at the real directory instead.

- Refuse changing or incomplete native JSON file generations before parsing or
  publishing local, window, chapter-window or remote transcript output.

- Reject nonfinite or negative coverage policies before native work and invalid
  measured durations at the shared coverage boundary; zero still disables gap rejection.

- Hold and verify chapter model generations around successful inference; require JSON output after zero exit without DTW retries.

- Detect sync URL media suffixes from the path and reject ambiguous, deeply nested or oversized sync transcription JSON before fetching.

- Bound profile revision snapshots and supervise cancellable version probes with a small output limit.
- Retire pyannote decoder descendants on every exit with finite cleanup; preserve primary failures and retain scratch on uncertain retirement.

- Capture profile provenance before inference, reject changed runtime generations and refresh CLI environment versions within 60 seconds.

- Validate profile token confidence and temporal ordering before word publication; assemble long words in linear work.

- Seal frozen numeric samples and require exact baseline bundle/truth identities
  for variants; admit constructor fields and decode/sanitizer generation settings.

- Correct pp62 evaluation to report aligned correct/wrong/unnamed/uncovered speech
  seconds, accepted public names and interval-union coverage. Retain the historical
  whole-segment score under a legacy name; remove misleading DER/primary keys.

- Publish complete deterministic suite generations with exact member/input hashes
  and one same-tag publisher; retain previous results and failed recovery stages.
  Validate portable tags before work and keep variants on the baseline embedding model.

- Publish frozen speaker bundles only after complete staged capture. Version 2
  records source/model generation hashes and refuses stale suite reuse; local
  model artifact lists are now required for capture. Standalone v1 replay remains.
  The code identity covers only capture-path modules, so postprocessing edits
  still replay the same bundles; job audio over the two-hour Pyannote bound is
  refused before ASR, and control timestamps admit episodes up to 24 hours.

- Admit bounded strict offline controls and contained frozen NumPy members;
  validate array geometry, aggregate work and finite samples before embeddings.

- Count all clear truth frames in offline speaker scores, expose coverage, and
  treat conflicting overlapping names as unnamed independent of cue order.

- Report incomplete whisper.cpp output as an error (archive item 408: with
  `-p 4`, whisper.cpp 1.9.5 merged three failed parts silently and exited 0).
  Each whisper.cpp run checks its raw segments against the normalized WAV's
  duration; a stretch without text longer than
  `VOXHELM_WHISPERCPP_MAX_GAP_SECONDS` (default 600, 0 disables) raises
  `IncompleteTranscriptError`, which fails closed without backend fallback
  (remote: retryable attempt; local job: failed; sync endpoint: HTTP 500).
  whisper-cli no longer runs with `-np`, so its library errors are kept;
  error-looking stderr lines are logged, also on exit 0, and a failed run's
  error carries them instead of stdout. Results record
  `media_duration_seconds` and `covered_seconds` in `result_metadata`, which the
  control plane now accepts from workers.

- Limit native-child reaping to a two-second cleanup budget and classify denied
  termination explicitly, including groups whose leader already exited. Retain
  writable scratch/partial exports when retirement is uncertain; never accept
  their output as successful. Preserve the harmless macOS zombie-only case by
  confirming group disappearance after reaping.

- Independently attempt every owned native-process cleanup stage, preserving
  primary read/timeout/cancellation failures and refusing success after cleanup
  errors; report secondary failures with fixed stage diagnostics.

- Close owned download readers once when response/header construction fails;
  preserve read/cancellation errors if closure also fails, with bounded private
  cleanup diagnostics and no close/read race.

- Decide cleanup outcomes from each run's own failure instead of
  `sys.exception()`, so a native run or reader close started while a caller is
  handling an unrelated exception still fails closed on cleanup errors and does
  not tag the caller's exception with the ownership note.

- Name reserved speech export files `voxhelm-speech-export-*` so retained
  partial exports can be identified during operator recovery.

- Include untied pyannote-label episode profiles for guests without reference
  audio. Preserve original overlap labels on relabelled short cues and expose
  ranked episode candidates; add offline guest/profile sanity diagnostics.

- Expose `diarization.known_speaker.short_segment_margin` (default 0.15,
  0 disables) through shared API/worker validation and summary metadata.

- Classify remaining 0.3–2.5 s uncertain cues against episode profiles from
  confident long segments; tile short audio and keep the mixed-speaker veto.
  The pass is skipped (`episode_voiceprint_skipped: "no_raw_turns"`) when the
  anonymous diarization produced no usable turns, since an unreferenced guest
  could not be detected then.

- Add repeatable offline speaker bundles and a six-episode baseline/variant runner.
  Capture both annotations without changing production turn selection; bundles
  pass the manifest's `num_speakers`/`min_speakers`/`max_speakers` to pyannote
  like the production request.

- Add offline speaker naming scores by frame, line duration and acceptance rule,
  including the consumer guest rule and baseline line comparisons.

- Quantize VTT/DOTE/Podlove cues once to integer milliseconds and ensure a
  minimum 1 ms duration, preserving verbose raw timing. Retain original drift
  words on empty/fully filtered retries and record an explicit rejection.

- Propagate WhisperKit caller cancellation through upload/header/body waits,
  closing owned resources and rejecting late success without fallback. Keep
  sidecar failure responses and logs private with fixed categories/status only.

- Stream WhisperKit responses under separate transcript/error byte limits and
  strict object JSON admission, with one elapsed deadline spanning connect,
  upload, headers and body; resource failures do not trigger backend fallback.

- Bound retained native inference stdout/stderr to 64 KiB (head and tail of each
  pipe; long whisper.cpp segment output is truncated, not refused) with
  owned process-group cleanup, also when a descendant keeps a pipe open after
  the child exited; admit standalone WhisperCpp transcript files
  through a regular no-follow descriptor and strict bounded object JSON.

- Update the locked OpenTelemetry API/SDK to 1.45.2 (semantic conventions
  0.66b2), clearing the existing dependency-audit finding CVE-2026-126360.

- Bound batch/worker JSON bytes and nesting, refuse duplicate members and
  nonfinite numbers, and normalize decoder resource failures before writes.
  The remote worker refuses a nonfinite request value locally before sending.
- Return fixed private batch backend and staging compensation diagnostics while
  preserving cleanup and retry records; cleanup failure logs name the staged id.
- Hash operator passwords before publication and commit each complete account
  update under a writer lock, with explicit retry on first-creation conflicts.
- Publish retention continuation by generation compare-and-swap so stale
  maintenance runs cannot replace newer progress, including explicit cursors.
- Hash only stable regular opened window models, invalidate cached hashes on
  change metadata and bind native window invocation to the checked descriptor.
  A symlinked configured model is resolved once to its regular file.
- Return owner-scoped 404 when a polled job disappears during reconciliation.

- Write Piper audio incrementally with 30-minute and 128 MiB PCM ceilings;
  require stable positive mono/stereo 16-bit geometry and a complete nonempty
  WAV before returning metadata. Invalid or failed output is cleaned up.

- Reject ambiguous case-insensitive voice aliases while preserving exact keys,
  defaults and pinned routing. Preserve synthesis/conversion failures and
  cancellation if temporary-file cleanup fails, logging a fixed recovery warning.

- Bound installed Kokoro style metadata and expanded array size before NumPy
  loading, matching exact ZIP key resolution; refuse changed ONNX, voicepack and
  Piper model/config generations before cache publication, allowing a stable retry.

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
