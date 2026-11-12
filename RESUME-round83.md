# Round 83 resume note

## B08–B10 outcome (2026-11-12)

The full scout report was not reachable from the implementing host, so the six
slices were implemented from their slugs after confirming each gap on `main`:
orphaned synthesis output after cancellation is deleted by its owner, outbound
writes have a 30-second deadline, open connections
(`VOXHELM_WYOMING_MAX_CONNECTIONS`, 16) and inference threads
(`VOXHELM_WYOMING_MAX_INFERENCE_THREADS`, 4, checked before the thread starts) are
capped, failure events no longer carry exception text (missing-voice messages
stay client-facing via `VoiceNotInstalledError`), and stop/cancellation aborts and
cancels every handler. Regressions: `tests/test_wyoming_lifecycle.py`.
Three Opus review rounds: round 1 found a missing-voice regression (fixed) and a
shutdown-order concern based on a newer wyoming `stop()` (wyoming 1.8.0 does not
await `wait_closed`; shutdown was still reordered and covered with real TCP
tests); rounds 2–3 were suggestion-only and applied. Round 83 is complete.

## B07 finisher outcome (2026-11-08)

Claude finisher reviewed the whole parked B07 diff against `ef4e7d1`. One finding,
fixed: `AudioStart`/`AudioChunk` admitted 3-byte samples, but the bundled Wyoming
converter (`pyaudioop`, Python 3.14) only supports 1, 2 and 4 bytes, so a 24-bit
stream raised inside conversion instead of a clean `invalid_event`. Width 3 is now
rejected at admission (regression added). A fuzz of 3000 random rate/width/channel
chunkings confirmed the cumulative `ceil(N * 16000 / rate)` bound never undercounts
the converter output. Re-review clean; merged to `main`. B08–B10 remain unstarted.

## Current continuation: B07 parked on 2026-11-08

Branch: `fix/r83-b07-utterance-geometry`, based on current default `main`
`ef4e7d1b92c019d3c57d21bc7bdab5a0737e0ac7` in a fresh SSH clone at
`/Users/jochen/workspaces/ws-cx-r83c-voxhelm/voxhelm`. Origin verified as
`git@github.com:ephes/voxhelm.git`. B06 is already merged (see earlier history below).

Reverified all three B07 slices against that main revision; none was already fixed.
Implemented in report order:

- `voxhelm-wyoming-chunk-shape-binding`: require AudioStart before chunks,
  matching rate/width/channels, and whole frames; refuse before conversion/inference.
- `voxhelm-wyoming-preconversion-frame-budget`: admit integer 8000–192000 Hz,
  1–4 byte samples, mono/stereo; compare the cumulative resampled frame upper
  bound with the utterance ceiling before converter entry.
- `voxhelm-wyoming-utterance-resampler-reset`: reset converter, audio buffer and
  shape at each Transcribe/AudioStart; retain converter state between chunks.

README, BACKLOG and CHANGELOG Unreleased match the implementation. Fourteen
synthetic regressions failed against unchanged main production code, including
byte-for-byte resampler restart checks. Five additional accepted boundary cases
cover 8000, 16000, 44100, 48000 and 192000 Hz. The Wyoming suite passed 87 tests.
Full local validation passed: `just check` completed with Ruff lint/format clean,
mypy clean on 123 source files, and **1600 passed, 6 skipped** in pytest.
The skips are existing optional model/dependency cases; no models were downloaded.
The locked all-extras dependency audit passed with no known vulnerabilities.

The required Opus review was attempted with `--model claude-opus-5-5 --effort high`.
The harness returned exit 2 / PROVIDER_ERROR / api_error; its provider text was
"Failed to authenticate: OAuth session expired and could not be refreshed".
No review verdict or findings exist; no merge was attempted. The review copy was
removed and the manifest reports no skips, truncations or redactions. Artifacts:
`../.cache/b07-review-1/`, `../.cache/b07-check.log`, `../.cache/b07-audit.log`.

The owner explicitly required parking the branch with this note and stopping on
OAuth failure. This parking commit is unreviewed and must not be merged yet.
B08–B10 remain entirely unstarted and were not individually reverified in this run.
Resume them in report order only after finishing B07.

After restoring OAuth, review the **entire B07 diff** from the main base above.
Because the work is now committed for parking, a default HEAD-only review would
be empty: use `--baseline-ref ef4e7d1b92c019d3c57d21bc7bdab5a0737e0ac7` with a
fresh run directory, `--record-baseline`, the same Opus model/effort and slice ID
`r83-voxhelm-b07`. State explicitly that this is the first valid review of the
whole parked bundle. Scope subsequent repair reviews to each recorded baseline.
Follow the owner's four-round/recurring-finding stop rules.
Before a clean merge, fetch and rebase if origin/main moved, then rerun checks;
merge with --no-ff, run full local checks after merge, push main, verify with
`git ls-remote`. Use `[skip ci]` in commits to suppress paid Actions. Do not
run/wait for Actions, open PRs, deploy, release/tag, or delete remote branches.

## Earlier B06 parking history (superseded by its finisher outcome below)

Parked at the owner's explicit OAuth-failure stop condition. **Do not merge
without a completed Opus review.** Origin is `git@github.com:ephes/voxhelm.git`;
integration/default branch is `main`. Fresh clone began at
`867f9c8168c0eda29cf3721bb723aa22da3e667f`, matching the scout baseline.

Source report: `~/workspaces/ws-work-planning/reviews/scout-round83.md`.
All eleven full slice descriptions, bundle ordering, cautions and validation
notes were read. B06's two gaps were reverified on current `origin/main` and
implemented; no slice was found already fixed. Remaining bundles have not been
individually reverified or implemented.

## Implemented B06, awaiting review

1. `voxhelm-wyoming-typed-event-admission`: consumed synthesis, transcription,
   voice and audio scalar fields checked before parsing/state changes/inference;
   malformed events receive one `invalid_event` response, then close.
2. `voxhelm-wyoming-event-elapsed-deadline`: one fixed 30-second asyncio timeout
   wraps the entire bounded event read, including header, data and payload.
   Timeout returns `event_timeout` and closes the client.

README, BACKLOG and CHANGELOG Unreleased are updated. Twelve malformed-event
cases and four fake-clock stalled/trickled read cases all failed against the
unchanged baseline; all 68 Wyoming tests pass after implementation, including
existing inclusive byte-boundary and normal protocol round trips.

Full local validation passed: `just check` completed with Ruff lint and format
clean, mypy clean on 123 source files, and **1581 passed, 6 skipped** in pytest.
The six skips are the existing optional model/dependency cases; no model was
downloaded for testing. The locked all-extras pip-audit passed with no known
vulnerabilities. Its tool installation needed workspace-local `UV_TOOL_DIR`
in addition to `UV_CACHE_DIR`; the default tool directory was sandbox-blocked.

Review attempt: the installed `claude-review-loop` harness was invoked with
`--model claude-opus-5-5 --effort high`. It returned exit 2, `PROVIDER_ERROR`,
before any review/tool use: **OAuth session expired and could not be refreshed**.
No review verdict or findings exist. Local artifacts live one directory above
this clone in `review-b06-r1/`, `b06-check.log` and `b06-audit.log`.

## Finisher procedure

Rerun the B06 review using the installed harness in a fresh run directory.
Review only this bundle, keeping B07-B10 separate. Fix/re-review while findings
shrink; stop at four rounds or recurrence of the same Critical/Warning. This
branch is intentionally committed only to preserve unfinished work under the
owner's explicit parking exception, not as a claim that the review gate passed.

Once review is clean or suggestion-only with dispositions recorded, fetch
immediately before merge. If `origin/main` moved, rebase onto it and rerun
checks. Merge locally with `--no-ff`, run full local checks again, push only
`main`, verify with `git ls-remote`. Never delete remote branches. Use
`[skip ci]` in parking/merge commit messages to avoid paid Actions runs.
No PRs, deployments, releases, tags, host access or work-app writes.

Continue in this report order, rechecking each slice against fresh trunk:

- B07: `voxhelm-wyoming-chunk-shape-binding` →
  `voxhelm-wyoming-preconversion-frame-budget` →
  `voxhelm-wyoming-utterance-resampler-reset`.
- B08: `voxhelm-wyoming-cancelled-synthesis-result-cleanup` →
  `voxhelm-wyoming-outbound-write-deadline`.
- B09: `voxhelm-wyoming-active-connection-admission` →
  `voxhelm-wyoming-prethread-inference-admission`.
- B10: `voxhelm-wyoming-failure-diagnostic-redaction` →
  `voxhelm-wyoming-server-cancellation-transport-cleanup`.

Keep synthetic fixtures and platform-independent tests; require each new
regression to fail on its own pre-fix baseline. Gates: `just check` (Ruff,
format, mypy, pytest), plus the locked dependency audit described in README.
Point `UV_CACHE_DIR` into the workspace `.cache/uv` when necessary. Do not use
or wait for GitHub Actions. No work under `~/projects` is authorized.

## Finisher outcome (round 84)

B06 reviewed by the Claude finisher (Opus, one round): no Critical/Warning
findings. Suggestion applied: typed-event rejections are now logged like
read-level rejections. Checks rerun outside the sandbox before the `--no-ff`
merge to `main`. B07-B10 remain open as listed above.
