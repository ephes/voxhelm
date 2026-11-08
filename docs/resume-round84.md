# Round 84 resume note

Work is parked on `fix/r84-b07-holder-safety`, based on `origin/main`
`ef4e7d1b92c019d3c57d21bc7bdab5a0737e0ac7`. Nothing was merged.
The fresh clone's origin is `git@github.com:ephes/voxhelm.git`.

## Completed implementation, review blocked

B07, in report order:

1. `voxhelm-lane-holder-publication-compensation`: compensate this attempt's
   holder under the control lock when publication/removal fails; post-admission
   logging is non-authoritative. Regressions cover a write that raises after
   creating the holder, waiter unlink failure, logging failure, another live
   holder's survival, retry and separate-process admission.
2. `voxhelm-lane-holder-read-errors-fail-closed`: distinguish missing/malformed
   records from operational read errors. PermissionError/EIO refuse admission
   without deleting live current or legacy holders.

All seven new parameter cases failed before the implementation. All 35 lane
scheduler tests pass afterward. README and Unreleased changelog are updated.
Full local validation results are recorded below.

The required review command used the installed `claude-review-loop` harness,
`--model claude-opus-5-5 --effort high`. It exited 2, PROVIDER_ERROR, with no
review performed: "Failed to authenticate: OAuth session expired and could not
be refreshed". There is no clean review verdict. The local artifacts are in
`../reviews/b07-r1/`; no findings were emitted. This checkpoint commit/push is
solely the owner's explicit exception for parking unfinished work.

## Resume sequence

Restore Claude OAuth outside this task, then fetch origin, rebase this branch
onto the latest `origin/main`, inspect any intervening changes, run `just check`
and complete the Opus review gate. Fix/re-review while making progress, at most
four rounds and stopping if a required finding recurs. Do not merge without a
passing review. Fetch immediately before merging, rebase and rerun checks if
origin moved, merge locally with `--no-ff`, run full local checks after merge,
push only main and verify with `git ls-remote`. No Actions fallback, PRs,
deploys, releases/tags, host access, work-app writes or remote branch deletion.

The remaining report slices were checked against current default-branch source;
the cited mechanisms remain present. No slice was skipped as already fixed.
They have no implementation, regression or review yet; re-verify each against
current trunk again when starting. Continue in bundle order:

- B08: `voxhelm-lane-cancel-during-control-lock-wait` then
  `voxhelm-lane-restart-pid-reuse-ownership`.
- B09: `voxhelm-speech-json-scalar-depth-admission` then
  `voxhelm-speech-public-error-diagnostic-boundary`.
- B10: `voxhelm-voice-discovery-unrelated-corruption-isolation` then
  `voxhelm-speech-streamed-file-response-lifetime`.
- B11: `voxhelm-speech-export-diagnostic-byte-ceiling` then
  `voxhelm-speech-export-owned-tree-deadline`.

Read each full description and validation notes in
`~/workspaces/ws-work-planning/reviews/scout-round84.md` before starting.
Synthetic fixtures only, platform-independent regressions that fail before the
fix, minimal public APIs, matching docs/Unreleased notes, no version bumps.

## Local validation

- `just check`: PASS (Ruff lint, Ruff format check, mypy over 123 source files,
  pytest: 1,588 passed, 6 skipped in 123.61 seconds).
- Focused `pytest tests/test_lane_scheduler.py -q`: 35 passed.
- Before implementation, all seven new parameter cases failed on the original
  source; no native inference or model downloads were used by these regressions.
- `git diff --check`: PASS.
- Optional-extras dependency audit was not run before parking. No Actions runs
  were queried, waited on or rerun.
