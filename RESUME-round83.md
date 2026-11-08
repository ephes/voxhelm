# Round 83 resume note

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
