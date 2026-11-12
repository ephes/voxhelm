# Round 84 B08 resume note

## B08 finisher outcome (2026-11-12)

Rebased onto `main` after the round-82 merge and reviewed together with the other
parked bundle (r84 B08 + r85 B16) by `claude-review-loop`
(`claude-opus-5-5`, effort high): no Critical or Warning findings. One suggestion,
declined: `transcriptions/audio_decode.run_audio_decode` resembles
`known_speaker._run_audio_decode`, but the known-speaker helper drains and counts
both pipes and supports an unlimited mode; unifying them would change that
established contract. `just check` passed (1,684 passed, 6 skipped); merged to `main`.
The remaining bundles listed below are still unstarted.

Branch: `fix/r84-b08-lane-cancellation-ownership`.
Base: `main` at `cab38107ca94e9f1e19b71e646f9a5c4cc2dbb5c` (fresh SSH clone).

## Gate and next action

Not reviewed; do not merge yet. The requested `claude-review-loop` invocation
with `--model claude-opus-5-5 --effort high` exited 2, `PROVIDER_ERROR`:
"Failed to authenticate: OAuth session expired and could not be refreshed".
No review findings or passing verdict were produced. No merge, default-branch
push, Actions invocation, release or deployment was performed.

Local artifacts (outside Git):
`/Users/jochen/workspaces/ws-cx-r84c-voxhelm/reviews/b08-r1/` contains the
result and logs; `b08-review-context.txt` at the workspace root records scope.
The harness reported no skipped, truncated or redacted files and removed its
throwaway review copy.

After authentication is restored, fetch current `origin/main`, re-verify this
bundle, and rebase if it moved. For a committed parked branch, review the whole
bundle against current trunk using `--baseline-ref origin/main` (without that
option the harness would review only uncommitted changes). Use a fresh run
directory and record a baseline for subsequent delta reviews. Follow the owner's
four-round/repeated-finding stop rule; merge only with a passing gate. Immediately
before merge fetch again, rebase if needed and rerun local checks. After a
`--no-ff` merge run full local checks before pushing `main`; avoid Actions.

## Implemented bundle B08

- `voxhelm-lane-cancel-during-control-lock-wait`: cancellable nonblocking flock
  retries cover initial waiter registration and queue scans. Failed/cancelled
  attempts remove only their UUID waiter without reacquiring the control lock.
- `voxhelm-lane-restart-pid-reuse-ownership`: new tickets retain an exclusive
  flock on their holder JSON inode until release. Marked abandoned records are
  reclaimable despite a reused PID. Unmarked older records retain PID liveness,
  and age remains warning-only. Documentation and Unreleased notes match.

Both bugs were reproduced on unchanged trunk. Four new regression cases fail
there and pass with the fix: two externally-held control-lock cancellation
phases, real abrupt child exit plus simulated unrelated PID reuse, and an old
live lock-owned holder protected independently of its numeric PID. No platform
PID range or actual PID reuse is assumed. Existing cross-process capacity and
legacy-record tests remain intact.

Validation: `just check` passed (Ruff, formatting, mypy; 1,612 passed, 6 optional
model tests skipped). Lane suite: 39 passed. All-extras locked dependency audit
with pip-audit 2.10.1: no known vulnerabilities. A final lint/format/mypy pass
also succeeded. The uv tool cache for the audit was redirected to the workspace
because its default tools directory was sandbox-blocked.

## Remaining work, in report bundle order

B07 publication compensation/read-error admission is already merged; current
trunk contains those fixes and their tests. Do not reimplement it.

The OAuth stop rule left these bundles unstarted and unverified by this worker:

1. B09: `voxhelm-speech-json-scalar-depth-admission`, then
   `voxhelm-speech-public-error-diagnostic-boundary`.
2. B10: `voxhelm-voice-discovery-unrelated-corruption-isolation`, then
   `voxhelm-speech-streamed-file-response-lifetime`.
3. B11: `voxhelm-speech-export-diagnostic-byte-ceiling`, then
   `voxhelm-speech-export-owned-tree-deadline`.

Re-verify each against the current default branch before implementing. The scout
report is `~/workspaces/ws-work-planning/reviews/scout-round84.md`. Another
worker may update trunk; use fresh fetch/rebase checks before every merge. No
PRs, host access, work-app writes, releases/tags or paid Actions. Parked branch
pushes are the owner's explicit exception to the default feature-push ban.
