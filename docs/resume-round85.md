# Round 85 resume note

## B16 finisher outcome (2026-11-12)

Rebased onto `main` after the round-82 merge and reviewed together with the other
parked bundle (r84 B08 + r85 B16) by `claude-review-loop`
(`claude-opus-5-5`, effort high): no Critical or Warning findings. One suggestion,
declined: `transcriptions/audio_decode.run_audio_decode` resembles
`known_speaker._run_audio_decode`, but the known-speaker helper drains and counts
both pipes and supports an unlimited mode; unifying them would change that
established contract. `just check` passed (1,684 passed, 6 skipped); merged to `main`.
The remaining bundles listed below are still unstarted.

Work is parked on `fix/r85-b16-pyannote`, based on `origin/main`
`cab38107ca94e9f1e19b71e646f9a5c4cc2dbb5c`. Nothing was merged.
The fresh SSH clone's origin is `git@github.com:ephes/voxhelm.git`.

## B16 implementation; review blocked

The three B16 mechanisms remained present on current main; none was already fixed:

1. `voxhelm-pyannote-decoded-audio-admission`: cap decoding at two hours of
   16 kHz mono PCM plus one sentinel sample, validate WAV geometry/frame count
   before reading PCM, reject truncated bytes before waveform conversion.
2. `voxhelm-pyannote-decode-diagnostic-budget`: discard unused stdout and stderr;
   kill/reap at 64 KiB stderr or the existing elapsed decode deadline, clean scratch
   files and keep private output out of public errors.
3. `voxhelm-pyannote-initialization-publishes-after-device`: configure a local
   candidate and publish the cached pipeline only after successful device placement.

README and Unreleased notes are updated. New synthetic regressions ran against
unmodified source: 13 failures and 1 normal-input control passed. Fixed targeted
suite: 34 passed (`tests/test_pyannote_resources.py tests/test_diarization.py`).
No model downloads or GPU inference were used in these regressions.

The required `claude-review-loop` ran with `--model claude-opus-5-5 --effort high`
and exited 2 (`PROVIDER_ERROR`), before reviewing anything. Claude reported:
"Failed to authenticate: OAuth session expired and could not be refreshed".
There is no review verdict. Local artifacts: `../reviews/b16-round1/`.
This checkpoint commit/push uses the owner's explicit exception for parking
unfinished work, with `[skip ci]` to avoid starting paid Actions checks.

## Resume sequence

Restore Claude OAuth outside this task. Fetch origin, rebase this branch onto
latest `origin/main`, inspect intervening changes, run `just check` and the Opus
review gate. Run a fresh first review of the complete B16 change (no completed
baseline review exists). Fix/re-review while making progress, at most four rounds,
stopping if a required finding recurs. Do not merge without a passing review.
Fetch immediately before merging; if origin moved, rebase and rerun checks.
Merge locally with `--no-ff`, run full local checks after merge, then push only
main with CI-skipping commit text and verify using `git ls-remote`.
No Actions runs, PRs, releases/tags, deploys, host access, work-app writes or remote
branch deletion. Do not push further feature branches except to park unfinished work.

The OAuth stop rule stopped all further implementation. Remaining bundles are
unstarted and must each be re-verified against current main before work begins.
Read every full description and validation note in
`~/workspaces/ws-work-planning/reviews/scout-round85.md`; continue in this order:

- B17: `voxhelm-diarization-turn-finite-domain`,
  `voxhelm-speaker-alignment-interval-work-budget`,
  `voxhelm-chapter-native-word-time-domain`.
- B18: `voxhelm-remote-manifest-preflight-operation-budget`,
  `voxhelm-remote-artifact-model-field-domain`,
  `voxhelm-remote-artifact-storage-key-uniqueness`.
- B19: `voxhelm-staged-cleanup-failure-fairness`,
  `voxhelm-window-native-json-file-admission`,
  `voxhelm-normalizer-owned-process-diagnostic-budget`.

Keep fixes minimal; synthetic, platform-independent regressions must fail before
the fix. Update matching docs/Unreleased notes without version bumps. One review
per bundle is permitted. No Actions fallback or waiting for Actions.

## Local validation

- `just check`: PASS (Ruff lint/format, mypy over 125 source files; pytest:
  1,622 passed, 6 skipped in 124.80 seconds).
- Focused Pyannote/diarization suite: 34 passed.
- Pre-fix synthetic regressions: 13 failed, 1 control passed.
- `git diff --check`: PASS.
- Locked all-extras dependency audit: PASS, no known vulnerabilities.
- No Actions runs were queried, waited on or rerun.
