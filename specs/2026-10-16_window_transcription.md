# Window transcription (whisper large-v3-turbo on one clip)

Status: 2026-10-16, implemented and verified locally (see "Verification");
not deployed. Used by the `ad-analysis-v1` drift check since slice 3
(`2026-10-16_drift_check.md`, D-28). Decision: D-27. Source: slice 2 of
the podcast server-side ad analysis,
`~/projects/podcast/docs/player/specs/server-ad-analysis-design.md`
("Voxhelm Adapter Contract", row "Drift check with turbo retry", part (a);
Slice Plan row 2). Builds on `2026-10-16_ad_analysis_profile.md` (D-26).

## Why

Parakeet drifts into English inside German speech in about 8-10 % of German
30 s windows. The research fix (`work/night-2026-10-14/round2-german-drift`
of the ad research workspace) re-transcribes each flagged window with whisper
large-v3-turbo, lets whisper detect the language on the clip (forcing German
garbled genuinely English stretches), and keeps the retry only when it
detects the episode language with p >= 0.9. Slice 3 runs that check inside
the `ad-analysis-v1` profile; this slice provides the retry itself and proves
it gives the research runner's (`retry.py`) output.

## Scope

In scope: `transcriptions/windows.py` with `transcribe_window(audio_path, t)`,
the pinned turbo model check, the identity record, `window_runtime_available()`,
settings `VOXHELM_WINDOW_MODEL` and `VOXHELM_WINDOW_TIMEOUT_SECONDS`, the
comparison tool `evals/window_retry_compare.py`.

Out of scope (slice 3, now `2026-10-16_drift_check.md`): the function-word
check, which windows are retried, the p >= 0.9 acceptance, the splice,
per-window records in `words.json`, the retry model hash in the profile
identity, advertising the profile only on workers where
`window_runtime_available()` holds.

## Interface: an internal function, no HTTP route (D-27)

The podcast spec calls part (a) a "window endpoint". Its only caller is the
profile's drift check (slice 3, and possibly slice 10's turbo windows), which
runs where the episode audio already is: in the Django Tasks worker for a
local job, inside `process_claim` on a remote worker. Delve never calls it;
it receives the spliced words with the job. So the endpoint is the function
`transcribe_window`, called in-process on the materialised source:

- an HTTP route would need the source on the serving host: a remote worker
  would upload its audio back to the control plane per window, or the control
  plane would keep a `source` copy it otherwise deletes after 24 h;
- a route is a new producer surface with its own auth, size limits, lane
  admission and retention duties, for no consumer;
- in-process, "nothing persists after the response" holds by construction:
  no request, job or artifact exists, and the clip lives in a temporary
  directory of the call.

If a producer later needs windows of an episode it holds, a
producer-authenticated synchronous route over a staged upload can wrap the
same function; that is a new decision.

## Design

`transcribe_window(audio_path, t, *, cancel_event=None) -> WindowTranscription`
re-transcribes the grid window `[t, t + 30)` of a local source file:

1. Resolve the runtime: `VOXHELM_WHISPERCPP_BIN` and `VOXHELM_WINDOW_MODEL`
   (a file name in `VOXHELM_MODEL_CACHE_DIR` or a path; default
   `ggml-large-v3-turbo.bin`). The model file's SHA-256 must equal
   `WINDOW_MODEL_SHA256`
   (`1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69`, the
   research file, FREEZE.txt prefix `1fc70f774d38eb16`); the hash is cached
   per path, size, inode and modification time. A missing binary, a missing
   file or another hash raises `BackendUnavailableError`. No other model or
   backend is ever built: not `VOXHELM_WHISPERCPP_MODEL`, not the batch
   chain.
2. Decode the clip with the runner's ffmpeg arguments: `-ss <max(0, t - 1)>
   -t 32.000` as input options (millisecond formatting), `-ac 1 -ar 16000
   -f wav`, into a temporary directory (`voxhelm-window-*`).
3. Run `whisper-cli -m <model> -f clip.wav -l auto -t 4 -p 1 -bs 1 -bo 2
   -tp 0.00 -tpi 0.20 -ml 60 -ojf -of <tmp>/window`. These are the decoding
   parameters of the runner's `whisper-server` 1.9.5 (`server.sh`: `-t 4`,
   server defaults best-of 2 and beam size -1, i.e. greedy; `retry.py` posts
   temperature 0.0 / temperature_inc 0.2; verbose_json turns on token
   timestamps; the server caps segments at 60 characters, which does not
   change tokens). `whisper-cli`'s own defaults (beam search 5, best-of 5)
   gave different words on 5 of 5 windows tried. `-np` is not passed: the
   language probability is only in whisper.cpp's log.
4. Language: the single `auto-detected language: <code> (p = <p>)` log line
   gives the code and probability (6 decimals as whisper.cpp prints them; the
   research server returned the unrounded float); the JSON result's language
   must be the same code. Otherwise the run fails.
5. Words, as `retry.py` on the server's verbose_json: tokens of all segments
   in order; tokens with id >= 50257 (end-of-text and every special token,
   which whisper-server leaves out of its words), tokens whose text starts
   with `[_` and whitespace-only tokens are skipped; a token starts a word
   when its text starts with a space (or there is no word yet) and its
   stripped text is not a substring of `.,!?;:`, otherwise its stripped text
   is appended and the word's end moves; times are clip start plus the token
   offset (whisper-cli's `offsets` in ms, the server's centiseconds * 0.01,
   the same value), rounded to 3 decimals; words whose start lies in
   `[t, t + 30)` are kept. Malformed output fails the run.

Result (`WindowTranscription.as_record()`):

```json
{"t": 1200.0, "clip_start": 1199.0, "language": "de",
 "language_probability": 0.999549,
 "words": [{"w": "Hallo,", "s": 1200.0, "e": 1200.34}, ...],
 "identity": {"engine": "whisper.cpp", "cli": "whisper-cli",
   "cli_version": "1.9.5", "model": "ggml-large-v3-turbo",
   "model_sha256": "1fc70f77...bc69",
   "settings": {"pad_seconds": 1.0, "window_seconds": 30.0,
     "clip_seconds": 32.0, "sample_rate": 16000, "language": "auto",
     "decoding": "greedy", "best_of": 2, "temperature": 0.0,
     "temperature_inc": 0.2, "threads": 4, "processors": 1,
     "max_len": 60, "token_timestamps": true}}}
```

`cli_version` comes from `whisper-cli --version`, cached per binary and
modification time; it is recorded, not enforced (as `parakeet-mlx` in D-26).

Nothing outlives the call: the temporary directory is removed on success,
failure, timeout (`VOXHELM_WINDOW_TIMEOUT_SECONDS`, default 120) and
cancellation; the function writes no job, artifact, staged upload or task
result and runs no database query; it logs no words, and its error messages
carry the tail of whisper.cpp's log, never its stdout (the transcript), so a
failing job cannot store window text in its error. The caller admits the run
through the lane scheduler (the profile is already admitted when slice 3
calls it).

## Tests

`tests/test_window_transcription.py` (fake `whisper-cli` and ffmpeg, fake
model file pinned by its hash):

- identity recorded (engine, CLI version, model name and SHA-256, settings)
  and the exact whisper.cpp and ffmpeg arguments; clip clamped at file start;
- no fallback: a model with another hash or a missing turbo model raises
  `BackendUnavailableError` before anything runs, although the batch model
  `ggml-large-v3.bin` sits next to it and `build_backend_service` would fail
  the test if called; a missing binary is unavailable; a replaced model file
  is hashed again;
- no residue: after success and after each failure (whisper.cpp exit status,
  no JSON, invalid JSON, no language line, malformed tokens, ffmpeg failure)
  the temporary root is empty and there is no job, artifact, staged upload or
  task result; failures carry no transcript text;
- word assembly and window filtering equal a verbatim copy of `retry.py`'s
  loop fed with what whisper-server returns for the same fixture tokens
  (special tokens by id and by text, punctuation before the first word,
  space-led punctuation, whitespace tokens, sub-words across a segment
  boundary, the `[t, t + 30)` bounds), at several clip offsets; malformed
  output; the language log line and its cross-check.

## Verification

### 40 research windows (2026-10-16)

`evals/window_retry_compare.py` on this Mac (M4 Max, shared), whisper.cpp
1.9.5 (Homebrew, the binary of the research runs), ffmpeg 9.0.2, the research
model `ws-podcast-ad-reassess-20261012/models/ggml-large-v3-turbo.bin`
(SHA-256 verified: `1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69`,
the FREEZE.txt prefix). 40 dev windows, evenly spaced through the research
auto-mode retry files: 25 flagged German windows (`retry-dev-cand-auto.jsonl`,
269), 5 unflagged German windows (`retry-dev-clean-auto.jsonl`, 30) and 10
windows of English shows (`retry-dev-enctrl-auto.jsonl`, 40), each through
`transcribe_window` on the research source file.

| Set | Windows | Language same | p same (6 decimals) | Words identical (text, start, end) | Words |
| --- | --- | --- | --- | --- | --- |
| German, flagged (`cand`) | 25 | 25 (de) | 25 | 25 | 2,005 |
| German, unflagged (`clean`) | 5 | 5 (de) | 5 | 5 | 445 |
| English shows (`enctrl`) | 10 | 10 (en) | 10 | 10 | 892 |
| All | 40 | 40 | 40 | 40 | 3,342 |

Every window's word list is identical to the research record (3,342 of
3,342 words, same text, same start and end to the millisecond). Detected
probabilities 0.999273-0.999909, equal to the research values rounded to 6
decimals. No `voxhelm-window-*` directory was left after any window and no
database query ran. Median time per window 1.54 s, including the model load
and ffmpeg; the research server, with the model loaded once,
took a median 1.79 s per clip under other load.

Windows: `cand` bitsundso-0 5910, dasklima-1 480, engkiosk-1 2880,
engkiosk-2 3660, fazki-1 210 / 1050 / 2640, happyboot-0 1380 / 2460 / 3450,
happyboot-1 180 / 1020 / 3600 / 4860, kassenzone-0 1140 / 1890,
kassenzone-1 1200, kassenzone-2 3060, lage-0 480 / 1200, lage-1 690,
lage-2 450 / 1110, lagedev-1 630, lageholdout-0917 240; `clean`
dasklima-1 2490, bitsundso-0 4560 / 3660, engkiosk-2 2280 / 1650; `enctrl`
homeassistant-1 2280, sedaily-1 1470, lwaidev-1 540, talkpython-1 1620,
latentspace-0 930, freakonomics-0 1200, pythonbytes-2 210, pythonbytes-1 990,
lwai-0 3480, pythonbytes-0 270.

The research server kept the model loaded across requests; the function
starts one `whisper-cli` per window, so no decoder state carries from one
window to the next. Identical words on all 40 windows indicate that the
server's results did not depend on the previous request either.

Rerun:

```bash
VOXHELM_WINDOW_MODEL=/Users/jochen/workspaces/ws-podcast-ad-reassess-20261012/models/ggml-large-v3-turbo.bin \
    uv run python evals/window_retry_compare.py --out /tmp/window-compare.json
```
