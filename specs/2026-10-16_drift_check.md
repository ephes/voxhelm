# German drift check in `ad-analysis-v1`

Status: 2026-10-16, implemented and verified locally (see "Verification",
gate G2 passes); not deployed. Decision: D-28. Source: slice 3 of the podcast
server-side ad analysis,
`~/projects/podcast/docs/player/specs/server-ad-analysis-design.md`
("Analysis Stages" 3, "Voxhelm Adapter Contract" row "Drift check with turbo
retry" part (b), gate G2, Slice Plan row 3). Builds on
`2026-10-16_ad_analysis_profile.md` (D-26) and
`2026-10-16_window_transcription.md` (D-27).

## Why

Parakeet drifts into English inside German speech: phrases or half sentences,
also inside ad reads ("Werbung Ende." became "They have all ended."). The ad
research (`work/night-2026-10-14/round2-german-drift` of the ad research
workspace, FREEZE.txt 2026-10-15T00:54:03Z) froze a cheap check and a whisper
large-v3-turbo retry that removes about 90 % of the excess English for about
20 s per German audio hour. The podcast spec puts that check inside the
profile, because it repairs an ASR defect, and wants its per-window records
returned with the words.

## Scope

In scope: `transcriptions/drift.py` (rule, records, splice); running it in
`run_profile_transcription` for German jobs; the `drift` part of `words.json`
and of the identity; the window runtime as a profile requirement (local
execution and worker capability); the control plane's check of a worker's
drift identity; `evals/drift_check_g2.py`.

Out of scope: drift checks for other languages, splicing the transcript
formats, the holdout run of G2 (the gate is dev-only), dropping "these" from
the English list or restricting whisper's language choice (each would be a new
profile).

## Design

### Rule (frozen research, reproduced verbatim)

1. Grid: the windows `[30k, 30k + 30)` of file time over the Parakeet word
   starts (`lib.py` `grid`, `int(s // 30)`); a window exists when at least one
   word starts in it.
2. Count: English and German function words of the window's text with the
   research lists and tokenizer (`lib.py` `EN`, `DE`, `toks`, `counts`: lower
   case, `’` -> `'`, everything but `a-z0-9äöüß'` is a separator, apostrophes
   stripped unless the token is an English function word; English words that
   are also German ones are removed from the English list). The lists are
   copied verbatim, `"these"` included (REPORT.md proposed dropping it after
   the holdout; the frozen rule keeps it). Their SHA-256 (canonical JSON of
   both sorted lists, `5dbea279...2432`) is in the identity.
3. Flag: `en >= 4 and en / max(1, en + de) >= 0.10` (`splice.py` with
   `4,0.1`).
4. Retry: `transcribe_window(audio, t)` (D-27): clip `[t - 1, t + 31]`,
   language detected on the clip, words starting in `[t, t + 30)`.
5. Accept: the detected language equals the job's `language_hint` and
   p >= 0.9 (the post-hoc rule the spec adopts); otherwise Parakeet's words
   stay.
6. Splice (`splice.py`): every Parakeet word starting in an accepted window
   is dropped, the retry's words are added, and the list is sorted by start
   (stable). Retry words carry `c: null`.

The check runs only for `language_hint` `de` (`DriftRule.languages`); for any
other hint or none it does nothing and says so in its record.

### Where it runs

`run_profile_transcription` resolves the Parakeet runtime and the window
runtime (`whisper-cli`, the turbo model with the pinned SHA-256) before
Parakeet runs, then runs Parakeet, builds the words, and runs the drift check
on them, all inside the lane admission the caller already holds (local Django
Tasks worker or `process_claim` on a remote worker). Each flagged window
starts one `whisper-cli` (D-27); windows run one after another.

No silent skip:

- a missing or unverified turbo model or `whisper-cli` fails a local job
  before Parakeet runs, whatever its language (the profile needs both
  runtimes; capability and identity are per profile, not per job language);
- remote workers advertise `ad-analysis-v1` only while Parakeet and
  `window_runtime_available()` both hold (re-checked every heartbeat), so a
  profile job waits for such a worker; a worker that loses the turbo model
  after claiming raises `BackendUnavailableError` and the attempt is
  retryable;
- a window retry that fails (ffmpeg, whisper.cpp, malformed output, model gone
  between Parakeet and the retry) fails the run.

Cancellation: the `cancel_event` is checked before Parakeet, before each
retry, inside each retry (`transcribe_window`) and after the last window;
`InferenceCancelled` propagates and no words are returned.

### Artifact: records inside `words.json` (D-28)

```json
{"schema_version": 1, "profile": "ad-analysis-v1", "identity": {...},
 "words": [{"w": "Also", "s": 5912.69, "e": 5913.06, "c": null}, ...],
 "drift": {
   "rule": "fw-en4-share10-v1", "checked": true, "language": "de", "reason": null,
   "summary": {"windows": 263, "flagged": 2, "accepted": 2, "rejected": 0,
               "retry_seconds": 3.259},
   "windows": [
     {"t": 0.0, "en": 0, "de": 15, "flagged": false,
      "words_before": 43, "words_after": 43},
     {"t": 180.0, "en": 17, "de": 10, "flagged": true,
      "words_before": 53, "words_after": 72, "language": "de", "p": 0.999651,
      "accepted": true, "retry_words": 72, "retry_seconds": 1.736}]}}
```

For a non-German job: `"checked": false`, `"reason": "language_hint 'en' is
not checked (checked: de)."` (or `"no language_hint ..."`), zero summary,
`"windows": []`, and the words are Parakeet's unchanged.

The records live in the words artifact, not in a sibling artifact or the job
metadata: Delve already fetches `words.json`; the records describe exactly
those words; the remote completion check (exactly one exposed `words.json`),
producer `DELETE` and the 7-day backstop cover them unchanged, because both
delete every artifact of a profile job. They hold no text, only counts,
languages, probabilities and timings. `retry_seconds` is a measurement, so
two runs of the same episode give the same words but not byte-identical
documents. `schema_version` stays 1: nothing with the slice 1 shape was ever
served.

The transcript formats (`transcript.json`, text, VTT) stay Parakeet's
sentences; only the words carry the splice.

### Identity

`identity.drift` (in `result.metadata.profile` and `words.json`):

```json
{"rule": "fw-en4-share10-v1", "languages": ["de"], "window_seconds": 30.0,
 "min_english": 4, "min_english_share": 0.1,
 "function_words_sha256": "5dbea279a0c276d25d5f7606093a73448cd7682f211807e1e95a896e1bce2432",
 "accept_language": "episode", "accept_probability": 0.9,
 "retry": {"engine": "whisper.cpp", "cli": "whisper-cli", "cli_version": "1.9.5",
   "model": "ggml-large-v3-turbo",
   "model_sha256": "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69",
   "settings": {...}}}
```

It is recorded for every job of the profile, German or not. For a remote
completion the control plane refuses (400) a `drift` that differs from its
own rule record or whose `retry` names another engine, model, hash or
settings, and keeps only the worker's `cli_version` (sanitized).

### Profile name (D-28)

The podcast spec says any change to the profile's settings is a new name.
Slice 1 is committed but was never deployed, served or cached by a producer,
and the podcast spec's own profile JSON pairs `voxhelmProfile:
ad-analysis-v1` with the drift block. So `ad-analysis-v1` is amended in place
to include the drift check; from its first deploy on, any change (rule, word
lists, thresholds, turbo model, decoding) is a new name.

### Cost

One `whisper-cli` start per flagged window (model load included): median
1.57 s per window in G2, 17.2 s per German audio hour, inside the 30 s
budget. A resident `whisper-server` would save the per-window model load but
adds a long-lived service per host; not worth it at this cost. An episode with
a long genuinely English part can flag up to ~90 % of its windows (research
holdout, Kassenzone-n3: about 75 s of retries).

## Tests

`tests/test_drift_check.py` (fake retry callable, or the fake Parakeet CLI
of `test_ad_analysis_profile.py` with the fake `whisper-cli` / ffmpeg / pinned
model of `tests/window_fakes.py`):

- rule: word lists and tokenizer equal a verbatim copy of `lib.py` (and the
  list hash is pinned); flag edges `4/36` flagged (share exactly 0.10),
  `4/37` not, `5/45` / `5/46`, `3/0` not, `4/0` flagged, on counts and on
  window text;
- splice: words at 29.999 / 30.0 / 59.999 / 60.0 with window 30 accepted:
  only 30.0 and 59.999 are replaced; several windows; equal to a verbatim
  copy of `splice.py`'s word list; retry words `c: null`;
- acceptance: `de` p 0.9 accepted, `de` p 0.899999, `en` p 0.999 and `nl`
  p 0.95 rejected (Parakeet's words stay, `words_after == words_before`);
- records: every window recorded, unflagged ones without retry fields, the
  summary counts;
- non-German jobs (`en`, `fr`, none): no retry, words unchanged, record says
  why; a local English job runs no `whisper-cli`, keeps Parakeet's words and
  still records the retry model hash;
- a failing retry fails the check;
- cancellation before the first window and while one runs (the next window
  is not retried, the retry got the event); through the profile:
  cancelled during the first window raises `InferenceCancelled` and leaves no
  temporary file; cancelled before the start runs nothing;
- local German job through the real `transcribe_window`: the flagged windows
  are exactly those of the rule, one window answering English keeps
  Parakeet's words, the others carry the retry's words, `words_after`
  matches, transcript artifacts unchanged, no temporary files;
- identity: `drift` with the rule, list hash, acceptance and the pinned
  retry model hash and whisper.cpp version, equal in metadata and artifact;
- window runtime missing, with another hash, or no `whisper-cli`: German and
  English jobs fail before Parakeet runs, no words artifact, no other engine;
  model lost after Parakeet: the job fails;
- `DELETE` and the backstop (finished 7 days 1 minute ago) remove the words
  artifact with its records, the row, and leave no stored file;
- workers: the profile is advertised only while the window runtime resolves
  (Parakeet still advertised without it) and again once it is back; a claim
  without the turbo model raises `BackendUnavailableError` (retryable)
  before Parakeet runs; a completion keeps the drift identity (worker's
  whisper.cpp version), and one without `drift`, with another threshold,
  list hash, retry hash, model or decoding setting is refused.

## Verification

### G2 on the 22 German development episodes (2026-10-16)

`evals/drift_check_g2.py run` on this Mac (M4 Max, 16 cores, 128 GB,
shared: 1-minute load average 4.8-13.5 during the run from other sessions'
work), whisper.cpp 1.9.5 (Homebrew), the research turbo model
(`ws-podcast-ad-reassess-20261012/models/ggml-large-v3-turbo.bin`, SHA-256
verified as pinned), `parakeet-mlx` 0.5.3 / `mlx` 0.32.4 with the pinned
snapshot `ed2b7e8c...`. Each episode (19.71 h in total) was staged with
`POST /v1/uploads`, run as an `ad-analysis-v1` job with `language_hint=de`
through the local `django_tasks` code path (Django test client, immediate
backend, throw-away test database, one process), its `words.json` fetched,
and then deleted with the producer `DELETE` (204 for all 22; no stored file
left afterwards). `evals/drift_check_g2.py analyse` compared the documents
with the research outputs.

The spec's reference "dev 7.6 % of German windows flagged" is REPORT.md's
share of *reference drift windows* (180 windows with at least 6 more English
function words than whole-file turbo); the frozen check itself flagged
215 of 2,376 (9.0 %). The gate is held to the frozen check.

| G2 measure | Research (frozen, dev) | This profile | Gate | Result |
| --- | --- | --- | --- | --- |
| Grid windows with Parakeet words | 2,369 (of 2,376 incl. turbo-only) | 2,369, counts identical on all (EN, DE, words) | — | — |
| Flagged windows | 215 | 215, the same set (0 only here, 0 only there) | set equal | pass |
| Flagged share | 9.0 % (215 / 2,376) | 9.05 % (215 / 2,376; 9.08 % of 2,369) | 9.0 ± 1 point | pass |
| Retries accepted (de, p >= 0.9) | 215 of 215 (p 0.999+) | 215 of 215 (lowest p 0.999252) | — | — |
| Excess English captured by flagged windows (`sweep.py`) | 2,493 / 2,783 (90 %) | 2,493 / 2,783 (89.6 %) | — | — |
| English function words in flagged windows, Parakeet -> retry (turbo reference) (`evalretry.py`) | 2,535 -> 50 (42) | 2,535 -> 50 (42); 211 of 215 within reference + 1 | — | — |
| Excess English removed over all windows (after the splice, against turbo) | — | 2,783 -> 302 (89.1 %) | >= 85 % | pass |
| Retry time per German audio hour | 19.7 s (whisper-server, model resident; median 1.79 s per window) | 17.2 s (339 s for 215 windows; median 1.57 s per window incl. ffmpeg and model load) | <= 30 s | pass |
| Worst episode | — | happyboot-1: 33 windows, 52.6 s | — | — |

The whole run took 1,059 s for 19.7 h (Parakeet and the drift check
together; the drift check is 32 % of it).

### Splice agreement with the research

Reference: `splice.py`'s word list built from the research Parakeet words
(`parakeet-chain/pk-dev`) and `retry-dev-cand-auto.jsonl`, restricted to the
frozen rule `4,0.1` and with the adopted acceptance (language German,
p >= 0.9) applied (all 215 pass it, as without it); and the research's
spliced detector lines `tx-dev/pk30s-fix` (`sentence_segs` on the spliced
words).

| | Episodes identical | Words |
| --- | --- | --- |
| Spliced word list (text, start, end) | 22 of 22 | 198,048 of 198,048 |
| Retry words of the 215 accepted windows | 215 of 215 windows | — |
| Detector lines (`sentence_segs`, start, end, text) vs `tx-dev/pk30s-fix` | 22 of 22 | — |

Rerun:

```bash
uv run python evals/drift_check_g2.py run --out /tmp/g2
uv run python evals/drift_check_g2.py analyse --out /tmp/g2
```
