# Mixed-speaker transcript segments (plan, step 1b deployed, turn corroboration implemented)

Status: 2026-10-12. Step 1b (sub-window veto) is implemented and deployed with
voxhelm 0.1.3 (control plane and both workers); steps 1a, 2 and 3 are still
plan. Turn corroboration (see "Uncertain segments without a veto") is
implemented but not deployed. Revised after a fresh-context review by gpt-6.1-sol (high
effort) on the same day; the main findings were checked against the source and
are folded in below. The open question is settled, see "Findings" below:
pyannote emitted no turn for the greeting, the sub-window classification does
separate the voices, and the job was transcribed by whisper.cpp.

## Problem

Voxhelm assigns exactly one speaker to each ASR segment. When a segment holds
speech from two people, the minority speaker's words are published under the
majority speaker's name, and the known-speaker sidecar marks that suggestion
as confident.

Observed on Django Chat episode 206 (job `f29c92d2-b629-4e3e-b2ba-1a8bb71d4a1d`,
two known speakers with approved voice references, strategy
`pyannote_known_speaker`):

| start | end | text | sidecar |
|---|---|---|---|
| 0.00 | 10.02 | "hello welcome to another episode of django chat i'm will vincent with carlton gibson hey carlton" | Will 0.62 / Carlton 0.05, confident |
| 10.02 | 16.40 | "hello will welcome back yeah we're back we did a sneak peek one for the developer survey but" | Will 0.63 / Carlton 0.10, confident |

The second segment opens with about 1.5 s of Carlton ("hello will welcome
back") followed by about 5 s of Will. `raw_diarization_speaker` is `Speaker 1`
for both segments. The consumer (django-cast) applied the confident
suggestion, so Carlton's greeting is shown as Will's.

Across the 454 segments of that job, 339 were confident and 115 uncertain.
Only 13% of segments contain punctuation, and segments are near-uniform
5-6 s windows that end mid-sentence, starting at second 0. Most other Django
Chat transcripts are 27-80% punctuated and begin with sentence-shaped
segments.

## Current behaviour in code

- `transcriptions/diarization.py::choose_speaker_for_segment` returns the label
  of the single turn with the longest overlap, not the label with the largest
  total coverage. With A on 0-3 s and 7-10 s and B on 3-7 s it picks B. A
  second speaker inside the segment is discarded.
- `extract_pyannote_annotation` prefers `speaker_diarization` over
  `exclusive_speaker_diarization`, so raw turns can overlap, and stretches
  without any turn exist.
- Raw turns are not persisted by either execution path (`jobs/services.py`,
  `jobs/remote_worker_cli.py`); only the per-segment label reaches the sidecar.
- `transcriptions/known_speaker.py::classify_segments` embeds the audio of the
  whole segment once and classifies that single embedding. `is_confident`
  looks only at duration, top similarity and margin. Whether the raw pyannote
  turns show more than one speaker inside the segment is not considered; only
  the dominant raw label is stored.
- `transcriptions/service.py`: `MlxWhisperBackend` passes
  `word_timestamps=False`; `WhisperCppBackend` writes `-oj` without word-level
  output. Both payload normalizers drop word/token fields, and
  `TranscriptionSegment` has no word representation, so a segment cannot be
  split later.
- By default `WhisperCppBackend` runs with `--max-context 0` and the MLX
  backend with `condition_on_previous_text=False`, set deliberately to stop
  long-form repetition loops. Both are overridable
  (`VOXHELM_WHISPERCPP_MAX_CONTEXT`, `VOXHELM_MLX_CONDITION_ON_PREVIOUS_TEXT`).
  The episode 206 job ran on whisper.cpp with the default `--max-context 0`
  (see "Findings").
- `pad_to_min_duration` appends silence, while `is_confident` checks the
  unpadded duration. Any segment shorter than `min_segment_duration` (1.5 s)
  is therefore always uncertain.

## Open question to settle first

Settled on 2026-10-12, see "Findings": pyannote did not emit a Carlton turn
inside 10.02-16.40, so the second branch below applies. The question as it
stood:

It was not known whether pyannote produced a separate turn for the 1.5 s
greeting at 10.0 s. The sidecar keeps only the dominant raw label per
segment, and the turns were not stored. Reproduce the full-episode
diarization with the same model and speaker-count hint, keep the turns, and
read the first minute. A run on a cropped first minute is only a preliminary
hint, because pyannote clustering changes with the input.

- If pyannote emitted a Carlton turn inside 10.02-16.40, step 1a catches this
  case.
- If it did not, nothing that relies on pyannote boundaries can, including the
  splitting in step 2. Only detection that is independent of pyannote helps:
  step 1b.

## Findings (2026-10-12)

Measured with `evals/mixed_speaker_segments_probe.py`, run with the deployed
worker's interpreter and env file on `atlas`, so the shipped code and the
installed versions are the ones under test: voxhelm 0.1.2, pyannote.audio
4.0.4, torch 2.12.0, `pyannote/speaker-diarization-3.1` on MPS, embedding model
`pyannote/wespeaker-voxceleb-resnet34-LM`. Audio: the public episode 206 MP3
(2195 s). References as in the job: source ranges from the "Breaking Django"
episode, Will 136.43-166.43 s and Carlton 27.44-57.44 s, six 5 s windows each.

### How the job was run

From the job record and the worker log:

- Worker `atlas`, `execution_mode: remote_pull`, claimed 11:44:48 and completed
  11:47:38 UTC, `processing_seconds` 169.6.
- STT backend `whisper.cpp`, model `ggml-large-v3.bin`, `requested_model: auto`,
  `language: en`, no prompt (the remote worker always passes `prompt=None`).
  `atlas` itself is configured with `VOXHELM_STT_BACKEND=mlx`, but the control
  plane resolves the backend for `auto` from its own setting and puts it in the
  claim, so the worker's preference does not apply.
- whisper.cpp arguments as built by `WhisperCppBackend` with the worker's
  settings: `-p 4 -l en -np --max-context 0 --suppress-nst`, no word timing.
- Diarization hint `num_speakers: 2`, strategy `pyannote_known_speaker`,
  thresholds at their defaults (1.5 s, margin 0.15, top similarity 0.55).

Re-running whisper.cpp on the full file through the worker code path
reproduces the job: 454 segments, 60 of them punctuated (13%), the same
boundaries and text for the first segments (0.00-10.02, 10.02-16.40, ...).

### pyannote turns in the first minute

Full-episode run with `num_speakers=2` (294 turns, 37 s). `SPEAKER_00` is
Will, `SPEAKER_01` is Carlton.

| start | end | length | label |
|---:|---:|---:|---|
| 5.077 | 19.066 | 13.99 | SPEAKER_00 |
| 19.336 | 29.967 | 10.63 | SPEAKER_00 |
| 27.588 | 27.824 | 0.24 | SPEAKER_01 |
| 29.967 | 36.025 | 6.06 | SPEAKER_01 |
| 36.025 | 36.042 | 0.02 | SPEAKER_00 |
| 36.413 | 40.430 | 4.02 | SPEAKER_00 |
| 41.307 | 54.166 | 12.86 | SPEAKER_01 |
| 52.883 | 56.410 | 3.53 | SPEAKER_00 |
| 56.410 | 62.030 | 5.62 | SPEAKER_01 |

One `SPEAKER_00` turn spans 5.08-19.07 s. It covers Will's introduction,
Carlton's "hello will welcome back" and Will's answer. There is no
`SPEAKER_01` turn anywhere inside 10.02-16.40. The run without a hint (328
turns, three labels, the third with 31 s in total) has the same 5.08-19.07
turn, so the result does not depend on the hint. `exclusive_speaker_diarization`
differs only where turns overlap (52.9-54.2 s) and has the same turn here.

Consequence: step 1a cannot catch this segment, and neither can the splitting
in step 2. The sub-window sweep below also places Carlton at about 18.8-22.5 s
("the long stretch until christmas"), again inside a `SPEAKER_00` turn, so the
greeting is not a single outlier.

### Sub-window classification of 10.02-16.40

Whole-segment classification reproduces the sidecar: 0.00-10.02 gives Will
0.619 / Carlton 0.054, and 10.02-16.40 gives Will 0.628 / Carlton 0.103.

1.5 s windows, 0.75 s hop:

| window | Will | Carlton | top | margin |
|---|---:|---:|---|---:|
| 10.02-11.52 | 0.045 | 0.088 | Carlton | 0.043 |
| 10.77-12.27 | 0.167 | 0.444 | Carlton | 0.277 |
| 11.52-13.02 | 0.474 | 0.252 | Will | 0.222 |
| 12.27-13.77 | 0.540 | 0.109 | Will | 0.432 |
| 13.02-14.52 | 0.516 | -0.034 | Will | 0.550 |
| 13.77-15.27 | 0.616 | -0.040 | Will | 0.656 |
| 14.52-16.02 | 0.600 | 0.015 | Will | 0.584 |

2.0 s windows with 0.5 s hop show the same change: 10.52-12.52 gives Carlton
0.422 / Will 0.169, and every window from 11.52 on is Will with a margin of
0.36 or more. 1.0 s windows are too noisy: the Carlton windows reach a margin
of at most 0.18 and the first second is undecided.

So the voices do separate, and step 1b can catch this segment. Three things
follow for its design:

- Short windows have low absolute similarity. Only two of the seven 1.5 s
  windows pass the existing `is_confident` rule, and the Carlton window peaks at
  0.444, under `min_top_similarity` 0.55. The veto needs its own rule, for
  example two windows with different top speakers that each clear a margin
  threshold. One window at margin 0.277 against several at 0.43-0.66 is the
  evidence available here; a margin threshold of 0.15 would fire, 0.30 would
  not.
- Windows without speech score near zero for both speakers. pyannote finds no
  speech before 5.08 s, and the 1.5 s windows inside 0.0-4.5 s have
  similarities of at most 0.07 in magnitude and margins under 0.04. Such
  windows must not count as a disagreeing speaker; a margin floor handles
  that.
- Cost is small: 6-12 ms per embedding on `atlas` after the model is
  loaded (12 ms while a diarization ran beside it), against about 40 s of embedding time for the whole job today. Seven
  windows per 6 s segment is roughly eight times as many embeddings.

### Observation for step 3

On a 90 s crop of the same file, the same whisper.cpp settings produce cased,
punctuated, sentence-shaped segments, with "Hello, Will. Welcome back." as its
own segment at 10.72-11.96. On the full file the output is lowercase and
unpunctuated from the first word. `--max-context 0` is set in both runs, so
disabled conditioning alone does not explain the lowercase output. What differs
is the input length, and with it how `-p 4` divides the audio. MLX on the same
crop was lowercase until about 83 s and punctuated after that. Step 3 should
test `-p 1` on the full file before testing prompts.

## Plan

### Step 1: stop calling mixed segments confident (small)

Two independent vetoes. Neither changes segment boundaries, indices or public
artifacts; both only turn a confident suggestion into an uncertain one.

**1a. Raw-turn coverage.** In `classify_segments`, compute per segment the
clipped coverage per normalized raw label (union of that label's turns inside
the segment, so same-label overlaps are not counted twice). The dominant label
is the one with the largest coverage. If another label covers at least a
threshold (proposal: 0.75 s or 15% of the segment, whichever is larger), set
`speaker_uncertain=True` and `speaker=None`.

**1b. Sub-window classification.** Embed sliding windows inside the segment
(for example 1.5 s windows with 0.75 s hop) and classify each against the
centroids. If windows disagree on the top speaker with a clear margin, veto
the segment as mixed. This is the only step that can catch a change pyannote
missed, and it is the cheaper experiment to run first on the 10.02-16.40
segment. Cost: several embeddings per segment instead of one; measure it.

For both: record the reason and the evidence in the sidecar segment as new
optional fields (version stays 1, fields are additive) and report
`mixed_segment_count` in the summary.

**1b as implemented.** `find_mixed_windows` in `transcriptions/known_speaker.py`
checks every segment that would be auto-accepted: 1.5 s windows, 0.75 s hop,
plus a window aligned to the segment end. A window that names a different top
speaker with margin >= 0.15 and top similarity >= 0.30 vetoes the segment. The
segment gets `speaker_uncertain_reason="mixed_speakers"` and the dissenting
windows in `speaker_mixed_windows`. The four values are request options
(`diarization.known_speaker.mixed_window_*`), `mixed_window_duration=0` turns
the veto off, and the contract is described in `known-speaker-diarization.md`.

Thresholds were chosen on the 454 segments of episode 206 (339 confident
without the veto). Number of confident segments vetoed, by window margin and
similarity floor, for 1.5 s windows with 0.75 s hop:

| margin | no floor | sim >= 0.30 | sim >= 0.40 |
|---:|---:|---:|---:|
| 0.10 | 28 | 25 | 19 |
| 0.15 | 25 | 23 | 17 |
| 0.20 | 22 | 22 | 17 |
| 0.25 | 18 | 18 | 15 |

The 10.02-16.40 segment is vetoed in every cell. With the defaults, 23 segments
(6.8% of the confident ones) become uncertain and 316 stay confident. Their
texts read as exchanges ("... as well as versioning yes", "... i don't want to
minimize the word bgt oh yeah so mark smith and i ..."), but they were not
checked against a speaker-labelled reference, so the false-veto rate is not
measured. The margin-only rule is not enough on its own: a window over intro
music reached margin 0.195 at a top similarity of 0.03. The veto adds 2198
embeddings for this episode, 1.7 s on `atlas` on top of 12.9 s.

Verified in production on 2026-10-12 (item 3 under "Order and verification"):
job `4d50f8c2-8074-42a2-981d-2f826e80a81e` re-ran episode 206 on worker `atlas`
with voxhelm 0.1.3 and reproduced the local numbers: 454 segments, 316
confident, 138 uncertain, `mixed_segment_count` 23, Will 116 / Carlton 200.
Segment 10.02-16.40 has `speaker: null` with reason `mixed_speakers` and the
Carlton window 10.77-12.27 (similarity 0.444214, margin 0.277257). django-cast
consumed the sidecar unchanged. In the first 75 s, 7 of 12 transcript lines now
carry no name; the vetoes at 16.40-21.84 and 28.02-33.34 look plausible from
the text but were not checked against the audio.

Thresholds checked on episode 205 ("Django Developers Survey 2026", public
feed MP3, 2997 s, same whisper.cpp settings and references, 2026-10-12): 710
segments, 495 confident before the veto (Will 258 / Carlton 237), 28 vetoed at
the defaults (5.7%, episode 206: 6.8%). Vetoed counts for 1.5 s / 0.75 s
windows, as no floor / similarity >= 0.30 / >= 0.40:

| Margin | Episode 205 | Episode 206 |
| --- | --- | --- |
| 0.10 | 36 / 29 / 25 | 28 / 25 / 19 |
| 0.15 | 33 / 28 / 24 | 25 / 23 / 17 |
| 0.20 | 28 / 24 / 23 | 22 / 22 / 17 |
| 0.25 | 21 / 20 / 20 | 18 / 18 / 15 |

As a proxy for a labelled reference, each confident segment was compared with
the raw pyannote turns (`num_speakers=2`): a segment counts as pyannote-mixed
when a second speaker's turns overlap it for at least 0.3 s. On episode 205 all
28 vetoed segments are pyannote-mixed; on episode 206, 19 of 23 are, and the
other four (10.02-16.40, 16.40-21.84, 338.26-343.50, 442.86-447.20) are in
stretches where pyannote emitted no second-speaker turn at all. The defaults
stay as they are. The proxy does not prove a veto is right, and it is not
independent of the embedding model family.

The same comparison shows what the veto leaves alone: 57 confident segments on
episode 205 and 38 on episode 206 are pyannote-mixed without a veto (18 and 14
when the second speaker needs at least 1.0 s). That is the population step 1a
would address; how many of them are backchannels was not examined.

Still open for 1b: no speaker-labelled reference exists, so the false-veto
rate is estimated by the proxy only.

#### Uncertain segments without a veto

The veto only removes names. After it, 115 of 454 segments on episode 206 and
215 of 710 on episode 205 are uncertain for another reason (2026-10-12, same
runs as above; "pyannote-mixed" as defined above):

| Reason | Ep. 206 segments / seconds | pyannote-mixed | Ep. 205 segments / seconds | pyannote-mixed |
| --- | --- | --- | --- | --- |
| shorter than 1.5 s | 46 / 33 | 4 | 110 / 98 | 8 |
| only similarity < 0.55 | 45 / 238 | 22 | 69 / 202 | 11 |
| similarity < 0.55 and margin < 0.15 | 20 / 100 | 15 | 28 / 125 | 18 |
| only margin < 0.15 | 4 / 22 | 4 | 8 / 56 | 8 |

"Only similarity" is the largest block by speech time and the one where the
embedding already picks a speaker with a clear margin. The diarization label
of each such segment was mapped to a known speaker through the confident
segments of the same job (episode 206: one label 114 Will / 1 Carlton, the
other 199 Carlton / 2 Will; episode 205: 237 / 1 and 229 / 0). Where pyannote
sees a single speaker on the segment, that mapped speaker equals the embedding
top speaker in 23 of 23 segments on episode 206 and 57 of 58 on episode 205.
The one disagreement is 12.30-15.26 on episode 205 ("Hello, Will. What are we
doing in summer? ..."), a greeting pyannote attributed to one speaker; the rule
below leaves it uncertain. Agreement does not depend visibly on the similarity
between 0.35 and 0.55, but only 11 single-speaker segments lie below 0.40.

**Turn corroboration as implemented** (`classify_segments`, contract in
`known-speaker-diarization.md`): a segment failing only `min_top_similarity`
is accepted when its similarity is at least `turn_corroborated_min_similarity`
(0.35), pyannote puts one speaker on it, that label maps to the same known
speaker, and the mixed-segment veto finds nothing. Run with the worker
dependencies on both episodes:

| | Episode 206 | Episode 205 |
| --- | --- | --- |
| confident before / after | 316 / 337 | 467 / 520 |
| uncertain before / after | 138 / 117 | 243 / 190 |
| corroborated | 21 | 53 |
| `mixed_segment_count` before / after | 23 / 25 | 28 / 28 |
| distribution after | Will 130 / Carlton 207 | Will 255 / Carlton 265 |

On episode 206 two candidates were stopped by the veto; they were uncertain
before and now also carry the `mixed_speakers` reason, which is why the mixed
count rises without any segment losing a name.

The accepted segments were read as text only: they sit inside longer runs of
the same speaker. They were not checked against the audio, and both episodes
have the same two speakers, so the rule is untested on episodes with guests
or more than two voices. Runtime is unchanged within noise (one extra veto
pass per corroborated candidate).

The change is not classifier-only. It also needs:

- `parse_known_speaker_config` and both job/claim config builders, which copy
  only the existing thresholds today;
- the remote completion summary allowlist in `jobs/remote_workers.py`, which
  would strip `mixed_segment_count`;
- tests for threshold propagation, serialization and local/remote summary
  parity, plus fragmented-turn and overlapping-turn cases for 1a;
- `specs/known-speaker-diarization.md` and usage/release docs in the same
  change, since acceptance policy, configuration and sidecar fields change.

Risk: more blank segments in fast banter. Check thresholds against at least
episodes 205 and 206.

### Step 2: split segments at speaker changes (larger)

Scope: this splits only where pyannote detected a change. It does not recover
changes pyannote missed.

a. Produce and keep word timing: `word_timestamps=True` for MLX and
   token-level JSON for whisper.cpp, grouped into words; a normalized word
   list on `TranscriptionSegment`; both payload normalizers; and a defined
   behaviour in the sanitizer, which rebuilds segments during text edits and
   loop collapse (preserve the words or invalidate them). Behind a setting
   until both backends and both execution paths are verified.
b. After diarization, coalesce adjacent same-label turns, then split each ASR
   segment at turn boundaries inside it. Define what happens to a word inside
   overlapping turns and to a word in a gap without any turn; do not rely on
   "the turn containing the midpoint" alone. Decide whether to split on
   `exclusive_speaker_diarization` while keeping the overlapping annotation as
   evidence for step 1a.
c. Short splits: never merge a short split into a neighbour of a different
   speaker, since that recreates the mixed segment. Keep it as its own
   uncertain segment or merge only into a same-speaker neighbour.
d. Classify each split separately. Splits under 1.5 s stay uncertain by the
   existing duration rule; padding adds silence, not evidence. State that as
   the expected outcome or define and validate a separate short-turn policy.
e. Render Podlove, DOTe and WebVTT from the split segments and keep sidecar
   indices aligned with them.

Risks: whisper.cpp word timing quality and its interaction with
`--max-context 0`; short windows give noisier embeddings; the sanitizer
(`transcriptions/sanitizer.py`) and the repetition-loop collapse operate on
segments and must run before splitting; segment counts change for consumers
that key editorial decisions on segment index (django-cast stores editor
decisions by segment position).

### Step 3: punctuation and sentence-shaped segments (experiment)

Hypothesis, unverified: with previous-text conditioning disabled, Whisper
drifts into lowercase, unpunctuated output, and then segments are cut by time
instead of by sentence, which makes mixed segments far more frequent.

Experiment on the episode 206 audio: compare the current settings against
(a) an initial prompt in punctuated, cased English and (b) whisper.cpp
`--carry-initial-prompt`, measuring punctuated-segment share, median segment
duration and the number of mixed-speaker segments. The backend and settings the
worker used are recorded under "Findings", together with an observation that
points at input length and `-p 4` instead of conditioning.

`sanitize_result` emits no counters and collapses loops before artifacts are
written, so a returning loop would be hidden. Capture raw decoder output and
compare loop statistics before and after sanitization, or add
instrumentation, before using this experiment as an adoption gate.

This experiment is cheap and should run before committing to step 2.

#### Results (2026-10-13, episode 206, `atlas`)

Measured on the full episode with `whisper-cli` and `ggml-large-v3.bin`, called
directly with the worker's arguments (`-l en -np --suppress-nst`) on the same
16 kHz WAV; "punctuated" is the share of segments containing `.`, `?`, `!` or
`,`. Raw decoder output, not sanitized.

| Variant | Segments | Punctuated | Words | Mean segment |
| --- | --- | --- | --- | --- |
| deployed: `-p 4 --max-context 0` | 454 | 13% | 7453 | 4.8 s |
| `-p 1 --max-context 0` | 446 | 21% | - | 4.9 s |
| `-p 1 --max-context -1` | 412 | 1% | - | 5.3 s |
| `-p 4 --max-context -1` | 418 | 1% | - | 5.2 s |
| `-p 4 --max-context 0 --prompt P` (with or without `--carry-initial-prompt`) | 454 | 13% | 7453 | 4.8 s |
| `-p 4 --max-context 32 --prompt P --carry-initial-prompt` | 339 | 98% | 5610 | 6.2 s |
| `-p 4 --max-context 64 --prompt P --carry-initial-prompt` | 442 | 82% | 6709 | 4.8 s |
| `-p 4 --max-context 32 --prompt P` | 873 | 88% | 6614 | 2.3 s |
| MLX `whisper-large-v3-mlx`, worker settings | 549 | 46% | - | 3.9 s |

`P` is "Hello, and welcome to another episode of Django Chat. I'm Will Vincent,
joined by Carlton Gibson. Hi, Carlton!".

What this shows:

- `-p 4` is not the cause, and the hypothesis is half right: enabling
  previous-text conditioning makes it worse, because the lowercase style of the
  first window is then carried through the whole file.
- **With `--max-context 0` whisper.cpp ignores `--prompt` entirely**: the output
  is identical with and without it. The API `prompt` parameter therefore has no
  effect on the whisper.cpp backend as deployed. Not fixed here.
- The style is decided per 30 s window. The first 90 s of the same WAV give 0%
  punctuated segments as full 30 s windows (`-p 1`) and 77% as 22.5 s chunks
  (`-p 4` on a 90 s cut); the first 300 s give 32% / 65% / 67% in 30 / 25 / 20 s
  chunks. MLX with the same model shows the same lowercase start, so this is
  model behaviour, not a whisper.cpp artefact.
- A carried, punctuated prompt with a small context budget fixes the style:
  sentence-shaped segments, "Hello, Will! Welcome back!" as its own segment
  10.72-11.96.
- It is not adoptable as is. The prompt leaks into the transcript ("welcome to
  another episode of Django Chat" 6 times, "I'm Will Vincent, joined by Carlton
  Gibson" 3 times at `--max-context 32`), single segments span a whole 30 s
  window, and the word count drops by 25% (10% at the other two variants).
  How much of that is dropped filler words and how much is lost speech was not
  examined. Loop statistics were not compared.
- Sentence-shaped segments do not by themselves produce more speaker names.
  Known-speaker classification on the 339-segment variant (veto and turn
  corroboration on): 253 confident, 86 uncertain, 12 mixed, 26 corroborated;
  1693 of 2105 segment-seconds are named, against 1752 of 2185 on the deployed
  segmentation. Mixed segments halve (25 to 12), but the short sentences fall
  under `min_segment_duration`: in the first 77 s, 11 of 19 lines stay unnamed,
  among them the 1.24 s greeting.

#### Follow-up (2026-10-13): no decoder setting fixes it reliably

- **The carried prompt is destructive, not just noisy.** At `--max-context 32`
  about 15 whole 30 s windows of episode 206 contain only the prompt text (for
  example 329.5-449.3 and 844.2-904.2), so roughly 7 minutes of speech are
  missing; that is the 25% word loss. Per 5-minute bin the deployed run has
  950-1070 words, this variant 590-900. The variant is rejected.
  `--max-context 64` loses up to 27% of the words in single bins as well.
- **VAD** (`--vad`, `ggml-silero-v5.1.2.bin`, `-p 1`): the first seconds become
  sentence segments with tight times ("Hello, Will. Welcome back."
  10.75-11.97), but over the episode only 12% of 406 segments are punctuated;
  7458 words, nothing lost. `--vad-max-speech-duration-s 20` changes nothing.
- **Model**: `ggml-large-v3-turbo.bin` behaves the same (13% punctuated at
  `-p 4`, 22% at `-p 1`), about four times faster.
- **Fixed chunks** of the first 600 s as separate files: 68% / 35% / 69% / 56%
  punctuated segments for 10 / 15 / 20 / 25 s chunks, against 8% as one file.
  Chunks shorter than a window help, but each chunk still flips a coin, and
  fixed cuts split words (about 4% fewer words).
- **Word timestamps are available regardless of the style**: `-ojf` returns
  token offsets in the lowercase output too (greeting: `hello@10.07 will@10.53
  welcome@10.85 back@11.37 yeah@11.70`, against 10.72-11.96 for the punctuated
  segment, so individual words are up to about 0.6 s early). `--dtw large.v3`
  did not change these offsets in this build.

Conclusion: sentence-shaped segments cannot be forced through whisper.cpp
options on this audio without losing speech. The remaining route to correct
speaker lines is to stop depending on Whisper's segment boundaries: take word
timestamps and re-cut segments at speaker changes (step 2, extended to the
sub-window evidence of step 1b for changes pyannote misses), plus a rule for
the short single-speaker segments this produces. Punctuation and casing would
then be a separate text-restoration question.

#### Prototype: re-cut at speaker changes from word timestamps (2026-10-13)

Not production code; scripts ran outside the repository against the worker's
dependencies. Method:

1. Word timestamps: `whisper-cli -ojf` with the deployed settings but `-p 1`.
   With `-p 4` the token offsets of the later chunks are relative to their
   chunk, so they cannot be used as they are; segment offsets are correct.
2. Speaker track: the known-speaker embedding of a 1.5 s window every 0.25 s
   over the whole file, classified against the job's centroids. A frame is
   clear when margin >= 0.15 and similarity >= 0.30 (the veto's thresholds).
   Clear frames of one speaker form runs (at least two frames); between runs
   of different speakers the boundary is the last frame where the top speaker
   flips.
3. Each Whisper segment is split where its words fall into different turns;
   pieces of a single word are folded into a neighbour.
4. A piece gets its turn's speaker unless a clear frame of another speaker
   lies inside it (0.4 s in from each edge) or, for pieces of 2.5 s and more,
   fewer than 30% of its frames are clear for that speaker.

| | Episode 206 | Episode 205 |
| --- | --- | --- |
| Whisper segments (`-p 1`) | 446 | 737 |
| segments after the re-cut | 521 | 814 |
| named | 491 (2118 of 2164 s) | 762 (2785 of 2885 s) |
| named, single pyannote speaker: agreeing with it | 388 of 397 | 660 of 664 |
| of those, pieces from a split: agreeing | 61 of 69 | 83 of 85 |
| deployed 0.1.4 for comparison: named | 337 of 454 | 520 of 710 |
| extra cost on `atlas` | 48 s track, 2195 s audio | 65 s track, 2997 s audio |

The opening of episode 206 after the re-cut:

```
  0.00-  9.32 Will     hello welcome to another episode of django chat i'm will vincent with carlton gibson
  9.32- 10.18 Will     hey carlton
 10.18- 12.28 Carlton  hello will welcome back yeah
 12.28- 15.76 Will     we're back we did a sneak peek one for the developer
 15.76- 19.47 Will     survey but this is the first start of the fall season so the
 19.47- 21.84 Carlton  long the long stretch until christmas
 21.84- 28.02 Will     and even this is number 206 it still took us what 20 minutes to figure out the logistics of
 28.02- 29.42 Will     the technology so
 29.42- 33.58 Carlton  i just getting the pipes lined up so they actually connect to each other is
 33.58- 36.55 -        harder than it should be yeah yeah okay well
 36.55- 40.34 Will     you look tanned and rested tell us where have you been
 40.34- 48.42 Carlton  well i've just got back um from pescara at brutto in italy where we had the second episode of
```

On episode 205 the block boundary error at 36.68 / 43.76 is cut correctly
("... how important is the survey but" Will / "historically it's been very
important so i" Carlton).

Limits: not checked against the audio by ear; pyannote agreement is the only
independent check and pyannote misses the very interjections this is for, so
disagreement on split pieces is expected and not all of it is error. Word
times are up to about 0.6 s early, so a boundary can be one word off ("yeah"
at 11.93 belongs to Will's next sentence). A piece can collapse to near-zero
length (episode 205, 12.30-12.33 "Hello, Will."). Both episodes have the same
two speakers. The naming rule is far more permissive than the deployed
acceptance policy and was not tuned. Text stays lowercase and unpunctuated.

## Order and verification

1. Done, see "Findings": full-episode diarization of episode 206 with
   retained turns, and the sub-window experiment (1b) on the 10.02-16.40
   segment.
2. Implement step 1 with the config, remote-summary, test and documentation
   work listed there. 1b is done, because it is the only veto that catches
   the episode 206 segment; 1a, which covers the cases where pyannote did see
   a second speaker, is still open.
3. Re-run episode 206 on the worker and compare the sidecar: the 10.02-16.40
   segment must no longer be confident for Will.
4. Run the step 3 experiment.
5. Decide on step 2 from those results, with its own spec update first.

## Out of scope

- django-cast changes. A separate change there already resolves the episode
  for `generate_transcripts --audio-id` and warns when no references are sent.
- Changing the anti-loop defaults without the step 3 experiment.
