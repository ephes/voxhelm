# Mixed-speaker transcript segments (step 1b, turn corroboration and re-cut deployed)

Status: 2026-10-12. Step 1b (sub-window veto) is implemented and deployed with
voxhelm 0.1.3 (control plane and both workers); steps 1a, 2 and 3 are still
plan. Turn corroboration (see "Uncertain segments without a veto") is
deployed with voxhelm 0.1.4, the re-cut at speaker changes with 0.1.5, the
guest guard with 0.1.6 and the diarization label summary with 0.1.7 (see
"Guest without a reference" and the sections after it). Revised after a fresh-context review by gpt-6.1-sol (high
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

Verified in production on 2026-10-13: job
`ba30a7c0-e1c5-4edb-8756-7f22da27aea9` re-ran episode 206 on worker `atlas`
with voxhelm 0.1.4 and reproduced the episode 206 column exactly (337 / 117,
21 corroborated, 25 mixed, Will 130 / Carlton 207); django-cast consumed the
sidecar unchanged. The opening of the episode is unchanged, as expected: 7 of
the first 12 lines carry no name.

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

#### Re-cut as implemented (2026-10-13, deployed with voxhelm 0.1.5)

The prototype is now `build_speaker_track`, `track_turns`, `resegment_result`
and the track acceptance in `classify_segments`
(`transcriptions/known_speaker.py`); the contract is in
`known-speaker-diarization.md`, "Re-cut at speaker changes". Word timings come
from `TranscribeParams.word_timestamps` (whisper.cpp `-ojf -p 1`). Run with the
worker dependencies on the `-p 1` transcripts, veto and turn corroboration on:

| | Episode 206 | Episode 205 |
| --- | --- | --- |
| segments without / with re-cut | 446 / 519 | 737 / 808 |
| confident without / with | 351 / 490 | 488 / 746 |
| uncertain without / with | 95 / 29 | 249 / 62 |
| segments from a split | 138 | 139 |
| named by: whole-segment rule / turn corroboration / speaker track | 365 / 44 / 81 | 493 / 71 / 182 |
| agreeing with a single pyannote speaker, same three groups | 315 of 317 / 44 of 44 / 30 of 37 | 431 of 431 / 71 of 71 / 142 of 146 |
| `mixed_segment_count` without / with | 16 / 0 | 35 / 4 |
| postprocess time on `atlas` without / with | 30 s / 71 s | 41 s / 97 s |

The opening of episode 206 is the same as in the prototype listing above; the
track names "hey carlton", the greeting, "long the long stretch until
christmas" and "the technology so". The speaker-track group is where pyannote
agreement is lowest on episode 206 (30 of 37): it contains the interjections
pyannote attributes to the surrounding speaker, such as the greeting, so the
disagreements are not all errors, but none of them was checked by ear.
Transcription with `-p 1` took 108 s instead of 69 s for episode 206.

Verified in production on 2026-10-13: job
`768290ef-e343-4702-8263-147befa96463` re-ran episode 206 on worker `atlas`
with voxhelm 0.1.5 and reproduced the episode 206 column and the opening
exactly (519 segments, 490 confident, 29 uncertain, 0 mixed, 44 corroborated,
138 from a split, 81 by track, Will 205 / Carlton 285). django-cast matched
all 519 sidecar segments to the Podlove, VTT and DOTe artifacts by start time
without changes on its side. `processing_seconds` was 250.3, against 142.1
for the 0.1.4 job; the local 190 s estimate left out the anonymous pyannote
run. Checked by ear on staging by the maintainer on 2026-10-13 for the
opening: judged acceptable, with two boundaries one word late. "yeah" at the
end of "hello will welcome back yeah" (10.18-12.28) is already Will, and
"the" at the end of "... fall season so the" (15.76-19.47) is already
Carlton. In the first case the word time is about 0.5 s early (token
`yeah@11.93`, spoken around 12.5 in the VAD run); in the second the word time
is right and the track boundary (19.375) is about 0.13 s late, within the
0.25 s track step. Two refinements were measured afterwards and not adopted
(see "Boundary refinements that did not help"). 41 segments are shorter than
1 s, 14 shorter than 0.5 s. The four under 0.3 s are Whisper's own segments;
the others include pieces of a split, such as "hey carlton" (0.86 s).

#### Boundary refinements that did not help (2026-10-13)

Both were measured on episodes 205 and 206 against the raw pyannote turns,
which have fine time resolution where pyannote sees a change. The adoption
rule was: take it only if it measurably helps.

**Snapping a cut to the quietest point nearby** (10 ms RMS envelope, smoothed
50-200 ms, search radius 0.25-0.6 s). Distance of a cut to the nearest
pyannote change gap, for cuts pyannote also sees within 1 s (81 of 98 on
episode 206, 104 of 116 on episode 205):

| | Ep. 206 mean / p90 / within 0.1 s | Ep. 205 mean / p90 / within 0.1 s |
| --- | --- | --- |
| track cut as deployed | 0.104 s / 0.314 s / 56 | 0.191 s / 0.576 s / 53 |
| snap, radius 0.25 s, 100 ms | 0.100 s / 0.387 s / 60 | 0.188 s / 0.624 s / 62 |
| snap, radius 0.4 s, 100 ms | 0.123 s / 0.452 s / 55 | 0.213 s / 0.654 s / 58 |
| snap, radius 0.6 s, 100 ms | 0.151 s / 0.600 s / 53 | 0.251 s / 0.705 s / 54 |

A few more cuts land very close, the tail gets worse, the mean does not move.
At the word level (words within 1 s of a cut, one pyannote speaker), agreement
with pyannote goes from 359 to 351 of 451 on episode 206 and from 403 to 404
of 500 on episode 205 at radius 0.25 s, with 52 and 58 words reassigned. On
the two reported words it fixes "the" (cut 19.375 to 19.31) but not "yeah".

**DTW word times** (`--dtw large.v3`, which only produces values with
`--no-flash-attn`). On the opening they are visibly better (`yeah` at
11.84-12.58 instead of 11.70-12.04, matching the VAD run), and "yeah" moves
to Will; "the" does not move. Over the episodes, the share of words near a
cut that are assigned like pyannote does not improve:

| Word time | Ep. 206 within 0.5 s / 1.0 s | Ep. 205 within 0.5 s / 1.0 s |
| --- | --- | --- |
| offset midpoint (deployed) | 72.8% / 79.4% | 71.9% / 79.8% |
| DTW of the first token | 72.4% / 78.9% | 70.6% / 81.2% |
| DTW midpoint | 71.0% / 78.0% | 71.4% / 81.1% |

Disabling flash attention also changes the transcript itself (episode 206: 459
instead of 446 segments and 7446 instead of 7475 words, so the segment
boundaries shift; the wording was not compared) and costs time
(124 s instead of 108 s for episode 206, 167 s for episode 205).

Neither is adopted. Limit of this evaluation: pyannote misses short
interjections, which is why agreement near cuts is only 70-80% to begin with,
so a real but small gain could hide in that noise. Settling it would need
hand-labelled word boundaries.

#### Guest without a reference (2026-10-13)

The most common request on the Django Chat staging catalogue is three
assigned speakers with references for the two hosts only (115 of 207
episodes), `num_speakers=3`. Measured locally with the host references on two
such episodes; "guest" is the pyannote label that is tied to no host.

| | "EuroPython 2026" (1425 s, guest 933 s) | "Deploy on Day One" (3061 s, guest 2111 s) |
| --- | --- | --- |
| guest speech named, re-cut off (0.1.4 behaviour) | 0 | 1 segment / 3 s |
| guest speech named, 0.1.5 | 0 | 72 segments / 266 s, 71 by the speaker track |
| guest speech named, with the diarization guard | 0 | 1 segment / 3 s |
| hosts named, 0.1.5 / with the guard | 123 / 116 segments | 309 / 215 segments |

0.1.5 therefore gives a guest whose voice resembles a host that host's name.
The guard (`raw_labels_are_known`) lets the track name a segment only when
its diarization labels are tied to known speakers. With it, episodes 205 and
206 are unchanged (746 and 490 confident, same opening). The remaining false
name comes from the whole-segment rule and predates the re-cut. Not examined:
guests with their own reference, more than three speakers, and whether the
re-cut places sensible boundaries inside guest speech (the track cannot tell
a guest from the host they resemble).

#### Guest guard in production and the lost guest names (2026-10-14)

voxhelm 0.1.6 (the guard above) was verified on staging on 2026-10-14:

| Post / audio | Job, worker | Result |
| --- | --- | --- |
| 24 / 22 "Django Deployments in 2025", Eric Matthes, no guest reference | `52d97f63-16bc-4258-a104-a20c7597a6de`, studio, ~562 s | 929 segments, 266 confident (Carlton 138 / Will 128), 663 uncertain, 0 mixed, 40 from a split, 34 by track, 13 corroborated, margin median 0.157. The guest's diarization label (2415 s) gets no name; the 2 s named Will on it are Will. |
| 209 / 203 Calvin Hendryx-Parker, reference from the same episode | `12af0897-f8c2-4aeb-bb81-8fd57ccb8e03`, atlas, ~349 s | 734 segments, 692 confident, 42 uncertain, 1 mixed, 96 by track. Guard not exercised; optimistic. |
| 210 / 204 Mia Bajic, reference from the same episode | `dd4ef443-4562-4bf0-bc2d-c6fc2cbb0e50`, studio, 191.6 s | 447 segments, 424 confident (Mia 260 / Carlton 138 / Will 26), 23 uncertain, 0 mixed, 42 from a split, 103 by track. No cross-assignments; optimistic. |

The guard worked as designed, but the maintainer judged post 24 broken: the
guest is listed on the episode and none of his lines carried a name. Before
the re-run they had. The earlier transcript was anonymous pyannote output
("Speaker N" in the Podlove/VTT/DOTe speaker fields), and django-cast named
the guest through an approved "Speaker 3 → Eric Matthes" mapping applied at
render time. A known-speaker run writes the public artifacts unlabeled, so
the artifact fingerprint changed, the approved mappings went stale, and
nothing named the guest any more. On staging 182 of 207 transcripts have
such approved mappings (520 rows), so every known-speaker re-run of one of
them would have lost a reviewed guest name the same way. The guard did not
cause this; 0.1.5 named the guest as a host instead.

Fix, fully automatic (maintainer decision, 2026-10-14: no hand-seeded guest
references):

- voxhelm 0.1.7 adds `raw_diarization_labels` to the known-speaker summary:
  every diarization label with its seconds of speech and the known speaker
  the confident-segment vote ties it to, or null (contract:
  `known-speaker-diarization.md`, "Response"). The control plane allowlists
  it for remote workers. Classification is unchanged. Deployed to the
  control plane and both workers on 2026-10-14.
- django-cast (develop `1f03c4c3`) names the guest at apply time when it is
  unambiguous: exactly one label tied to no known speaker, every known
  speaker tied to a label, the audio in exactly one episode, and exactly one
  visible assigned contributor without a reference. The uncertain segments on
  that label get the contributor's name; segments flagged `mixed_speakers`
  do not, and editor decisions win. For sidecars from 0.1.6 and earlier it
  falls back to the same vote over the segments.

Re-applied to the existing 0.1.6 sidecar of post 24, without a re-run:
Speaker 3 is named "Eric Matthes", 625 lines carry his name, Carlton 138,
Will 128, 38 stay blank. Episodes where everyone has a reference are not
changed by the rule.

Boundary errors seen in the text on staging, not checked by ear:

- Post 24, 30.00 s: Will's segment starts with Eric's "thank you for having
  me". 01:28: Eric's trailing "nice." in Will's segment (one word; one-word
  pieces are folded on purpose, `RESEGMENT_MIN_PIECE_WORDS = 2`).
- Post 210, 24.96 s: Carlton's "hello will" inside Will's segment; unclear
  "oh yes" (end of Mia's 31.69 segment) and "uh i'll be there" (end of Will's
  37.74 segment).

Both 30.00 s and 24.96 s sit where the ASR text flips to lowercase and
unpunctuated, at a whisper 30 s window boundary. The 30.00 s error was
reproduced locally with the production settings (whisper.cpp `-ojf -p 1 -l
en -np --max-context 0 --suppress-nst`, pyannote `num=3`; 919 segments, 280
confident, 639 uncertain, close to staging). Cause: the speaker track only
compares against the references, so a guest without one produces no run of
its own; Eric → Will is not a change for the track and the re-cut never
splits there. pyannote does see it: "thank you for having me" (words
30.27-31.46) lies on the guest's label (turn 30.73-31.35). The 24.96 s case
is different (Carlton has a reference); probably too short for a track run
(two clear frames at a 0.25 s hop with 1.5 s windows). Not examined yet.

#### Re-cut at the turns of unmapped labels (2026-10-14, deployed with voxhelm 0.1.8)

A first pass runs the postprocess as deployed and reads which diarization
labels the confident segments tie to known speakers. A second pass overlays
the pyannote turns of the other labels on the speaker-track turns before the
re-cut, so a segment is also split where speech moves to or from a label
nobody is tied to. Measured locally with the host references; the metric is
the seconds of speech on unmapped labels (per pyannote) inside segments that
carry a host's name:

| Episode (pyannote) | guest speech in named segments, 0.1.6 → prototype | named s | known-speaker speech left unnamed | segments |
| --- | --- | --- | --- | --- |
| Post 24, Eric (`num=3`) | 30.3 s / 20 segments → 11.7 s / 5 | 1243 → 1266 | 117 → 96 s | 919 → 1107 |
| "Deploy on Day One", Calvin (`num=3`) | 76.4 s / 45 → 26.2 s / 11 | 903 → 850 | 118 → 151 s | 700 → 781 |
| "EuroPython 2026", Mia, no Mia reference (`num=3`) | 14.9 s / 10 → 2.7 s / 1 | 417 → 437 | 41 → 14 s | 375 → 408 |
| Episode 206 (`num=2`) | 0 → 0, identical | | | |
| Episode 205 (`num=2`) | 0 → 0, identical | | | |

The 30.00 s case is fixed ("thank you for having me" becomes its own
segment, which the django-cast rule now names after the guest). Caveats: the
metric is circular, because it is measured against pyannote and the
prototype cuts at pyannote turns; on "Deploy on Day One" 33 s of host speech
lose their name; segments grow by 10-20 %; and the first pass costs roughly
40-60 s per job. A cheaper label map (for example from the track frames,
which exist before the re-cut) is untested and could map a guest who
resembles a host to that host. With guest naming in place this is the next
candidate; it would be judged on the re-run staging transcript of post 24.

Implemented in `run_known_speaker_postprocess` (`unmapped_label_turns`,
`overlay_unmapped_turns`; contract: "Re-cut at speaker changes"). The label
map is the same vote as `raw_diarization_labels`; the second pass runs only
when at least one label is tied to a known speaker and at least one is not.
Re-measured with the implementation on the same data: all five rows of the
table above are reproduced exactly, episodes 205 and 206 classify once and are
identical to 0.1.7, and on post 24 the 30.00 s segment becomes "thank you for
having me" (30.00-31.46, unnamed, Speaker 3) followed by Will's "so i was
thinking of you recently ..." (31.46-36.12). With the django-cast rule the
first piece is named after the guest. Wall times in this run did not show the
extra pass clearly (model warm-up falls into the first pass); on staging
`processing_seconds` of the post 24 re-run will show it.

Verified in production on 2026-10-14: job
`25f1753f-453f-4a1a-9c4b-e3284ac40507` re-ran post 24 on worker `studio` with
voxhelm 0.1.8: 1132 segments, 297 confident (Carlton 154 / Will 143), 835
uncertain, 0 mixed, 356 from a split, 47 by track, 18 corroborated, close to
the local 1107. `raw_diarization_labels`: Speaker 1 → Carlton (627 s),
Speaker 2 → Will (706 s), Speaker 3 → null (2459 s). Per pyannote label, the
guest's label has 2403 s unnamed and 4 s named Will; Will's label has 704 s
Will, 2 s Carlton and 32 s unnamed. After django-cast applied the
suggestions with the guest rule, the public transcript names Eric 768 lines,
Carlton 154, Will 143, and leaves 67 blank (0.1.6 sidecar re-applied: 625 / 138 / 128 /
38). The opening now reads: 20.00-29.70 Will ("... welcome back, Eric"),
30.00-31.46 Eric ("thank you for having me"), 31.46-36.12 Will ("so i was
thinking of you recently ..."). From 30 s on the text is still lowercase and
unpunctuated (see step 3). `processing_seconds` was 675.9 for 4079 s of
audio. The 0.1.6 run of the same post took about 562 s wall time on the same
worker, which is a different measure, so the cost of the second pass is not
isolated. Not yet checked by ear.

#### Speaker count and ASR pilot (2026-10-15)

Trigger: on episode 219 ("Hiring in 2026 - Jon Gould", 55.8 min, Will,
Carlton and Jon, references for all three) 61 % of segments were unnamed on
staging and the first 20 s carried wrong names. Locally with the same
references: pyannote with `num_speakers=3` gives labels of 895 s / 1988 s /
45 s, so it spends one label on noise and merges Carlton and Jon. Without a
count it gives Will 889 s, Jon 1378 s, Carlton 611 s and two noise labels
(32 s, 18 s). Carlton's "hello will" (11.68-12.40) is on Will's label in
every run, and the speaker track has no clear frame there.

Gold standard: the maintainer labelled 166 sentences (Parakeet sentence
boundaries, about 10 min) in eight excerpts of six episodes: 219 (opening and
middle), Eric Matthes (opening and middle), Mia Bajic, Calvin Hendryx-Parker,
205 and 206 (openings). 157 single-speaker sentences that lie fully inside
their excerpt are scored per word, after django-cast's guest rule:

| ASR / pyannote count | right | wrong name | unnamed |
| --- | --- | --- | --- |
| whisper.cpp, count pinned (production until 2026-10-15) | 92.2 % | 2.5 % | 5.2 % |
| whisper.cpp, no count | 95.4 % | 2.5 % | 2.0 % |
| Parakeet TDT 0.6B v3, count pinned | 93.4 % | 1.1 % | 5.5 % |
| Parakeet TDT 0.6B v3, no count | 96.1 % | 1.1 % | 2.9 % |

Without a count pyannote gives the same turns on the Eric, Mia, Calvin and
205 episodes; on 206 it adds a 31 s label nobody is tied to (unnamed segments
14 -> 32 with whisper; the scored opening is the same in both runs). Dropping the
count added no wrong name anywhere. django-cast stopped sending
`num_speakers` for known-speaker jobs on 2026-10-15 (`7205f730`).

Most remaining wrong names are in Calvin's opening (9.7-10.6 % of its
words): pyannote puts his "Thanks, Will." and "jumping right in ... Let's do
this." on Will's label, and the track hears Will for the first 232 s (the
guest resembles that host). The track names these pieces although their own
embeddings are near zero for everyone.

Short segments: segments under `min_segment_duration` are zero-padded before
embedding (`pad_to_min_duration`). Unpadded, eight hand-picked short pieces
of the 219 opening all score highest for the right person ("Hello, Will."
Carlton 0.22 / Jon 0.15 / Will 0.09); padded, that one flips to Will 0.05 and
margins fall to about 0.03. Not changed yet.

Parakeet (parakeet-mlx 0.5.3, Python 3.13 venv, model
`mlx-community/parakeet-tdt-0.6b-v3`): punctuation and casing throughout,
sentences that end at speaker changes, "DjangoCon"/"PyCon" where whisper.cpp
wrote "jenga con"/"pike on", word times closer to the pyannote turns, and
about 8x faster than whisper.cpp `-p 1`. Names stay wrong ("John Gould",
"Foxy Talent"). With the default 120 s chunks punctuation is lost near chunk
seams (run-on "sentences" of 30-115 s). Without chunking (local attention)
peak memory footprint is 40-51 GB for 1:17-2:16 h episodes; with
`--chunk-duration 600 --overlap-duration 30 --local-attention` it is 11.5 GB
for 2:16 h (65 s). (Correction, 2026-10-15: the pilot's Parakeet files were
unchunked; chunked runs have long unpunctuated stretches on some episodes,
see "Local verification" in `2026-10-15_parakeet_asr_backend.md`.) whisper.cpp needs about 5.8 GB
and 525 s for the same file. German (two python-podcast episodes): both are
usable; Parakeet has consistent punctuation but occasional meaning-changing
word errors, whisper.cpp is a bit more literal and 58 % of its segments start
lowercase. Word error rates were not measured.

Glossary correction (episode title, summary and show notes as glossary): a
local Qwen3.6-27B (4-bit MLX, 93 s for the episode) proposes replacements,
and a filter keeps only those whose original and replacement are similar
(ratio >= 0.6) and that do not merely extend a name. On 219 that kept 26
fixes, 24 right ("John" -> "Jon", "Foxy Talent" -> "Foxley Talent", "Gilbert
Rile" -> "Gilbert Ryle"); the two wrong ones came from a missing glossary
entry. Qwen2.5-0.5B and Gemma 4 E2B were unusable.

Four speakers, checked without gold labels (Parakeet, host references only):
on "Django Fellows - Natalia Bidart & Sarah Boyce" and "PyCon US 2026 -
Elaine Wong & Jon Banafato" pyannote gives the same four labels with
`num_speakers=4` and without a count. Both hosts' labels are tied to them and
named almost entirely (Fellows: 1255 of 1278 s; PyCon: 1041 of 1068 s); the
two guests' labels carry a host name for 0 s and 7 s and are otherwise
unnamed, because neither guest has a reference here and django-cast's guest
rule needs exactly one untied label.

Limits: one listener; the sentence boundaries come from Parakeet; ten
minutes are a sample.

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
