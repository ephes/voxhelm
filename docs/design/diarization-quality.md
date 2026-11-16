# Speaker names for short segments

Status: slices 1–5 implemented and merged to `main`, 2026-11-15. Real-data
qualification done (see [Clerk qualification](#clerk-qualification-2026-11-15)):
all pooled targets met, but the episode-80 guest safety target is missed, so
the server was not deployed; remote workers need a PyPI release first.
Slices 6–7 remain deferred.

Trigger (maintainer, 2026-11-15): transcript quality is acceptable overall,
but short interjections often carry no speaker or the wrong one. The wish is
a simple improvement without new infrastructure.

## Summary

- Today about 10 % of transcript lines carry no name, but they hold only
  2-3 % of the speech. 80 % of them are shorter than 1.5 s.
- The cause is one line of the acceptance policy plus one preprocessing step:
  segments under 1.5 s can never be accepted on their own voiceprint, and they
  are padded with silence before embedding, which destroys the embedding
  (right speaker in 40-55 % of segments under 1 s, against 70-89 % unpadded).
- Filling such lines from the neighbouring line or from the pyannote label is
  not an option: on blank short segments the neighbour is right in 42 % and
  the label in 60 % of cases, because the typical blank line is an
  interjection by the *other* person.
- Chosen design: classify the still-unnamed short segments a second time,
  unpadded, against voiceprints taken from the episode itself (the segments
  already accepted in this job), and include a voiceprint for a guest who has
  no reference. Measured in a spike: two thirds of the blank short lines get a
  name, 86 % of those names are right, and the two Django Chat test episodes
  go from 69 and 61 blank lines to about 18 and 19.
- A second, independent change is to align against pyannote's exclusive
  turns (+3 points of correct label time on two gold episodes).
- No new models, services or dependencies. Extra cost is about 10 s of
  embedding time per episode.

## Goal and non-goals

Goal: fewer unnamed and fewer wrongly named short lines in known-speaker
transcripts (Django Chat, python-podcast), measured against the existing
truth data, without making long lines worse.

Non-goals:

- Speech the ASR drops. 3.9 % of the truth speech in the pilot episodes has
  no transcript segment at all (interjections during crosstalk). That is an
  ASR question and stays as it is.
- New segment boundaries. The episode voiceprint pass names existing
  segments; it does not split or merge them, so start times and segment
  counts stay stable for consumers. (The exclusive-turns change can shift
  existing cuts in guest episodes, see "Chosen design".)
- Anonymous diarization quality (speakers merged by pyannote clustering, see
  `specs/diarization-quality-research.md`).
- New diarization backends, hosted services, or a new embedding model.
- django-cast changes. The consumer needs none for this design.
- Punctuation, casing and word errors.

## Current state

### Pipeline

1. **ASR.** Four local backends in `transcriptions/service.py`, three of
   which matter here (`WhisperKitBackend`, line 562, is an experimental
   opt-in without word timings and is not used for these jobs):
   `MlxWhisperBackend` (line 165, `word_timestamps=False` at line 185),
   `WhisperCppBackend` (line 200; `-ojf` and one processor when word timings
   are requested, lines 232-239) and `ParakeetBackend` (line 329; 600 s
   chunks, local attention, optional 30 s line cap, lines 391-398).
   Known-speaker jobs take Parakeet when
   `VOXHELM_KNOWN_SPEAKER_STT_BACKEND=parakeet` (`config/settings.py:408`,
   routing at `transcriptions/service.py:1024-1056`), with whisper.cpp as the
   fallback. The two Django Chat jobs measured below ran on Parakeet (job
   metadata). Only Parakeet and whisper.cpp return word timings (`TranscriptionWord`, `transcriptions/service.py:82`).
2. **Sanitizer.** `sanitize_result` (`transcriptions/sanitizer.py:69`) drops
   punctuation-only cues and collapses repetition loops before diarization.
3. **Anonymous diarization.** `PyannoteDiarizationBackend`
   (`transcriptions/diarization.py:150`) runs
   `pyannote/speaker-diarization-3.1` (worker setting
   `VOXHELM_PYANNOTE_MODEL`). `extract_pyannote_annotation`
   (`transcriptions/diarization.py:267-276`) prefers `speaker_diarization`,
   whose turns may overlap, over `exclusive_speaker_diarization`.
   `choose_speaker_for_segment` (`transcriptions/diarization.py:427-451`)
   gives a segment the label of the single turn with the longest overlap.
   For the anonymous strategy this is the final answer
   (`apply_speaker_labels`, line 323).
4. **Known-speaker postprocess**, `run_known_speaker_postprocess`
   (`transcriptions/known_speaker.py:1449`), for jobs with
   `strategy=pyannote_known_speaker`:
   - One centroid per speaker from up to 16 windows of 5 s per reference
     (`REFERENCE_WINDOW_SECONDS`, `REFERENCE_MAX_WINDOWS`, lines 141-142;
     `build_speaker_centroids`, line 716). Embedding model
     `pyannote/wespeaker-voxceleb-resnet34-LM` (line 68).
   - Speaker track: a 1.5 s window every 0.25 s over the whole file
     (`build_speaker_track`, line 758), turned into turns (`track_turns`,
     line 798). At most 32,768 frames; longer audio gets a wider step instead
     of failing (`speaker_track_hop`, line 699; about 0.33 s for three hours).
   - Re-cut: each segment is split where its words fall into different track
     turns (`split_segment_at_turns`, line 844); pieces of one word or under
     0.3 s are folded into a neighbour (lines 114-115). When pyannote has a
     label no known speaker is tied to, its turns are cut in as well
     (`overlay_unmapped_turns`, line 980).
   - Classification, `classify_segments` (line 1165). A segment is accepted
     on its own embedding when it lasts at least 1.5 s, its top similarity is
     at least 0.55 and its margin at least 0.15 (`is_confident`, line 412;
     defaults at lines 69-71). Before embedding, a shorter segment is padded
     with zeros to 1.5 s (`pad_to_min_duration`, line 663, called at
     line 1208). Three more rules follow: the mixed-segment veto
     (`find_mixed_windows`, line 457), turn corroboration (line 1148) and the
     speaker track (`track_speaker_for_segment`, line 1051), which is the only
     rule that can name a segment under 1.5 s. The track names a segment only
     when every pyannote label on it is tied to a known speaker
     (`raw_labels_are_known`, line 1124).
   - Everything else is `speaker_uncertain=true` with `speaker=null`. The
     sidecar keeps the candidates and the pyannote label with the longest
     overlap (`raw_diarization_speaker`, line 1281).
5. **Consumer.** django-cast applies the confident names with `smooth=False`
   (`src/cast/views/transcript.py:319`). Before that it names a guest without
   a reference: when exactly one pyannote label is tied to nobody and exactly
   one contributor has no reference, every uncertain, non-mixed segment on
   that label gets the guest's name
   (`src/cast/transcripts/services.py:203-226`,
   `src/cast/transcripts/known_speakers.py:122-170`).

Already in place and therefore not proposed again: word timings and the
re-cut at speaker changes (0.1.5), no pinned speaker count for known-speaker
jobs (measured as harmful on 2026-10-15), and Parakeet for sentence-shaped
segments. The history and the numbers are in
`specs/2026-10-12_mixed_speaker_segments.md` and
`specs/2026-10-15_parakeet_asr_backend.md`.

### What this means for a short interjection

A 0.7 s "Yeah." from Carlton inside Will's sentence:

- pyannote usually has no turn of its own for it, so its label is Will's.
- Its own embedding cannot name it: the duration rule excludes it, and the
  zero padding pushes every similarity towards zero anyway.
- The track can name it only if at least one clear 1.5 s frame is centred
  inside those 0.7 s. Such a frame is mostly Will's speech.
- So it stays blank, or, if it sits on the guest's label, django-cast gives
  it the guest's name without looking at the voice.

## Measurements

All measurements are read-only: published transcripts, stored sidecars, the
control-plane job table, and spikes in a scratch directory on `studio` run
with the deployed worker's interpreter (voxhelm 0.1.13, pyannote.audio 4.0.4,
torch 2.12.0). No job was submitted and nothing was changed on a host.

### Data

- **Django Chat staging**, two episodes with both hosts referenced and a
  guest without a reference at the time of the job: "Databases, Django APIs,
  and AI - Haki Benita" (audio 210, job `da97bdc8`, 2026-11-13, 67 min) and
  "Django in Space - Loes Crama" (audio 209, job `c0e84980`, 2026-10-22,
  55 min). No speaker truth exists for them; they are judged by counts and by
  reading the text.
- **python-podcast pilot**, eight episodes (14.4 h) with stored sidecars from
  the Parakeet pilot of 2026-10-17 and a speaker truth at 10 frames per
  second: three from hand-corrected multitrack transcripts (audio 77, 78, 80)
  and five from per-speaker stems (2, 3, 59, 64, 67). A frame has a truth
  speaker only where one speaker is clearly alone. The truth files and the
  scoring method come from the bulk run
  (`var/handoff/2026-10-19_python_podcast_parakeet_bulk/`, gitignored, on
  `studio`); `specs/2026-10-15_parakeet_asr_backend.md` describes them.
- The embedding spike ran on four of the eight (59, 77, 78, 80; 4.6 h, 2094
  segments under 2.5 s, 1738 of them with a clear truth speaker: at least
  three truth frames, 80 % of them one speaker).

### Before: how much is unnamed, and where

Published lines after django-cast applied the names:

| | lines | unnamed lines | unnamed speech time |
| --- | --- | --- | --- |
| Django Chat, Haki Benita | 877 | 69 (7.9 %) | 2.3 % |
| Django Chat, Loes Crama | 608 | 61 (10.0 %) | 2.3 % |
| python-podcast, 8 episodes | 11,392 | 1,172 (10.3 %) | 3.3 % |

Unnamed lines by line duration:

| duration | Haki Benita | Loes Crama | python-podcast (8) |
| --- | --- | --- | --- |
| under 0.5 s | 9 of 38 (24 %) | 2 of 21 (10 %) | 227 of 502 (45 %) |
| 0.5-1 s | 35 of 99 (35 %) | 35 of 79 (44 %) | 539 of 1173 (46 %) |
| 1-1.5 s | 11 of 88 (12 %) | 12 of 49 (24 %) | 188 of 1158 (16 %) |
| 1.5-2.5 s | 10 of 147 (7 %) | 7 of 87 (8 %) | 84 of 2135 (4 %) |
| 2.5-5 s | 2 of 216 (1 %) | 4 of 127 (3 %) | 87 of 2948 (3 %) |
| 5 s and more | 2 of 289 (1 %) | 1 of 245 (0 %) | 47 of 3476 (1 %) |

Against the truth, the eight python-podcast episodes score 94.9 % of speech
frames right, 2.2 % wrong and 2.8 % unnamed (per episode: 93.2-96.3 % right,
1.3-3.1 % wrong, 1.9-5.2 % unnamed). Another 3.9 % of the truth speech has no
transcript segment (see non-goals).

### Why the lines are unnamed

The sidecars of the two Django Chat jobs, by the rule that decided each
segment:

| | Haki Benita | Loes Crama |
| --- | --- | --- |
| named by the whole-segment rule | 277 | 202 |
| named by turn corroboration | 14 | 9 |
| named by the speaker track | 73 (68 under 1.5 s) | 48 (43 under 1.5 s) |
| named by django-cast's guest rule | 444 (102 under 1.5 s) | 288 (57 under 1.5 s) |
| unnamed, on a host's pyannote label, under 1.5 s | 54 | 48 |
| unnamed, on a host's label, 1.5 s or longer, weak embedding | 12 | 12 |
| unnamed, mixed-segment veto | 2 | 0 |
| unnamed, no pyannote turn | 1 | 1 |

Apart from the vetoed ones, every unnamed line sits on a host's label or on
none. Lines on the guest's label are not unnamed, because the guest rule
names all of them. Of the 54 and 48 short ones, 46 and 37 miss both the
similarity and the margin threshold; the examples below have top
similarities of 0.01-0.17, the signature of the zero padding.

The opening of the Haki Benita episode shows both problems (sidecar values,
then the published name):

```
 16.48  0.80s  pyannote: Carlton's label  top sim 0.17  -> unnamed    "Hey Carlton."
 24.08  1.92s  pyannote: Carlton's label  top sim 0.15  -> unnamed    "I was Thank you for having me."
 26.00  0.80s  pyannote: Carlton's label  top sim 0.01  -> unnamed    "No, no, no."
1223.60 0.80s  pyannote: guest's label                  -> Haki Benita "Go on, Will."
```

"Hey Carlton." is Will, "thank you for having me" is the guest, and "Go on,
Will." is a host; pyannote put all of them on the label of the surrounding
speaker.

### Which signal knows who spoke a short segment

python-podcast, four episodes, segments under 2.5 s with a clear truth
speaker. "Label" is the known speaker tied to the segment's pyannote label,
"neighbour" the name of the previous line, "padded" today's embedding against
the reference centroids. "Tiled" repeats the segment's own audio up to 1.5 s
instead of padding it with zeros. "Episode" centroids are the mean embedding
of up to 150 segments per speaker that the whole-segment rule accepted in the
same job (3 s or longer, highest margin first).

Share of segments whose top candidate is the right speaker:

| duration | n | label | neighbour | padded, reference | tiled, reference | tiled, episode |
| --- | --- | --- | --- | --- | --- | --- |
| under 0.6 s | 146 | 59 % | 44 % | 40 % | 71 % | 75 % |
| 0.6-1.0 s | 334 | 69 % | 46 % | 55 % | 87 % | 89 % |
| 1.0-1.5 s | 441 | 74 % | 54 % | 93 % | 95 % | 96 % |
| 1.5-2.5 s | 817 | 87 % | 61 % | 98 % | 98 % | 98 % |

The same for the 295 segments that are unnamed today, with the share that
clears a margin of 0.15 ("coverage") and the share of those that is right
("precision"):

| signal | top candidate right | coverage at 0.15 | precision at 0.15 |
| --- | --- | --- | --- |
| neighbour | 42 % | | |
| neighbour, only where previous and next line agree (90) | 52 % | | |
| label | 60 % | | |
| padded, reference (today) | 58 % | 2 % | 100 % |
| tiled, reference | 78 % | 41 % | 85 % |
| unpadded as is, episode | 82 % | 66 % | 87 % |
| tiled, episode | 82 % | 66 % | 86 % |
| 0.2 s more audio on each side, episode | 82 % | 65 % | 87 % |

Findings:

- Zero padding is the largest single loss. Under 1 s it turns a usable
  embedding into noise. Tiling and the unpadded audio are equivalent; tiling
  keeps the input length the model already sees today.
- Episode centroids do not change who is on top, but they raise the margins:
  66 % instead of 41 % of the blank segments clear 0.15. Reference clips come
  from other episodes with another microphone and room. Collecting longer or
  more reference clips per speaker addresses the same gap with manual work and
  is not needed.
- Precision does not rise with the margin (86 % at 0.15, 86 % at 0.20, 88 % at
  0.25, 91 % at 0.30 with 22 % coverage) and a similarity floor adds one
  point. Part of the remaining 14 % is truth and boundary noise; it was not
  separated from real errors. Per episode: 100 % (59), 81 % (77), 100 % (78),
  71 % (80).
- For comparison, short lines that are named today by the speaker track are
  87 % right under 1 s and 97 % between 1 and 1.5 s. The new names are as
  reliable as the ones already published for that length.
- The neighbour and the label are worse than a coin toss weighted by
  speaking time. Filling blank lines from either (which is what `smooth=True`
  does) adds more wrong names than right ones on short lines.

Rules simulated on the stored sidecars of all eight episodes, without new
embeddings ("precision" over the newly named segments with a truth speaker):

| rule | right | wrong | unnamed | precision of new names |
| --- | --- | --- | --- | --- |
| today | 94.9 % | 2.2 % | 2.8 % | |
| name by pyannote label | 96.2 % | 2.9 % | 0.9 % | 56 % |
| carry the previous name forward | 96.4 % | 3.6 % | 0.0 % | 41 % |
| name when label and padded top candidate agree | 95.9 % | 2.4 % | 1.7 % | 72 % |

These fill the frame statistics, because blank segments are short, but most
of the names they add to short lines are wrong.

Effect of "tiled, episode, margin 0.15" on the four spike episodes: of the
speech frames inside blank short segments (2.0 % of all truth speech), 52 %
become right, 10 % wrong and 38 % stay unnamed. Per episode that moves about
94.4 / 1.7 / 3.9 % (right / wrong / unnamed) to about 95.4 / 1.9 / 2.7 %, and
the unnamed lines from 495 to roughly 210 of 4199 (11.8 % to about 5 %).

### Guest without a reference

For the two Django Chat episodes the spike also built a centroid for the
untied pyannote label from its uncertain single-label segments of 3 s and
more, and classified every uncertain segment (tiled under 1.5 s) against the
two host episode centroids and that guest centroid.

| | Haki Benita | Loes Crama |
| --- | --- | --- |
| sanity: guest-rule lines of 3 s and more whose top candidate is the guest | 255 of 256 | 194 of 194 |
| unnamed lines decided at margin 0.15 | 51 of 69 | 42 of 61 |
| of those: a host / the guest | 39 / 12 | 40 / 2 |
| unnamed lines left | 18 (2.1 %) | 19 (3.1 %) |
| short lines named after the guest today that the voiceprint gives to a host | 17 of 102 | 8 of 57 |

Read against the text, the decisions are plausible where the text gives it
away: "Hey Carlton." goes to Will, "I was Thank you for having me." to the
guest, "okay, fair enough." to Carlton, and "Go on, Will.", "Oh, that's the
problem." and "Yeah, yeah, and why not?" move from the guest to a host.
"Hello, Will." at 10.32 s in the Loes Crama episode stays undecided
(similarity 0.02). Nobody listened to these lines; the listening check is
part of the rollout slice.

### pyannote variants

Episodes 78 and 80, no speaker count, labels mapped to truth speakers by
majority. "Label right" is over ASR segments under 1.5 s with a clear truth
speaker.

| model, annotation | 78: frames right / label right | 80: frames right / label right |
| --- | --- | --- |
| speaker-diarization-3.1, regular (today) | 91.2 % / 64 % | 87.1 % / 61 % |
| speaker-diarization-3.1, exclusive | 94.0 % / 76 % | 90.7 % / 66 % |
| community-1, regular | 90.6 % / 64 % | 86.2 % / 60 % |
| community-1, exclusive | 93.2 % / 75 % | 89.3 % / 61 % |

Both models run in about one minute per episode. `community-1` is not better
here. The exclusive annotation is: about 3 points more correct label time and
better labels on short segments. Overlapping turns were flattened for this
comparison by letting the shorter turn win, which is not the same as
Voxhelm's longest-overlap choice per segment, so the slice below re-measures
it through the shipped code. Even the best labels are right on only 66-76 %
of short segments, far below the tiled embedding.

## Options

| option | evidence | decision |
| --- | --- | --- |
| A. Tile short segments instead of zero padding | top candidate right 40-55 % to 71-87 % under 1 s | adopt, as part of C |
| B. Longer or more reference clips per speaker | same gap as C, needs manual work per speaker | not needed |
| C. Second pass against episode voiceprints for unnamed short segments | names 66 % of blank short lines at 86 % precision | adopt |
| D. Voiceprint for the guest's label inside C | 51 of 69 and 42 of 61 blank lines decided; 17 and 8 lines move from guest to host | adopt |
| E. Fill from the pyannote label | 60 % right on blank short lines | reject |
| F. Fill from neighbours (`smooth=True`, sandwich) | 42-52 % right | reject |
| G. Align on pyannote's exclusive turns | +3 points correct label time | adopt if the harness confirms |
| H. `speaker-diarization-community-1` | not better on two episodes | reject for now |
| I. Tune pyannote segmentation parameters (`min_duration_off`, collar) | not measured; pyannote assigns the interjection to the surrounding speaker in every run so far | not pursued |
| J. Pin or bound the speaker count from the contributor list | measured harmful on 2026-10-15 (a label spent on noise, two people merged) | already removed |
| K. Word-level speaker assignment and re-cut | deployed since 0.1.5 | already done |
| L. Lower `min_segment_duration` globally | padded embeddings are 55 % right at 0.6-1 s, so it needs A first; then it equals C without the stricter evidence | covered by C |
| M. Finer speaker track (shorter window or hop) | 1.0 s windows were too noisy on 2026-10-12; cost grows with the frame count | reject |

## Chosen design

Two changes in Voxhelm. The first keeps segment boundaries, the sidecar
version and the public artifacts as they are. The second is a separate,
smaller change with a wider reach: it can move the cut at unmapped labels in
known-speaker jobs with a guest, and it changes the `Speaker N` labels that
anonymous jobs publish.

### 1. Episode voiceprint pass for short unnamed segments

After `classify_segments` has run (both passes, where the re-cut at unmapped
labels applies), a new step in `run_known_speaker_postprocess`:

1. **Episode centroids for known speakers.** For each known speaker, embed up
   to 150 segments that the whole-segment rule accepted for that speaker
   (no accept reason, not vetoed), each 3 s or longer, highest margin first,
   and average them. A known speaker with fewer than 5 such segments does not
   take part in the pass: no short segment is named after them. Reference
   centroids are never mixed with episode centroids, because episode
   centroids score systematically higher and would win by default.
2. **Centroids for untied labels.** For each pyannote label that the label
   vote ties to nobody (`summarize_raw_labels`, line 1359), take the
   uncertain, non-mixed segments of 3 s and more whose only label it is
   (`single_raw_label`, line 1093), longest first, up to 150. With at least 5,
   build a centroid named after the label.
3. **Whether the pass runs.** pyannote puts a short interjection on the label
   of the surrounding speaker, so the label of a short segment says little
   about who spoke it, and a guest's interjection usually sits on a host's
   label. The only protection against naming it after a host is that the
   guest's own voiceprint competes. The pass therefore runs for a job only
   when at least two centroids exist and every untied label with 30 s of
   speech or more has a centroid. Otherwise nothing changes for that job and
   the summary says why (`episode_voiceprint_skipped`: `"unresolved_label"`
   or `"too_few_centroids"`). Untied labels under 30 s are treated as noise.
4. **Candidates.** Every segment that is still uncertain, not flagged by the
   mixed-segment veto, and between 0.3 s and 2.5 s long, whatever its pyannote
   label.
5. **Classification.** Take the segment's audio, tiled to 1.5 s when shorter,
   embed it, and compare with all centroids from 1 and 2. The decision needs a
   margin of at least `short_segment_margin` (default 0.15) between the first
   and the second candidate.
   - Top is a known speaker: the mixed-segment veto runs against that
     speaker first, exactly as it does for the speaker track
     (`find_mixed_windows` with the reference centroids and the existing
     options; it only has windows for segments longer than
     `mixed_window_duration`, here the candidates of 1.5-2.5 s). If it finds a
     window for another speaker, the segment stays uncertain with
     `speaker_uncertain_reason="mixed_speakers"`. Otherwise the segment gets
     that `speaker`, `speaker_uncertain=false` and
     `speaker_accept_reason="episode_voiceprint"`. The spike did not apply
     the veto; it can only lower the number of new names among the
     1.5-2.5 s candidates, which are about one in seven of the unnamed lines.
   - Top is an untied label: the segment stays uncertain, and its
     `raw_diarization_speaker` becomes that label. django-cast's guest rule
     then names it like every other segment on the guest's label. When this
     changes the value, the label with the longest overlap is kept in
     `raw_diarization_overlap_speaker`, so the pyannote result stays
     auditable.
   - No decision: unchanged.
6. **Guest lines that are a host's.** Uncertain short segments on an untied
   label are candidates like any other. When a known speaker wins with the
   margin, the segment becomes confident for that speaker, and django-cast
   applies the confident name before the guest rule. This is what moves
   "Go on, Will." away from the guest.

Sidecar: segments the pass decided carry an additional optional
`speaker_episode_candidates` list (`speaker` or `label`, and `similarity`),
and the existing `speaker_candidates`, `speaker_confidence` and
`speaker_margin` keep describing the whole-segment embedding against the
references. The summary gets `episode_voiceprint_segment_count`,
`episode_voiceprint_relabelled_count`, `episode_voiceprint_skipped` (null
when the pass ran) and `short_segment_margin`. Version stays 1; all fields
are additive.

Contract change: `raw_diarization_speaker` is today always the label with the
longest overlap. With this design it is the label the segment is assigned to,
which for relabelled segments is the voiceprint's choice; the overlap label
moves to `raw_diarization_overlap_speaker` on exactly those segments.
`raw_diarization_labels` in the summary keeps counting seconds from the raw
turns. `specs/known-speaker-diarization.md` is updated with the slice that
introduces it.

Configuration: one request option, `diarization.known_speaker.short_segment_margin`
(finite, non-negative, default 0.15), following the existing options in
`KNOWN_SPEAKER_NUMBER_OPTIONS`. `0` turns the pass off, as it does for
`turn_corroborated_min_similarity` and `resegment_hop`; unlike
`auto_accept_margin`, it does not mean "no margin required", and the contract
says so. Running the pass without a margin is not offered. The other numbers
(150, 5, 3 s, 30 s, 0.3 s, 2.5 s) are module constants like the track's.

Reasons for this shape:

- It uses the evidence that measurably knows the answer (the segment's own
  voice) and leaves the line blank otherwise. It never copies a name from a
  neighbour or a label.
- It runs after the existing rules and only on what they left unnamed, so
  everything Voxhelm names today keeps its name. Rule 6 only touches lines
  Voxhelm itself left unnamed.
- Rule 3 replaces the track's label guard (`raw_labels_are_known`) for this
  pass. That guard checks the labels on the segment, which the measurements
  show to be the wrong place to look for short segments; requiring a
  voiceprint for every substantial untied label is stricter where it matters
  and skips the whole job otherwise.
- No consumer change: confident names and `raw_diarization_speaker` are the
  two fields django-cast already reads.

Cost: at most 150 embeddings per speaker and label plus one per candidate,
under 1,000 for a typical episode, about 10 s at the measured 6-12 ms per
embedding. Jobs took 220-280 s in total.

Why not also replace the zero padding in the first pass: segments under
1.5 s cannot be accepted there, their padded embedding is used for nothing
but the sidecar's candidate list, and leaving it alone keeps the first pass
byte-identical. Removing `pad_to_min_duration` can follow once this pass has
proven itself.

### 2. Exclusive turns for alignment

`extract_pyannote_annotation` returns `exclusive_speaker_diarization` when the
result has one, otherwise what it returns today. This affects every consumer
of the turns: `apply_speaker_labels` for anonymous jobs, and for
known-speaker jobs the label vote, turn corroboration, the track guard, the
overlay of unmapped turns and `raw_diarization_speaker`. It is adopted only
if the harness shows that the naming metrics do not get worse (see the
slice).

## Evaluation harness

A script in `evals/` (like the existing probes, not part of the service) with
three steps, so that a variant can be scored in minutes and without the live
service. It runs alongside `evals/known_speaker_eval.py`, which stays as it
is: that script scores one episode (pp_64) from local FLAC reference tracks
and a labelled transcript by segment accuracy, and it classifies the gold
transcript's segments without ASR, word timings or pyannote turns, so it
cannot exercise the re-cut, the track or this pass. The new script takes the
production inputs and reports the published names.

1. `bundle`: for one audio file and its references, run the ASR and the
   anonymous diarization once through the shipped code path with the worker's
   env file, and store the transcription result with word timings, the raw
   turns (regular and exclusive) and the reference ranges in a directory
   outside the repository.
2. `run`: load a bundle, run `run_known_speaker_postprocess` with a given
   configuration and write the sidecar. Only embeddings are computed, so a
   run takes one to two minutes per episode.
3. `score`: compare a sidecar with a truth file and print the metrics below.
   It applies django-cast's guest rule the way
   `resolve_unreferenced_label_names` does, so the score describes the
   published names.

Test set:

- python-podcast episodes 59, 77, 78 and 80 (4.6 h; two, three, four and
  three known speakers, one of them with a guest without a reference) as the
  required set, the other four pilot episodes as the extended set. Truth: the
  existing 10 frames-per-second files. The audio is public; the truth files
  and bundles stay outside the repository, and the document records where.
- Django Chat audio 209 and 210 without truth: line counts, plus a
  before/after listing of every line whose name changes, for the listening
  check.

Metrics:

- frame level, over truth speech inside transcript segments: right, wrong,
  unnamed (the three numbers of the bulk evaluation);
- line level: share of unnamed lines, total and per duration bucket (under
  0.5, 0.5-1, 1-1.5, 1.5-2.5, 2.5-5, 5 s and more);
- per accept reason: number of segments, and the share that is right among
  those with a clear truth speaker (at least three truth frames, 80 % one
  speaker);
- lines whose name differs from the baseline run, with the share that is
  right before and after.

A diarization error rate is deliberately not the headline metric: the product
is names on transcript lines, pyannote's own turns are only an input, and
time-weighted rates hide the short lines this work is about. The frame
numbers are the closest equivalent and are comparable with the earlier
evaluations.

## Risks and failure modes

- **A wrong name where a blank was.** About one in seven new names is wrong
  on the spike data. That equals the accuracy of today's track-named short
  lines, and the margin is a request option. The maintainer decides whether
  that trade is wanted (open question 1).
- **Episode centroid built from wrong segments.** If the first pass names a
  guest's long passages after a host (measured once, on an opening with a
  guest who resembles a host), the host's episode centroid drifts towards the
  guest. Mitigations: only segments accepted on their own embedding at the
  full thresholds, highest margin first, capped at 150. The harness reports
  per-reason precision, which would show it.
- **Speakers with little speech.** A known speaker with fewer than 5 accepted
  long segments is left out of the pass. Their short lines stay unnamed, or,
  if their voice is close to another participant's, can be named after that
  participant. The same holds for a person on an untied label with under
  30 s of speech. Both are a handful of lines per episode at most; the
  harness reports new names per speaker.
- **A guest whose label has no voiceprint.** The pass is skipped for the job,
  so such episodes keep today's result. The summary field makes these jobs
  countable.
- **Similar voices.** In-room recordings with four similar voices (the
  DjangoCon episode in `specs/diarization-quality-research.md`) have small
  margins; the pass then decides less, not wrong more often, as long as the
  margin holds. Episode 80 (71 % precision) is the weakest in the spike and
  is in the required set.
- **Two guests without references.** Each untied label gets a centroid, so
  host interjections are still separated. django-cast's guest rule needs
  exactly one untied label, so the guests themselves stay unnamed as today.
- **An untied label that is noise or music.** Its centroid attracts only
  noise; segments relabelled to it stay unnamed, as today.
- **Truth quality.** Stem truth is weaker with crosstalk, and only frames
  with a single clear speaker count. Interjections spoken over another
  speaker are therefore under-represented in every number here.
- **Exclusive turns.** They change turn boundaries, and with them the overlay
  cut at unmapped labels, so segment counts in guest episodes can shift
  slightly. Consumers match by start time; the slice checks that all sidecar
  starts still match the artifacts. They also change the published
  `Speaker N` labels of anonymous jobs, which the known-speaker harness does
  not see; the slice measures those separately.
- **Re-runs replace reviewed transcripts.** A re-run changes the sidecar and
  the artifact fingerprint, and editor decisions and stale mappings follow
  django-cast's existing rules. This design does not require re-running
  anything but the test episodes (open question 2).
- **Runtime and memory.** About 10 s more embedding time per job; no new
  model is loaded.

## Test and qualification plan

- Unit tests for the new pure functions with a fake embedding backend, as in
  `tests/test_known_speaker.py`: tiling, episode centroid selection, a known
  speaker left out for lack of segments, untied-label centroids, both skip
  conditions, the candidate rule, the three outcomes of the classification,
  the mixed-segment veto stopping an acceptance, the preserved overlap label, cancellation between embeddings, and `short_segment_margin=0`
  leaving the outcome identical to today.
- Contract tests: option validation at submission, propagation into stored
  and claimed configuration, the request fingerprint, the new summary keys
  through the remote completion allowlist, local and remote summary parity,
  serialization of the additive segment fields.
- Harness on the required set before and after each behaviour change, with
  the tables recorded in this document.
- Staging: re-run audio 209 and 210 after the release is installed, compare
  the line counts, and hand the before/after listing to the maintainer for a
  listening check.
- Gates as usual: `just lint && just typecheck && just test`.

## Implementation plan

Seven slices, each 30-90 minutes, in this order. "Required set" is
python-podcast 59, 77, 78, 80. B stands for the baseline the harness records
in slice 2 with the unchanged code.

1. **Scoring.** `evals/speaker_naming_eval.py score` and
   `tests/test_speaker_naming_eval.py`: frame, line and per-reason metrics
   from a sidecar and a truth file, including the guest rule.
   Acceptance: on the stored pilot sidecar of episode 59 it reports 96.3 %
   right, 1.5 % wrong, 2.2 % unnamed and 101 of 1221 unnamed lines (each
   within 0.1 point or one line); unit tests cover the bucket and precision
   arithmetic on a synthetic fixture.
2. **Bundles and baseline.** `bundle` and `run` in the same script; bundles
   for the required set and for Django Chat 209 and 210.
   Acceptance: `run` with default options on a bundle is deterministic (two
   runs give identical sidecars); the baseline table B for the required set
   is added to this document; right is within 93-97 % and unnamed lines
   within 7-15 % per episode (the pilot range), otherwise the difference to
   the pilot is explained in the document.
3. **Episode voiceprint pass for known speakers.** `transcriptions/known_speaker.py`
   (centroids, candidates, classification, accept reason, summary counters,
   `short_segment_margin` in `KnownSpeakerConfig`), `tests/test_known_speaker.py`.
   No centroids for untied labels yet, so the pass is skipped for jobs with
   an untied label of 30 s or more (episode 80 and both Django Chat episodes).
   Acceptance on the required set, against B: unnamed lines down by at least
   35 % relative (spike estimate without episode 80: 48 %); frame right up by
   at least 0.5 points (estimate: 0.8); frame wrong up by at most 0.4 points
   (estimate: 0.2); segments with the new reason at least 82 % right (spike:
   86 %); every segment named in B keeps its name; the sidecar of episode 80
   has the same segments as in B and `episode_voiceprint_skipped` set;
   postprocess time up by at most 15 %; with `short_segment_margin=0` the
   segments and all summary keys that exist today equal B, and the new
   summary keys report zero counts.
4. **Contract and plumbing.** `jobs/services.py` (`parse_known_speaker_config`,
   `known_speaker_config_for_job`), `jobs/remote_worker_cli.py` (claimed
   configuration), `jobs/remote_workers.py` (summary allowlist),
   `jobs/fingerprints.py` if the option is part of the fingerprint,
   `specs/known-speaker-diarization.md`, `README.md`, `CHANGELOG.md`, tests.
   Acceptance: the option is validated, stored, claimed and summarized
   identically for local and remote execution; a job without the option
   behaves like one with 0.15; the contract document describes the pass, the
   new reason and the new fields.
5. **Untied labels.** Centroids for untied labels, relabelling with
   `raw_diarization_overlap_speaker`, hosts winning on a guest's label,
   `speaker_episode_candidates`; the contract change for
   `raw_diarization_speaker` in `specs/known-speaker-diarization.md`; tests.
   Acceptance: on Django Chat 210 the unnamed lines after the guest rule drop
   from 69 to at most 30 (spike: 18) and on 209 from 61 to at most 30
   (spike: 19); at least 95 % of the guest-rule lines of 3 s and more have
   the guest's label as top candidate (spike: 255 of 256, 194 of 194); on the
   required set against B, unnamed lines are down by at least 40 % (spike:
   58 %), frame right is up by at least 0.6 points (spike: 1.0) and frame
   wrong by at most 0.4 points (spike: 0.2); in episode 80, of the short
   segments whose truth speaker is the guest without a reference (48 in the
   pilot sidecar), at most 2 more than in B carry a known speaker's name; the
   listing of lines moved from the guest to a host is produced (spike: 17 and
   8 lines).
6. **Exclusive turns.** `transcriptions/diarization.py`
   (`extract_pyannote_annotation`), `tests/test_diarization.py`, harness runs
   from the stored regular and exclusive turns, `CHANGELOG.md`.
   Acceptance: on episodes 78 and 80 the share of short segments whose
   longest-overlap label is the right speaker rises by at least 3 points
   (spike: 11 and 5 with a different flattening); on the required set, frame
   right does not fall and frame wrong does not rise by more than 0.1 point
   against slice 5; every sidecar start matches an artifact start. Anonymous
   jobs: `apply_speaker_labels` on the bundles of the required set with the
   regular and the exclusive turns, labels mapped to truth speakers by
   majority; frame right must not fall by more than 0.1 point on any episode,
   and no bundle may fail with "did not overlap transcript segments". If a
   criterion fails, the change is not merged and the result is recorded here.
7. **Rollout check.** After the release is installed on the control plane
   and both workers by the usual deploy: re-run Django Chat 209 and 210 on
   staging and compare with the harness prediction.
   Acceptance: unnamed lines at most 30 on each; `processing_seconds` within
   15 % of the earlier jobs (281.6 s and 222.0 s); django-cast matches every
   sidecar segment to the artifacts; the before/after listing is handed to
   the maintainer for the listening check; results are recorded in this
   document.

Slices 3 and 5 change which lines carry a name, slice 6 changes label
boundaries, and slices 1, 2, 4 and 7 change no naming behaviour.

## Open owner questions

In German, because they are addressed to the maintainer.

1. Die neue Regel benennt etwa zwei Drittel der bisher leeren kurzen Zeilen;
   ungefähr jede siebte dieser neuen Zuordnungen ist falsch, so wie heute
   schon bei kurzen Zeilen. Ist dir ein gelegentlich falscher Name lieber als
   eine leere Zeile? Empfehlung: ja, mit der gemessenen Schwelle 0,15. Ohne
   Antwort wird es so gebaut; die Schwelle lässt sich pro Anfrage ändern oder
   abschalten.
2. Sollen nach dem Rollout schon veröffentlichte Transkripte (python-podcast
   in Produktion, Django Chat auf Staging) neu erzeugt werden? Empfehlung:
   zuerst nur die beiden Testfolgen, ein Massenlauf erst nach deinem
   Hörtest. Ohne Antwort bleibt alles Veröffentlichte unverändert und nur
   neue Folgen profitieren.
3. Magst du nach dem Rollout etwa zehn Minuten lang eine Liste der
   geänderten Zeilen der Haki-Benita- und der Loes-Crama-Folge gegenhören?
   Ohne Hörtest stützt sich die Abnahme nur auf die python-podcast-Messung.

## Sandbox implementation (2026-11-15)

Slice 1 adds `uv run python -m evals.speaker_naming_eval score SIDECAR
--truth TRUTH --guest NAME --out REPORT`. Omit `--guest` when every contributor
has a reference; repeat it for multiple unreferenced contributors (the guest rule
then stays off). `--baseline SIDECAR` lists changed names. Truth input is JSON
`{"fps":10,"frames":["display name",null]}`: frame i samples i/fps, null excludes
silence/overlap/unclear speech. Convert stored truth to these display names before
scoring. Every non-null truth frame counts once, including uncovered speech (unnamed).
`frames.covered` and `coverage_pct` measure transcript coverage separately.
Overlapping cues agree only if all published labels agree, including null;
conflicts count as unnamed. Intervals are half-open and cue order has no effect.
Frame percentages from older reports omitted uncovered speech and used first-cue
arbitration; they are not directly comparable. Line metrics remain cue-based.
Clear-line precision includes unnamed clear lines in its denominator.
The committed synthetic fixture scores 15/30 right, 5/30 wrong, 10/30 unnamed
frames and 1/4 unnamed lines with `--guest Guest`; it is not a pilot measurement.
The pilot sidecars/truth are only documented on studio under
`var/handoff/2026-10-19_python_podcast_parakeet_bulk/`. No host access is authorized
for this worker, so episode 59's 96.3/1.5/2.2 and 101/1221 acceptance remain unverified.

### Slice 2: repeatable bundles and baseline handoff

No production audio, truth, models or pilot bundles are included in this clone.
The historical files are on studio at the path above; there is no supported
truth-download endpoint in the repository. No host access or live jobs were used.
The clerk measured B and the new pass on real bundles; the numbers are in
[Clerk qualification](#clerk-qualification-2026-11-15).

Clerk preparation (on the measurement machine, outside this sandbox): put the
six local audio files and reference audio in `/tmp/voxhelm-speaker-naming/`.
Create `manifests/{pp59,pp77,pp78,pp80,dc209,dc210}.json`, with paths relative
to each manifest. Each entry of `references` is one source range (start/end in
seconds) or a complete clip (omit both); repeat the id/name for multiple ranges:

```json
{
  "audio": "../audio/pp59.m4a",
  "language": "de",
  "model": "auto",
  "references": [
    {"id":"jochen", "name":"Jochen", "audio":"../audio/reference.flac", "start":0, "end":30}
  ],
  "unreferenced_speakers": [],
  "num_speakers": 2
}
```

`num_speakers`, `min_speakers` and `max_speakers` are optional and are passed
to pyannote exactly like the job's `diarization` counts (same validation as the
API: positive integers, `num_speakers` not combined with min/max, min <= max).
Omit them when the job set none. Copy them from the job being reproduced: the
frozen turns depend on them.

Supply all real participant references, English for Django Chat, and its single
unreferenced guest name. Convert the four historical 10 fps truth arrays into
`truth/{pp59,pp77,pp78,pp80}.json` with the JSON schema above. The source arrays'
serialization is not documented in this repository; do not infer names from IDs.
The fixture in `tests/fixtures/speaker_naming/` demonstrates the score schema.

Exact baseline command at the slice-2 commit (with the deployed model cache):

```bash
HF_HUB_OFFLINE=1 uv run --extra diarization python -m evals.speaker_naming_eval suite \
  /tmp/voxhelm-speaker-naming --env-file "$HOME/Library/Application Support/voxhelm-remote-worker/worker.env" \
  --tag baseline
```

`bundle MANIFEST --env-file ENV --out NEW_DIRECTORY` performs ASR through the
production backend chain with sanitization and word timings and captures regular
and exclusive anonymous turns once. It freezes float32 audio/reference windows,
reference source ranges and the complete transcription in `bundle.json` + NPY
files. Run on the same machine/model versions; numerical GPU determinism is
verified, not assumed. `run BUNDLE --env-file ENV --config OPTIONS.json --out
SIDECAR.json` replays only embeddings and postprocessing; the separate
`.timing.json` holds runtime so sidecars can compare byte for byte.
`suite` does two replays per episode and fails on unequal sidecars. It writes
`results/TAG/EPISODE{,.score}.json`, including every changed line with text,
frame and line metrics, and postprocess time. Variant tags compare with baseline;
missing python-podcast truth fails instead of silently reporting counts only.
Bundles and all real measurements stay outside the repository.

Local mock qualification: a three-second one-speaker bundle roundtrips word
boundaries, both annotations and source ranges; two replays have identical
sidecars (1/1 confident Alice). This validates the harness, not the six real
baselines or timing/quality targets. Slices 3 and 5 must still be measured by the
clerk before review/merge qualification.

### Slice 3: known episode profiles

Known-speaker episode profiles and the short-cue pass are implemented. Profiles
require five whole-segment acceptances of at least 3 s, take the 150 highest
margins per speaker, and never mix reference and episode centroids. Candidate
audio repeats to 1.5 s without silence; a known winner must pass the existing
mixed veto against reference profiles. Named cues and segment boundaries remain
unchanged. Summary counts describe accepted known names; relabelled count is
zero in this slice. Skip reasons are `disabled`, `no_raw_turns` (added at merge
review: without anonymous turns a guest cannot be detected), `unresolved_label`,
`too_few_centroids`, or null when the pass ran. These additive summary keys are
allowlisted for remote completion so local gates stay coherent.

From this commit onward the baseline command **must** disable the new pass:

```bash
printf '%s\n' '{"short_segment_margin":0}' > /tmp/voxhelm-speaker-naming/baseline-options.json
HF_HUB_OFFLINE=1 uv run --extra diarization python -m evals.speaker_naming_eval suite \
  /tmp/voxhelm-speaker-naming --env-file "$HOME/Library/Application Support/voxhelm-remote-worker/worker.env" \
  --config /tmp/voxhelm-speaker-naming/baseline-options.json --tag baseline
HF_HUB_OFFLINE=1 uv run --extra diarization python -m evals.speaker_naming_eval suite \
  /tmp/voxhelm-speaker-naming --env-file "$HOME/Library/Application Support/voxhelm-remote-worker/worker.env" \
  --tag known-episode
```

Reuse exactly the same bundles. The second command produces per-line changes,
per-reason precision, frame metrics and runtime for comparison with B. Real
35 % unnamed-line reduction, +0.5 point right / <=+0.4 wrong, >=82 % new-name
precision and <=15 % runtime increase remain unmeasured in this sandbox.
Synthetic qualification: the uncertain 0.7 s B interjection on A's raw label is
accepted as B; all ten already named long cues retain their names. Disabled
execution performs no second-pass embeddings and adds zero summary counts.

### Slice 4: request and worker contract

The threshold uses the existing shared numeric option allowlist and config loader;
no parallel parsing or fingerprint implementation was added. API/storage/claim
and local/remote summary regressions cover 0, 0.15 and custom values, rejection
of malformed/nonfinite/negative values, changed request fingerprints and null
skip-reason preservation. README and the known-speaker contract document the
pass and the disable semantics. The default 0.15 follows the owner's stated
preference; no production runs or rollout were performed.

### Slice 5: untied-label profiles and final qualification handoff

Untied labels now compete through profiles from at least five uncertain,
non-mixed single-label cues of at least 3 s (longest first, maximum 150).
Substantial labels without profiles still skip the whole job. Known winners keep
the reference-based mixed veto. Label winners stay uncertain for the consumer's
guest rule; changed assignments preserve even a null original overlap label.
Optional episode candidates use disjoint speaker/label identities. No first-pass
reference evidence, raw-turn speech totals, cue starts or artifact boundaries are
rewritten. Slice 6 (exclusive turn selection) and slice 7 (rollout) were not done.

After producing B with the disabled configuration above, the clerk must run:

```bash
HF_HUB_OFFLINE=1 uv run --extra diarization python -m evals.speaker_naming_eval suite \
  /tmp/voxhelm-speaker-naming --env-file "$HOME/Library/Application Support/voxhelm-remote-worker/worker.env" \
  --tag with-guest
```

The suite now also writes `EPISODE.profiles.json`: for each untied label, the
number of long guest-rule cues, how often its own profile is top, and that
percentage (target >=95 % on dc209/dc210). These extra diagnostic embeddings run
**after** the measured production postprocess and are excluded from its runtime.
For an individual replay add `run ... --profile-checks` to produce the same file.
`EPISODE.score.json` includes `guest_to_host_lines` with start/end/text for the
listening check, `new_names_by_speaker`, and `short_guest_known_names.before/after`
for the episode-80 safety criterion (at most two additional known names on clear
short guest cues). It also retains all changed lines and their truth outcomes.
No truth means frame/precision percentages are null, not zero or invented.

Synthetic measurements: guest-profile sanity 5/5 (100 %); a guest interjection
on a host label moves to the guest label with the original overlap retained;
a host interjection on a guest label becomes a confident host. These are mock
qualifications, not evidence for <=30 unnamed Django Chat lines, >=40 % relative
unnamed reduction, +0.6 point right / <=+0.4 wrong, or production runtime. All real
cells in B and all corresponding acceptance targets remained pending clerk data at
implementation time; see [Clerk qualification](#clerk-qualification-2026-11-15).

The six-episode suite checks two identical sidecars per replay. It never submits
a job, changes a published transcript, deploys, or selects exclusive turns.
Measurement inputs and output stay under `/tmp/voxhelm-speaker-naming`, outside
Git; archive that directory before temporary storage is cleared.

### Local gates and review status

Every slice ran `just check` (Ruff lint/format, mypy, full pytest), with no models
or services needed. Slices 1–4: respectively 2106, 2108, 2128, 2146 passed;
each had 6 model-dependent skips. Slice 5: 2160 passed, 6 skipped. The locked all-extras pip-audit found no known vulnerabilities.
Logs are outside the clone in
`~/workspaces/ws-work-planning/coordination/codex/logs/cx-voxhelm-diarization-a/`.
Per the final worker instruction, no Claude command or sandbox cross-family
review was run. At implementation time the branch was parked for external review
and real-data qualification; the clerk's review, measurements and merge follow in
[Clerk qualification](#clerk-qualification-2026-11-15).

## Clerk qualification (2026-11-15)

Measured on `studio` against the real episodes; no job was submitted and no
published transcript changed.

### Setup

- Inputs from the control plane (read-only): the job requests of the pilot jobs
  (pp59 `e04859f0`, pp77 `46009d86`, pp78 `065c031f`, pp80 `8a74983c`) and of
  the Django Chat staging jobs (dc209 `c0e84980`, dc210 `da97bdc8`) give the
  audio URLs, the reference ranges and the `num_speakers` hints; the stored
  speakers sidecars of these jobs served as cross-check. Truth: the pilot's
  10 fps arrays (`truth-{59,77,78,80}.json` under
  `var/handoff/2026-10-19_python_podcast_parakeet_bulk/pilot/`), whose entries
  are display names or empty strings, converted 1:1 (empty = null).
- Interpreter: the deployed remote worker's tool environment
  (`~/.local/share/uv/tools/voxhelm`, voxhelm 0.1.13 deps, torch 2.12.0,
  pyannote.audio 4.0.4) with the code under test on `PYTHONPATH`; worker env
  file `/etc/voxhelm-remote-worker/worker.env`, `HF_HUB_OFFLINE=1`, and
  `VOXHELM_KNOWN_SPEAKER_STT_BACKEND=parakeet` (a control-plane setting the
  worker env lacks; without it the harness would transcribe with MLX Whisper
  instead of Parakeet as production does).
- Run from a directory without Python packages: `python -m` puts the working
  directory before `PYTHONPATH`, so running the "old" variant from a branch
  checkout silently imports the branch code.
- One bundle per episode (ASR + both annotations, 1.5-4.5 min each), then
  three replays of the same bundles: **old** = `origin/main` 3d32848 code,
  **B** = branch with `{"short_segment_margin": 0}`, **new** = branch defaults
  (0.15, untied-label profiles). Every replay ran twice; all sidecars were
  identical (deterministic). B equals old segment by segment on all six
  episodes; only the new summary keys differ.
- The bundles reproduce production: B on pp59, pp77 and pp78 scores exactly
  like the stored production sidecars (for example pp59 96.17 / 1.68 / 2.15 %,
  101 of 1221 lines unnamed; the slice-1 reference values 96.3 / 1.5 / 2.2 were
  the spike's rounding, the 101/1221 lines match). pp80 and the Django Chat
  episodes differ by one to three lines from a fresh ASR run.
- Data (outside Git): `~/workspaces/ws-s1-voxhelm-merge/speaker-naming/`
  (`manifests/`, `truth/`, `bundles/`, `results/{old-main,margin0,with-guest}/`).

### Results

Frames: share of truth speech inside transcript lines (right / wrong /
unnamed). Lines: published names after django-cast's guest rule. "New-pass
names" are lines with accept reason `episode_voiceprint` that have a clear
truth speaker.

| episode | B frames % | new frames % | B unnamed lines | new unnamed lines | change | new-pass names right |
| --- | --- | --- | --- | --- | --- | --- |
| pp59 | 96.17 / 1.68 / 2.15 | 96.94 / 1.76 / 1.30 | 101/1221 | 40/1221 | −60.4 % | 44/44 (100 %) |
| pp77 | 93.55 / 1.59 / 4.86 | 94.78 / 1.91 / 3.31 | 245/1764 | 119/1764 | −51.4 % | 73/92 (79.3 %) |
| pp78 | 93.20 / 1.55 / 5.25 | 94.33 / 1.59 / 4.07 | 70/501 | 39/501 | −44.3 % | 20/20 (100 %) |
| pp80 | 93.42 / 3.38 / 3.20 | 94.61 / 3.88 / 1.52 | 79/710 | 32/710 | −59.5 % | 23/36 (63.9 %) |
| required set | 94.24 / 1.90 / — | 95.31 / 2.14 / — | 495 | 230 | **−53.5 %** | **160/192 (83.3 %)** |
| dc209 | no truth | no truth | 61/609 | 23/609 | −62.3 % | 46 named |
| dc210 | no truth | no truth | 69/878 | 20/878 | −71.0 % | 55 named |

Other observations:

- No line named in B lost its name. Renamed lines occur only where a host's
  episode profile beats the guest's untied label: pp80 11 (of the 8 with a
  clear truth speaker, 3 right and 5 wrong), dc209 10, dc210 18.
- Guest-profile sanity (long guest-rule cues whose own label profile ranks
  first): pp80 46/46, dc209 184/184, dc210 226/226 (100 %).
- pp80 clear short cues whose truth speaker is the guest Sebastian and that
  carry a known (host) name: 1 in B, 6 new.
- Postprocess time B → new: +2.0, +1.1, −15.8, +0.8, −0.3, +2.3 % (pp59 …
  dc210). The runs shared the machine with other jobs, so only differences
  well beyond a few percent would be meaningful; none is.

### Target check

| target (slices 3/5, against B) | result | |
| --- | --- | --- |
| unnamed lines on the required set −35 % (slice 3) / −40 % (slice 5) | −53.5 % | met |
| frame right +0.5 / +0.6 points | +1.07 | met |
| frame wrong at most +0.4 points | +0.24 (pp80 alone +0.50) | met pooled |
| new-reason names at least 82 % right | 83.3 % (pp77 79.3 %, pp80 63.9 %) | met pooled, narrowly |
| Django Chat unnamed lines at most 30 | 23 and 20 | met |
| guest-rule lines ≥ 3 s with the guest label on top, at least 95 % | 100 % on all three guest episodes | met |
| pp80: at most 2 more clear short guest cues with a known name than B | +5 (1 → 6) | **missed** |
| postprocess time at most +15 % | within noise | met |
| `short_segment_margin=0` equals today | identical segments on all six | met |

Verdict: the pooled targets hold, but the episode-80 guest safety target is
missed. The misses all come from the same mechanism: hosts' episode profiles
winning short cues on the guest's untied label (pp80: 5 of 8 clear relabels
wrong). Per the instruction "deploy only if the targets are met", the control
plane was **not** deployed. The change is merged to `main`; it reaches jobs
only through a voxhelm release for the remote workers (all transcription jobs
run `remote_pull`), which is the owner's call.

The Django Chat listening list (all lines whose published name changes between
old and new, same ASR and turns) is in
`~/workspaces/ws-s1-voxhelm-merge/hoertest-django-chat.md`. Re-generating the
two staging episodes was deferred: with the workers still on 0.1.13 a re-run
would use the old naming code, so it could not show the change and would only
replace the staging transcripts.

Possible follow-ups (owner decision): keep the default 0.15 and accept the
guest cases, or require a larger margin for a known speaker to win on an untied
label (all five additional pp80 guest cues with a host name come from that
relabel step), then re-measure with the same bundles (an `evals` `run` replay
takes about 3 minutes per episode).

### Offline input admission (round 110)

Controls use strict UTF-8 object JSON: regular no-follow files, at most 64 MiB,
128 levels, unique members and finite numbers. Errors identify schema locations,
not control contents. Truth admits at most 1,000,000 null/nonempty-name frames,
FPS in (0, 1000], and at most 5,000,000 cue/frame visits. Cues/turns/words admit
50,000 entries, timestamps in [0, 86400] seconds with end >= start (an
arithmetic bound, so long production episodes stay scorable); booleans are
not numbers. Manifests admit 128 references/guest names and references of at
most 600 seconds. Malformed controls fail before embedding/model startup.

Frozen members must be relative regular `.npy` files contained in the chosen
bundle, with no `.`/`..`, backslashes, absolute paths or symlink components;
hard-linked members are refused too.
All opened NumPy 1.0/2.0 headers and exact file lengths are checked before sample
allocation: nonempty native float32 mono vectors, at most two hours at 16 kHz
for audio (the production Pyannote decode bound, `PYANNOTE_MAX_FRAMES`; capture
decodes the job audio with that allowance first and refuses longer sources
before ASR or diarization start), five seconds per reference window, 1000 windows and 512 MiB aggregate
sample bytes (including repeated members). Samples must be finite before any
embedding. After header admission fixed-count raw float32 reads avoid reparsing
a potentially modified header; pickle is never enabled or used. Opened files
must remain unchanged during reads. These are offline capacity limits, not
changes to production inference admission.

### Frozen generation publication and identity (bundle version 2)

`bundle` captures into a uniquely owned sibling `.NAME.stage-*` directory, checks
the complete manifest/arrays, then renames it into an absent destination under
a nonblocking `.NAME.lock` publisher lock. Existing destinations are refused,
including incomplete legacy directories. A failed attempt removes only its own
stage and can be retried. Process death may retain a stage; after verifying no
publisher is running, archive or remove that exact stage and retry. Lock files
remain; advisory ownership is released automatically, so do not unlink locks.

Version 2 binds the canonical manifest, actual source/reference bytes, explicit
local model artifacts, backend settings, installed package versions and the
capture-path transcription modules (`CAPTURE_CODE` in
`evals/speaker_generations.py`: ASR service, diarization, decode, sanitizer and
their imports). `transcriptions/known_speaker.py` is deliberately excluded: it is
the postprocessing under comparison, and its decode/window output is frozen as
sample-sealed arrays, so old/branch code replays of the same bundles keep working.
When capture code gains a new `transcriptions` import, add it to `CAPTURE_CODE`
(a regression test checks the import closure). In addition to the example above manifests must now include:

```json
"generation_files": {
  "asr": ["../models/asr-weights.bin", "../bin/whisper-cli"],
  "diarization": ["../models/segmentation.bin", "../models/embedding.bin"]
}
```

Each role requires 1–128 regular files (relative to the manifest, or absolute).
Enumerate the actual weights, configuration/tokenizers and native executable
used by that role, including all possible configured fallback backends. These
are explicit local generation inputs: the harness does not download or guess
model cache identities. Use an immutable local snapshot and offline inference;
a mutable remote-only model cannot qualify as exact frozen suite evidence.
Hashes are checked before and after capture. The suite recomputes identities
before replay and refuses a changed manifest, source, reference, model, code or
backend configuration. Choose a new bundle/root on mismatch; historical evidence
is never overwritten automatically. Postprocessing options are deliberately
outside the bundle identity so variants reuse the same frozen inputs.
Version-1 bundles remain readable by standalone `run`; suite reuse requires
version 2 and an explicit rebuild in a new directory. Old reports are historical
measurements, not proof of the new input/scoring contract.

Version-2 bundle manifests also seal each admitted member's raw float32 samples
with `sample_sha256`; substituted finite arrays fail before embedding. Completion
manifests transitively bind those hashes through the frozen bundle JSON hash.
Variants must match the baseline's exact bundle JSON and truth hashes before
replay, as well as its embedding model. Backend/decode/sanitizer settings are
recorded as a digest so credential-bearing endpoints are not copied into evidence.
Include the actual decoder and backend executables among generation files.

### Complete suite result generations and tags

Tags are one portable ASCII component, 1–64 characters, starting with an
alphanumeric character and containing only alphanumerics, `_`, `-`, `.`. Trailing
dots and reserved device names are refused before creating directories or
starting a backend. `baseline` is the explicit comparison baseline.

The suite validates and snapshots all manifests, truth, baseline sidecars and
configuration before replay. Reused bundles must match source identities;
variants retain the baseline embedding model. It stages all six episodes,
profiles and scores under `results/.TAG.stage-*`, then writes `complete.json`
with exact generated-member hashes, full effective configuration and per-episode
bundle/truth/baseline identities. Only deterministic, complete generations may
publish. A nonblocking `.TAG.lock` excludes cooperating same-tag publishers.

A completed generation lives in `results/.TAG.generation-*`; `results/TAG` is an
atomically replaced relative symlink. Readers that need a multi-file snapshot
must resolve the tag once and read that generation. Previous generations are
retained, including when a later episode or write fails. Failed stages are also
retained, carry no completion authority, and are named in the exception note.
After verifying no publisher is running, archive/remove only the intended stage
or obsolete generation. Never delete the persistent lock file. A process may
die after completion but before pointer publication; its unreferenced complete
generation is retained and the previous tag remains authoritative. Existing
legacy real tag directories must be archived/moved explicitly before a rerun;
the harness will not overwrite or mix them. Recreate legacy baselines with the
new format before running variants. No production model campaign is implied.

### pp62 production metric contract (version 2)

`python -m evals.pp62_production_eval --gold GOLD --sidecar SIDECAR --out REPORT`
now reports `speech_seconds.top1` and `.public`: exact correct, wrong, unnamed,
uncovered and covered seconds, gold seconds, accuracy over all gold speech,
union coverage and accuracy conditional on coverage. Gold tracks overlapping
one another contribute separate speaker-seconds. Predictions use half-open
intervals clipped to each gold interval; duplicate/same-label overlaps count
once, conflicting labels (including candidate abstention) count as covered but
unnamed. Touching endpoints do not overlap. Top1 uses the first candidate only;
public uses the accepted `speaker` only when `speaker_uncertain` is exactly
false and the accepted speaker is non-null. Null/uncertain public labels abstain
and do not contribute public coverage or public segment counts.

One correct predicted second over ten gold seconds yields one correct and nine
uncovered seconds (10% accuracy/coverage). Two right and three wrong seconds over
ten yield 20% accuracy and 50% coverage. All percentages with an empty denominator
are null. Both compatibility coverage fields now use actual covered seconds;
`public_confident_time_weighted_accuracy_pct` is correct / covered public seconds.

`legacy_duration_weighted_segment_label` preserves the old full-gold-duration
winner calculation under an accurate name (largest summed-overlap label, lexical
tie break); duplicate vote weight remains part of that historical calculation.
Its public view now uses accepted names. `PRIMARY_time_weighted_top1_accuracy_pct`
and `time_weighted_DER_pct` are removed: the old calculation was neither aligned
speech-second accuracy nor DER. No metric here measures silence false alarms,
full overlap-aware DER or collars. Historical reports and qualification tables
above retain their historical meaning and must be remeasured for comparison;
do not relabel their numbers as version-2 evidence.
