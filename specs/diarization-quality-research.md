# Diarization Quality Research

Status: research note / planning input
Date: 2026-05-27

## Context

Voxhelm can add speaker labels to batch transcription artifacts by running a
diarization backend after STT and aligning diarization turns to Whisper
transcript segments. The current implemented backend is pyannote.

django-cast can request diarization for podcast episodes and can pass an exact
speaker count derived from episode contributor assignments. It then maps
anonymous labels such as `Speaker 1` to contributors. This mapping only works
when diarization has already separated the speakers into useful clusters.

## Problem Episode

Python Podcast episode:

- "Live von der DjangoCon Europe 2025 in Dublin - Tag 2"
- German mono MP3, about 66 minutes
- Live-room conversational podcast
- Four expected male speakers: Dominik, Jochen, Johannes, Ronny
- django-cast passed `diarization.num_speakers = 4`
- Voxhelm stored `{"enabled": true, "num_speakers": 4}` correctly

Observed Podlove transcript distribution from the four-speaker run:

- `Speaker 1`: about 969 segments
- `Speaker 2`: about 234 segments
- `Speaker 3`: about 979 segments
- `Speaker 4`: 9 segments

An older three-speaker run was nearly identical:

- `982 / 972 / 237`

Therefore `num_speakers=4` technically produced four labels, but pyannote mostly
split out a tiny fourth cluster instead of separating the real fourth speaker.

Known failed passage, spoken by Johannes but assigned to another label:

> Also, sie hat ja da auch so ein paar Sachen erwähnt, die einfach richtig
> schlimme Folgen haben. Ja, JP Morgan hat halt eine von den Berechnungen, haben
> sie... Hups. Falsch. Addiert statt zu mitteln und dann haben sie ihre
> Risikobewertung einfach um Faktor 2 falsch gehabt und Milliarden an Dollar
> verloren.

Approximate transcript time: `00:38:16-00:38:32`.

## Short Diagnosis

This looks like a diarization quality failure in the acoustic speaker clustering
stage, not a request/metadata/mapping bug.

The request path worked:

- django-cast passed the expected speaker count.
- Voxhelm stored the diarization metadata.
- pyannote returned four labels.
- Voxhelm aligned the returned turns to transcript segments.

The failure is that pyannote could not form a useful separate cluster for one
real speaker. In a mono live-room recording with four male speakers, similar
voices, crosstalk, room bleed, short turns, and uneven amounts of clean solo
speech can make the fourth speaker collapse into another cluster. Forcing four
speakers can then create a token tiny cluster rather than recover the missing
speaker.

## Current Implementation Notes

Relevant implementation shape:

- Voxhelm uses pyannote for diarization.
- Speaker-count hints are represented as `DiarizationParams` and passed to the
  pyannote pipeline as `num_speakers`, `min_speakers`, and/or `max_speakers`.
- Voxhelm normalizes backend labels to stable generic labels such as `Speaker 1`.
- Transcript segments are labeled by choosing the diarization turn with maximum
  timestamp overlap.
- Speaker-label mapping in django-cast maps anonymous labels to contributors,
  but cannot fix merged speaker clusters.
- The audio available to production is mono; there is no production channel
  separation today.

pyannote's current speaker diarization pipeline can return both:

- `speaker_diarization`
- `exclusive_speaker_diarization`

The exclusive output removes overlapping speech turns and is intended for
downstream transcription alignment. For transcript labeling, it is likely a
better default than the overlapping diarization annotation, though it does not
solve a whole-speaker merge by itself.

In current pyannote logic, `min_speakers=4, max_speakers=4` is effectively
normalized to exact `num_speakers=4`, so it is not expected to behave materially
differently from `num_speakers=4`.

## Separated Track Findings

For this episode, local pre-mastering FLAC tracks exist. They are potentially
much more valuable than mono blind diarization because each track can provide a
known-speaker activity signal.

However, those tracks are currently local only and are not available on the
production website. Production transcription runs against the mastered audio
that has gone through Auphonic. Auphonic may apply processing that changes the
timeline, especially silence cutting.

This creates an important constraint:

> Track-derived speaker activity is only directly usable when the speaker tracks
> and the transcription audio share the same timeline.

If the separated FLAC tracks are pre-Auphonic and the transcription audio is a
post-Auphonic mastered MP3 with silence removed, direct timestamps from the FLAC
tracks may drift or become nonlinearly wrong. In that case, a naive "local FLAC
VAD -> production transcript labels" pipeline can produce confident but wrong
speaker labels.

If Auphonic can provide post-mastering speaker-track FLACs, or if it can provide
an edit/cut time map from raw input to mastered output, track-derived speaker
activity becomes much more practical.

## Offline Experiment Results, 2026-05-27

A local experiment used:

- production/mastered audio: `https://d2mmy4gxasde9x.cloudfront.net/cast_audio/pp_64.m4a`
- local FLAC tracks:
  - `pp_64-01-Jochen.flac`
  - `pp_64-02-Johannes.flac`
  - `pp_64-03-Ronny.flac`
  - `pp_64-04-Dominik.flac`
- local labeled transcript: `pp_64/transcripts/pp_64.json`

### Timeline / FLAC activity check

The production/local rendered M4A duration is `3975.09s`. The local FLAC tracks
are `4023.65s`, about `48.6s` longer. This confirms that direct timestamps from
the local FLAC tracks cannot be assumed to match the production/mastered audio.

A simple per-track energy/VAD experiment against labeled transcript timestamps
was not reliable enough:

- same-timestamp accuracy: about 55%
- best constant-offset accuracy: about 64%
- the known Johannes passage was not cleanly dominant on the Johannes track by
  simple energy, likely because of bleed, editing/timeline changes, or both

Conclusion: use local pre-Auphonic FLAC tracks as timing labels only after
explicit timeline alignment. Without that, they are better used as speaker
reference material.

### pyannote 3.1 baseline and tuning

Testing `pyannote/speaker-diarization-3.1` on the production M4A with
`num_speakers=4` reproduced the failure shape:

- baseline cluster distribution over labeled transcript segments: roughly
  `566 / 549 / 127 / 7`
- `exclusive_speaker_diarization` did not materially change the result
- one real speaker was effectively missing as a useful cluster

For model `pyannote/speaker-diarization-3.1`, the loaded pipeline used
agglomerative clustering parameters rather than the VBx parameters exposed by
newer/community pipeline defaults. The instantiated clustering parameters were
approximately:

```python
{
    "segmentation": {"min_duration_off": 0.0},
    "clustering": {
        "method": "centroid",
        "min_cluster_size": 12,
        "threshold": 0.7045654963945799,
    },
}
```

A tuned experiment with a lower clustering threshold recovered a much more
balanced fourth cluster:

```python
{
    "segmentation": {"min_duration_off": 0.0},
    "clustering": {
        "method": "centroid",
        "min_cluster_size": 12,
        "threshold": 0.55,
    },
}
```

Observed result:

- cluster distribution: roughly `555 / 384 / 183 / 129`
- segment accuracy against the local labeled transcript after majority mapping:
  about 87%
- the known Johannes passage mapped to Johannes

This does not prove the setting generalizes, but it is a concrete candidate for
a retry/tuning strategy when default pyannote produces a pathological cluster
imbalance.

`pyannote/speaker-diarization-community-1` was initially blocked by gated-model
access. The repeatable spike below was later able to load it with the local
Hugging Face token and accepted terms.

### Known-speaker voiceprint experiment

A speaker embedding experiment used:

```text
pyannote/wespeaker-voxceleb-resnet34-LM
```

Two reference strategies were tested against production mono transcript
segments:

1. Mono reference snippets from the production audio using known local transcript
   labels.
2. Local FLAC speaker tracks as reference material, then classifying production
   mono audio segments.

Results on a held-out segment sample:

- mono references -> mono segments: `240 / 243` correct, about 98.8%
- FLAC references -> mono segments: `241 / 243` correct, about 99.2%
- the known Johannes passage was correctly classified as Johannes in both
  strategies

The FLAC-reference margins were lower than mono-reference margins, but still
promising. This is the strongest current evidence for known-speaker podcasts:
local FLAC tracks may be more useful as voiceprint/reference material than as
untrusted timing labels.

## Repeatable Research Spike, 2026-05-27

A repeatable scratch prototype was created outside the repository at:

```text
/tmp/voxhelm-diarization-spike/research_spike.py
```

The prototype writes JSON result artifacts under `/tmp/voxhelm-diarization-spike`
and keeps generated audio/features out of the repo.

### Environment and timing reproduction

The production M4A downloaded from CloudFront has duration `3975.093696s`.
Each local separated FLAC track has duration `4023.652271s`. The FLAC tracks
are therefore about `48.56s` longer than the mastered production file.

The local labeled transcript contains 1,249 segments:

- Dominik: 183 segments, `434.28s`
- Jochen: 149 segments, `408.50s`
- Johannes: 382 segments, `1060.74s`
- Ronny: 535 segments, `1553.00s`

The repeatable per-track energy/VAD check confirmed that direct local-FLAC
timing is unsafe against the mastered transcript timeline:

- same transcript timestamps: `53.8%` segment accuracy
- best constant offset in `[-60s, +60s]`: `+16s`, `62.8%` segment accuracy

The known Johannes passage is mostly assigned to Jochen by this simple direct
energy method at same timestamps. With the best constant offset it is still
mixed and not reliable. This supports the conclusion that local pre-Auphonic
FLACs are useful as voice references, not as direct timing labels, unless a
mastered-timeline alignment or post-mastering stems are available.

### Known-speaker voiceprint prototype

The repeatable voiceprint prototype used:

```text
pyannote/wespeaker-voxceleb-resnet34-LM
```

Reference strategy:

1. Decode each local speaker FLAC to 16 kHz mono.
2. Select 16 high-energy, spaced 5-second windows from each separated track.
3. Compute one embedding per window.
4. Average and normalize one centroid per known speaker.
5. Decode the production M4A to 16 kHz mono.
6. Embed transcript segment audio windows and assign to the nearest speaker
   centroid by cosine similarity.
7. Report top similarity, second similarity, and margin.

Results over all 1,249 transcript segments, padding short segments to a minimum
embedding window of 1.5 seconds:

- accuracy: `95.4%`
- predicted distribution: Dominik `191`, Jochen `138`, Johannes `390`, Ronny
  `530`
- margin median: `0.372`
- margin p10: `0.202`

Confusion matrix, rows = local transcript speaker, columns = predicted speaker:

| speaker | Dominik | Jochen | Johannes | Ronny |
| --- | ---: | ---: | ---: | ---: |
| Dominik | 174 | 1 | 5 | 3 |
| Jochen | 10 | 133 | 5 | 1 |
| Johannes | 4 | 2 | 367 | 9 |
| Ronny | 3 | 2 | 13 | 517 |

Results restricted to segments at least 1.5 seconds long:

- evaluated segments: `1020`
- skipped short segments: `229`
- accuracy: `99.4%`
- predicted distribution: Dominik `135`, Jochen `107`, Johannes `315`, Ronny
  `463`
- margin median: `0.384`
- margin p10: `0.252`

Confusion matrix for the `>=1.5s` subset:

| speaker | Dominik | Jochen | Johannes | Ronny |
| --- | ---: | ---: | ---: | ---: |
| Dominik | 132 | 0 | 0 | 1 |
| Jochen | 3 | 106 | 1 | 0 |
| Johannes | 0 | 1 | 314 | 0 |
| Ronny | 0 | 0 | 0 | 462 |

Margin thresholds are useful for production confidence handling. On the all
segment run:

- margin `>=0.10`: `1204 / 1249` coverage, `97.3%` accuracy
- margin `>=0.15`: `1175 / 1249` coverage, `98.3%` accuracy
- margin `>=0.20`: `1129 / 1249` coverage, `98.5%` accuracy

On the `>=1.5s` subset:

- margin `>=0.15`: `1002 / 1020` coverage, `99.6%` accuracy
- margin `>=0.20`: `981 / 1020` coverage, `99.6%` accuracy

The known passage around `00:38:16-00:38:32` was correctly classified by the
voiceprint prototype:

- Johannes segment `2295.689-2298.089`: Johannes, margin `0.248`
- Johannes segment `2298.329-2302.009`: Johannes, margin `0.413`
- Johannes segment `2302.009-2303.269`: Johannes, margin `0.449`
- Dominik interruption `2304.609-2304.969`: Dominik, margin `0.006`
- Johannes segment `2304.969-2309.409`: Johannes, margin `0.261`
- Johannes segment `2309.409-2311.709`: Johannes, margin `0.334`

The short Dominik "Falsch." interruption is correctly classified but has a very
low margin. That is a useful production signal: very short or low-margin
segments should be represented as uncertain rather than silently auto-accepted.

### Embedding model comparison after Hugging Face approval

After the gated Hugging Face access was approved, the voiceprint prototype was
rerun with `pyannote/embedding` and compared against the previous
`pyannote/wespeaker-voxceleb-resnet34-LM` result. The same reference strategy
was used: 16 high-energy 5-second windows per local speaker FLAC, one centroid
per contributor, and production mono transcript segments classified by cosine
similarity.

`pyannote/embedding` loads successfully with the local token, but requires
`omegaconf` in the Python environment. The comparison command used a temporary
`uv --with omegaconf` dependency. If this model becomes a supported option,
`omegaconf` must be added to Voxhelm's diarization extra.

| embedding model | segment set | evaluated | accuracy | margin median | margin p10 |
| --- | --- | ---: | ---: | ---: | ---: |
| `pyannote/wespeaker-voxceleb-resnet34-LM` | all segments | 1249 | `95.4%` | `0.372` | `0.202` |
| `pyannote/embedding` | all segments | 1249 | `94.9%` | `0.242` | `0.102` |
| `pyannote/wespeaker-voxceleb-resnet34-LM` | `>=1.5s` | 1020 | `99.4%` | `0.384` | `0.252` |
| `pyannote/embedding` | `>=1.5s` | 1020 | `98.4%` | `0.258` | `0.148` |

At equivalent margin thresholds, `pyannote/embedding` can be very precise but
has lower coverage because its margins are smaller:

| embedding model | segment set | threshold | coverage | accuracy |
| --- | --- | ---: | ---: | ---: |
| `pyannote/wespeaker-voxceleb-resnet34-LM` | all segments | `>=0.15` | `1175 / 1249` | `98.3%` |
| `pyannote/embedding` | all segments | `>=0.15` | `1032 / 1249` | `99.0%` |
| `pyannote/wespeaker-voxceleb-resnet34-LM` | `>=1.5s` | `>=0.15` | `1002 / 1020` | `99.6%` |
| `pyannote/embedding` | `>=1.5s` | `>=0.15` | `910 / 1020` | `99.9%` |
| `pyannote/wespeaker-voxceleb-resnet34-LM` | `>=1.5s` | `>=0.20` | `981 / 1020` | `99.6%` |
| `pyannote/embedding` | `>=1.5s` | `>=0.20` | `761 / 1020` | `100.0%` |

Both models correctly classify the Johannes parts of the known passage.
`pyannote/embedding` misclassifies the short Dominik "Falsch." interruption as
Johannes with a low margin (`0.067`), while the WeSpeaker model classifies it as
Dominik with an even lower margin (`0.006`). In both cases this interruption
should be treated as uncertain rather than accepted automatically.

Current conclusion: keep `pyannote/wespeaker-voxceleb-resnet34-LM` as the
preferred known-speaker embedding model for this podcast workflow. It has
slightly better raw accuracy, much better margin separation, higher useful
coverage at practical thresholds, and no newly observed `omegaconf` dependency.
`pyannote/embedding` remains a viable alternative for high-precision/low-coverage
review modes, but it is not the better default based on this episode.

### pyannote tuning comparison

The repeatable pyannote sweep used `pyannote/speaker-diarization-3.1`,
`num_speakers=4`, preloaded ffmpeg-decoded audio, and majority mapping from
anonymous labels to the local transcript speakers.

Baseline reproduced the bad cluster distribution:

- `speaker_diarization`: raw segment distribution `566 / 549 / 127 / 7`
- majority-mapped accuracy: `77.4%`
- `exclusive_speaker_diarization`: raw segment distribution `569 / 549 / 124 / 7`
- majority-mapped accuracy: `77.1%`

Baseline `speaker_diarization` confusion matrix:

| speaker | Dominik | Jochen | Johannes | Ronny |
| --- | ---: | ---: | ---: | ---: |
| Dominik | 0 | 0 | 169 | 14 |
| Jochen | 3 | 115 | 26 | 5 |
| Johannes | 1 | 7 | 348 | 26 |
| Ronny | 3 | 5 | 23 | 504 |

The tuned thresholds improved the cluster distribution and accuracy:

| threshold | annotation | raw distribution | accuracy |
| ---: | --- | --- | ---: |
| 0.50 | speaker | `555 / 376 / 187 / 131` | `86.9%` |
| 0.50 | exclusive | `553 / 383 / 184 / 129` | `87.2%` |
| 0.55 | speaker | `555 / 377 / 186 / 131` | `87.0%` |
| 0.55 | exclusive | `553 / 384 / 183 / 129` | `87.3%` |
| 0.60 | speaker | `555 / 377 / 186 / 131` | `87.0%` |
| 0.60 | exclusive | `553 / 384 / 183 / 129` | `87.3%` |
| 0.65 | speaker | `555 / 377 / 186 / 131` | `87.0%` |
| 0.65 | exclusive | `553 / 384 / 183 / 129` | `87.3%` |

Best tuned confusion matrix, threshold `0.55` with
`exclusive_speaker_diarization`:

| speaker | Dominik | Jochen | Johannes | Ronny |
| --- | ---: | ---: | ---: | ---: |
| Dominik | 144 | 0 | 25 | 14 |
| Jochen | 9 | 117 | 16 | 7 |
| Johannes | 23 | 8 | 324 | 27 |
| Ronny | 7 | 4 | 19 | 505 |

`pyannote/speaker-diarization-community-1` is available with the local Hugging
Face token and was also tested with `num_speakers=4`. Its instantiated defaults
were:

```python
{
    "segmentation": {"min_duration_off": 0.0},
    "clustering": {"threshold": 0.6, "Fa": 0.07, "Fb": 0.8},
}
```

Community-1 did not materially outperform the tuned 3.1 run on this episode:

- `speaker_diarization`: raw distribution `562 / 357 / 201 / 129`, accuracy
  `86.5%`
- `exclusive_speaker_diarization`: raw distribution `563 / 356 / 203 / 127`,
  accuracy `86.9%`

Community-1 exclusive confusion matrix:

| speaker | Dominik | Jochen | Johannes | Ronny |
| --- | ---: | ---: | ---: | ---: |
| Dominik | 147 | 0 | 20 | 16 |
| Jochen | 14 | 116 | 12 | 7 |
| Johannes | 37 | 7 | 310 | 28 |
| Ronny | 5 | 4 | 14 | 512 |

In this repeatable pyannote run, the known Johannes passage maps to Johannes
after majority mapping for both baseline and tuned runs. The short Dominik
interruption in the middle maps to Johannes. This does not invalidate the
observed user-facing failure in django-cast, because anonymous speaker labels
can be mapped differently in the consumer and the baseline cluster is still
pathological: Dominik is mostly merged into the large Johannes-like cluster, and
one produced label has only 7 transcript segments.

### Recommendation from the spike

pyannote threshold tuning is useful as a cheap retry/fallback, but it is not a
sufficient production answer for known-speaker podcasts. On this episode it
improves segment accuracy from about `77%` to about `87%`, but known-speaker
voiceprints are substantially stronger: `95%` over all segments and `99%+` over
segments long enough for a stable embedding.

Recommended production direction:

1. Keep pyannote diarization as the anonymous baseline and fallback.
2. Add bad-diarization detection for pathological label distributions.
3. Prefer `exclusive_speaker_diarization` for transcript alignment when
   pyannote returns it.
4. Add a known-speaker embedding postprocessor for episodes with contributor
   reference clips.
5. Classify transcript segments directly from the mastered mono audio first,
   because the final artifact is segment-labeled and direct segment
   classification avoids depending on pyannote clusters that may have already
   merged speakers.
6. Use diarization turns only as a smoothing or fallback signal, not as the
   primary unit for known-speaker recovery, unless the turns are already known
   to be single-speaker and high quality.
7. Start with conservative auto-accept rules such as duration `>=1.5s`, margin
   `>=0.15`, and a top similarity floor around `0.55-0.60`. Treat short or
   low-margin segments as uncertain.
8. Preserve raw pyannote labels, voiceprint candidates, margins, and uncertainty
   flags in metadata so django-cast can expose manual correction instead of
   hiding the uncertainty.

Potential Voxhelm API/data additions:

```json
{
  "diarization": {
    "enabled": true,
    "num_speakers": 4,
    "strategy": "pyannote_known_speaker",
    "known_speakers": [
      {
        "id": "contributor-id",
        "name": "Johannes",
        "references": [
          {
            "kind": "clip_artifact",
            "artifact": "artifact-id-or-url"
          },
          {
            "kind": "source_range",
            "audio_artifact": "artifact-id-or-url",
            "start": 123.45,
            "end": 153.45
          }
        ]
      }
    ],
    "known_speaker": {
      "embedding_model": "pyannote/wespeaker-voxceleb-resnet34-LM",
      "min_segment_duration": 1.5,
      "auto_accept_margin": 0.15,
      "min_top_similarity": 0.55
    }
  }
}
```

Potential transcript segment metadata:

```json
{
  "speaker": "Johannes",
  "speaker_source": "known_speaker_voiceprint",
  "speaker_confidence": 0.81,
  "speaker_margin": 0.33,
  "speaker_candidates": [
    {"speaker": "Johannes", "similarity": 0.81},
    {"speaker": "Dominik", "similarity": 0.47}
  ],
  "speaker_uncertain": false,
  "raw_diarization_speaker": "SPEAKER_02"
}
```

For uncertain segments, set `speaker_uncertain=true` and either keep the
best-effort speaker with candidates or use `speaker=null` / `Unknown` depending
on the consumer contract. django-cast should be able to store contributor
reference clips or choose existing private media artifacts as references.
Voxhelm should own model-versioned embedding extraction and centroid caching for
repeated use.

Contributor voice references should be modeled in django-cast as private
enrollment/reference material attached to `Contributor`, not as public
contributor profile metadata. A reference does not have to come from the current
episode. Cross-episode clips should be useful for recurring contributors and are
probably the production-friendly path, as long as the clips contain clean solo
speech and the person is one of the expected episode contributors. Same-episode
references are still expected to be strongest because microphone, room,
language, energy, and mastering chain match the target audio. The next
validation should therefore compare references from earlier Python Podcast
episodes against `pp_64` before relying on cross-episode enrollment.

Recommended ownership split:

- django-cast stores private contributor reference clips or private source
  ranges, with consent/review state and source metadata.
- Voxhelm owns embedding extraction, model-versioned centroid caching, and
  segment/turn classification, because embedding model changes are a Voxhelm
  implementation detail.
- When the embedding model changes, Voxhelm re-extracts centroids from the
  stored references instead of requiring django-cast to migrate model-specific
  embedding blobs.
- django-cast stores returned speaker suggestions, confidence/margin metadata,
  and uncertainty flags as reviewable transcript state rather than treating
  speaker identity as automatically final.

Concrete next implementation slice if approved:

1. Add pyannote quality metadata and imbalance warnings to Voxhelm artifacts.
2. Prefer `exclusive_speaker_diarization` for alignment.
3. Add a non-default known-speaker postprocessor behind an explicit request flag
   that accepts contributor reference clips, computes model-versioned
   centroids, classifies mastered transcript segments, and emits per-segment
   candidate/margin metadata.
4. Add django-cast support for private contributor reference clips and a review
   UI path for uncertain segments.

Status update, 2026-05-29: item 3 is implemented. The known-speaker
voiceprint postprocessor and its `pyannote_known_speaker` request/response
contract landed; see `known-speaker-diarization.md`. It classifies mastered
mono transcript segments directly against contributor reference centroids,
keeps anonymous pyannote as a fallback/debug signal recorded per segment, and
emits a reviewable `speakers` artifact with candidates, confidence, margin, and
uncertainty. Items 1, 2, and 4 (pyannote imbalance warnings, exclusive-output
alignment, and the django-cast review UI) remain open follow-ups.

## Improvement Options

### 1. Add diagnostics and bad-diarization detection

Lowest-risk next product improvement.

Detect cases where the requested speaker count was technically produced but the
effective distribution is suspicious, for example:

- expected speakers: 4
- produced labels: 4
- one label has less than about 1-3% of speaking time or transcript segments
- the four-speaker distribution is nearly identical to a three-speaker run

Possible output metadata:

- requested speaker count
- produced label count
- effective speaker count
- per-label turn count
- per-label speaking duration
- per-label transcript segment count
- warnings such as `imbalanced_speaker_clusters`

This should not silently fail the job by default, but it should make the output
quality visible and allow django-cast/admin UI to warn users.

### 2. Export raw diarization artifacts

Add raw debug artifacts for diarization jobs:

- raw pyannote turns
- exclusive pyannote turns
- RTTM if practical
- speaker distribution summary

This would make failures inspectable without reverse-engineering Podlove output.

### 3. Prefer exclusive diarization for transcript alignment

When pyannote returns `exclusive_speaker_diarization`, use it for assigning
speakers to transcript segments.

Expected benefit:

- fewer overlap/boundary artifacts
- better match to single-speaker transcript segment formats

Expected limitation:

- will not fix the Johannes-style whole-speaker merge if the underlying cluster
  is already wrong.

### 4. pyannote model and parameter experiments

Run offline experiments on this episode before adding production-facing knobs.

Candidate model:

- `pyannote/speaker-diarization-community-1`

The repeatable spike can load this model with the current local Hugging Face
token. Its default settings did not beat tuned `speaker-diarization-3.1` on
`pp_64`, but it remains worth testing across more episodes before choosing a
default.

For `pyannote/speaker-diarization-3.1`, candidate agglomerative-clustering
parameters to sweep are:

- `clustering.threshold`: start with `0.55` because it improved this episode;
  also try `0.50`, `0.60`, `0.65`, `0.70`, `0.75`, `0.80`
- `clustering.method`: keep `centroid` initially
- `clustering.min_cluster_size`: try `6`, `8`, `12`, `16`
- `segmentation.min_duration_off`: `0.0`, `0.2`, `0.5`

For newer/community pyannote pipelines that expose VBx parameters, candidate
parameters are:

- `clustering.threshold`: `0.50`, `0.55`, `0.60`, `0.65`, `0.70`
- `clustering.Fa`: `0.03`, `0.07`, `0.15`
- `clustering.Fb`: `0.5`, `0.8`, `1.5`
- `segmentation.min_duration_off`: `0.0`, `0.2`, `0.5`

Also test `embedding_exclude_overlap=True` if constructing the pipeline directly
allows it. This may improve embeddings in crosstalk-heavy audio, but can hurt if
there is too little clean non-overlapping speech.

### 5. Track-derived speaker activity when timelines match

If post-mastering per-speaker FLACs are available, or if the mastered timeline
can be mapped back to the local tracks, use the tracks as an oracle-like
speaker-activity source.

Recommended shape:

1. Transcribe the final mastered mix.
2. Run speech activity detection on each aligned speaker track.
3. Suppress bleed/crosstalk by comparing per-track activity/energy.
4. Convert activity regions to speaker turns with known speaker names.
5. Align those turns to the transcript segments.

This should outperform mono pyannote for known multi-track podcast recordings.

Do not use pre-Auphonic track timestamps directly against post-Auphonic mastered
audio unless alignment has been verified.

### 6. Known-speaker voiceprint identification

Use clean local FLAC tracks as speaker reference material rather than direct
timing labels.

Possible approach:

1. Extract clean reference samples for each contributor.
2. Compute speaker embeddings for references.
3. Compute embeddings for diarized turns or transcript-aligned audio windows.
4. Assign windows to the nearest known speaker when confidence is high.
5. Keep anonymous labels or require manual review when confidence is low.

This may help recover Johannes even when mono unsupervised clustering merges him,
but it needs careful confidence handling and privacy consideration.

The first offline experiment strongly supports this path: using
`pyannote/wespeaker-voxceleb-resnet34-LM`, local FLAC references classified a
held-out sample of production mono segments with about 99% accuracy and correctly
identified the known Johannes passage.

### 7. Alternative diarization backends

Evaluate outside the main Python 3.14 Voxhelm process first:

- NVIDIA NeMo diarization / MSDD
- SpeechBrain diarization or embeddings
- WeSpeaker / CAM++ embeddings
- pyannoteAI hosted/premium models if privacy/cost are acceptable
- WhisperX integration only as a convenience wrapper; its diarization quality is
  usually tied to pyannote unless configured otherwise

NeMo is a strong comparison candidate but is best treated as a Linux/CUDA or
remote/container experiment rather than a macOS Apple Silicon in-process
Voxhelm dependency.

### 8. Manual range-level corrections

Speaker mapping alone cannot fix merged clusters. django-cast should eventually
support range-level speaker corrections.

Potential data model:

```json
{
  "start": 2296.0,
  "end": 2312.5,
  "speaker": "Johannes",
  "source": "manual",
  "note": "Corrected merged diarization"
}
```

Corrections should be stored as overlays on top of generated transcript
artifacts. Rendering/export can apply overlays without destroying the original
Voxhelm result.

## Recommended Experiment Plan

### Phase 0: completed local probe

A first local probe has already established:

- direct local-FLAC VAD is not reliable against the production/mastered timeline
  without explicit alignment
- `pyannote/speaker-diarization-3.1` threshold `0.55` is a promising retry
  candidate for this episode
- known-speaker embeddings using local FLAC references are highly promising

### Phase A: create a small gold set

Hand-label a small set of passages before tuning:

- the known Johannes passage around `00:38:16-00:38:32`
- 5-10 additional Johannes passages
- representative passages for Dominik, Jochen, and Ronny
- some overlap/crosstalk sections

Measure:

- segment-level speaker accuracy
- Johannes precision and recall
- per-speaker speaking-time distribution
- DER/JER if RTTM-style annotations are created

### Phase B: pyannote comparison

Run these offline on the episode:

1. Current deployed baseline: verify the exact production pyannote model/config,
   then compare it with the local `pyannote/speaker-diarization-3.1`,
   `num_speakers=4` baseline.
2. Same model with `min_speakers=4,max_speakers=4` to confirm equivalence.
3. Same model using `exclusive_speaker_diarization` for alignment.
4. `pyannote/speaker-diarization-community-1`, `num_speakers=4`, exclusive
   output. This now loads locally; repeat on more episodes because it did not
   beat tuned `speaker-diarization-3.1` on `pp_64`.
5. Small parameter sweep over the clustering parameters exposed by the selected
   pyannote model. For `speaker-diarization-3.1`, start with agglomerative
   threshold `0.55` and compare against the pretrained default threshold around
   `0.7045`.
6. Voiceprint post-processing using `pyannote/wespeaker-voxceleb-resnet34-LM`
   and contributor reference clips from local FLAC tracks. Keep it as the
   preferred embedding model for now; `pyannote/embedding` was tested after HF
   approval and worked, but had lower raw accuracy, smaller margins, and an
   additional `omegaconf` dependency.

### Phase C: separated-track feasibility

Inspect the local FLAC tracks and production mastered audio:

- Are durations identical?
- Do all files start at the same content point?
- Is there constant offset only, or nonlinear drift?
- Did Auphonic cut silence?
- Can Auphonic export post-mastering per-speaker tracks?
- Can Auphonic export cut/edit metadata?

If timelines match, run per-track VAD/activity detection and compare against the
gold set.

If timelines do not match, use FLACs as voiceprint/reference material or manual
correction aids instead of direct timestamp sources. This is the current case for
`pp_64`: the local FLAC tracks are about `48.6s` longer than the mastered M4A.

### Phase D: external backend comparison

Run at least one external diarization comparison:

- NeMo diarization with oracle speaker count = 4
- export RTTM
- compare the same gold passages

Keep this out of the main Voxhelm dependency stack until it proves useful.

## Implementation Implications

Potential Voxhelm follow-ons:

- prefer exclusive pyannote output for transcript labels
- emit raw diarization/debug artifacts
- store diarization quality metadata and warnings
- add retry/experiment support for pyannote parameter sets
- add a pluggable diarization backend boundary for pyannote, track-derived
  activity, external NeMo, or known-speaker identification
- add a known-speaker postprocessor that computes segment/turn embeddings from
  the mastered mono audio and maps them to contributor reference embeddings
- optionally support multi-input jobs if aligned speaker tracks become available

Potential django-cast follow-ons:

- show diarization quality warnings in the transcript/admin UI
- keep speaker mapping for well-separated anonymous clusters
- add range-level speaker correction overlays for merged clusters
- optionally manage private, reusable contributor voice reference clips or
  source ranges for known-speaker recognition
- optionally attach post-mastering speaker stems to a transcript generation
  request when available

## Risks and Trade-offs

- pyannote tuning is cheap but may only move errors around.
- exclusive diarization improves alignment semantics but cannot recover a missing
  acoustic cluster.
- track-derived activity can be very accurate, but only with timeline-aligned
  tracks.
- pre-Auphonic FLACs against post-Auphonic MP3 are risky because silence cutting
  can create nonlinear timestamp mismatch.
- voiceprint identification can help known-speaker podcasts, but needs clean
  references, confidence thresholds, and privacy handling.
- NeMo and similar alternatives may improve quality but add deployment/runtime
  complexity, especially on macOS Apple Silicon and Python 3.14.
- hosted diarization APIs may improve quality but introduce cost and audio
  privacy concerns.
- manual correction is reliable but costs human time.

## Success Criteria

Minimum success for the problem episode:

- the `00:38:16-00:38:32` Johannes passage is labeled Johannes
- the fourth speaker is no longer represented by only a tiny number of segments
- speaker distribution is plausible for a four-speaker episode

Better success:

- Johannes recall on the hand-labeled gold windows is at least about 80%
- Johannes precision remains acceptable, about 80% or better
- labels for the other three speakers do not regress significantly
- Voxhelm detects and reports pathological cluster imbalance

Operational success:

- bad diarization is visible in metadata/UI instead of silently accepted
- raw turns are available for debugging
- django-cast has a path to manually correct merged-speaker regions

## Open Questions

- Which pyannote model/config is deployed in current production runs, and does
  it match the local `pyannote/speaker-diarization-3.1` baseline?
- Can Auphonic provide post-mastering per-speaker FLACs for future episodes?
- Can Auphonic provide a silence-cut/edit time map?
- Are local pre-Auphonic tracks offset-only relative to the mastered MP3, or is
  there nonlinear drift?
- How much clean solo speech does Johannes have in this episode?
- Is sending audio to pyannoteAI or another hosted API acceptable?
- Are contributor voice reference clips acceptable to store and process?
- How well do cross-episode contributor voice references perform compared with
  same-episode references on Python Podcast material?
- Should Voxhelm receive private signed clip URLs, copied job artifacts, or
  precomputed embeddings from django-cast?
- Should Voxhelm support multi-input transcription jobs, or should track-derived
  speaker activity be precomputed outside Voxhelm and uploaded as an artifact?
