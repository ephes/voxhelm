# Known-Speaker Diarization Contract

Status: implemented
Date: 2026-05-29

Anonymous pyannote diarization is fallback-only for known-speaker podcasts (see
`diarization-quality-research.md`). This adds a known-speaker voiceprint
postprocessor: transcript segments from the mastered mono audio are classified
directly against contributor reference voiceprints, which the research measured
at ~95% over all segments and ~99%+ over segments long enough for a stable
embedding.

The episode voiceprint extension for short segments is described in
`../docs/design/diarization-quality.md`.

## Request

`POST /v1/jobs` with `job_type=transcribe`. The existing anonymous diarization
contract is unchanged; known-speaker mode is opt-in through
`diarization.strategy`.

```json
{
  "diarization": {
    "enabled": true,
    "strategy": "pyannote_known_speaker",
    "known_speakers": [
      {
        "id": "12",
        "name": "Johannes",
        "references": [
          {
            "kind": "source_range",
            "audio": {"kind": "url", "url": "https://cdn.example.com/pp_60.m4a"},
            "start": 120.0,
            "end": 150.0
          },
          {
            "kind": "clip_artifact",
            "audio": {"kind": "url", "url": "https://cdn.example.com/clip.wav"}
          }
        ]
      }
    ],
    "known_speaker": {
      "embedding_model": "pyannote/wespeaker-voxceleb-resnet34-LM",
      "min_segment_duration": 1.5,
      "auto_accept_margin": 0.15,
      "min_top_similarity": 0.55,
      "mixed_window_duration": 1.5,
      "mixed_window_hop": 0.75,
      "mixed_window_margin": 0.15,
      "mixed_window_min_similarity": 0.3,
      "turn_corroborated_min_similarity": 0.35,
      "resegment_hop": 0.25
    }
  }
}
```

Rules:

- `strategy` is `pyannote` (default, anonymous) or `pyannote_known_speaker`.
  When omitted or `pyannote`, the stored/normalized payload is unchanged from
  the prior anonymous contract.
- `known_speakers` and `known_speaker` require
  `strategy=pyannote_known_speaker`. `known_speakers` must be a non-empty list;
  each entry needs a unique non-empty `id`, a non-empty `name`, and a non-empty
  `references` list.
- A reference is `clip_artifact` (whole referenced audio) or `source_range`
  (a `start`/`end` window, `start` < `end`).
- Reference `audio` is `{kind: "url", url}`. URL hosts must be on
  `VOXHELM_ALLOWED_URL_HOSTS` (the production path is source ranges into
  already-public mastered audio). `{kind: "upload", upload_id}` references are
  rejected with 400 at submission: fetching them would need signed private
  delivery, which neither local execution nor remote workers implement, and
  accepting them only to fail after the full STT run wasted GPU time.
- `known_speaker` thresholds default to the research recommendation: WeSpeaker
  embeddings, `min_segment_duration=1.5`, `auto_accept_margin=0.15`,
  `min_top_similarity=0.55`.
- The `mixed_window_*` options drive the mixed-segment veto (see "Acceptance
  policy"): `mixed_window_duration=1.5` and `mixed_window_hop=0.75` in seconds,
  `mixed_window_margin=0.15`, `mixed_window_min_similarity=0.3`. All are
  optional, finite and non-negative; `mixed_window_hop` must be at least 0.1
  and `mixed_window_duration` either 0 or at least 0.5.
  `mixed_window_duration=0` turns the veto off.
- `turn_corroborated_min_similarity=0.35` is the lower similarity floor for
  segments the anonymous diarization corroborates (see "Turn corroboration").
  Optional, finite and non-negative; `0` turns it off.
- `resegment_hop=0.25` is the step in seconds of the speaker track used to
  re-cut segments at speaker changes (see "Re-cut at speaker changes"). `0`
  turns the re-cut off; otherwise it must be at least 0.1. The re-cut is also
  off while `mixed_window_duration` is 0.
- `short_segment_margin=0.15` controls the second episode-voiceprint pass.
  It must be a finite non-negative number (booleans/strings are rejected).
  `0` disables this pass entirely; it does **not** mean acceptance without a
  margin. Larger values require more separation between the two best profiles.
- `num_speakers`/`min_speakers`/`max_speakers` may still be sent; they drive the
  anonymous pyannote run. Its labels are recorded per segment and feed turn
  corroboration, the speaker-track guard, the re-cut at unmapped labels and
  `raw_diarization_labels`. Clients should not pin `num_speakers` to the number
  of participants for known-speaker jobs: pyannote can then spend a label on
  noise and merge two people (measured 2026-10-15 on Django Chat, see
  `2026-10-12_mixed_speaker_segments.md`, "Speaker count and ASR pilot").

## Reference source lifetime and interval decoding

Both local and remote reference loaders apply a 9,600,000-frame ceiling (ten
minutes at 16 kHz; 19.2 MB PCM16 and 38.4 MB float32). This applies to complete
clips and requested intervals, not ordinary full job audio. ffmpeg emits at most
one frame past the ceiling so oversized inputs fail rather than silently truncate.
The WAV channel/width/rate and frame count are validated before reading samples
or converting to float32. Exactly-at-limit references pass.

Decoding uses quiet ffmpeg output and drains/discards stdout and stderr with a
combined 64 KiB byte budget for references. Full job audio drains and discards
diagnostics without a byte ceiling. Overflow kills/reaps the decoder and returns a fixed
KnownSpeakerError. Timeout, nonzero exit and overflow all remove temporary WAVs;
raw diagnostics never appear in API errors.

Local and remote workers materialize each exact reference URL once per job attempt.
References to the job's original URL reuse its already downloaded source, including
video sources before audio extraction. Query strings remain part of URL identity;
distinct URLs are independent snapshots. URL policy checks still apply on reuse.
Reference-owned files are removed once reference windows are built, including on
failure; the main input remains owned by its caller. There is no cross-job URL cache.

For `source_range`, ffmpeg uses output seek and duration limits to write only the
requested PCM interval before allocating NumPy samples. Output seeking may decode
and discard preceding compressed frames to preserve codec priming and resampler
history; CPU work can therefore grow with the interval offset, while PCM storage
and NumPy memory are bounded by the requested duration. Start/end use the same nearest-sample rounding
at 16 kHz as the former array slice; ranges crossing EOF end at EOF and ranges beyond
EOF are empty. Reference window selection receives only those interval samples.
Whole `clip_artifact` references decode from the beginning within the reference
ceiling; the main job postprocessing decode remains a whole source. Decode deadlines and temporary-file cleanup apply to both paths.

## Response

A new exposed `speakers` artifact (`transcript.speakers.json`) carries the
reviewable per-segment suggestions. Known-speaker results are suggestions, so
the public Podlove/DOTe/VTT artifacts are intentionally left **unlabeled**; the
consumer applies speaker identity only after review/approval. Each sidecar
segment still carries the confident speaker name (or `null` when uncertain).

```json
{
  "version": 1,
  "summary": {
    "strategy": "pyannote_known_speaker",
    "embedding_model": "pyannote/wespeaker-voxceleb-resnet34-LM",
    "embedding_version": "pyannote/wespeaker-voxceleb-resnet34-LM",
    "known_speakers": ["Dominik", "Johannes"],
    "min_segment_duration": 1.5,
    "auto_accept_margin": 0.15,
    "min_top_similarity": 0.55,
    "mixed_window_duration": 1.5,
    "mixed_window_hop": 0.75,
    "mixed_window_margin": 0.15,
    "mixed_window_min_similarity": 0.3,
    "turn_corroborated_min_similarity": 0.35,
    "resegment_hop": 0.25,
    "speaker_track_hop": 0.25,
    "segment_count": 1249,
    "confident_segment_count": 1175,
    "uncertain_segment_count": 74,
    "mixed_segment_count": 12,
    "turn_corroborated_segment_count": 40,
    "resegmented_segment_count": 130,
    "speaker_track_segment_count": 80,
    "confident_speaker_distribution": {"Johannes": 390, "Dominik": 191},
    "margin_median": 0.372,
    "raw_diarization_available": true,
    "raw_diarization_labels": {
      "Speaker 1": {"speaker": "Johannes", "seconds": 1630.2},
      "Speaker 2": {"speaker": "Dominik", "seconds": 1188.7},
      "Speaker 3": {"speaker": null, "seconds": 912.4}
    }
  },
  "segments": [
    {
      "index": 0,
      "start": 2295.689,
      "end": 2298.089,
      "speaker": "Johannes",
      "speaker_source": "known_speaker_voiceprint",
      "speaker_confidence": 0.81,
      "speaker_margin": 0.33,
      "speaker_candidates": [
        {"speaker": "Johannes", "similarity": 0.81},
        {"speaker": "Dominik", "similarity": 0.48}
      ],
      "speaker_uncertain": false,
      "raw_diarization_speaker": "Speaker 2"
    }
  ]
}
```

A segment the mixed-segment veto downgraded carries two more fields. They are
absent on every other segment, and the artifact `version` stays 1:

```json
{
  "index": 1,
  "start": 10.02,
  "end": 16.4,
  "speaker": null,
  "speaker_confidence": 0.627918,
  "speaker_margin": 0.525005,
  "speaker_uncertain": true,
  "speaker_uncertain_reason": "mixed_speakers",
  "speaker_mixed_windows": [
    {"start": 10.77, "end": 12.27, "speaker": "Carlton", "similarity": 0.444214, "margin": 0.277257}
  ]
}
```

`speaker_candidates`, `speaker_confidence` and `speaker_margin` still describe
the whole segment, so its main speaker remains visible for review.
`speaker_mixed_windows` lists only the windows that voted for another speaker.
`uncertain_segment_count` includes the `mixed_segment_count` segments.

A segment accepted by turn corroboration carries
`"speaker_accept_reason": "turn_corroborated"`; whole-segment acceptances
have no accept reason. Its `speaker_confidence` is below `min_top_similarity` by
construction. `confident_segment_count` and `confident_speaker_distribution`
include the `turn_corroborated_segment_count` segments.

A segment named by the speaker track carries
`"speaker_accept_reason": "speaker_track"` and is counted in
`speaker_track_segment_count`. Its `speaker` can differ from the first entry
of `speaker_candidates`, which still describes the whole-segment embedding.
`resegmented_segment_count` is the number of segments that came out of a
split. `segment_count` and the segment `index` refer to the re-cut segments.
`speaker_track_hop` is the step the speaker track actually used: the
configured `resegment_hop`, or a wider step for very long audio (see "Cancellation
and track work budget"); it is `null` when no track was built.

`raw_diarization_labels` lists every label of the anonymous diarization run
with its seconds of speech from the raw turns and the
known speaker it is tied to by the vote described under "Turn corroboration",
or `null` when it is tied to nobody. It is `{}` without diarization turns.
Voxhelm only names speakers it has a reference for; a consumer that knows the
episode's other participants can use the untied labels to name them. For
example, with exactly one participant without a reference and exactly one
untied label, the uncertain segments whose `raw_diarization_speaker` is that
label are most likely that participant (the "Speaker N" mapping such a
consumer keeps for anonymous transcripts does not survive a known-speaker
run, because the public artifacts are unlabeled). Labels are numbered per
job, so they cannot be matched to the labels of another run.

The job `result.metadata.diarization` also carries a `known_speaker_summary`
(the summary block above) and a non-sensitive `known_speakers` list of
`{id, name, reference_count}` — never the private reference URLs/ranges.

## Acceptance policy

A segment is auto-accepted (gets a sidecar `speaker`) when
`duration >= min_segment_duration` and `top_similarity >= min_top_similarity`
and `margin >= auto_accept_margin`, and the mixed-segment veto below does not
object. A segment that misses only the similarity threshold can still be
accepted through "Turn corroboration" below. Every other segment is
`speaker_uncertain=true` with `speaker=null`, keeping the best-effort
candidate list for review. The
sidecar is a suggestion: it never writes the public transcript artifacts. Raw
anonymous pyannote labels are preserved per segment for audit and remapping.

Each classification normalizes the probe embedding once and reuses it across
centroid scores. Cosine scores use finite, equal-dimensional vectors, and ties
are ordered by speaker name.

### Mixed-segment veto

One embedding per segment hides a second voice: a segment that holds a short
interjection from another speaker still classifies confidently as its main
speaker, and anonymous pyannote often emits no turn for such an interjection
(see `2026-10-12_mixed_speaker_segments.md`). A segment that passes the rule
above is therefore checked again:

- Windows of `mixed_window_duration` seconds slide over the segment in steps of
  `mixed_window_hop`, plus one window aligned to the segment end. Segments no
  longer than one window are not checked, and a window that the end of the
  audio clips to under 0.5 s is skipped.
- Each window is classified against the same centroids. It counts as evidence
  of another speaker when its top speaker differs from the segment's, its
  margin is at least `mixed_window_margin` and its top similarity is at least
  `mixed_window_min_similarity`. The similarity floor is lower than
  `min_top_similarity` because short windows score lower than whole segments;
  it still excludes music and silence, which resemble no speaker.
- One such window is enough: the segment becomes `speaker_uncertain=true` with
  `speaker=null`, `speaker_uncertain_reason="mixed_speakers"` and the windows in
  `speaker_mixed_windows`.

The veto only turns confident suggestions into uncertain ones. It does not
change segment boundaries, indices or the public transcript artifacts, and it
does not split the segment. Only segments that would otherwise be accepted are
checked (by the rule above or by turn corroboration), so the extra cost is a
few embeddings per such segment.

### Turn corroboration

`min_top_similarity` rejects many single-speaker segments whose embedding is
merely weak (short phrases, laughter in the voice, a different microphone
distance) although the margin to the other speakers is clear. The anonymous
pyannote run is independent evidence for those. A segment that fails only the
similarity threshold is accepted anyway when all of this holds:

- `duration >= min_segment_duration`, `margin >= auto_accept_margin` and
  `turn_corroborated_min_similarity <= top_similarity < min_top_similarity`.
- The diarization puts one speaker on the segment: other diarization labels
  overlap it for less than 0.3 s in total.
- That diarization label stands for the segment's top speaker in this job. A
  label is tied to a known speaker when at least 5 confident segments with
  only that label exist and at least 90% of them name the same speaker.
  Segments the mixed-segment veto downgraded do not vote.
- The mixed-segment veto finds no window for another speaker. If it does, the
  segment stays uncertain and carries `speaker_uncertain_reason` and
  `speaker_mixed_windows` like any vetoed segment, so `mixed_segment_count`
  counts it as well.

Such a segment gets its `speaker`, `speaker_uncertain=false` and
`speaker_accept_reason="turn_corroborated"`. Without raw diarization turns, or
with `turn_corroborated_min_similarity=0`, nothing is corroborated. The rule
never settles a close call between two known speakers and never overrides the
veto. The measurements are in `2026-10-12_mixed_speaker_segments.md`.

### Re-cut at speaker changes

Whisper often emits long, unpunctuated segments cut by time, so a segment can
start with one speaker and end with another. Naming or blanking such a segment
is wrong either way. When word timings are available, Voxhelm therefore
re-cuts the segments before classifying them:

- Word timings come from whisper.cpp or, with
  `VOXHELM_KNOWN_SPEAKER_STT_BACKEND=parakeet`, from Parakeet (falling back
  to whisper.cpp; see `2026-10-15_parakeet_asr_backend.md`). Known-speaker
  jobs with the re-cut on run whisper.cpp with full JSON output on one
  processor (token offsets are chunk-relative with several). Other backends
  return no word timings; their segments are left as they are and no track
  is built.
- The speaker track classifies a window of `mixed_window_duration` seconds
  every `resegment_hop` seconds over the whole file against the centroids. A
  frame is clear when it reaches `mixed_window_margin` and
  `mixed_window_min_similarity`.
- Clear frames of one speaker form a run of at least two frames. Between runs
  of different speakers the turn boundary is the last place where the top
  speaker flips, or the middle of the gap.
- A segment is split where its words fall into different turns. Pieces of a
  single word, or whose words or emitted boundaries span under 0.3 s, are
  folded into a neighbour, so no piece is empty or shares its start with the
  next one. A piece starts at its
  first word (the first keeps the segment start) and ends where the next one
  starts (the last keeps the segment end); its text is its words.
- The track only compares against the references, so it sees no change to
  someone without one, such as a guest. When the anonymous diarization has
  labels nobody is tied to (by the vote described under "Turn
  corroboration") and at least one label that is tied, the segments are
  classified once, and then the re-cut is repeated from Whisper's segments
  with the diarization turns of the untied labels cut into the track turns,
  each as a speaker of its own, and the segments are classified again. This
  costs a second classification pass on those jobs only; jobs where every
  label is tied, or none is, are re-cut once as before.
- The transcript artifacts and the sidecar are both written from the re-cut
  segments, so their start times match. The full transcript text is unchanged.

After the whole-segment rules, a segment that is still uncertain and was not
flagged by the veto takes the speaker the track supports for it, but only
when the anonymous diarization puts known speakers on it: every diarization
label overlapping the segment for 0.3 s or more (or the longest one, if none
does) must be tied to a known speaker in this job by the vote described under
"Turn corroboration". The label need not be the track's speaker, since
pyannote often gives a short interjection to the surrounding speaker. Without
raw diarization turns, or on a label nobody was tied to, the track names
nothing. This matters for episodes with a guest who has no reference: the
track only compares against the known references, and a guest's voice can
resemble a host's. The track supports a speaker when: every clear
frame centred inside the segment (0.4 s in from each edge when it is longer
than 0.8 s) names the same speaker, there is at least one such frame, and for
segments of 2.5 s and more at least 30% of the frames are clear. This is not
applied when the segment is at least `min_segment_duration` long and its own
embedding names another speaker with `margin >= auto_accept_margin`. The
mixed-segment veto runs against the track's speaker first; if it finds a
window for another speaker, the segment stays uncertain with
`speaker_uncertain_reason="mixed_speakers"`.

The re-cut changes the number and the boundaries of segments compared with
0.1.4; consumers that match sidecar and transcript by start time are not
affected. Word boundaries can be off by a word, because word timings are
approximate. Cost: one embedding per `resegment_hop` of audio and a slower
whisper.cpp run (see `2026-10-12_mixed_speaker_segments.md`).

## Ownership

- Voxhelm owns embedding extraction, model-versioned centroids, and
  classification. The embedding model is a Voxhelm implementation detail; when
  it changes, Voxhelm re-extracts centroids from the caller's references.
- django-cast owns private reference storage, consent/review state, and the
  reviewable transcript suggestion state. It sends contributor ids + names +
  private reference clips/ranges, never public profile URLs.

## Implementation

- `transcriptions/known_speaker.py`: config, reference/centroid math,
  classification, accept policy, mixed-segment veto, sidecar artifact, pyannote
  embedding backend
  (lazy, behind `KnownSpeakerEmbeddingBackendProtocol`).
- `jobs/services.py`: request parsing/validation, reference fetching, job
  orchestration, `speakers` artifact persistence, dedup by full diarization
  payload.
- `jobs/models.py`: `JobArtifact.Kind.TRANSCRIPT_SPEAKERS`
  (migration `0006_alter_jobartifact_kind`).

### Configuration and display-name admission

API submission, stored/claimed reconstruction and direct postprocessing use the
same numeric domain: finite JSON numbers (no booleans or numeric strings),
nonnegative thresholds, and durations/hops at most 600 seconds. This caps padding
allocation too. Mixed-window duration is zero (disabled) or at least 0.5 seconds;
mixed-window hop is at least 0.1 seconds; resegment hop is zero (disabled) or at
least 0.1 seconds. Zero still disables turn corroboration. Thresholds above the
reachable cosine score/margin remain usable to disable acceptance.

The existing sidecar keys are display names. Different speaker IDs must have
unique names after Unicode NFKC normalization, whitespace collapse and casefold;
submission rejects collisions. Stored/direct reference sets are checked before
embedding too. Multiple clips for the same ID and exact display name still merge
into one centroid; an ID carrying conflicting display names is refused.

### Cancellation and track work budget

Known-speaker postprocessing accepts the existing cancellation event and checks
it before and after every reference, track, segment and mixed-window embedding,
and between classification passes. The remote worker passes its lease-loss event;
revocation raises the existing cancellation exception, then the worker reports
lease loss without publishing a sidecar. In-process inference already running
finishes its current call before cancellation can take effect.

Track admission happens before any embedding in a postprocessing pass and before
any sample slicing in direct track construction. At most 32,768 track frames (one
embedding each) are built. The default 0.25-second hop covers about two hours and
sixteen minutes within that budget; longer audio widens the track step just
enough to stay within 32,768 frames (about 0.33 seconds for a three-hour
episode) instead of failing the job after transcription. The step used is
recorded as `speaker_track_hop` in the summary. Set `resegment_hop` to zero to
disable the track.

### Interval processing work

Tiny-piece folding retains the first/last segment boundaries, original word
objects and word order, minimum word/duration rules, and left-neighbour preference.
It folds linked contiguous word ranges in amortized linear work rather than
copying accumulated word lists or scanning all pieces after every fold.
Overlapping turns use an endpoint sweep with a heap: latest start wins, with
lexicographically greatest speaker name breaking equal-start ties. Adjacent turns
of the same speaker merge. Resolution takes O(n log n) work; unmapped overlays
look up the first possible overlap and visit only intersecting resolved intervals.

## Episode voiceprints for short cues

After all first-pass classification and any re-cut at unmapped labels, uncertain,
non-mixed cues of 0.3–2.5 s receive a second opinion from the episode itself.
At least five whole-segment accepted cues of 3 s or more are required per known
speaker. Only cues with no accept reason and no mixed windows train a profile;
the 150 highest-margin cues are averaged. No reference centroid is mixed into
these episode centroids. A speaker with fewer than five eligible cues is omitted.
Short cue audio is repeated to 1.5 s without silence or adjacent speech. A decision
requires the configured margin between the best and second profile. The existing
mixed veto is run against reference centroids before accepting a known winner.
Existing confident names and all transcript boundaries remain unchanged.

A known winner carries `speaker_accept_reason="episode_voiceprint"` and
`speaker_uncertain=false`. `speaker_candidates`, `speaker_confidence` and
`speaker_margin` still describe the first-pass reference embedding and may disagree
with the accepted name. Version 1 stays unchanged; all additions are optional.

The summary adds `short_segment_margin`, `episode_voiceprint_segment_count`
(new known-name acceptances), `episode_voiceprint_relabelled_count` (actual raw
label changes), and `episode_voiceprint_skipped`. A null skip means the pass ran;
`disabled` means option 0, `no_raw_turns` means the anonymous diarization produced
no usable turns (so an unreferenced guest could not be detected),
`unresolved_label` means a substantial untied label has no episode profile, and
`too_few_centroids` means fewer than two profiles.
Counts are zero on skipped jobs. The raw label vote and raw speech seconds
continue to use first-pass whole-segment acceptances and anonymous turns.

API submission, stored job execution and claimed worker execution share the same
numeric validation/default. Explicit thresholds are stored and included in the
existing request fingerprint, so changing the threshold creates a distinct job
for the same task_ref. Omitted and explicit 0.15 have the same execution behavior;
their stored request shapes/fingerprints need not be equal (historical fingerprints
are not rewritten). Local and remote summaries retain all four new fields,
including null for a pass that ran. Disable with:

```json
{"short_segment_margin": 0}
```

For complete runnable request objects, use the request example above with this
option added. Qualification commands and unmeasured real-data criteria live in
`../docs/design/diarization-quality.md`; no rollout is implied by this extension.

### Untied-label episode profiles and raw-label assignment

For every label the first-pass vote ties to nobody, the pass averages up to 150
uncertain, non-mixed single-label cues of at least 3 s, longest first. Five cues
are required. "Single label" uses the existing corroboration overlap tolerance
(other-label speech must total less than 0.3 s). Profiles include labels even
when the caller supplies no guest name or reference. At least two profiles must
exist overall, and every untied label with at least 30 s of raw speech needs a
profile; otherwise the whole pass is skipped. Smaller unresolved labels do not
block it. Known speakers with too few eligible long cues still have no profile.

Known-speaker and untied-label profiles compete on every eligible short cue,
regardless of its raw label. A host can therefore win on a guest's label, and a
guest's label can win on a host's label. On a label winner, `speaker` remains null
and `speaker_uncertain` remains true. No guest identity is invented by Voxhelm;
the consumer applies its usual exactly-one-unreferenced-participant / exactly-one-
untied-label rule, excluding mixed cues. Multiple untied labels stay distinct.

**Contract change:** `raw_diarization_speaker` is now the assigned label, normally
the longest-overlap label but possibly the episode voiceprint's winner. Only when
it changes, `raw_diarization_overlap_speaker` records the original overlap label
(including null when no turn overlapped). The raw summary seconds and label/name
votes remain based on the unchanged anonymous turns and original whole-segment
acceptances. No exclusive-turn selection is introduced by this change.

A margin-qualified decision adds `speaker_episode_candidates`, a ranked list of
objects containing either `speaker` or `label`, plus `similarity`. Reference
candidates/confidence/margin retain their original meanings. The episode list is
also available when the mixed veto blocks a known winner; that cue stays uncertain
with `speaker_uncertain_reason="mixed_speakers"`. No-decision cues have no episode
list. A label decision that retains the original label carries episode candidates
but no overlap field and does not increment `episode_voiceprint_relabelled_count`.

Example of a guest-label decision (the usual reference evidence is omitted here):

```json
{
  "speaker": null,
  "speaker_uncertain": true,
  "raw_diarization_speaker": "Speaker 3",
  "raw_diarization_overlap_speaker": "Speaker 1",
  "speaker_episode_candidates": [
    {"label":"Speaker 3", "similarity":0.88},
    {"speaker":"Will", "similarity":0.52},
    {"speaker":"Carlton", "similarity":0.41}
  ]
}
```
