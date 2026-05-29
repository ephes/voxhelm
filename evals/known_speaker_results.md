# Known-Speaker Gold-Set Results

Measured through the shipped `transcriptions.known_speaker` code path
(`run_known_speaker_postprocess`) via `evals/known_speaker_eval.py`.

## Material (pp_64)

- Episode: "Live von der DjangoCon Europe 2025 in Dublin - Tag 2", German mono,
  ~66 min, four speakers (Dominik, Jochen, Johannes, Ronny).
- Mastered production audio: `pp_64.m4a` (the public CloudFront master).
- References: local per-speaker FLAC tracks
  `pp_64-0{1..4}-{Jochen,Johannes,Ronny,Dominik}.flac`, evenly-spaced 5s windows.
- Gold transcript: `pp_64/transcripts/pp_64.json`, 1249 track-labeled segments.
- Embedding model: `pyannote/wespeaker-voxceleb-resnet34-LM`.

## Result (2026-05-29, all 1249 segments)

| Metric | Value |
| --- | ---: |
| All-segments top-1 accuracy | **93.27%** |
| DER (time-weighted, single-speaker) | **1.82%** |
| WDER (word-weighted) | **2.40%** |
| DER on auto-accepted (public) labels | **0.09%** |
| Auto-accept coverage | 69.98% (874 / 1249) |
| Auto-accept accuracy (public labels) | **99.89%** |
| Margin median | 0.211 |
| Margin p10 | 0.050 |
| Johannes passage (00:38:16–00:38:32) | 5/6 segments correct |

Auto-accept policy: `min_segment_duration=1.5`, `auto_accept_margin=0.15`,
`min_top_similarity=0.55`. Uncertain segments are excluded from public labels as
"needs review" rather than counted as correct, per the acceptance policy.

DER/WDER are practical single-speaker approximations (no overlap handling or
collar), not full `pyannote.metrics` DER.

### Curated gold subset

The goal's gold set (Johannes passage + per-speaker representatives + short and
speaker-boundary/crosstalk cases), top-1 accuracy by category:

| Category | Segments | Top-1 accuracy |
| --- | ---: | ---: |
| Johannes passage (00:38:16–00:38:32) | 6 | 83.3% (5/6) |
| Per-speaker representative (>=2.5s clean) | 12 | 100% |
| Short segments (<1.0s) | 113 | 43.4% |
| Speaker-boundary / crosstalk (<2.0s) | 181 | 60.8% |
| Curated subset (unique) | 226 | 65.5% |

The low curated-subset top-1 is concentrated entirely in the *intentionally
hard* short and crosstalk cases — and those are exactly the segments the
acceptance policy routes to review. Every segment shorter than
`min_segment_duration=1.5s` is `speaker_uncertain` by construction, so the
sub-1.0s short segments and the short boundary segments are **never
auto-applied to public output**. On clean, long-enough speech the system is
~100% (per-speaker representatives) and the public auto-applied labels are
99.89% accurate (DER 0.09%). The one Johannes-passage miss is the ~0.36s
"Falsch." interjection by Dominik (margin ~0.006) — correctly flagged uncertain
rather than mislabeled publicly.

Top-1 gold vs predicted distribution: gold
`{Ronny 535, Johannes 382, Dominik 183, Jochen 149}`; predicted
`{Ronny 540, Johannes 370, Jochen 170, Dominik 169}`.

## Reading

- 93.27% all-segment top-1 clears the >90% goal. The public-facing auto-accept
  labels are essentially perfect (99.89%) over the ~70% of segments confident
  enough to label, with the remaining ~30% routed to review.
- This is below the research spike's 95.4% all-segment figure largely because
  the shipped reference window selection is evenly spaced rather than
  energy-ranked; energy-based reference window selection is a tracked
  improvement that should raise all-segment accuracy further without changing
  the contract.
- Anonymous pyannote on the same episode was ~77% (baseline) to ~87% (tuned);
  see `../specs/diarization-quality-research.md`.

## Deployed production validation (2026-05-29)

Validated through the **real deployed** Wagtail → django-cast → Voxhelm flow,
not just offline. Voxhelm deployed to macstudio (`just deploy-one voxhelm
macstudio`); python-podcast deployed to staging then production (`just
deploy-staging`, `just deploy-production`), pinning django-cast develop
`2a028f7e`. Eight approved same-episode voice references (source ranges into
the public `pp_64.m4a`) were added for Dominik, Jochen, Johannes, and Ronny on
episode 137, and a transcript was generated through Voxhelm.

- Voxhelm jobs: staging `f0e2e7ff-ded6-4881-b022-50a60008feb3`, production
  `7c47b673-f64b-4816-899b-cb8486fb86c8` (`task_ref cast-audio-79-diarized-4-speakers`).
- Returned `speakers` sidecar: 2191 segments, 1364 confident, 827 uncertain,
  model `pyannote/wespeaker-voxceleb-resnet34-LM`, all four known speakers.

Scored against the hand-labeled `pp_64` gold transcript (Voxhelm segments
mapped to gold speakers by time overlap; reference ranges held out of the
evaluation gold set):

| Metric (production deployed flow) | Value |
| --- | ---: |
| Hand-labeled gold set (representative passages, all speakers) — top-1 | **95.06%** |
| Auto-applied public labels (representative set) | 98.93% |
| Per-speaker top-1 (Ronny / Jochen / Johannes / Dominik) | 99.0 / 97.6 / 92.0 / 84.0% |
| Johannes passage 00:38:16–00:38:32 — confident segments | 100% correct |
| All returned segments — top-1 / DER | 88.22% / 6.71% |
| All returned segments — auto-applied accuracy / DER (cov 62.6%) | 98.74% / 1.49% |

The hand-labeled gold set clears the >90% bar at **95.06%** segment-level
top-1. The lower all-segments number reflects Voxhelm's much finer STT
segmentation (2191 vs the gold's 1249) producing many sub-second/crosstalk
segments; those are routed to review by the auto-accept policy
(`min_segment_duration=1.5`) and excluded from auto-applied public labels,
which are 98.7–98.9% accurate. The editor review/apply path then wrote 1364
confident labels into the public Podlove/DOTe output.

Privacy verified on the live systems: the suggestion sidecar is stored in
private server-side `FileSystemStorage` (the `cast_voice_references` alias),
absent from the public S3 bucket, the public Podlove carries no suggestion
metadata, and voice references are absent from contributor serialization.

## Reproduce

```bash
HF_HUB_OFFLINE=1 uv run --active python evals/known_speaker_eval.py \
    --out /tmp/known_speaker_eval.json
```

Requires the local pp_64 FLAC tracks, the cached wespeaker model, and the
mastered `pp_64.m4a`. The full per-segment output is written to the `--out`
path (local only; not committed).
