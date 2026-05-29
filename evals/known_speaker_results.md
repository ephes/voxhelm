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

## Cross-episode production validation — pp_62 "Bytes und Strings" (2026-05-29)

Second, stronger validation: the **deployed production** known-speaker flow run
on a *different* episode than the references were taken from. This tests
cross-recording voiceprint generalization (different day, mics, levels) and
removes any same-episode-reference concern.

- Target: episode 135 "Bytes und Strings" (audio 77 = `pp_62`, German mono,
  1:50:33, three speakers Jochen/Dominik/Johannes),
  <https://python-podcast.de/show/bytes-und-strings/>.
- References: the **approved pp_64 source-range references** for Jochen,
  Dominik, Johannes (source ranges into audio 79 = `pp_64`). Because the
  references live entirely in a *different* episode, the whole pp_62 gold audio
  is held out — **zero evaluation leakage**.
- Real deployed flow: Wagtail episode → `enqueue_audio_transcript_generation`
  (episode 135, 3 contributors) → Voxhelm. No code/config changes were needed;
  the shipped `pyannote_known_speaker` strategy was used as deployed.
- Voxhelm job `8b833cc9-fa91-4dce-b009-e8c3678107e1`
  (`task_ref cast-audio-77-diarized-3-speakers`, `num_speakers=3`), generation
  #4 succeeded (~9 min). Returned `speakers` sidecar: 3701 segments, 2019
  confident, 1682 uncertain, model `pyannote/wespeaker-voxceleb-resnet34-LM`,
  all three known speakers.
- Deployed commits: django-cast `d6ce2c79` (pinned in prod `uv.lock`), voxhelm
  `6498b5e` (`known-speaker-diarization`), python-podcast `7fe1d37`.

Gold = the authoritative multitrack-derived transcript
(`~/Documents/REAPER Media/pp_62/transcripts/pp_62.json`, 1817 segments labelled
by isolated per-speaker track), aligned to the same mastered audio the
production job transcribed (gold span 0.006–6632.1s vs mastered 6633.9s).

| Metric (production deployed flow, full episode) | Value |
| --- | ---: |
| **Time-weighted top-1 accuracy (primary, = 1−DER)** | **96.76%** |
| Time-weighted DER | 3.24% |
| Gold-speech coverage (overlapped by a production segment) | 99.49% |
| Auto-applied (public) labels — time-weighted accuracy | 97.74% |
| Auto-applied (public) labels — segment-level accuracy | 98.56% |
| Auto-applied (public) labels — coverage | 88.84% |
| Per-speaker top-1 (Jochen / Johannes / Dominik) | 97.74 / 97.04 / 93.09% |
| All returned segments — segment-count top-1 | 88.95% |

Metric definition (per the goal): gold = authoritative multitrack speaker per
gold segment; non-speech/music is excluded by construction (no gold segment).
**Primary** is gold-centric and full-episode: for every gold segment the
production *top-1 candidate* covering the largest share of that segment is
compared to the gold speaker, weighted by gold-segment duration
(numerator = correctly-attributed gold speech-seconds; denominator = **total**
gold speech-seconds — every gold segment). Gold speech not overlapped by any
production segment cannot match and so counts as a miss rather than being
dropped from the denominator; the overlapped fraction (99.49%) is reported
separately as coverage.
Segment-count top-1 is production-centric (each production segment → its
max-overlap gold speaker). The auto-applied figures count only confident
(non-uncertain) segments — the ones that become public after review.

The full-episode **96.76% time-weighted top-1 (DER 3.24%) clears the >90% goal
on the live production result**, with cross-episode references and no leakage.
As on pp_64, the lower raw segment-count number (88.95%) is concentrated in
sub-second/crosstalk segments that the auto-accept policy
(`min_segment_duration=1.5`) routes to review and never applies publicly; the
public-facing labels are 98.56% accurate.

### Applied to public output

The suggestions were then approved and applied to the public transcript via
`Transcript.apply_known_speaker_suggestions(smooth=True)` (7402 segments
labelled across Podlove + DOTe, full coverage, no blank segments). The public
Podlove now carries per-segment `speaker`/`voice` and DOTe carries
`speakerDesignation` (CloudFront serves the labelled
`cast_transcript/audio-77.podlove.json`), so the Podlove web-player transcript
shows Jochen/Dominik/Johannes.

Scoring the **visible applied labels** (carry-forward smoothing fills the
review-gated uncertain gaps) against gold: **91.18% time-weighted top-1 /
DER 8.82%** at 99.49% coverage — still clears >90%. Smoothing is slightly below
the 96.76% raw top-1 because carry-forward attributes uncertain interjections to
the surrounding speaker; the underlying per-segment suggestions remain available
in the private sidecar for finer review.

Privacy verified on the live system. The `speakers` sidecar is stored in the
private server-side `FileSystemStorage` (`/home/python-podcast/site/private_media/...`)
and is not reachable on the public CDN (CloudFront 403). Before the editor apply
step the public output was fully unlabeled (all 3701 Podlove entries had empty
`speaker`/`voice`, all 3701 DOTe lines empty `speakerDesignation`, no WebVTT
`<v …>` tags), confirming that generating suggestions does not by itself expose
identities. After the editor apply step the public Podlove/DOTe carry only the
approved speaker *labels* (names) — the raw per-segment candidates, confidence,
margin and uncertainty flags stay solely in the private sidecar, which is never
served publicly. WebVTT is still left unlabeled (tracked follow-up). No serializer,
feed, API, or repository-serialization path references `voice_references` or the
transcript `speakers` field.

**Hardening note (latent, non-leaking):** `Transcript.speakers.url` currently
renders an S3-style URL because the private `cast_voice_references`
`FileSystemStorage` inherits the public `MEDIA_URL` as its `base_url`. The file
is never written to S3 (FileSystemStorage on local disk; confirmed CDN 403), so
this is not an active exposure, but the storage should be configured with a
non-public `base_url` (or no public URL) so `.url` cannot point at a public host
if a future code path calls it.

## Reproduce

pp_64 offline harness:

```bash
HF_HUB_OFFLINE=1 uv run --active python evals/known_speaker_eval.py \
    --out /tmp/known_speaker_eval.json
```

Requires the local pp_64 FLAC tracks, the cached wespeaker model, and the
mastered `pp_64.m4a`. The full per-segment output is written to the `--out`
path (local only; not committed).

pp_62 production scorer (scores the deployed `speakers` sidecar against the
multitrack gold):

```bash
# Download the private sidecar from prod (Transcript.speakers for audio 77),
# then score it. The sidecar is private suggestion data and is NOT committed.
python3 evals/pp62_production_eval.py --sidecar /tmp/pp_62.speakers.json \
    --out /tmp/pp62_production_eval.json
```

Requires `~/Documents/REAPER Media/pp_62/transcripts/pp_62.json` (gold) and the
downloaded production sidecar.
