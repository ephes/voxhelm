# Known-Speaker Diarization Contract

Status: implemented
Date: 2026-05-29

Anonymous pyannote diarization is fallback-only for known-speaker podcasts (see
`diarization-quality-research.md`). This adds a known-speaker voiceprint
postprocessor: transcript segments from the mastered mono audio are classified
directly against contributor reference voiceprints, which the research measured
at ~95% over all segments and ~99%+ over segments long enough for a stable
embedding.

## Request

`POST /v1/jobs` with `job_type=transcribe`. The existing anonymous diarization
contract is unchanged; known-speaker mode is opt-in through
`diarization.strategy`.

```json
{
  "diarization": {
    "enabled": true,
    "num_speakers": 4,
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
      "min_top_similarity": 0.55
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
- Reference `audio` reuses the job-input descriptor shape: `{kind: "url", url}`
  or `{kind: "upload", upload_id}`. URL hosts must be on
  `VOXHELM_ALLOWED_URL_HOSTS`. Execution currently fetches `url` references
  (the production path is source ranges into already-public mastered audio);
  `upload` references are accepted by the contract but not yet fetched at
  execution time.
- `known_speaker` thresholds default to the research recommendation: WeSpeaker
  embeddings, `min_segment_duration=1.5`, `auto_accept_margin=0.15`,
  `min_top_similarity=0.55`.
- `num_speakers`/`min_speakers`/`max_speakers` may still be sent; they drive the
  anonymous pyannote run kept as a fallback/debug signal whose raw label is
  recorded per segment.

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
    "segment_count": 1249,
    "confident_segment_count": 1175,
    "uncertain_segment_count": 74,
    "confident_speaker_distribution": {"Johannes": 390, "Dominik": 191},
    "margin_median": 0.372,
    "raw_diarization_available": true
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

The job `result.metadata.diarization` also carries a `known_speaker_summary`
(the summary block above) and a non-sensitive `known_speakers` list of
`{id, name, reference_count}` — never the private reference URLs/ranges.

## Acceptance policy

A segment is auto-accepted (gets a sidecar `speaker`) only when
`duration >= min_segment_duration` and `top_similarity >= min_top_similarity`
and `margin >= auto_accept_margin`. Otherwise it is `speaker_uncertain=true`
with `speaker=null`, keeping the best-effort candidate list for review. The
sidecar is a suggestion: it never writes the public transcript artifacts. Raw
anonymous pyannote labels are preserved per segment for audit and remapping.

## Ownership

- Voxhelm owns embedding extraction, model-versioned centroids, and
  classification. The embedding model is a Voxhelm implementation detail; when
  it changes, Voxhelm re-extracts centroids from the caller's references.
- django-cast owns private reference storage, consent/review state, and the
  reviewable transcript suggestion state. It sends contributor ids + names +
  private reference clips/ranges, never public profile URLs.

## Implementation

- `transcriptions/known_speaker.py`: config, reference/centroid math,
  classification, accept policy, sidecar artifact, pyannote embedding backend
  (lazy, behind `KnownSpeakerEmbeddingBackendProtocol`).
- `jobs/services.py`: request parsing/validation, reference fetching, job
  orchestration, `speakers` artifact persistence, dedup by full diarization
  payload.
- `jobs/models.py`: `JobArtifact.Kind.TRANSCRIPT_SPEAKERS`
  (migration `0006_alter_jobartifact_kind`).
