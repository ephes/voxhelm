"""Gold-set evaluation for the known-speaker voiceprint postprocessor.

Runs the *shipped* ``transcriptions.known_speaker`` code path end to end against
real material: contributor reference FLAC tracks build the centroids, the
mastered production audio is classified segment by segment, and predictions are
scored against a hand/track-labeled gold transcript.

This is offline validation tooling, not part of the service. Audio paths are
local-only; pass them as args or rely on the pp_64 defaults. The wespeaker model
is loaded from the local Hugging Face cache (set HF_HUB_OFFLINE=1).

Usage (from the voxhelm venv):

    HF_HUB_OFFLINE=1 uv run --active python evals/known_speaker_eval.py \
        --out /tmp/known_speaker_eval.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from transcriptions.known_speaker import (  # noqa: E402
    KnownSpeakerConfig,
    PyannoteEmbeddingBackend,
    ReferenceAudio,
    decode_mono_16k,
    extract_reference_windows,
    run_known_speaker_postprocess,
)
from transcriptions.service import TranscriptionResult, TranscriptionSegment  # noqa: E402

REAPER = Path.home() / "Documents/REAPER Media/pp_64"
DEFAULT_PRODUCTION_AUDIO = Path("/tmp/voxhelm-diarization-spike/pp_64.m4a")
DEFAULT_GOLD = REAPER / "transcripts/pp_64.json"
DEFAULT_REFERENCES = {
    "Jochen": REAPER / "pp_64-01-Jochen.flac",
    "Johannes": REAPER / "pp_64-02-Johannes.flac",
    "Ronny": REAPER / "pp_64-03-Ronny.flac",
    "Dominik": REAPER / "pp_64-04-Dominik.flac",
}
# The known passage the research flagged: spoken by Johannes, mislabeled by
# anonymous pyannote (~00:38:16-00:38:32).
JOHANNES_PASSAGE = (2295.0, 2312.5)


def load_gold(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())


def build_references(
    reference_paths: dict[str, Path], backend: PyannoteEmbeddingBackend
) -> list[ReferenceAudio]:
    references = []
    for name, path in reference_paths.items():
        print(f"references: {name} <- {path.name}", flush=True)
        samples = decode_mono_16k(path)
        windows = extract_reference_windows(samples, 16000)
        references.append(ReferenceAudio(speaker_id=name.lower(), name=name, windows=windows))
    return references


def confusion_matrix(pairs: list[tuple[str, str]]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {}
    for gold, predicted in pairs:
        matrix.setdefault(gold, Counter())[predicted] += 1
    return {gold: dict(counts) for gold, counts in matrix.items()}


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production-audio", type=Path, default=DEFAULT_PRODUCTION_AUDIO)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--out", type=Path, default=Path("/tmp/known_speaker_eval.json"))
    parser.add_argument("--limit", type=int, default=0, help="evaluate first N segments only")
    args = parser.parse_args()

    config = KnownSpeakerConfig()
    backend = PyannoteEmbeddingBackend(
        model_name=config.embedding_model,
        auth_token=os.getenv("VOXHELM_HUGGINGFACE_TOKEN") or "cached-offline",
        device_name="auto",
    )

    references = build_references(DEFAULT_REFERENCES, backend)

    gold = load_gold(args.gold)
    if args.limit:
        gold = gold[: args.limit]
    print(f"decoding production audio: {args.production_audio}", flush=True)
    job_audio = decode_mono_16k(args.production_audio)

    result = TranscriptionResult(
        text="",
        language="de",
        segments=[
            TranscriptionSegment(id=i, start=float(g["start"]), end=float(g["end"]), text=g["text"])
            for i, g in enumerate(gold)
        ],
    )

    print(f"classifying {len(gold)} segments through run_known_speaker_postprocess", flush=True)
    outcome = run_known_speaker_postprocess(
        result,
        references=references,
        job_audio_samples=job_audio,
        raw_turns=[],
        config=config,
        backend=backend,
    )

    gold_speakers = [g["speaker"] for g in gold]
    # Top-1 prediction = best candidate (the model's guess) regardless of confidence.
    top1 = [seg.speaker_candidates[0].speaker for seg in outcome.segments]
    correct_all = sum(1 for g, p in zip(gold_speakers, top1, strict=True) if g == p)

    # Auto-accept policy: uncertain segments are "needs review", excluded from
    # the public-label accuracy/coverage rather than counted as correct.
    accepted = [
        (g, seg)
        for g, seg in zip(gold_speakers, outcome.segments, strict=True)
        if not seg.speaker_uncertain
    ]
    accepted_correct = sum(1 for g, seg in accepted if seg.speaker == g)

    # Johannes passage check.
    passage = [
        (g["speaker"], seg)
        for g, seg in zip(gold, outcome.segments, strict=True)
        if JOHANNES_PASSAGE[0] <= float(g["start"]) <= JOHANNES_PASSAGE[1]
    ]
    passage_report = [
        {
            "start": seg.start,
            "end": seg.end,
            "gold": gold_speaker,
            "predicted_top1": seg.speaker_candidates[0].speaker,
            "accepted_speaker": seg.speaker,
            "confidence": seg.speaker_confidence,
            "margin": seg.speaker_margin,
            "uncertain": seg.speaker_uncertain,
        }
        for gold_speaker, seg in passage
    ]

    margins = [seg.speaker_margin for seg in outcome.segments]
    report = {
        "embedding_model": config.embedding_model,
        "segment_count": len(gold),
        "all_segments_top1_accuracy": round(correct_all / len(gold), 4),
        "auto_accept_policy": {
            "min_segment_duration": config.min_segment_duration,
            "auto_accept_margin": config.auto_accept_margin,
            "min_top_similarity": config.min_top_similarity,
            "accepted_segments": len(accepted),
            "coverage": round(len(accepted) / len(gold), 4),
            "accepted_accuracy": round(accepted_correct / len(accepted), 4) if accepted else 0.0,
        },
        "margin_median": round(quantile(margins, 0.5), 4),
        "margin_p10": round(quantile(margins, 0.1), 4),
        "confusion_matrix_top1": confusion_matrix(list(zip(gold_speakers, top1, strict=True))),
        "predicted_distribution_top1": dict(Counter(top1)),
        "gold_distribution": dict(Counter(gold_speakers)),
        "johannes_passage": passage_report,
        "johannes_passage_correct": all(
            row["predicted_top1"] == row["gold"]
            for row in passage_report
            if row["gold"] == "Johannes"
        ),
    }
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    skip = {"confusion_matrix_top1", "johannes_passage"}
    summary = {k: v for k, v in report.items() if k not in skip}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
