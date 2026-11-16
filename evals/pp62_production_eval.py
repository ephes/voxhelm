"""Offline pp62 production evaluation with explicit version-2 speech-second metrics.

Gold intervals define the speech denominator (overlapping gold tracks contribute
speaker-seconds independently). Predictions are clipped and unioned per gold
interval. Conflicting simultaneous labels abstain; endpoints are half-open.
Public metrics use only explicitly confident accepted names. Candidate-top1 is a
separate view. No collar, silence false-alarm or full diarization DER is claimed.
The historical whole-gold-segment winner score is retained under an explicit
legacy name; old DER-labelled reports are not comparable to speech seconds.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from evals.speaker_inputs import interval, items, require, text


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def build_gold_index(gold: list[dict]) -> list[tuple[float, float, str]]:
    return [(float(g["start"]), float(g["end"]), g["speaker"]) for g in gold]


def gold_speaker_for(
    seg_start: float, seg_end: float, gold: list[tuple[float, float, str]]
) -> str | None:
    """Gold speaker overlapping a production segment the most (None if no overlap)."""
    best_spk, best_ov = None, 0.0
    for gs, ge, spk in gold:
        if ge < seg_start:
            continue
        if gs > seg_end:
            break
        ov = overlap(seg_start, seg_end, gs, ge)
        if ov > best_ov:
            best_ov, best_spk = ov, spk
    return best_spk


def prediction_label(segment: dict, *, confident_only: bool) -> str | None:
    if confident_only:
        return segment.get("speaker") if segment.get("speaker_uncertain") is False else None
    candidates = segment.get("speaker_candidates") or []
    return candidates[0]["speaker"] if candidates else None


def aligned_intervals(
    gs: float, ge: float, segs: list[dict], *, confident_only: bool
) -> Iterator[tuple[float, str | None, bool]]:
    """Yield duration, unanimous label and union coverage, independent of order."""
    events: dict[float, Counter[str | None]] = {gs: Counter(), ge: Counter()}
    for segment in segs:
        label = prediction_label(segment, confident_only=confident_only)
        if confident_only and label is None:
            continue
        start, end = max(gs, float(segment["start"])), min(ge, float(segment["end"]))
        if start < end:
            events.setdefault(start, Counter())[label] += 1
            events.setdefault(end, Counter())[label] -= 1
    active: Counter[str | None] = Counter()
    previous = gs
    for boundary in sorted(events):
        if boundary > previous:
            labels = {label for label, count in active.items() if count > 0}
            yield (
                boundary - previous,
                next(iter(labels)) if len(labels) == 1 else None,
                bool(labels),
            )
        active.update(events[boundary])
        for label in events[boundary]:
            if active[label] == 0:
                del active[label]
        previous = boundary


def production_label_for(
    gs: float, ge: float, segs: list[dict], *, confident_only: bool
) -> tuple[str | None, float]:
    """Legacy largest summed label vote, with independent interval-union coverage.

    Duplicate predictions retain legacy vote weight only; ties use lexical label
    order. New speech-second accuracy uses unanimous aligned intervals instead.
    """
    by_label: dict[str, float] = defaultdict(float)
    for segment in segs:
        label = prediction_label(segment, confident_only=confident_only)
        if label is not None:
            by_label[label] += overlap(gs, ge, float(segment["start"]), float(segment["end"]))
    by_label = {label: seconds for label, seconds in by_label.items() if seconds > 0}
    winner = min(by_label, key=lambda label: (-by_label[label], label)) if by_label else None
    covered = sum(
        duration
        for duration, _, covered in aligned_intervals(gs, ge, segs, confident_only=confident_only)
        if covered
    )
    return winner, covered


def speech_seconds_for(
    gs: float, ge: float, gold: str, segs: list[dict], *, confident_only: bool
) -> dict[str, float]:
    counts = dict.fromkeys(
        (
            "correct_seconds",
            "wrong_seconds",
            "unnamed_seconds",
            "uncovered_seconds",
            "covered_seconds",
        ),
        0.0,
    )
    for duration, label, covered in aligned_intervals(gs, ge, segs, confident_only=confident_only):
        key = (
            "uncovered_seconds"
            if not covered
            else "unnamed_seconds"
            if label is None
            else "correct_seconds"
            if label == gold
            else "wrong_seconds"
        )
        counts[key] += duration
        if covered:
            counts["covered_seconds"] += duration
    return counts


def pct(n: float, d: float) -> float | None:
    return round(100.0 * n / d, 2) if d else None


def evaluate(gold_raw: list[dict], side: dict[str, Any]) -> dict[str, Any]:
    """Pure report arithmetic, shared by CLI and synthetic regression fixtures."""
    for segment in items(gold_raw, "gold"):
        interval(segment, "gold[]")
        text(segment.get("speaker"), "gold[].speaker")
    segs = items(side.get("segments", []), "production.segments")
    for segment in segs:
        interval(segment, "production.segments[]")
        text(segment.get("speaker"), "production.speaker", nullable=True)
        require(type(segment.get("speaker_uncertain")) is bool, "production.uncertain")
        for candidate in items(
            segment.get("speaker_candidates") or [], "production.candidates", 128
        ):
            require(isinstance(candidate, dict), "production.candidates[]")
            text(candidate.get("speaker"), "production.candidates[].speaker")
    gold = sorted(build_gold_index(gold_raw))
    summary = side.get("summary", {})
    counted = matched = counted_conf = matched_conf = 0
    for segment in segs:
        speaker = gold_speaker_for(float(segment["start"]), float(segment["end"]), gold)
        if speaker is None:
            continue
        counted += 1
        matched += prediction_label(segment, confident_only=False) == speaker
        public = prediction_label(segment, confident_only=True)
        if public is not None:
            counted_conf += 1
            matched_conf += public == speaker
    total = legacy_top_correct = legacy_public_correct = legacy_public_total = 0.0
    per_speaker: dict[str, Counter[str]] = defaultdict(Counter)
    seconds: dict[str, Counter[str]] = {"top1": Counter(), "public": Counter()}
    for gs, ge, speaker in gold:
        duration = ge - gs
        if duration <= 0:
            continue
        total += duration
        label, _ = production_label_for(gs, ge, segs, confident_only=False)
        legacy_top_correct += duration if label == speaker else 0
        public, covered = production_label_for(gs, ge, segs, confident_only=True)
        if covered:
            legacy_public_total += duration
            legacy_public_correct += duration if public == speaker else 0
        for view, confident in (("top1", False), ("public", True)):
            interval_counts = speech_seconds_for(gs, ge, speaker, segs, confident_only=confident)
            seconds[view].update(interval_counts)
            if not confident:
                per_speaker[speaker].update(interval_counts)
    views = {}
    for view, counts in seconds.items():
        views[view] = {
            **{
                key: counts[key]
                for key in (
                    "correct_seconds",
                    "wrong_seconds",
                    "unnamed_seconds",
                    "uncovered_seconds",
                    "covered_seconds",
                )
            },
            "gold_seconds": total,
            "accuracy_pct": pct(counts["correct_seconds"], total),
            "coverage_pct": pct(counts["covered_seconds"], total),
            "covered_accuracy_pct": pct(counts["correct_seconds"], counts["covered_seconds"]),
        }
    return {
        "metric_version": 2,
        "gold_segments": len(gold),
        "production_segments": len(segs),
        "production_summary": {
            key: summary.get(key)
            for key in (
                "strategy",
                "embedding_model",
                "known_speakers",
                "segment_count",
                "confident_segment_count",
                "uncertain_segment_count",
                "confident_speaker_distribution",
                "auto_accept_margin",
                "min_segment_duration",
                "min_top_similarity",
            )
        },
        "speech_seconds": views,
        "legacy_duration_weighted_segment_label": {
            "top1_accuracy_pct": pct(legacy_top_correct, total),
            "public_accuracy_pct": pct(legacy_public_correct, legacy_public_total),
            "definition": "Whole gold duration credited to largest summed-overlap label; not DER",
        },
        "time_weighted_coverage_pct": views["top1"]["coverage_pct"],
        "public_confident_time_weighted_coverage_pct": views["public"]["coverage_pct"],
        "public_confident_time_weighted_accuracy_pct": views["public"]["covered_accuracy_pct"],
        "segment_count_top1_accuracy_pct": pct(matched, counted),
        "segment_count_evaluated": counted,
        "public_confident_segment_accuracy_pct": pct(matched_conf, counted_conf),
        "public_confident_segments_evaluated": counted_conf,
        "per_speaker_time_weighted_accuracy_pct": {
            speaker: pct(
                counts["correct_seconds"],
                sum(
                    counts[key]
                    for key in (
                        "correct_seconds",
                        "wrong_seconds",
                        "unnamed_seconds",
                        "uncovered_seconds",
                    )
                ),
            )
            for speaker, counts in sorted(per_speaker.items())
        },
        "per_speaker_gold_seconds": {
            speaker: sum(
                counts[key]
                for key in (
                    "correct_seconds",
                    "wrong_seconds",
                    "unnamed_seconds",
                    "uncovered_seconds",
                )
            )
            for speaker, counts in sorted(per_speaker.items())
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path.home() / "Documents/REAPER Media/pp_62/transcripts/pp_62.json"
    ap.add_argument("--gold", type=Path, default=here)
    ap.add_argument("--sidecar", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("/tmp/pp62_production_eval.json"))
    args = ap.parse_args()

    report = evaluate(load_json(args.gold), load_json(args.sidecar))
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
