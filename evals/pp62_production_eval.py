"""Score the *deployed production* known-speaker result for pp_62
("Bytes und Strings") against the multitrack-derived gold transcript.

Inputs are both local files:

* ``--gold``    REAPER multitrack gold transcript (``transcripts/pp_62.json``),
                each segment carrying an authoritative per-track ``speaker``.
* ``--sidecar`` the production ``Transcript.speakers`` sidecar downloaded from
                the live site (the Voxhelm ``speakers`` artifact: per-segment
                top candidate, full candidate list, confidence, margin and an
                uncertainty flag).

References used by the production run are source ranges into *pp_64* (a
different episode), so the pp_62 gold audio is fully held out — no evaluation
leakage.

Metric definitions (documented for the goal):

* Gold = authoritative multitrack speaker per gold segment.
* A production segment is mapped to a gold speaker by maximum time overlap.
* **Time-weighted top-1 accuracy (primary)** is gold-centric: for every gold
  segment, the production *top-1 candidate* covering the largest share of that
  segment is compared to the gold speaker, weighted by gold-segment duration.
  This equals ``1 - DER`` (single-speaker approximation; no overlap/collar).
  Numerator = gold speech-seconds whose production top-1 label matches gold;
  denominator = total gold speech-seconds (every gold segment). Non-speech /
  music (no gold segment) is excluded by construction. Gold speech with no
  overlapping production segment cannot match, so it is counted as a miss
  (lowers accuracy); the unmatched fraction is reported separately as
  ``time_weighted_coverage_pct`` rather than removed from the denominator.
* **Segment-count top-1 accuracy** is production-centric: each production
  segment mapped to its max-overlap gold speaker, unweighted.
* **Auto-applied (public) accuracy / coverage** counts only confident
  (non-uncertain) production segments — the ones that become public labels.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


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


def production_label_for(
    gs: float, ge: float, segs: list[dict], *, confident_only: bool
) -> tuple[str | None, float]:
    """Production top-1 label covering the most of a gold segment.

    Returns (label, covered_overlap_seconds). ``confident_only`` restricts to
    non-uncertain (public) segments.
    """
    by_label: dict[str, float] = defaultdict(float)
    covered = 0.0
    for s in segs:
        if confident_only and s.get("speaker_uncertain"):
            continue
        ov = overlap(gs, ge, float(s["start"]), float(s["end"]))
        if ov <= 0:
            continue
        cands = s.get("speaker_candidates") or []
        top = cands[0]["speaker"] if cands else (s.get("speaker") or None)
        if top is None:
            continue
        by_label[top] += ov
        covered += ov
    if not by_label:
        return None, covered
    return max(by_label.items(), key=lambda kv: kv[1])[0], covered


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path.home() / "Documents/REAPER Media/pp_62/transcripts/pp_62.json"
    ap.add_argument("--gold", type=Path, default=here)
    ap.add_argument("--sidecar", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("/tmp/pp62_production_eval.json"))
    args = ap.parse_args()

    gold_raw = load_json(args.gold)
    gold = build_gold_index(gold_raw)
    side = load_json(args.sidecar)
    segs = side.get("segments", [])
    summary = side.get("summary", {})

    # ---- Segment-count top-1 (production-centric) ----
    counted = matched = 0
    counted_conf = matched_conf = 0
    for s in segs:
        gspk = gold_speaker_for(float(s["start"]), float(s["end"]), gold)
        if gspk is None:
            continue
        cands = s.get("speaker_candidates") or []
        top = cands[0]["speaker"] if cands else None
        counted += 1
        if top == gspk:
            matched += 1
        if not s.get("speaker_uncertain"):
            counted_conf += 1
            if (s.get("speaker") or None) == gspk:
                matched_conf += 1

    # ---- Time-weighted top-1 (gold-centric, primary = 1-DER) ----
    tw_total = tw_correct = tw_covered = 0.0
    per_spk_total: dict[str, float] = defaultdict(float)
    per_spk_correct: dict[str, float] = defaultdict(float)
    # confident/public time-weighted
    cw_total = cw_correct = cw_covered = 0.0
    for gs, ge, spk in gold:
        dur = ge - gs
        if dur <= 0:
            continue
        tw_total += dur
        per_spk_total[spk] += dur
        label, covered = production_label_for(gs, ge, segs, confident_only=False)
        tw_covered += min(covered, dur)
        if label == spk:
            tw_correct += dur
            per_spk_correct[spk] += dur
        # public (confident) view
        clabel, ccovered = production_label_for(gs, ge, segs, confident_only=True)
        if ccovered > 0:
            cw_total += dur
            cw_covered += min(ccovered, dur)
            if clabel == spk:
                cw_correct += dur

    def pct(n: float, d: float) -> float:
        return round(100.0 * n / d, 2) if d else 0.0

    report = {
        "gold_segments": len(gold),
        "production_segments": len(segs),
        "production_summary": {
            "strategy": summary.get("strategy"),
            "embedding_model": summary.get("embedding_model"),
            "known_speakers": summary.get("known_speakers"),
            "segment_count": summary.get("segment_count"),
            "confident_segment_count": summary.get("confident_segment_count"),
            "uncertain_segment_count": summary.get("uncertain_segment_count"),
            "confident_speaker_distribution": summary.get("confident_speaker_distribution"),
            "auto_accept_margin": summary.get("auto_accept_margin"),
            "min_segment_duration": summary.get("min_segment_duration"),
            "min_top_similarity": summary.get("min_top_similarity"),
        },
        "PRIMARY_time_weighted_top1_accuracy_pct": pct(tw_correct, tw_total),
        "time_weighted_DER_pct": pct(tw_total - tw_correct, tw_total),
        "time_weighted_coverage_pct": pct(tw_covered, tw_total),
        "segment_count_top1_accuracy_pct": pct(matched, counted),
        "segment_count_evaluated": counted,
        "public_confident_time_weighted_accuracy_pct": pct(cw_correct, cw_total),
        "public_confident_time_weighted_coverage_pct": pct(cw_total, tw_total),
        "public_confident_segment_accuracy_pct": pct(matched_conf, counted_conf),
        "public_confident_segments_evaluated": counted_conf,
        "per_speaker_time_weighted_accuracy_pct": {
            spk: pct(per_spk_correct[spk], per_spk_total[spk]) for spk in sorted(per_spk_total)
        },
        "per_speaker_gold_seconds": {
            spk: round(per_spk_total[spk], 1) for spk in sorted(per_spk_total)
        },
    }
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
