from itertools import permutations

import pytest

from evals import pp62_production_eval as ev


def segment(start, end, candidate="Alice", speaker="Alice", uncertain=False):
    return {
        "start": start,
        "end": end,
        "speaker": speaker,
        "speaker_uncertain": uncertain,
        "speaker_candidates": [{"speaker": candidate}] if candidate else [],
    }


def test_speech_seconds_do_not_credit_an_entire_gold_segment():
    report = ev.evaluate(
        [{"start": 0, "end": 10, "speaker": "Alice"}], {"segments": [segment(0, 1)]}
    )
    seconds = report["speech_seconds"]["top1"]
    assert seconds["correct_seconds"] == 1
    assert seconds["uncovered_seconds"] == 9
    assert seconds["accuracy_pct"] == 10
    assert report["legacy_duration_weighted_segment_label"]["top1_accuracy_pct"] == 100
    assert "time_weighted_DER_pct" not in report


def test_exact_duration_weights_for_right_wrong_and_uncovered():
    seconds = ev.speech_seconds_for(
        0, 10, "Alice", [segment(0, 2), segment(2, 5, "Bob")], confident_only=False
    )
    assert seconds == {
        "correct_seconds": 2,
        "wrong_seconds": 3,
        "unnamed_seconds": 0,
        "uncovered_seconds": 5,
        "covered_seconds": 5,
    }


def test_public_metrics_use_accepted_speaker_and_exclude_abstentions():
    segs = [
        segment(0, 2, candidate="Alice", speaker="Bob"),
        segment(2, 3, speaker=None),
        segment(3, 4, uncertain=True),
    ]
    assert ev.production_label_for(0, 2, segs, confident_only=True) == ("Bob", 2)
    report = ev.evaluate([{"start": 0, "end": 4, "speaker": "Bob"}], {"segments": segs})
    assert report["speech_seconds"]["public"]["correct_seconds"] == 2
    assert report["speech_seconds"]["public"]["covered_seconds"] == 2
    assert report["speech_seconds"]["top1"]["wrong_seconds"] == 4
    assert report["public_confident_segment_accuracy_pct"] == 100
    assert report["segment_count_top1_accuracy_pct"] == 0


def test_duplicate_predictions_cover_only_the_interval_union():
    segs = [segment(0, 1)] * 10
    assert ev.production_label_for(0, 10, segs, confident_only=False) == ("Alice", 1)
    report = ev.evaluate([{"start": 0, "end": 10, "speaker": "Alice"}], {"segments": segs})
    assert report["time_weighted_coverage_pct"] == 10
    assert report["public_confident_time_weighted_coverage_pct"] == 10


def test_overlap_conflicts_abstain_and_coverage_is_order_independent():
    segs = [segment(0, 2), segment(1, 3, "Bob"), segment(3, 4)]
    views = [
        ev.speech_seconds_for(0, 5, "Alice", list(p), confident_only=False)
        for p in permutations(segs)
    ]
    assert all(v == views[0] for v in views)
    assert views[0] == {
        "correct_seconds": 2,
        "wrong_seconds": 1,
        "unnamed_seconds": 1,
        "uncovered_seconds": 1,
        "covered_seconds": 4,
    }
    assert all(
        ev.production_label_for(0, 5, list(p), confident_only=False)[1] == 4
        for p in permutations(segs)
    )


@pytest.mark.parametrize("gold,segs", [([], []), ([{"start": 0, "end": 1, "speaker": "A"}], [])])
def test_empty_predictions_have_explicit_stable_metrics(gold, segs):
    report = ev.evaluate(gold, {"segments": segs})
    assert report["speech_seconds"]["top1"]["correct_seconds"] == 0
