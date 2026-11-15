from copy import deepcopy
from pathlib import Path

import pytest

from evals.speaker_naming_eval import duration_bucket, published_names, read_json, score

FIXTURE = Path(__file__).parent / "fixtures/speaker_naming"


def test_score_fixture_arithmetic_and_changes():
    sidecar = read_json(FIXTURE / "sidecar.json")
    truth = read_json(FIXTURE / "truth.json")
    report = score(sidecar, truth, guests=["Guest"])
    assert report["frames"]["total"] == 30
    assert [report["frames"][key] for key in ("right", "wrong", "unnamed")] == [15, 5, 10]
    assert report["lines"]["unnamed_pct"] == 25
    assert report["accept_reasons"]["guest_rule"]["right_pct"] == 100
    assert report["accept_reasons"]["speaker_track"]["right_pct"] == 0
    assert report["duration_buckets"]["1-1.5"]["unnamed_pct"] == 50
    variant = deepcopy(sidecar)
    variant["segments"][2].update(speaker="Alice", speaker_uncertain=False)
    diff = score(variant, truth, guests=["Guest"], baseline=sidecar)
    assert len(diff["changes"]) == 1
    assert diff["changes_precision"]["after_right_pct"] == 100
    assert diff["changes_precision"]["before_right_pct"] == 0


@pytest.mark.parametrize(
    "duration,bucket",
    [(0.49, "<0.5"), (0.5, "0.5-1"), (1, "1-1.5"), (1.5, "1.5-2.5"), (2.5, "2.5-5"), (5, ">=5")],
)
def test_bucket_edges(duration, bucket):
    assert duration_bucket(duration) == bucket


def test_guest_requires_one_label_one_guest_and_no_mixed_segment():
    sidecar = read_json(FIXTURE / "sidecar.json")
    assert published_names(sidecar, ["Guest", "Other"])[1] is None
    sidecar["segments"][1]["speaker_uncertain_reason"] = "mixed_speakers"
    assert published_names(sidecar, ["Guest"])[1] is None
    sidecar["summary"]["raw_diarization_labels"]["Speaker 3"] = {"speaker": None}
    assert published_names(sidecar, ["Guest"])[1] is None


def test_frames_count_overlap_once_and_clear_truth_needs_three_frames():
    sidecar = read_json(FIXTURE / "sidecar.json")
    sidecar["segments"][1]["start"] = 0
    report = score(sidecar, {"fps": 10, "frames": ["Alice", "Alice", None]}, guests=[])
    assert report["frames"]["total"] == 2
    assert report["lines"]["clear_lines"] == 0
