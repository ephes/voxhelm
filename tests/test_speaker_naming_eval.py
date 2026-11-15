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


def test_bundle_uses_production_paths_and_replays_deterministically(tmp_path, monkeypatch):
    from dataclasses import asdict

    import numpy

    from evals.speaker_naming_eval import create_bundle, run_bundle, write_json
    from transcriptions.diarization import PyannoteDiarizationBackend, SpeakerTurn
    from transcriptions.known_speaker import KnownSpeakerConfig
    from transcriptions.service import TranscriptionResult, TranscriptionSegment, TranscriptionWord

    result = TranscriptionResult(
        text="Hello",
        language="en",
        segments=[
            TranscriptionSegment(
                id=0, start=0, end=3, text="Hello", words=(TranscriptionWord(0, 3, "Hello"),)
            )
        ],
    )
    captured = []

    def transcribe(path, params):
        assert path == tmp_path / "episode.wav"
        assert params.word_timestamps
        return result

    def decode(path, **kwargs):
        captured.append(kwargs)
        return numpy.ones(48000, dtype="float32")

    monkeypatch.setattr("transcriptions.service.transcribe_audio", transcribe)
    monkeypatch.setattr("transcriptions.known_speaker.decode_mono_16k", decode)
    monkeypatch.setattr(
        PyannoteDiarizationBackend,
        "diarize_annotations",
        lambda *args: {
            "regular": [SpeakerTurn(0, 3, "raw")],
            "exclusive": [SpeakerTurn(0, 2.9, "raw")],
        },
    )

    class Backend:
        embedding_version = "fixture"

        def embed(self, samples, sample_rate):
            return [1.0, 0.0]

    monkeypatch.setattr(
        "transcriptions.known_speaker.get_known_speaker_backend", lambda _: Backend()
    )
    manifest = tmp_path / "manifest.json"
    write_json(
        manifest,
        {
            "audio": "episode.wav",
            "language": "en",
            "references": [{"id": "alice", "name": "Alice", "start": 0, "end": 3}],
        },
    )
    bundle = tmp_path / "bundle"
    create_bundle(manifest, bundle)
    frozen = read_json(bundle / "bundle.json")
    assert frozen["result"] == asdict(result) | {
        "segments": [asdict(result.segments[0]) | {"words": [asdict(result.segments[0].words[0])]}]
    }
    assert frozen["turns"]["exclusive"][0]["end"] == 2.9
    assert captured[1]["start"] == 0
    assert captured[1]["end"] == 3
    config = asdict(KnownSpeakerConfig(resegment_hop=0))
    first, _ = run_bundle(bundle, config)
    second, _ = run_bundle(bundle, config)
    assert first == second
    assert first["segments"][0]["speaker"] == "Alice"
    assert first["segments"][0]["text"] == "Hello"


def test_capture_annotations_keeps_regular_choice(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from transcriptions.diarization import PyannoteDiarizationBackend

    class Annotation:
        def __init__(self, end):
            self.end = end

        def itertracks(self, **kwargs):
            return [(SimpleNamespace(start=0, end=self.end), 0, "raw")]

    backend = PyannoteDiarizationBackend(model_name="fixture", auth_token="fixture")
    calls = []

    def pipeline(*args, **kwargs):
        calls.append(1)
        return SimpleNamespace(
            speaker_diarization=Annotation(3), exclusive_speaker_diarization=Annotation(2)
        )

    monkeypatch.setattr(backend, "_load_pipeline", lambda: pipeline)
    monkeypatch.setattr("transcriptions.diarization.load_audio_for_pyannote", lambda _: {})
    annotations = backend.diarize_annotations(tmp_path / "audio.wav")
    assert len(calls) == 1
    assert annotations["regular"][0].end == 3
    assert annotations["exclusive"][0].end == 2
    assert backend.diarize(tmp_path / "audio.wav") == annotations["regular"]
