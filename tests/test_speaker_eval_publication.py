"""Frozen bundle and complete suite publication, using synthetic backends."""

import numpy as np
import pytest

from evals import speaker_naming_eval as ev
from transcriptions.service import TranscriptionResult, TranscriptionSegment


@pytest.fixture
def capture(tmp_path, monkeypatch):
    from transcriptions.diarization import PyannoteDiarizationBackend, SpeakerTurn

    (tmp_path / "audio.wav").write_bytes(b"synthetic audio")
    (tmp_path / "model.bin").write_bytes(b"synthetic model")
    manifest = tmp_path / "manifest.json"
    ev.write_json(
        manifest,
        {
            "audio": "audio.wav",
            "references": [{"id": "a", "name": "Alice", "start": 0, "end": 1}],
            "generation_files": {"asr": ["model.bin"], "diarization": ["model.bin"]},
        },
    )
    monkeypatch.setattr(
        "transcriptions.service.transcribe_audio",
        lambda *a, **k: TranscriptionResult("hi", "en", [TranscriptionSegment(0, 0, 1, "hi")]),
    )
    monkeypatch.setattr(
        PyannoteDiarizationBackend,
        "diarize_annotations",
        lambda *a, **k: {"regular": [SpeakerTurn(0, 1, "a")], "exclusive": []},
    )
    monkeypatch.setattr(
        "transcriptions.known_speaker.decode_mono_16k",
        lambda *a, **k: np.ones(16000, dtype=np.float32),
    )
    return manifest, tmp_path / "bundle"


@pytest.mark.parametrize("stage", ["asr", "reference", "manifest"])
def test_bundle_failure_never_publishes_and_retry_succeeds(capture, monkeypatch, stage):
    manifest, destination = capture
    with monkeypatch.context() as patch:

        def fail(*args, **kwargs):
            raise RuntimeError("synthetic failure")

        if stage == "asr":
            patch.setattr("transcriptions.service.transcribe_audio", fail)
        elif stage == "reference":

            def decode(path, **kwargs):
                if kwargs:
                    fail()
                return np.ones(16000, dtype=np.float32)

            patch.setattr("transcriptions.known_speaker.decode_mono_16k", decode)
        else:
            patch.setattr(ev, "write_json", fail)
        with pytest.raises(RuntimeError):
            ev.create_bundle(manifest, destination)
    assert not destination.exists()
    ev.create_bundle(manifest, destination)
    assert (destination / "bundle.json").is_file()
    assert not list(destination.parent.glob(".bundle.stage-*"))


def test_existing_bundle_is_never_overwritten(capture):
    manifest, destination = capture
    destination.mkdir()
    (destination / "sentinel").write_bytes(b"old")
    with pytest.raises((ValueError, FileExistsError)):
        ev.create_bundle(manifest, destination)
    assert (destination / "sentinel").read_bytes() == b"old"


@pytest.mark.parametrize("change", ["manifest", "audio", "model", "settings"])
def test_frozen_identity_rejects_changed_sources_before_replay(
    capture, monkeypatch, settings, change
):
    manifest, destination = capture
    ev.create_bundle(manifest, destination)
    if change == "manifest":
        value = ev.read_json(manifest)
        value["unreferenced_speakers"] = ["Guest"]
        ev.write_json(manifest, value)
    elif change == "settings":
        settings.VOXHELM_PYANNOTE_MODEL = "new-model"
    else:
        (manifest.parent / ("audio.wav" if change == "audio" else "model.bin")).write_bytes(b"new")
    with pytest.raises(ValueError, match="identity"):
        ev.verify_bundle_identity(manifest, destination)


def test_unchanged_identity_and_postprocess_variants_reuse(capture):
    manifest, destination = capture
    ev.create_bundle(manifest, destination)
    ev.verify_bundle_identity(manifest, destination)
    # Postprocess configuration is intentionally absent from the frozen identity.
    assert ev.read_json(destination / "bundle.json")["version"] == 2


@pytest.fixture
def suite_root(capture, monkeypatch):
    manifest, _ = capture
    root = manifest.parent
    (root / "manifests").mkdir()
    (root / "truth").mkdir()
    for episode in ("pp59", "pp77", "pp78", "pp80", "dc209", "dc210"):
        value = ev.read_json(manifest)
        value["audio"] = "../audio.wav"
        value["generation_files"] = {key: ["../model.bin"] for key in ("asr", "diarization")}
        path = root / "manifests" / f"{episode}.json"
        ev.write_json(path, value)
        ev.create_bundle(path, root / "bundles" / episode)
        if episode.startswith("pp"):
            ev.write_json(root / "truth" / f"{episode}.json", {"fps": 1, "frames": ["Alice"]})

    def replay(bundle, config, **kwargs):
        if kwargs.get("diagnostics_path"):
            ev.write_json(kwargs["diagnostics_path"], {})
        return {
            "summary": {},
            "segments": [{"start": 0, "end": 1, "speaker": "Alice", "speaker_uncertain": False}],
        }, 1

    monkeypatch.setattr(ev, "run_bundle", replay)
    return root, replay


def contents(path):
    return {p.name: p.read_bytes() for p in path.iterdir()}


@pytest.mark.parametrize("failure", ["episode-five", "score-write"])
def test_failed_suite_preserves_previous_complete_generation(suite_root, monkeypatch, failure):
    root, replay = suite_root
    ev.run_suite(root, {}, "baseline")
    output = root / "results" / "baseline"
    before = contents(output)
    if failure == "episode-five":

        def run(bundle, config, **kwargs):
            if bundle.name == "dc209":
                raise RuntimeError("late episode")
            return replay(bundle, config, **kwargs)

        monkeypatch.setattr(ev, "run_bundle", run)
    else:
        original = ev.write_json

        def write(path, value):
            if path.name == "dc209.score.json":
                raise RuntimeError("late write")
            original(path, value)

        monkeypatch.setattr(ev, "write_json", write)
    with pytest.raises(RuntimeError):
        ev.run_suite(root, {}, "baseline")
    assert contents(output) == before
    stages = list((root / "results").glob(".baseline.stage-*"))
    assert len(stages) == 1
    assert not (stages[0] / "complete.json").exists()


def test_same_tag_publishers_are_exclusive_and_completion_lists_exact_members(
    suite_root, monkeypatch
):
    root, replay = suite_root
    attempted = False

    def run(bundle, config, **kwargs):
        nonlocal attempted
        if not attempted:
            attempted = True
            with pytest.raises(ValueError, match="publisher"):
                ev.run_suite(root, {}, "baseline")
        return replay(bundle, config, **kwargs)

    monkeypatch.setattr(ev, "run_bundle", run)
    ev.run_suite(root, {}, "baseline")
    output = root / "results" / "baseline"
    complete = ev.read_json(output / "complete.json")
    assert set(complete["members"]) == set(contents(output)) - {"complete.json"}
    assert len(complete["episodes"]) == 6
    assert complete["config"]["embedding_model"]


@pytest.mark.parametrize(
    "tag", ["../manifests", "/tmp/escaped", "a/b", "a\\b", ".", "..", "a" * 65]
)
def test_invalid_suite_tag_refuses_before_any_work(tmp_path, monkeypatch, tag):
    monkeypatch.setattr(ev, "create_bundle", lambda *a: pytest.fail("backend"))
    with pytest.raises(ValueError, match="tag"):
        ev.run_suite(tmp_path, {}, tag)
    assert not (tmp_path / "results").exists()
