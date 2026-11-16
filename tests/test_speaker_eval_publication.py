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
