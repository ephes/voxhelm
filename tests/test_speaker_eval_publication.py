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
                if "start" in kwargs:
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


def test_long_job_audio_refuses_before_asr(capture, monkeypatch):
    from evals.speaker_inputs import MAX_AUDIO_SECONDS
    from transcriptions.diarization import PYANNOTE_MAX_FRAMES
    from transcriptions.known_speaker import KnownSpeakerError

    manifest, destination = capture
    assert MAX_AUDIO_SECONDS * 16000 == PYANNOTE_MAX_FRAMES
    allowances = []

    def decode(path, **kwargs):
        allowances.append(kwargs.get("max_frames"))
        raise KnownSpeakerError("over the decoded frame limit")

    monkeypatch.setattr("transcriptions.known_speaker.decode_mono_16k", decode)
    monkeypatch.setattr(
        "transcriptions.service.transcribe_audio", lambda *a, **k: pytest.fail("ASR started")
    )
    with pytest.raises(KnownSpeakerError):
        ev.create_bundle(manifest, destination)
    assert allowances == [PYANNOTE_MAX_FRAMES]
    assert not destination.exists()


def test_capture_code_identity_excludes_postprocess_and_covers_capture_imports(capture):
    import ast
    from pathlib import Path

    from evals.speaker_generations import CAPTURE_CODE

    manifest, destination = capture
    ev.create_bundle(manifest, destination)
    identity = ev.read_json(destination / "bundle.json")["identity"]
    # Old/new postprocessing code must be able to replay the same frozen bundles.
    assert "known_speaker.py" not in identity["code"]
    assert set(identity["code"]) == set(CAPTURE_CODE)
    package = Path(ev.__file__).resolve().parent.parent / "transcriptions"
    pending, seen = ["service.py", "diarization.py"], set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        for node in ast.walk(ast.parse((package / name).read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = "transcriptions" if node.level == 1 else node.module or ""
                if node.level == 1 and node.module:
                    base = "transcriptions." + node.module
                if base == "transcriptions":
                    # `from transcriptions import formats` imports submodules by name.
                    modules = [f"transcriptions.{alias.name}" for alias in node.names]
                else:
                    modules = [base]
            for module in modules:
                if module.startswith("transcriptions."):
                    pending.append(module.removeprefix("transcriptions.").split(".")[0] + ".py")
    assert seen <= set(CAPTURE_CODE)


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


def test_frozen_sample_replacement_refuses_before_embedding(capture, monkeypatch):
    manifest, destination = capture
    ev.create_bundle(manifest, destination)
    np.save(destination / "audio.npy", np.zeros(16000, dtype=np.float32))
    monkeypatch.setattr(
        "transcriptions.known_speaker.get_known_speaker_backend",
        lambda *a: pytest.fail("changed frozen samples reached embedding"),
    )
    with pytest.raises(ValueError, match="bundle"):
        ev.run_bundle(destination, {})


@pytest.mark.parametrize("change", ["truth", "bundle"])
def test_variant_reuses_exact_baseline_inputs_before_replay(suite_root, monkeypatch, change):
    root, _ = suite_root
    ev.run_suite(root, {}, "baseline")
    if change == "truth":
        ev.write_json(root / "truth/pp59.json", {"fps": 1, "frames": ["Bob"]})
    else:
        path = root / "bundles/pp59/bundle.json"
        value = ev.read_json(path)
        value["result"]["text"] = "changed frozen ASR"
        ev.write_json(path, value)
    monkeypatch.setattr(ev, "run_bundle", lambda *a, **k: pytest.fail("stale baseline replayed"))
    with pytest.raises(ValueError, match="baseline"):
        ev.run_suite(root, {}, "variant")


def test_sealed_manifest_must_fit_control_limit_before_publication(capture, monkeypatch):
    import json

    from evals import speaker_inputs

    manifest, destination = capture
    ev.create_bundle(manifest, destination)
    unsealed = ev.read_json(destination / "bundle.json")
    del unsealed["sample_sha256"]
    size = len((json.dumps(unsealed, indent=2, ensure_ascii=False) + "\n").encode())
    monkeypatch.setattr(speaker_inputs, "JSON_MAX_BYTES", size)
    other = destination.with_name("other-bundle")
    with pytest.raises(ValueError, match="control"):
        ev.create_bundle(manifest, other)
    assert not other.exists()
