"""Synthetic offline evaluation admission regressions; no models or audio files."""

import hashlib
import os
import subprocess
import sys

import numpy as np
import pytest

from evals import speaker_naming_eval as ev


def tiny_bundle(path):
    path.mkdir()
    np.save(path / "audio.npy", np.ones(16, dtype=np.float32))
    value = {
        "version": 1,
        "audio": "audio.npy",
        "references": [],
        "manifest": {"audio": "source.wav", "references": []},
        "result": {"text": "hi", "language": "en", "segments": []},
        "turns": {"regular": [], "exclusive": []},
    }
    ev.write_json(path / "bundle.json", value)
    return value


@pytest.mark.parametrize("raw", ['{"fps":1,"fps":2}', '{"fps":NaN}', "[" * 130 + "0" + "]" * 130])
def test_json_boundary_rejects_ambiguous_controls(tmp_path, raw):
    path = tmp_path / "control.json"
    path.write_text(raw)
    with pytest.raises(ValueError, match="control"):
        ev.read_json(path)


def test_nonregular_and_oversized_controls(tmp_path, monkeypatch):
    from evals import speaker_inputs as inputs

    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError):
        ev.read_json(fifo)
    monkeypatch.setattr(inputs, "JSON_MAX_BYTES", 2)
    path = tmp_path / "size.json"
    path.write_text("{} ")
    with pytest.raises(ValueError):
        ev.read_json(path)
    path.write_text("{}")
    assert ev.read_json(path) == {}


@pytest.mark.parametrize(
    "truth",
    [
        {"fps": True, "frames": []},
        {"fps": 1e308, "frames": []},
        {"fps": 10, "frames": [4]},
        {"fps": 10, "frames": "Alice"},
    ],
)
def test_truth_shape_is_admitted_before_scoring(truth):
    with pytest.raises(ValueError):
        ev.score({"summary": {}, "segments": []}, truth)


def test_work_is_admitted_before_scoring(monkeypatch):
    from evals import speaker_inputs as inputs

    monkeypatch.setattr(inputs, "MAX_FRAME_VISITS", 3)
    side = {
        "summary": {},
        "segments": [{"start": 0, "end": 1, "speaker": "Alice", "speaker_uncertain": False}] * 2,
    }
    with pytest.raises(ValueError):
        ev.score(side, {"fps": 2, "frames": ["Alice"] * 2})


@pytest.mark.parametrize("name", ["../outside.npy", "/outside.npy", "link.npy", "linked/audio.npy"])
def test_bundle_members_cannot_escape(tmp_path, monkeypatch, name):
    path = tmp_path / "bundle"
    value = tiny_bundle(path)
    np.save(tmp_path / "outside.npy", np.zeros(1, dtype=np.float32))
    (path / "link.npy").symlink_to(tmp_path / "outside.npy")
    (path / "linked").symlink_to(tmp_path, target_is_directory=True)
    value["audio"] = name
    ev.write_json(path / "bundle.json", value)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("outside load"))
    with pytest.raises(ValueError):
        ev.run_bundle(path, {})


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("reused_member", ["reference", "job audio"])
def test_bundle_refuses_reused_member_names(tmp_path, monkeypatch, version, reused_member):
    path = tmp_path / "bundle"
    value = tiny_bundle(path)
    value["version"] = version
    value["references"] = [
        {"id": "a", "name": "Speaker A", "windows": ["first.npy"]},
        {"id": "b", "name": "Speaker B", "windows": ["second.npy"]},
    ]
    samples = np.ones(16, dtype=np.float32)
    for name in ("first.npy", "second.npy"):
        np.save(path / name, samples)
    if version == 2:
        value["sample_sha256"] = {
            name: hashlib.sha256(samples).hexdigest()
            for name in ("audio.npy", "first.npy", "second.npy")
        }

    class Backend:
        embedding_version = "fixture"

        def embed(self, samples, sample_rate):
            return [1.0, 0.0]

    monkeypatch.setattr(
        "transcriptions.known_speaker.get_known_speaker_backend", lambda _: Backend()
    )
    ev.write_json(path / "bundle.json", value)
    sidecar, _ = ev.run_bundle(path, {})
    assert sidecar["summary"]["known_speakers"] == ["Speaker A", "Speaker B"]

    value["references"][1]["windows"] = [
        "first.npy" if reused_member == "reference" else "audio.npy"
    ]
    if version == 2:
        del value["sample_sha256"]["second.npy"]
    ev.write_json(path / "bundle.json", value)
    monkeypatch.setattr(ev, "load_arrays", lambda *a, **k: pytest.fail("array admission started"))
    with pytest.raises(ValueError, match=r"bundle\.member\.duplicate"):
        ev.run_bundle(path, {})


def test_bundle_refuses_member_alias_on_case_insensitive_filesystem(tmp_path, monkeypatch):
    path = tmp_path / "bundle"
    value = tiny_bundle(path)
    if not (path / "AUDIO.npy").exists():
        pytest.skip("filesystem distinguishes these names")
    value["references"] = [{"id": "a", "name": "Speaker A", "windows": ["AUDIO.npy"]}]
    ev.write_json(path / "bundle.json", value)
    monkeypatch.setattr(
        "transcriptions.known_speaker.get_known_speaker_backend",
        lambda *a: pytest.fail("backend started"),
    )
    with pytest.raises(ValueError, match=r"bundle\.array") as raised:
        ev.run_bundle(path, {})
    assert "bundle.member.duplicate" in str(raised.value.__cause__)


@pytest.mark.parametrize(
    "array",
    [
        np.zeros((1, 2), dtype=np.float32),
        np.zeros(2, dtype=np.int64),
        np.array([np.nan], dtype=np.float32),
    ],
)
def test_invalid_array_refuses_before_backend(tmp_path, monkeypatch, array):
    path = tmp_path / "bundle"
    tiny_bundle(path)
    np.save(path / "audio.npy", array)
    monkeypatch.setattr(
        "transcriptions.known_speaker.get_known_speaker_backend",
        lambda *a: pytest.fail("backend started"),
    )
    with pytest.raises(ValueError):
        ev.run_bundle(path, {})


def test_huge_header_and_aggregate_refuse_before_load(tmp_path, monkeypatch):
    from evals import speaker_inputs as inputs

    path = tmp_path / "bundle"
    tiny_bundle(path)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("allocated"))
    monkeypatch.setattr(inputs, "MAX_ARRAY_BYTES", 63)
    with pytest.raises(ValueError):
        ev.run_bundle(path, {})
    with (path / "audio.npy").open("wb") as stream:
        np.lib.format.write_array_header_1_0(
            stream, {"descr": "<f4", "fortran_order": False, "shape": (2**50,)}
        )
    with pytest.raises(ValueError):
        ev.run_bundle(path, {})


def test_bundle_array_admission_under_low_descriptor_limit(tmp_path):
    path = tmp_path / "bundle"
    value = tiny_bundle(path)
    (path / "references" / "speaker").mkdir(parents=True)
    windows = [f"references/speaker/{index}.npy" for index in range(128)]
    for name in windows:
        np.save(path / name, np.ones(16, dtype=np.float32))
    value["references"] = [{"id": "a", "name": "Speaker A", "windows": windows}]
    ev.write_json(path / "bundle.json", value)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import resource
import sys
from pathlib import Path

import numpy

from evals.speaker_inputs import load_arrays, read_json, validate_bundle

root = Path(sys.argv[1])
members = validate_bundle(read_json(root / "bundle.json"))
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (64 if soft < 0 else min(soft, 64), hard))
arrays = load_arrays(root, members)
assert len(arrays) == 129
assert all(numpy.array_equal(array, numpy.ones(16, dtype=numpy.float32)) for array in arrays)
""",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_bundle_refuses_member_replaced_after_header_preflight(tmp_path, monkeypatch):
    path = tmp_path / "bundle"
    value = tiny_bundle(path)
    value["references"] = [{"id": "a", "name": "Speaker A", "windows": ["refs/window.npy"]}]
    ev.write_json(path / "bundle.json", value)
    original = path / "refs"
    replacement = path / "replacement"
    for directory in (original, replacement):
        directory.mkdir()
        np.save(directory / "window.npy", np.ones(16, dtype=np.float32))

    read_header = np.lib.format.read_array_header_1_0
    header_count = 0

    def replace_after_last_header(*args, **kwargs):
        nonlocal header_count
        header = read_header(*args, **kwargs)
        header_count += 1
        if header_count == 2:
            # Swapping the parent leaves the already opened member's metadata intact.
            original.rename(path / "old")
            replacement.rename(original)
        return header

    monkeypatch.setattr(np.lib.format, "read_array_header_1_0", replace_after_last_header)
    monkeypatch.setattr(
        "transcriptions.known_speaker.get_known_speaker_backend",
        lambda *a: pytest.fail("backend started"),
    )
    with pytest.raises(ValueError, match=r"bundle\.array") as raised:
        ev.run_bundle(path, {})
    assert "bundle.array.changed" in str(raised.value.__cause__)


@pytest.mark.parametrize("role", ["result", "turn"])
def test_bundle_constructor_shapes_refuse_before_backend(tmp_path, monkeypatch, role):
    path = tmp_path / "bundle"
    value = tiny_bundle(path)
    if role == "result":
        value["result"]["unexpected"] = "invalid"
    else:
        value["turns"]["regular"] = [{"start": 0, "end": 1, "speaker": "a", "extra": 1}]
    ev.write_json(path / "bundle.json", value)
    monkeypatch.setattr(
        "transcriptions.known_speaker.get_known_speaker_backend",
        lambda *a: pytest.fail("bad shape reached backend"),
    )
    with pytest.raises(ValueError):
        ev.run_bundle(path, {})


def test_bundle_hardlink_member_refuses_before_backend(tmp_path, monkeypatch):
    path = tmp_path / "bundle"
    tiny_bundle(path)
    outside = tmp_path / "outside.npy"
    (path / "audio.npy").rename(outside)
    os.link(outside, path / "audio.npy")
    monkeypatch.setattr(
        "transcriptions.known_speaker.get_known_speaker_backend",
        lambda *a: pytest.fail("outside hardlink reached backend"),
    )
    with pytest.raises(ValueError):
        ev.run_bundle(path, {})


def test_long_episode_timestamps_are_admitted():
    from evals import speaker_inputs as inputs

    three_hours = 3 * 60 * 60
    assert inputs.interval({"start": three_hours - 1, "end": three_hours}, "x") == (
        three_hours - 1,
        three_hours,
    )
    with pytest.raises(ValueError):
        inputs.interval({"start": 0, "end": inputs.MAX_SECONDS + 1}, "x")
