"""Output ownership for the standalone speaker CLIs, using synthetic files."""

import json
import os
import sys
from typing import Any

import pytest

from evals import known_speaker_eval as known
from evals import mixed_speaker_segments_probe as probe
from transcriptions import known_speaker as ks


@pytest.mark.parametrize(
    "command,source",
    [
        ("known", "gold"),
        ("known", "production"),
        ("known", "reference"),
        ("subwindow", "audio"),
        ("subwindow", "reference"),
        ("stt", "audio"),
        ("turns", "audio"),
        ("stt", "env"),
    ],
)
@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_source_output_alias_refused_before_work(monkeypatch, tmp_path, command, source, alias):
    paths = {key: tmp_path / key for key in ("gold", "production", "reference", "audio", "env")}
    for path in paths.values():
        path.write_bytes(b"synthetic source bytes")
    paths["gold"].write_text(json.dumps([{"start": 0, "end": 1, "speaker": "Alice", "text": "hi"}]))
    before = {key: path.read_bytes() for key, path in paths.items()}
    target = paths[source]
    if alias != "direct":
        out = tmp_path / "report.json"
        if alias == "symlink":
            out.symlink_to(target)
        else:
            os.link(target, out)
    else:
        out = target
    if command == "known":
        argv = [
            "eval",
            "--gold",
            str(paths["gold"]),
            "--production-audio",
            str(paths["production"]),
            "--out",
            str(out),
        ]
        monkeypatch.setattr(known, "DEFAULT_REFERENCES", {"Alice": paths["reference"]})
        monkeypatch.setattr(
            known, "PyannoteEmbeddingBackend", lambda **kw: pytest.fail("backend acquired")
        )
        main = known.main
    else:
        argv = [
            "probe",
            "--env-file",
            str(paths["env"]),
            command,
            str(paths["audio"]),
            "--out",
            str(out),
        ]
        if command == "subwindow":
            argv += [str(paths["reference"]), "--reference", "Alice=0-1", "--target", "0-1"]
        monkeypatch.setattr(probe, "setup", lambda *a: pytest.fail("setup reached"))
        main = probe.main
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert {key: path.read_bytes() for key, path in paths.items()} == before


def test_late_output_alias_rechecked_before_publication(monkeypatch, tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"synthetic input")
    out = tmp_path / "report.json"
    args = probe.build_parser().parse_args(
        [
            "subwindow",
            str(audio),
            str(audio),
            "--reference",
            "Alice=0-1",
            "--target",
            "0-1",
            "--window",
            "1-1",
            "--out",
            str(out),
        ]
    )
    monkeypatch.setattr(ks, "decode_mono_16k", lambda *a: [1] * 16000)

    class Backend:
        def embed(self, *args):
            if not out.exists():
                os.link(audio, out)
            return [1.0, 0.0]

    monkeypatch.setattr(ks, "get_known_speaker_backend", lambda *a: Backend())
    with pytest.raises(ValueError, match="output"):
        probe.run_subwindow(args)
    assert audio.read_bytes() == b"synthetic input"


@pytest.mark.parametrize("failure", ["serialization", "write", "replace"])
def test_failed_publication_preserves_complete_report(monkeypatch, tmp_path, failure):
    import tempfile
    from pathlib import Path

    from evals import speaker_reports as reports

    out = tmp_path / "report.json"
    old = b'{"generation":"old"}\n'
    out.write_bytes(old)
    report: dict[str, Any] = {"generation": "new"}
    error: type[Exception] = OSError
    if failure == "serialization":
        report["generation"] = object()
        error = TypeError
    elif failure == "write":
        real_temporary = tempfile.NamedTemporaryFile
        real_write = Path.write_text

        class FailingWriter:
            def __init__(self, *args, **kwargs):
                self.file = real_temporary(*args, **kwargs)
                self.name = self.file.name

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.file.__exit__(*args)

            def write(self, data):
                self.file.write(data[:3])
                self.file.flush()
                raise OSError("injected write failure")

        def fail_in_place(path, text, *args, **kwargs):
            real_write(path, text[:3], *args, **kwargs)
            raise OSError("injected write failure")

        monkeypatch.setattr(tempfile, "NamedTemporaryFile", FailingWriter)
        monkeypatch.setattr(Path, "write_text", fail_in_place)
    else:

        def fail_replace(*args):
            raise OSError("injected replace failure")

        monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(error):
        reports.publish_report(report, out, [])
    assert out.read_bytes() == old
    assert list(tmp_path.iterdir()) == [out]


def test_successful_publication_is_complete_and_removes_owned_temporary(tmp_path):
    from evals.speaker_reports import publish_report

    out = tmp_path / "report.json"
    out.write_text('{"generation":"old"}')
    report = {"generation": "new", "text": "synthetic umlaut ü"}
    publish_report(report, out, [])
    assert json.loads(out.read_text(encoding="utf-8")) == report
    assert list(tmp_path.iterdir()) == [out]


@pytest.mark.parametrize("boundary", ["before", "after"])
def test_process_death_at_replace_keeps_old_or_new_report(tmp_path, boundary):
    import subprocess
    from pathlib import Path

    out = tmp_path / "report.json"
    out.write_text('{"generation":"old"}')
    code = """
import os
import sys
from pathlib import Path
from evals.speaker_reports import publish_report
real_replace = os.replace
def stop_at_boundary(source, dest):
    if sys.argv[2] == "before":
        os._exit(73)
    real_replace(source, dest)
    os._exit(73)
os.replace = stop_at_boundary
publish_report({"generation": "new"}, Path(sys.argv[1]), [])
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(out), boundary],
        cwd=Path(__file__).resolve().parents[1],
        timeout=10,
    )
    assert result.returncode == 73
    assert json.loads(out.read_text()) == {"generation": "old" if boundary == "before" else "new"}
    # An abrupt exit can leave an owned sibling temporary; the final path is complete.
    assert all(path.name.startswith(".report.json.") for path in tmp_path.iterdir() if path != out)


def test_destination_revalidated_after_temporary_is_closed(monkeypatch, tmp_path):
    import tempfile

    from evals.speaker_reports import publish_report

    source = tmp_path / "source.wav"
    source.write_bytes(b"synthetic source")
    out = tmp_path / "report.json"
    real_temporary = tempfile.NamedTemporaryFile

    class SwapOnClose:
        def __init__(self, *args, **kwargs):
            self.file = real_temporary(*args, **kwargs)

        def __enter__(self):
            return self.file

        def __exit__(self, *args):
            self.file.__exit__(*args)
            os.link(source, out)

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", SwapOnClose)
    with pytest.raises(ValueError, match="output"):
        publish_report({"complete": True}, out, [source])
    assert source.read_bytes() == b"synthetic source"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["report.json", "source.wav"]
