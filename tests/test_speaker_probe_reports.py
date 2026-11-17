"""Output ownership for the standalone speaker CLIs, using synthetic files."""

import json
import os
import sys

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
