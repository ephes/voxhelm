"""Synthetic native inference output admission; no model or media is used."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from transcriptions import service


@pytest.mark.parametrize("pipe", [1, 2])
def test_native_diagnostic_flood_is_refused_and_reaped(monkeypatch, pipe):
    children = []
    popen = subprocess.Popen

    def record(*args, **kwargs):
        child = popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", record)
    with pytest.raises(RuntimeError, match="diagnostic output limit"):
        service.run_cancellable_process(
            [sys.executable, "-c", f"import os; os.write({pipe}, b'x' * 2000000)"],
            cancel_event=None,
        )
    assert children[0].returncode is not None
    assert children[0].stdout is not None and children[0].stdout.closed
    assert children[0].stderr is not None and children[0].stderr.closed


@pytest.mark.parametrize(
    "kind", ["duplicate", "deep", "nonfinite", "array", "oversize", "link", "fifo"]
)
def test_whispercpp_refuses_untrusted_transcript(monkeypatch, tmp_path, kind):
    monkeypatch.setattr(service, "resolve_whispercpp_binary", lambda _: "synthetic-cli")
    monkeypatch.setattr(service, "resolve_whispercpp_model_path", lambda _: tmp_path / "model")
    monkeypatch.setattr(service, "normalize_audio_to_wav_16k_mono", lambda **_: None)

    def produce(args, **kwargs):
        path = Path(args[args.index("-of") + 1]).with_suffix(".json")
        if kind == "link":
            target = tmp_path / "target.json"
            target.write_text("{}")
            path.symlink_to(target)
        elif kind == "fifo":
            os.mkfifo(path)
        elif kind == "oversize":
            with path.open("wb") as handle:
                handle.truncate(64 * 1024 * 1024 + 1)
        else:
            path.write_text(
                {
                    "duplicate": '{"result":{},"result":{}}',
                    "deep": '{"a":' + "[" * 130 + "0" + "]" * 130 + "}",
                    "nonfinite": '{"a":NaN}',
                    "array": "[]",
                }[kind]
            )
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(service, "run_cancellable_process", produce)
    backend = service.WhisperCppBackend(binary_path="cli", model_name="model", processors=1)
    with pytest.raises(
        RuntimeError, match="whisper.cpp transcription produced invalid JSON output"
    ):
        backend.transcribe(tmp_path / "input.wav", service.TranscribeParams("auto", None, None))


def test_native_output_keeps_admitted_language_evidence():
    from transcriptions.windows import detected_language

    completed = service.run_cancellable_process(
        [
            sys.executable,
            "-c",
            "import sys; sys.stderr.write("
            "'auto-detected language: de (p = 0.990000)\\n' + 'x' * 8192)",
        ],
        cancel_event=None,
    )
    assert detected_language(completed.stderr, {"result": {"language": "de"}}) == ("de", 0.99)
