"""Synthetic native inference output admission; no model or media is used."""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile

from tests.test_service import write_wav
from transcriptions import service
from transcriptions.errors import TranscriptionError
from transcriptions.owned_process import OWNED_PROCESS_DIAGNOSTIC_MAX_BYTES


@pytest.mark.parametrize("pipe", [1, 2])
def test_native_output_flood_keeps_bounded_head_and_tail_and_reaps(monkeypatch, pipe):
    children = []
    popen = subprocess.Popen

    def record(*args, **kwargs):
        child = popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", record)
    completed = service.run_cancellable_process(
        [
            sys.executable,
            "-c",
            f"import os; os.write({pipe}, b'HEAD' + b'x' * 2000000 + b'TAIL'); os._exit(3)",
        ],
        cancel_event=None,
    )
    flooded = completed.stdout if pipe == 1 else completed.stderr
    assert completed.returncode == 3
    assert flooded.startswith("HEAD") and flooded.endswith("TAIL")
    assert "bytes of output omitted]" in flooded
    retained = len(completed.stdout.encode()) + len(completed.stderr.encode())
    assert retained <= OWNED_PROCESS_DIAGNOSTIC_MAX_BYTES + 64
    assert children[0].returncode is not None
    assert children[0].stdout is not None and children[0].stdout.closed
    assert children[0].stderr is not None and children[0].stderr.closed


def test_long_successful_native_output_is_not_refused():
    # whisper-cli prints each segment to stdout; a long episode writes far more
    # than the retained 64 KiB and must still succeed.
    segment = "[00:00:00.000 --> 00:00:02.000]  some transcript text\n"
    completed = service.run_cancellable_process(
        [
            sys.executable,
            "-c",
            "import sys; sys.stderr.write('auto-detected language: de\\n'); "
            f"sys.stdout.write({segment!r} * 5000)",
        ],
        cancel_event=None,
    )
    assert completed.returncode == 0
    assert completed.stderr == "auto-detected language: de\n"
    assert completed.stdout.startswith(segment)
    assert completed.stdout.endswith(segment)


def test_language_evidence_after_a_long_model_load_log_is_kept():
    # Window runs parse "auto-detected language:" from stderr; it follows the
    # model-load log, which some builds make long, while stdout floods.
    completed = service.run_cancellable_process(
        [
            sys.executable,
            "-c",
            "import sys; sys.stderr.write('ggml_metal: kernel loaded\\n' * 1600); "
            "sys.stderr.write('auto-detected language: de (p = 0.987654)\\n'); "
            "sys.stderr.write('whisper_print_timings: per window\\n' * 1000); sys.stderr.flush(); "
            "sys.stdout.write('[00:00:00.000 --> 00:00:02.000]  text\\n' * 20000)",
        ],
        cancel_event=None,
    )
    assert completed.returncode == 0
    # The line sits past 40 KiB, followed by more than the 8 KiB tail of later log
    # lines: only the 52 KiB stderr head keeps it.
    assert "bytes of output omitted]" in completed.stderr
    assert completed.stderr.count("auto-detected language: de (p = 0.987654)") == 1


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


@pytest.mark.parametrize(
    "record",
    [
        {"text": None, "timestamps": {"from": "00:00:00,000", "to": "00:00:01,000"}},
        {"text": 42, "timestamps": {"from": "00:00:00,000", "to": "00:00:01,000"}},
        {"text": "Synthetic", "timestamps": {"to": "00:00:01,000"}},
        {"text": "Synthetic", "timestamps": {"from": "00:00:00,000"}},
        {"text": "Synthetic", "timestamps": {"from": "NaN:00:00,000", "to": "00:00:01,000"}},
        {"text": "Synthetic", "timestamps": {"from": "invalid", "to": "00:00:01,000"}},
        {"text": "Synthetic"},
        ["Synthetic"],
    ],
)
def test_whispercpp_rejects_invalid_segment_before_coverage(client, monkeypatch, tmp_path, record):
    monkeypatch.setattr(service, "resolve_whispercpp_binary", lambda _: "synthetic-cli")
    monkeypatch.setattr(service, "resolve_whispercpp_model_path", lambda _: tmp_path / "model")
    monkeypatch.setattr(
        service,
        "normalize_audio_to_wav_16k_mono",
        lambda **kwargs: write_wav(kwargs["output_path"]),
    )
    valid = {"text": "Valid", "timestamps": {"from": "00:00:00,000", "to": "00:00:01,000"}}
    records = [valid, record]

    def produce(args, **kwargs):
        path = Path(args[args.index("-of") + 1]).with_suffix(".json")
        path.write_text(json.dumps({"transcription": records}))
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(service, "run_cancellable_process", produce)
    coverage = Mock(wraps=service.check_transcript_coverage)
    monkeypatch.setattr(service, "check_transcript_coverage", coverage)
    backend = service.WhisperCppBackend(binary_path="cli", model_name="model", processors=1)
    params = service.TranscribeParams("auto", None, None)

    with pytest.raises(
        TranscriptionError, match="whisper.cpp transcription produced invalid segment"
    ):
        backend.transcribe(tmp_path / "input.wav", params)
    monkeypatch.setattr(
        service,
        "get_backend_services_for_model",
        lambda _model: [service.BackendInvocation("whispercpp", backend)],
    )
    response = client.post(
        "/v1/audio/transcriptions",
        data={
            "file": SimpleUploadedFile("input.wav", b"synthetic audio", content_type="audio/wav"),
            "model": "whisper-1",
            "response_format": "vtt",
        },
        HTTP_AUTHORIZATION="Bearer test-token",
    )
    assert response.status_code == 500
    assert "invalid segment" in response.json()["error"]["message"]
    coverage.assert_not_called()

    records[1] = {"text": " \n "}
    result = backend.transcribe(tmp_path / "input.wav", params)
    assert result.segments == [service.TranscriptionSegment(id=0, start=0, end=1, text="Valid")]
    coverage.assert_called_once()


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


def test_exited_child_with_pipe_holding_descendant_returns_and_retires_it(tmp_path):
    import time

    pid_file = tmp_path / "grandchild.pid"
    script = (
        "import subprocess, sys; "
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        f"open({str(pid_file)!r}, 'w').write(str(p.pid)); "
        "print('done', flush=True)"
    )
    started = time.monotonic()
    completed = service.run_cancellable_process([sys.executable, "-c", script], cancel_event=None)
    assert time.monotonic() - started < 15
    assert completed.returncode == 0
    assert completed.stdout.startswith("done")
    grandchild = int(pid_file.read_text())
    for _ in range(100):
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        os.kill(grandchild, 9)
        raise AssertionError("pipe-holding descendant survived")
