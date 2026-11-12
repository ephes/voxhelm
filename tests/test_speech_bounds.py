"""Speech endpoint admission, public errors, voice discovery, streaming and export bounds."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

from synthesis.service import (
    SynthesisResult,
    VoiceNotInstalledError,
    discover_installed_voices,
    export_audio,
)
from tests.test_synthesis import write_voice_fixture

PRIVATE = "/private/synthetic-secret-path"


class SpeechResult:
    def __init__(self, path: Path) -> None:
        self.audio_path = path
        self.backend_name = "piper"
        self.voice_name = "synthetic"
        self.language = "en"


def post_speech(client, body: bytes | str):
    return client.post(
        "/v1/audio/speech",
        data=body,
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )


def refuse_synthesis(monkeypatch) -> None:
    def fail(*args, **kwargs):
        pytest.fail("request must be refused before synthesis")

    monkeypatch.setattr("synthesis.views.synthesize_text", fail)


# --- JSON scalar and depth admission ------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        '{"model": "tts-1", "input": "Hi", "speed": NaN}',
        '{"model": "tts-1", "input": "Hi", "speed": Infinity}',
        '{"model": "tts-1", "input": "Hi", "speed": -Infinity}',
        '{"model": "tts-1", "input": "Hi", "extra": ' + "9" * 5000 + "}",
    ],
    ids=["nan", "infinity", "negative-infinity", "integer-digit-limit"],
)
def test_speech_rejects_unrepresentable_json_scalars(client, monkeypatch, body) -> None:
    refuse_synthesis(monkeypatch)
    response = post_speech(client, body)
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "Request body was not valid JSON."


@pytest.mark.parametrize(
    "speed", ["true", "false", "1" + "0" * 400, "1e400"], ids=["true", "false", "huge", "overflow"]
)
def test_speech_rejects_non_numeric_or_unbounded_speed(client, monkeypatch, speed) -> None:
    refuse_synthesis(monkeypatch)
    response = post_speech(client, '{"model": "tts-1", "input": "Hi", "speed": ' + speed + "}")
    assert response.status_code == 400
    assert "'speed'" in response.json()["error"]["message"]


def test_speech_parser_nesting_limit_is_invalid_json(client, monkeypatch) -> None:
    """Nesting limits vary between Python builds; inject the parser's failure."""
    refuse_synthesis(monkeypatch)

    original = json.loads

    def nested(text, **kwargs):
        if "parse_constant" not in kwargs:  # the test client's own response parsing
            return original(text, **kwargs)
        raise RecursionError("maximum recursion depth exceeded while decoding a JSON array")

    monkeypatch.setattr("synthesis.views.json.loads", nested)
    response = post_speech(client, '{"model": "tts-1", "input": "Hi"}')
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "Request body was not valid JSON."


def test_speech_deeply_nested_body_is_a_client_error(client, monkeypatch) -> None:
    refuse_synthesis(monkeypatch)
    depth = 200_000
    body = '{"model": "tts-1", "input": "Hi", "extra": ' + "[" * depth + "]" * depth + "}"
    response = post_speech(client, body)
    # Either the parser refuses the depth or the request is otherwise rejected;
    # it must never become a server error.
    assert response.status_code == 400


# --- Public error diagnostic boundary ------------------------------------------------------


def test_speech_backend_failure_hides_diagnostics(client, monkeypatch, caplog) -> None:
    def fail(text, params):
        raise RuntimeError(f"model missing at {PRIVATE}")

    monkeypatch.setattr("synthesis.views.synthesize_text", fail)
    response = post_speech(client, json.dumps({"model": "tts-1", "input": "Hi"}))
    assert response.status_code == 500
    assert response.json()["error"] == {
        "message": "Speech synthesis failed.",
        "type": "server_error",
    }
    assert PRIVATE in caplog.text


def test_speech_missing_voice_is_a_client_error(client, monkeypatch) -> None:
    def fail(text, params):
        raise VoiceNotInstalledError("Requested voice 'synthetic' is not installed.")

    monkeypatch.setattr("synthesis.views.synthesize_text", fail)
    response = post_speech(client, json.dumps({"model": "tts-1", "input": "Hi"}))
    assert response.status_code == 400
    assert response.json()["error"] == {
        "message": "Requested voice 'synthetic' is not installed.",
        "type": "invalid_request_error",
    }


def test_speech_export_failure_hides_converter_output(client, monkeypatch, settings, tmp_path):
    source = tmp_path / "speech.wav"
    source.write_bytes(b"RIFFtest")
    monkeypatch.setattr(
        "synthesis.views.synthesize_text", lambda text, params: SpeechResult(source)
    )
    fake_ffmpeg = tmp_path / "ffmpeg"
    fake_ffmpeg.write_text(
        f"#!{sys.executable}\nimport sys\nsys.stderr.write({PRIVATE!r})\nraise SystemExit(1)\n"
    )
    fake_ffmpeg.chmod(0o755)
    settings.VOXHELM_FFMPEG_BIN = str(fake_ffmpeg)
    response = post_speech(
        client, json.dumps({"model": "tts-1", "input": "Hi", "response_format": "mp3"})
    )
    assert response.status_code == 500
    assert PRIVATE not in response.content.decode()
    assert not source.exists()


# --- Voice discovery isolation ---------------------------------------------------------------


@pytest.mark.parametrize(
    "corruption",
    [b"{not json", b"\xff\xfe", b"[]", b'{"speaker_id_map": 5}'],
    ids=["syntax", "encoding", "not-object", "bad-speakers"],
)
@pytest.mark.parametrize("configured", [True, False])
def test_unrelated_corrupt_voice_does_not_hide_other_voices(
    tmp_path, corruption, configured
) -> None:
    write_voice_fixture(tmp_path, "en_US-good-medium")
    write_voice_fixture(tmp_path, "de_DE-broken-high")
    (tmp_path / "de_DE-broken-high.onnx.json").write_bytes(corruption)
    discovered = discover_installed_voices(
        voice_dir=tmp_path,
        configured_voices=["de_DE-broken-high", "en_US-good-medium"] if configured else [],
    )
    assert list(discovered) == ["en_US-good-medium"]


# --- Streamed file response lifetime -----------------------------------------------------------


def test_speech_response_streams_after_owned_files_are_removed(client, monkeypatch, tmp_path):
    source = tmp_path / "speech.wav"
    payload = b"RIFF" + b"x" * 200_000
    source.write_bytes(payload)
    monkeypatch.setattr(
        "synthesis.views.synthesize_text", lambda text, params: SpeechResult(source)
    )
    response = post_speech(client, json.dumps({"model": "tts-1", "input": "Hi"}))
    assert response.status_code == 200
    assert response.streaming
    assert response["Content-Length"] == str(len(payload))
    # The temporary WAV is already unlinked; the open handle still serves it whole.
    assert not source.exists()
    assert b"".join(response.streaming_content) == payload


# --- Export diagnostic ceiling and owned process tree ------------------------------------------


def _result(tmp_path: Path) -> SynthesisResult:
    source = tmp_path / "source.wav"
    source.write_bytes(b"RIFF")
    return SynthesisResult(source, "piper", "piper", "voice", "en", 16000, 2, 1, 1.0)


def _pid_gone(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        time.sleep(0.05)
    return False


def test_export_diagnostic_output_is_capped(tmp_path, settings) -> None:
    pid_file = tmp_path / "pid"
    fake_ffmpeg = tmp_path / "ffmpeg"
    fake_ffmpeg.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, sys, time\n"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
        "pathlib.Path(sys.argv[-1]).write_bytes(b'partial')\n"
        "sys.stderr.write('x' * (1024 * 1024)); sys.stderr.flush()\n"
        "time.sleep(30)\n"
    )
    fake_ffmpeg.chmod(0o755)
    settings.VOXHELM_FFMPEG_BIN = str(fake_ffmpeg)
    settings.VOXHELM_AUDIO_EXPORT_TIMEOUT_SECONDS = 20
    started = time.monotonic()
    with pytest.raises(RuntimeError, match=r"^Audio conversion exceeded the diagnostic"):
        export_audio(_result(tmp_path), output_format="mp3")
    assert time.monotonic() - started < 10
    assert _pid_gone(int(pid_file.read_text()))


def test_export_deadline_covers_descendants_holding_pipes(tmp_path, settings) -> None:
    """A converter that exits while a descendant keeps stderr open is bounded and killed."""
    child_pid_file = tmp_path / "child-pid"
    fake_ffmpeg = tmp_path / "ffmpeg"
    fake_ffmpeg.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, subprocess, sys\n"
        "pathlib.Path(sys.argv[-1]).write_bytes(b'converted')\n"
        f"child = subprocess.Popen([{sys.executable!r}, '-c', 'import time; time.sleep(60)'])\n"
        f"pathlib.Path({str(child_pid_file)!r}).write_text(str(child.pid))\n"
    )
    fake_ffmpeg.chmod(0o755)
    settings.VOXHELM_FFMPEG_BIN = str(fake_ffmpeg)
    settings.VOXHELM_AUDIO_EXPORT_TIMEOUT_SECONDS = 1
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="export deadline"):
        export_audio(_result(tmp_path), output_format="ogg")
    assert time.monotonic() - started < 10
    assert _pid_gone(int(child_pid_file.read_text()))


def test_export_success_kills_lingering_descendants(tmp_path, settings) -> None:
    """Descendants that detached from stderr do not outlive a successful export."""
    child_pid_file = tmp_path / "child-pid"
    fake_ffmpeg = tmp_path / "ffmpeg"
    fake_ffmpeg.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, subprocess, sys\n"
        "pathlib.Path(sys.argv[-1]).write_bytes(b'converted')\n"
        "child = subprocess.Popen(\n"
        f"    [{sys.executable!r}, '-c', 'import time; time.sleep(60)'],\n"
        "    stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL,\n"
        ")\n"
        f"pathlib.Path({str(child_pid_file)!r}).write_text(str(child.pid))\n"
    )
    fake_ffmpeg.chmod(0o755)
    settings.VOXHELM_FFMPEG_BIN = str(fake_ffmpeg)
    exported = export_audio(_result(tmp_path), output_format="mp3")
    assert exported.path.read_bytes() == b"converted"
    exported.path.unlink()
    assert _pid_gone(int(child_pid_file.read_text()))


@pytest.mark.django_db(transaction=True)
def test_asgi_speech_response_streams_without_materializing(monkeypatch, tmp_path) -> None:
    import asyncio
    import warnings

    from django.http import StreamingHttpResponse
    from django.test import AsyncClient

    source = tmp_path / "speech.wav"
    payload = b"RIFF" + b"y" * 300_000
    source.write_bytes(payload)
    monkeypatch.setattr(
        "synthesis.views.synthesize_text", lambda text, params: SpeechResult(source)
    )

    async def flow() -> None:
        response = await AsyncClient().post(
            "/v1/audio/speech",
            data=json.dumps({"model": "tts-1", "input": "Hi"}),
            content_type="application/json",
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 200
        assert isinstance(response, StreamingHttpResponse)
        assert response.is_async
        assert response["Content-Length"] == str(len(payload))
        assert not source.exists()
        iterator = response.streaming_content
        first = await anext(iterator)
        assert len(first) <= 65536
        rest = b"".join([chunk async for chunk in iterator])
        assert first + rest == payload

    with warnings.catch_warnings():
        # Django warns when it must materialize a synchronous iterator under ASGI.
        warnings.simplefilter("error")
        asyncio.run(flow())
