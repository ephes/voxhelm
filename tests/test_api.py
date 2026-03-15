from __future__ import annotations

import io
import json
import tempfile
import wave
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile

from config.settings import env_tokens
from transcriptions.errors import ApiError
from transcriptions.service import TranscribeParams, TranscriptionResult, TranscriptionSegment


class DummyBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, TranscribeParams]] = []

    def transcribe(self, audio_path: Path, params: TranscribeParams) -> TranscriptionResult:
        self.calls.append((audio_path, params))
        return TranscriptionResult(
            text="Hello world",
            language=params.language or "en",
            segments=[
                TranscriptionSegment(id=0, start=0.0, end=1.5, text="Hello"),
                TranscriptionSegment(id=1, start=1.5, end=3.0, text="world"),
            ],
        )


class DummySpeechResult:
    def __init__(self, audio_path: Path) -> None:
        self.audio_path = audio_path


def wav_bytes(*, frames: int = 320) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setframerate(16000)
        wav_file.setsampwidth(2)
        wav_file.setnchannels(1)
        wav_file.writeframes(b"\x01\x00" * frames)
    return buffer.getvalue()


def test_health_endpoint(client):
    response = client.get("/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_transcription_requires_bearer_token(client):
    response = client.post("/v1/audio/transcriptions", data={})

    assert response.status_code == 401
    assert response.json()["error"]["type"] == "authentication_error"


def test_upload_transcription_returns_json(client, monkeypatch):
    backend = DummyBackend()
    monkeypatch.setattr("transcriptions.service.get_backend_service", lambda: backend)
    upload = SimpleUploadedFile("sample.mp3", b"mp3-bytes", content_type="audio/mpeg")

    response = client.post(
        "/v1/audio/transcriptions",
        data={"file": upload, "model": "gpt-4o-mini-transcribe", "prompt": "context"},
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert response.status_code == 200
    assert response.json() == {"text": "Hello world"}
    assert backend.calls[0][1].prompt == "context"


def test_upload_transcription_uses_non_interactive_scheduler_lane(client, monkeypatch):
    lanes: list[str] = []

    @contextmanager
    def fake_admit(lane: str):
        lanes.append(lane)
        yield object()

    monkeypatch.setattr("transcriptions.service.admit_local_inference", fake_admit)
    monkeypatch.setattr("transcriptions.service.get_backend_service", lambda: DummyBackend())
    upload = SimpleUploadedFile("sample.mp3", b"mp3-bytes", content_type="audio/mpeg")

    response = client.post(
        "/v1/audio/transcriptions",
        data={"file": upload, "model": "gpt-4o-mini-transcribe"},
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert response.status_code == 200
    assert lanes == ["non-interactive"]


def test_upload_transcription_emits_debug_log(client, monkeypatch):
    debug_calls: list[dict[str, object]] = []
    monkeypatch.setattr("transcriptions.service.get_backend_service", lambda: DummyBackend())
    monkeypatch.setattr(
        "transcriptions.views.emit_transcription_debug_log",
        lambda **kwargs: debug_calls.append(kwargs),
    )
    upload = SimpleUploadedFile("sample.wav", wav_bytes(), content_type="audio/wav")

    response = client.post(
        "/v1/audio/transcriptions",
        data={"file": upload, "model": "whisper-1", "language": "en"},
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert response.status_code == 200
    assert len(debug_calls) == 1
    debug_payload = debug_calls[0]
    audio_shape = cast(dict[str, Any], debug_payload["audio_shape"])
    assert debug_payload["source"] == "http.audio_transcriptions"
    assert debug_payload["request_model"] == "whisper-1"
    assert debug_payload["request_language"] == "en"
    assert debug_payload["prompt"] is None
    assert "path" not in audio_shape
    assert audio_shape["suffix"] == ".wav"
    assert audio_shape["rate"] == 16000
    assert audio_shape["channels"] == 1


def test_text_response_format(client, monkeypatch):
    monkeypatch.setattr("transcriptions.service.get_backend_service", lambda: DummyBackend())
    upload = SimpleUploadedFile("sample.mp3", b"mp3-bytes", content_type="audio/mpeg")

    response = client.post(
        "/v1/audio/transcriptions",
        data={"file": upload, "model": "whisper-1", "response_format": "text"},
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert response.status_code == 200
    assert response["Content-Type"].startswith("text/plain")
    assert response.content.decode() == "Hello world"


def test_verbose_json_response_format(client, monkeypatch):
    monkeypatch.setattr("transcriptions.service.get_backend_service", lambda: DummyBackend())
    upload = SimpleUploadedFile("sample.mp3", b"mp3-bytes", content_type="audio/mpeg")

    response = client.post(
        "/v1/audio/transcriptions",
        data={"file": upload, "model": "whisper-1", "response_format": "verbose_json"},
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["text"] == "Hello world"
    assert payload["segments"][0]["id"] == 0
    assert payload["segments"][1]["text"] == "world"


def test_vtt_response_format(client, monkeypatch):
    monkeypatch.setattr("transcriptions.service.get_backend_service", lambda: DummyBackend())
    upload = SimpleUploadedFile("sample.mp3", b"mp3-bytes", content_type="audio/mpeg")

    response = client.post(
        "/v1/audio/transcriptions",
        data={"file": upload, "model": "whisper-1", "response_format": "vtt"},
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    text = response.content.decode()
    assert response.status_code == 200
    assert text.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.500" in text


def test_sync_contract_rejects_dote_response_format(client):
    upload = SimpleUploadedFile("sample.mp3", b"mp3-bytes", content_type="audio/mpeg")

    response = client.post(
        "/v1/audio/transcriptions",
        data={"file": upload, "model": "whisper-1", "response_format": "dote"},
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert response.status_code == 400
    assert "json, text, verbose_json, or vtt" in response.json()["error"]["message"]


def test_sync_contract_rejects_podlove_response_format(client):
    upload = SimpleUploadedFile("sample.mp3", b"mp3-bytes", content_type="audio/mpeg")

    response = client.post(
        "/v1/audio/transcriptions",
        data={"file": upload, "model": "whisper-1", "response_format": "podlove"},
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert response.status_code == 400
    assert "json, text, verbose_json, or vtt" in response.json()["error"]["message"]


def test_env_tokens_rejects_reserved_label_in_json_syntax(monkeypatch):
    monkeypatch.setenv("VOXHELM_BEARER_TOKENS", '{"__operator_ui__": "secret"}')

    try:
        env_tokens("VOXHELM_BEARER_TOKENS")
    except ValueError as exc:
        assert "reserved label" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected reserved bearer token label to be rejected")


def test_url_mode_uses_allowlist(client, monkeypatch, settings):
    backend = DummyBackend()
    monkeypatch.setattr("transcriptions.service.get_backend_service", lambda: backend)
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}

    def fake_download(*, source_url: str):
        assert source_url == "https://media.example.com/episode.mp3"
        path = Path(settings.BASE_DIR) / "tmp-test.mp3"
        path.write_bytes(b"mp3-bytes")
        return path

    monkeypatch.setattr("transcriptions.views.download_allowed_url_to_tempfile", fake_download)

    response = client.post(
        "/v1/audio/transcriptions",
        data=json.dumps(
            {"url": "https://media.example.com/episode.mp3", "model": "gpt-4o-mini-transcribe"}
        ),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert response.status_code == 200
    assert response.json()["text"] == "Hello world"
    assert backend.calls


def test_url_mode_rejects_non_allowlisted_hosts(client):
    response = client.post(
        "/v1/audio/transcriptions",
        data=json.dumps({"url": "https://blocked.example.com/file.mp3", "model": "whisper-1"}),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert response.status_code == 400
    assert "allowlist" in response.json()["error"]["message"]


def test_url_download_cleanup_on_size_limit(monkeypatch, settings, tmp_path):
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}
    settings.VOXHELM_MAX_URL_DOWNLOAD_BYTES = 4

    class DummyHeaders:
        def get_content_type(self) -> str:
            return "audio/mpeg"

    class DummyResponse:
        headers = DummyHeaders()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def geturl(self) -> str:
            return "https://media.example.com/file.mp3"

        def read(self, _size: int) -> bytes:
            if hasattr(self, "_done"):
                return b""
            self._done = True
            return b"12345"

    created_paths: list[Path] = []
    original_named_temporary_file = tempfile.NamedTemporaryFile

    def fake_named_temporary_file(*args, **kwargs):
        kwargs = {"dir": tmp_path, **kwargs}
        handle = original_named_temporary_file(*args, **kwargs)
        created_paths.append(Path(handle.name))
        return handle

    monkeypatch.setattr(
        "transcriptions.input_media.urlopen",
        lambda request, timeout: DummyResponse(),
    )
    monkeypatch.setattr(
        "transcriptions.input_media.tempfile.NamedTemporaryFile",
        fake_named_temporary_file,
    )
    from transcriptions.input_media import download_allowed_url_to_tempfile

    try:
        download_allowed_url_to_tempfile(source_url="https://media.example.com/file.mp3")
    except ApiError as exc:
        assert "download limit" in exc.message
    else:  # pragma: no cover
        raise AssertionError("Expected ApiError for oversized remote media")

    assert created_paths
    assert all(not path.exists() for path in created_paths)


def test_invalid_model_is_rejected(client):
    upload = SimpleUploadedFile("sample.mp3", b"mp3-bytes", content_type="audio/mpeg")

    response = client.post(
        "/v1/audio/transcriptions",
        data={"file": upload, "model": "unknown-model"},
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_whisperkit_model_is_accepted_when_backend_is_enabled(client, monkeypatch, settings):
    settings.VOXHELM_WHISPERKIT_ENABLED = True
    settings.VOXHELM_WHISPERKIT_MODEL = "large-v3-v20240930"
    monkeypatch.setattr(
        "transcriptions.views.transcribe_audio",
        lambda audio_path, params: TranscriptionResult(
            text="Hallo Welt",
            language="de",
            segments=[TranscriptionSegment(id=0, start=0.0, end=1.0, text="Hallo Welt")],
            backend_name="whisperkit",
            model_name="large-v3-v20240930",
        ),
    )
    upload = SimpleUploadedFile("sample.mp3", b"mp3-bytes", content_type="audio/mpeg")

    response = client.post(
        "/v1/audio/transcriptions",
        data={"file": upload, "model": "whisperkit"},
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert response.status_code == 200
    assert response.json() == {"text": "Hallo Welt"}


def test_whisperkit_model_is_rejected_when_backend_is_disabled(client, settings):
    settings.VOXHELM_WHISPERKIT_ENABLED = False
    settings.VOXHELM_WHISPERKIT_MODEL = "large-v3-v20240930"
    upload = SimpleUploadedFile("sample.mp3", b"mp3-bytes", content_type="audio/mpeg")

    response = client.post(
        "/v1/audio/transcriptions",
        data={"file": upload, "model": "whisperkit"},
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert response.status_code == 400
    assert "Unsupported model 'whisperkit'" in response.json()["error"]["message"]


def test_upload_limit_is_enforced(client, settings):
    settings.VOXHELM_MAX_UPLOAD_BYTES = 4
    upload = SimpleUploadedFile("sample.mp3", b"12345", content_type="audio/mpeg")

    response = client.post(
        "/v1/audio/transcriptions",
        data={"file": upload, "model": "whisper-1"},
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert response.status_code == 400
    assert "25 MiB" in response.json()["error"]["message"] or "exceeded" in response.json()[
        "error"
    ]["message"]


@pytest.mark.django_db
def test_sync_upload_limit_remains_in_place_when_batch_staging_exists(client, settings):
    settings.VOXHELM_MAX_UPLOAD_BYTES = 4
    settings.VOXHELM_BATCH_MAX_STAGED_UPLOAD_BYTES = 8
    upload = SimpleUploadedFile("sample.mp3", b"12345", content_type="audio/mpeg")

    sync_response = client.post(
        "/v1/audio/transcriptions",
        data={"file": upload, "model": "whisper-1"},
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    staged_response = client.post(
        "/v1/uploads",
        data={"file": SimpleUploadedFile("sample.mp3", b"12345", content_type="audio/mpeg")},
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert sync_response.status_code == 400
    assert staged_response.status_code == 201


def test_speech_endpoint_requires_bearer_token(client):
    response = client.post(
        "/v1/audio/speech",
        data=json.dumps({"model": "tts-1", "input": "Hello world"}),
        content_type="application/json",
    )

    assert response.status_code == 401
    assert response.json()["error"]["type"] == "authentication_error"


def test_speech_endpoint_returns_audio(client, monkeypatch, tmp_path):
    audio_path = tmp_path / "speech.wav"
    audio_path.write_bytes(b"RIFFtest")

    monkeypatch.setattr(
        "synthesis.views.synthesize_text",
        lambda text, params: DummySpeechResult(audio_path),
    )
    monkeypatch.setattr(
        "synthesis.views.export_audio",
        lambda result, output_format: type(
            "ExportedAudio",
            (),
            {"path": result.audio_path, "format_name": output_format, "content_type": "audio/wav"},
        )(),
    )

    response = client.post(
        "/v1/audio/speech",
        data=json.dumps({"model": "tts-1", "input": "Hello world", "response_format": "wav"}),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert response.status_code == 200
    assert response["Content-Type"] == "audio/wav"
    assert response.content == b"RIFFtest"


def test_speech_endpoint_uses_non_interactive_scheduler_lane(client, monkeypatch, tmp_path):
    lanes: list[str] = []
    audio_path = tmp_path / "speech.wav"
    audio_path.write_bytes(b"RIFFtest")

    @contextmanager
    def fake_admit(lane: str):
        lanes.append(lane)
        yield object()

    monkeypatch.setattr("synthesis.service.admit_local_inference", fake_admit)
    monkeypatch.setattr(
        "synthesis.service.get_backend_service",
        lambda: type(
            "Backend",
            (),
            {
                "synthesize": lambda self, text, params: DummySpeechResult(audio_path),
            },
        )(),
    )
    monkeypatch.setattr(
        "synthesis.views.export_audio",
        lambda result, output_format: type(
            "ExportedAudio",
            (),
            {"path": result.audio_path, "format_name": output_format, "content_type": "audio/wav"},
        )(),
    )

    response = client.post(
        "/v1/audio/speech",
        data=json.dumps({"model": "tts-1", "input": "Hello world", "response_format": "wav"}),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert response.status_code == 200
    assert lanes == ["non-interactive"]


def test_speech_endpoint_rejects_oversized_input(client, settings):
    settings.VOXHELM_TTS_MAX_INPUT_CHARS = 4

    response = client.post(
        "/v1/audio/speech",
        data=json.dumps({"model": "tts-1", "input": "Hello world"}),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert response.status_code == 400
    assert "character limit" in response.json()["error"]["message"]


def test_speech_endpoint_rejects_out_of_range_speed(client):
    response = client.post(
        "/v1/audio/speech",
        data=json.dumps({"model": "tts-1", "input": "Hello world", "speed": 100}),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert response.status_code == 400
    assert "between 0.25 and 4.0" in response.json()["error"]["message"]
