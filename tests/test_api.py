from __future__ import annotations

import json
import tempfile
from pathlib import Path

from django.core.files.uploadedfile import SimpleUploadedFile

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

    monkeypatch.setattr("transcriptions.views.urlopen", lambda request, timeout: DummyResponse())
    monkeypatch.setattr(
        "transcriptions.views.tempfile.NamedTemporaryFile",
        fake_named_temporary_file,
    )

    from transcriptions.views import ApiError, download_allowed_url_to_tempfile

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
