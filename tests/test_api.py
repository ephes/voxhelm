from __future__ import annotations

import asyncio
import io
import json
import tempfile
import threading
import time
import wave
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import pytest
from django.core.asgi import get_asgi_application
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test.client import encode_multipart

from config.settings import (
    env_tokens,
    validate_disjoint_token_values,
    validate_positive_int,
    validate_remote_pull_artifact_backend,
    validate_remote_pull_s3_configuration,
    validate_remote_pull_worker_tokens,
    validate_transcription_execution_mode,
)
from synthesis.service import SynthesizeParams
from transcriptions.errors import ApiError
from transcriptions.input_media import write_upload_to_tempfile
from transcriptions.service import (
    BackendInvocation,
    InferenceCancelled,
    TranscribeParams,
    TranscriptionResult,
    TranscriptionSegment,
    transcribe_audio,
)
from transcriptions.views import parse_transcription_request


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
    def __init__(
        self,
        audio_path: Path,
        *,
        backend_name: str = "piper",
        voice_name: str = "en_US-lessac-medium",
        language: str | None = "en",
    ) -> None:
        self.audio_path = audio_path
        self.backend_name = backend_name
        self.voice_name = voice_name
        self.language = language


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
    def fake_admit(lane: str, *, cancel_event=None):
        del cancel_event
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


def test_env_tokens_rejects_empty_token_values(monkeypatch):
    monkeypatch.setenv("VOXHELM_WORKER_TOKENS", "atlas= ")

    with pytest.raises(ValueError, match="empty token value"):
        env_tokens("VOXHELM_WORKER_TOKENS")


def test_env_tokens_normalizes_json_labels_and_values(monkeypatch):
    monkeypatch.setenv("VOXHELM_WORKER_TOKENS", '{" atlas ": " atlas-token "}')

    assert env_tokens("VOXHELM_WORKER_TOKENS") == {"atlas": "atlas-token"}


def test_env_tokens_rejects_json_whitespace_token_values(monkeypatch):
    monkeypatch.setenv("VOXHELM_WORKER_TOKENS", '{"atlas": " "}')

    with pytest.raises(ValueError, match="empty token value"):
        env_tokens("VOXHELM_WORKER_TOKENS")


def test_env_tokens_rejects_json_whitespace_labels(monkeypatch):
    monkeypatch.setenv("VOXHELM_WORKER_TOKENS", '{" ": "atlas-token"}')

    with pytest.raises(ValueError, match="empty label"):
        env_tokens("VOXHELM_WORKER_TOKENS")


def test_worker_tokens_reject_producer_token_overlap():
    with pytest.raises(ValueError, match="must not share bearer token values"):
        validate_disjoint_token_values(
            "VOXHELM_BEARER_TOKENS",
            {"archive": "shared-secret"},
            "VOXHELM_WORKER_TOKENS",
            {"atlas": "shared-secret"},
        )


def test_validate_positive_int_rejects_zero_or_negative_values():
    assert validate_positive_int("VOXHELM_REMOTE_WORKER_MAX_ATTEMPTS", 1) == 1

    with pytest.raises(ValueError, match="positive integer"):
        validate_positive_int("VOXHELM_REMOTE_WORKER_MAX_ATTEMPTS", 0)

    with pytest.raises(ValueError, match="positive integer"):
        validate_positive_int("VOXHELM_REMOTE_WORKER_LEASE_SECONDS", -1)


def test_validate_transcription_execution_mode_rejects_unknown_values():
    assert validate_transcription_execution_mode("django_tasks") == "django_tasks"
    assert validate_transcription_execution_mode("remote_pull") == "remote_pull"

    with pytest.raises(ValueError, match="VOXHELM_TRANSCRIPTION_EXECUTION_MODE"):
        validate_transcription_execution_mode("remote-pull")


def test_remote_pull_requires_worker_tokens():
    with pytest.raises(ValueError, match="VOXHELM_WORKER_TOKENS"):
        validate_remote_pull_worker_tokens("remote_pull", {})

    validate_remote_pull_worker_tokens("remote_pull", {"atlas": "atlas-token"})
    validate_remote_pull_worker_tokens("django_tasks", {})


def test_remote_pull_requires_shared_artifact_backend():
    with pytest.raises(ValueError, match="remote_pull requires VOXHELM_ARTIFACT_BACKEND"):
        validate_remote_pull_artifact_backend("remote_pull", "filesystem")


def test_remote_pull_allows_s3_artifact_backend():
    validate_remote_pull_artifact_backend("remote_pull", "s3")
    validate_remote_pull_artifact_backend("django_tasks", "filesystem")


def test_remote_pull_requires_complete_s3_artifact_configuration():
    with pytest.raises(ValueError, match="requires complete S3 artifact configuration"):
        validate_remote_pull_s3_configuration(
            "remote_pull",
            "s3",
            {
                "VOXHELM_ARTIFACT_S3_ENDPOINT_URL": "",
                "VOXHELM_ARTIFACT_S3_ACCESS_KEY_ID": "access",
                "VOXHELM_ARTIFACT_S3_SECRET_ACCESS_KEY": "",
                "VOXHELM_ARTIFACT_BUCKET": "voxhelm",
            },
        )


def test_remote_pull_s3_configuration_allows_non_remote_modes():
    validate_remote_pull_s3_configuration(
        "django_tasks",
        "filesystem",
        {
            "VOXHELM_ARTIFACT_S3_ENDPOINT_URL": "",
            "VOXHELM_ARTIFACT_S3_ACCESS_KEY_ID": "",
        },
    )


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
        lambda *args, **kwargs: type(
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


def _stub_export_audio(monkeypatch) -> None:
    monkeypatch.setattr(
        "synthesis.views.export_audio",
        lambda result, output_format: type(
            "ExportedAudio",
            (),
            {"path": result.audio_path, "format_name": output_format, "content_type": "audio/wav"},
        )(),
    )


def test_speech_endpoint_sets_voxhelm_headers(client, monkeypatch, tmp_path):
    audio_path = tmp_path / "speech.wav"
    audio_path.write_bytes(b"RIFFtest")

    monkeypatch.setattr(
        "synthesis.views.synthesize_text",
        lambda text, params: DummySpeechResult(
            audio_path, backend_name="kokoro", voice_name="kokoro-martin", language="de"
        ),
    )
    _stub_export_audio(monkeypatch)

    response = client.post(
        "/v1/audio/speech",
        data=json.dumps({"model": "auto", "input": "Ein hinreichend langer Satz."}),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert response.status_code == 200
    assert response["X-Voxhelm-Backend"] == "kokoro"
    assert response["X-Voxhelm-Voice"] == "kokoro-martin"
    assert response["X-Voxhelm-Language"] == "de"


def test_speech_endpoint_routing_field_reaches_params(client, monkeypatch, tmp_path):
    audio_path = tmp_path / "speech.wav"
    audio_path.write_bytes(b"RIFFtest")
    captured: list[SynthesizeParams] = []

    def fake_synthesize(text: str, params: SynthesizeParams) -> DummySpeechResult:
        captured.append(params)
        # The endpoint unlinks the audio file in its finally block, so recreate it
        # for each request (this helper serves two POSTs).
        audio_path.write_bytes(b"RIFFtest")
        return DummySpeechResult(audio_path)

    monkeypatch.setattr("synthesis.views.synthesize_text", fake_synthesize)
    _stub_export_audio(monkeypatch)

    bypass = client.post(
        "/v1/audio/speech",
        data=json.dumps(
            {
                "model": "kokoro",
                "input": "Force this voice.",
                "voice": "kokoro-martin",
                "routing": False,
            }
        ),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )
    default = client.post(
        "/v1/audio/speech",
        data=json.dumps({"model": "auto", "input": "Route this text please."}),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert bypass.status_code == 200
    assert default.status_code == 200
    assert captured[0].routing is False
    # Omitting the field defaults to routing enabled for the request.
    assert captured[1].routing is True


def test_speech_endpoint_rejects_non_boolean_routing(client):
    response = client.post(
        "/v1/audio/speech",
        data=json.dumps({"model": "tts-1", "input": "Hello world", "routing": "yes"}),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert response.status_code == 400
    assert "must be a boolean" in response.json()["error"]["message"]


def test_language_routing_requires_extra_at_startup(monkeypatch):
    import importlib.util

    from config.settings import validate_language_routing_dependencies

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ValueError, match="'routing' extra is not installed"):
        validate_language_routing_dependencies(True)


def test_language_routing_disabled_skips_dependency_check(monkeypatch):
    import importlib.util

    from config.settings import validate_language_routing_dependencies

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    # Disabled routing must never trip the startup dependency check.
    validate_language_routing_dependencies(False)


MULTIPART_BOUNDARY = "VoxhelmAsgiBoundary"


def _multipart_upload_body(*, model: str = "whisper-1") -> tuple[bytes, str]:
    upload = SimpleUploadedFile("sample.wav", wav_bytes(), content_type="audio/wav")
    body = encode_multipart(MULTIPART_BOUNDARY, {"model": model, "file": upload})
    return body, f"multipart/form-data; boundary={MULTIPART_BOUNDARY}"


class ScriptedConnection:
    """ASGI receive/send pair that delivers the body, then a disconnect on demand."""

    def __init__(self, body: bytes) -> None:
        self.body = body
        self.disconnect = asyncio.Event()
        self.disconnect_delivered = threading.Event()
        self.sent: list[Mapping[str, Any]] = []
        self._body_delivered = False

    async def receive(self) -> dict[str, Any]:
        if not self._body_delivered:
            self._body_delivered = True
            return {"type": "http.request", "body": self.body, "more_body": False}
        await self.disconnect.wait()
        self.disconnect_delivered.set()
        return {"type": "http.disconnect"}

    async def send(self, message: Mapping[str, Any]) -> None:
        self.sent.append(message)


def _asgi_scope(*, body: bytes, content_type: str) -> dict[str, Any]:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/audio/transcriptions",
        "raw_path": b"/v1/audio/transcriptions",
        "root_path": "",
        "query_string": b"",
        "headers": [
            (b"host", b"testserver"),
            (b"authorization", b"Bearer test-token"),
            (b"content-type", content_type.encode("ascii")),
            (b"content-length", str(len(body)).encode("ascii")),
        ],
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
    }


async def _await_flag(flag: threading.Event, *, timeout: float = 10.0) -> bool:
    """Wait for a worker-thread event without occupying an executor thread."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if flag.is_set():
            return True
        await asyncio.sleep(0.01)
    return flag.is_set()


async def _settle_delivered_disconnect(passes: int = 100) -> None:
    """Run the loop until a delivered ``http.disconnect`` has cancelled the view.

    Everything between ``receive()`` returning the disconnect and the view
    resuming with ``CancelledError`` happens on this event loop with no thread
    and no I/O in between, so draining the ready queue is enough. Waiting for
    the view to *enter* a phase needs a real signal instead, because that side
    is thread-backed.
    """
    for _ in range(passes):
        await asyncio.sleep(0)


def _wait_until_gone(path: Path, *, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not path.exists():
            return True
        time.sleep(0.02)
    return not path.exists()


def test_asgi_disconnect_during_inference_cancels_and_cleans_up(monkeypatch):
    running = threading.Event()
    observed: dict[str, Any] = {}

    def fake_transcribe_audio(audio_path: Path, params: TranscribeParams):
        observed["audio_path"] = audio_path
        observed["cancel_event"] = params.cancel_event
        running.set()
        assert params.cancel_event is not None
        params.cancel_event.wait(timeout=10)
        raise InferenceCancelled("cancelled by the client")

    monkeypatch.setattr("transcriptions.views.transcribe_audio", fake_transcribe_audio)
    body, content_type = _multipart_upload_body()
    connection = ScriptedConnection(body)

    async def scenario() -> None:
        application = get_asgi_application()
        call = asyncio.ensure_future(
            application(
                _asgi_scope(body=body, content_type=content_type),
                connection.receive,
                connection.send,
            )
        )
        assert await _await_flag(running)
        connection.disconnect.set()
        await asyncio.wait_for(call, timeout=10)

    asyncio.run(scenario())

    cancel_event = cast(threading.Event, observed["cancel_event"])
    assert cancel_event.is_set()
    assert connection.sent == []
    assert _wait_until_gone(cast(Path, observed["audio_path"]))


def test_asgi_disconnect_during_parsing_finishes_the_body_read(monkeypatch):
    parser_entered = threading.Event()
    release = threading.Event()
    parsed_paths: list[Path] = []
    existed_after_parse: list[bool] = []
    backend_calls: list[Path] = []
    real_parse = parse_transcription_request

    def blocking_parse(request):
        parser_entered.set()
        # Only read the ASGI body stream once the disconnect has been delivered
        # and the view has already been cancelled.
        assert release.wait(timeout=10)
        parsed = real_parse(request)
        parsed_paths.append(parsed.input_path)
        existed_after_parse.append(parsed.input_path.exists())
        return parsed

    def unexpected_transcribe(audio_path: Path, params: TranscribeParams):
        del params
        backend_calls.append(audio_path)
        return TranscriptionResult(text="", language=None, segments=[])

    monkeypatch.setattr("transcriptions.views.parse_transcription_request", blocking_parse)
    monkeypatch.setattr("transcriptions.views.transcribe_audio", unexpected_transcribe)
    body, content_type = _multipart_upload_body()
    connection = ScriptedConnection(body)

    async def scenario() -> None:
        application = get_asgi_application()
        call = asyncio.ensure_future(
            application(
                _asgi_scope(body=body, content_type=content_type),
                connection.receive,
                connection.send,
            )
        )
        # The parse phase is thread-backed, so wait for the worker to report it
        # rather than assuming a number of loop turns.
        assert await _await_flag(parser_entered)
        connection.disconnect.set()
        assert await _await_flag(connection.disconnect_delivered)
        # Hand the cancellation all the way through to the view before the body
        # is touched, so the parse provably reads an already-cancelled request.
        await _settle_delivered_disconnect()
        release.set()
        await asyncio.wait_for(call, timeout=10)

    asyncio.run(scenario())

    # The real parser ran after the disconnect and still produced a temp file,
    # which proves the request body was readable throughout parsing.
    assert existed_after_parse == [True]
    assert backend_calls == []
    assert connection.sent == []
    assert _wait_until_gone(parsed_paths[0])


def test_asgi_disconnect_during_parsing_survives_a_parse_error(monkeypatch):
    parser_entered = threading.Event()
    release = threading.Event()
    parse_calls: list[bool] = []
    backend_calls: list[Path] = []

    def failing_parse(request):
        del request
        parser_entered.set()
        assert release.wait(timeout=10)
        parse_calls.append(True)
        raise ApiError("Unsupported uploaded media type for transcription.")

    def unexpected_transcribe(audio_path: Path, params: TranscribeParams):
        del params
        backend_calls.append(audio_path)
        return TranscriptionResult(text="", language=None, segments=[])

    monkeypatch.setattr("transcriptions.views.parse_transcription_request", failing_parse)
    monkeypatch.setattr("transcriptions.views.transcribe_audio", unexpected_transcribe)
    body, content_type = _multipart_upload_body()
    connection = ScriptedConnection(body)

    async def scenario() -> None:
        application = get_asgi_application()
        call = asyncio.ensure_future(
            application(
                _asgi_scope(body=body, content_type=content_type),
                connection.receive,
                connection.send,
            )
        )
        assert await _await_flag(parser_entered)
        connection.disconnect.set()
        assert await _await_flag(connection.disconnect_delivered)
        await _settle_delivered_disconnect()
        release.set()
        await asyncio.wait_for(call, timeout=10)

    asyncio.run(scenario())

    # The parser really ran and really failed; the view swallowed neither.
    assert parse_calls == [True]
    assert backend_calls == []
    assert connection.sent == []


class RecordingExecutor(ThreadPoolExecutor):
    """Single-worker executor that reports every job handed to it."""

    def __init__(self) -> None:
        super().__init__(max_workers=1)
        self._counter_lock = threading.Lock()
        self.submissions = 0

    def submit(self, fn, /, *args, **kwargs):
        with self._counter_lock:
            self.submissions += 1
        return super().submit(fn, *args, **kwargs)


def test_asgi_disconnect_with_saturated_executor_cancels_before_the_backend(monkeypatch):
    blocker_started = threading.Event()
    blocker_release = threading.Event()
    cancelled_in_worker = threading.Event()
    produced: dict[str, Any] = {}
    raised: list[BaseException] = []
    backend_calls: list[Path] = []
    executor = RecordingExecutor()

    class NeverCalledBackend:
        def transcribe(self, audio_path: Path, params: TranscribeParams):
            del params
            backend_calls.append(audio_path)
            return TranscriptionResult(text="", language=None, segments=[])

    def blocker() -> None:
        blocker_started.set()
        blocker_release.wait(timeout=10)

    def parse_then_saturate(chunks, *, suffix: str) -> Path:
        path = write_upload_to_tempfile(chunks, suffix=suffix)
        produced["path"] = path
        # Occupy the one worker so the inference phase can only be queued.
        executor.submit(blocker)
        return path

    def recording_transcribe(audio_path: Path, params: TranscribeParams):
        try:
            return transcribe_audio(audio_path, params)
        except InferenceCancelled as exc:
            raised.append(exc)
            cancelled_in_worker.set()
            raise
        except BaseException as exc:  # pragma: no cover - defensive
            raised.append(exc)
            cancelled_in_worker.set()
            raise

    monkeypatch.setattr("transcriptions.views.write_upload_to_tempfile", parse_then_saturate)
    monkeypatch.setattr("transcriptions.views.transcribe_audio", recording_transcribe)
    monkeypatch.setattr(
        "transcriptions.service.get_backend_services_for_model",
        lambda _request_model: [BackendInvocation("stub", NeverCalledBackend())],
    )
    body, content_type = _multipart_upload_body()
    connection = ScriptedConnection(body)

    async def wait_for_submissions(count: int, *, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if executor.submissions >= count:
                return True
            await asyncio.sleep(0.01)
        return executor.submissions >= count

    async def scenario() -> None:
        asyncio.get_running_loop().set_default_executor(executor)
        application = get_asgi_application()
        call = asyncio.ensure_future(
            application(
                _asgi_scope(body=body, content_type=content_type),
                connection.receive,
                connection.send,
            )
        )
        # Submissions in order: the parse phase, the blocker, the inference phase.
        assert await _await_flag(blocker_started)
        assert await wait_for_submissions(3)
        connection.disconnect.set()
        await asyncio.wait_for(call, timeout=10)
        blocker_release.set()
        # Stay inside the loop until the queued worker has finished cleaning up,
        # otherwise loop shutdown would cancel it before it ever ran.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if cancelled_in_worker.is_set() and not cast(Path, produced["path"]).exists():
                break
            await asyncio.sleep(0.01)

    try:
        asyncio.run(scenario())
    finally:
        blocker_release.set()
        executor.shutdown(wait=True)

    assert backend_calls == []
    assert len(raised) == 1
    assert isinstance(raised[0], InferenceCancelled)
    assert connection.sent == []
    assert not cast(Path, produced["path"]).exists()


def test_json_request_rejects_invalid_model_before_downloading(client, monkeypatch, settings):
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}
    downloads: list[str] = []

    def fake_download(*, source_url: str):
        downloads.append(source_url)
        path = Path(settings.BASE_DIR) / "tmp-test-invalid-model.mp3"
        path.write_bytes(b"mp3-bytes")
        return path

    monkeypatch.setattr("transcriptions.views.download_allowed_url_to_tempfile", fake_download)

    response = client.post(
        "/v1/audio/transcriptions",
        data=json.dumps(
            {"url": "https://media.example.com/episode.mp3", "model": "not-a-model"}
        ),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert response.status_code == 400
    assert "Unsupported model 'not-a-model'." in response.json()["error"]["message"]
    assert downloads == []
