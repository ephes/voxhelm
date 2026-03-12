from __future__ import annotations

import threading
import time
from pathlib import Path

from transcriptions.service import (
    BackendInvocation,
    BackendUnavailableError,
    TranscribeParams,
    TranscriptionResult,
    TranscriptionSegment,
    get_backend_services_for_model,
    normalize_whispercpp_payload,
    resolve_backend_name_for_model,
    resolve_whispercpp_binary,
    resolve_whispercpp_model_path,
    timestamp_to_seconds,
    transcribe_audio,
)


class SerialBackend:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def transcribe(self, audio_path: Path, params: TranscribeParams) -> TranscriptionResult:
        del audio_path, params
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.05)
        with self.lock:
            self.active -= 1
        return TranscriptionResult(text="ok", language="en", segments=[])


def test_transcribe_audio_serializes_backend_access(monkeypatch) -> None:
    backend = SerialBackend()
    monkeypatch.setattr(
        "transcriptions.service.get_backend_services_for_model",
        lambda _request_model: [BackendInvocation("stub", backend)],
    )
    params = TranscribeParams(request_model="whisper-1", prompt=None, language=None)
    errors: list[Exception] = []

    def worker() -> None:
        try:
            transcribe_audio(Path("/tmp/sample.mp3"), params)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    first = threading.Thread(target=worker)
    second = threading.Thread(target=worker)
    first.start()
    second.start()
    first.join()
    second.join()

    assert errors == []
    assert backend.max_active == 1


class UnavailableBackend:
    def transcribe(self, audio_path: Path, params: TranscribeParams) -> TranscriptionResult:
        del audio_path, params
        raise BackendUnavailableError("primary unavailable")


class WorkingBackend:
    def transcribe(self, audio_path: Path, params: TranscribeParams) -> TranscriptionResult:
        del audio_path, params
        return TranscriptionResult(text="fallback", language="de", segments=[])


def test_transcribe_audio_uses_fallback_backend(monkeypatch) -> None:
    monkeypatch.setattr(
        "transcriptions.service.get_backend_services_for_model",
        lambda _request_model: [
            BackendInvocation("whispercpp", UnavailableBackend()),
            BackendInvocation("mlx", WorkingBackend()),
        ],
    )

    result = transcribe_audio(
        Path("/tmp/sample.mp3"),
        TranscribeParams(request_model="whisper-1", prompt=None, language="de"),
    )

    assert result.text == "fallback"


def test_transcribe_audio_raises_when_all_backends_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        "transcriptions.service.get_backend_services_for_model",
        lambda _request_model: [
            BackendInvocation("whispercpp", UnavailableBackend()),
            BackendInvocation("mlx", UnavailableBackend()),
        ],
    )

    try:
        transcribe_audio(
            Path("/tmp/sample.mp3"),
            TranscribeParams(request_model="whisper-1", prompt=None, language="de"),
        )
    except RuntimeError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected RuntimeError when all backends are unavailable.")

    assert "No configured STT backend is available." in message
    assert "whispercpp: primary unavailable" in message
    assert "mlx: primary unavailable" in message


def test_resolve_backend_name_for_model_uses_aliases_and_explicit_models(settings) -> None:
    settings.VOXHELM_STT_BACKEND = "whispercpp"
    settings.VOXHELM_MLX_MODEL = "mlx-community/whisper-large-v3-mlx"
    settings.VOXHELM_WHISPERCPP_MODEL = "ggml-large-v3.bin"

    assert resolve_backend_name_for_model("auto") == "whispercpp"
    assert resolve_backend_name_for_model("whisper-1") == "whispercpp"
    assert resolve_backend_name_for_model("gpt-4o-mini-transcribe") == "whispercpp"
    assert resolve_backend_name_for_model(settings.VOXHELM_MLX_MODEL) == "mlx"
    assert resolve_backend_name_for_model(settings.VOXHELM_WHISPERCPP_MODEL) == "whispercpp"


def test_get_backend_services_for_model_only_adds_fallback_for_auto_requests(
    monkeypatch, settings
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_service_for_backend_name(backend_name: str, *, request_model: str) -> str:
        calls.append((backend_name, request_model))
        return f"{backend_name}:{request_model}"

    monkeypatch.setattr(
        "transcriptions.service.service_for_backend_name",
        fake_service_for_backend_name,
    )
    settings.VOXHELM_STT_BACKEND = "whispercpp"
    settings.VOXHELM_STT_FALLBACK_BACKEND = "mlx"
    settings.VOXHELM_MLX_MODEL = "mlx-community/whisper-large-v3-mlx"
    settings.VOXHELM_WHISPERCPP_MODEL = "ggml-large-v3.bin"

    auto_services = get_backend_services_for_model("whisper-1")
    explicit_services = get_backend_services_for_model(settings.VOXHELM_MLX_MODEL)
    settings.VOXHELM_STT_FALLBACK_BACKEND = "whispercpp"
    no_extra_services = get_backend_services_for_model("auto")
    settings.VOXHELM_STT_FALLBACK_BACKEND = ""
    empty_fallback_services = get_backend_services_for_model("auto")

    assert [service.name for service in auto_services] == ["whispercpp", "mlx"]
    assert [service.name for service in explicit_services] == ["mlx"]
    assert [service.name for service in no_extra_services] == ["whispercpp"]
    assert [service.name for service in empty_fallback_services] == ["whispercpp"]
    assert calls == [
        ("whispercpp", "whisper-1"),
        ("mlx", "whisper-1"),
        ("mlx", settings.VOXHELM_MLX_MODEL),
        ("whispercpp", "auto"),
        ("whispercpp", "auto"),
    ]


def test_normalize_whispercpp_payload_builds_segments_and_language() -> None:
    payload = {
        "result": {"language": "de"},
        "transcription": [
            {
                "text": "Hallo",
                "timestamps": {"from": "00:00:01,250", "to": "00:00:02.500"},
            },
            {
                "text": "Welt",
                "timestamps": {"from": "00:00:02,500", "to": "00:00:03,750"},
            },
        ],
    }

    result = normalize_whispercpp_payload(payload, model_name="ggml-large-v3.bin")

    assert result.text == "Hallo Welt"
    assert result.language == "de"
    assert result.backend_name == "whisper.cpp"
    assert result.model_name == "ggml-large-v3.bin"
    assert result.segments == [
        TranscriptionSegment(id=0, start=1.25, end=2.5, text="Hallo"),
        TranscriptionSegment(id=1, start=2.5, end=3.75, text="Welt"),
    ]


def test_timestamp_to_seconds_accepts_dot_and_comma_formats() -> None:
    assert timestamp_to_seconds("00:01:02,345") == 62.345
    assert timestamp_to_seconds("00:01:02.345") == 62.345


def test_timestamp_to_seconds_raises_helpful_error_on_invalid_input() -> None:
    try:
        timestamp_to_seconds("bad-timestamp")
    except ValueError as exc:
        assert str(exc) == "Invalid whisper.cpp timestamp 'bad-timestamp'."
    else:  # pragma: no cover
        raise AssertionError("Expected invalid timestamp to raise ValueError.")


def test_resolve_whispercpp_binary_supports_absolute_paths(tmp_path: Path) -> None:
    binary = tmp_path / "whisper-cli"
    binary.write_text("", encoding="utf-8")

    assert resolve_whispercpp_binary(str(binary)) == str(binary)


def test_resolve_whispercpp_binary_raises_for_missing_binary(monkeypatch) -> None:
    monkeypatch.setattr("transcriptions.service.shutil.which", lambda _name: None)

    try:
        resolve_whispercpp_binary("whisper-cli")
    except BackendUnavailableError as exc:
        assert str(exc) == "whisper.cpp binary 'whisper-cli' was not found in PATH."
    else:  # pragma: no cover
        raise AssertionError("Expected missing binary to raise BackendUnavailableError.")


def test_resolve_whispercpp_model_path_prefers_cache_dir_for_relative_names(
    tmp_path: Path, settings
) -> None:
    settings.VOXHELM_MODEL_CACHE_DIR = tmp_path
    model = tmp_path / "ggml-large-v3.bin"
    model.write_text("model", encoding="utf-8")

    assert resolve_whispercpp_model_path("ggml-large-v3.bin") == model


def test_resolve_whispercpp_model_path_supports_absolute_paths(tmp_path: Path) -> None:
    model = tmp_path / "ggml-large-v3.bin"
    model.write_text("model", encoding="utf-8")

    assert resolve_whispercpp_model_path(str(model)) == model


def test_resolve_whispercpp_model_path_raises_for_missing_model(tmp_path: Path, settings) -> None:
    settings.VOXHELM_MODEL_CACHE_DIR = tmp_path

    try:
        resolve_whispercpp_model_path("ggml-large-v3.bin")
    except BackendUnavailableError as exc:
        assert "ggml-large-v3.bin" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected missing model to raise BackendUnavailableError.")
