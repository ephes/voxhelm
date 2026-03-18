from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from transcriptions.service import (
    BackendInvocation,
    BackendUnavailableError,
    TranscribeParams,
    TranscriptionResult,
    TranscriptionSegment,
    WhisperCppBackend,
    WhisperKitBackend,
    get_backend_services_for_model,
    normalize_interactive_transcript,
    normalize_whispercpp_payload,
    resolve_backend_name_for_model,
    resolve_model_name_for_backend,
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
    settings.VOXHELM_WHISPERKIT_MODEL = "large-v3-v20240930"

    assert resolve_backend_name_for_model("auto") == "whispercpp"
    assert resolve_backend_name_for_model("whisper-1") == "whispercpp"
    assert resolve_backend_name_for_model("gpt-4o-mini-transcribe") == "whispercpp"
    assert resolve_backend_name_for_model(settings.VOXHELM_MLX_MODEL) == "mlx"
    assert resolve_backend_name_for_model(settings.VOXHELM_WHISPERCPP_MODEL) == "whispercpp"
    assert resolve_backend_name_for_model("whisperkit") == "whisperkit"
    assert resolve_backend_name_for_model(settings.VOXHELM_WHISPERKIT_MODEL) == "whisperkit"


def test_resolve_model_name_for_backend_maps_whisperkit_alias_to_configured_model(settings) -> None:
    settings.VOXHELM_WHISPERKIT_MODEL = "large-v3-v20240930"

    assert (
        resolve_model_name_for_backend(request_model="whisperkit", backend_name="whisperkit")
        == "large-v3-v20240930"
    )
    assert (
        resolve_model_name_for_backend(
            request_model="auto",
            backend_name="whisperkit",
        )
        == "large-v3-v20240930"
    )


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
    settings.VOXHELM_WHISPERKIT_MODEL = "large-v3-v20240930"

    auto_services = get_backend_services_for_model("whisper-1")
    explicit_services = get_backend_services_for_model(settings.VOXHELM_MLX_MODEL)
    whisperkit_services = get_backend_services_for_model("whisperkit")
    settings.VOXHELM_STT_FALLBACK_BACKEND = "whispercpp"
    no_extra_services = get_backend_services_for_model("auto")
    settings.VOXHELM_STT_FALLBACK_BACKEND = ""
    empty_fallback_services = get_backend_services_for_model("auto")

    assert [service.name for service in auto_services] == ["whispercpp", "mlx"]
    assert [service.name for service in explicit_services] == ["mlx"]
    assert [service.name for service in whisperkit_services] == ["whisperkit"]
    assert [service.name for service in no_extra_services] == ["whispercpp"]
    assert [service.name for service in empty_fallback_services] == ["whispercpp"]
    assert calls == [
        ("whispercpp", "whisper-1"),
        ("mlx", "whisper-1"),
        ("mlx", settings.VOXHELM_MLX_MODEL),
        ("whisperkit", "whisperkit"),
        ("whispercpp", "auto"),
        ("whispercpp", "auto"),
    ]


def test_whisperkit_backend_normalizes_verbose_json_payload(monkeypatch) -> None:
    backend = WhisperKitBackend(
        enabled=True,
        base_url="http://127.0.0.1:50060/v1",
        model_name="large-v3-v20240930",
        timeout_seconds=900,
    )
    call_args: list[tuple[Path, str | None, str | None]] = []

    def fake_call(**kwargs) -> dict[str, object]:
        call_args.append(
            (
                kwargs["audio_path"],
                kwargs["language"],
                kwargs["prompt"],
            )
        )
        return {
            "language": "de",
            "text": "Hallo Welt",
            "segments": [
                {"id": 0, "start": 0.0, "end": 1.0, "text": "Hallo"},
                {"id": 1, "start": 1.0, "end": 2.0, "text": "Welt"},
            ],
        }

    monkeypatch.setattr("transcriptions.service.call_whisperkit_server", fake_call)

    result = backend.transcribe(
        Path("/tmp/sample.wav"),
        TranscribeParams(
            request_model="whisperkit",
            prompt="Podcast transcript",
            language="de",
        ),
    )

    assert call_args == [(Path("/tmp/sample.wav"), "de", "Podcast transcript")]
    assert result.text == "Hallo Welt"
    assert result.language == "de"
    assert result.backend_name == "whisperkit"
    assert result.model_name == "large-v3-v20240930"
    assert result.segments == [
        TranscriptionSegment(id=0, start=0.0, end=1.0, text="Hallo"),
        TranscriptionSegment(id=1, start=1.0, end=2.0, text="Welt"),
    ]


def test_whisperkit_backend_reports_unreachable_server_as_unavailable(monkeypatch) -> None:
    backend = WhisperKitBackend(
        enabled=True,
        base_url="http://127.0.0.1:50060/v1",
        model_name="large-v3-v20240930",
        timeout_seconds=900,
    )
    monkeypatch.setattr(
        "transcriptions.service.call_whisperkit_server",
        lambda **kwargs: (_ for _ in ()).throw(OSError("connection refused")),
    )

    try:
        backend.transcribe(
            Path("/tmp/sample.wav"),
            TranscribeParams(request_model="whisperkit", prompt=None, language="de"),
        )
    except BackendUnavailableError as exc:
        assert "not reachable" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected unreachable WhisperKit server to raise.")


def test_whisperkit_backend_rejects_disabled_backend() -> None:
    backend = WhisperKitBackend(
        enabled=False,
        base_url="http://127.0.0.1:50060/v1",
        model_name="large-v3-v20240930",
        timeout_seconds=900,
    )

    try:
        backend.transcribe(
            Path("/tmp/sample.wav"),
            TranscribeParams(request_model="whisperkit", prompt=None, language="de"),
        )
    except BackendUnavailableError as exc:
        assert "WhisperKit is disabled" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected disabled WhisperKit backend to raise.")


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


def test_whispercpp_backend_normalizes_input_audio_before_transcribing(
    monkeypatch, tmp_path: Path, settings
) -> None:
    backend = WhisperCppBackend(
        binary_path="/tmp/fake-whisper-cli",
        model_name="ggml-large-v3.bin",
        processors=4,
    )
    input_path = tmp_path / "sample.m4a"
    input_path.write_bytes(b"not-really-audio")
    model_path = tmp_path / "ggml-large-v3.bin"
    model_path.write_text("model", encoding="utf-8")
    settings.VOXHELM_FFMPEG_BIN = "/tmp/fake-ffmpeg"
    calls: list[list[str]] = []

    monkeypatch.setattr(
        "transcriptions.service.resolve_whispercpp_binary",
        lambda _path: "/tmp/fake-whisper-cli",
    )
    monkeypatch.setattr(
        "transcriptions.service.resolve_whispercpp_model_path",
        lambda _name: model_path,
    )

    def fake_run(args, capture_output, text, check):
        del capture_output, text, check
        call = [str(part) for part in args]
        calls.append(call)
        if call[0] == settings.VOXHELM_FFMPEG_BIN:
            Path(call[-1]).write_bytes(b"RIFFfakewav")
            return type("Completed", (), {"returncode": 0, "stderr": "", "stdout": ""})()
        output_base = Path(call[call.index("-of") + 1])
        output_base.with_suffix(".json").write_text(
            json.dumps(
                {
                    "result": {"language": "de"},
                    "transcription": [
                        {
                            "text": "Hallo Welt",
                            "timestamps": {"from": "00:00:00,000", "to": "00:00:01,000"},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return type("Completed", (), {"returncode": 0, "stderr": "", "stdout": ""})()

    monkeypatch.setattr("transcriptions.service.subprocess.run", fake_run)

    result = backend.transcribe(
        input_path,
        TranscribeParams(request_model="whisper-1", prompt=None, language="de"),
    )

    assert len(calls) == 2
    assert calls[0][:5] == [settings.VOXHELM_FFMPEG_BIN, "-y", "-i", str(input_path), "-vn"]
    assert calls[0][-1].endswith("input.wav")
    assert calls[1][0] == "/tmp/fake-whisper-cli"
    assert calls[1][calls[1].index("-f") + 1].endswith("input.wav")
    assert result.text == "Hallo Welt"
    assert result.language == "de"


def test_whispercpp_backend_surfaces_ffmpeg_normalization_failure(
    monkeypatch, tmp_path: Path, settings
) -> None:
    backend = WhisperCppBackend(
        binary_path="/tmp/fake-whisper-cli",
        model_name="ggml-large-v3.bin",
        processors=4,
    )
    input_path = tmp_path / "sample.m4a"
    input_path.write_bytes(b"not-really-audio")
    model_path = tmp_path / "ggml-large-v3.bin"
    model_path.write_text("model", encoding="utf-8")
    settings.VOXHELM_FFMPEG_BIN = "/tmp/fake-ffmpeg"

    monkeypatch.setattr(
        "transcriptions.service.resolve_whispercpp_binary",
        lambda _path: "/tmp/fake-whisper-cli",
    )
    monkeypatch.setattr(
        "transcriptions.service.resolve_whispercpp_model_path",
        lambda _name: model_path,
    )

    def fake_run(args, capture_output, text, check):
        del args, capture_output, text, check
        return type(
            "Completed",
            (),
            {"returncode": 1, "stderr": "decoder exploded", "stdout": ""},
        )()

    monkeypatch.setattr("transcriptions.service.subprocess.run", fake_run)

    try:
        backend.transcribe(
            input_path,
            TranscribeParams(request_model="whisper-1", prompt=None, language="de"),
        )
    except RuntimeError as exc:
        assert str(exc) == "ffmpeg audio normalization failed: decoder exploded"
    else:  # pragma: no cover
        raise AssertionError("Expected ffmpeg normalization failure to raise RuntimeError.")


def test_whispercpp_backend_raises_when_transcript_json_is_missing(
    monkeypatch, tmp_path: Path, settings
) -> None:
    backend = WhisperCppBackend(
        binary_path="/tmp/fake-whisper-cli",
        model_name="ggml-large-v3.bin",
        processors=4,
    )
    input_path = tmp_path / "sample.m4a"
    input_path.write_bytes(b"not-really-audio")
    model_path = tmp_path / "ggml-large-v3.bin"
    model_path.write_text("model", encoding="utf-8")
    settings.VOXHELM_FFMPEG_BIN = "/tmp/fake-ffmpeg"

    monkeypatch.setattr(
        "transcriptions.service.resolve_whispercpp_binary",
        lambda _path: "/tmp/fake-whisper-cli",
    )
    monkeypatch.setattr(
        "transcriptions.service.resolve_whispercpp_model_path",
        lambda _name: model_path,
    )

    def fake_run(args, capture_output, text, check):
        del capture_output, text, check
        call = [str(part) for part in args]
        if call[0] == settings.VOXHELM_FFMPEG_BIN:
            Path(call[-1]).write_bytes(b"RIFFfakewav")
        return type("Completed", (), {"returncode": 0, "stderr": "", "stdout": ""})()

    monkeypatch.setattr("transcriptions.service.subprocess.run", fake_run)

    try:
        backend.transcribe(
            input_path,
            TranscribeParams(request_model="whisper-1", prompt=None, language="de"),
        )
    except RuntimeError as exc:
        assert str(exc) == (
            "whisper.cpp transcription failed: transcript.json was not produced."
        )
    else:  # pragma: no cover
        raise AssertionError("Expected missing transcript.json to raise RuntimeError.")


def test_normalize_interactive_transcript_strips_german_leading_fillers() -> None:
    assert (
        normalize_interactive_transcript(
            "Okay, und wie ist denn die Temperatur im Wintergarten?",
            language="de-DE",
        )
        == "wie ist die Temperatur im Wintergarten?"
    )


def test_normalize_interactive_transcript_preserves_unknown_or_empty_only_filler() -> None:
    assert normalize_interactive_transcript("Okay.", language="de") == "Okay."
    assert normalize_interactive_transcript("bonjour salon", language="fr") == "bonjour salon"


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
