from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from transcriptions.service import (
    BackendInvocation,
    BackendUnavailableError,
    InferenceCancelled,
    MlxWhisperBackend,
    TranscribeParams,
    TranscriptionResult,
    TranscriptionSegment,
    WhisperCppBackend,
    WhisperKitBackend,
    _normalize_audio_for_whispercpp,
    build_backend_service,
    get_backend_services_for_model,
    normalize_interactive_transcript,
    normalize_transcription_payload,
    normalize_whispercpp_payload,
    resolve_backend_name_for_model,
    resolve_model_name_for_backend,
    resolve_whispercpp_binary,
    resolve_whispercpp_model_path,
    run_cancellable_process,
    timestamp_to_seconds,
    transcribe_audio,
)


class OverlappingBackend:
    """Backend that only returns once a second concurrent call has arrived."""

    def __init__(self) -> None:
        self.barrier = threading.Barrier(2, timeout=2)

    def transcribe(self, audio_path: Path, params: TranscribeParams) -> TranscriptionResult:
        del audio_path, params
        self.barrier.wait()
        return TranscriptionResult(text="ok", language="en", segments=[])


def _run_in_threads(target, count: int = 2) -> list[BaseException]:
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            target()
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert all(not thread.is_alive() for thread in threads)
    return errors


def test_transcribe_audio_allows_subprocess_backends_to_overlap(monkeypatch) -> None:
    backend = OverlappingBackend()
    monkeypatch.setattr(
        "transcriptions.service.get_backend_services_for_model",
        lambda _request_model: [BackendInvocation("stub", backend)],
    )
    params = TranscribeParams(request_model="whisper-1", prompt=None, language=None)

    errors = _run_in_threads(lambda: transcribe_audio(Path("/tmp/sample.mp3"), params))

    # The barrier only clears when both calls are inside the backend at once.
    assert errors == []


def test_mlx_backend_serializes_concurrent_calls(monkeypatch) -> None:
    state = {"active": 0, "max_active": 0}
    state_lock = threading.Lock()

    class _FakeMlxWhisper:
        @staticmethod
        def transcribe(_audio, **kwargs):
            del kwargs
            with state_lock:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            time.sleep(0.05)
            with state_lock:
                state["active"] -= 1
            return {"text": "Hallo", "language": "de", "segments": []}

    monkeypatch.setitem(sys.modules, "mlx_whisper", _FakeMlxWhisper)
    backend = MlxWhisperBackend(model_name="mlx-community/whisper-large-v3-mlx")
    params = TranscribeParams(request_model="whisper-1", prompt=None, language="de")

    errors = _run_in_threads(lambda: backend.transcribe(Path("/tmp/sample.wav"), params))

    assert errors == []
    assert state["max_active"] == 1


class UnavailableBackend:
    def transcribe(self, audio_path: Path, params: TranscribeParams) -> TranscriptionResult:
        del audio_path, params
        raise BackendUnavailableError("primary unavailable")


class WorkingBackend:
    def transcribe(self, audio_path: Path, params: TranscribeParams) -> TranscriptionResult:
        del audio_path, params
        return TranscriptionResult(text="fallback", language="de", segments=[])


def test_normalize_transcription_payload_drops_backend_speaker_labels() -> None:
    result = normalize_transcription_payload(
        {
            "text": "Hello",
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 1.0,
                    "text": "Hello",
                    "speaker": "SPEAKER_00",
                }
            ],
        }
    )

    assert result.segments == [TranscriptionSegment(id=0, start=0.0, end=1.0, text="Hello")]


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


def test_transcribe_audio_sanitizes_repeated_loop_segments(monkeypatch) -> None:
    loop = [
        TranscriptionSegment(
            id=index, start=float(index), end=float(index + 1), text="Das ist auch sehr subjektiv."
        )
        for index in range(18)
    ]

    class LoopBackend:
        def transcribe(self, audio_path: Path, params: TranscribeParams) -> TranscriptionResult:
            del audio_path, params
            return TranscriptionResult(
                text=" ".join(segment.text for segment in loop),
                language="de",
                segments=loop,
            )

    monkeypatch.setattr(
        "transcriptions.service.get_backend_services_for_model",
        lambda _request_model: [BackendInvocation("stub", LoopBackend())],
    )

    result = transcribe_audio(
        Path("/tmp/sample.mp3"),
        TranscribeParams(request_model="whisper-1", prompt=None, language="de"),
    )

    assert len(result.segments) == 1
    assert result.segments[0].text == "Das ist auch sehr subjektiv."
    assert result.text == "Das ist auch sehr subjektiv."


def test_transcribe_audio_respects_disabled_sanitizer(monkeypatch, settings) -> None:
    settings.VOXHELM_SANITIZE_TRANSCRIPT = False
    loop = [
        TranscriptionSegment(
            id=index, start=float(index), end=float(index + 1), text="Das ist auch sehr subjektiv."
        )
        for index in range(18)
    ]

    class LoopBackend:
        def transcribe(self, audio_path: Path, params: TranscribeParams) -> TranscriptionResult:
            del audio_path, params
            return TranscriptionResult(text="", language="de", segments=list(loop))

    monkeypatch.setattr(
        "transcriptions.service.get_backend_services_for_model",
        lambda _request_model: [BackendInvocation("stub", LoopBackend())],
    )

    result = transcribe_audio(
        Path("/tmp/sample.mp3"),
        TranscribeParams(request_model="whisper-1", prompt=None, language="de"),
    )

    assert len(result.segments) == 18


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

    def fake_run(args, **kwargs):
        assert kwargs["encoding"] == "utf-8"
        assert kwargs["errors"] == "replace"
        call = [str(part) for part in args]
        calls.append(call)
        assert call[0] == settings.VOXHELM_FFMPEG_BIN
        Path(call[-1]).write_bytes(b"RIFFfakewav")
        return type("Completed", (), {"returncode": 0, "stderr": "", "stdout": ""})()

    def fake_whisper_cli(args, *, cancel_event, **kwargs):
        del kwargs
        assert cancel_event is None
        call = [str(part) for part in args]
        calls.append(call)
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
        return subprocess.CompletedProcess(call, 0, "", "")

    monkeypatch.setattr("transcriptions.service.subprocess.run", fake_run)
    monkeypatch.setattr("transcriptions.service.run_cancellable_process", fake_whisper_cli)

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


def _capture_whispercpp_args(
    backend: WhisperCppBackend, *, monkeypatch, tmp_path: Path, settings
) -> list[str]:
    """Run a fully faked whisper.cpp transcription and return the whisper-cli argv."""
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

    def fake_run(args, **kwargs):
        del kwargs
        call = [str(part) for part in args]
        calls.append(call)
        Path(call[-1]).write_bytes(b"RIFFfakewav")
        return type("Completed", (), {"returncode": 0, "stderr": "", "stdout": ""})()

    def fake_whisper_cli(args, **kwargs):
        del kwargs
        call = [str(part) for part in args]
        calls.append(call)
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
        return subprocess.CompletedProcess(call, 0, "", "")

    monkeypatch.setattr("transcriptions.service.subprocess.run", fake_run)
    monkeypatch.setattr("transcriptions.service.run_cancellable_process", fake_whisper_cli)
    backend.transcribe(
        input_path,
        TranscribeParams(request_model="whisper-1", prompt=None, language="de"),
    )
    return calls[1]


def test_whispercpp_backend_emits_anti_hallucination_flags_by_default(
    monkeypatch, tmp_path: Path, settings
) -> None:
    backend = WhisperCppBackend(
        binary_path="/tmp/fake-whisper-cli",
        model_name="ggml-large-v3.bin",
        processors=4,
    )
    args = _capture_whispercpp_args(
        backend, monkeypatch=monkeypatch, tmp_path=tmp_path, settings=settings
    )
    # max-context 0 disables conditioning on previous text (the loop trigger).
    assert args[args.index("--max-context") + 1] == "0"
    assert "--suppress-nst" in args


def test_whispercpp_backend_honors_context_and_nst_settings(
    monkeypatch, tmp_path: Path, settings
) -> None:
    backend = WhisperCppBackend(
        binary_path="/tmp/fake-whisper-cli",
        model_name="ggml-large-v3.bin",
        processors=4,
        max_context=-1,
        suppress_nst=False,
    )
    args = _capture_whispercpp_args(
        backend, monkeypatch=monkeypatch, tmp_path=tmp_path, settings=settings
    )
    assert args[args.index("--max-context") + 1] == "-1"
    assert "--suppress-nst" not in args


def test_mlx_backend_disables_condition_on_previous_text_by_default(monkeypatch) -> None:
    captured: list[dict[str, object]] = []

    class _FakeMlxWhisper:
        @staticmethod
        def transcribe(_audio, **kwargs):
            captured.append(kwargs)
            return {"text": "Hallo", "language": "de", "segments": []}

    monkeypatch.setitem(sys.modules, "mlx_whisper", _FakeMlxWhisper)

    MlxWhisperBackend(model_name="mlx-community/whisper-large-v3-mlx").transcribe(
        Path("/tmp/sample.wav"),
        TranscribeParams(request_model="whisper-1", prompt=None, language="de"),
    )
    MlxWhisperBackend(
        model_name="mlx-community/whisper-large-v3-mlx",
        condition_on_previous_text=True,
    ).transcribe(
        Path("/tmp/sample.wav"),
        TranscribeParams(request_model="whisper-1", prompt=None, language="de"),
    )

    assert captured[0]["condition_on_previous_text"] is False
    assert captured[1]["condition_on_previous_text"] is True


def test_build_backend_service_wires_anti_hallucination_settings(settings) -> None:
    settings.VOXHELM_WHISPERCPP_MAX_CONTEXT = 7
    settings.VOXHELM_WHISPERCPP_SUPPRESS_NST = False
    settings.VOXHELM_MLX_CONDITION_ON_PREVIOUS_TEXT = True

    cpp = build_backend_service(backend_name="whispercpp", model_name="ggml-large-v3.bin")
    assert isinstance(cpp, WhisperCppBackend)
    assert cpp.max_context == 7
    assert cpp.suppress_nst is False

    mlx = build_backend_service(backend_name="mlx", model_name="mlx-community/whisper-large-v3-mlx")
    assert isinstance(mlx, MlxWhisperBackend)
    assert mlx.condition_on_previous_text is True


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

    def fake_run(args, **kwargs):
        del args, kwargs
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


def test_whispercpp_audio_normalization_handles_non_utf8_ffmpeg_stderr(
    tmp_path: Path, settings
) -> None:
    fake_ffmpeg = tmp_path / "fake-ffmpeg.py"
    fake_ffmpeg.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "sys.stderr.buffer.write(b'bad byte: \\xf0')\n"
        "raise SystemExit(1)\n",
        encoding="utf-8",
    )
    fake_ffmpeg.chmod(0o755)
    input_path = tmp_path / "sample.mp3"
    input_path.write_bytes(b"not-really-audio")
    output_path = tmp_path / "normalized.wav"
    settings.VOXHELM_FFMPEG_BIN = str(fake_ffmpeg)

    try:
        _normalize_audio_for_whispercpp(input_path=input_path, output_path=output_path)
    except RuntimeError as exc:
        assert str(exc) == "ffmpeg audio normalization failed: bad byte: �"
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

    def fake_run(args, **kwargs):
        del kwargs
        call = [str(part) for part in args]
        assert call[0] == settings.VOXHELM_FFMPEG_BIN
        Path(call[-1]).write_bytes(b"RIFFfakewav")
        return type("Completed", (), {"returncode": 0, "stderr": "", "stdout": ""})()

    monkeypatch.setattr("transcriptions.service.subprocess.run", fake_run)
    monkeypatch.setattr(
        "transcriptions.service.run_cancellable_process",
        lambda args, **kwargs: subprocess.CompletedProcess([str(part) for part in args], 0, "", ""),
    )

    try:
        backend.transcribe(
            input_path,
            TranscribeParams(request_model="whisper-1", prompt=None, language="de"),
        )
    except RuntimeError as exc:
        assert str(exc) == ("whisper.cpp transcription failed: transcript.json was not produced.")
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


def _record_popen(monkeypatch) -> list[subprocess.Popen[str]]:
    """Capture every child started by run_cancellable_process for reaping assertions."""
    created: list[subprocess.Popen[str]] = []
    real_popen = subprocess.Popen

    def factory(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        created.append(process)
        return process

    monkeypatch.setattr("transcriptions.service.subprocess.Popen", factory)
    return created


def test_run_cancellable_process_returns_output_without_cancel_event() -> None:
    completed = run_cancellable_process(
        [sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr)"],
        cancel_event=None,
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == "out"
    assert completed.stderr.strip() == "err"


def _cancel_once_ready(ready_path: Path, cancel_event: threading.Event) -> threading.Thread:
    """Set ``cancel_event`` only after the child reported that it is running."""

    def watcher() -> None:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not ready_path.exists():
            time.sleep(0.01)
        cancel_event.set()

    thread = threading.Thread(target=watcher, daemon=True)
    thread.start()
    return thread


def _assert_pipes_closed(process: subprocess.Popen[str]) -> None:
    assert process.stdout is not None and process.stdout.closed
    assert process.stderr is not None and process.stderr.closed


def test_run_cancellable_process_terminates_sleeping_child(monkeypatch, tmp_path: Path) -> None:
    created = _record_popen(monkeypatch)
    ready_path = tmp_path / "ready"
    cancel_event = threading.Event()
    watcher = _cancel_once_ready(ready_path, cancel_event)
    started_at = time.monotonic()

    with pytest.raises(InferenceCancelled):
        run_cancellable_process(
            [
                sys.executable,
                "-c",
                "import pathlib, sys, time; "
                "pathlib.Path(sys.argv[1]).write_text('ready'); time.sleep(30)",
                str(ready_path),
            ],
            cancel_event=cancel_event,
            poll_seconds=0.1,
        )
    watcher.join(timeout=5)

    assert time.monotonic() - started_at < 5.0
    assert created[0].returncode == -signal.SIGTERM
    _assert_pipes_closed(created[0])


def test_run_cancellable_process_escalates_to_sigkill(monkeypatch, tmp_path: Path) -> None:
    created = _record_popen(monkeypatch)
    ready_path = tmp_path / "ready"
    cancel_event = threading.Event()
    watcher = _cancel_once_ready(ready_path, cancel_event)
    started_at = time.monotonic()

    with pytest.raises(InferenceCancelled):
        run_cancellable_process(
            [
                sys.executable,
                "-c",
                "import pathlib, signal, sys, time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "pathlib.Path(sys.argv[1]).write_text('ready'); time.sleep(30)",
                str(ready_path),
            ],
            cancel_event=cancel_event,
            poll_seconds=0.1,
            terminate_grace_seconds=0.5,
        )
    watcher.join(timeout=5)

    assert time.monotonic() - started_at < 5.0
    # The handler ignored SIGTERM, so only the SIGKILL escalation can have ended it.
    assert created[0].returncode == -signal.SIGKILL
    _assert_pipes_closed(created[0])


def test_run_cancellable_process_reaps_and_closes_pipes_when_communicate_fails(
    monkeypatch,
) -> None:
    created: list[subprocess.Popen[str]] = []
    real_popen = subprocess.Popen

    def factory(*args, **kwargs):
        process = real_popen(*args, **kwargs)

        def failing_communicate(*_args, **_kwargs):
            raise OSError("communicate exploded")

        monkeypatch.setattr(process, "communicate", failing_communicate)
        created.append(process)
        return process

    monkeypatch.setattr("transcriptions.service.subprocess.Popen", factory)

    with pytest.raises(OSError, match="communicate exploded"):
        run_cancellable_process(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cancel_event=None,
        )

    assert created[0].returncode == -signal.SIGKILL
    _assert_pipes_closed(created[0])


def test_run_cancellable_process_survives_cancel_racing_child_exit(monkeypatch) -> None:
    created = _record_popen(monkeypatch)
    cancel_event = threading.Event()
    timer = threading.Timer(0.2, cancel_event.set)
    timer.start()
    started_at = time.monotonic()

    try:
        try:
            completed = run_cancellable_process(
                [sys.executable, "-c", "import time; time.sleep(0.2)"],
                cancel_event=cancel_event,
                poll_seconds=0.05,
            )
        except InferenceCancelled:
            completed = None
    finally:
        timer.cancel()

    assert time.monotonic() - started_at < 3.0
    assert created[0].returncode is not None
    if completed is not None:
        assert completed.returncode == 0


class CountingUnavailableBackend:
    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, audio_path: Path, params: TranscribeParams) -> TranscriptionResult:
        del audio_path, params
        self.calls += 1
        raise BackendUnavailableError("primary unavailable")


class CancellingBackend:
    def __init__(self, cancel_event: threading.Event) -> None:
        self.cancel_event = cancel_event
        self.calls = 0

    def transcribe(self, audio_path: Path, params: TranscribeParams) -> TranscriptionResult:
        del audio_path, params
        self.calls += 1
        self.cancel_event.set()
        raise BackendUnavailableError("primary unavailable")


class CountingWorkingBackend:
    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, audio_path: Path, params: TranscribeParams) -> TranscriptionResult:
        del audio_path, params
        self.calls += 1
        return TranscriptionResult(text="fallback", language="de", segments=[])


def test_transcribe_audio_skips_all_backends_when_cancelled_before_start(monkeypatch) -> None:
    primary = CountingUnavailableBackend()
    fallback = CountingWorkingBackend()
    monkeypatch.setattr(
        "transcriptions.service.get_backend_services_for_model",
        lambda _request_model: [
            BackendInvocation("whispercpp", primary),
            BackendInvocation("mlx", fallback),
        ],
    )
    cancel_event = threading.Event()
    cancel_event.set()

    with pytest.raises(InferenceCancelled):
        transcribe_audio(
            Path("/tmp/sample.mp3"),
            TranscribeParams(
                request_model="whisper-1",
                prompt=None,
                language=None,
                cancel_event=cancel_event,
            ),
        )

    assert primary.calls == 0
    assert fallback.calls == 0


def test_transcribe_audio_does_not_fall_back_after_cancel_during_primary(monkeypatch) -> None:
    cancel_event = threading.Event()
    primary = CancellingBackend(cancel_event)
    fallback = CountingWorkingBackend()
    monkeypatch.setattr(
        "transcriptions.service.get_backend_services_for_model",
        lambda _request_model: [
            BackendInvocation("whispercpp", primary),
            BackendInvocation("mlx", fallback),
        ],
    )

    with pytest.raises(InferenceCancelled):
        transcribe_audio(
            Path("/tmp/sample.mp3"),
            TranscribeParams(
                request_model="whisper-1",
                prompt=None,
                language=None,
                cancel_event=cancel_event,
            ),
        )

    assert primary.calls == 1
    assert fallback.calls == 0


def test_transcribe_audio_releases_scheduler_slot_when_cancelled(
    monkeypatch, tmp_path: Path, settings
) -> None:
    settings.VOXHELM_LANE_SCHEDULER_ENABLED = True
    settings.VOXHELM_LANE_SCHEDULER_DIR = tmp_path
    settings.VOXHELM_LANE_SCHEDULER_STALE_SECONDS = 60
    settings.VOXHELM_LANE_SCHEDULER_INTERACTIVE_SLOTS = 1
    settings.VOXHELM_LANE_SCHEDULER_NON_INTERACTIVE_SLOTS = 1
    cancel_event = threading.Event()
    primary = CancellingBackend(cancel_event)
    monkeypatch.setattr(
        "transcriptions.service.get_backend_services_for_model",
        lambda _request_model: [BackendInvocation("whispercpp", primary)],
    )

    with pytest.raises(InferenceCancelled):
        transcribe_audio(
            Path("/tmp/sample.mp3"),
            TranscribeParams(
                request_model="whisper-1",
                prompt=None,
                language=None,
                cancel_event=cancel_event,
            ),
        )

    holders_dir = tmp_path / "holders"
    assert not holders_dir.exists() or list(holders_dir.iterdir()) == []


def test_mlx_backend_cancels_caller_waiting_for_the_lock(monkeypatch) -> None:
    entered = threading.Event()
    release = threading.Event()
    calls: list[dict[str, object]] = []

    class _FakeMlxWhisper:
        @staticmethod
        def transcribe(_audio, **kwargs):
            calls.append(kwargs)
            entered.set()
            release.wait(timeout=5)
            return {"text": "Hallo", "language": "de", "segments": []}

    monkeypatch.setitem(sys.modules, "mlx_whisper", _FakeMlxWhisper)
    backend = MlxWhisperBackend(model_name="mlx-community/whisper-large-v3-mlx")
    holder_params = TranscribeParams(request_model="whisper-1", prompt=None, language="de")
    cancel_event = threading.Event()
    waiter_params = TranscribeParams(
        request_model="whisper-1",
        prompt=None,
        language="de",
        cancel_event=cancel_event,
    )
    waiter_errors: list[BaseException] = []

    holder = threading.Thread(
        target=lambda: backend.transcribe(Path("/tmp/first.wav"), holder_params)
    )
    holder.start()
    assert entered.wait(timeout=5)

    def waiter() -> None:
        try:
            backend.transcribe(Path("/tmp/second.wav"), waiter_params)
        except BaseException as exc:
            waiter_errors.append(exc)

    second = threading.Thread(target=waiter)
    second.start()
    # The holder still owns _MLX_LOCK, so the waiter cannot pass its checkpoint
    # before the event is set.
    cancel_event.set()
    release.set()
    holder.join(timeout=5)
    second.join(timeout=5)

    assert len(calls) == 1
    assert len(waiter_errors) == 1
    assert isinstance(waiter_errors[0], InferenceCancelled)


def test_whispercpp_backend_raises_before_starting_whisper_cli_when_cancelled(
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
    cancel_event = threading.Event()
    whisper_cli_calls: list[list[str]] = []

    monkeypatch.setattr(
        "transcriptions.service.resolve_whispercpp_binary",
        lambda _path: "/tmp/fake-whisper-cli",
    )
    monkeypatch.setattr(
        "transcriptions.service.resolve_whispercpp_model_path",
        lambda _name: model_path,
    )

    def fake_run(args, **kwargs):
        del kwargs
        call = [str(part) for part in args]
        Path(call[-1]).write_bytes(b"RIFFfakewav")
        # The client disconnects while ffmpeg normalizes the upload.
        cancel_event.set()
        return type("Completed", (), {"returncode": 0, "stderr": "", "stdout": ""})()

    def fake_whisper_cli(args, **kwargs):  # pragma: no cover - must never run
        del kwargs
        whisper_cli_calls.append([str(part) for part in args])
        raise AssertionError("whisper-cli must not start after cancellation.")

    monkeypatch.setattr("transcriptions.service.subprocess.run", fake_run)
    monkeypatch.setattr("transcriptions.service.run_cancellable_process", fake_whisper_cli)

    with pytest.raises(InferenceCancelled):
        backend.transcribe(
            input_path,
            TranscribeParams(
                request_model="whisper-1",
                prompt=None,
                language="de",
                cancel_event=cancel_event,
            ),
        )

    assert whisper_cli_calls == []


def _pid_is_gone(pid: int, *, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError, PermissionError:
            return True
        time.sleep(0.02)
    try:
        os.kill(pid, 0)
    except ProcessLookupError, PermissionError:
        return True
    return False


def test_transcribe_audio_kills_real_whisper_cli_and_releases_the_slot(
    monkeypatch, tmp_path: Path, settings
) -> None:
    settings.VOXHELM_LANE_SCHEDULER_ENABLED = True
    settings.VOXHELM_LANE_SCHEDULER_DIR = tmp_path
    settings.VOXHELM_LANE_SCHEDULER_STALE_SECONDS = 60
    settings.VOXHELM_LANE_SCHEDULER_INTERACTIVE_SLOTS = 1
    settings.VOXHELM_LANE_SCHEDULER_NON_INTERACTIVE_SLOTS = 1
    settings.VOXHELM_FFMPEG_BIN = "/tmp/fake-ffmpeg"

    started_path = tmp_path / "started"
    pid_path = tmp_path / "child.pid"
    fake_cli = tmp_path / "fake-whisper-cli.sh"
    # `exec` keeps the recorded shell pid, so the pidfile names the process that
    # run_cancellable_process signals and reaps.
    fake_cli.write_text(
        f'#!/bin/sh\necho $$ > "{pid_path}"\ntouch "{started_path}"\nexec sleep 30\n',
        encoding="utf-8",
    )
    fake_cli.chmod(0o755)
    model_path = tmp_path / "ggml-large-v3.bin"
    model_path.write_text("model", encoding="utf-8")
    input_path = tmp_path / "sample.m4a"
    input_path.write_bytes(b"not-really-audio")

    monkeypatch.setattr(
        "transcriptions.service.resolve_whispercpp_binary",
        lambda _path: str(fake_cli),
    )
    monkeypatch.setattr(
        "transcriptions.service.resolve_whispercpp_model_path",
        lambda _name: model_path,
    )

    def fake_ffmpeg_run(args, **kwargs):
        del kwargs
        call = [str(part) for part in args]
        Path(call[-1]).write_bytes(b"RIFFfakewav")
        return type("Completed", (), {"returncode": 0, "stderr": "", "stdout": ""})()

    monkeypatch.setattr("transcriptions.service.subprocess.run", fake_ffmpeg_run)
    backend = WhisperCppBackend(
        binary_path=str(fake_cli),
        model_name="ggml-large-v3.bin",
        processors=1,
    )
    monkeypatch.setattr(
        "transcriptions.service.get_backend_services_for_model",
        lambda _request_model: [BackendInvocation("whispercpp", backend)],
    )

    cancel_event = threading.Event()
    watcher = _cancel_once_ready(started_path, cancel_event)

    with pytest.raises(InferenceCancelled):
        transcribe_audio(
            input_path,
            TranscribeParams(
                request_model="whisper-1",
                prompt=None,
                language="de",
                cancel_event=cancel_event,
            ),
        )
    watcher.join(timeout=5)

    child_pid = int(pid_path.read_text(encoding="utf-8").strip())
    assert _pid_is_gone(child_pid)
    holders_dir = tmp_path / "holders"
    assert not holders_dir.exists() or list(holders_dir.iterdir()) == []
