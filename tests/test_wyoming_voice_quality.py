from __future__ import annotations

import array
import math
from pathlib import Path

import pytest

import transcriptions.wyoming as wyoming_module
from synthesis.service import RoutingDetection
from transcriptions.service import TranscribeParams, TranscriptionResult
from transcriptions.wyoming import (
    WyomingSttConfig,
    needs_language_retry,
    normalize_loudness,
    transcribe_pcm,
    warm_up_stt,
)


def tone(amplitude: int, samples: int = 16000) -> bytes:
    return array.array(
        "h", (round(amplitude * math.sin(2 * math.pi * 440 * i / 16000)) for i in range(samples))
    ).tobytes()


def rms_dbfs(audio: bytes) -> float:
    values = array.array("h")
    values.frombytes(audio)
    return 20 * math.log10(math.sqrt(sum(v * v for v in values) / len(values)) / 32768)


def peak(audio: bytes) -> int:
    values = array.array("h")
    values.frombytes(audio)
    return max(abs(v) for v in values)


def test_quiet_voice_is_raised_to_the_target() -> None:
    quiet = tone(2000)  # about -27 dBFS RMS
    out = normalize_loudness(quiet, width=2, target_dbfs=-19.0)
    # -27 dBFS needs +8 dB, exactly the boost limit.
    assert rms_dbfs(out) == pytest.approx(min(-19.0, rms_dbfs(quiet) + 8.0), abs=0.2)


def test_loud_voice_is_lowered_to_the_target() -> None:
    out = normalize_loudness(tone(12000), width=2, target_dbfs=-19.0)
    assert rms_dbfs(out) == pytest.approx(-19.0, abs=0.2)


def test_boost_never_lifts_peaks_above_the_ceiling() -> None:
    # A spiky signal: low RMS, one full-scale peak.
    values = array.array("h", [100] * 16000)
    values[100] = 30000
    out = normalize_loudness(values.tobytes(), width=2, target_dbfs=-19.0)
    assert peak(out) <= round(0.89 * 32767) + 1


@pytest.mark.parametrize(
    ("audio", "width", "target"),
    [(bytes(3200), 2, -19.0), (tone(2000), 2, None), (b"\x10\x20\x30", 1, -19.0)],
)
def test_silence_disabled_and_other_widths_pass_unchanged(audio, width, target) -> None:
    assert normalize_loudness(audio, width=width, target_dbfs=target) == audio


@pytest.mark.parametrize(
    ("text", "language", "detected", "expected"),
    [
        ("Uh, that's linking.", "de", "en", True),
        ("Yep.", "de", "en", True),
        ("Das linke.", "de", "de", False),
        ("Turn the light off.", "en", "en", False),
        ("", "de", "en", False),
        ("Yep.", None, "en", False),
        ("Yep.", "fr", "en", False),
    ],
)
def test_needs_language_retry(monkeypatch, text, language, detected, expected) -> None:
    monkeypatch.setattr(
        wyoming_module, "detect_routing_language", lambda value: RoutingDetection(detected, 0.6)
    )
    assert needs_language_retry(text, language) is expected


def test_needs_language_retry_is_false_without_a_detector(monkeypatch) -> None:
    def broken(value: str):
        raise ImportError("lingua")

    monkeypatch.setattr(wyoming_module, "detect_routing_language", broken)
    assert needs_language_retry("Yep.", "de") is False


def test_parakeet_english_transcript_is_retried_with_whisper_in_german(
    monkeypatch, settings
) -> None:
    settings.VOXHELM_MLX_MODEL = "mlx-community/whisper-large-v3-mlx"
    calls: list[TranscribeParams] = []

    def fake_transcribe(audio_path: Path, params: TranscribeParams) -> TranscriptionResult:
        calls.append(params)
        if params.backend == "parakeet-mlx":
            return TranscriptionResult(
                text="Uh, that's linking.", language=None, segments=[], backend_name="parakeet-mlx"
            )
        return TranscriptionResult(
            text="Das linke.", language="de", segments=[], backend_name="mlx-whisper"
        )

    monkeypatch.setattr(wyoming_module, "transcribe_audio", fake_transcribe)
    monkeypatch.setattr(
        wyoming_module, "detect_routing_language", lambda value: RoutingDetection("en", 0.7)
    )

    result = transcribe_pcm(
        bytes(3200),
        request_model="parakeet-tdt-0.6b-v3",
        prompt=None,
        language="de",
        backend="parakeet-mlx",
    )

    assert result.text == "Das linke."
    assert [(p.backend, p.language, p.request_model) for p in calls] == [
        ("parakeet-mlx", "de", "parakeet-tdt-0.6b-v3"),
        ("mlx", "de", "mlx-community/whisper-large-v3-mlx"),
    ]


def test_whisper_fallback_result_is_not_retried(monkeypatch) -> None:
    calls: list[TranscribeParams] = []

    def fake_transcribe(audio_path: Path, params: TranscribeParams) -> TranscriptionResult:
        calls.append(params)
        return TranscriptionResult(
            text="Yep.", language="de", segments=[], backend_name="mlx-whisper"
        )

    monkeypatch.setattr(wyoming_module, "transcribe_audio", fake_transcribe)
    monkeypatch.setattr(
        wyoming_module, "detect_routing_language", lambda value: RoutingDetection("en", 0.7)
    )
    transcribe_pcm(
        bytes(3200), request_model="m", prompt=None, language="de", backend="parakeet-mlx"
    )
    assert len(calls) == 1


def test_parakeet_warm_up_also_loads_whisper(monkeypatch, settings) -> None:
    settings.VOXHELM_MLX_MODEL = "mlx-community/whisper-large-v3-mlx"
    seen: list[str | None] = []

    def fake_transcribe(audio_path: Path, params: TranscribeParams) -> TranscriptionResult:
        seen.append(params.backend)
        return TranscriptionResult(text="", language="de", segments=[], backend_name="x")

    monkeypatch.setattr(wyoming_module, "transcribe_audio", fake_transcribe)
    config = WyomingSttConfig(
        host="127.0.0.1",
        port=10300,
        backend="parakeet-mlx",
        model="parakeet-tdt-0.6b-v3",
        language=None,
        languages=("de", "en"),
        prompt=None,
    )
    warm_up_stt(config)
    assert seen == ["parakeet-mlx", "mlx"]


def test_record_voice_turn_is_off_by_default(settings, tmp_path) -> None:
    settings.VOXHELM_WYOMING_STT_RECORD_DIR = ""
    wyoming_module.record_voice_turn(bytes(320), text="x", backend="b", language="de")
    assert list(tmp_path.iterdir()) == []


def test_record_voice_turn_keeps_newest_owner_only(settings, tmp_path) -> None:
    import json
    import os
    import stat
    import wave

    target = tmp_path / "rec"
    settings.VOXHELM_WYOMING_STT_RECORD_DIR = str(target)
    settings.VOXHELM_WYOMING_STT_RECORD_MAX = 2
    target.mkdir(mode=0o750)
    (target / "legacy.wav").write_bytes(b"legacy audio")
    (target / "legacy.json").write_text("legacy metadata")
    root = target / "voxhelm-turns"
    root.mkdir()
    incomplete = root / "00000000000000000001-00000000000000000000000000000000"
    incomplete.mkdir()
    (incomplete / "audio.wav").write_bytes(b"incomplete")
    (root / "other.wav").write_bytes(b"unrelated")
    for index in range(3):
        wyoming_module.record_voice_turn(
            bytes(320), text=f"t{index}", backend="parakeet-mlx", language="de"
        )
    assert (target / "legacy.wav").read_bytes() == b"legacy audio"
    assert (target / "legacy.json").read_text() == "legacy metadata"
    assert (incomplete / "audio.wav").read_bytes() == b"incomplete"
    assert (root / "other.wav").read_bytes() == b"unrelated"
    assert stat.S_IMODE(target.stat().st_mode) == 0o750
    wavs = sorted(root.glob("*/metadata.json"))
    wavs = [path.with_name("audio.wav") for path in wavs]
    assert len(wavs) == 2
    assert stat.S_IMODE(os.stat(root).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(wavs[0]).st_mode) == 0o600
    texts = [json.loads(w.with_name("metadata.json").read_text())["text"] for w in wavs]
    assert texts == ["t1", "t2"]
    with wave.open(str(wavs[0]), "rb") as wav_file:
        assert wav_file.getframerate() == 16000


def test_recording_publishes_complete_turn_and_recovers_pending(settings, tmp_path, monkeypatch):
    settings.VOXHELM_WYOMING_STT_RECORD_DIR = str(tmp_path)
    root = tmp_path / "voxhelm-turns"
    record = wyoming_module.record_voice_turn

    record(bytes(320), text="first", backend="b", language="de")
    first = next(root.glob("[0-9]*"))
    original_open = wyoming_module.os.open

    def fail_metadata(path, *args, **kwargs):
        if getattr(path, "name", None) == "metadata.json":
            raise OSError("simulated disk failure")
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(wyoming_module.os, "open", fail_metadata)
        record(bytes(320), text="failed", backend="b", language="de")
    assert list(root.glob("[0-9]*")) == [first]

    abandoned = root / (".pending-" + first.name)
    abandoned.mkdir()
    (abandoned / "audio.wav").write_bytes(b"interrupted write")
    unrelated = root / ".pending-elsewhere"
    unrelated.mkdir()
    record(bytes(320), text="next", backend="b", language="de")
    assert first.is_dir()
    assert not abandoned.exists()
    assert unrelated.is_dir()
    assert len(list(root.glob("[0-9]*/metadata.json"))) == 2


def test_recording_retains_newest_complete_turns_within_byte_budget(settings, tmp_path, caplog):
    settings.VOXHELM_WYOMING_STT_RECORD_DIR = str(tmp_path)
    settings.VOXHELM_WYOMING_STT_RECORD_MAX = 10
    settings.VOXHELM_WYOMING_STT_RECORD_MAX_BYTES = 1000
    root = tmp_path / "voxhelm-turns"
    record = wyoming_module.record_voice_turn

    for name, size in [("first", 120), ("large", 440), ("newest", 460), ("oversized", 1100)]:
        record(bytes(size), text=name, backend="b", language="de")

    import json

    files = sorted(root.glob("[0-9]*/metadata.json"))
    assert [json.loads(path.read_bytes())["text"] for path in files] == ["first", "newest"]
    assert sum(path.stat().st_size for path in root.glob("[0-9]*/*")) <= 1000
    assert "Wyoming STT recording exceeds byte budget" in caplog.text
