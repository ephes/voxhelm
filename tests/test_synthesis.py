from __future__ import annotations

import sys
from pathlib import Path

import pytest

from synthesis.service import (
    ExportedAudio,
    InstalledVoice,
    PiperBackend,
    SynthesisResult,
    VoiceRegistry,
    build_voice_metadata,
    build_voice_registry,
    discover_installed_voices,
    export_audio,
)


def write_voice_fixture(tmp_path: Path, voice_name: str) -> tuple[Path, Path]:
    model_path = tmp_path / f"{voice_name}.onnx"
    config_path = tmp_path / f"{voice_name}.onnx.json"
    model_path.write_bytes(b"model")
    config_path.write_text('{"speaker_id_map": {"speaker_0": 0}}', encoding="utf-8")
    return model_path, config_path


def test_discover_installed_voices_reads_configured_voices(tmp_path: Path) -> None:
    write_voice_fixture(tmp_path, "en_US-lessac-medium")

    discovered = discover_installed_voices(
        voice_dir=tmp_path,
        configured_voices=["en_US-lessac-medium"],
    )

    assert list(discovered) == ["en_US-lessac-medium"]
    assert discovered["en_US-lessac-medium"].languages == ("en", "en_US")


def test_build_voice_metadata_reads_speakers(tmp_path: Path) -> None:
    model_path, config_path = write_voice_fixture(tmp_path, "de_DE-thorsten-high")

    metadata = build_voice_metadata(
        voice_name="de_DE-thorsten-high",
        model_path=model_path,
        config_path=config_path,
    )

    assert metadata.speakers == ("speaker_0",)


def test_piper_backend_resolves_voice_by_language(tmp_path: Path) -> None:
    model_path, config_path = write_voice_fixture(tmp_path, "en_US-lessac-medium")
    backend = PiperBackend(
        voice_dir=tmp_path,
        configured_voices=["en_US-lessac-medium"],
        default_voice="en_US-lessac-medium",
        language_voices={"en": "en_US-lessac-medium"},
    )

    resolved = backend.resolve_voice(voice=None, language="en")

    assert resolved == InstalledVoice(
        key="en_US-lessac-medium",
        name="en_US-lessac-medium",
        backend="piper",
        languages=("en", "en_US"),
        artifacts={"model": model_path, "config": config_path},
        speakers=("speaker_0",),
    )
    assert resolved.model_path == model_path
    assert resolved.config_path == config_path


def test_build_voice_registry_lists_piper_voices(tmp_path: Path, settings) -> None:
    write_voice_fixture(tmp_path, "en_US-lessac-medium")
    settings.VOXHELM_TTS_BACKEND = "piper"
    settings.VOXHELM_PIPER_VOICE_DIR = tmp_path
    settings.VOXHELM_PIPER_VOICES = ["en_US-lessac-medium"]

    registry = build_voice_registry()

    assert [voice.key for voice in registry.voices] == ["en_US-lessac-medium"]
    assert registry.voices[0].backend == "piper"
    assert registry.default_backend == "piper"


def test_voice_registry_dispatches_known_voice_to_its_backend() -> None:
    registry = VoiceRegistry(
        voices=(
            InstalledVoice(
                key="kokoro-martin",
                name="kokoro-martin",
                backend="kokoro",
                languages=("de",),
                artifacts={},
            ),
        ),
        default_backend="piper",
    )

    assert registry.resolve_backend(voice="kokoro-martin", request_model="auto") == "kokoro"
    # Case-insensitive alias resolves to the same backend.
    assert registry.resolve_backend(voice="KOKORO-MARTIN", request_model="auto") == "kokoro"


def test_voice_registry_unknown_or_unpinned_voice_uses_default_backend() -> None:
    registry = VoiceRegistry(voices=(), default_backend="piper")

    assert registry.resolve_backend(voice=None, request_model="auto") == "piper"
    assert registry.resolve_backend(voice="", request_model="tts-1") == "piper"
    # A pinned voice the registry does not know (e.g. a language alias) also
    # falls back to the default backend so that backend resolves it internally.
    assert registry.resolve_backend(voice="de", request_model="auto") == "piper"


def test_voice_registry_default_backend_follows_configuration() -> None:
    registry = VoiceRegistry(voices=(), default_backend="kokoro")

    assert registry.resolve_backend(voice=None, request_model="auto") == "kokoro"


def test_export_audio_returns_wav_without_conversion(tmp_path: Path) -> None:
    wav_path = tmp_path / "speech.wav"
    wav_path.write_bytes(b"RIFF")
    result = SynthesisResult(
        audio_path=wav_path,
        backend_name="piper",
        model_name="piper",
        voice_name="en_US-lessac-medium",
        language="en",
        sample_rate=22050,
        sample_width=2,
        channels=1,
        duration_seconds=1.0,
    )

    exported = export_audio(result, output_format="wav")

    assert exported == ExportedAudio(path=wav_path, format_name="wav", content_type="audio/wav")


def test_export_audio_handles_non_utf8_ffmpeg_stderr(tmp_path: Path, settings) -> None:
    fake_ffmpeg = tmp_path / "fake-ffmpeg.py"
    fake_ffmpeg.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "sys.stderr.buffer.write(b'bad byte: \\xf0')\n"
        "raise SystemExit(1)\n",
        encoding="utf-8",
    )
    fake_ffmpeg.chmod(0o755)
    wav_path = tmp_path / "speech.wav"
    wav_path.write_bytes(b"RIFF")
    result = SynthesisResult(
        audio_path=wav_path,
        backend_name="piper",
        model_name="piper",
        voice_name="en_US-lessac-medium",
        language="en",
        sample_rate=22050,
        sample_width=2,
        channels=1,
        duration_seconds=1.0,
    )
    settings.VOXHELM_FFMPEG_BIN = str(fake_ffmpeg)

    with pytest.raises(RuntimeError, match="Audio conversion failed: bad byte: �"):
        export_audio(result, output_format="mp3")


def test_export_audio_rejects_unknown_format(tmp_path: Path) -> None:
    wav_path = tmp_path / "speech.wav"
    wav_path.write_bytes(b"RIFF")
    result = SynthesisResult(
        audio_path=wav_path,
        backend_name="piper",
        model_name="piper",
        voice_name="en_US-lessac-medium",
        language="en",
        sample_rate=22050,
        sample_width=2,
        channels=1,
        duration_seconds=1.0,
    )

    with pytest.raises(RuntimeError, match="Unsupported audio format"):
        export_audio(result, output_format="flac")
