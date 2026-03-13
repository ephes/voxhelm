from __future__ import annotations

from pathlib import Path

import pytest

from synthesis.service import (
    ExportedAudio,
    InstalledVoice,
    PiperBackend,
    SynthesisResult,
    build_voice_metadata,
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
        languages=("en", "en_US"),
        model_path=model_path,
        config_path=config_path,
        speakers=("speaker_0",),
    )


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
