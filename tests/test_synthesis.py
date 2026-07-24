from __future__ import annotations

import sys
from pathlib import Path

import pytest

from synthesis.service import (
    ExportedAudio,
    InstalledVoice,
    PiperBackend,
    RoutingDetection,
    SynthesisResult,
    SynthesizeParams,
    VoiceRegistry,
    build_voice_metadata,
    build_voice_registry,
    detect_routing_language,
    discover_installed_voices,
    export_audio,
    synthesize_text,
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


# --------------------------------------------------------------------------- #
# Language routing (synthesize_text)
# --------------------------------------------------------------------------- #


class CapturingBackend:
    """Backend stub that records the params it receives and echoes them back."""

    def __init__(self) -> None:
        self.params: SynthesizeParams | None = None

    def synthesize(self, text: str, params: SynthesizeParams) -> SynthesisResult:
        del text
        self.params = params
        return SynthesisResult(
            audio_path=Path("/tmp/routing-test.wav"),
            backend_name="kokoro",
            model_name="kokoro",
            voice_name=params.voice or "",
            language=params.language,
            sample_rate=24000,
            sample_width=2,
            channels=1,
            duration_seconds=1.0,
        )


def _registry_with(voice_key: str, backend: str = "kokoro") -> VoiceRegistry:
    return VoiceRegistry(
        voices=(
            InstalledVoice(
                key=voice_key, name=voice_key, backend=backend, languages=(), artifacts={}
            ),
        ),
        default_backend="piper",
    )


def _run_synthesize(monkeypatch, text: str, params: SynthesizeParams) -> CapturingBackend:
    backend = CapturingBackend()
    monkeypatch.setattr("synthesis.service.get_backend_service", lambda _params: backend)
    synthesize_text(text, params)
    return backend


def _forbid_detector(monkeypatch) -> None:
    def _fail(_text: str):  # pragma: no cover - only invoked on floor-logic bugs
        raise AssertionError("detector must not be consulted")

    monkeypatch.setattr("synthesis.service.detect_routing_language", _fail)


def test_synthesize_text_routing_off_passes_identical_params(monkeypatch, settings) -> None:
    # Regression: with routing disabled the params reach the backend unchanged
    # (same object) and the detector is never consulted.
    settings.VOXHELM_TTS_LANGUAGE_ROUTING = False
    settings.VOXHELM_TTS_LANGUAGE_VOICES = {"en": "kokoro-af_heart"}
    _forbid_detector(monkeypatch)
    params = SynthesizeParams(
        request_model="tts-1", voice="de_DE-thorsten-high", language="de", speed=1.0
    )

    backend = _run_synthesize(monkeypatch, "This is clearly English text right here.", params)

    assert backend.params is params


def test_synthesize_text_routes_german_text_to_mapped_voice(monkeypatch, settings) -> None:
    settings.VOXHELM_TTS_LANGUAGE_ROUTING = True
    settings.VOXHELM_TTS_LANGUAGE_VOICES = {"de": "kokoro-martin", "en": "kokoro-af_heart"}
    monkeypatch.setattr(
        "synthesis.service.detect_routing_language",
        lambda _text: RoutingDetection("de", 0.99),
    )
    monkeypatch.setattr(
        "synthesis.service.build_voice_registry", lambda: _registry_with("kokoro-martin")
    )
    params = SynthesizeParams(
        request_model="auto", voice="en_US-lessac-medium", language="en", speed=1.0
    )

    backend = _run_synthesize(monkeypatch, "Ein hinreichend langer deutscher Satz.", params)

    # Mapped voice AND detected language are effective end-to-end.
    assert backend.params is not None
    assert backend.params.voice == "kokoro-martin"
    assert backend.params.language == "de"


def test_synthesize_text_routes_english_text_to_mapped_voice(monkeypatch, settings) -> None:
    settings.VOXHELM_TTS_LANGUAGE_ROUTING = True
    settings.VOXHELM_TTS_LANGUAGE_VOICES = {"de": "kokoro-martin", "en": "kokoro-af_heart"}
    monkeypatch.setattr(
        "synthesis.service.detect_routing_language",
        lambda _text: RoutingDetection("en", 0.97),
    )
    monkeypatch.setattr(
        "synthesis.service.build_voice_registry", lambda: _registry_with("kokoro-af_heart")
    )
    # Pinned to a German voice/language, yet English text overrides both.
    params = SynthesizeParams(
        request_model="auto", voice="de_DE-thorsten-high", language="de", speed=1.0
    )

    backend = _run_synthesize(monkeypatch, "A sufficiently long English sentence here.", params)

    assert backend.params is not None
    assert backend.params.voice == "kokoro-af_heart"
    assert backend.params.language == "en"


def test_synthesize_text_short_text_stays_on_pinned_voice(monkeypatch, settings) -> None:
    settings.VOXHELM_TTS_LANGUAGE_ROUTING = True
    settings.VOXHELM_TTS_LANGUAGE_VOICES = {"en": "kokoro-af_heart"}
    # Below the 20-char floor: the detector must not even be consulted.
    _forbid_detector(monkeypatch)
    params = SynthesizeParams(
        request_model="auto", voice="de_DE-thorsten-high", language="de", speed=1.0
    )

    backend = _run_synthesize(monkeypatch, "Okay", params)

    assert backend.params is params


def test_synthesize_text_low_confidence_stays_on_pinned_voice(monkeypatch, settings) -> None:
    settings.VOXHELM_TTS_LANGUAGE_ROUTING = True
    settings.VOXHELM_TTS_LANGUAGE_VOICES = {"en": "kokoro-af_heart"}
    monkeypatch.setattr(
        "synthesis.service.detect_routing_language",
        lambda _text: RoutingDetection("en", 0.50),
    )
    params = SynthesizeParams(
        request_model="auto", voice="de_DE-thorsten-high", language="de", speed=1.0
    )

    backend = _run_synthesize(monkeypatch, "Ambiguous but long enough text here.", params)

    assert backend.params is params


def test_synthesize_text_unmapped_language_stays_on_pinned_voice(monkeypatch, settings) -> None:
    settings.VOXHELM_TTS_LANGUAGE_ROUTING = True
    settings.VOXHELM_TTS_LANGUAGE_VOICES = {"en": "kokoro-af_heart"}
    # Detected German, but only English is mapped -> no routing.
    monkeypatch.setattr(
        "synthesis.service.detect_routing_language",
        lambda _text: RoutingDetection("de", 0.99),
    )
    params = SynthesizeParams(
        request_model="auto", voice="de_DE-thorsten-high", language="de", speed=1.0
    )

    backend = _run_synthesize(monkeypatch, "Ein hinreichend langer deutscher Satz.", params)

    assert backend.params is params


def test_synthesize_text_unregistered_voice_warns_and_falls_back(
    monkeypatch, settings, caplog
) -> None:
    settings.VOXHELM_TTS_LANGUAGE_ROUTING = True
    settings.VOXHELM_TTS_LANGUAGE_VOICES = {"de": "kokoro-martin"}
    monkeypatch.setattr(
        "synthesis.service.detect_routing_language",
        lambda _text: RoutingDetection("de", 0.99),
    )
    # Mapped voice is not in the registry (empty registry) -> fall back + warn.
    monkeypatch.setattr(
        "synthesis.service.build_voice_registry",
        lambda: VoiceRegistry(voices=(), default_backend="piper"),
    )
    params = SynthesizeParams(
        request_model="auto", voice="de_DE-thorsten-high", language="de", speed=1.0
    )

    with caplog.at_level("WARNING"):
        backend = _run_synthesize(monkeypatch, "Ein hinreichend langer deutscher Satz.", params)

    assert backend.params is params
    assert "kokoro-martin" in caplog.text


def test_synthesize_text_registry_failure_warns_and_falls_back(
    monkeypatch, settings, caplog
) -> None:
    settings.VOXHELM_TTS_LANGUAGE_ROUTING = True
    settings.VOXHELM_TTS_LANGUAGE_VOICES = {"de": "kokoro-martin"}
    monkeypatch.setattr(
        "synthesis.service.detect_routing_language",
        lambda _text: RoutingDetection("de", 0.99),
    )

    def _broken_registry() -> VoiceRegistry:
        raise OSError("voice artifact unreadable")

    monkeypatch.setattr("synthesis.service.build_voice_registry", _broken_registry)
    params = SynthesizeParams(
        request_model="auto", voice="de_DE-thorsten-high", language="de", speed=1.0
    )

    with caplog.at_level("WARNING"):
        backend = _run_synthesize(monkeypatch, "Ein hinreichend langer deutscher Satz.", params)

    assert backend.params is params
    assert "registry lookup failed" in caplog.text


def test_synthesize_text_routing_bypass_field_skips_detection(monkeypatch, settings) -> None:
    settings.VOXHELM_TTS_LANGUAGE_ROUTING = True
    settings.VOXHELM_TTS_LANGUAGE_VOICES = {"en": "kokoro-af_heart"}
    # routing=False on the request must skip detection entirely.
    _forbid_detector(monkeypatch)
    params = SynthesizeParams(
        request_model="kokoro",
        voice="kokoro-martin",
        language=None,
        speed=1.0,
        routing=False,
    )

    backend = _run_synthesize(monkeypatch, "A sufficiently long English sentence here.", params)

    assert backend.params is params


def test_detect_routing_language_is_restricted_to_de_en() -> None:
    pytest.importorskip("lingua")

    german = detect_routing_language("Das Wetter ist heute wirklich schön und sehr warm.")
    english = detect_routing_language("The weather today is really nice and very warm.")
    french = detect_routing_language("Bonjour, comment allez-vous aujourd'hui mes chers amis?")

    assert german is not None and german.language == "de" and german.confidence >= 0.90
    assert english is not None and english.language == "en" and english.confidence >= 0.90
    # A third language is forced into one of the two restricted candidates.
    assert french is not None and french.language in {"de", "en"}


def test_synthesize_text_real_lingua_routes_english(monkeypatch, settings) -> None:
    # End-to-end routing with the REAL lingua detector (not monkeypatched).
    pytest.importorskip("lingua")
    settings.VOXHELM_TTS_LANGUAGE_ROUTING = True
    settings.VOXHELM_TTS_LANGUAGE_VOICES = {"de": "kokoro-martin", "en": "kokoro-af_heart"}
    monkeypatch.setattr(
        "synthesis.service.build_voice_registry", lambda: _registry_with("kokoro-af_heart")
    )
    params = SynthesizeParams(
        request_model="auto", voice="de_DE-thorsten-high", language="de", speed=1.0
    )

    backend = _run_synthesize(
        monkeypatch, "The weather today is really nice and very warm.", params
    )

    assert backend.params is not None
    assert backend.params.voice == "kokoro-af_heart"
    assert backend.params.language == "en"


# --------------------------------------------------------------------------- #
# Backend dispatch through the REAL registry (routing vs. explicit model)
# --------------------------------------------------------------------------- #


def _configure_real_registry(tmp_path: Path, settings) -> None:
    """Populate settings with a real Piper + Kokoro registry backed by fixtures.

    Both backends contribute one voice via on-disk artifacts so
    ``build_voice_registry`` / ``resolve_backend`` run for real (not monkeypatched)
    and the backend is chosen from the registry, not a stub.
    """
    piper_dir = tmp_path / "piper"
    piper_dir.mkdir()
    (piper_dir / "de_DE-thorsten-high.onnx").write_bytes(b"model")
    (piper_dir / "de_DE-thorsten-high.onnx.json").write_text("{}", encoding="utf-8")
    kokoro_dir = tmp_path / "kokoro"
    kokoro_dir.mkdir()
    (kokoro_dir / "kokoro-martin.onnx").write_bytes(b"onnx")
    (kokoro_dir / "voices-martin.npz").write_bytes(b"npz")

    settings.VOXHELM_TTS_BACKEND = "piper"
    settings.VOXHELM_PIPER_VOICE_DIR = piper_dir
    settings.VOXHELM_PIPER_VOICES = ["de_DE-thorsten-high"]
    settings.VOXHELM_KOKORO_MODEL_DIR = kokoro_dir
    settings.VOXHELM_KOKORO_MODELS = {
        "kokoro-martin": {
            "model": "kokoro-martin.onnx",
            "voicepack": "voices-martin.npz",
            "voicepack_key": "martin",
            "language": "de",
        }
    }
    settings.VOXHELM_KOKORO_DEFAULT_VOICE = ""
    settings.VOXHELM_ESPEAK_LIBRARY = ""


def _stub_kokoro_synthesize(monkeypatch) -> dict:
    """Stub the real KokoroBackend.synthesize (no ONNX/espeak) and echo params."""
    captured: dict = {}

    def fake_synth(self, text, params):  # noqa: ANN001 - test stub
        del text
        captured["params"] = params
        return SynthesisResult(
            audio_path=Path("/tmp/dispatch-test.wav"),
            backend_name="kokoro",
            model_name="kokoro",
            voice_name=params.voice or "",
            language=params.language,
            sample_rate=24000,
            sample_width=2,
            channels=1,
            duration_seconds=1.0,
        )

    monkeypatch.setattr("synthesis.kokoro.KokoroBackend.synthesize", fake_synth)
    return captured


def test_synthesize_text_routing_releases_pinned_model_to_kokoro_backend(
    tmp_path: Path, settings, monkeypatch
) -> None:
    # A request pinned to the Piper model, whose text routes to a Kokoro-mapped
    # voice, must dispatch to the KOKORO backend: routing releases the pinned
    # request_model so registry dispatch follows the mapped voice's backend.
    _configure_real_registry(tmp_path, settings)
    settings.VOXHELM_TTS_LANGUAGE_ROUTING = True
    settings.VOXHELM_TTS_LANGUAGE_VOICES = {"de": "kokoro-martin", "en": "kokoro-af_heart"}
    monkeypatch.setattr(
        "synthesis.service.detect_routing_language",
        lambda _text: RoutingDetection("de", 0.99),
    )
    captured = _stub_kokoro_synthesize(monkeypatch)

    params = SynthesizeParams(
        request_model="piper", voice="de_DE-thorsten-high", language="de", speed=1.0
    )
    result = synthesize_text("Ein hinreichend langer deutscher Satz.", params)

    # Real dispatch selected the Kokoro backend for the routed voice...
    assert result.backend_name == "kokoro"
    assert result.voice_name == "kokoro-martin"
    # ...and the explicitly pinned model was released to the auto name so it can
    # never hold the routed voice on the wrong (Piper) backend.
    assert captured["params"].request_model == "auto"
    assert captured["params"].voice == "kokoro-martin"
    assert captured["params"].language == "de"


def test_synthesize_text_explicit_kokoro_model_forces_backend_with_routing_off(
    tmp_path: Path, settings, monkeypatch
) -> None:
    # Existing explicit-model semantics stay intact: with routing OFF, an explicit
    # request_model="kokoro" still forces the Kokoro backend even though the
    # default backend is Piper and no Kokoro voice is pinned.
    _configure_real_registry(tmp_path, settings)
    settings.VOXHELM_TTS_LANGUAGE_ROUTING = False
    _forbid_detector(monkeypatch)
    captured = _stub_kokoro_synthesize(monkeypatch)

    params = SynthesizeParams(
        request_model="kokoro", voice=None, language=None, speed=1.0
    )
    result = synthesize_text("Ein hinreichend langer deutscher Satz.", params)

    assert result.backend_name == "kokoro"
    # request_model reaches the backend unchanged (routing did not touch it).
    assert captured["params"].request_model == "kokoro"
