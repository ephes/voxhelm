from __future__ import annotations

import asyncio
import io
import wave
from pathlib import Path

import pytest
from django.core.exceptions import ImproperlyConfigured
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event, read_event
from wyoming.info import Describe, Info
from wyoming.tts import Synthesize

from synthesis.service import SynthesizeParams
from transcriptions.service import TranscribeParams, TranscriptionResult
from transcriptions.wyoming import (
    WyomingSttConfig,
    WyomingSttEventHandler,
    build_wyoming_info,
    get_wyoming_stt_config,
)


class DummyWriter:
    def __init__(self) -> None:
        self.buffer = bytearray()

    def writelines(self, chunks) -> None:
        for chunk in chunks:
            self.write(chunk)

    def write(self, chunk: bytes) -> None:
        self.buffer.extend(chunk)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        return None


def decode_single_event(writer: DummyWriter):
    return read_event(io.BytesIO(bytes(writer.buffer)))


def decode_all_events(writer: DummyWriter) -> list[Event]:
    stream = io.BytesIO(bytes(writer.buffer))
    events: list[Event] = []
    while True:
        event = read_event(stream)
        if event is None:
            return events
        events.append(event)


def limit_config(max_audio_seconds: int = 1) -> WyomingSttConfig:
    return WyomingSttConfig(
        host="127.0.0.1",
        port=10300,
        backend="whispercpp",
        model="ggml-large-v3.bin",
        language="de",
        languages=("de",),
        prompt=None,
        max_audio_seconds=max_audio_seconds,
    )


# One second of 16 kHz, 16-bit, mono PCM.
ONE_SECOND_PCM = b"\x01\x00" * 16000


def pcm_chunk(audio: bytes):
    return AudioChunk(rate=16000, width=2, channels=1, audio=audio).event()


def test_get_wyoming_stt_config_uses_dedicated_defaults(settings) -> None:
    settings.VOXHELM_STT_BACKEND = "whispercpp"
    settings.VOXHELM_WHISPERCPP_MODEL = "ggml-large-v3.bin"
    settings.VOXHELM_WYOMING_STT_BACKEND = "mlx"
    settings.VOXHELM_MLX_MODEL = "mlx-community/whisper-large-v3-mlx"
    settings.VOXHELM_WYOMING_STT_MODEL = ""
    settings.VOXHELM_WYOMING_STT_LANGUAGE = "de"
    settings.VOXHELM_WYOMING_STT_LANGUAGES = ["de", "en"]
    settings.VOXHELM_WYOMING_STT_PROMPT = "Home Assistant command"

    config = get_wyoming_stt_config()

    assert config.backend == "mlx"
    assert config.model == "mlx-community/whisper-large-v3-mlx"
    assert config.language == "de"
    assert config.languages == ("de", "en")
    assert config.prompt == "Home Assistant command"


def test_build_wyoming_info_announces_model_and_languages() -> None:
    info = build_wyoming_info(
        WyomingSttConfig(
            host="0.0.0.0",
            port=10300,
            backend="whispercpp",
            model="ggml-large-v3.bin",
            language="de",
            languages=("de", "en"),
            prompt=None,
        )
    )

    asr_program = info.asr[0]
    assert asr_program.name == "voxhelm"
    assert asr_program.models[0].name == "ggml-large-v3.bin"
    assert asr_program.models[0].languages == ["de", "en"]
    assert info.tts[0].name == "voxhelm"


def test_wyoming_handler_describe_returns_info() -> None:
    writer = DummyWriter()
    config = WyomingSttConfig(
        host="127.0.0.1",
        port=10300,
        backend="whispercpp",
        model="ggml-large-v3.bin",
        language="de",
        languages=("de",),
        prompt=None,
    )

    async def run_flow() -> None:
        handler = WyomingSttEventHandler(
            config,
            build_wyoming_info(config),
            asyncio.StreamReader(),
            writer,
        )
        await handler.handle_event(Describe().event())

    asyncio.run(run_flow())

    event = decode_single_event(writer)
    assert event is not None
    assert Info.is_type(event.type)


def test_wyoming_handler_transcribes_audio(monkeypatch) -> None:
    writer = DummyWriter()
    config = WyomingSttConfig(
        host="127.0.0.1",
        port=10300,
        backend="whispercpp",
        model="ggml-large-v3.bin",
        language="de",
        languages=("de",),
        prompt=None,
    )
    calls: list[tuple[Path, TranscribeParams]] = []

    def fake_transcribe(audio_path: Path, params: TranscribeParams) -> TranscriptionResult:
        calls.append((audio_path, params))
        with wave.open(str(audio_path), "rb") as wav_file:
            assert wav_file.getframerate() == 16000
            assert wav_file.getsampwidth() == 2
            assert wav_file.getnchannels() == 1
        return TranscriptionResult(text="hallo welt", language="de", segments=[])

    monkeypatch.setattr("transcriptions.wyoming.transcribe_audio", fake_transcribe)

    async def run_flow() -> None:
        handler = WyomingSttEventHandler(
            config,
            build_wyoming_info(config),
            asyncio.StreamReader(),
            writer,
        )
        await handler.handle_event(Transcribe(name="ggml-large-v3.bin", language="de").event())
        await handler.handle_event(AudioStart(rate=8000, width=2, channels=2).event())
        await handler.handle_event(
            AudioChunk(
                rate=8000,
                width=2,
                channels=2,
                audio=b"\x01\x00\x01\x00" * 320,
            ).event()
        )
        await handler.handle_event(AudioStop().event())

    asyncio.run(run_flow())

    event = decode_single_event(writer)
    assert event is not None
    assert Transcript.is_type(event.type)
    transcript = Transcript.from_event(event)
    assert transcript.text == "hallo welt"
    assert calls[0][1].request_model == "ggml-large-v3.bin"
    assert calls[0][1].language == "de"
    assert calls[0][1].scheduler_lane == "interactive"


def test_wyoming_handler_emits_debug_log(monkeypatch) -> None:
    writer = DummyWriter()
    config = WyomingSttConfig(
        host="127.0.0.1",
        port=10300,
        backend="mlx",
        model="mlx-community/whisper-large-v3-mlx",
        language="en",
        languages=("en",),
        prompt="Home Assistant command",
    )
    debug_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        "transcriptions.wyoming.transcribe_audio",
        lambda audio_path, params: TranscriptionResult(
            text="turn on the office light",
            language="en",
            segments=[],
            backend_name="mlx-whisper",
            model_name="mlx-community/whisper-large-v3-mlx",
        ),
    )
    monkeypatch.setattr(
        "transcriptions.wyoming.emit_transcription_debug_log",
        lambda **kwargs: debug_calls.append(kwargs),
    )

    async def run_flow() -> None:
        handler = WyomingSttEventHandler(
            config,
            build_wyoming_info(config),
            asyncio.StreamReader(),
            writer,
        )
        await handler.handle_event(Transcribe(language="en").event())
        await handler.handle_event(AudioStart(rate=8000, width=2, channels=2).event())
        await handler.handle_event(
            AudioChunk(
                rate=8000,
                width=2,
                channels=2,
                audio=b"\x01\x00\x01\x00" * 320,
            ).event()
        )
        await handler.handle_event(AudioStop().event())

    asyncio.run(run_flow())

    assert len(debug_calls) == 1
    debug_payload = debug_calls[0]
    assert debug_payload["source"] == "wyoming"
    assert debug_payload["request_model"] == "mlx-community/whisper-large-v3-mlx"
    assert debug_payload["request_language"] == "en"
    assert debug_payload["prompt"] == "Home Assistant command"
    assert debug_payload["audio_shape"] == {
        "input": {
            "bytes": 1280,
            "channels": 2,
            "duration_seconds": 0.04,
            "rate": 8000,
            "width": 2,
        },
        "converted": {
            "bytes": 1278,
            "channels": 1,
            "duration_seconds": 0.04,
            "rate": 16000,
            "width": 2,
        },
    }


def test_wyoming_handler_normalizes_leading_fillers(monkeypatch, settings) -> None:
    writer = DummyWriter()
    settings.VOXHELM_WYOMING_STT_NORMALIZE_TRANSCRIPT = True
    config = WyomingSttConfig(
        host="127.0.0.1",
        port=10300,
        backend="mlx",
        model="mlx-community/whisper-large-v3-mlx",
        language="de",
        languages=("de", "en"),
        prompt=None,
    )

    monkeypatch.setattr(
        "transcriptions.wyoming.transcribe_audio",
        lambda audio_path, params: TranscriptionResult(
            text="Okay wie ist denn die Temperatur im Wintergarten?",
            language="de",
            segments=[],
            backend_name="mlx-whisper",
            model_name="mlx-community/whisper-large-v3-mlx",
        ),
    )

    async def run_flow() -> None:
        handler = WyomingSttEventHandler(
            config,
            build_wyoming_info(config),
            asyncio.StreamReader(),
            writer,
        )
        await handler.handle_event(Transcribe(language="de").event())
        await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
        await handler.handle_event(
            AudioChunk(
                rate=16000,
                width=2,
                channels=1,
                audio=b"\x01\x00" * 320,
            ).event()
        )
        await handler.handle_event(AudioStop().event())

    asyncio.run(run_flow())

    event = decode_single_event(writer)
    assert event is not None
    assert Transcript.is_type(event.type)
    transcript = Transcript.from_event(event)
    assert transcript.text == "wie ist die Temperatur im Wintergarten?"


def test_wyoming_handler_returns_error_for_empty_audio() -> None:
    writer = DummyWriter()
    config = WyomingSttConfig(
        host="127.0.0.1",
        port=10300,
        backend="whispercpp",
        model="ggml-large-v3.bin",
        language=None,
        languages=("en",),
        prompt=None,
    )

    async def run_flow() -> None:
        handler = WyomingSttEventHandler(
            config,
            build_wyoming_info(config),
            asyncio.StreamReader(),
            writer,
        )
        await handler.handle_event(AudioStop().event())

    asyncio.run(run_flow())

    event = decode_single_event(writer)
    assert event is not None
    assert Error.is_type(event.type)
    assert Error.from_event(event).code == "empty_audio"


def test_wyoming_handler_synthesizes_audio(monkeypatch, tmp_path: Path) -> None:
    writer = DummyWriter()
    config = WyomingSttConfig(
        host="127.0.0.1",
        port=10300,
        backend="whispercpp",
        model="ggml-large-v3.bin",
        language="en",
        languages=("en",),
        prompt=None,
    )
    speech_path = tmp_path / "speech.wav"
    with wave.open(str(speech_path), "wb") as wav_file:
        wav_file.setframerate(22050)
        wav_file.setsampwidth(2)
        wav_file.setnchannels(1)
        wav_file.writeframes(b"\x01\x00" * 2205)

    calls: list[tuple[str, SynthesizeParams]] = []

    def fake_synthesize(text: str, params: SynthesizeParams):
        calls.append((text, params))
        return type("SpeechResult", (), {"audio_path": speech_path})()

    monkeypatch.setattr(
        "transcriptions.wyoming.synthesize_text",
        fake_synthesize,
    )
    monkeypatch.setattr(
        "transcriptions.wyoming.cleanup_paths",
        lambda *paths: None,
    )

    async def run_flow() -> None:
        handler = WyomingSttEventHandler(
            config,
            build_wyoming_info(config),
            asyncio.StreamReader(),
            writer,
        )
        await handler.handle_event(Synthesize(text="Hello world").event())

    asyncio.run(run_flow())

    stream = io.BytesIO(bytes(writer.buffer))
    events: list[Event] = []
    while True:
        event = read_event(stream)
        if event is None:
            break
        events.append(event)

    assert len(events) >= 2
    assert AudioStart.is_type(events[0].type)
    assert AudioStop.is_type(events[-1].type)
    assert calls[0][1].scheduler_lane == "interactive"


def test_wyoming_handler_cleans_up_audio_after_synthesis_write_failure(
    monkeypatch, tmp_path: Path
) -> None:
    writer = DummyWriter()
    config = WyomingSttConfig(
        host="127.0.0.1",
        port=10300,
        backend="whispercpp",
        model="ggml-large-v3.bin",
        language="en",
        languages=("en",),
        prompt=None,
    )
    speech_path = tmp_path / "speech.wav"
    speech_path.write_bytes(b"RIFFboom")
    cleaned_paths: list[Path] = []

    monkeypatch.setattr(
        "transcriptions.wyoming.synthesize_text",
        lambda text, params: type("SpeechResult", (), {"audio_path": speech_path})(),
    )

    async def fail_write_audio(self, audio_path: Path) -> None:
        del self, audio_path
        raise RuntimeError("write exploded")

    monkeypatch.setattr(
        WyomingSttEventHandler,
        "_write_synthesized_audio",
        fail_write_audio,
    )
    monkeypatch.setattr(
        "transcriptions.wyoming.cleanup_paths",
        lambda *paths: cleaned_paths.extend(paths),
    )

    async def run_flow() -> None:
        handler = WyomingSttEventHandler(
            config,
            build_wyoming_info(config),
            asyncio.StreamReader(),
            writer,
        )
        await handler.handle_event(Synthesize(text="Hello world").event())

    asyncio.run(run_flow())

    event = decode_single_event(writer)
    assert event is not None
    assert Error.is_type(event.type)
    assert Error.from_event(event).code == "synthesis_failed"
    assert cleaned_paths == [speech_path]


def test_get_wyoming_stt_config_reads_max_audio_seconds(settings) -> None:
    settings.VOXHELM_WYOMING_STT_MAX_AUDIO_SECONDS = 45

    config = get_wyoming_stt_config()

    assert config.max_audio_seconds == 45
    assert config.max_audio_bytes == 45 * 32000


@pytest.mark.parametrize("value", [0, -5])
def test_get_wyoming_stt_config_rejects_non_positive_max_audio_seconds(settings, value) -> None:
    settings.VOXHELM_WYOMING_STT_MAX_AUDIO_SECONDS = value

    with pytest.raises(ImproperlyConfigured):
        get_wyoming_stt_config()


def test_wyoming_config_defaults_to_120_second_audio_cap() -> None:
    config = WyomingSttConfig(
        host="127.0.0.1",
        port=10300,
        backend="whispercpp",
        model="ggml-large-v3.bin",
        language=None,
        languages=("en",),
        prompt=None,
    )

    assert config.max_audio_seconds == 120


def test_wyoming_handler_rejects_audio_over_cap(monkeypatch) -> None:
    writer = DummyWriter()
    config = limit_config(max_audio_seconds=1)
    calls: list[Path] = []
    monkeypatch.setattr(
        "transcriptions.wyoming.transcribe_audio",
        lambda audio_path, params: calls.append(audio_path),
    )
    results: list[bool] = []

    async def run_flow() -> WyomingSttEventHandler:
        handler = WyomingSttEventHandler(
            config,
            build_wyoming_info(config),
            asyncio.StreamReader(),
            writer,
        )
        results.append(await handler.handle_event(Transcribe(language="de").event()))
        results.append(
            await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
        )
        results.append(await handler.handle_event(pcm_chunk(ONE_SECOND_PCM[:-2])))
        results.append(await handler.handle_event(pcm_chunk(b"\x01\x00\x01\x00")))
        return handler

    handler = asyncio.run(run_flow())

    assert results == [True, True, True, False]
    events = decode_all_events(writer)
    assert len(events) == 1
    assert Error.is_type(events[0].type)
    error = Error.from_event(events[0])
    assert error.code == "audio_too_long"
    assert "1 second limit" in (error.text or "")
    assert calls == []
    assert handler.audio_buffer.tell() == 0
    assert handler.audio_shape.converted_bytes == 0


def test_wyoming_handler_cap_applies_to_converted_audio(monkeypatch) -> None:
    """The cap is checked after conversion to 16 kHz mono, not on the raw input bytes."""
    writer = DummyWriter()
    config = limit_config(max_audio_seconds=1)
    calls: list[Path] = []

    def fake_transcribe(audio_path: Path, params: TranscribeParams) -> TranscriptionResult:
        calls.append(audio_path)
        return TranscriptionResult(text="ok", language="de", segments=[])

    monkeypatch.setattr("transcriptions.wyoming.transcribe_audio", fake_transcribe)

    async def run_flow() -> None:
        handler = WyomingSttEventHandler(
            config,
            build_wyoming_info(config),
            asyncio.StreamReader(),
            writer,
        )
        await handler.handle_event(Transcribe(language="de").event())
        await handler.handle_event(AudioStart(rate=16000, width=2, channels=2).event())
        # 0.75 s of 16 kHz stereo: 48000 raw bytes (over the 32000-byte cap) but only
        # 24000 bytes once converted to mono.
        assert await handler.handle_event(
            AudioChunk(
                rate=16000, width=2, channels=2, audio=b"\x01\x00\x01\x00" * 12000
            ).event()
        )
        await handler.handle_event(AudioStop().event())

    asyncio.run(run_flow())

    events = decode_all_events(writer)
    assert len(events) == 1
    assert Transcript.is_type(events[0].type)
    assert len(calls) == 1


def test_wyoming_handler_transcribes_audio_exactly_at_cap(monkeypatch) -> None:
    writer = DummyWriter()
    config = limit_config(max_audio_seconds=1)
    received: list[int] = []

    def fake_transcribe(audio_path: Path, params: TranscribeParams) -> TranscriptionResult:
        with wave.open(str(audio_path), "rb") as wav_file:
            received.append(wav_file.getnframes())
        return TranscriptionResult(text="hallo", language="de", segments=[])

    monkeypatch.setattr("transcriptions.wyoming.transcribe_audio", fake_transcribe)

    async def run_flow() -> None:
        handler = WyomingSttEventHandler(
            config,
            build_wyoming_info(config),
            asyncio.StreamReader(),
            writer,
        )
        await handler.handle_event(Transcribe(language="de").event())
        await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
        assert await handler.handle_event(pcm_chunk(ONE_SECOND_PCM[:16000]))
        assert await handler.handle_event(pcm_chunk(ONE_SECOND_PCM[16000:]))
        await handler.handle_event(AudioStop().event())

    asyncio.run(run_flow())

    events = decode_all_events(writer)
    assert len(events) == 1
    assert Transcript.is_type(events[0].type)
    assert Transcript.from_event(events[0]).text == "hallo"
    assert received == [16000]


def test_wyoming_handler_rejects_text_over_tts_limit(monkeypatch, settings) -> None:
    writer = DummyWriter()
    settings.VOXHELM_TTS_MAX_INPUT_CHARS = 10
    config = limit_config()
    calls: list[str] = []
    monkeypatch.setattr(
        "transcriptions.wyoming.synthesize_text",
        lambda text, params: calls.append(text),
    )

    async def run_flow() -> bool:
        handler = WyomingSttEventHandler(
            config,
            build_wyoming_info(config),
            asyncio.StreamReader(),
            writer,
        )
        return await handler.handle_event(Synthesize(text="x" * 11).event())

    keep_open = asyncio.run(run_flow())

    assert keep_open is False
    events = decode_all_events(writer)
    assert len(events) == 1
    assert Error.is_type(events[0].type)
    error = Error.from_event(events[0])
    assert error.code == "text_too_long"
    assert "10 character limit" in (error.text or "")
    assert calls == []


def test_wyoming_handler_accepts_text_at_tts_limit(monkeypatch, settings, tmp_path: Path) -> None:
    writer = DummyWriter()
    settings.VOXHELM_TTS_MAX_INPUT_CHARS = 10
    config = limit_config()
    speech_path = tmp_path / "speech.wav"
    with wave.open(str(speech_path), "wb") as wav_file:
        wav_file.setframerate(22050)
        wav_file.setsampwidth(2)
        wav_file.setnchannels(1)
        wav_file.writeframes(b"\x01\x00" * 100)
    calls: list[str] = []

    def fake_synthesize(text: str, params: SynthesizeParams):
        calls.append(text)
        return type("SpeechResult", (), {"audio_path": speech_path})()

    monkeypatch.setattr("transcriptions.wyoming.synthesize_text", fake_synthesize)
    monkeypatch.setattr("transcriptions.wyoming.cleanup_paths", lambda *paths: None)

    async def run_flow() -> bool:
        handler = WyomingSttEventHandler(
            config,
            build_wyoming_info(config),
            asyncio.StreamReader(),
            writer,
        )
        # Surrounding whitespace is not counted, matching the HTTP endpoint.
        return await handler.handle_event(Synthesize(text="  " + "x" * 10 + "  ").event())

    keep_open = asyncio.run(run_flow())

    assert keep_open is True
    assert calls == ["x" * 10]
    events = decode_all_events(writer)
    assert AudioStart.is_type(events[0].type)
    assert AudioStop.is_type(events[-1].type)


def test_wyoming_handler_passes_stripped_text_to_synthesis(monkeypatch, settings) -> None:
    """Stripped padding is neither counted nor passed to the synthesis backend."""
    writer = DummyWriter()
    settings.VOXHELM_TTS_MAX_INPUT_CHARS = 10
    config = limit_config()
    calls: list[str] = []

    def fake_synthesize(text: str, params: SynthesizeParams):
        calls.append(text)
        raise RuntimeError("stop after capturing input")

    monkeypatch.setattr("transcriptions.wyoming.synthesize_text", fake_synthesize)

    async def run_flow() -> None:
        handler = WyomingSttEventHandler(
            config,
            build_wyoming_info(config),
            asyncio.StreamReader(),
            writer,
        )
        await handler.handle_event(Synthesize(text=" " * 100_000 + "hello" + " " * 100_000).event())

    asyncio.run(run_flow())

    assert calls == ["hello"]
