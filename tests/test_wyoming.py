from __future__ import annotations

import asyncio
import io
import wave
from pathlib import Path

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import read_event
from wyoming.info import Describe, Info

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


def test_get_wyoming_stt_config_uses_dedicated_defaults(settings) -> None:
    settings.VOXHELM_STT_BACKEND = "whispercpp"
    settings.VOXHELM_WHISPERCPP_MODEL = "ggml-large-v3.bin"
    settings.VOXHELM_WYOMING_STT_BACKEND = "mlx"
    settings.VOXHELM_MLX_MODEL = "mlx-community/whisper-large-v3-mlx"
    settings.VOXHELM_WYOMING_STT_MODEL = ""
    settings.VOXHELM_WYOMING_STT_LANGUAGE = "de"
    settings.VOXHELM_WYOMING_STT_LANGUAGES = ["de", "en"]

    config = get_wyoming_stt_config()

    assert config.backend == "mlx"
    assert config.model == "mlx-community/whisper-large-v3-mlx"
    assert config.language == "de"
    assert config.languages == ("de", "en")


def test_build_wyoming_info_announces_model_and_languages() -> None:
    info = build_wyoming_info(
        WyomingSttConfig(
            host="0.0.0.0",
            port=10300,
            backend="whispercpp",
            model="ggml-large-v3.bin",
            language="de",
            languages=("de", "en"),
        )
    )

    asr_program = info.asr[0]
    assert asr_program.name == "voxhelm"
    assert asr_program.models[0].name == "ggml-large-v3.bin"
    assert asr_program.models[0].languages == ["de", "en"]


def test_wyoming_handler_describe_returns_info() -> None:
    writer = DummyWriter()
    config = WyomingSttConfig(
        host="127.0.0.1",
        port=10300,
        backend="whispercpp",
        model="ggml-large-v3.bin",
        language="de",
        languages=("de",),
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


def test_wyoming_handler_returns_error_for_empty_audio() -> None:
    writer = DummyWriter()
    config = WyomingSttConfig(
        host="127.0.0.1",
        port=10300,
        backend="whispercpp",
        model="ggml-large-v3.bin",
        language=None,
        languages=("en",),
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
