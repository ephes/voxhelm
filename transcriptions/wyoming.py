from __future__ import annotations

import asyncio
import io
import logging
import os
import signal
import tempfile
import time
import wave
from dataclasses import dataclass, replace
from pathlib import Path

from django.conf import settings
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import (
    AsrModel,
    AsrProgram,
    Attribution,
    Describe,
    Info,
    TtsProgram,
    TtsVoice,
    TtsVoiceSpeaker,
)
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import Synthesize

from synthesis.service import (
    SynthesizeParams,
    cleanup_paths,
    discover_installed_voices,
    synthesize_text,
)

from .observability import emit_transcription_debug_log
from .service import (
    TranscribeParams,
    normalize_interactive_transcript,
    resolve_model_name_for_backend,
    transcribe_audio,
)

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class WyomingSttConfig:
    host: str
    port: int
    backend: str
    model: str
    language: str | None
    languages: tuple[str, ...]
    prompt: str | None

    @property
    def uri(self) -> str:
        return f"tcp://{self.host}:{self.port}"


@dataclass
class WyomingAudioShape:
    input_rate: int | None = None
    input_width: int | None = None
    input_channels: int | None = None
    input_bytes: int = 0
    converted_rate: int = 16000
    converted_width: int = 2
    converted_channels: int = 1
    converted_bytes: int = 0

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "converted": {
                "bytes": self.converted_bytes,
                "channels": self.converted_channels,
                "duration_seconds": _audio_duration_seconds(
                    byte_count=self.converted_bytes,
                    rate=self.converted_rate,
                    width=self.converted_width,
                    channels=self.converted_channels,
                ),
                "rate": self.converted_rate,
                "width": self.converted_width,
            }
        }
        if self.input_rate is not None:
            payload["input"] = {
                "bytes": self.input_bytes,
                "channels": self.input_channels,
                "duration_seconds": _audio_duration_seconds(
                    byte_count=self.input_bytes,
                    rate=self.input_rate,
                    width=self.input_width,
                    channels=self.input_channels,
                ),
                "rate": self.input_rate,
                "width": self.input_width,
            }
        return payload


def _audio_duration_seconds(
    *,
    byte_count: int,
    rate: int | None,
    width: int | None,
    channels: int | None,
) -> float | None:
    if not rate or not width or not channels:
        return None
    bytes_per_second = rate * width * channels
    if bytes_per_second <= 0:
        return None
    return round(byte_count / bytes_per_second, 3)


def get_wyoming_stt_config() -> WyomingSttConfig:
    backend = settings.VOXHELM_WYOMING_STT_BACKEND or settings.VOXHELM_STT_BACKEND
    model = settings.VOXHELM_WYOMING_STT_MODEL or resolve_model_name_for_backend(
        request_model="auto",
        backend_name=backend,
    )
    language = settings.VOXHELM_WYOMING_STT_LANGUAGE or None
    languages = tuple(dict.fromkeys(settings.VOXHELM_WYOMING_STT_LANGUAGES))
    if not languages:
        languages = (language,) if language else ("en",)
    return WyomingSttConfig(
        host=settings.VOXHELM_WYOMING_STT_HOST,
        port=settings.VOXHELM_WYOMING_STT_PORT,
        backend=backend,
        model=model,
        language=language,
        languages=languages,
        prompt=settings.VOXHELM_WYOMING_STT_PROMPT or None,
    )


def build_wyoming_info(config: WyomingSttConfig) -> Info:
    return Info(
        asr=[
            AsrProgram(
                name="voxhelm",
                description="Voxhelm Wyoming speech-to-text",
                attribution=Attribution(
                    name="Voxhelm",
                    url="https://github.com/jochen/Voxhelm",
                ),
                installed=True,
                version="0.1.0",
                models=[
                    AsrModel(
                        name=config.model,
                        description=f"{config.backend} via Voxhelm",
                        attribution=Attribution(
                            name=config.backend,
                            url="https://github.com/rhasspy/wyoming",
                        ),
                        installed=True,
                        languages=list(config.languages),
                        version="0.1.0",
                    )
                ],
            )
        ],
        tts=[
            TtsProgram(
                name="voxhelm",
                description="Voxhelm Wyoming text-to-speech",
                attribution=Attribution(
                    name="Voxhelm",
                    url="https://github.com/jochen/Voxhelm",
                ),
                installed=True,
                version="0.1.0",
                voices=[
                    TtsVoice(
                        name=voice.key,
                        description=voice.name,
                        attribution=Attribution(
                            name="Piper",
                            url="https://github.com/OHF-Voice/piper1-gpl",
                        ),
                        installed=True,
                        version="0.1.0",
                        languages=list(voice.languages),
                        speakers=[TtsVoiceSpeaker(name=speaker) for speaker in voice.speakers]
                        or None,
                    )
                    for voice in discover_installed_voices(
                        voice_dir=settings.VOXHELM_PIPER_VOICE_DIR,
                        configured_voices=list(settings.VOXHELM_PIPER_VOICES),
                    ).values()
                ],
                supports_synthesize_streaming=False,
            )
        ],
    )


class WyomingSttEventHandler(AsyncEventHandler):
    def __init__(
        self,
        config: WyomingSttConfig,
        info: Info,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.info_event = info.event()
        self.audio_buffer = io.BytesIO()
        self.audio_converter = AudioChunkConverter(rate=16000, width=2, channels=1)
        self.request_model = config.model
        self.request_language = config.language
        self.audio_shape = WyomingAudioShape()

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.info_event)
            return True

        if Synthesize.is_type(event.type):
            synthesize = Synthesize.from_event(event)
            requested_voice = None
            requested_language = None
            speech_result = None
            if synthesize.voice is not None:
                requested_voice = synthesize.voice.name
                requested_language = synthesize.voice.language
            try:
                speech_result = await asyncio.to_thread(
                    synthesize_text,
                    synthesize.text,
                    SynthesizeParams(
                        request_model="auto",
                        voice=requested_voice,
                        language=requested_language,
                        speed=1.0,
                    ),
                )
                await self._write_synthesized_audio(speech_result.audio_path)
            except Exception as exc:
                _LOGGER.exception("Wyoming TTS synthesis failed")
                await self.write_event(Error(text=str(exc), code="synthesis_failed").event())
            finally:
                if speech_result is not None:
                    cleanup_paths(speech_result.audio_path)
            return True

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            self.request_model = (transcribe.name or self.config.model).strip()
            self.request_language = transcribe.language or self.config.language
            self.audio_buffer = io.BytesIO()
            self.audio_shape = WyomingAudioShape()
            return True

        if AudioStart.is_type(event.type):
            start = AudioStart.from_event(event)
            self.audio_shape = WyomingAudioShape(
                input_rate=start.rate,
                input_width=start.width,
                input_channels=start.channels,
            )
            self.audio_buffer = io.BytesIO()
            return True

        if AudioChunk.is_type(event.type):
            raw_chunk = AudioChunk.from_event(event)
            self.audio_shape.input_bytes += len(raw_chunk.audio)
            chunk = self.audio_converter.convert(raw_chunk)
            self.audio_shape.converted_bytes += len(chunk.audio)
            self.audio_buffer.write(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            if self.audio_buffer.tell() == 0:
                await self.write_event(
                    Error(text="No audio payload received.", code="empty_audio").event()
                )
                return False

            try:
                started_at = time.monotonic()
                result = await asyncio.to_thread(
                    self._transcribe_current_audio,
                    self.audio_buffer.getvalue(),
                )
                raw_transcript = result.text
                if settings.VOXHELM_WYOMING_STT_NORMALIZE_TRANSCRIPT:
                    normalized_text = normalize_interactive_transcript(
                        result.text,
                        language=result.language or self.request_language,
                    )
                    if normalized_text and normalized_text != result.text:
                        result = replace(result, text=normalized_text)
            except Exception as exc:
                _LOGGER.exception(
                    "Wyoming STT transcription failed model=%s language=%s audio=%s",
                    self.request_model,
                    self.request_language or "auto",
                    self.audio_shape.as_dict(),
                )
                await self.write_event(
                    Error(text=str(exc), code="transcription_failed").event()
                )
                return False

            emit_transcription_debug_log(
                source="wyoming",
                audio_shape=self.audio_shape.as_dict(),
                request_model=self.request_model,
                request_language=self.request_language,
                prompt=self.config.prompt,
                result=result,
                duration_ms=int((time.monotonic() - started_at) * 1000),
                raw_transcript=raw_transcript,
            )
            await self.write_event(
                Transcript(
                    text=result.text,
                    language=result.language or self.request_language,
                ).event()
            )
            return False

        return True

    def _transcribe_current_audio(self, audio_bytes: bytes):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            temp_path = Path(handle.name)
        try:
            with wave.open(str(temp_path), "wb") as wav_file:
                wav_file.setframerate(16000)
                wav_file.setsampwidth(2)
                wav_file.setnchannels(1)
                wav_file.writeframes(audio_bytes)

            return transcribe_audio(
                temp_path,
                TranscribeParams(
                    request_model=self.request_model,
                    prompt=self.config.prompt,
                    language=self.request_language,
                ),
            )
        finally:
            temp_path.unlink(missing_ok=True)

    async def _write_synthesized_audio(self, audio_path: Path) -> None:
        with wave.open(str(audio_path), "rb") as wav_reader:
            rate = wav_reader.getframerate()
            width = wav_reader.getsampwidth()
            channels = wav_reader.getnchannels()

            await self.write_event(
                AudioStart(
                    rate=rate,
                    width=width,
                    channels=channels,
                ).event()
            )

            frames_per_chunk = settings.VOXHELM_WYOMING_SAMPLES_PER_CHUNK
            while True:
                chunk = wav_reader.readframes(frames_per_chunk)
                if not chunk:
                    break
                await self.write_event(
                    AudioChunk(
                        audio=chunk,
                        rate=rate,
                        width=width,
                        channels=channels,
                    ).event()
                )

        await self.write_event(AudioStop().event())


async def run_wyoming_stt_server(*, config: WyomingSttConfig | None = None) -> None:
    resolved_config = config or get_wyoming_stt_config()
    info = build_wyoming_info(resolved_config)
    server = AsyncServer.from_uri(resolved_config.uri)
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    for handled_signal in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(handled_signal, stop_event.set)
        except NotImplementedError:  # pragma: no cover
            pass

    def handler_factory(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> WyomingSttEventHandler:
        return WyomingSttEventHandler(resolved_config, info, reader, writer)

    await server.start(handler_factory)
    _LOGGER.info(
        "Wyoming STT/TTS listening on %s:%s using backend=%s model=%s language=%s",
        resolved_config.host,
        resolved_config.port,
        resolved_config.backend,
        resolved_config.model,
        resolved_config.language or "auto",
    )
    await stop_event.wait()
    await server.stop()


def main() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(run_wyoming_stt_server())
