from __future__ import annotations

import asyncio
import io
import logging
import os
import signal
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

from django.conf import settings
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

from .service import TranscribeParams, resolve_model_name_for_backend, transcribe_audio

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class WyomingSttConfig:
    host: str
    port: int
    backend: str
    model: str
    language: str | None
    languages: tuple[str, ...]

    @property
    def uri(self) -> str:
        return f"tcp://{self.host}:{self.port}"


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
        ]
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

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.info_event)
            return True

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            self.request_model = (transcribe.name or self.config.model).strip()
            self.request_language = transcribe.language or self.config.language
            self.audio_buffer = io.BytesIO()
            return True

        if AudioStart.is_type(event.type):
            self.audio_buffer = io.BytesIO()
            return True

        if AudioChunk.is_type(event.type):
            chunk = self.audio_converter.convert(AudioChunk.from_event(event))
            self.audio_buffer.write(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            if self.audio_buffer.tell() == 0:
                await self.write_event(
                    Error(text="No audio payload received.", code="empty_audio").event()
                )
                return False

            try:
                result = await asyncio.to_thread(
                    self._transcribe_current_audio,
                    self.audio_buffer.getvalue(),
                )
            except Exception as exc:
                _LOGGER.exception("Wyoming STT transcription failed")
                await self.write_event(
                    Error(text=str(exc), code="transcription_failed").event()
                )
                return False

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
                    prompt=None,
                    language=self.request_language,
                ),
            )
        finally:
            temp_path.unlink(missing_ok=True)


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
        "Wyoming STT listening on %s:%s using backend=%s model=%s language=%s",
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
