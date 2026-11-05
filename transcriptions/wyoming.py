from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import signal
import tempfile
import time
import wave
from dataclasses import dataclass, replace
from pathlib import Path

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
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

from lane_scheduler import LANE_INTERACTIVE
from synthesis.service import (
    SynthesizeParams,
    build_voice_registry,
    cleanup_paths,
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

# Audio is buffered after conversion to 16 kHz, 16-bit, mono PCM.
_CONVERTED_RATE = 16000
_CONVERTED_WIDTH = 2
_CONVERTED_CHANNELS = 1
_CONVERTED_BYTES_PER_SECOND = _CONVERTED_RATE * _CONVERTED_WIDTH * _CONVERTED_CHANNELS
DEFAULT_MAX_AUDIO_SECONDS = 120
DEFAULT_MAX_EVENT_DATA_BYTES = 256 * 1024

_PIPER_ATTRIBUTION = Attribution(
    name="Piper",
    url="https://github.com/OHF-Voice/piper1-gpl",
)
_VOXHELM_ATTRIBUTION = Attribution(
    name="Voxhelm",
    url="https://github.com/jochen/Voxhelm",
)
_KOKORO_ATTRIBUTION = Attribution(
    name="Kokoro",
    url="https://github.com/thewh1teagle/kokoro-onnx",
)


def _tts_voice_attribution(backend: str) -> Attribution:
    if backend == "piper":
        return _PIPER_ATTRIBUTION
    if backend == "kokoro":
        return _KOKORO_ATTRIBUTION
    return _VOXHELM_ATTRIBUTION


@dataclass(frozen=True)
class WyomingSttConfig:
    host: str
    port: int
    backend: str
    model: str
    language: str | None
    languages: tuple[str, ...]
    prompt: str | None
    max_audio_seconds: int = DEFAULT_MAX_AUDIO_SECONDS
    max_event_data_bytes: int = DEFAULT_MAX_EVENT_DATA_BYTES
    # None derives the per-event payload limit from the STT audio cap.
    max_event_payload_bytes_override: int | None = None

    @property
    def max_audio_bytes(self) -> int:
        return self.max_audio_seconds * _CONVERTED_BYTES_PER_SECOND

    @property
    def max_event_payload_bytes(self) -> int:
        if self.max_event_payload_bytes_override is not None:
            return self.max_event_payload_bytes_override
        return self.max_audio_bytes

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
    max_audio_seconds = settings.VOXHELM_WYOMING_STT_MAX_AUDIO_SECONDS
    if max_audio_seconds <= 0:
        raise ImproperlyConfigured(
            "VOXHELM_WYOMING_STT_MAX_AUDIO_SECONDS must be a positive integer."
        )
    max_event_data_bytes = settings.VOXHELM_WYOMING_MAX_EVENT_DATA_BYTES
    if max_event_data_bytes <= 0:
        raise ImproperlyConfigured(
            "VOXHELM_WYOMING_MAX_EVENT_DATA_BYTES must be a positive integer."
        )
    max_event_payload_bytes = settings.VOXHELM_WYOMING_MAX_EVENT_PAYLOAD_BYTES
    if max_event_payload_bytes is not None and max_event_payload_bytes <= 0:
        raise ImproperlyConfigured(
            "VOXHELM_WYOMING_MAX_EVENT_PAYLOAD_BYTES must be a positive integer."
        )
    return WyomingSttConfig(
        host=settings.VOXHELM_WYOMING_STT_HOST,
        port=settings.VOXHELM_WYOMING_STT_PORT,
        backend=backend,
        model=model,
        language=language,
        languages=languages,
        prompt=settings.VOXHELM_WYOMING_STT_PROMPT or None,
        max_audio_seconds=max_audio_seconds,
        max_event_data_bytes=max_event_data_bytes,
        max_event_payload_bytes_override=max_event_payload_bytes,
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
                        attribution=_tts_voice_attribution(voice.backend),
                        installed=True,
                        version="0.1.0",
                        languages=list(voice.languages),
                        speakers=[TtsVoiceSpeaker(name=speaker) for speaker in voice.speakers]
                        or None,
                    )
                    for voice in build_voice_registry().voices
                ],
                supports_synthesize_streaming=False,
            )
        ],
    )


class WyomingEventRejected(Exception):
    """A client event was refused before its body was read."""

    def __init__(self, code: str, text: str) -> None:
        super().__init__(text)
        self.code = code
        self.text = text


def _declared_length(header: dict, key: str, limit: int) -> int:
    value = header.get(key)
    if value is None:
        return 0
    # bool is an int subclass; JSON true/false is not a length.
    if type(value) is not int or value < 0:
        raise WyomingEventRejected("invalid_event", f"Invalid {key} in event header.")
    if value > limit:
        raise WyomingEventRejected(
            "event_too_large",
            f"Event {key} {value} exceeds the configured limit of {limit} bytes.",
        )
    return value


async def read_bounded_event(
    reader: asyncio.StreamReader,
    *,
    max_data_bytes: int,
    max_payload_bytes: int,
) -> Event | None:
    """Read one Wyoming event, refusing oversized or malformed ones before reading their body.

    Mirrors ``wyoming.event.async_read_event`` (wyoming 1.8.0), whose ``readexactly`` calls
    trust the client-declared ``data_length`` and ``payload_length``. The header line itself
    is bounded by the stream reader's limit (64 KiB by default). Returns ``None`` when the
    client disconnects, and raises ``WyomingEventRejected`` for an event that must be refused.
    """
    try:
        json_line = await reader.readline()
    except ValueError as exc:  # header line longer than the stream reader limit
        raise WyomingEventRejected(
            "event_too_large", "Event header exceeds the line length limit."
        ) from exc
    if not json_line:
        return None

    try:
        header = json.loads(json_line)
    except ValueError as exc:
        raise WyomingEventRejected("invalid_event", "Event header is not valid JSON.") from exc
    if not isinstance(header, dict) or not isinstance(header.get("type"), str):
        raise WyomingEventRejected("invalid_event", "Event header has no valid type.")
    data = header.get("data")
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise WyomingEventRejected("invalid_event", "Event data is not a JSON object.")

    data_length = _declared_length(header, "data_length", max_data_bytes)
    payload_length = _declared_length(header, "payload_length", max_payload_bytes)

    try:
        if data_length > 0:
            data_bytes = await reader.readexactly(data_length)
            try:
                extra_data = json.loads(data_bytes)
            except ValueError as exc:
                raise WyomingEventRejected(
                    "invalid_event", "Event data is not valid JSON."
                ) from exc
            if not isinstance(extra_data, dict):
                raise WyomingEventRejected("invalid_event", "Event data is not a JSON object.")
            data.update(extra_data)

        payload: bytes | None = None
        if payload_length > 0:
            payload = await reader.readexactly(payload_length)
    except asyncio.IncompleteReadError:
        # The client disconnected in the middle of an event.
        return None

    return Event(type=header["type"], data=data, payload=payload)


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
        self.audio_converter = AudioChunkConverter(
            rate=_CONVERTED_RATE,
            width=_CONVERTED_WIDTH,
            channels=_CONVERTED_CHANNELS,
        )
        self.request_model = config.model
        self.request_language = config.language
        self.audio_shape = WyomingAudioShape()

    async def run(self) -> None:
        """Receive events like ``AsyncEventHandler.run``, but through the bounded reader.

        Replaces the library loop (wyoming 1.8.0) so that a client-declared length can never
        make the process buffer more than the configured per-event limits, and closes the
        connection when the loop ends.
        """
        self._is_running = True
        try:
            while self._is_running:
                try:
                    event = await read_bounded_event(
                        self.reader,
                        max_data_bytes=self.config.max_event_data_bytes,
                        max_payload_bytes=self.config.max_event_payload_bytes,
                    )
                except WyomingEventRejected as exc:
                    _LOGGER.warning("Wyoming event rejected (%s): %s", exc.code, exc.text)
                    await self.write_event(Error(text=exc.text, code=exc.code).event())
                    break
                if event is None:
                    break
                if not (await self.handle_event(event)):
                    break
        finally:
            await self.disconnect()
            # The library leaves the transport open when the loop ends; close it so a
            # refused or finished client does not keep the socket.
            self.writer.close()

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.info_event)
            return True

        if Synthesize.is_type(event.type):
            synthesize = Synthesize.from_event(event)
            # Strip once and synthesize the validated text, as the HTTP endpoint does.
            text = synthesize.text.strip()
            max_chars = settings.VOXHELM_TTS_MAX_INPUT_CHARS
            if len(text) > max_chars:
                _LOGGER.warning(
                    "Wyoming TTS text rejected: %s characters exceeds limit of %s",
                    len(text),
                    max_chars,
                )
                await self.write_event(
                    Error(
                        text=f"Input text exceeded the configured {max_chars} character limit.",
                        code="text_too_long",
                    ).event()
                )
                return False
            requested_voice = None
            requested_language = None
            speech_result = None
            if synthesize.voice is not None:
                requested_voice = synthesize.voice.name
                requested_language = synthesize.voice.language
            try:
                speech_result = await asyncio.to_thread(
                    synthesize_text,
                    text,
                    SynthesizeParams(
                        request_model="auto",
                        voice=requested_voice,
                        language=requested_language,
                        speed=1.0,
                        scheduler_lane=LANE_INTERACTIVE,
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
            max_audio_bytes = self.config.max_audio_bytes
            if self.audio_buffer.tell() + len(chunk.audio) > max_audio_bytes:
                _LOGGER.warning(
                    "Wyoming STT audio rejected: buffered audio exceeds limit of %s seconds",
                    self.config.max_audio_seconds,
                )
                self.audio_buffer = io.BytesIO()
                self.audio_shape = WyomingAudioShape()
                await self.write_event(
                    Error(
                        text=(
                            "Audio exceeded the configured "
                            f"{self.config.max_audio_seconds} second limit."
                        ),
                        code="audio_too_long",
                    ).event()
                )
                return False
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
                await self.write_event(Error(text=str(exc), code="transcription_failed").event())
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
                wav_file.setframerate(_CONVERTED_RATE)
                wav_file.setsampwidth(_CONVERTED_WIDTH)
                wav_file.setnchannels(_CONVERTED_CHANNELS)
                wav_file.writeframes(audio_bytes)

            return transcribe_audio(
                temp_path,
                TranscribeParams(
                    request_model=self.request_model,
                    prompt=self.config.prompt,
                    language=self.request_language,
                    scheduler_lane=LANE_INTERACTIVE,
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
