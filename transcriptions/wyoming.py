from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import signal
import tempfile
import threading
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event, async_write_event
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
    SynthesisInputError,
    SynthesizeParams,
    build_voice_registry,
    cleanup_paths,
    synthesize_text,
)

from .errors import ApiError
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
EVENT_READ_TIMEOUT_SECONDS = 30
EVENT_WRITE_TIMEOUT_SECONDS = 30
DEFAULT_MAX_CONNECTIONS = 16
DEFAULT_MAX_INFERENCE_THREADS = 4

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
    max_connections: int = DEFAULT_MAX_CONNECTIONS
    max_inference_threads: int = DEFAULT_MAX_INFERENCE_THREADS

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
    max_connections = settings.VOXHELM_WYOMING_MAX_CONNECTIONS
    if max_connections <= 0:
        raise ImproperlyConfigured("VOXHELM_WYOMING_MAX_CONNECTIONS must be a positive integer.")
    max_inference_threads = settings.VOXHELM_WYOMING_MAX_INFERENCE_THREADS
    if max_inference_threads <= 0:
        raise ImproperlyConfigured(
            "VOXHELM_WYOMING_MAX_INFERENCE_THREADS must be a positive integer."
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
        max_connections=max_connections,
        max_inference_threads=max_inference_threads,
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


@dataclass
class WyomingAdmission:
    """Process-wide connection and inference-thread admission for one server.

    The inference count is released by the worker thread itself when it finishes,
    so threads that outlive a cancelled or disconnected request keep counting.
    """

    max_connections: int = DEFAULT_MAX_CONNECTIONS
    max_inference_threads: int = DEFAULT_MAX_INFERENCE_THREADS
    connections: int = 0
    inference_threads: int = 0
    closing: bool = False
    handlers: dict[asyncio.Task, WyomingSttEventHandler] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def try_open_connection(self) -> bool:
        with self.lock:
            if self.connections >= self.max_connections:
                return False
            self.connections += 1
            return True

    def close_connection(self) -> None:
        with self.lock:
            self.connections -= 1

    def try_start_inference(self) -> bool:
        with self.lock:
            if self.inference_threads >= self.max_inference_threads:
                return False
            self.inference_threads += 1
            return True

    def finish_inference(self) -> None:
        with self.lock:
            self.inference_threads -= 1


class WyomingInferenceBusy(Exception):
    """No inference thread could be admitted for this request."""


class WyomingWriteTimeout(Exception):
    """A client did not accept an outbound event before the write deadline."""


class _OwnedResult:
    """Hand a worker result to its awaiting request, or clean it up if abandoned."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.abandoned = False
        self.result: object | None = None


async def run_owned_inference[T](
    admission: WyomingAdmission,
    func: Callable[[], T],
    *,
    cleanup: Callable[[T], None] | None = None,
) -> T:
    """Run ``func`` in a thread after admission; never leak its result on cancellation.

    Cancellation cannot stop the thread. If the awaiting request is cancelled first,
    the thread cleans up its own result when it finishes; if the thread finished
    first, the cancelled request cleans it up. Cleanup never depends on the event
    loop still running.
    """
    if not admission.try_start_inference():
        raise WyomingInferenceBusy
    owned = _OwnedResult()

    def call() -> T:
        try:
            result = func()
            with owned.lock:
                if not owned.abandoned:
                    owned.result = result
                    return result
            if cleanup is not None:
                cleanup(result)
            return result
        finally:
            admission.finish_inference()

    try:
        return await asyncio.to_thread(call)
    except asyncio.CancelledError:
        with owned.lock:
            owned.abandoned = True
            delivered = owned.result
            owned.result = None
        if delivered is not None and cleanup is not None:
            cleanup(delivered)  # type: ignore[arg-type]
        raise


def _cleanup_speech_result(result: object) -> None:
    cleanup_paths(result.audio_path)  # type: ignore[attr-defined]


def _public_failure_text(exc: Exception, fallback: str) -> str:
    """Only request errors are client-facing; other diagnostics stay in the server log."""
    if isinstance(exc, ApiError):
        return exc.message
    if isinstance(exc, SynthesisInputError):
        return str(exc)
    return fallback


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
    """Apply one elapsed deadline to the header and both bounded event bodies."""
    try:
        async with asyncio.timeout(EVENT_READ_TIMEOUT_SECONDS):
            return await _read_bounded_event(
                reader, max_data_bytes=max_data_bytes, max_payload_bytes=max_payload_bytes
            )
    except TimeoutError as exc:
        raise WyomingEventRejected("event_timeout", "Event read deadline exceeded.") from exc


async def _read_bounded_event(
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


def _validate_typed_event(event: Event) -> None:
    """Check consumed fields before library dataclass parsing or handler state changes."""
    data = event.data
    invalid = False
    if Synthesize.is_type(event.type):
        invalid = not isinstance(data.get("text"), str)
        voice = data.get("voice", {})
        if not isinstance(voice, dict):
            invalid = True
        else:
            invalid |= any(
                value is not None and not isinstance(value, str)
                for value in (voice.get(key) for key in ("name", "language", "speaker"))
            )
    elif Transcribe.is_type(event.type):
        invalid = any(
            value is not None and not isinstance(value, str)
            for value in (data.get("name"), data.get("language"))
        )
    elif AudioStart.is_type(event.type) or AudioChunk.is_type(event.type):
        invalid = any(
            type(data.get(key)) is not int or data[key] <= 0
            for key in ("rate", "width", "channels")
        )
        if not invalid:
            invalid = (
                not 8000 <= data["rate"] <= 192000
                or data["width"] not in (1, 2, 4)
                or data["channels"] not in (1, 2)
            )
    if (
        AudioStart.is_type(event.type)
        or AudioChunk.is_type(event.type)
        or AudioStop.is_type(event.type)
    ):
        timestamp = data.get("timestamp")
        invalid |= timestamp is not None and type(timestamp) is not int
    if invalid:
        raise WyomingEventRejected("invalid_event", "Invalid event fields.")


class WyomingSttEventHandler(AsyncEventHandler):
    def __init__(
        self,
        config: WyomingSttConfig,
        info: Info,
        *args,
        admission: WyomingAdmission | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.admission = admission or WyomingAdmission(
            max_connections=config.max_connections,
            max_inference_threads=config.max_inference_threads,
        )
        self.info_event = info.event()
        self._reset_audio()
        self.request_model = config.model
        self.request_language = config.language

    def _reset_audio(self) -> None:
        self.audio_buffer = io.BytesIO()
        self.audio_converter = AudioChunkConverter(
            rate=_CONVERTED_RATE,
            width=_CONVERTED_WIDTH,
            channels=_CONVERTED_CHANNELS,
        )
        self.audio_shape = WyomingAudioShape()

    async def run(self) -> None:
        """Receive events like ``AsyncEventHandler.run``, but through the bounded reader.

        Replaces the library loop (wyoming 1.8.0) so that a client-declared length can never
        make the process buffer more than the configured per-event limits, and closes the
        connection when the loop ends.
        """
        self._is_running = True
        task = asyncio.current_task()
        admitted = not self.admission.closing and self.admission.try_open_connection()
        if task is not None:
            self.admission.handlers[task] = self
        try:
            if not admitted:
                if self.admission.closing:
                    _LOGGER.info("Wyoming connection refused: server is shutting down")
                else:
                    _LOGGER.warning(
                        "Wyoming connection refused: %s active connections",
                        self.config.max_connections,
                    )
                await self.write_event(
                    Error(text="Too many active connections.", code="server_busy").event()
                )
                return
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
        except (WyomingWriteTimeout, ConnectionError) as exc:
            _LOGGER.warning("Wyoming client write failed: %s", type(exc).__name__)
        finally:
            if admitted:
                self.admission.close_connection()
            if task is not None:
                self.admission.handlers.pop(task, None)
            try:
                await self.disconnect()
            finally:
                # The library leaves the transport open when the loop ends; close it so
                # a refused, finished or cancelled client does not keep the socket.
                self.writer.close()

    def abort(self) -> None:
        """Drop the connection immediately, discarding any unsent output."""
        self._is_running = False
        transport = getattr(self.writer, "transport", None)
        if transport is not None:
            transport.abort()
        else:
            self.writer.close()

    async def write_event(self, event: Event) -> None:
        """Write one event, aborting the connection if the client stops reading."""
        try:
            async with asyncio.timeout(EVENT_WRITE_TIMEOUT_SECONDS):
                await async_write_event(event, self.writer)
        except TimeoutError as exc:
            self.abort()
            raise WyomingWriteTimeout("Outbound event write deadline exceeded.") from exc

    async def _write_busy(self) -> None:
        _LOGGER.warning(
            "Wyoming inference refused: %s inference threads active",
            self.config.max_inference_threads,
        )
        await self.write_event(Error(text="Inference capacity is busy.", code="busy").event())

    async def handle_event(self, event: Event) -> bool:
        try:
            _validate_typed_event(event)
        except WyomingEventRejected as exc:
            _LOGGER.warning("Wyoming event rejected (%s): %s", exc.code, exc.text)
            await self.write_event(Error(text=exc.text, code=exc.code).event())
            return False

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
            params = SynthesizeParams(
                request_model="auto",
                voice=requested_voice,
                language=requested_language,
                speed=1.0,
                scheduler_lane=LANE_INTERACTIVE,
            )
            try:
                speech_result = await run_owned_inference(
                    self.admission,
                    lambda: synthesize_text(text, params),
                    cleanup=_cleanup_speech_result,
                )
                await self._write_synthesized_audio(speech_result.audio_path)
            except WyomingInferenceBusy:
                await self._write_busy()
                return False
            except WyomingWriteTimeout, ConnectionError:
                raise
            except Exception as exc:
                _LOGGER.exception("Wyoming TTS synthesis failed")
                await self.write_event(
                    Error(
                        text=_public_failure_text(exc, "Speech synthesis failed."),
                        code="synthesis_failed",
                    ).event()
                )
            finally:
                if speech_result is not None:
                    cleanup_paths(speech_result.audio_path)
            return True

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            self.request_model = (transcribe.name or self.config.model).strip()
            self.request_language = transcribe.language or self.config.language
            self._reset_audio()
            return True

        if AudioStart.is_type(event.type):
            start = AudioStart.from_event(event)
            self._reset_audio()
            self.audio_shape = WyomingAudioShape(
                input_rate=start.rate,
                input_width=start.width,
                input_channels=start.channels,
            )
            return True

        if AudioChunk.is_type(event.type):
            raw_chunk = AudioChunk.from_event(event)
            if (raw_chunk.rate, raw_chunk.width, raw_chunk.channels) != (
                self.audio_shape.input_rate,
                self.audio_shape.input_width,
                self.audio_shape.input_channels,
            ) or len(raw_chunk.audio) % (raw_chunk.width * raw_chunk.channels):
                await self.write_event(
                    Error(
                        text="Invalid audio chunk geometry or sequence.", code="invalid_event"
                    ).event()
                )
                return False

            input_bytes = self.audio_shape.input_bytes + len(raw_chunk.audio)
            input_frames = input_bytes // (raw_chunk.width * raw_chunk.channels)
            # A fresh fixed-geometry stream produces at most ceil(N * out_rate / in_rate)
            # frames. Use cumulative frames so chunk rounding cannot erode the budget.
            output_frames_bound = (
                input_frames * _CONVERTED_RATE + raw_chunk.rate - 1
            ) // raw_chunk.rate
            max_audio_bytes = self.config.max_audio_bytes
            if output_frames_bound * _CONVERTED_WIDTH > max_audio_bytes:
                _LOGGER.warning(
                    "Wyoming STT audio rejected: buffered audio exceeds limit of %s seconds",
                    self.config.max_audio_seconds,
                )
                self._reset_audio()
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
            chunk = self.audio_converter.convert(raw_chunk)
            self.audio_shape.input_bytes = input_bytes
            self.audio_shape.converted_bytes += len(chunk.audio)
            self.audio_buffer.write(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            if self.audio_buffer.tell() == 0:
                await self.write_event(
                    Error(text="No audio payload received.", code="empty_audio").event()
                )
                return False

            audio_bytes = self.audio_buffer.getvalue()
            try:
                started_at = time.monotonic()
                result = await run_owned_inference(
                    self.admission,
                    lambda: self._transcribe_current_audio(audio_bytes),
                )
                raw_transcript = result.text
                if settings.VOXHELM_WYOMING_STT_NORMALIZE_TRANSCRIPT:
                    normalized_text = normalize_interactive_transcript(
                        result.text,
                        language=result.language or self.request_language,
                    )
                    if normalized_text and normalized_text != result.text:
                        result = replace(result, text=normalized_text)
            except WyomingInferenceBusy:
                await self._write_busy()
                return False
            except Exception as exc:
                _LOGGER.exception(
                    "Wyoming STT transcription failed model=%s language=%s audio=%s",
                    self.request_model,
                    self.request_language or "auto",
                    self.audio_shape.as_dict(),
                )
                await self.write_event(
                    Error(
                        text=_public_failure_text(exc, "Transcription failed."),
                        code="transcription_failed",
                    ).event()
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
    admission = WyomingAdmission(
        max_connections=resolved_config.max_connections,
        max_inference_threads=resolved_config.max_inference_threads,
    )

    def request_stop() -> None:
        stop_event.set()

    installed_signals: list[signal.Signals] = []
    for handled_signal in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(handled_signal, request_stop)
        except NotImplementedError, RuntimeError, ValueError:  # pragma: no cover
            continue
        installed_signals.append(handled_signal)

    def handler_factory(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> WyomingSttEventHandler:
        return WyomingSttEventHandler(resolved_config, info, reader, writer, admission=admission)

    try:
        await server.start(handler_factory)
        # Everything after a successful start runs under the finally, so a logging
        # failure or cancellation still stops the listener and owned handlers.
        try:
            _LOGGER.info(
                "Wyoming STT/TTS listening on %s:%s using backend=%s model=%s language=%s",
                resolved_config.host,
                resolved_config.port,
                resolved_config.backend,
                resolved_config.model,
                resolved_config.language or "auto",
            )
            await stop_event.wait()
        finally:
            await shutdown_wyoming_server(server, admission)
    finally:
        # Remove the SIGINT/SIGTERM handlers installed above so the caller's loop keeps
        # no stale reference to this server's stop event, but leave a handler someone
        # registered after us (another sidecar on the same loop) in place. asyncio has
        # one handler per signal and no public lookup, so a handler registered before
        # is replaced while the sidecar runs and not restored afterwards.
        for handled_signal in installed_signals:
            if _loop_signal_callback(loop, handled_signal) in (request_stop, None):
                loop.remove_signal_handler(handled_signal)


def _loop_signal_callback(loop: asyncio.AbstractEventLoop, handled: signal.Signals) -> object:
    """The callback currently registered for ``handled`` on ``loop``, if visible.

    Reads the private registry of the Unix selector loop; ``None`` when it is not
    available (other loop implementations), in which case the caller removes the
    handler it installed.
    """
    handlers = getattr(loop, "_signal_handlers", None)
    if not isinstance(handlers, dict):
        return None
    return getattr(handlers.get(handled), "_callback", None)


async def shutdown_wyoming_server(server: AsyncServer, admission: WyomingAdmission) -> None:
    """Abort and cancel every handler, stop listening, then drop late arrivals.

    Aborting (rather than closing) discards output buffered for a client that stopped
    reading, so no half-closed transport lingers. Runs on normal stop and when the
    serving task is cancelled. Inference threads cannot be interrupted; their results
    are cleaned up when they finish.
    """

    async def cancel_handlers() -> None:
        handlers = list(admission.handlers.items())
        for task, handler in handlers:
            handler.abort()
            task.cancel()
        if handlers:
            await asyncio.gather(*(task for task, _ in handlers), return_exceptions=True)

    async def shutdown() -> None:
        # Handlers that start from now on refuse immediately and close.
        admission.closing = True
        await cancel_handlers()
        await server.stop()
        # Best effort for handlers that already started; any later one sees
        # ``closing`` and refuses on its own, so correctness does not rely on this yield.
        await asyncio.sleep(0)
        await cancel_handlers()

    await asyncio.shield(shutdown())


def main() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(run_wyoming_stt_server())
