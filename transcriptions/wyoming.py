from __future__ import annotations

import array
import asyncio
import fcntl
import io
import json
import logging
import math
import os
import re
import signal
import stat
import sys
import tempfile
import threading
import time
import wave
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from uuid import uuid4

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

from lane_scheduler import LANE_INTERACTIVE, InferenceCancelled
from synthesis.service import (
    SynthesisInputError,
    SynthesizeParams,
    build_voice_registry,
    cleanup_paths,
    detect_routing_language,
    synthesize_text,
)

from .errors import ApiError
from .observability import emit_transcription_debug_log
from .service import (
    PARAKEET_INPROCESS_BACKEND_NAME,
    BackendUnavailableError,
    TranscribeParams,
    normalize_interactive_transcript,
    resolve_model_name_for_backend,
    transcribe_audio,
)

_LOGGER = logging.getLogger(__name__)


def _best_effort_warning(message: str) -> None:
    with suppress(Exception):
        _LOGGER.warning(message)


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
    # Run one silent inference at startup so the model is loaded before the first
    # voice turn (VOXHELM_WYOMING_STT_WARMUP). Off for directly built configs.
    warm_up: bool = False

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
        warm_up=settings.VOXHELM_WYOMING_STT_WARMUP,
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


_RECORDING_QUEUE_SIZE = 4
_RECORDING_DRAIN_SECONDS = 2


class _RecordingWriter:
    """Own a single off-loop write and a bounded queue for the listener."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[Callable[[], None]] = asyncio.Queue(_RECORDING_QUEUE_SIZE)

    def submit(self, write: Callable[[], None]) -> None:
        with suppress(asyncio.QueueFull):
            self.queue.put_nowait(write)

    async def run(self) -> None:
        while True:
            write = await self.queue.get()
            try:
                await asyncio.to_thread(write)
            finally:
                self.queue.task_done()

    async def close(self, worker: asyncio.Task[None]) -> None:
        try:
            async with asyncio.timeout(_RECORDING_DRAIN_SECONDS):
                await self.queue.join()
        except TimeoutError:
            _best_effort_warning("Wyoming recording shutdown timed out; queued captures dropped")
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)


class WyomingSttEventHandler(AsyncEventHandler):
    def __init__(
        self,
        config: WyomingSttConfig,
        info: Info,
        *args,
        admission: WyomingAdmission | None = None,
        recording_writer: _RecordingWriter | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.admission = admission or WyomingAdmission(
            max_connections=config.max_connections,
            max_inference_threads=config.max_inference_threads,
        )
        self.recording_writer = recording_writer
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

            if self.recording_writer is not None:
                self.recording_writer.submit(
                    partial(
                        record_voice_turn,
                        audio_bytes,
                        text=result.text,
                        backend=result.backend_name,
                        language=self.request_language,
                        raw_text=raw_transcript,
                        result_language=result.language,
                    )
                )
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
        return transcribe_pcm(
            audio_bytes,
            request_model=self.request_model,
            prompt=self.config.prompt,
            language=self.request_language,
            backend=explicit_backend(self.config),
        )

    async def _write_synthesized_audio(self, audio_path: Path) -> None:
        with wave.open(str(audio_path), "rb") as wav_reader:
            rate = wav_reader.getframerate()
            width = wav_reader.getsampwidth()
            channels = wav_reader.getnchannels()

            target = settings.VOXHELM_WYOMING_TTS_TARGET_DBFS
            gain = 1.0
            if target is not None and width == 2:
                gain = await asyncio.to_thread(_measure_wav_gain, audio_path, target)

            await self.write_event(
                AudioStart(
                    rate=rate,
                    width=width,
                    channels=channels,
                ).event()
            )

            frames_per_chunk = settings.VOXHELM_WYOMING_SAMPLES_PER_CHUNK
            while audio := wav_reader.readframes(frames_per_chunk):
                await self.write_event(
                    AudioChunk(
                        audio=_apply_pcm16_gain(audio, gain),
                        rate=rate,
                        width=width,
                        channels=channels,
                    ).event()
                )

        await self.write_event(AudioStop().event())


def transcribe_pcm(
    audio_bytes: bytes,
    *,
    request_model: str,
    prompt: str | None,
    language: str | None,
    backend: str | None = None,
    warm_up_only: bool = False,
    cancel_event: threading.Event | None = None,
):
    """Transcribe 16 kHz mono 16-bit PCM on the interactive lane."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        with wave.open(str(temp_path), "wb") as wav_file:
            wav_file.setframerate(_CONVERTED_RATE)
            wav_file.setsampwidth(_CONVERTED_WIDTH)
            wav_file.setnchannels(_CONVERTED_CHANNELS)
            wav_file.writeframes(audio_bytes)

        result = transcribe_audio(
            temp_path,
            TranscribeParams(
                request_model=request_model,
                prompt=prompt,
                language=language,
                scheduler_lane=LANE_INTERACTIVE,
                backend=backend,
                allow_backend_fallback=not warm_up_only,
                cancel_event=cancel_event,
            ),
        )
        if (
            not warm_up_only
            and backend == PARAKEET_INPROCESS_BACKEND_NAME
            and result.backend_name == PARAKEET_INPROCESS_BACKEND_NAME
            and needs_language_retry(result.text, language)
        ):
            try:
                result = transcribe_audio(
                    temp_path,
                    TranscribeParams(
                        request_model=settings.VOXHELM_MLX_MODEL,
                        prompt=prompt,
                        language=language,
                        scheduler_lane=LANE_INTERACTIVE,
                        backend="mlx",
                        cancel_event=cancel_event,
                    ),
                )
            except BackendUnavailableError:
                _best_effort_warning("Wyoming STT language retry unavailable; serving first pass")
            else:
                with suppress(Exception):
                    _LOGGER.info("Wyoming STT language retry succeeded; serving retry pass")
        return result
    finally:
        temp_path.unlink(missing_ok=True)


_TTS_MAX_BOOST_DB = 8.0
_TTS_MAX_CUT_DB = 8.0
_TTS_PEAK_CEILING = 0.89  # about -1 dBFS


_TTS_MEASUREMENT_BLOCK_BYTES = 64 * 1024


def _pcm16_samples(audio: bytes) -> array.array[int]:
    samples = array.array("h")
    samples.frombytes(audio[: len(audio) - len(audio) % 2])
    if sys.byteorder != "little":
        samples.byteswap()
    return samples


def _loudness_gain(*, peak: int, squares: int, count: int, target_dbfs: float) -> float:
    if peak == 0:
        return 1.0
    rms = math.sqrt(squares / count)
    gain_db = target_dbfs - 20 * math.log10(rms / 32768)
    gain_db = max(-_TTS_MAX_CUT_DB, min(_TTS_MAX_BOOST_DB, gain_db))
    return min(10 ** (gain_db / 20), _TTS_PEAK_CEILING * 32767 / peak)


def _measure_wav_gain(audio_path: Path, target_dbfs: float) -> float:
    peak = squares = count = 0
    # This reader belongs to the worker even if its awaiting client disconnects.
    with wave.open(str(audio_path), "rb") as reader:
        frames = max(1, _TTS_MEASUREMENT_BLOCK_BYTES // (2 * reader.getnchannels()))
        while audio := reader.readframes(frames):
            samples = _pcm16_samples(audio)
            peak = max(peak, max(map(abs, samples), default=0))
            squares += sum(sample * sample for sample in samples)
            count += len(samples)
    return _loudness_gain(peak=peak, squares=squares, count=count, target_dbfs=target_dbfs)


def _apply_pcm16_gain(audio: bytes, gain: float) -> bytes:
    if abs(gain - 1.0) < 0.01:
        return audio
    samples = _pcm16_samples(audio)
    scaled = array.array("h", (max(-32768, min(32767, round(sample * gain))) for sample in samples))
    if sys.byteorder != "little":
        scaled.byteswap()
    return scaled.tobytes()


def normalize_loudness(audio: bytes, *, width: int, target_dbfs: float | None) -> bytes:
    """Scale 16-bit PCM towards the target RMS, with the Wyoming gain and peak limits."""
    if target_dbfs is None or width != 2:
        return audio
    samples = _pcm16_samples(audio)
    gain = _loudness_gain(
        peak=max(map(abs, samples), default=0),
        squares=sum(sample * sample for sample in samples),
        count=len(samples),
        target_dbfs=target_dbfs,
    )
    return _apply_pcm16_gain(audio, gain)


_TURN_NAME = re.compile(r"[0-9]{20}-[0-9a-f]{32}")
_TURN_FILES = {"audio.wav", "metadata.json"}


def _is_owned_turn(path: Path, *, complete: bool) -> bool:
    if not stat.S_ISDIR(path.lstat().st_mode):
        return False
    entries = {entry.name: entry for entry in path.iterdir()}
    valid_names = entries.keys() == _TURN_FILES if complete else entries.keys() <= _TURN_FILES
    return valid_names and all(stat.S_ISREG(entry.lstat().st_mode) for entry in entries.values())


def _remove_turn(path: Path) -> None:
    for name in _TURN_FILES:
        (path / name).unlink(missing_ok=True)
    path.rmdir()


def record_voice_turn(
    audio_bytes: bytes,
    *,
    text: str,
    backend: str,
    language: str | None,
    raw_text: str | None = None,
    result_language: str | None = None,
) -> None:
    """Keep the turn's audio and transcript for an STT comparison, if enabled.

    Off unless VOXHELM_WYOMING_STT_RECORD_DIR is set. The recorder namespace and files are
    owner-only, and only the newest
    VOXHELM_WYOMING_STT_RECORD_MAX turns are kept. Failures are logged and
    never affect the voice turn.
    """
    directory = settings.VOXHELM_WYOMING_STT_RECORD_DIR
    if not directory or not audio_bytes:
        return
    try:
        root = Path(directory) / "voxhelm-turns"
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not stat.S_ISDIR(root.lstat().st_mode):
            raise OSError("Invalid recording namespace")
        os.chmod(root, 0o700)
        lock_fd = os.open(root / ".writer.lock", os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            _commit_recording(
                root,
                audio_bytes,
                {
                    "metadata_version": 2,
                    "text": text,
                    "backend": backend,
                    "language": language,
                    "raw_text": text if raw_text is None else raw_text,
                    "result_language": result_language,
                },
            )
        finally:
            os.close(lock_fd)
    except Exception:
        _best_effort_warning("Wyoming STT recording failed")


_PENDING_PREFIX = ".pending-"
# The wave writer emits the standard 44-byte header for this PCM format.
_PCM_WAV_HEADER_BYTES = 44


def _commit_recording(
    root: Path, audio_bytes: bytes, metadata: dict[str, str | int | None]
) -> None:
    for path in root.iterdir():
        if (
            path.name.startswith(_PENDING_PREFIX)
            and _TURN_NAME.fullmatch(path.name.removeprefix(_PENDING_PREFIX))
            and _is_owned_turn(path, complete=False)
        ):
            _remove_turn(path)
    metadata_bytes = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
    budget = settings.VOXHELM_WYOMING_STT_RECORD_MAX_BYTES
    if _PCM_WAV_HEADER_BYTES + len(audio_bytes) + len(metadata_bytes) > budget:
        _best_effort_warning("Wyoming STT recording exceeds byte budget")
        _prune_recordings(root, budget)
        return

    name = f"{time.time_ns():020d}-{uuid4().hex}"
    stage = root / (_PENDING_PREFIX + name)
    stage.mkdir(mode=0o700)
    try:
        with open(
            os.open(stage / "audio.wav", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb"
        ) as raw:
            with wave.open(raw, "wb") as wav_file:
                wav_file.setframerate(_CONVERTED_RATE)
                wav_file.setsampwidth(_CONVERTED_WIDTH)
                wav_file.setnchannels(_CONVERTED_CHANNELS)
                wav_file.writeframes(audio_bytes)
        with open(
            os.open(stage / "metadata.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb"
        ) as handle:
            handle.write(metadata_bytes)
        stage.rename(root / name)
    finally:
        if stage.exists():
            _remove_turn(stage)

    _prune_recordings(root, budget)


def _prune_recordings(root: Path, budget: int) -> None:
    keep = max(1, settings.VOXHELM_WYOMING_STT_RECORD_MAX)
    complete = sorted(
        path
        for path in root.iterdir()
        if _TURN_NAME.fullmatch(path.name) and _is_owned_turn(path, complete=True)
    )
    for turn in reversed(complete):
        size = sum((turn / name).stat().st_size for name in _TURN_FILES)
        if keep and size <= budget:
            keep -= 1
            budget -= size
            continue
        pending = turn.rename(root / (_PENDING_PREFIX + turn.name))
        _remove_turn(pending)


def needs_language_retry(text: str, language: str | None) -> bool:
    """True when Parakeet's transcript is not in the pipeline's language.

    Parakeet detects the language itself and turns short German follow-ups
    ("Das linke", "Ja") into English ("Uh, that's linking.", "Yep."); the
    conversation agent then answers in English. Only de/en are judged, with
    the TTS routing detector (``routing`` extra); without it nothing retries.
    """
    expected = (language or "").strip().lower()[:2]
    if expected not in {"de", "en"} or not text.strip():
        return False
    try:
        detection = detect_routing_language(text.strip())
    except Exception:
        return False
    return detection is not None and detection.language != expected


def explicit_backend(config: WyomingSttConfig) -> str | None:
    """Backends the model name cannot select are passed to the service by name."""
    if config.backend == PARAKEET_INPROCESS_BACKEND_NAME:
        return config.backend
    return None


def warm_up_stt(config: WyomingSttConfig, *, cancel_event: threading.Event | None = None) -> None:
    """Load the STT model with one second of silence; the transcript is discarded."""
    primary_backend = explicit_backend(config)
    targets = [(primary_backend, config.model)]
    if primary_backend == PARAKEET_INPROCESS_BACKEND_NAME:
        targets.append(("mlx", settings.VOXHELM_MLX_MODEL))

    for backend, model in targets:
        if cancel_event is not None and cancel_event.is_set():
            break
        started = time.monotonic()
        try:
            transcribe_pcm(
                bytes(_CONVERTED_BYTES_PER_SECOND),
                request_model=model,
                prompt=config.prompt,
                language=config.language or next(iter(config.languages), None),
                backend=backend,
                warm_up_only=True,
                cancel_event=cancel_event,
            )
        except InferenceCancelled:
            with suppress(Exception):
                _LOGGER.info("Wyoming STT warm-up cancelled backend=%s", backend or config.backend)
            break
        except Exception:
            with suppress(Exception):
                _LOGGER.exception(
                    "Wyoming STT warm-up failed backend=%s", backend or config.backend
                )
        else:
            with suppress(Exception):
                _LOGGER.info(
                    "Wyoming STT warm-up finished backend=%s seconds=%.1f",
                    backend or config.backend,
                    time.monotonic() - started,
                )


_WARM_UP_DRAIN_SECONDS = 2


class _WarmUp:
    """Track completion in the native worker, independently of listener cancellation."""

    def __init__(self, config: WyomingSttConfig) -> None:
        self.cancel_event = threading.Event()
        self.finished = asyncio.Event()
        loop = asyncio.get_running_loop()

        def run() -> None:
            try:
                warm_up_stt(config, cancel_event=self.cancel_event)
            except Exception:
                _best_effort_warning("Wyoming STT warm-up worker failed")
            finally:
                if self.cancel_event.is_set():
                    with suppress(Exception):
                        _LOGGER.info("Wyoming STT warm-up thread retired")
                # The listener and its loop may already have exited after a timeout.
                with suppress(RuntimeError):
                    loop.call_soon_threadsafe(self.finished.set)

        # Native model loading may ignore cancellation. A detached warm-up does
        # not hold up asyncio's default-executor shutdown or keep the process alive.
        threading.Thread(target=run, name="wyoming-stt-warm-up", daemon=True).start()

    async def drain(self) -> None:
        try:
            async with asyncio.timeout(_WARM_UP_DRAIN_SECONDS):
                await self.finished.wait()
        except TimeoutError:
            _best_effort_warning("Wyoming STT warm-up still running at shutdown; detached")


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

    recording_writer = _RecordingWriter() if settings.VOXHELM_WYOMING_STT_RECORD_DIR else None

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
        return WyomingSttEventHandler(
            resolved_config,
            info,
            reader,
            writer,
            admission=admission,
            recording_writer=recording_writer,
        )

    recording_task: asyncio.Task[None] | None = None
    warm_up: _WarmUp | None = None
    try:
        await server.start(handler_factory)
        # Everything after a successful start runs under the finally, so a logging
        # failure or cancellation still stops the listener and owned handlers.
        try:
            if recording_writer is not None:
                recording_task = asyncio.create_task(recording_writer.run())
            _LOGGER.info(
                "Wyoming STT/TTS listening on %s:%s using backend=%s model=%s language=%s",
                resolved_config.host,
                resolved_config.port,
                resolved_config.backend,
                resolved_config.model,
                resolved_config.language or "auto",
            )
            if resolved_config.warm_up:
                # In a thread so the listener answers describe/info at once; a voice
                # turn that arrives meanwhile waits on the same model lock.
                warm_up = _WarmUp(resolved_config)
            await stop_event.wait()
        finally:
            if warm_up is not None:
                warm_up.cancel_event.set()
            try:
                await shutdown_wyoming_server(server, admission)
            finally:
                try:
                    if warm_up is not None:
                        await warm_up.drain()
                finally:
                    if recording_writer is not None and recording_task is not None:
                        await recording_writer.close(recording_task)
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
