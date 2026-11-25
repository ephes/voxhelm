from __future__ import annotations

import asyncio
import json
import logging
import math
import mimetypes
import os
import re
import shutil
import subprocess
import threading
from collections.abc import AsyncGenerator, Mapping
from contextlib import aclosing
from dataclasses import dataclass, field, replace
from pathlib import Path
from stat import S_ISREG
from threading import Lock
from time import monotonic
from typing import Any, Protocol, cast
from urllib.parse import urlparse
from uuid import uuid4

import httpx
from django.conf import settings

from lane_scheduler import LANE_NON_INTERACTIVE, InferenceCancelled, admit_local_inference
from transcriptions import formats as transcript_formats
from transcriptions.coverage import (
    IncompleteTranscriptError,
    TranscriptCoverage,
    check_transcript_coverage,
    validate_max_gap_seconds,
    wav_duration_seconds,
)
from transcriptions.json_input import decode_json_object
from transcriptions.native_output import file_generation, read_native_json
from transcriptions.owned_process import (
    OwnedProcessMessages,
    process_output_directory,
    run_native_process,
    run_owned_process,
)
from transcriptions.sanitizer import sanitize_result

NATIVE_TRANSCRIPT_MAX_BYTES = 64 * 1024 * 1024
WHISPERKIT_RESPONSE_MAX_BYTES = 64 * 1024 * 1024
WHISPERKIT_ERROR_MAX_BYTES = 64 * 1024

AUTO_BACKEND_MODEL_NAMES = {"auto", "gpt-4o-mini-transcribe", "whisper-1"}
WHISPERKIT_BACKEND_MODEL_NAMES = {"whisperkit"}
PARAKEET_BACKEND_NAME = "parakeet"
# Parakeet loaded once into this process (Wyoming sidecar); needs the
# ``parakeet`` extra. The ``parakeet`` backend above runs the CLI per call.
PARAKEET_INPROCESS_BACKEND_NAME = "parakeet-mlx"
# Parakeet TDT 0.6B v3 detects the language itself; these are the 25 it supports.
PARAKEET_LANGUAGES = frozenset(
    "bg cs da de el en es et fi fr hr hu it lt lv mt nl pl pt ro ru sk sl sv uk".split()
)
PARAKEET_MODEL_FILES = ("config.json", "model.safetensors")
_LOGGER = logging.getLogger(__name__)
LEADING_FILLER_WORDS = {
    "de": ("okay", "ok", "und", "also", "äh", "ah", "hm", "hmm"),
    "en": ("okay", "ok", "and", "so", "well"),
}
LEADING_FILLER_PATTERNS = {
    language: re.compile(
        r"^(?:" + "|".join(re.escape(word) for word in words) + r")(?:[\s,.;:!?\-…]+|$)",
        re.IGNORECASE,
    )
    for language, words in LEADING_FILLER_WORDS.items()
}
# whisper.cpp stderr lines worth keeping: its library logs errors such as
# "failed to encode" or "failed to decode" through the log callback, which
# ``-np`` would silence. Bounded so a noisy run cannot flood logs or errors.
_WHISPERCPP_ERROR_LINE = re.compile(r"fail|error|abort|out of memory", re.IGNORECASE)
WHISPERCPP_DIAGNOSTIC_MAX_LINES = 20
WHISPERCPP_DIAGNOSTIC_MAX_LINE_CHARS = 300
GERMAN_DISCOURSE_PARTICLE_PATTERN = re.compile(
    r"^(?P<prefix>(?:wie|was)\s+ist)\s+(?P<particle>denn|eigentlich|mal)\s+",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TranscribeParams:
    request_model: str
    prompt: str | None
    language: str | None
    scheduler_lane: str = LANE_NON_INTERACTIVE
    # Ask the backend for per-word timings on each segment. Backends that cannot
    # provide them return segments without words. Only known-speaker jobs with
    # the re-cut set it, so it also selects their backend chain (see
    # ``known_speaker_backend_chain``).
    word_timestamps: bool = False
    # Cooperative cancellation: set by the caller (the sync HTTP view on client
    # disconnect). Excluded from equality/repr so parameter comparisons and log
    # lines stay unchanged.
    cancel_event: threading.Event | None = field(default=None, compare=False, repr=False)
    # Run exactly this backend (then the mlx fallback if it is unavailable)
    # instead of resolving one from the model name. Only the Wyoming sidecar
    # sets it, for ``parakeet-mlx``.
    backend: str | None = None
    allow_backend_fallback: bool = True


@dataclass(frozen=True)
class TranscriptionWord:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TranscriptionSegment:
    id: int
    start: float
    end: float
    text: str
    speaker: str | None = None
    # Per-word timings, only when requested and the backend provides them. They
    # are working data for postprocessing and are not rendered into artifacts.
    words: tuple[TranscriptionWord, ...] = ()

    def as_verbose_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "seek": int(self.start * 100),
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }
        if self.speaker:
            payload["speaker"] = self.speaker
        return payload


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str | None
    segments: list[TranscriptionSegment]
    backend_name: str = ""
    model_name: str = ""
    # Set by backends that measure their output against the media duration
    # (whisper.cpp); ``None`` otherwise.
    coverage: TranscriptCoverage | None = None


@dataclass(frozen=True)
class BackendInvocation:
    name: str
    service: BackendProtocol


class BackendUnavailableError(RuntimeError):
    """Raised when a backend is not available on the current host."""


class BackendProtocol(Protocol):
    def transcribe(self, audio_path: Path, params: TranscribeParams) -> TranscriptionResult: ...


def normalize_interactive_transcript(text: str, *, language: str | None) -> str:
    normalized = " ".join(text.split()).strip()
    if not normalized:
        return normalized

    language_key = normalize_language_code(language)
    pattern = LEADING_FILLER_PATTERNS.get(language_key)
    if pattern is None:
        return normalized

    candidate = normalized
    while True:
        stripped = pattern.sub("", candidate, count=1).lstrip(" ,.;:!?-…")
        if not stripped or stripped == candidate:
            candidate = normalized if not stripped else candidate
            break
        candidate = stripped

    if language_key == "de":
        candidate = GERMAN_DISCOURSE_PARTICLE_PATTERN.sub(r"\g<prefix> ", candidate)

    return candidate


def normalize_language_code(language: str | None) -> str:
    if not language:
        return ""
    return language.strip().lower().replace("_", "-").split("-", 1)[0]


class MlxWhisperBackend:
    def __init__(self, *, model_name: str, condition_on_previous_text: bool = False) -> None:
        self.model_name = model_name
        self.condition_on_previous_text = condition_on_previous_text

    def transcribe(self, audio_path: Path, params: TranscribeParams) -> TranscriptionResult:
        try:
            import mlx_whisper
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise BackendUnavailableError(
                "mlx-whisper is not installed. Install the project dependencies first."
            ) from exc

        # mlx-whisper keeps model state in this process, so overlapping calls are
        # not safe here. Other processes (Wyoming sidecar) have their own lock.
        with _MLX_LOCK:
            _raise_if_cancelled(params)
            payload = mlx_whisper.transcribe(
                str(audio_path),
                path_or_hf_repo=self.model_name,
                word_timestamps=False,
                initial_prompt=params.prompt,
                language=params.language,
                # Off by default: conditioning on prior text is Whisper's main long-form
                # repetition-loop trigger (see settings). compression_ratio_threshold and
                # temperature fallback stay at mlx-whisper defaults.
                condition_on_previous_text=self.condition_on_previous_text,
            )
        return normalize_transcription_payload(
            payload,
            backend_name="mlx-whisper",
            model_name=self.model_name,
        )


class WhisperCppBackend:
    def __init__(
        self,
        *,
        binary_path: str,
        model_name: str,
        processors: int,
        max_context: int = 0,
        suppress_nst: bool = True,
        max_gap_seconds: float = 600.0,
    ) -> None:
        validate_max_gap_seconds(max_gap_seconds)
        self.binary_path = binary_path
        self.model_name = model_name
        self.processors = processors
        self.max_context = max_context
        self.suppress_nst = suppress_nst
        self.max_gap_seconds = max_gap_seconds

    def transcribe(self, audio_path: Path, params: TranscribeParams) -> TranscriptionResult:
        executable = resolve_whispercpp_binary(self.binary_path)
        model_path = resolve_whispercpp_model_path(self.model_name)

        with process_output_directory(prefix="voxhelm-whispercpp-") as temp_dir:
            temp_root = Path(temp_dir)
            normalized_audio_path = temp_root / "input.wav"
            normalize_audio_to_wav_16k_mono(
                input_path=audio_path,
                output_path=normalized_audio_path,
            )

            output_base = temp_root / "transcript"
            # Token offsets of the later chunks are chunk-relative when
            # whisper.cpp splits the audio across processors, so word timings
            # need a single processor.
            processors = 1 if params.word_timestamps else self.processors
            args = [
                executable,
                "-m",
                str(model_path),
                "-f",
                str(normalized_audio_path),
                "-ojf" if params.word_timestamps else "-oj",
                "-of",
                str(output_base),
                "-p",
                str(processors),
                "-l",
                params.language or "auto",
                # No ``-np``: it silences whisper.cpp's library log, including the
                # errors of a part that fails under ``-p N``.
                # Anti-hallucination: bound decoder context (0 = do not condition on
                # previously generated text, the main long-form repetition-loop trigger).
                "--max-context",
                str(self.max_context),
            ]
            if self.suppress_nst:
                # Drop non-speech tokens so music/silence stretches stop producing
                # hallucinated filler (e.g. outro-music loops).
                args.append("--suppress-nst")
            if params.prompt:
                args.extend(["--prompt", params.prompt])

            _raise_if_cancelled(params)
            completed = run_cancellable_process(args, cancel_event=params.cancel_event)
            diagnostics = whispercpp_error_lines(completed.stderr)
            if completed.returncode != 0:
                detail = "; ".join(diagnostics or whispercpp_tail_lines(completed.stderr))
                if detail:
                    raise RuntimeError(f"whisper.cpp transcription failed: {detail}")
                raise RuntimeError("whisper.cpp transcription failed.")
            if diagnostics:
                # Exit 0 with error lines is how a lost ``-p N`` part shows up.
                _LOGGER.warning(
                    "whisper.cpp exited 0 but logged errors processors=%s: %s",
                    processors,
                    " | ".join(diagnostics),
                )

            json_path = output_base.with_suffix(".json")
            try:
                payload = decode_json_object(
                    read_native_json(json_path, limit=NATIVE_TRANSCRIPT_MAX_BYTES)
                )
            except (OSError, RuntimeError, ValueError, RecursionError) as exc:
                raise RuntimeError(
                    "whisper.cpp transcription produced invalid JSON output."
                ) from exc
            _raise_if_cancelled(params)
            result = normalize_whispercpp_payload(payload, model_name=self.model_name)
            # Measured on the normalized WAV while it still exists: the raw
            # segments against the real duration, before any sanitizing.
            try:
                coverage = check_transcript_coverage(
                    result.segments,
                    media_duration_seconds=wav_duration_seconds(normalized_audio_path),
                    max_gap_seconds=self.max_gap_seconds,
                    backend_name="whisper.cpp",
                )
            except IncompleteTranscriptError as exc:
                _LOGGER.error(
                    "%s processors=%s stderr_errors=%s",
                    exc,
                    processors,
                    " | ".join(diagnostics) or "none",
                )
                raise
        return replace(result, coverage=coverage)


def whispercpp_error_lines(stderr: object) -> list[str]:
    """The last error-looking stderr lines of a whisper.cpp run, bounded."""
    if not isinstance(stderr, str):
        return []
    lines = [
        line.strip()[:WHISPERCPP_DIAGNOSTIC_MAX_LINE_CHARS]
        for line in stderr.splitlines()
        if _WHISPERCPP_ERROR_LINE.search(line)
    ]
    return lines[-WHISPERCPP_DIAGNOSTIC_MAX_LINES:]


def whispercpp_tail_lines(stderr: object, *, count: int = 5) -> list[str]:
    """The last nonblank stderr lines, bounded, for a failure without error lines."""
    if not isinstance(stderr, str):
        return []
    lines = [
        line.strip()[:WHISPERCPP_DIAGNOSTIC_MAX_LINE_CHARS]
        for line in stderr.splitlines()
        if line.strip()
    ]
    return lines[-count:]


NORMALIZATION_MESSAGES = OwnedProcessMessages(
    start="ffmpeg audio normalization could not start ffmpeg.",
    deadline="ffmpeg audio normalization exceeded the configured decode deadline.",
    diagnostics="ffmpeg audio normalization exceeded the diagnostic output limit.",
    failure="ffmpeg audio normalization failed.",
)


def normalize_audio_to_wav_16k_mono(*, input_path: Path, output_path: Path) -> None:
    """Decode to 16 kHz mono PCM WAV with an owned process tree and bounded diagnostics.

    ffmpeg runs in its own process group under ``VOXHELM_AUDIO_DECODE_TIMEOUT_SECONDS``
    at error log level; a 2 KiB stderr prefix is logged on failure and the rest is
    drained and discarded. Tool output never goes into the raised error.
    """
    run_owned_process(
        [
            settings.VOXHELM_FFMPEG_BIN,
            "-nostdin",
            "-hide_banner",
            "-nostats",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ],
        timeout=settings.VOXHELM_AUDIO_DECODE_TIMEOUT_SECONDS,
        messages=NORMALIZATION_MESSAGES,
        # Damaged but decodable user media can log many errors; keep memory bounded
        # without failing a decode that succeeds.
        fail_on_diagnostic_overflow=False,
    )
    if not output_path.exists():
        raise RuntimeError(NORMALIZATION_MESSAGES.failure)


class ParakeetBackend:
    """NVIDIA Parakeet TDT through the ``parakeet-mlx`` command line.

    The CLI lives in its own Python 3.13 tool environment, so Voxhelm runs it
    as a child process instead of importing it. The model is a local
    directory deployed at a pinned commit, and the child runs with
    ``HF_HUB_OFFLINE=1`` so it can never fetch other weights.
    """

    def __init__(
        self,
        *,
        binary_path: str,
        model_dir: str,
        model_name: str,
        chunk_seconds: int,
        overlap_seconds: int,
        max_sentence_seconds: float | None,
        timeout_seconds: int,
        local_attention: bool = True,
    ) -> None:
        self.binary_path = binary_path
        self.model_dir = model_dir
        self.model_name = model_name
        self.chunk_seconds = chunk_seconds
        self.overlap_seconds = overlap_seconds
        self.max_sentence_seconds = max_sentence_seconds
        self.timeout_seconds = timeout_seconds
        self.local_attention = local_attention

    def transcribe(self, audio_path: Path, params: TranscribeParams) -> TranscriptionResult:
        payload = self.run(audio_path, params)
        return normalize_parakeet_payload(payload, model_name=self.model_name)

    def run(self, audio_path: Path, params: TranscribeParams) -> dict[str, Any]:
        """Run the CLI once and return its JSON payload (sentences with tokens)."""
        if not parakeet_supports_language(params.language):
            raise BackendUnavailableError(
                f"Parakeet does not support language '{params.language}'."
            )
        executable, model_dir = resolve_parakeet_runtime(
            binary_path=self.binary_path, model_dir=self.model_dir
        )

        with process_output_directory(prefix="voxhelm-parakeet-") as temp_dir:
            temp_root = Path(temp_dir)
            normalized_audio_path = temp_root / "input.wav"
            normalize_audio_to_wav_16k_mono(
                input_path=audio_path,
                output_path=normalized_audio_path,
            )
            output_dir = temp_root / "out"
            args = [
                executable,
                "--model",
                str(model_dir),
                "--output-format",
                "json",
                "--output-dir",
                str(output_dir),
                "--output-template",
                "transcript",
                "--chunk-duration",
                str(self.chunk_seconds),
                "--overlap-duration",
                str(self.overlap_seconds),
                "--local-attention" if self.local_attention else "--full-attention",
            ]
            if self.max_sentence_seconds:
                args.extend(["--max-duration", f"{self.max_sentence_seconds:g}"])
            args.append(str(normalized_audio_path))

            _raise_if_cancelled(params)
            started = monotonic()
            completed = run_cancellable_process(
                args,
                cancel_event=params.cancel_event,
                timeout_seconds=self.timeout_seconds,
                env=parakeet_child_env(),
            )
            if completed.returncode != 0:
                detail = "\n".join(
                    part.strip()
                    for part in (completed.stderr, completed.stdout)
                    if isinstance(part, str) and part.strip()
                )
                if detail:
                    raise RuntimeError(f"Parakeet transcription failed: {detail}")
                raise RuntimeError("Parakeet transcription failed.")

            json_path = output_dir / "transcript.json"
            if not json_path.exists():
                raise RuntimeError(
                    "Parakeet transcription failed: transcript.json was not produced."
                )
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError("Parakeet transcription produced invalid JSON.") from exc
        _LOGGER.info(
            "parakeet transcription finished model=%s seconds=%.1f",
            self.model_name,
            monotonic() - started,
        )
        if not isinstance(payload, dict):
            raise RuntimeError("Parakeet transcription produced an unexpected payload.")
        return payload


def parakeet_child_env() -> dict[str, str]:
    """The child's environment: offline, and no ``PARAKEET_*`` overrides.

    The CLI reads most options from ``PARAKEET_*`` variables as well; dropping
    them keeps the invocation fully determined by the arguments.
    """
    env = {key: value for key, value in os.environ.items() if not key.startswith("PARAKEET_")}
    env["HF_HUB_OFFLINE"] = "1"
    return env


def resolve_parakeet_runtime(*, binary_path: str, model_dir: str) -> tuple[str, Path]:
    """Return the CLI path and model directory, or raise ``BackendUnavailableError``."""
    if not model_dir:
        raise BackendUnavailableError(
            "Parakeet is not configured. Set VOXHELM_PARAKEET_MODEL_DIR to the deployed model."
        )
    # Always absolute: parakeet-mlx tries the Hugging Face Hub first, so a
    # relative path shaped like a repo id could load cached unpinned weights.
    directory = Path(model_dir).expanduser().resolve()
    missing = [name for name in PARAKEET_MODEL_FILES if not (directory / name).is_file()]
    if missing:
        raise BackendUnavailableError(
            f"Parakeet model directory '{directory}' is missing {', '.join(missing)}."
        )
    if "/" in binary_path:
        if not Path(binary_path).is_file():
            raise BackendUnavailableError(f"parakeet-mlx was not found at '{binary_path}'.")
        return binary_path, directory
    resolved = shutil.which(binary_path)
    if resolved is None:
        raise BackendUnavailableError(f"parakeet-mlx '{binary_path}' was not found in PATH.")
    return resolved, directory


def normalize_parakeet_payload(payload: object, *, model_name: str) -> TranscriptionResult:
    """Turn ``parakeet-mlx --output-format json`` output into a result with words.

    One segment per sentence. Words are built from the sentence's tokens: a
    token starting with a space starts a word, any other token (sub-word
    piece, punctuation) is appended to the current word. Words keep their
    punctuation, because the speaker-change re-cut rebuilds text from words.
    A malformed token list drops the words of that sentence only.
    """
    if not isinstance(payload, dict):
        raise RuntimeError("Parakeet transcription produced an unexpected payload.")
    raw_sentences = payload.get("sentences")
    segments: list[TranscriptionSegment] = []
    if isinstance(raw_sentences, list):
        for raw_sentence in raw_sentences:
            if not isinstance(raw_sentence, dict):
                continue
            segment_text = raw_sentence.get("text")
            start = _finite_seconds(raw_sentence.get("start"))
            end = _finite_seconds(raw_sentence.get("end"))
            if not isinstance(segment_text, str) or not segment_text.strip():
                continue
            if start is None or end is None:
                continue
            end = max(end, start)
            segments.append(
                TranscriptionSegment(
                    id=len(segments),
                    start=start,
                    end=end,
                    text=segment_text.strip(),
                    words=parakeet_sentence_words(
                        raw_sentence.get("tokens"), segment_start=start, segment_end=end
                    ),
                )
            )
    raw_text = payload.get("text")
    text = raw_text.strip() if isinstance(raw_text, str) else ""
    if not text:
        text = " ".join(segment.text for segment in segments)
    return TranscriptionResult(
        text=text,
        language=None,
        segments=segments,
        backend_name="parakeet",
        model_name=model_name,
    )


def parakeet_sentence_words(
    raw_tokens: object,
    *,
    segment_start: float,
    segment_end: float,
) -> tuple[TranscriptionWord, ...]:
    if not isinstance(raw_tokens, list):
        return ()
    words: list[tuple[float, float, str]] = []
    for raw_token in raw_tokens:
        if not isinstance(raw_token, dict):
            return ()
        token_text = raw_token.get("text")
        token_start = _finite_seconds(raw_token.get("start"))
        token_end = _finite_seconds(raw_token.get("end"))
        if not isinstance(token_text, str) or token_start is None or token_end is None:
            return ()
        if not token_text:
            continue
        # Clamped into the sentence and kept non-decreasing, as for whisper.cpp.
        floor = words[-1][0] if words else segment_start
        token_start = min(max(token_start, floor), segment_end)
        token_end = min(max(token_end, token_start), segment_end)
        if words and not token_text.startswith(" "):
            previous_start, previous_end, previous_text = words[-1]
            words[-1] = (previous_start, max(previous_end, token_end), previous_text + token_text)
        else:
            words.append((token_start, token_end, token_text.strip()))
    return tuple(
        TranscriptionWord(start=start, end=end, text=text) for start, end, text in words if text
    )


def _finite_seconds(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    seconds = float(value)
    return seconds if math.isfinite(seconds) else None


_PARAKEET_INPROCESS_LOCK = threading.Lock()
type _ParakeetGeneration = tuple[Path, tuple[tuple[int, ...], ...]]
_PARAKEET_INPROCESS_MODELS: dict[Path, tuple[_ParakeetGeneration, Any]] = {}


def _parakeet_installed_generation(configured: Path) -> _ParakeetGeneration:
    directory = configured.resolve()
    try:
        files = [(directory / name).stat() for name in PARAKEET_MODEL_FILES]
    except OSError as exc:
        raise BackendUnavailableError("Parakeet model files are unavailable.") from exc
    if not all(S_ISREG(metadata.st_mode) for metadata in files):
        raise BackendUnavailableError("Parakeet model files must be regular files.")
    return directory, tuple(map(file_generation, files))


class ParakeetInProcessBackend:
    """Parakeet TDT through ``parakeet_mlx`` in this process, model kept loaded.

    For short interactive audio (Home Assistant voice commands), where the CLI
    backend's per-call model load would cost more than the inference.
    """

    def __init__(self, *, model_dir: str, model_name: str) -> None:
        self.model_dir = model_dir
        self.model_name = model_name

    def transcribe(self, audio_path: Path, params: TranscribeParams) -> TranscriptionResult:
        if not parakeet_supports_language(params.language):
            raise BackendUnavailableError(
                f"Parakeet does not support language '{params.language}'."
            )
        try:
            from parakeet_mlx import from_pretrained  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise BackendUnavailableError(
                "parakeet-mlx is not installed. Install the 'parakeet' extra."
            ) from exc
        if not self.model_dir:
            raise BackendUnavailableError("VOXHELM_PARAKEET_MODEL_DIR is not set.")
        configured = Path(self.model_dir).expanduser().absolute()
        with _PARAKEET_INPROCESS_LOCK:
            _raise_if_cancelled(params)
            generation = _parakeet_installed_generation(configured)
            resident = _PARAKEET_INPROCESS_MODELS.get(configured)
            if resident is None or resident[0] != generation:
                del resident
                _PARAKEET_INPROCESS_MODELS.pop(configured, None)
                model = from_pretrained(str(generation[0]))
                if _parakeet_installed_generation(configured) != generation:
                    raise BackendUnavailableError("Parakeet model changed while loading.")
                _PARAKEET_INPROCESS_MODELS[configured] = (generation, model)
            else:
                model = resident[1]
            _raise_if_cancelled(params)
            result = model.transcribe(str(audio_path))
        payload = {
            "text": result.text,
            "sentences": [
                {
                    "text": sentence.text,
                    "start": sentence.start,
                    "end": sentence.end,
                    "tokens": [
                        {"text": token.text, "start": token.start, "end": token.end}
                        for token in sentence.tokens
                    ],
                }
                for sentence in result.sentences
            ],
        }
        normalized = normalize_parakeet_payload(payload, model_name=self.model_name)
        return replace(normalized, backend_name=PARAKEET_INPROCESS_BACKEND_NAME)


class WhisperKitBackend:
    def __init__(
        self,
        *,
        enabled: bool,
        base_url: str,
        model_name: str,
        timeout_seconds: int,
    ) -> None:
        self.enabled = enabled
        self.base_url = base_url
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds

    def transcribe(self, audio_path: Path, params: TranscribeParams) -> TranscriptionResult:
        if not self.enabled:
            raise BackendUnavailableError(
                "WhisperKit is disabled. Set VOXHELM_WHISPERKIT_ENABLED=true to use it."
            )

        _raise_if_cancelled(params)
        try:
            payload = call_whisperkit_server(
                base_url=self.base_url,
                audio_path=audio_path,
                model_name=self.model_name,
                language=params.language,
                prompt=params.prompt,
                timeout_seconds=self.timeout_seconds,
                cancel_event=params.cancel_event,
            )
        except OSError:
            _raise_if_cancelled(params)
            _LOGGER.warning("WhisperKit sidecar failure: unreachable.")
            # Only transport-level failures are treated as backend unavailability.
            # HTTP/server-side failures stay visible instead of silently switching backends.
            raise BackendUnavailableError(
                "WhisperKit server is not reachable. Check the sidecar process and URL config."
            ) from None

        _raise_if_cancelled(params)
        try:
            result = normalize_transcription_payload(
                payload,
                backend_name="whisperkit",
                model_name=self.model_name,
            )
        except TypeError, ValueError, OverflowError:
            _raise_if_cancelled(params)
            _LOGGER.warning("WhisperKit sidecar failure: invalid transcript.")
            raise RuntimeError("WhisperKit server returned an invalid transcript.") from None
        _raise_if_cancelled(params)
        return result


_MLX_LOCK = Lock()


def _raise_if_cancelled(params: TranscribeParams) -> None:
    cancel_event = params.cancel_event
    if cancel_event is not None and cancel_event.is_set():
        raise InferenceCancelled("Local inference was cancelled by the caller.")


class ChildProcessTimeout(RuntimeError):
    """Raised when a child process outlives its ``timeout_seconds``."""


def run_cancellable_process(
    args: list[str],
    *,
    cancel_event: threading.Event | None,
    poll_seconds: float = 0.25,
    terminate_grace_seconds: float = 5.0,
    timeout_seconds: float | None = None,
    env: Mapping[str, str] | None = None,
    pass_fds: tuple[int, ...] = (),
) -> subprocess.CompletedProcess[str]:
    """Run native inference with bounded diagnostics and owned-group cleanup."""

    def check_cancelled() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise InferenceCancelled("Local inference was cancelled by the caller.")

    try:
        return run_native_process(
            args,
            check_cancelled=check_cancelled,
            timeout=timeout_seconds,
            poll_seconds=poll_seconds,
            terminate_grace_seconds=terminate_grace_seconds,
            env=env,
            pass_fds=pass_fds,
        )
    except subprocess.TimeoutExpired as exc:
        raise ChildProcessTimeout(
            f"Child process exceeded its {timeout_seconds:g} s time limit."
        ) from exc


def normalize_transcription_payload(
    payload: dict[str, Any],
    *,
    backend_name: str = "",
    model_name: str = "",
) -> TranscriptionResult:
    text = str(payload.get("text", "")).strip()
    language_value = payload.get("language")
    language = str(language_value).strip() if isinstance(language_value, str) else None
    raw_segments = payload.get("segments")
    segments: list[TranscriptionSegment] = []

    if isinstance(raw_segments, list):
        for index, raw_segment in enumerate(raw_segments):
            if not isinstance(raw_segment, dict):
                continue
            segment_text = str(raw_segment.get("text", "")).strip()
            if not segment_text:
                continue
            segments.append(
                TranscriptionSegment(
                    id=int(raw_segment.get("id", index)),
                    start=float(raw_segment.get("start", 0.0)),
                    end=float(raw_segment.get("end") or raw_segment.get("start") or 0.0),
                    text=segment_text,
                )
            )

    if not segments and text:
        segments = [TranscriptionSegment(id=0, start=0.0, end=0.0, text=text)]

    return TranscriptionResult(
        text=text,
        language=language,
        segments=segments,
        backend_name=backend_name,
        model_name=model_name,
    )


def normalize_whispercpp_payload(
    payload: dict[str, Any],
    *,
    model_name: str,
) -> TranscriptionResult:
    raw_segments = payload.get("transcription")
    segments: list[TranscriptionSegment] = []

    if isinstance(raw_segments, list):
        for index, raw_segment in enumerate(raw_segments):
            if not isinstance(raw_segment, dict):
                continue
            timestamps = raw_segment.get("timestamps")
            if not isinstance(timestamps, dict):
                continue
            segment_text = str(raw_segment.get("text", "")).strip()
            if not segment_text:
                continue
            start = timestamp_to_seconds(str(timestamps.get("from", "00:00:00,000")))
            end = timestamp_to_seconds(str(timestamps.get("to", "00:00:00,000")))
            segments.append(
                TranscriptionSegment(
                    id=index,
                    start=start,
                    end=end,
                    text=segment_text,
                    words=whispercpp_segment_words(
                        raw_segment.get("tokens"), segment_start=start, segment_end=end
                    ),
                )
            )

    text = " ".join(segment.text for segment in segments).strip()
    return TranscriptionResult(
        text=text,
        language=str(payload.get("result", {}).get("language", "")).strip() or None,
        segments=segments,
        backend_name="whisper.cpp",
        model_name=model_name,
    )


def whispercpp_segment_words(
    raw_tokens: object,
    *,
    segment_start: float,
    segment_end: float,
) -> tuple[TranscriptionWord, ...]:
    """Build per-word timings from the tokens of a ``-ojf`` whisper.cpp segment.

    Sub-word tokens (no leading space) extend the previous word; special
    tokens such as ``[_BEG_]`` are skipped. Timings are clamped into the
    segment and kept non-decreasing. Any malformed token drops the words of
    the whole segment, because a partial word list would misplace the rest.
    """
    if not isinstance(raw_tokens, list) or segment_end <= segment_start:
        return ()
    words: list[tuple[float, float, str]] = []
    # A whitespace-only token separates words although it carries no text.
    boundary_pending = True
    for raw_token in raw_tokens:
        if not isinstance(raw_token, dict):
            return ()
        token_text = raw_token.get("text")
        if not isinstance(token_text, str):
            return ()
        if token_text.startswith("[_"):
            continue
        if not token_text.strip():
            boundary_pending = True
            continue
        offsets = raw_token.get("offsets")
        if not isinstance(offsets, dict):
            return ()
        raw_from, raw_to = offsets.get("from"), offsets.get("to")
        if (
            isinstance(raw_from, bool)
            or isinstance(raw_to, bool)
            or not isinstance(raw_from, int | float)
            or not isinstance(raw_to, int | float)
        ):
            return ()
        try:
            from_seconds, to_seconds = float(raw_from) / 1000.0, float(raw_to) / 1000.0
        except OverflowError:
            return ()
        if not math.isfinite(from_seconds) or not math.isfinite(to_seconds):
            return ()
        floor = words[-1][0] if words else segment_start
        token_start = min(max(from_seconds, floor), segment_end)
        token_end = min(max(to_seconds, token_start), segment_end)
        if words and not boundary_pending and not token_text.startswith(" "):
            previous_start, previous_end, previous_text = words[-1]
            words[-1] = (previous_start, max(previous_end, token_end), previous_text + token_text)
        else:
            words.append((token_start, token_end, token_text.strip()))
        boundary_pending = False
    return tuple(TranscriptionWord(start=start, end=end, text=text) for start, end, text in words)


def call_whisperkit_server(
    *,
    base_url: str,
    audio_path: Path,
    model_name: str,
    language: str | None,
    prompt: str | None,
    timeout_seconds: float,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    def check_cancelled() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise InferenceCancelled("WhisperKit inference was cancelled by the caller.")

    check_cancelled()
    try:
        parsed = urlparse(base_url)
    except ValueError:
        raise BackendUnavailableError("WhisperKit sidecar URL configuration is invalid.") from None
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise BackendUnavailableError(
            "VOXHELM_WHISPERKIT_BASE_URL must be an absolute http(s) URL."
        )

    boundary = f"voxhelm-{uuid4().hex}"
    endpoint_path = (parsed.path.rstrip("/") or "") + "/audio/transcriptions"
    query = f"?{parsed.query}" if parsed.query else ""
    request_path = endpoint_path + query
    file_content_type = mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream"

    fields: list[tuple[str, str]] = [
        ("model", model_name),
        ("response_format", "verbose_json"),
        ("timestamp_granularities[]", "segment"),
    ]
    if language:
        fields.append(("language", language))
    if prompt:
        fields.append(("prompt", prompt))

    payload_prefix = bytearray()
    for name, value in fields:
        payload_prefix.extend(
            render_multipart_field(boundary=boundary, name=name, value=value).encode("utf-8")
        )
    payload_prefix.extend(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{audio_path.name}"\r\n'
            f"Content-Type: {file_content_type}\r\n\r\n"
        ).encode("utf-8")
    )
    payload_suffix = f"\r\n--{boundary}--\r\n".encode("utf-8")
    try:
        content_length = len(payload_prefix) + audio_path.stat().st_size + len(payload_suffix)
    except OSError:
        raise RuntimeError("WhisperKit request media could not be read.") from None

    async def content():
        yield bytes(payload_prefix)
        with audio_path.open("rb") as handle:
            while chunk := handle.read(64 * 1024):
                yield chunk
        yield payload_suffix

    async def exchange() -> dict[str, Any]:
        try:
            async with asyncio.timeout(timeout_seconds):
                # Use the public transport directly: AsyncClient's request log
                # includes full URLs, which can carry private query parameters.
                async with httpx.AsyncHTTPTransport() as transport:
                    request = httpx.Request(
                        "POST",
                        f"{parsed.scheme}://{parsed.netloc}{request_path}",
                        content=content(),
                        headers={
                            "Content-Type": f"multipart/form-data; boundary={boundary}",
                            "Content-Length": str(content_length),
                            "Accept-Encoding": "identity",
                        },
                    )
                    response = await transport.handle_async_request(request)
                    try:
                        limit = (
                            WHISPERKIT_ERROR_MAX_BYTES
                            if response.status_code >= 400
                            else WHISPERKIT_RESPONSE_MAX_BYTES
                        )
                        body = bytearray()
                        async with aclosing(
                            cast(AsyncGenerator[bytes, None], response.aiter_raw())
                        ) as chunks:
                            async for chunk in chunks:
                                if len(chunk) > limit - len(body):
                                    raise RuntimeError(
                                        "WhisperKit server exceeded the response byte limit."
                                    )
                                body.extend(chunk)
                        if response.status_code >= 400:
                            status = response.status_code
                            _LOGGER.warning("WhisperKit sidecar failure: HTTP %s.", status)
                            raise RuntimeError(
                                f"WhisperKit transcription failed with HTTP {status}."
                            )
                        try:
                            return decode_json_object(bytes(body))
                        except (ValueError, RecursionError) as exc:
                            raise RuntimeError("WhisperKit server returned invalid JSON.") from exc
                    finally:
                        await response.aclose()
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise RuntimeError(
                "WhisperKit exchange exceeded the configured elapsed deadline."
            ) from exc
        except httpx.ConnectError as exc:
            raise OSError("WhisperKit server is not reachable.") from exc
        except httpx.HTTPError:
            _LOGGER.warning("WhisperKit sidecar failure: response transport.")
            raise RuntimeError("WhisperKit response transport failed.") from None
        except OSError:
            raise RuntimeError("WhisperKit request media could not be read.") from None
        except ValueError, httpx.InvalidURL:
            raise RuntimeError("WhisperKit sidecar configuration is invalid.") from None

    # Match the worker-control transport: own an isolated loop so sync callers can
    # interrupt the entire upload/read and close streams before releasing the lane.
    # A cancelled OS DNS lookup may finish later, without owning an HTTP stream/file.
    loop = asyncio.new_event_loop()
    task = loop.create_task(exchange())

    async def cancel_on_signal() -> None:
        assert cancel_event is not None
        while not cancel_event.is_set():
            await asyncio.sleep(0.05)
        task.cancel()

    watcher = loop.create_task(cancel_on_signal()) if cancel_event is not None else None
    try:
        try:
            payload = loop.run_until_complete(task)
        except asyncio.CancelledError:
            check_cancelled()
            raise
        check_cancelled()
        return payload
    finally:
        if watcher is not None:
            watcher.cancel()
            loop.run_until_complete(asyncio.gather(watcher, return_exceptions=True))
        if not task.done():
            task.cancel()
            loop.run_until_complete(asyncio.gather(task, return_exceptions=True))
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


def render_multipart_field(*, boundary: str, name: str, value: str) -> str:
    return f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'


def timestamp_to_seconds(timestamp: str) -> float:
    normalized = timestamp.replace(".", ",")
    try:
        hours, minutes, seconds = normalized.split(":")
        whole_seconds, milliseconds = seconds.split(",")
        return (
            float(hours) * 3600
            + float(minutes) * 60
            + float(whole_seconds)
            + float(milliseconds) / 1000
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid whisper.cpp timestamp '{timestamp}'.") from exc


def render_verbose_json(result: TranscriptionResult) -> dict[str, Any]:
    return transcript_formats.render_verbose_json(result)


def render_vtt(result: TranscriptionResult) -> str:
    return transcript_formats.render_vtt(result)


def get_backend_service() -> BackendProtocol:
    return build_backend_service(
        backend_name=settings.VOXHELM_STT_BACKEND,
        model_name=resolve_model_name_for_backend(
            request_model="auto",
            backend_name=settings.VOXHELM_STT_BACKEND,
        ),
    )


def get_backend_services_for_model(request_model: str) -> list[BackendInvocation]:
    primary_backend = resolve_backend_name_for_model(request_model)
    services = [
        BackendInvocation(
            primary_backend,
            service_for_backend_name(primary_backend, request_model=request_model),
        )
    ]
    fallback_backend = settings.VOXHELM_STT_FALLBACK_BACKEND.strip()
    if (
        is_auto_backend_model_request(request_model)
        and fallback_backend
        and fallback_backend != primary_backend
    ):
        services.append(
            BackendInvocation(
                fallback_backend,
                service_for_backend_name(fallback_backend, request_model=request_model),
            )
        )
    return services


def known_speaker_backend_chain(
    request_model: str, *, language: str | None = None
) -> list[str] | None:
    """Ordered backends for a known-speaker job with the re-cut, or ``None``.

    Only automatic model requests are routed, and only when
    ``VOXHELM_KNOWN_SPEAKER_STT_BACKEND`` names Parakeet. whisper.cpp comes
    next because it is the only other backend that returns word timings;
    after it the configured chain follows, without words and so without the
    re-cut, exactly as such a job behaves without routing. A language Parakeet
    does not support leaves it out of the chain.
    """
    if settings.VOXHELM_KNOWN_SPEAKER_STT_BACKEND != PARAKEET_BACKEND_NAME:
        return None
    if not is_auto_backend_model_request(request_model):
        return None
    chain: list[str] = []
    for name in (
        PARAKEET_BACKEND_NAME if parakeet_supports_language(language) else "",
        "whispercpp",
        settings.VOXHELM_STT_BACKEND,
        settings.VOXHELM_STT_FALLBACK_BACKEND.strip(),
    ):
        if name and name not in chain:
            chain.append(name)
    return chain


def parakeet_supports_language(language: str | None) -> bool:
    """A job without a language is passed through; Parakeet detects it."""
    normalized = normalize_language_code(language)
    return not normalized or normalized in PARAKEET_LANGUAGES


def get_backend_services_for_params(params: TranscribeParams) -> list[BackendInvocation]:
    if params.backend:
        return explicit_backend_chain(params)
    # Word timings are asked for exactly by known-speaker jobs with the re-cut.
    chain = (
        known_speaker_backend_chain(params.request_model, language=params.language)
        if params.word_timestamps
        else None
    )
    if chain is None:
        return get_backend_services_for_model(params.request_model)
    return [
        BackendInvocation(name, service_for_backend_name(name, request_model=params.request_model))
        for name in chain
    ]


def explicit_backend_chain(params: TranscribeParams) -> list[BackendInvocation]:
    """The requested backend, then mlx-whisper if that one is unavailable."""
    assert params.backend
    chain = [
        BackendInvocation(
            params.backend,
            build_backend_service(
                backend_name=params.backend,
                model_name=resolve_model_name_for_backend(
                    request_model=params.request_model, backend_name=params.backend
                ),
            ),
        )
    ]
    if params.backend != "mlx":
        chain.append(
            BackendInvocation(
                "mlx",
                build_backend_service(backend_name="mlx", model_name=settings.VOXHELM_MLX_MODEL),
            )
        )
    return chain


def service_for_backend_name(backend_name: str, *, request_model: str) -> BackendProtocol:
    if backend_name == settings.VOXHELM_STT_BACKEND and is_auto_backend_model_request(
        request_model
    ):
        return get_backend_service()
    return build_backend_service(
        backend_name=backend_name,
        model_name=resolve_model_name_for_backend(
            request_model=request_model,
            backend_name=backend_name,
        ),
    )


def build_backend_service(*, backend_name: str, model_name: str) -> BackendProtocol:
    if backend_name == "mlx":
        return MlxWhisperBackend(
            model_name=model_name,
            condition_on_previous_text=settings.VOXHELM_MLX_CONDITION_ON_PREVIOUS_TEXT,
        )
    if backend_name == "whispercpp":
        return WhisperCppBackend(
            binary_path=settings.VOXHELM_WHISPERCPP_BIN,
            model_name=model_name,
            processors=settings.VOXHELM_WHISPERCPP_PROCESSORS,
            max_context=settings.VOXHELM_WHISPERCPP_MAX_CONTEXT,
            suppress_nst=settings.VOXHELM_WHISPERCPP_SUPPRESS_NST,
            max_gap_seconds=settings.VOXHELM_WHISPERCPP_MAX_GAP_SECONDS,
        )
    if backend_name == "whisperkit":
        return WhisperKitBackend(
            enabled=settings.VOXHELM_WHISPERKIT_ENABLED,
            base_url=settings.VOXHELM_WHISPERKIT_BASE_URL,
            model_name=model_name,
            timeout_seconds=settings.VOXHELM_WHISPERKIT_TIMEOUT_SECONDS,
        )
    if backend_name == PARAKEET_BACKEND_NAME:
        return ParakeetBackend(
            binary_path=settings.VOXHELM_PARAKEET_BIN,
            model_dir=settings.VOXHELM_PARAKEET_MODEL_DIR,
            model_name=model_name,
            chunk_seconds=settings.VOXHELM_PARAKEET_CHUNK_SECONDS,
            overlap_seconds=settings.VOXHELM_PARAKEET_OVERLAP_SECONDS,
            max_sentence_seconds=settings.VOXHELM_PARAKEET_MAX_SENTENCE_SECONDS,
            timeout_seconds=settings.VOXHELM_PARAKEET_TIMEOUT_SECONDS,
        )
    if backend_name == PARAKEET_INPROCESS_BACKEND_NAME:
        return ParakeetInProcessBackend(
            model_dir=settings.VOXHELM_PARAKEET_MODEL_DIR,
            model_name=model_name,
        )
    raise RuntimeError(f"Unsupported STT backend '{backend_name}'.")


def resolve_backend_name_for_model(request_model: str) -> str:
    if is_auto_backend_model_request(request_model):
        return settings.VOXHELM_STT_BACKEND
    if request_model in WHISPERKIT_BACKEND_MODEL_NAMES:
        return "whisperkit"
    if request_model == settings.VOXHELM_WHISPERCPP_MODEL:
        return "whispercpp"
    if request_model == settings.VOXHELM_MLX_MODEL:
        return "mlx"
    if request_model == settings.VOXHELM_WHISPERKIT_MODEL:
        return "whisperkit"
    return settings.VOXHELM_STT_BACKEND


def resolve_model_name_for_backend(*, request_model: str, backend_name: str) -> str:
    if request_model in WHISPERKIT_BACKEND_MODEL_NAMES:
        if backend_name == "whisperkit":
            return settings.VOXHELM_WHISPERKIT_MODEL
        raise RuntimeError(
            "Cannot resolve WhisperKit request model "
            f"'{request_model}' for backend '{backend_name}'."
        )
    if not is_auto_backend_model_request(request_model):
        return request_model
    if backend_name == "whispercpp":
        return settings.VOXHELM_WHISPERCPP_MODEL
    if backend_name == "mlx":
        return settings.VOXHELM_MLX_MODEL
    if backend_name == "whisperkit":
        return settings.VOXHELM_WHISPERKIT_MODEL
    if backend_name in (PARAKEET_BACKEND_NAME, PARAKEET_INPROCESS_BACKEND_NAME):
        return settings.VOXHELM_PARAKEET_MODEL_NAME
    raise RuntimeError(f"Unsupported STT backend '{backend_name}'.")


def is_auto_backend_model_request(request_model: str) -> bool:
    return request_model in AUTO_BACKEND_MODEL_NAMES


def resolve_whispercpp_binary(binary_path: str) -> str:
    if "/" in binary_path:
        candidate = Path(binary_path)
        if not candidate.exists():
            raise BackendUnavailableError(f"whisper.cpp binary was not found at '{binary_path}'.")
        return str(candidate)
    resolved = shutil.which(binary_path)
    if resolved is None:
        raise BackendUnavailableError(f"whisper.cpp binary '{binary_path}' was not found in PATH.")
    return resolved


def resolve_whispercpp_model_path(model_name: str) -> Path:
    configured = Path(model_name).expanduser()
    if configured.is_absolute() or configured.parent != Path("."):
        candidate = configured
    else:
        candidate = settings.VOXHELM_MODEL_CACHE_DIR / model_name
    if not candidate.exists():
        raise BackendUnavailableError(
            "whisper.cpp model "
            f"'{candidate}' is missing. Deploy the model before selecting this backend."
        )
    return candidate


def transcribe_audio(audio_path: Path, params: TranscribeParams) -> TranscriptionResult:
    # Concurrency model: subprocess backends (whisper.cpp) and sidecar backends
    # (WhisperKit) are bounded by the cross-process lane scheduler alone, so two
    # of them may overlap on the GPU. Backends that keep model state in this
    # process take their own per-backend lock (see ``_MLX_LOCK``); that lock is
    # process-local, so the Wyoming sidecar serializes independently.
    with admit_local_inference(params.scheduler_lane, cancel_event=params.cancel_event):
        _raise_if_cancelled(params)
        unavailable_errors: list[str] = []
        invocations = get_backend_services_for_params(params)
        if not params.allow_backend_fallback:
            invocations = invocations[:1]
        for invocation in invocations:
            _raise_if_cancelled(params)
            try:
                result = invocation.service.transcribe(audio_path, params)
            except BackendUnavailableError as exc:
                unavailable_errors.append(f"{invocation.name}: {exc}")
                continue
            return sanitize_result(
                result,
                enabled=settings.VOXHELM_SANITIZE_TRANSCRIPT,
                repeat_threshold=settings.VOXHELM_SANITIZE_REPEAT_THRESHOLD,
            )

        # A cancellation that arrived while the last backend ran must surface as
        # InferenceCancelled, never as a backend-availability failure.
        _raise_if_cancelled(params)
        joined = "; ".join(unavailable_errors)
        raise BackendUnavailableError(f"No configured STT backend is available. {joined}")


def serialize_health() -> str:
    return json.dumps({"status": "ok"})
