"""Window transcription: whisper large-v3-turbo on one 32 s clip of a local source.

The German drift check of the ``ad-analysis-v1`` profile (podcast server-side ad
analysis) re-transcribes a flagged 30 s grid window ``[t, t + 30)`` with whisper
large-v3-turbo and keeps the result only when whisper detects the episode language on
the clip. This module is that retry, reproducing the research runner
(``round2-german-drift/retry.py``) exactly:

- the clip ``[max(0, t - 1), max(0, t - 1) + 32)`` is decoded by ffmpeg from the source
  file to 16 kHz mono WAV with the runner's arguments;
- ``whisper-cli`` runs the pinned ggml model (identified by its SHA-256) with the
  decoding the runner's ``whisper-server`` used: language detected on the clip (never
  forced), greedy with best-of 2, temperature 0.0 with 0.2 increments, 4 threads, one
  processor, segments of at most 60 characters, token timestamps;
- words are assembled from the tokens as the runner does and kept when their start lies
  in ``[t, t + 30)``, in file time; the detected language and its probability come back
  with them.

It never runs another model: a missing binary, a missing model file or one whose hash is
not the pinned one raises ``BackendUnavailableError``. Nothing outlives the call: the
clip and whisper.cpp's output live in a temporary directory removed on every exit path,
no job, artifact or task row is written, and error messages carry whisper.cpp's log, not
its transcript. The caller admits the run through the lane scheduler (the profile runs
it inside its own admission).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import tempfile
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from django.conf import settings

from lane_scheduler import InferenceCancelled
from transcriptions.service import (
    BackendUnavailableError,
    resolve_whispercpp_binary,
    resolve_whispercpp_model_path,
    run_cancellable_process,
)

LOGGER = logging.getLogger(__name__)

WINDOW_ENGINE = "whisper.cpp"
WINDOW_CLI = "whisper-cli"
WINDOW_MODEL_NAME = "ggml-large-v3-turbo"
# The ggml file of the research runs (``models/ggml-large-v3-turbo.bin`` of the ad
# research workspace, FREEZE.txt prefix 1fc70f774d38eb16). A different file is a
# different engine, so it is refused rather than used.
WINDOW_MODEL_SHA256 = "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69"
# whisper.cpp's end-of-text token for the multilingual large-v3 vocabulary; it and every
# token above it are special (timestamps, language, task tokens). whisper-server leaves
# them out of its verbose_json words, which is what the research runner read.
WHISPER_TOKEN_EOT = 50257
WINDOW_SECONDS = 30.0
WINDOW_PAD_SECONDS = 1.0
SAMPLE_RATE = 16000
_PUNCTUATION = ".,!?;:"
_DETECTED_LANGUAGE = re.compile(
    r"auto-detected language: (?P<code>[a-z]{2,3}) \(p = (?P<p>[0-9]+(?:\.[0-9]+)?)\)"
)
_VERSION = re.compile(r"whisper\.cpp version:\s*(\S+)")
_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_ERROR_LOG_LINES = 20


@dataclass(frozen=True)
class WindowSettings:
    """Every whisper.cpp and clip setting that changes the result."""

    pad_seconds: float = WINDOW_PAD_SECONDS
    window_seconds: float = WINDOW_SECONDS
    sample_rate: int = SAMPLE_RATE
    language: str = "auto"
    beam_size: int = 1
    best_of: int = 2
    temperature: float = 0.0
    temperature_inc: float = 0.2
    threads: int = 4
    processors: int = 1
    max_len: int = 60

    @property
    def clip_seconds(self) -> float:
        return self.window_seconds + 2 * self.pad_seconds

    def record(self) -> dict[str, Any]:
        return {
            "pad_seconds": self.pad_seconds,
            "window_seconds": self.window_seconds,
            "clip_seconds": self.clip_seconds,
            "sample_rate": self.sample_rate,
            "language": self.language,
            "decoding": "greedy",
            "best_of": self.best_of,
            "temperature": self.temperature,
            "temperature_inc": self.temperature_inc,
            "threads": self.threads,
            "processors": self.processors,
            "max_len": self.max_len,
            "token_timestamps": True,
        }


WINDOW_SETTINGS = WindowSettings()


@dataclass(frozen=True)
class WindowTranscription:
    t: float
    clip_start: float
    language: str
    language_probability: float
    words: list[dict[str, Any]]
    identity: dict[str, Any]

    def as_record(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "clip_start": self.clip_start,
            "language": self.language,
            "language_probability": self.language_probability,
            "words": self.words,
            "identity": self.identity,
        }


def resolve_window_runtime() -> tuple[str, Path]:
    """The ``whisper-cli`` path and the verified turbo model, or ``BackendUnavailableError``."""
    executable = resolve_whispercpp_binary(settings.VOXHELM_WHISPERCPP_BIN)
    model_path = resolve_whispercpp_model_path(settings.VOXHELM_WINDOW_MODEL)
    digest = model_sha256(model_path)
    if digest != WINDOW_MODEL_SHA256:
        raise BackendUnavailableError(
            f"Window model '{model_path}' has SHA-256 {digest}, "
            f"not the pinned {WINDOW_MODEL_NAME} {WINDOW_MODEL_SHA256}."
        )
    return executable, model_path


def window_runtime_available() -> bool:
    """True when this host can run window transcription (binary and pinned model)."""
    try:
        resolve_window_runtime()
    except BackendUnavailableError:
        return False
    return True


def transcribe_window(
    audio_path: Path,
    t: float,
    *,
    cancel_event: threading.Event | None = None,
) -> WindowTranscription:
    """Re-transcribe the window ``[t, t + 30)`` of ``audio_path`` with the pinned turbo model.

    ``t`` is in seconds of file time. Raises ``BackendUnavailableError`` when the binary or
    the pinned model is missing, ``RuntimeError`` when ffmpeg or whisper.cpp fails or its
    output is malformed, ``InferenceCancelled`` when ``cancel_event`` is set.
    """
    if isinstance(t, bool) or not isinstance(t, int | float) or not math.isfinite(t) or t < 0:
        raise ValueError("Window start must be a finite, non-negative number of seconds.")
    window = WINDOW_SETTINGS
    raise_if_cancelled(cancel_event)
    executable, model_path = resolve_window_runtime()
    clip_start = max(0.0, float(t) - window.pad_seconds)
    with tempfile.TemporaryDirectory(prefix="voxhelm-window-") as temp_dir:
        temp_root = Path(temp_dir)
        clip_path = temp_root / "clip.wav"
        decode_clip(
            audio_path,
            clip_path,
            start=clip_start,
            seconds=window.clip_seconds,
            cancel_event=cancel_event,
        )
        # The process runner looks at the event only while it waits, so a fast child
        # can finish unnoticed; check between the steps and before returning.
        raise_if_cancelled(cancel_event)
        output_base = temp_root / "window"
        args = [
            executable,
            "-m",
            str(model_path),
            "-f",
            str(clip_path),
            "-l",
            window.language,
            "-t",
            str(window.threads),
            "-p",
            str(window.processors),
            "-bs",
            str(window.beam_size),
            "-bo",
            str(window.best_of),
            "-tp",
            f"{window.temperature:.2f}",
            "-tpi",
            f"{window.temperature_inc:.2f}",
            "-ml",
            str(window.max_len),
            "-ojf",
            "-of",
            str(output_base),
        ]
        try:
            completed = run_cancellable_process(
                args,
                cancel_event=cancel_event,
                timeout_seconds=settings.VOXHELM_WINDOW_TIMEOUT_SECONDS,
            )
        except OSError as exc:
            raise RuntimeError(f"whisper.cpp could not be started: {exc}") from exc
        # stdout carries the transcript; only the log may go into an error message.
        if completed.returncode != 0:
            raise RuntimeError(f"whisper.cpp window transcription failed: {_log_tail(completed)}")
        json_path = output_base.with_suffix(".json")
        try:
            raw = json_path.read_bytes()
        except OSError as exc:
            raise RuntimeError("whisper.cpp window transcription produced no JSON output.") from exc
        try:
            payload = json.loads(raw.decode("utf-8", errors="replace"), strict=False)
        except json.JSONDecodeError as exc:
            raise RuntimeError("whisper.cpp window transcription produced invalid JSON.") from exc
        language, probability = detected_language(completed.stderr, payload)
    words = window_words(payload, clip_start=clip_start, t=float(t))
    # Identity may run ``whisper-cli --version`` the first time; resolve it before the
    # last check so a cancellation during it is not answered with a result.
    identity = window_identity(executable)
    raise_if_cancelled(cancel_event)
    return WindowTranscription(
        t=float(t),
        clip_start=clip_start,
        language=language,
        language_probability=probability,
        words=words,
        identity=identity,
    )


def raise_if_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise InferenceCancelled("Window transcription was cancelled by the caller.")


def decode_clip(
    audio_path: Path,
    clip_path: Path,
    *,
    start: float,
    seconds: float,
    cancel_event: threading.Event | None = None,
) -> None:
    """ffmpeg-decode ``[start, start + seconds)`` to 16 kHz mono WAV, as the research runner.

    ``-ss`` and ``-t`` are input options (before ``-i``), formatted to milliseconds.
    """
    args = [
        settings.VOXHELM_FFMPEG_BIN,
        "-nostdin",
        "-v",
        "error",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{seconds:.3f}",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        str(WINDOW_SETTINGS.sample_rate),
        "-f",
        "wav",
        str(clip_path),
    ]
    try:
        completed = run_cancellable_process(
            args,
            cancel_event=cancel_event,
            timeout_seconds=settings.VOXHELM_AUDIO_DECODE_TIMEOUT_SECONDS,
        )
    except OSError as exc:
        raise RuntimeError(f"ffmpeg could not be started: {exc}") from exc
    if completed.returncode != 0 or not clip_path.is_file():
        raise RuntimeError(f"ffmpeg clip decoding failed: {_log_tail(completed)}")


def detected_language(stderr: str, payload: object) -> tuple[str, float]:
    """The language whisper.cpp detected on the clip and its probability.

    whisper-cli writes the probability only to its log (6 decimals); the JSON result
    carries the code, and both must agree.
    """
    matches = _DETECTED_LANGUAGE.findall(stderr or "")
    if len(matches) != 1:
        raise RuntimeError(
            "whisper.cpp window transcription did not report exactly one detected language."
        )
    code, probability = matches[0]
    result = payload.get("result") if isinstance(payload, dict) else None
    reported = result.get("language") if isinstance(result, dict) else None
    if reported != code:
        raise RuntimeError(
            "whisper.cpp window transcription reported inconsistent languages "
            f"({code!r} in the log, {reported!r} in the result)."
        )
    return code, float(probability)


def window_words(payload: object, *, clip_start: float, t: float) -> list[dict[str, Any]]:
    """Words ``{w, s, e}`` in file time from ``whisper-cli -ojf`` output, as the research runner.

    Tokens from every segment in order. Special tokens (id at or above end-of-text,
    which whisper-server leaves out), tokens whose text starts with ``[_`` and
    whitespace-only tokens are skipped. A token starts a new word when its text starts
    with a space (or no word exists yet) and its stripped text is not a substring of
    ``.,!?;:``; otherwise its stripped text is appended to the current word, whose end
    moves to the token's end. Times are clip start plus the token offset, rounded to 3
    decimals. Kept are the words whose start lies in ``[t, t + 30)``. Malformed output
    (missing segments, tokens, text, id or offsets) raises ``RuntimeError``.
    """
    segments = payload.get("transcription") if isinstance(payload, dict) else None
    if not isinstance(segments, list):
        raise RuntimeError("whisper.cpp output has no transcription list.")
    words: list[dict[str, Any]] = []
    for segment in segments:
        tokens = segment.get("tokens") if isinstance(segment, dict) else None
        if not isinstance(tokens, list):
            raise RuntimeError("whisper.cpp output has a segment without tokens.")
        for token in tokens:
            text, token_id, start_ms, end_ms = _token_fields(token)
            if token_id >= WHISPER_TOKEN_EOT:
                continue
            if text.startswith("[_") or not text.strip():
                continue
            start = clip_start + start_ms / 1000
            end = clip_start + end_ms / 1000
            if (text.startswith(" ") or not words) and text.strip() not in _PUNCTUATION:
                words.append({"w": text.strip(), "s": round(start, 3), "e": round(end, 3)})
            elif words:
                words[-1]["w"] += text.strip()
                words[-1]["e"] = round(end, 3)
    window_end = t + WINDOW_SETTINGS.window_seconds
    return [word for word in words if t <= word["s"] < window_end]


def _token_fields(token: object) -> tuple[str, int, int, int]:
    if not isinstance(token, dict):
        raise RuntimeError("whisper.cpp output has a malformed token.")
    text = token.get("text")
    token_id = token.get("id")
    offsets = token.get("offsets")
    if not isinstance(text, str):
        raise RuntimeError("whisper.cpp output has a token without text.")
    if isinstance(token_id, bool) or not isinstance(token_id, int):
        raise RuntimeError("whisper.cpp output has a token without id.")
    if not isinstance(offsets, dict):
        raise RuntimeError("whisper.cpp output has a token without offsets.")
    values: list[int] = []
    for key in ("from", "to"):
        value = offsets.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError("whisper.cpp output has a token without offsets.")
        values.append(value)
    return text, token_id, values[0], values[1]


def window_identity(executable: str) -> dict[str, Any]:
    return {
        "engine": WINDOW_ENGINE,
        "cli": WINDOW_CLI,
        "cli_version": whispercpp_version(executable),
        "model": WINDOW_MODEL_NAME,
        "model_sha256": WINDOW_MODEL_SHA256,
        "settings": WINDOW_SETTINGS.record(),
    }


def model_sha256(model_path: Path) -> str:
    """SHA-256 of the model file, cached per path, size, inode and modification time."""
    try:
        stat = model_path.stat()
    except OSError as exc:
        raise BackendUnavailableError(f"Window model '{model_path}' is not readable.") from exc
    return _model_sha256(str(model_path), stat.st_size, stat.st_ino, stat.st_mtime_ns)


@lru_cache(maxsize=4)
def _model_sha256(path: str, size: int, inode: int, mtime_ns: int) -> str:
    del size, inode, mtime_ns
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            while chunk := handle.read(_HASH_CHUNK_BYTES):
                digest.update(chunk)
    except OSError as exc:
        raise BackendUnavailableError(f"Window model '{path}' is not readable.") from exc
    return digest.hexdigest()


def whispercpp_version(executable: str) -> str:
    path = Path(executable)
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        mtime_ns = 0
    return _whispercpp_version(str(path), mtime_ns)


@lru_cache(maxsize=4)
def _whispercpp_version(executable: str, mtime_ns: int) -> str:
    del mtime_ns
    try:
        completed = run_cancellable_process(
            [executable, "--version"], cancel_event=None, timeout_seconds=30
        )
    except (OSError, RuntimeError) as exc:
        LOGGER.warning("whisper.cpp version check failed executable=%s error=%s", executable, exc)
        return ""
    match = _VERSION.search(f"{completed.stdout}\n{completed.stderr}")
    return match.group(1) if match else ""


def _log_tail(completed: Any) -> str:
    lines = [line for line in (completed.stderr or "").splitlines() if line.strip()]
    return "\n".join(lines[-_ERROR_LOG_LINES:]) or f"exit status {completed.returncode}"
