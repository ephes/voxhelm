"""Chapter windows: whisper large-v3-turbo word windows around sponsor-chapter edges.

The podcast backend (Delve) checks the text around each sponsor-chapter edge of an
episode to move the edge onto the advertisement's real start or end (Server-Side Ad
Analysis v1, stage 10). Its gate (G6) passes only on the words the ad research
transcribed for that check (``work/night-2026-10-14/round2-chapter-textcheck``,
``asrwin120.py``, FREEZE.txt 2026-10-15T00:58:02Z), not on Parakeet's. This module
reproduces that transcription exactly, as the ``turbo-window-v1`` rule (D-29):

- for a mark ``t`` (the chapter edge, whole milliseconds, file time) the clip
  ``[max(0, t - 60 + shift), +120 s)`` is decoded by ffmpeg to 16 kHz mono WAV with the
  research arguments (``-ss`` and ``-t`` as input options, 3 decimals);
- ``whisper-cli`` runs the pinned ggml turbo model (the drift check's file, SHA-256
  pinned in ``transcriptions.windows``) in English with DTW token timestamps
  (``-dtw large.v3.turbo``), one word per segment (``-ml 1 -sow``), the research prompt
  ``Hello, and welcome back. Thanks!`` and plain JSON output; the decoding settings are
  whisper.cpp 1.9.5's defaults, passed explicitly (4 threads, 1 processor, beam size 5,
  best-of 5, temperature 0.00 with 0.20 increments);
- when whisper.cpp exits with an error (its DTW sometimes aborts) the clip is shifted by
  0.7, -0.7, 1.9 and -1.9 s in turn, as the research did; a window none of the five
  clips transcribes is returned with ``ok: false`` and no words (the edge keeps its
  mark), its exit statuses recorded;
- words are every non-blank segment's stripped text with the clip start plus its
  offsets in seconds, rounded to 3 decimals, in output order (whisper.cpp's JSON is
  decoded as UTF-8 with replacement characters, as the research read it).

It never runs another model: a missing ``whisper-cli``, a missing turbo model or one with
another hash raises ``BackendUnavailableError`` (resolved by the caller before
anything runs). Nothing outlives the call: clips and whisper.cpp output live in one
temporary directory removed on every exit path, and error messages carry whisper.cpp's
or ffmpeg's log, never a transcript. The caller admits the run through the lane
scheduler (the profile runs it inside its own admission).
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any

from django.conf import settings

from transcriptions import windows as window_module
from transcriptions.service import (
    BackendUnavailableError,
    run_cancellable_process,
)
from transcriptions.windows import raise_if_cancelled, resolve_window_runtime, whispercpp_version

CHAPTER_WINDOWS_RULE = "turbo-window-v1"
CHAPTER_WINDOWS_ARTIFACT_NAME = "chapter_windows.json"
CHAPTER_WINDOWS_ARTIFACT_FORMAT = "chapter_windows"
CHAPTER_WINDOWS_CONTENT_TYPE = "application/json"
CHAPTER_WINDOWS_SCHEMA_VERSION = 1
# Delve checks at most 128 sponsor blocks, two edges each.
MAX_CHAPTER_MARKS = 256
# 24 h in milliseconds; Delve analyses at most 6 h.
MAX_CHAPTER_MARK_MS = 24 * 60 * 60 * 1000
RESEARCH_PROMPT = "Hello, and welcome back. Thanks!"
_ERROR_LOG_LINES = 20


class ChapterMarkError(ValueError):
    """A job's chapter marks are not ones the profile can transcribe (request fault)."""


@dataclass(frozen=True)
class ChapterWindowSettings:
    """Every clip and whisper.cpp setting that changes a window's words."""

    half_window_seconds: float = 60.0
    sample_rate: int = 16000
    language: str = "en"
    dtw: str = "large.v3.turbo"
    max_len: int = 1
    split_on_word: bool = True
    prompt: str = RESEARCH_PROMPT
    shifts_seconds: tuple[float, ...] = (0.0, 0.7, -0.7, 1.9, -1.9)
    beam_size: int = 5
    best_of: int = 5
    temperature: float = 0.0
    temperature_inc: float = 0.2
    threads: int = 4
    processors: int = 1

    @property
    def clip_seconds(self) -> float:
        return 2 * self.half_window_seconds

    def record(self) -> dict[str, Any]:
        return {
            "half_window_seconds": self.half_window_seconds,
            "clip_seconds": self.clip_seconds,
            "sample_rate": self.sample_rate,
            "language": self.language,
            "dtw": self.dtw,
            "max_len": self.max_len,
            "split_on_word": self.split_on_word,
            "prompt": self.prompt,
            "prompt_sha256": hashlib.sha256(self.prompt.encode("utf-8")).hexdigest(),
            "shifts_seconds": list(self.shifts_seconds),
            "decoding": "beam_search",
            "beam_size": self.beam_size,
            "best_of": self.best_of,
            "temperature": self.temperature,
            "temperature_inc": self.temperature_inc,
            "threads": self.threads,
            "processors": self.processors,
        }


CHAPTER_WINDOW_SETTINGS = ChapterWindowSettings()


@dataclass(frozen=True)
class ChapterWindows:
    """The windows of one job: the ``chapter_windows.json`` document and its summary."""

    identity: dict[str, Any]
    duration_ms: int
    windows: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "windows": len(self.windows),
            "ok": sum(1 for window in self.windows if window["ok"]),
            "seconds": round(sum(window["seconds"] for window in self.windows), 3),
        }

    def document(self, *, profile: str) -> dict[str, Any]:
        return {
            "schema_version": CHAPTER_WINDOWS_SCHEMA_VERSION,
            "profile": profile,
            "rule": CHAPTER_WINDOWS_RULE,
            "identity": self.identity,
            "duration_ms": self.duration_ms,
            "summary": self.summary(),
            "windows": self.windows,
        }

    def metadata_record(self) -> dict[str, Any]:
        """What the job result's metadata keeps (no words)."""
        return {"identity": self.identity, **self.summary()}


def parse_chapter_marks(value: object) -> list[int] | None:
    """A request's ``chapter_marks_ms``: ``None`` when absent, else the checked marks.

    Marks are whole milliseconds of file time, at least one and at most
    ``MAX_CHAPTER_MARKS``, each in ``[0, 24 h]``, strictly increasing (so unique). Raises
    ``ChapterMarkError`` otherwise; whether they lie within the media is checked once the
    source is materialised (``check_marks_within``).
    """
    if value is None:
        return None
    if not isinstance(value, list):
        raise ChapterMarkError("chapter_marks_ms must be a list of whole milliseconds.")
    if not value:
        raise ChapterMarkError("chapter_marks_ms must not be empty; omit it for no windows.")
    if len(value) > MAX_CHAPTER_MARKS:
        raise ChapterMarkError(f"chapter_marks_ms holds at most {MAX_CHAPTER_MARKS} marks.")
    marks: list[int] = []
    for mark in value:
        if isinstance(mark, bool) or not isinstance(mark, int):
            raise ChapterMarkError("chapter_marks_ms must hold whole milliseconds (integers).")
        if not 0 <= mark <= MAX_CHAPTER_MARK_MS:
            raise ChapterMarkError(
                f"chapter_marks_ms must lie in [0, {MAX_CHAPTER_MARK_MS}] milliseconds."
            )
        if marks and mark <= marks[-1]:
            raise ChapterMarkError("chapter_marks_ms must be strictly increasing.")
        marks.append(mark)
    return marks


def resolve_ffprobe_binary() -> str:
    """``VOXHELM_FFPROBE_BIN`` as an executable path, or ``BackendUnavailableError``."""
    binary = settings.VOXHELM_FFPROBE_BIN
    if "/" in binary:
        if not Path(binary).is_file():
            raise BackendUnavailableError(f"ffprobe binary was not found at '{binary}'.")
        return binary
    resolved = shutil.which(binary)
    if resolved is None:
        raise BackendUnavailableError(f"ffprobe binary '{binary}' was not found in PATH.")
    return resolved


def ffprobe_runtime_available() -> bool:
    try:
        resolve_ffprobe_binary()
    except BackendUnavailableError:
        return False
    return True


def media_duration_ms(
    audio_path: Path, *, ffprobe: str, cancel_event: threading.Event | None = None
) -> int:
    """The container duration ffprobe reports, in whole milliseconds (rounded half up).

    The same measure Delve takes of the file (``smartskipfetch`` ``FFprobe``:
    ``format.duration``, ``math.Round(seconds * 1000)``).
    """
    try:
        completed = run_cancellable_process(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            cancel_event=cancel_event,
            timeout_seconds=settings.VOXHELM_AUDIO_DECODE_TIMEOUT_SECONDS,
        )
    except OSError as exc:
        raise RuntimeError(f"ffprobe could not be started: {exc}") from exc
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe could not read the media: {_log_tail(completed)}")
    try:
        seconds = float((completed.stdout or "").strip())
    except ValueError as exc:
        raise RuntimeError("ffprobe reported no media duration.") from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise RuntimeError("ffprobe reported no media duration.")
    return math.floor(seconds * 1000 + 0.5)


def check_marks_within(marks_ms: list[int], duration_ms: int) -> None:
    """Every mark must lie within the media (at most its last millisecond)."""
    beyond = [mark for mark in marks_ms if mark > duration_ms]
    if beyond:
        raise ChapterMarkError(
            f"{len(beyond)} chapter mark(s) lie after the media's end ({duration_ms} ms)."
        )


def chapter_windows_identity(executable: str) -> dict[str, Any]:
    return {
        "rule": CHAPTER_WINDOWS_RULE,
        "engine": window_module.WINDOW_ENGINE,
        "cli": window_module.WINDOW_CLI,
        "cli_version": whispercpp_version(executable),
        "model": window_module.WINDOW_MODEL_NAME,
        "model_sha256": window_module.WINDOW_MODEL_SHA256,
        "settings": CHAPTER_WINDOW_SETTINGS.record(),
    }


def expected_chapter_windows_identity() -> dict[str, Any]:
    """The identity without the whisper.cpp version (the part a worker must not change)."""
    return {
        "rule": CHAPTER_WINDOWS_RULE,
        "engine": window_module.WINDOW_ENGINE,
        "cli": window_module.WINDOW_CLI,
        "model": window_module.WINDOW_MODEL_NAME,
        "model_sha256": window_module.WINDOW_MODEL_SHA256,
        "settings": CHAPTER_WINDOW_SETTINGS.record(),
    }


def run_chapter_windows(
    audio_path: Path,
    marks_ms: list[int],
    *,
    executable: str,
    model_path: Path,
    duration_ms: int,
    cancel_event: threading.Event | None = None,
) -> ChapterWindows:
    """Transcribe the window of every mark, one after another, in mark order.

    ``executable`` and ``model_path`` come from ``resolve_window_runtime`` (the pinned
    turbo model), which is checked again before the first window and after every clip
    whisper.cpp failed on: ``BackendUnavailableError`` when the model went away or
    changed. Raises ``RuntimeError`` when ffmpeg fails, whisper.cpp cannot be
    started, times out or writes malformed JSON, and ``InferenceCancelled`` when
    ``cancel_event`` is set; a window whose five clips all end in a whisper.cpp error is
    ``ok: false``.
    """
    check_marks_within(marks_ms, duration_ms)
    resolve_window_runtime()
    windows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="voxhelm-chapter-windows-") as temp_dir:
        temp_root = Path(temp_dir)
        for mark in marks_ms:
            raise_if_cancelled(cancel_event)
            windows.append(
                transcribe_chapter_window(
                    audio_path,
                    mark,
                    executable=executable,
                    model_path=model_path,
                    temp_root=temp_root,
                    cancel_event=cancel_event,
                )
            )
    identity = chapter_windows_identity(executable)
    raise_if_cancelled(cancel_event)
    return ChapterWindows(identity=identity, duration_ms=duration_ms, windows=windows)


def transcribe_chapter_window(
    audio_path: Path,
    mark_ms: int,
    *,
    executable: str,
    model_path: Path,
    temp_root: Path,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    """One window ``{mark_ms, ok, clip_start, shift, exit_codes, seconds, words}``."""
    window = CHAPTER_WINDOW_SETTINGS
    t = mark_ms / 1000
    started = monotonic()
    clip_path = temp_root / "x.wav"
    output_base = temp_root / "x"
    json_path = temp_root / "x.json"
    exit_codes: list[int] = []
    ok = False
    clip_start = 0.0
    shift = 0.0
    for shift in window.shifts_seconds:
        raise_if_cancelled(cancel_event)
        clip_start = max(0.0, t - window.half_window_seconds + shift)
        decode_chapter_clip(audio_path, clip_path, start=clip_start, cancel_event=cancel_event)
        raise_if_cancelled(cancel_event)
        json_path.unlink(missing_ok=True)
        args = [
            executable,
            "-m",
            str(model_path),
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
            "-dtw",
            window.dtw,
            "-ml",
            str(window.max_len),
            "-sow",
            "--prompt",
            window.prompt,
            "-oj",
            "-of",
            str(output_base),
            "-np",
            str(clip_path),
        ]
        try:
            completed = run_cancellable_process(
                args,
                cancel_event=cancel_event,
                timeout_seconds=settings.VOXHELM_WINDOW_TIMEOUT_SECONDS,
            )
        except OSError as exc:
            raise RuntimeError(f"whisper.cpp could not be started: {exc}") from exc
        exit_codes.append(completed.returncode)
        if completed.returncode == 0 and json_path.is_file():
            ok = True
            break
        # A failed clip is the research's DTW abort only while the pinned model is still
        # there; a model that went away (or was replaced) fails the run instead of
        # passing as an untranscribed window.
        resolve_window_runtime()
    words: list[dict[str, Any]] = []
    if ok:
        words = chapter_window_words(json_path.read_bytes(), clip_start=clip_start)
    json_path.unlink(missing_ok=True)
    clip_path.unlink(missing_ok=True)
    return {
        "mark_ms": mark_ms,
        "ok": ok,
        "clip_start": round(clip_start, 3),
        "shift": shift,
        "exit_codes": exit_codes,
        "seconds": round(monotonic() - started, 3),
        "words": words,
    }


def decode_chapter_clip(
    audio_path: Path,
    clip_path: Path,
    *,
    start: float,
    cancel_event: threading.Event | None = None,
) -> None:
    """ffmpeg-decode ``[start, start + 120)`` to 16 kHz mono WAV, as ``asrwin120.py``."""
    args = [
        settings.VOXHELM_FFMPEG_BIN,
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{CHAPTER_WINDOW_SETTINGS.clip_seconds:.3f}",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        str(CHAPTER_WINDOW_SETTINGS.sample_rate),
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
        raise RuntimeError(f"ffmpeg chapter clip decoding failed: {_log_tail(completed)}")


def chapter_window_words(raw: bytes, *, clip_start: float) -> list[dict[str, Any]]:
    """Words ``{w, s, e}`` in file seconds from ``whisper-cli -oj -ml 1 -sow`` output.

    As ``asrwin120.py``: the bytes decoded as UTF-8 with replacement characters (a word
    split inside a multi-byte character stays readable), every segment whose stripped
    text is not empty, start and end the clip start plus the segment's offsets in
    seconds, rounded to 3 decimals. Malformed output raises ``RuntimeError``.
    """
    try:
        payload = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("whisper.cpp chapter window produced invalid JSON.") from exc
    segments = payload.get("transcription") if isinstance(payload, dict) else None
    if not isinstance(segments, list):
        raise RuntimeError("whisper.cpp chapter window output has no transcription list.")
    words: list[dict[str, Any]] = []
    for segment in segments:
        if not isinstance(segment, dict):
            raise RuntimeError("whisper.cpp chapter window output has a malformed segment.")
        text = segment.get("text")
        offsets = segment.get("offsets")
        if not isinstance(text, str) or not isinstance(offsets, dict):
            raise RuntimeError("whisper.cpp chapter window output has a malformed segment.")
        values: list[int] = []
        for key in ("from", "to"):
            value = offsets.get(key)
            if isinstance(value, bool) or not isinstance(value, int):
                raise RuntimeError("whisper.cpp chapter window output has malformed offsets.")
            values.append(value)
        stripped = text.strip()
        if not stripped:
            continue
        words.append(
            {
                "w": stripped,
                "s": round(clip_start + values[0] / 1000, 3),
                "e": round(clip_start + values[1] / 1000, 3),
            }
        )
    return words


def _log_tail(completed: Any) -> str:
    lines = [line for line in (completed.stderr or "").splitlines() if line.strip()]
    return "\n".join(lines[-_ERROR_LOG_LINES:]) or f"exit status {completed.returncode}"
