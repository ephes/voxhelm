"""Transcript coverage against the real media duration.

whisper.cpp 1.9.5 with ``-p N`` (N > 1) runs parts 2..N of the audio in worker
threads, ignores their return codes and merges whatever segments they produced.
A part that fails mid-way (encode or decode error) leaves a hole in the
transcript while ``whisper-cli`` still exits 0. Archive item 408 (2026-11-05)
came back "succeeded" with more than half of a 3 h episode missing.

The check here measures which stretches of the media no segment covers and
rejects a transcript whose largest uncovered stretch (at the start, between
segments or at the end) is longer than a threshold. It is a heuristic for
spoken-word media: a long stretch of music or silence without speech fails it
as well, and losses shorter than the threshold stay undetected.
"""

from __future__ import annotations

import math
import wave
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class IncompleteTranscriptError(RuntimeError):
    """A backend reported success, but its transcript leaves part of the media uncovered."""


class _Timed(Protocol):
    @property
    def start(self) -> float: ...

    @property
    def end(self) -> float: ...


@dataclass(frozen=True)
class TranscriptCoverage:
    """How much of the media the transcript's segments cover, in seconds."""

    media_duration_seconds: float
    covered_seconds: float
    largest_gap_start: float
    largest_gap_end: float

    @property
    def largest_gap_seconds(self) -> float:
        return self.largest_gap_end - self.largest_gap_start


def wav_duration_seconds(path: Path) -> float:
    """Duration of a PCM WAV file (frames / rate); ``RuntimeError`` if unreadable."""
    try:
        with wave.open(str(path), "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
    except (OSError, EOFError, wave.Error) as exc:
        raise RuntimeError("The normalized audio's duration could not be measured.") from exc
    if rate <= 0:
        raise RuntimeError("The normalized audio's duration could not be measured.")
    return frames / rate


def measure_transcript_coverage(
    segments: Iterable[_Timed], *, media_duration_seconds: float
) -> TranscriptCoverage:
    """Coverage from the sorted union of segment intervals, clamped to the media.

    Segments with non-finite times are ignored. A zero-length segment covers no
    time but still splits a gap, since it shows text at that point. The largest
    gap includes the stretch before the first and after the last segment; an
    empty transcript has one gap over the whole media.
    """
    duration = media_duration_seconds if math.isfinite(media_duration_seconds) else 0.0
    duration = max(duration, 0.0)
    intervals: list[tuple[float, float]] = []
    for segment in segments:
        start, end = float(segment.start), float(segment.end)
        if not math.isfinite(start) or not math.isfinite(end):
            continue
        start = min(max(start, 0.0), duration)
        end = min(max(end, start), duration)
        intervals.append((start, end))
    intervals.sort()

    covered = 0.0
    cursor = 0.0
    gap_start, gap_end = 0.0, 0.0
    for start, end in intervals:
        if start > cursor and start - cursor > gap_end - gap_start:
            gap_start, gap_end = cursor, start
        if end > cursor:
            covered += end - max(start, cursor)
            cursor = end
    if duration > cursor and duration - cursor > gap_end - gap_start:
        gap_start, gap_end = cursor, duration
    return TranscriptCoverage(
        media_duration_seconds=duration,
        covered_seconds=covered,
        largest_gap_start=gap_start,
        largest_gap_end=gap_end,
    )


def check_transcript_coverage(
    segments: Iterable[_Timed],
    *,
    media_duration_seconds: float,
    max_gap_seconds: float,
    backend_name: str,
) -> TranscriptCoverage:
    """Measure coverage; raise ``IncompleteTranscriptError`` on a gap above the threshold.

    ``max_gap_seconds`` 0 disables the rejection (coverage is still measured).
    A gap exactly at the threshold passes.
    """
    coverage = measure_transcript_coverage(segments, media_duration_seconds=media_duration_seconds)
    if max_gap_seconds > 0 and coverage.largest_gap_seconds > max_gap_seconds:
        raise IncompleteTranscriptError(
            f"{backend_name} transcript is incomplete: no text from "
            f"{coverage.largest_gap_start:.0f} s to {coverage.largest_gap_end:.0f} s "
            f"of {coverage.media_duration_seconds:.0f} s "
            f"(covered {coverage.covered_seconds:.0f} s; "
            f"largest allowed gap {max_gap_seconds:g} s)."
        )
    return coverage


def coverage_metadata(coverage: TranscriptCoverage | None) -> dict[str, float]:
    """``media_duration_seconds`` and ``covered_seconds`` for job result metadata."""
    if coverage is None:
        return {}
    return {
        "media_duration_seconds": round(coverage.media_duration_seconds, 3),
        "covered_seconds": round(coverage.covered_seconds, 3),
    }
