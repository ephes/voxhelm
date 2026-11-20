from __future__ import annotations

import math
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Protocol

from transcriptions.errors import TranscriptionError

if TYPE_CHECKING:
    from transcriptions.service import TranscriptionResult


class SegmentLike(Protocol):
    id: int
    start_ms: int
    end_ms: int
    text: str
    speaker: str | None


def validate_timestamps(result: TranscriptionResult) -> None:
    """Reject invalid backend cues, including empty cues, before rendering."""
    for segment in result.segments:
        if any(
            not math.isfinite(value) or not math.isfinite(value * 1000)
            for value in (segment.start, segment.end)
        ):
            raise TranscriptionError(
                "Transcript cue timestamps must be finite and representable in milliseconds."
            )


def render_verbose_json(result: TranscriptionResult) -> dict[str, Any]:
    validate_timestamps(result)
    segments: list[dict[str, Any]] = []
    for segment in result.segments:
        payload = {
            "id": segment.id,
            "seek": int(segment.start * 100),
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
        }
        if segment.speaker:
            payload["speaker"] = segment.speaker
        segments.append(payload)
    return {
        "task": "transcribe",
        "language": result.language,
        "text": result.text,
        "segments": segments,
    }


def render_text(result: TranscriptionResult) -> str:
    validate_timestamps(result)
    return result.text


def render_vtt(result: TranscriptionResult) -> str:
    lines = ["WEBVTT", ""]
    for segment in _iter_normalized_segments(result):
        cue_text = escape_vtt_cue_text(segment.text)
        if not cue_text:
            continue
        timestamp_line = (
            f"{format_milliseconds(segment.start_ms)} --> {format_milliseconds(segment.end_ms)}"
        )
        lines.append(timestamp_line)
        lines.append(cue_text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def escape_vtt_cue_text(text: str) -> str:
    """Return ``text`` as a single, safe WebVTT cue payload line.

    WebVTT treats ``&`` as an entity start and ``<`` as a tag start, a payload
    line must not contain ``-->``, and a blank line ends the cue. Line breaks are
    collapsed to one space, ``-->`` becomes ``->``, and ``&``, ``<`` and ``>``
    are written as character references.
    """
    # Scan each character once. An unanchored whitespace/newline regex retries
    # long interior space runs at every position when no newline follows.
    parts: list[str] = []
    whitespace_start: int | None = None
    has_line_break = False
    for index, char in enumerate(text):
        if char.isspace():
            if whitespace_start is None:
                whitespace_start = index
            has_line_break = has_line_break or char in "\r\n"
        else:
            if whitespace_start is not None:
                parts.append(" " if has_line_break else text[whitespace_start:index])
                whitespace_start = None
                has_line_break = False
            parts.append(char)
    # Pending trailing whitespace is trimmed; ordinary interior spacing survives.
    single_line = "".join(parts).strip()
    single_line = single_line.replace("-->", "->")
    return single_line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_dote(result: TranscriptionResult) -> dict[str, Any]:
    return {
        "lines": [
            {
                "startTime": format_milliseconds(segment.start_ms, separator=","),
                "endTime": format_milliseconds(segment.end_ms, separator=","),
                "speakerDesignation": segment.speaker or "",
                "text": segment.text,
            }
            for segment in _iter_normalized_segments(result)
        ]
    }


def render_podlove(result: TranscriptionResult) -> dict[str, Any]:
    transcripts = []
    for segment in _iter_normalized_segments(result):
        transcripts.append(
            {
                "start": format_milliseconds(segment.start_ms),
                "start_ms": segment.start_ms,
                "end": format_milliseconds(segment.end_ms),
                "end_ms": segment.end_ms,
                "speaker": segment.speaker or "",
                "voice": segment.speaker or "",
                "text": segment.text,
            }
        )
    return {"version": 1, "transcripts": transcripts}


def normalized_segments(result: TranscriptionResult) -> list[SegmentLike]:
    """Return the validated, normalized cues as a list."""
    return list(_iter_normalized_segments(result))


def _iter_normalized_segments(result: TranscriptionResult) -> Iterator[SegmentLike]:
    validate_timestamps(result)
    if result.segments:
        for segment in result.segments:
            if segment.text.strip():
                yield SimpleSegment(
                    id=segment.id,
                    start=segment.start,
                    end=max(segment.start, segment.end),
                    text=segment.text,
                    speaker=segment.speaker,
                )
    elif text := result.text.strip():
        yield SimpleSegment(id=0, start=0.0, end=0.0, text=text, speaker=None)


class SimpleSegment:
    def __init__(
        self,
        *,
        id: int,
        start: float,
        end: float,
        text: str,
        speaker: str | None,
    ) -> None:
        self.id = id
        self.start = start
        self.end = end
        # Quantize once. Integer endpoints also preserve a positive duration for
        # very large finite times, where adding 0.001 to a float would do nothing.
        self.start_ms = max(int(round(start * 1000)), 0)
        self.end_ms = max(int(round(end * 1000)), self.start_ms + 1)
        self.text = text
        self.speaker = speaker


def format_milliseconds(milliseconds: int, *, separator: str = ".") -> str:
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}{separator}{millis:03}"


def format_vtt_timestamp(seconds: float) -> str:
    return format_milliseconds(max(int(round(seconds * 1000)), 0))


def format_dote_timestamp(seconds: float) -> str:
    return format_milliseconds(max(int(round(seconds * 1000)), 0), separator=",")


def format_podlove_timestamp(seconds: float) -> str:
    return format_milliseconds(max(int(round(seconds * 1000)), 0))
