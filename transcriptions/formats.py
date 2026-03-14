from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from transcriptions.service import TranscriptionResult


class SegmentLike(Protocol):
    id: int
    start: float
    end: float
    text: str


def render_verbose_json(result: TranscriptionResult) -> dict[str, Any]:
    return {
        "task": "transcribe",
        "language": result.language,
        "text": result.text,
        "segments": [
            {
                "id": segment.id,
                "seek": int(segment.start * 100),
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
            }
            for segment in result.segments
        ],
    }


def render_text(result: TranscriptionResult) -> str:
    return result.text


def render_vtt(result: TranscriptionResult) -> str:
    lines = ["WEBVTT", ""]
    for segment in result.segments:
        timestamp_line = (
            f"{format_vtt_timestamp(segment.start)} --> {format_vtt_timestamp(segment.end)}"
        )
        lines.append(timestamp_line)
        lines.append(segment.text)
        lines.append("")
    if len(lines) == 2 and result.text:
        lines.extend(["00:00:00.000 --> 00:00:00.000", result.text, ""])
    return "\n".join(lines).rstrip() + "\n"


def render_dote(result: TranscriptionResult) -> dict[str, Any]:
    return {
        "lines": [
            {
                "startTime": format_dote_timestamp(segment.start),
                "endTime": format_dote_timestamp(segment.end),
                "speakerDesignation": "",
                "text": segment.text,
            }
            for segment in normalized_segments(result)
        ]
    }


def render_podlove(result: TranscriptionResult) -> dict[str, Any]:
    transcripts = []
    for segment in normalized_segments(result):
        start_ms = max(int(round(segment.start * 1000)), 0)
        end_ms = max(int(round(segment.end * 1000)), 0)
        transcripts.append(
            {
                "start": format_podlove_timestamp(segment.start),
                "start_ms": start_ms,
                "end": format_podlove_timestamp(segment.end),
                "end_ms": end_ms,
                "speaker": "",
                "voice": "",
                "text": segment.text,
            }
        )
    return {"version": 1, "transcripts": transcripts}


def normalized_segments(result: TranscriptionResult) -> list[SegmentLike]:
    if result.segments:
        return [
            SimpleSegment(
                id=segment.id,
                start=segment.start,
                end=max(segment.start, segment.end),
                text=segment.text,
            )
            for segment in result.segments
            if segment.text.strip()
        ]
    if result.text.strip():
        return [SimpleSegment(id=0, start=0.0, end=0.0, text=result.text.strip())]
    return []


class SimpleSegment:
    def __init__(self, *, id: int, start: float, end: float, text: str) -> None:
        self.id = id
        self.start = start
        self.end = end
        self.text = text


def format_vtt_timestamp(seconds: float) -> str:
    milliseconds = max(int(round(seconds * 1000)), 0)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"


def format_dote_timestamp(seconds: float) -> str:
    milliseconds = max(int(round(seconds * 1000)), 0)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{whole_seconds:02},{millis:03}"


def format_podlove_timestamp(seconds: float) -> str:
    milliseconds = max(int(round(seconds * 1000)), 0)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{whole_seconds:02}.{millis:03}"
