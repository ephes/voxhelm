from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from django.conf import settings


@dataclass(frozen=True)
class TranscribeParams:
    request_model: str
    prompt: str | None
    language: str | None


@dataclass(frozen=True)
class TranscriptionSegment:
    id: int
    start: float
    end: float
    text: str

    def as_verbose_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "seek": int(self.start * 100),
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str | None
    segments: list[TranscriptionSegment]


class BackendProtocol(Protocol):
    def transcribe(self, audio_path: Path, params: TranscribeParams) -> TranscriptionResult: ...


class MlxWhisperBackend:
    def __init__(self, *, model_name: str) -> None:
        self.model_name = model_name

    def transcribe(self, audio_path: Path, params: TranscribeParams) -> TranscriptionResult:
        try:
            import mlx_whisper
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError(
                "mlx-whisper is not installed. Install the project dependencies first."
            ) from exc

        payload = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=self.model_name,
            word_timestamps=False,
            initial_prompt=params.prompt,
            language=params.language,
        )
        return normalize_transcription_payload(payload)


_TRANSCRIPTION_LOCK = Lock()


def normalize_transcription_payload(payload: dict[str, Any]) -> TranscriptionResult:
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

    return TranscriptionResult(text=text, language=language, segments=segments)


def render_verbose_json(result: TranscriptionResult) -> dict[str, Any]:
    return {
        "task": "transcribe",
        "language": result.language,
        "text": result.text,
        "segments": [segment.as_verbose_json() for segment in result.segments],
    }


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


def format_vtt_timestamp(seconds: float) -> str:
    milliseconds = max(int(round(seconds * 1000)), 0)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"


@lru_cache(maxsize=1)
def get_backend_service() -> BackendProtocol:
    return MlxWhisperBackend(model_name=settings.VOXHELM_MLX_MODEL)


def transcribe_audio(audio_path: Path, params: TranscribeParams) -> TranscriptionResult:
    # MLX inference is not safe to run concurrently inside this long-lived process.
    with _TRANSCRIPTION_LOCK:
        return get_backend_service().transcribe(audio_path, params)


def serialize_health() -> str:
    return json.dumps({"status": "ok"})
