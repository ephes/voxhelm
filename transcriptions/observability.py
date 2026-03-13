from __future__ import annotations

import json
import logging
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Any

from django.conf import settings

if TYPE_CHECKING:
    from .service import TranscriptionResult

_LOGGER = logging.getLogger("voxhelm.stt")


def summarize_audio_file(audio_path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "suffix": audio_path.suffix.lower(),
    }
    try:
        summary["bytes"] = audio_path.stat().st_size
    except OSError as exc:
        summary["stat_error"] = str(exc)
        return summary

    if audio_path.suffix.lower() != ".wav":
        return summary

    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            width = wav_file.getsampwidth()
            channels = wav_file.getnchannels()
    except (OSError, wave.Error) as exc:
        summary["wav_error"] = str(exc)
        return summary

    duration_seconds = 0.0
    if rate > 0:
        duration_seconds = round(frames / rate, 3)

    summary.update(
        {
            "rate": rate,
            "width": width,
            "channels": channels,
            "frames": frames,
            "duration_seconds": duration_seconds,
        }
    )
    return summary


def emit_transcription_debug_log(
    *,
    source: str,
    audio_shape: dict[str, Any],
    request_model: str,
    request_language: str | None,
    prompt: str | None,
    result: TranscriptionResult,
    duration_ms: int | None,
    raw_transcript: str | None = None,
) -> None:
    if not settings.VOXHELM_STT_DEBUG_LOGGING:
        return

    payload = {
        "audio_shape": audio_shape,
        "backend": result.backend_name,
        "duration_ms": duration_ms,
        "language": {
            "requested": request_language,
            "resolved": result.language,
        },
        "model": {
            "requested": request_model,
            "resolved": result.model_name,
        },
        "prompt": _truncate(prompt, limit=160),
        "source": source,
        "transcript": _truncate(result.text, limit=300),
    }
    if (
        raw_transcript is not None
        and raw_transcript.strip()
        and raw_transcript.strip() != result.text
    ):
        payload["transcript_raw"] = _truncate(raw_transcript, limit=300)
    _LOGGER.warning("stt_debug %s", json.dumps(payload, sort_keys=True))


def _truncate(value: str | None, *, limit: int) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit].rstrip()}..."
