from __future__ import annotations

import json
import logging
import os
import wave
from pathlib import Path
from stat import S_ISREG
from typing import TYPE_CHECKING, Any

from django.conf import settings

if TYPE_CHECKING:
    from .service import TranscriptionResult

_LOGGER = logging.getLogger("voxhelm.stt")


def summarize_audio_file(audio_path: Path) -> dict[str, Any]:
    suffix = audio_path.suffix.lower()
    summary: dict[str, Any] = {"suffix": suffix}
    try:
        if suffix != ".wav":
            summary["bytes"] = audio_path.stat().st_size
            return summary

        # Opening a FIFO must not block before we can check the file type.
        with open(
            audio_path, "rb", opener=lambda path, flags: os.open(path, flags | os.O_NONBLOCK)
        ) as handle:
            metadata = os.fstat(handle.fileno())
            if not S_ISREG(metadata.st_mode):
                summary["stat_error"] = "not_regular_file"
                return summary
            summary["bytes"] = metadata.st_size
            try:
                with wave.open(handle, "rb") as wav_file:
                    frames = wav_file.getnframes()
                    rate = wav_file.getframerate()
                    width = wav_file.getsampwidth()
                    channels = wav_file.getnchannels()
            except (OSError, EOFError, wave.Error) as exc:
                summary["wav_error"] = str(exc)
                return summary
    except OSError as exc:
        summary["stat_error"] = str(exc)
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
