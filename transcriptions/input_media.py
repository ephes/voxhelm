from __future__ import annotations

import mimetypes
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Final

from django.conf import settings

from jobs.media import fetch_allowed_url_to_tempfile

SUPPORTED_SUFFIXES: Final[dict[str, str]] = {
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".mpeg": "audio/mpeg",
    ".mpga": "audio/mpeg",
    ".oga": "audio/ogg",
    ".ogg": "audio/ogg",
    ".wav": "audio/wav",
}
CONTENT_TYPE_SUFFIXES: Final[dict[str, str]] = {
    "audio/flac": ".flac",
    "audio/mp4": ".m4a",
    "audio/mpeg": ".mp3",
    "audio/mpga": ".mpga",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
}


def write_upload_to_tempfile(chunks: Iterable[bytes], *, suffix: str) -> Path:
    file_handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        for chunk in chunks:
            file_handle.write(chunk)
    finally:
        file_handle.close()
    return Path(file_handle.name)


def download_allowed_url_to_tempfile(*, source_url: str) -> Path:
    # Shared SSRF-guarded fetcher: allowlist, scheme and resolved-IP checks on
    # every redirect hop (see jobs.media.fetch_allowed_url_to_tempfile).
    return fetch_allowed_url_to_tempfile(
        source_url=source_url,
        accept="audio/*;q=1.0,*/*;q=0.1",
        max_bytes=settings.VOXHELM_MAX_URL_DOWNLOAD_BYTES,
        detect_suffix=detect_suffix,
        unsupported_message="Unsupported remote media type for transcription.",
        limit_message="Remote media exceeded the configured download limit.",
    ).path


def detect_suffix(filename_or_url: str, content_type: str) -> str:
    lower_name = filename_or_url.lower()
    for suffix in SUPPORTED_SUFFIXES:
        if lower_name.endswith(suffix):
            return suffix
    if content_type in CONTENT_TYPE_SUFFIXES:
        return CONTENT_TYPE_SUFFIXES[content_type]
    guessed = mimetypes.guess_extension(content_type, strict=False) or ""
    return guessed if guessed in SUPPORTED_SUFFIXES else ""
