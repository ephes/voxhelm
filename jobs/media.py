from __future__ import annotations

import mimetypes
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from django.conf import settings

from transcriptions.errors import ApiError

AUDIO_SUFFIXES: Final[dict[str, str]] = {
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".mpeg": "audio/mpeg",
    ".mpga": "audio/mpeg",
    ".oga": "audio/ogg",
    ".ogg": "audio/ogg",
    ".wav": "audio/wav",
}
VIDEO_SUFFIXES: Final[dict[str, str]] = {
    ".avi": "video/x-msvideo",
    ".m4v": "video/x-m4v",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
}
CONTENT_TYPE_SUFFIXES: Final[dict[str, str]] = {
    "audio/flac": ".flac",
    "audio/mp4": ".m4a",
    "audio/mpeg": ".mp3",
    "audio/mpga": ".mpga",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
    "video/x-m4v": ".m4v",
    "video/x-matroska": ".mkv",
    "video/x-msvideo": ".avi",
}
SUPPORTED_SUFFIXES: Final[dict[str, str]] = {
    **AUDIO_SUFFIXES,
    **VIDEO_SUFFIXES,
}


@dataclass(frozen=True)
class DownloadedMedia:
    path: Path
    content_type: str
    source_url: str


def detect_media_suffix(filename_or_url: str, content_type: str) -> str:
    lower_name = filename_or_url.lower()
    for suffix in SUPPORTED_SUFFIXES:
        if lower_name.endswith(suffix):
            return suffix
    if content_type in CONTENT_TYPE_SUFFIXES:
        return CONTENT_TYPE_SUFFIXES[content_type]
    guessed = mimetypes.guess_extension(content_type, strict=False) or ""
    return guessed if guessed in SUPPORTED_SUFFIXES else ""


def is_video_path(path: Path, *, content_type: str | None = None) -> bool:
    if path.suffix.lower() in VIDEO_SUFFIXES:
        return True
    return bool(content_type and content_type.startswith("video/"))


def download_allowed_media(*, source_url: str) -> DownloadedMedia:
    parsed = urlparse(source_url)
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise ApiError("URL input must include a hostname.")
    if hostname not in settings.VOXHELM_ALLOWED_URL_HOSTS:
        raise ApiError("URL host is not in the configured allowlist.")
    if parsed.scheme == "https":
        pass
    elif parsed.scheme == "http":
        if hostname not in settings.VOXHELM_TRUSTED_HTTP_HOSTS:
            raise ApiError("Plain HTTP URLs are only allowed for trusted internal hosts.")
    else:
        raise ApiError("Only https URLs are allowed by default.")

    request = Request(
        source_url,
        headers={"User-Agent": "voxhelm/0.1", "Accept": "audio/*,video/*;q=1.0,*/*;q=0.1"},
    )
    temp_path: Path | None = None
    try:
        with urlopen(request, timeout=settings.VOXHELM_URL_FETCH_TIMEOUT_SECONDS) as response:
            content_type = (response.headers.get_content_type() or "").lower()
            final_url = response.geturl() or source_url
            suffix = detect_media_suffix(final_url, content_type)
            if not suffix:
                raise ApiError("Unsupported remote media type for batch transcription.")
            temp_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=suffix).name)
            total = 0
            with temp_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > settings.VOXHELM_BATCH_MAX_DOWNLOAD_BYTES:
                        raise ApiError("Remote media exceeded the configured batch download limit.")
                    handle.write(chunk)
            return DownloadedMedia(
                path=temp_path,
                content_type=content_type,
                source_url=final_url,
            )
    except HTTPError as exc:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise ApiError(f"URL fetch failed with HTTP {exc.code}.") from exc
    except URLError as exc:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise ApiError(f"URL fetch failed: {exc.reason}.") from exc
    except OSError as exc:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise ApiError(f"URL fetch failed: {exc}.") from exc
    except ApiError:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
    raise AssertionError("Unexpected URL download flow.")


def extract_audio_from_video(*, source_path: Path) -> Path:
    target_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name)
    try:
        subprocess.run(
            [
                settings.VOXHELM_FFMPEG_BIN,
                "-y",
                "-i",
                str(source_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                str(target_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        target_path.unlink(missing_ok=True)
        stderr = exc.stderr.strip() or exc.stdout.strip() or "unknown ffmpeg error"
        raise RuntimeError(f"ffmpeg audio extraction failed: {stderr}") from exc
    return target_path
