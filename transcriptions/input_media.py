from __future__ import annotations

import mimetypes
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Final
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from django.conf import settings

from transcriptions.errors import ApiError

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
        headers={"User-Agent": "voxhelm/0.1", "Accept": "audio/*;q=1.0,*/*;q=0.1"},
    )
    temp_path: Path | None = None
    try:
        with urlopen(request, timeout=settings.VOXHELM_URL_FETCH_TIMEOUT_SECONDS) as response:
            content_type = (response.headers.get_content_type() or "").lower()
            final_url = response.geturl() or source_url
            suffix = detect_suffix(final_url, content_type)
            if not suffix:
                raise ApiError("Unsupported remote media type for transcription.")
            temp_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=suffix).name)
            total = 0
            with temp_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > settings.VOXHELM_MAX_URL_DOWNLOAD_BYTES:
                        raise ApiError("Remote media exceeded the configured download limit.")
                    handle.write(chunk)
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
    assert temp_path is not None
    return temp_path


def detect_suffix(filename_or_url: str, content_type: str) -> str:
    lower_name = filename_or_url.lower()
    for suffix in SUPPORTED_SUFFIXES:
        if lower_name.endswith(suffix):
            return suffix
    if content_type in CONTENT_TYPE_SUFFIXES:
        return CONTENT_TYPE_SUFFIXES[content_type]
    guessed = mimetypes.guess_extension(content_type, strict=False) or ""
    return guessed if guessed in SUPPORTED_SUFFIXES else ""
