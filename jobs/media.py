from __future__ import annotations

import http.client
import ipaddress
import mimetypes
import socket
import subprocess
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import (
    HTTPHandler,
    HTTPRedirectHandler,
    HTTPSHandler,
    ProxyHandler,
    Request,
    build_opener,
)

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
    source_url: str = ""
    source_name: str = "input"
    source_kind: str = "url"


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


def reserve_temp_media_path(*, suffix: str) -> Path:
    return Path(tempfile.NamedTemporaryFile(delete=False, suffix=suffix).name)


def write_uploaded_media_to_tempfile(chunks: Iterable[bytes], *, suffix: str) -> Path:
    file_handle = reserve_temp_media_path(suffix=suffix).open("wb")
    try:
        for chunk in chunks:
            file_handle.write(chunk)
    finally:
        file_handle.close()
    return Path(file_handle.name)


MAX_URL_REDIRECTS: Final[int] = 5
_FETCH_CHUNK_BYTES: Final[int] = 1024 * 1024
# "This network" (0.0.0.0/8) can reach the local host on some platforms; never
# connect there, even for hosts allowed to resolve to private addresses.
_THIS_NETWORK: Final = ipaddress.ip_network("0.0.0.0/8")
# Cloud instance-metadata endpoints outside link-local space (AWS IPv6, Alibaba,
# Oracle). Refused even for hosts allowed to resolve to private addresses.
_METADATA_ADDRESSES: Final = frozenset(
    ipaddress.ip_address(address)
    for address in ("fd00:ec2::254", "100.100.100.200", "192.0.0.192")
)


def validate_allowed_media_url(source_url: str) -> None:
    parsed = urlparse(source_url)
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise ApiError("URL input must include a hostname.")
    if hostname not in settings.VOXHELM_ALLOWED_URL_HOSTS:
        raise ApiError("URL host is not in the configured allowlist.")
    if parsed.scheme == "https":
        return
    elif parsed.scheme == "http":
        if hostname not in settings.VOXHELM_TRUSTED_HTTP_HOSTS:
            raise ApiError("Plain HTTP URLs are only allowed for trusted internal hosts.")
        return
    else:
        raise ApiError("Only https URLs are allowed by default.")


def private_url_hosts() -> set[str]:
    """Allowlisted hosts that may resolve to non-public (private/loopback/CGNAT) IPs."""
    configured: set[str] = getattr(settings, "VOXHELM_PRIVATE_URL_HOSTS", set())
    trusted_http = settings.VOXHELM_TRUSTED_HTTP_HOSTS
    return {host.lower() for host in (*configured, *trusted_http)}


def validate_resolved_address(hostname: str, address: str) -> None:
    """Reject a resolved IP the media fetcher must never connect to.

    Link-local (incl. cloud metadata 169.254.169.254), the other known cloud
    metadata addresses, multicast, reserved, unspecified and 0.0.0.0/8
    addresses are always rejected. Any other
    non-global address (RFC 1918, loopback, CGNAT/Tailscale, IPv6 ULA, ...) is
    only accepted for hosts listed in VOXHELM_PRIVATE_URL_HOSTS or
    VOXHELM_TRUSTED_HTTP_HOSTS.
    """
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError as exc:
        raise ApiError("URL host resolved to an invalid address.") from exc
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if (
        ip.is_link_local
        or ip in _METADATA_ADDRESSES
        or ip.is_multicast
        # ::1 sits inside the reserved ::/8 block; loopback is gated below.
        or (ip.is_reserved and not ip.is_loopback)
        or ip.is_unspecified
        or (isinstance(ip, ipaddress.IPv4Address) and ip in _THIS_NETWORK)
    ):
        raise ApiError("URL host resolved to a blocked network address.")
    if not ip.is_global and hostname.lower() not in private_url_hosts():
        raise ApiError("URL host resolved to a non-public network address.")


def resolve_host_addresses(
    host: str, port: int
) -> list[tuple[int, int, int, tuple[Any, ...]]]:
    """Resolve host for TCP; split out so tests can stub DNS."""
    return [
        (family, socktype, proto, sockaddr)
        for family, socktype, proto, _canonname, sockaddr in socket.getaddrinfo(
            host, port, type=socket.SOCK_STREAM
        )
    ]


def _guarded_create_connection(
    address: tuple[str, int],
    timeout: Any = socket._GLOBAL_DEFAULT_TIMEOUT,  # type: ignore[attr-defined]
    source_address: tuple[str, int] | None = None,
    *,
    all_errors: bool = False,
) -> socket.socket:
    """socket.create_connection that validates and pins the resolved IPs.

    Every resolved address is checked before connecting, and the socket
    connects to exactly the address that was checked, so a DNS answer that
    changes between validation and connect (rebinding) cannot slip through.
    """
    host, port = address
    try:
        resolved = resolve_host_addresses(host, port)
    except socket.gaierror as exc:
        raise ApiError(f"URL fetch failed: could not resolve host ({exc}).") from exc
    if not resolved:
        raise ApiError("URL fetch failed: host did not resolve.")
    for _family, _socktype, _proto, sockaddr in resolved:
        validate_resolved_address(host, str(sockaddr[0]))
    last_error: OSError | None = None
    for family, socktype, proto, sockaddr in resolved:
        sock = socket.socket(family, socktype, proto)
        try:
            if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:  # type: ignore[attr-defined]
                sock.settimeout(timeout)
            if source_address:
                sock.bind(source_address)
            sock.connect(sockaddr)
            return sock
        except OSError as exc:
            sock.close()
            last_error = exc
    assert last_error is not None
    raise last_error


class _GuardedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._create_connection = _guarded_create_connection


class _GuardedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._create_connection = _guarded_create_connection


class _GuardedHTTPHandler(HTTPHandler):
    def http_open(self, req: Request) -> Any:
        return self.do_open(_GuardedHTTPConnection, req)


class _GuardedHTTPSHandler(HTTPSHandler):
    def https_open(self, req: Request) -> Any:
        return self.do_open(_GuardedHTTPSConnection, req, context=self._context)  # type: ignore[attr-defined]


class _AllowlistRedirectHandler(HTTPRedirectHandler):
    """Re-validate every redirect hop against the URL allowlist and cap hops."""

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Request | None:
        hops = getattr(req, "voxhelm_redirect_hops", 0) + 1
        try:
            if hops > MAX_URL_REDIRECTS:
                raise ApiError(f"URL fetch exceeded {MAX_URL_REDIRECTS} redirects.")
            try:
                validate_allowed_media_url(newurl)
            except ApiError as exc:
                raise ApiError(f"URL redirect rejected: {exc.message}") from exc
        except ApiError:
            fp.close()
            raise
        new_request = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_request is not None:
            new_request.voxhelm_redirect_hops = hops  # type: ignore[attr-defined]
        return new_request


def open_allowed_url(request: Request, *, timeout: float) -> Any:
    """Open an already-validated URL with SSRF guards on every hop.

    Proxies from the environment are ignored (the proxy would otherwise be the
    host that gets connected to), every redirect target is re-checked against
    the allowlist, and every TCP connection is pinned to a validated address.
    """
    opener = build_opener(
        ProxyHandler({}),
        _GuardedHTTPHandler(),
        _GuardedHTTPSHandler(),
        _AllowlistRedirectHandler(),
    )
    return opener.open(request, timeout=timeout)


@dataclass(frozen=True)
class FetchedUrl:
    path: Path
    content_type: str
    final_url: str


def fetch_allowed_url_to_tempfile(
    *,
    source_url: str,
    accept: str,
    max_bytes: int,
    detect_suffix: Callable[[str, str], str],
    unsupported_message: str,
    limit_message: str,
) -> FetchedUrl:
    """Download an allowlisted URL into a temp file; the shared SSRF-guarded fetcher."""
    validate_allowed_media_url(source_url)
    request = Request(source_url, headers={"User-Agent": "voxhelm/0.1", "Accept": accept})
    temp_path: Path | None = None
    try:
        with open_allowed_url(
            request, timeout=settings.VOXHELM_URL_FETCH_TIMEOUT_SECONDS
        ) as response:
            final_url = response.geturl() or source_url
            # Redundant with the per-hop check; keeps the final URL honest.
            validate_allowed_media_url(final_url)
            content_type = (response.headers.get_content_type() or "").lower()
            suffix = detect_suffix(final_url, content_type)
            if not suffix:
                raise ApiError(unsupported_message)
            temp_path = reserve_temp_media_path(suffix=suffix)
            total = 0
            with temp_path.open("wb") as handle:
                while True:
                    chunk = response.read(_FETCH_CHUNK_BYTES)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise ApiError(limit_message)
                    handle.write(chunk)
            return FetchedUrl(path=temp_path, content_type=content_type, final_url=final_url)
    except HTTPError as exc:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise ApiError(f"URL fetch failed with HTTP {exc.code}.") from exc
    except URLError as exc:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        if isinstance(exc.reason, ApiError):
            raise exc.reason from exc
        raise ApiError(f"URL fetch failed: {exc.reason}.") from exc
    except OSError as exc:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise ApiError(f"URL fetch failed: {exc}.") from exc
    except ApiError:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def download_allowed_media(*, source_url: str) -> DownloadedMedia:
    fetched = fetch_allowed_url_to_tempfile(
        source_url=source_url,
        accept="audio/*,video/*;q=1.0,*/*;q=0.1",
        max_bytes=settings.VOXHELM_BATCH_MAX_DOWNLOAD_BYTES,
        detect_suffix=detect_media_suffix,
        unsupported_message="Unsupported remote media type for batch transcription.",
        limit_message="Remote media exceeded the configured batch download limit.",
    )
    return DownloadedMedia(
        path=fetched.path,
        content_type=fetched.content_type,
        source_url=fetched.final_url,
        source_name=Path(urlparse(fetched.final_url).path or "input").name or "input",
        source_kind="url",
    )


def extract_audio_from_video(*, source_path: Path) -> Path:
    target_path = reserve_temp_media_path(suffix=".wav")
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
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.CalledProcessError as exc:
        target_path.unlink(missing_ok=True)
        stderr = exc.stderr.strip() or exc.stdout.strip() or "unknown ffmpeg error"
        raise RuntimeError(f"ffmpeg audio extraction failed: {stderr}") from exc
    return target_path
