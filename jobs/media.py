from __future__ import annotations

import http.client
import ipaddress
import logging
import mimetypes
import queue
import socket
import ssl
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Iterable
from contextvars import ContextVar
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

LOGGER = logging.getLogger(__name__)

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
    """Copy uploaded chunks into a reserved temp file and return its path.

    The caller owns the returned file and removes it when done. A failed copy
    (broken upload stream, disk write error) removes the partial file here, so
    an interrupted upload leaves no local media behind.
    """
    temp_path = reserve_temp_media_path(suffix=suffix)
    try:
        with temp_path.open("wb") as file_handle:
            for chunk in chunks:
                file_handle.write(chunk)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return temp_path


MAX_URL_REDIRECTS: Final[int] = 5
_FETCH_CHUNK_BYTES: Final[int] = 1024 * 1024
# "This network" (0.0.0.0/8) can reach the local host on some platforms; never
# connect there, even for hosts allowed to resolve to private addresses.
_THIS_NETWORK: Final = ipaddress.ip_network("0.0.0.0/8")
# Cloud instance-metadata endpoints outside link-local space (AWS IPv6, Alibaba,
# Oracle). Refused even for hosts allowed to resolve to private addresses.
_METADATA_ADDRESSES: Final = frozenset(
    ipaddress.ip_address(address) for address in ("fd00:ec2::254", "100.100.100.200", "192.0.0.192")
)
# Upstream statuses worth another try: request timeout, rate limiting, server errors.
_TRANSIENT_HTTP_STATUSES: Final = frozenset({408, 425, 429, 500, 502, 503, 504})
_BATCH_FETCH_RETRY_DELAY_SECONDS: Final[float] = 5.0
_BATCH_FETCH_MAX_RETRY_DELAY_SECONDS: Final[float] = 30.0
_ELAPSED_BUDGET_MESSAGE: Final[str] = "URL fetch exceeded the configured elapsed budget."


class TransientFetchError(ApiError):
    """A URL fetch failure that another try may avoid (budget, network, 5xx).

    Allowlist, SSRF, redirect-limit, media-type and size-limit rejections stay plain
    ApiError: they fail the same way on every try.
    """


class _FetchDeadline:
    """One watchdog for all sockets in a fetch, including redirect responses."""

    def __init__(self, seconds: float) -> None:
        self.ends_at = time.monotonic() + seconds
        self.sockets: list[socket.socket] = []
        self.lock = threading.Lock()
        self.timer = threading.Timer(seconds, self.interrupt)
        self.timer.daemon = True

    def remaining(self) -> float:
        remaining = self.ends_at - time.monotonic()
        if remaining <= 0:
            raise TransientFetchError(_ELAPSED_BUDGET_MESSAGE)
        return remaining

    def register(self, sock: socket.socket) -> None:
        with self.lock:
            self.remaining()
            sock.settimeout(self.remaining())
            # Keep a descriptor for the same connection even when TLS wrapping
            # detaches the original socket. shutdown() interrupts both descriptors.
            self.sockets.append(sock.dup())

    def interrupt(self) -> None:
        with self.lock:
            for sock in self.sockets:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass


_FETCH_DEADLINE: ContextVar[_FetchDeadline | None] = ContextVar("fetch_deadline", default=None)


def _resolve_with_deadline(host: str, port: int) -> list[tuple[int, int, int, tuple[Any, ...]]]:
    deadline = _FETCH_DEADLINE.get()
    if deadline is None:
        return resolve_host_addresses(host, port)
    # getaddrinfo has no timeout API. A daemon resolver may finish later, but it
    # never opens sockets or owns media files after the caller's deadline.
    result: queue.Queue[Any] = queue.Queue(maxsize=1)

    def resolve() -> None:
        try:
            result.put(resolve_host_addresses(host, port))
        except Exception as exc:
            result.put(exc)

    threading.Thread(target=resolve, daemon=True).start()
    try:
        resolved = result.get(timeout=deadline.remaining())
    except queue.Empty as exc:
        raise TransientFetchError(_ELAPSED_BUDGET_MESSAGE) from exc
    deadline.remaining()
    if isinstance(resolved, Exception):
        raise resolved
    return resolved


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


def resolve_host_addresses(host: str, port: int) -> list[tuple[int, int, int, tuple[Any, ...]]]:
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
        resolved = _resolve_with_deadline(host, port)
    except socket.gaierror as exc:
        raise TransientFetchError(f"URL fetch failed: could not resolve host ({exc}).") from exc
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
            deadline = _FETCH_DEADLINE.get()
            if deadline is not None:
                deadline.register(sock)
            sock.connect(sockaddr)
            if deadline is not None:
                deadline.remaining()
            return sock
        except OSError as exc:
            sock.close()
            last_error = exc
        except BaseException:
            sock.close()
            raise
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
        deadline = _FETCH_DEADLINE.get()
        if deadline is not None:
            try:
                req.timeout = deadline.remaining()
            except ApiError:
                fp.close()
                raise
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
    timeout_seconds: float | None = None,
) -> FetchedUrl:
    """Download an allowlisted URL into a temp file; the shared SSRF-guarded fetcher.

    timeout_seconds is the elapsed budget for the whole fetch and defaults to
    VOXHELM_URL_FETCH_TIMEOUT_SECONDS. Failures another try may fix raise
    TransientFetchError; this function never retries by itself.
    """
    validate_allowed_media_url(source_url)
    request = Request(source_url, headers={"User-Agent": "voxhelm/0.1", "Accept": accept})
    temp_path: Path | None = None
    if timeout_seconds is None:
        timeout_seconds = settings.VOXHELM_URL_FETCH_TIMEOUT_SECONDS
    deadline = _FetchDeadline(timeout_seconds)
    marker = _FETCH_DEADLINE.set(deadline)
    deadline.timer.start()
    try:
        with open_allowed_url(request, timeout=deadline.remaining()) as response:
            deadline.remaining()
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
                    deadline.remaining()
                    # read1 performs at most one underlying read, so slow progress
                    # cannot hide inside a read(size) loop until the chunk fills.
                    read = getattr(response, "read1", response.read)
                    chunk = read(_FETCH_CHUNK_BYTES)
                    deadline.remaining()
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise ApiError(limit_message)
                    handle.write(chunk)
            # http.client's read1 reports a connection closed before Content-Length
            # as a normal end of body; a truncated file must not pass as the media.
            expected = response.headers.get("Content-Length")
            if expected is not None and expected.isdigit() and total < int(expected):
                raise TransientFetchError("URL fetch ended before the announced content length.")
            return FetchedUrl(path=temp_path, content_type=content_type, final_url=final_url)
    except HTTPError as exc:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        error_class = TransientFetchError if exc.code in _TRANSIENT_HTTP_STATUSES else ApiError
        raise error_class(f"URL fetch failed with HTTP {exc.code}.") from exc
    except URLError as exc:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        if isinstance(exc.reason, ApiError):
            raise exc.reason from exc
        # A certificate the host presents now fails the same way on every try.
        error_class = (
            ApiError
            if isinstance(exc.reason, ssl.SSLCertVerificationError)
            else TransientFetchError
        )
        raise error_class(f"URL fetch failed: {exc.reason}.") from exc
    except (OSError, http.client.HTTPException) as exc:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise TransientFetchError(f"URL fetch failed: {exc}.") from exc
    except ApiError:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
    except BaseException:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
    finally:
        deadline.timer.cancel()
        with deadline.lock:
            for sock in deadline.sockets:
                sock.close()
        _FETCH_DEADLINE.reset(marker)


def _retry_sleep(seconds: float) -> None:
    """Back off between batch fetch tries; split out so tests can skip the wait."""
    time.sleep(seconds)


def download_allowed_media(*, source_url: str) -> DownloadedMedia:
    """Fetch batch media (job input or speaker reference) for the server or a remote worker.

    Each try gets VOXHELM_BATCH_URL_FETCH_TIMEOUT_SECONDS. A TransientFetchError is
    retried up to VOXHELM_BATCH_URL_FETCH_ATTEMPTS tries in total, so a slow or
    dropped download does not use up a job attempt; any other error fails at once.
    """
    attempts = settings.VOXHELM_BATCH_URL_FETCH_ATTEMPTS
    attempt = 1
    while True:
        try:
            fetched = fetch_allowed_url_to_tempfile(
                source_url=source_url,
                accept="audio/*,video/*;q=1.0,*/*;q=0.1",
                max_bytes=settings.VOXHELM_BATCH_MAX_DOWNLOAD_BYTES,
                detect_suffix=detect_media_suffix,
                unsupported_message="Unsupported remote media type for batch transcription.",
                limit_message="Remote media exceeded the configured batch download limit.",
                timeout_seconds=settings.VOXHELM_BATCH_URL_FETCH_TIMEOUT_SECONDS,
            )
            break
        except TransientFetchError as exc:
            if attempt >= attempts:
                raise
            delay = min(
                _BATCH_FETCH_RETRY_DELAY_SECONDS * 2 ** (attempt - 1),
                _BATCH_FETCH_MAX_RETRY_DELAY_SECONDS,
            )
            LOGGER.warning(
                "batch media fetch failed, retrying try=%s/%s delay_seconds=%s host=%s detail=%s",
                attempt,
                attempts,
                delay,
                urlparse(source_url).hostname or "",
                exc.message,
            )
            _retry_sleep(delay)
            attempt += 1
    return DownloadedMedia(
        path=fetched.path,
        content_type=fetched.content_type,
        source_url=fetched.final_url,
        source_name=Path(urlparse(fetched.final_url).path or "input").name or "input",
        source_kind="url",
    )


def extract_audio_from_video(*, source_path: Path) -> Path:
    target_path = reserve_temp_media_path(suffix=".wav")
    completed = False
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
            timeout=settings.VOXHELM_VIDEO_EXTRACTION_TIMEOUT_SECONDS,
        )
        completed = True
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip() or (exc.stdout or "").strip() or "unknown ffmpeg error"
        raise RuntimeError(f"ffmpeg audio extraction failed: {stderr[:2000]}") from exc
    except subprocess.TimeoutExpired as exc:
        # run() kills and waits for the child before raising TimeoutExpired.
        raise RuntimeError(
            "ffmpeg audio extraction exceeded the configured processing budget."
        ) from exc
    except OSError as exc:
        raise RuntimeError("ffmpeg audio extraction could not start or complete.") from exc
    finally:
        if not completed:
            target_path.unlink(missing_ok=True)
    return target_path
