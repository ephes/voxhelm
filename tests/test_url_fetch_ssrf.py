"""SSRF guards for URL media fetches: redirects, resolved IPs and DNS pinning."""

from __future__ import annotations

import socket
import tempfile
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.request import Request

import pytest

from jobs import media
from jobs.media import (
    MAX_URL_REDIRECTS,
    _AllowlistRedirectHandler,
    download_allowed_media,
    validate_resolved_address,
)
from transcriptions.errors import ApiError
from transcriptions.input_media import download_allowed_url_to_tempfile

AUDIO_BYTES = b"ID3fake-mp3-payload"


class _Handler(BaseHTTPRequestHandler):
    requests_seen: list[tuple[str, str]] = []

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def do_GET(self) -> None:  # noqa: N802
        type(self).requests_seen.append((self.path, self.headers.get("Host", "")))
        port = self.server.server_address[1]  # type: ignore[index]
        if self.path == "/file.mp3":
            self.send_response(200)
            self.send_header("Content-Type", "audio/mpeg")
            self.send_header("Content-Length", str(len(AUDIO_BYTES)))
            self.end_headers()
            self.wfile.write(AUDIO_BYTES)
            return
        targets = {
            "/same-host": f"http://127.0.0.1:{port}/file.mp3",
            "/relative": "/file.mp3",
            "/metadata": "http://169.254.169.254/latest/meta-data/",
            "/foreign": "https://evil.example.net/file.mp3",
            "/downgrade": "http://media.example.com/file.mp3",
            "/other-loopback-name": f"http://localhost:{port}/file.mp3",
            "/ftp": "ftp://127.0.0.1/file.mp3",
        }
        if self.path.startswith("/loop/"):
            hop = int(self.path.rsplit("/", 1)[1])
            location = f"/loop/{hop + 1}"
        elif self.path.startswith("/chain/"):
            remaining = int(self.path.rsplit("/", 1)[1])
            location = "/file.mp3" if remaining <= 1 else f"/chain/{remaining - 1}"
        else:
            location = targets.get(self.path, "")
        if not location:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()


@pytest.fixture
def media_server() -> Iterator[str]:
    _Handler.requests_seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def isolated_tempdir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return tmp_path


@pytest.fixture
def local_allowlist(settings: Any) -> None:
    # The local test server is a trusted plain-HTTP host on loopback.
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"127.0.0.1", "media.example.com"}
    settings.VOXHELM_TRUSTED_HTTP_HOSTS = {"127.0.0.1"}
    settings.VOXHELM_PRIVATE_URL_HOSTS = set()
    settings.VOXHELM_BATCH_MAX_DOWNLOAD_BYTES = 1024
    settings.VOXHELM_MAX_URL_DOWNLOAD_BYTES = 1024


def _fetch_media(url: str) -> Path:
    return download_allowed_media(source_url=url).path


def _fetch_input(url: str) -> Path:
    return download_allowed_url_to_tempfile(source_url=url)


ENTRY_POINTS = [
    pytest.param(_fetch_media, id="jobs.media.download_allowed_media"),
    pytest.param(_fetch_input, id="transcriptions.download_allowed_url_to_tempfile"),
]


@pytest.mark.parametrize("fetch", ENTRY_POINTS)
@pytest.mark.parametrize("path", ["/same-host", "/relative", "/chain/3"])
def test_redirect_within_allowlist_is_followed(
    fetch, path, media_server, local_allowlist, isolated_tempdir
):
    result = fetch(f"{media_server}{path}")
    try:
        assert result.read_bytes() == AUDIO_BYTES
        assert result.suffix == ".mp3"
    finally:
        result.unlink(missing_ok=True)


@pytest.mark.parametrize("fetch", ENTRY_POINTS)
@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("/metadata", "allowlist"),
        ("/foreign", "allowlist"),
        ("/other-loopback-name", "allowlist"),
        ("/downgrade", "Plain HTTP"),
    ],
)
def test_redirect_outside_allowlist_is_rejected(
    fetch, path, message, media_server, local_allowlist, isolated_tempdir
):
    with pytest.raises(ApiError) as excinfo:
        fetch(f"{media_server}{path}")
    assert "redirect rejected" in excinfo.value.message
    assert message in excinfo.value.message
    assert [seen for seen, _host in _Handler.requests_seen] == [path]
    assert list(isolated_tempdir.iterdir()) == []


@pytest.mark.parametrize("fetch", ENTRY_POINTS)
def test_redirect_to_non_http_scheme_is_rejected(
    fetch, media_server, local_allowlist, isolated_tempdir
):
    with pytest.raises(ApiError):
        fetch(f"{media_server}/ftp")
    assert list(isolated_tempdir.iterdir()) == []


@pytest.mark.parametrize("fetch", ENTRY_POINTS)
def test_redirect_hops_are_capped(fetch, media_server, local_allowlist, isolated_tempdir):
    with pytest.raises(ApiError) as excinfo:
        fetch(f"{media_server}/loop/0")
    assert f"exceeded {MAX_URL_REDIRECTS} redirects" in excinfo.value.message
    assert len(_Handler.requests_seen) == MAX_URL_REDIRECTS + 1
    assert list(isolated_tempdir.iterdir()) == []


def test_redirect_chain_at_cap_succeeds(media_server, local_allowlist, isolated_tempdir):
    result = _fetch_media(f"{media_server}/chain/{MAX_URL_REDIRECTS}")
    try:
        assert result.read_bytes() == AUDIO_BYTES
    finally:
        result.unlink(missing_ok=True)


@pytest.mark.parametrize("fetch", ENTRY_POINTS)
def test_size_limit_after_redirect_leaves_no_temp_file(
    fetch, media_server, local_allowlist, isolated_tempdir, settings
):
    settings.VOXHELM_BATCH_MAX_DOWNLOAD_BYTES = 4
    settings.VOXHELM_MAX_URL_DOWNLOAD_BYTES = 4
    with pytest.raises(ApiError) as excinfo:
        fetch(f"{media_server}/same-host")
    assert "download limit" in excinfo.value.message
    assert list(isolated_tempdir.iterdir()) == []


def test_https_to_http_redirect_is_rejected_by_handler(settings):
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}
    settings.VOXHELM_TRUSTED_HTTP_HOSTS = set()

    class _Fp:
        closed = False

        def close(self) -> None:
            self.closed = True

    fp = _Fp()
    with pytest.raises(ApiError) as excinfo:
        _AllowlistRedirectHandler().redirect_request(
            Request("https://media.example.com/a.mp3"),
            fp,
            302,
            "Found",
            {},
            "http://media.example.com/a.mp3",
        )
    assert "Plain HTTP" in excinfo.value.message
    assert fp.closed


@pytest.mark.parametrize(
    "address",
    [
        "169.254.169.254",
        "fd00:ec2::254",
        "100.100.100.200",
        "192.0.0.192",
        "fe80::1",
        "::ffff:169.254.169.254",
        "0.0.0.0",
        "0.1.2.3",
        "::",
        "224.0.0.1",
        "240.0.0.1",
    ],
)
def test_blocked_addresses_rejected_even_for_private_hosts(settings, address):
    settings.VOXHELM_PRIVATE_URL_HOSTS = {"media.example.com"}
    settings.VOXHELM_TRUSTED_HTTP_HOSTS = {"media.example.com"}
    with pytest.raises(ApiError, match="blocked network address"):
        validate_resolved_address("media.example.com", address)


@pytest.mark.parametrize(
    "address",
    ["10.0.0.5", "192.168.1.10", "172.16.0.1", "127.0.0.1", "::1", "100.93.155.101", "fd00::1"],
)
def test_non_public_addresses_need_private_host_opt_in(settings, address):
    settings.VOXHELM_PRIVATE_URL_HOSTS = set()
    settings.VOXHELM_TRUSTED_HTTP_HOSTS = set()
    with pytest.raises(ApiError, match="non-public network address"):
        validate_resolved_address("media.example.com", address)

    settings.VOXHELM_PRIVATE_URL_HOSTS = {"Media.Example.com"}
    validate_resolved_address("media.example.com", address)

    settings.VOXHELM_PRIVATE_URL_HOSTS = set()
    settings.VOXHELM_TRUSTED_HTTP_HOSTS = {"media.example.com"}
    validate_resolved_address("media.example.com", address)


def test_public_addresses_are_accepted(settings):
    settings.VOXHELM_PRIVATE_URL_HOSTS = set()
    settings.VOXHELM_TRUSTED_HTTP_HOSTS = set()
    validate_resolved_address("media.example.com", "65.8.102.6")
    validate_resolved_address("media.example.com", "2600:9000:208a:5c00:3:44a7:6140:21")


def _stub_dns(monkeypatch: pytest.MonkeyPatch, answers: dict[str, list[str]]) -> list[str]:
    lookups: list[str] = []

    def fake_resolve(host: str, port: int):
        lookups.append(host)
        result = []
        for address in answers[host]:
            family = socket.AF_INET6 if ":" in address else socket.AF_INET
            sockaddr = (address, port, 0, 0) if family == socket.AF_INET6 else (address, port)
            result.append((family, socket.SOCK_STREAM, socket.IPPROTO_TCP, sockaddr))
        return result

    monkeypatch.setattr(media, "resolve_host_addresses", fake_resolve)
    return lookups


@pytest.mark.parametrize("fetch", ENTRY_POINTS)
def test_allowlisted_host_resolving_to_metadata_is_rejected(
    fetch, monkeypatch, settings, isolated_tempdir
):
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}
    settings.VOXHELM_TRUSTED_HTTP_HOSTS = set()
    lookups = _stub_dns(monkeypatch, {"media.example.com": ["169.254.169.254"]})
    with pytest.raises(ApiError, match="blocked network address"):
        fetch("https://media.example.com/file.mp3")
    assert lookups == ["media.example.com"]
    assert list(isolated_tempdir.iterdir()) == []


@pytest.mark.parametrize("fetch", ENTRY_POINTS)
def test_allowlisted_public_host_resolving_privately_is_rejected(
    fetch, monkeypatch, settings, isolated_tempdir
):
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}
    settings.VOXHELM_TRUSTED_HTTP_HOSTS = set()
    settings.VOXHELM_PRIVATE_URL_HOSTS = set()
    _stub_dns(monkeypatch, {"media.example.com": ["10.0.0.5"]})
    with pytest.raises(ApiError, match="non-public network address"):
        fetch("https://media.example.com/file.mp3")


def test_mixed_dns_answer_with_blocked_address_is_rejected(monkeypatch, settings):
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}
    settings.VOXHELM_TRUSTED_HTTP_HOSTS = set()
    _stub_dns(monkeypatch, {"media.example.com": ["65.8.102.6", "169.254.169.254"]})
    with pytest.raises(ApiError, match="blocked network address"):
        download_allowed_media(source_url="https://media.example.com/file.mp3")


def test_connection_is_pinned_to_the_validated_address(
    media_server, monkeypatch, settings, isolated_tempdir
):
    # The trusted internal host resolves (once) to the loopback test server; the
    # socket connects to exactly that validated address and keeps the Host header.
    port = int(media_server.rsplit(":", 1)[1])
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"internal.example.lan"}
    settings.VOXHELM_TRUSTED_HTTP_HOSTS = {"internal.example.lan"}
    lookups = _stub_dns(monkeypatch, {"internal.example.lan": ["127.0.0.1"]})
    connected: list[Any] = []
    original_connect = socket.socket.connect

    def recording_connect(self, address):
        connected.append(address)
        return original_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", recording_connect)
    result = download_allowed_media(source_url=f"http://internal.example.lan:{port}/file.mp3")
    try:
        assert result.path.read_bytes() == AUDIO_BYTES
    finally:
        result.path.unlink(missing_ok=True)
    assert lookups == ["internal.example.lan"]
    assert connected == [("127.0.0.1", port)]
    assert _Handler.requests_seen == [("/file.mp3", f"internal.example.lan:{port}")]


def test_environment_proxy_is_ignored(media_server, local_allowlist, monkeypatch, isolated_tempdir):
    monkeypatch.setenv("http_proxy", "http://203.0.113.1:9")
    monkeypatch.setenv("HTTP_PROXY", "http://203.0.113.1:9")
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    result = _fetch_media(f"{media_server}/file.mp3")
    try:
        assert result.read_bytes() == AUDIO_BYTES
    finally:
        result.unlink(missing_ok=True)
