from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from jobs import remote_worker_cli as worker_cli


@contextmanager
def control_server(*, status=200, body=b'{"job": {"id": "job-1"}}', drip=None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):  # noqa: N802
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            try:
                self.send_response(status)
                if drip == "headers":
                    # Each header arrives within the socket timeout.
                    self.wfile.flush()
                    for _ in range(30):
                        self.send_header("X-Progress", "yes")
                        self.flush_headers()
                        self.wfile.flush()
                        time.sleep(0.02)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if drip == "body":
                    for byte in body:
                        self.wfile.write(bytes([byte]))
                        self.wfile.flush()
                        time.sleep(0.02)
                else:
                    self.wfile.write(body)
            except BrokenPipeError, ConnectionResetError:
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def client(base_url, timeout=2):
    return worker_cli.WorkerClient(
        config=worker_cli.WorkerConfig(
            base_url=base_url,
            worker_id="synthetic-worker",
            token="synthetic-worker-credential",
            hostname="test-host",
            http_timeout_seconds=timeout,
        )
    )


@pytest.mark.parametrize("status", [200, 500])
def test_oversized_control_body_is_bounded(status, monkeypatch):
    monkeypatch.setattr(worker_cli, "MAX_CONTROL_RESPONSE_BYTES", 64, raising=False)
    body = b'{"job": {}, "private": "' + b"synthetic-secret" * 100 + b'"}'
    with control_server(status=status, body=body) as url:
        if status >= 400:
            with pytest.raises(worker_cli.WorkerHttpError) as http_error:
                client(url).claim(capabilities={})
            assert http_error.value.status == status
            detail = str(http_error.value)
        else:
            with pytest.raises(worker_cli.WorkerError, match="byte limit") as error:
                client(url).claim(capabilities={})
            detail = str(error.value)
    assert "synthetic-secret" not in detail


@pytest.mark.parametrize("status", [200, 500])
@pytest.mark.parametrize("drip", ["body", "headers"])
def test_dripping_control_response_has_elapsed_deadline(status, drip):
    with control_server(status=status, drip=drip) as url:
        started = time.monotonic()
        if status >= 400 and drip == "body":
            with pytest.raises(worker_cli.WorkerHttpError) as http_error:
                client(url, timeout=2).claim(capabilities={})
            assert http_error.value.status == status
            detail = str(http_error.value)
        else:
            with pytest.raises(worker_cli.WorkerError, match="elapsed budget") as error:
                client(url, timeout=0.15).claim(capabilities={})
            detail = str(error.value)
        elapsed = time.monotonic() - started
    assert elapsed < 0.9
    assert "synthetic-worker-credential" not in detail


def test_http_error_detail_does_not_echo_remote_body():
    with control_server(
        status=403, body=b"synthetic-worker-credential\nhttps://private.invalid"
    ) as url:
        with pytest.raises(worker_cli.WorkerHttpError) as error:
            client(url).claim(capabilities={})
    assert error.value.status == 403
    assert "synthetic-worker-credential" not in str(error.value)
    assert "private.invalid" not in str(error.value)


@pytest.mark.parametrize("status,body,expected", [(204, b"", None), (200, b'{"job": {}}', {})])
def test_small_claims_still_work(status, body, expected):
    with control_server(status=status, body=body) as url:
        assert client(url).claim(capabilities={}) == expected


@pytest.mark.parametrize("status", [200, 500])
@pytest.mark.parametrize("failure", ["oversize", "drip", "invalid", "ok"])
def test_streams_close_and_failed_claims_never_start_inference(
    status, failure, monkeypatch, caplog
):
    import asyncio
    import logging
    from unittest.mock import Mock

    import httpx

    closed = []
    consumed: list[bytes] = []
    streams: list[list[bytes]] = []

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for part in streams[-1]:
                if failure == "drip":
                    await asyncio.sleep(0.1)
                consumed.append(part)
                yield part

        async def aclose(self):
            closed.append(True)

    def respond(request):
        if request.url.path.endswith("workers/heartbeat"):
            streams.append([b"{}"])
            return httpx.Response(200, stream=Stream())
        if failure == "oversize":
            streams.append([b'{"job": {}, "secret": "', b"synthetic-secret" * 10, b'"}'])
        elif failure == "invalid":
            streams.append([b"\xff"])
        elif failure == "drip":
            streams.append([b"{}"] * 30)
        else:
            streams.append([b'{"job": {"id": "job-1", "lease_token": "lease"}}'])
        # No length header: the cap also applies to streamed/chunked responses.
        return httpx.Response(status, stream=Stream())

    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        worker_cli.httpx,
        "AsyncClient",
        lambda **kwargs: original_client(transport=httpx.MockTransport(respond), **kwargs),
    )
    monkeypatch.setattr(worker_cli, "MAX_CONTROL_RESPONSE_BYTES", 64)
    process = Mock()
    monkeypatch.setattr(worker_cli, "process_claim", process)
    worker = client("http://synthetic.invalid", timeout=0.3)
    caplog.set_level(logging.ERROR)
    if failure == "ok" and status == 200:
        assert worker_cli.run_worker_once(config=worker.config, client=worker, capabilities={})
        process.assert_called_once()
    else:
        with pytest.raises((worker_cli.WorkerError, worker_cli.WorkerHttpError)) as error:
            worker_cli.run_worker_once(config=worker.config, client=worker, capabilities={})
        process.assert_not_called()
        worker_cli.LOGGER.error("worker poll failed: %s", error.value)
        assert "synthetic-secret" not in caplog.text
        assert "synthetic-worker-credential" not in caplog.text
    assert len(streams) == 2
    assert closed == [True, True]
    if status >= 400:
        assert consumed == [b"{}"]  # heartbeat only; no error-body read


def test_slow_dns_does_not_extend_request_deadline(monkeypatch):
    import socket

    started = threading.Event()
    finished = threading.Event()

    def slow_lookup(*args, **kwargs):
        started.set()
        try:
            time.sleep(1.2)
            raise socket.gaierror("synthetic lookup failed")
        finally:
            finished.set()

    monkeypatch.setattr(socket, "getaddrinfo", slow_lookup)
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    before = time.monotonic()
    try:
        with pytest.raises(worker_cli.WorkerError, match="elapsed budget"):
            client("http://synthetic.invalid", timeout=0.15).claim(capabilities={})
        elapsed = time.monotonic() - before
        assert started.is_set()
        assert elapsed < 0.9
    finally:
        assert finished.wait(2)


@pytest.mark.parametrize("status", [401, 403, 404, 409])
def test_error_status_preserves_nonretryable_decision(status):
    with control_server(status=status, body=b"secret" * 10000, drip="body") as url:
        with pytest.raises(worker_cli.WorkerHttpError) as error:
            client(url).claim(capabilities={})
    assert error.value.status == status
    assert not worker_cli.retryable_worker_error(error.value)


def test_keyboard_interrupt_cancels_request_and_closes_stream(monkeypatch):
    import asyncio

    import httpx

    closed = []
    started = []

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            started.append(True)
            await asyncio.sleep(10)
            yield b"{}"

        async def aclose(self):
            closed.append(True)

    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        worker_cli.httpx,
        "AsyncClient",
        lambda **kwargs: original_client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=Stream())),
            **kwargs,
        ),
    )
    new_loop = asyncio.new_event_loop

    def interrupt():
        raise KeyboardInterrupt

    def interrupted_loop():
        loop = new_loop()
        loop.call_later(0.1, interrupt)
        return loop

    monkeypatch.setattr(worker_cli.asyncio, "new_event_loop", interrupted_loop)
    with pytest.raises(KeyboardInterrupt):
        client("http://synthetic.invalid").claim(capabilities={})
    assert started == [True]
    assert closed == [True]


def test_compressed_response_is_not_decompressed(monkeypatch):
    import gzip

    import httpx

    compressed = gzip.compress(b'{"job": {}, "padding": "' + b"x" * (2 * 1024 * 1024) + b'"}')
    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        worker_cli.httpx,
        "AsyncClient",
        lambda **kwargs: original_client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, headers={"Content-Encoding": "gzip"}, stream=httpx.ByteStream(compressed)
                )
            ),
            **kwargs,
        ),
    )
    with pytest.raises(worker_cli.WorkerError, match="invalid JSON"):
        client("http://synthetic.invalid").claim(capabilities={})


def test_worker_uses_stdlib_tls_context_and_safe_transport_error(monkeypatch):
    import ssl

    import httpx

    context = ssl.create_default_context()
    monkeypatch.setattr(worker_cli.ssl, "create_default_context", lambda: context)
    original_client = httpx.AsyncClient
    used = []

    def fail(request):
        raise httpx.ConnectError("synthetic-worker-credential https://private.invalid")

    def make_client(**kwargs):
        used.append(kwargs["verify"])
        return original_client(transport=httpx.MockTransport(fail), **kwargs)

    monkeypatch.setattr(worker_cli.httpx, "AsyncClient", make_client)
    with pytest.raises(worker_cli.WorkerError, match="ConnectError") as error:
        client("http://synthetic.invalid").claim(capabilities={})
    assert used == [context]
    assert "synthetic-worker-credential" not in str(error.value)
    assert "private.invalid" not in str(error.value)


def test_poll_exception_log_does_not_echo_error_body(monkeypatch, caplog):
    import logging

    import httpx

    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        worker_cli.httpx,
        "AsyncClient",
        lambda **kwargs: original_client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    403,
                    stream=httpx.ByteStream(b"synthetic-worker-credential https://private.invalid"),
                )
            ),
            **kwargs,
        ),
    )
    monkeypatch.setattr(worker_cli, "setup_django_for_worker", lambda: None)
    monkeypatch.setattr(
        worker_cli, "parse_worker_config", lambda args: client("http://synthetic.invalid").config
    )
    monkeypatch.setattr(worker_cli, "log_startup", lambda config: None)
    monkeypatch.setattr(worker_cli, "build_capabilities", lambda: {})
    caplog.set_level(logging.ERROR)
    assert worker_cli.main(["--once"]) == 1
    assert "worker poll failed" in caplog.text
    assert "HTTP 403: Forbidden" in caplog.text
    assert "synthetic-worker-credential" not in caplog.text
    assert "private.invalid" not in caplog.text
