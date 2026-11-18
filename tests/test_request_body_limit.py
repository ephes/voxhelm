"""The ASGI entry point bounds request bodies before Django spools them.

Django's ASGI handler copies the whole body into a temporary file before URL
routing, authentication or any view runs. These tests drive the real
``config.asgi.application`` with a fake transport and count how many body chunks
it pulls, so an unbounded spool shows up as consumed chunks, not as disk usage.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from tempfile import SpooledTemporaryFile
from types import ModuleType
from typing import Any

import pytest
from django.http import HttpResponse
from django.urls import path
from django.views.decorators.csrf import csrf_exempt

MIB = 1024 * 1024


def asgi_scope(
    method: str, path: str, headers: list[tuple[bytes, bytes]] | None = None, root_path: str = ""
) -> dict[str, Any]:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": root_path + path,
        "raw_path": (root_path + path).encode(),
        "root_path": root_path,
        "query_string": b"",
        "headers": [(b"host", b"localhost"), *(headers or [])],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8000),
    }


@pytest.fixture
def body_spools(settings, monkeypatch):
    settings.FILE_UPLOAD_MAX_MEMORY_SIZE = 1  # even a synthetic chunk rolls to disk
    opened = []

    def open_spool(*args, **kwargs):
        spool = SpooledTemporaryFile(*args, **kwargs)
        opened.append(spool)
        return spool

    monkeypatch.setattr("django.core.handlers.asgi.tempfile.SpooledTemporaryFile", open_spool)
    return opened


class BodyConnection:
    def __init__(self) -> None:
        self.events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.waiting = asyncio.Event()
        self.sent: list[Mapping[str, Any]] = []
        self.pulls = 0

    def chunk(self, *, more: bool = True) -> None:
        self.events.put_nowait({"type": "http.request", "body": b"data", "more_body": more})

    async def receive(self) -> dict[str, Any]:
        self.pulls += 1
        if self.events.empty():
            self.waiting.set()
        return await self.events.get()

    async def send(self, message: Mapping[str, Any]) -> None:
        self.sent.append(message)


def run_asgi(
    *,
    method: str,
    path: str,
    chunks: int,
    chunk_size: int = MIB,
    headers: list[tuple[bytes, bytes]] | None = None,
    root_path: str = "",
) -> tuple[int, dict[bytes, bytes], bytes, int]:
    """Send ``chunks`` body chunks; return status, headers, body and chunks pulled."""
    from config.asgi import application

    pulled = 0
    delivered_empty = False
    finished = asyncio.Event()
    start: dict[str, Any] = {}
    body = bytearray()

    async def receive() -> dict[str, Any]:
        nonlocal pulled, delivered_empty
        if pulled < chunks:
            pulled += 1
            return {
                "type": "http.request",
                "body": b"x" * chunk_size,
                "more_body": pulled < chunks,
            }
        if chunks == 0 and not delivered_empty:
            delivered_empty = True  # later calls wait for the disconnect
            return {"type": "http.request", "body": b"", "more_body": False}
        await finished.wait()
        return {"type": "http.disconnect"}

    async def send(message: Mapping[str, Any]) -> None:
        if message["type"] == "http.response.start":
            start.update(message)
        elif message["type"] == "http.response.body":
            body.extend(message.get("body", b""))
            if not message.get("more_body", False):
                finished.set()

    scope = asgi_scope(method, path, headers, root_path)

    async def main() -> None:
        await asyncio.wait_for(application(scope, receive, send), timeout=60)

    asyncio.run(main())
    response_headers = {key.lower(): value for key, value in start.get("headers", [])}
    return int(start.get("status", 0)), response_headers, bytes(body), pulled


def test_unauthenticated_unbounded_stream_is_cut_off_before_django_spools_it(body_spools):
    status, headers, body, pulled = run_asgi(method="POST", path="/v1/health", chunks=64)

    assert status == 413
    assert pulled <= 4
    assert headers[b"connection"] == b"close"
    assert headers[b"x-frame-options"] == b"DENY"
    assert headers[b"x-content-type-options"] == b"nosniff"
    assert json.loads(body)["error"]["type"] == "invalid_request_error"
    assert len(body_spools) == 1 and body_spools[0].closed


def test_non_ascii_digit_content_length_falls_back_to_the_streamed_count():
    status, _headers, _body, pulled = run_asgi(
        method="POST",
        path="/v1/health",
        chunks=16,
        headers=[(b"content-length", "\u00b2".encode("latin-1"))],
    )

    assert status == 413
    assert pulled <= 4


def test_declared_oversized_length_is_refused_without_reading_the_body():
    status, headers, _body, pulled = run_asgi(
        method="POST",
        path="/v1/uploads",
        chunks=64,
        headers=[
            (b"content-length", str(600 * MIB).encode()),
            (b"authorization", b"Bearer test-token"),
        ],
    )

    assert status == 413
    assert pulled == 0
    assert headers[b"connection"] == b"close"


def test_invalid_bearer_token_does_not_unlock_the_staged_upload_ceiling():
    status, _headers, _body, pulled = run_asgi(
        method="POST",
        path="/v1/uploads",
        chunks=16,
        headers=[
            (b"authorization", b"Bearer wrong-token"),
            (b"content-type", b"multipart/form-data; boundary=x"),
        ],
    )

    assert status == 413
    assert pulled <= 4


def test_valid_bearer_token_admits_an_upload_above_the_default_ceiling():
    # A 5 MiB body exceeds the 3 MiB default; with a producer token the sync
    # transcription route admits it and the view answers (unsupported type).
    status, _headers, body, pulled = run_asgi(
        method="POST",
        path="/v1/audio/transcriptions",
        chunks=5,
        headers=[
            (b"authorization", b"Bearer test-token"),
            (b"content-type", b"application/octet-stream"),
        ],
    )

    assert pulled == 5
    assert status == 400
    assert "Unsupported content type" in json.loads(body)["error"]["message"]


def test_small_unauthenticated_body_still_reaches_the_view():
    status, _headers, _body, pulled = run_asgi(
        method="POST",
        path="/v1/jobs",
        chunks=1,
        chunk_size=1024,
        headers=[(b"content-type", b"application/json")],
    )

    assert pulled == 1
    assert status == 401


def test_get_without_body_passes_through():
    status, _headers, body, _pulled = run_asgi(method="GET", path="/v1/health", chunks=0)

    assert status == 200
    assert json.loads(body) == {"status": "ok"}


@pytest.mark.parametrize("drip", [False, True], ids=["stalled", "drip-fed"])
def test_body_deadline_closes_django_spool_before_408(settings, monkeypatch, body_spools, drip):
    from config.asgi import application

    settings.VOXHELM_ASGI_BODY_TIMEOUT_SECONDS = 10

    async def scenario():
        loop = asyncio.get_running_loop()
        real_time = loop.time
        elapsed = 0
        monkeypatch.setattr(loop, "time", lambda: real_time() + elapsed)
        connection = BodyConnection()
        connection.chunk()
        if drip:
            connection.chunk()
        received = 0

        async def receive():
            nonlocal received, elapsed
            received += 1
            if drip:
                elapsed += 4
                if received < 3:
                    return await connection.receive()
                return {"type": "http.request", "body": b"last", "more_body": False}
            return await connection.receive()

        async def send(message):
            assert len(body_spools) == 1 and body_spools[0].closed
            await connection.send(message)

        scope = asgi_scope("POST", "/v1/health", root_path="/voxhelm")
        task = asyncio.create_task(application(scope, receive, send))
        if not drip:
            await asyncio.wait_for(connection.waiting.wait(), timeout=1)
            assert len(body_spools) == 1 and not body_spools[0].closed
            elapsed = 11
        await asyncio.wait_for(task, timeout=20 if drip else 1)
        start, body = connection.sent
        assert start["status"] == 408
        headers = dict(start["headers"])
        assert headers[b"connection"] == b"close"
        assert headers[b"cache-control"] == b"no-store"
        assert json.loads(body["body"])["error"] == {
            "message": "Request body receipt timed out.",
            "type": "invalid_request_error",
        }

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel", [False, True], ids=["disconnect", "cancel"])
def test_abandoned_partial_body_closes_django_spool(body_spools, cancel):
    from config.asgi import application

    async def scenario():
        connection = BodyConnection()
        connection.chunk()
        task = asyncio.create_task(
            application(asgi_scope("POST", "/"), connection.receive, connection.send)
        )
        await asyncio.wait_for(connection.waiting.wait(), timeout=1)
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            connection.events.put_nowait({"type": "http.disconnect"})
            await asyncio.wait_for(task, timeout=1)
        assert connection.sent == []
        assert len(body_spools) == 1 and body_spools[0].closed

    asyncio.run(scenario())


def test_completed_upload_can_run_past_receipt_deadline(settings, monkeypatch, body_spools):
    from config.asgi import application

    settings.VOXHELM_ASGI_BODY_TIMEOUT_SECONDS = 10

    async def scenario():
        loop = asyncio.get_running_loop()
        real_time = loop.time
        elapsed = 0
        monkeypatch.setattr(loop, "time", lambda: real_time() + elapsed)
        connection = BodyConnection()

        @csrf_exempt
        async def view(request):
            nonlocal elapsed
            elapsed = 11
            await asyncio.sleep(0)
            assert request.read() == b"data"
            return HttpResponse(b"done")

        urls = ModuleType("body_deadline_urls")
        urls.urlpatterns = [path("v1/audio/transcriptions", view)]  # type: ignore[attr-defined]
        settings.ROOT_URLCONF = urls
        connection.chunk(more=False)
        scope = asgi_scope("POST", "/v1/audio/transcriptions")
        await asyncio.wait_for(application(scope, connection.receive, connection.send), timeout=60)
        assert connection.sent[0]["status"] == 200
        assert connection.sent[1]["body"] == b"done"
        assert len(body_spools) == 1 and body_spools[0].closed

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "finish", ["complete", "timeout", "disconnect", "cancel", "parse-error", "cancel-after-body"]
)
def test_spool_admission_rejects_excess_and_reuses_one_slot(
    settings, monkeypatch, body_spools, finish
):
    from config.asgi import django_application
    from config.body_limit import RequestBodyLimitMiddleware

    settings.VOXHELM_ASGI_MAX_ACTIVE_UPLOADS = 2
    settings.VOXHELM_ASGI_BODY_TIMEOUT_SECONDS = 10
    application = RequestBodyLimitMiddleware(django_application)

    async def scenario():
        loop = asyncio.get_running_loop()
        real_time = loop.time
        elapsed = 0
        monkeypatch.setattr(loop, "time", lambda: real_time() + elapsed)
        view_started = asyncio.Event()
        cleanup_started = asyncio.Event()
        allow_cleanup = asyncio.Event()

        @csrf_exempt
        async def view(request):
            if finish == "parse-error":
                return HttpResponse(str(len(request.POST)))
            if finish == "complete":
                return HttpResponse()
            view_started.set()
            try:
                await asyncio.Future()
            finally:
                cleanup_started.set()
                await allow_cleanup.wait()
            return HttpResponse()

        if finish in {"complete", "parse-error", "cancel-after-body"}:
            urls = ModuleType("body_admission_urls")
            urls.urlpatterns = [path("", view)]  # type: ignore[attr-defined]
            settings.ROOT_URLCONF = urls

        chunked = [(b"transfer-encoding", b"chunked")]
        # Both holders share the operator pool; the producer pool is separate.
        other_operator = asgi_scope("POST", "/", chunked)
        headers = [(b"content-type", b"multipart/form-data")] if finish == "parse-error" else []
        operator = asgi_scope("POST", "/", [*chunked, *headers])

        def start(scope):
            connection = BodyConnection()
            connection.chunk()
            task = asyncio.create_task(application(scope, connection.receive, connection.send))
            return task, connection

        async def assert_busy(scope):
            excess = BodyConnection()
            excess.chunk(more=False)
            spools_before = len(body_spools)
            await asyncio.wait_for(application(scope, excess.receive, excess.send), timeout=2)
            assert excess.sent[0]["status"] == 503
            assert dict(excess.sent[0]["headers"])[b"connection"] == b"close"
            assert json.loads(excess.sent[1]["body"])["error"] == {
                "message": "Server is busy.",
                "type": "server_error",
            }
            assert excess.pulls == 0
            assert len(body_spools) == spools_before

        task, connection = start(operator)
        await asyncio.wait_for(connection.waiting.wait(), timeout=1)
        try:
            if finish == "timeout":
                elapsed = 5  # The second holder starts with its own later deadline.
            other, held = start(other_operator)
            await asyncio.wait_for(held.waiting.wait(), timeout=1)
            assert len(body_spools) == 2
            assert all(not spool.closed for spool in body_spools)
            await assert_busy(operator)

            if finish == "cancel-after-body":
                connection.chunk(more=False)
                await asyncio.wait_for(view_started.wait(), timeout=1)
                task.cancel()
                await asyncio.wait_for(cleanup_started.wait(), timeout=1)
                assert not body_spools[0].closed
                await assert_busy(other_operator)
                task.cancel()
                await assert_busy(other_operator)
                allow_cleanup.set()
            elif finish == "cancel":
                task.cancel()
            elif finish == "disconnect":
                connection.events.put_nowait({"type": "http.disconnect"})
            elif finish == "timeout":
                elapsed = 11
            else:
                # A normal form response or Django's malformed multipart response.
                connection.chunk(more=False)

            if finish.startswith("cancel"):
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                await asyncio.wait_for(task, timeout=2)
            if finish == "parse-error":
                assert connection.sent[0]["status"] == 400
            elif finish == "timeout":
                assert connection.sent[0]["status"] == 408
            elif finish == "complete":
                assert connection.sent[0]["status"] == 200
            else:
                assert connection.sent == []
            assert body_spools[0].closed
            assert not body_spools[1].closed

            replacement, fresh = start(other_operator)
            await asyncio.wait_for(fresh.waiting.wait(), timeout=1)
            assert len(body_spools) == 3 and not body_spools[2].closed
            await assert_busy(operator)
            for remaining in (fresh, held):
                remaining.events.put_nowait({"type": "http.disconnect"})
            await asyncio.wait_for(asyncio.gather(replacement, other), timeout=2)
            assert all(spool.closed for spool in body_spools)
        finally:
            allow_cleanup.set()

    asyncio.run(scenario())


def test_admission_counts_only_large_bodies_in_separate_pools(settings, body_spools):
    from config.asgi import django_application
    from config.body_limit import RequestBodyLimitMiddleware

    settings.VOXHELM_ASGI_MAX_ACTIVE_UPLOADS = 1
    application = RequestBodyLimitMiddleware(django_application)
    token = (b"authorization", b"Bearer test-token")
    length = (b"content-length", b"1000")

    async def answer(scope, body=b""):
        connection = BodyConnection()
        connection.events.put_nowait({"type": "http.request", "body": body, "more_body": False})
        task = asyncio.create_task(application(scope, connection.receive, connection.send))
        await asyncio.wait_for(task, timeout=5)
        return connection.sent[0]["status"]

    async def hold(scope, holders):
        connection = BodyConnection()
        connection.chunk()
        task = asyncio.create_task(application(scope, connection.receive, connection.send))
        await asyncio.wait_for(connection.waiting.wait(), timeout=1)
        holders.append((task, connection))

    async def scenario():
        holders: list[tuple[asyncio.Task[None], BodyConnection]] = []
        operator = asgi_scope("POST", "/", [length])  # anonymous operator form post
        staged = asgi_scope("POST", "/v1/uploads", [token, length])
        sync = asgi_scope("POST", "/v1/audio/transcriptions", [token, length])
        try:
            await hold(operator, holders)
            await hold(staged, holders)
            # Full operator and staged pools leave synchronous transcription alone.
            assert await answer(sync) != 503
            await hold(sync, holders)

            # Every pool is full: further large-body requests are refused.
            for scope in (operator, staged, sync):
                assert await answer(scope) == 503

            # Bodyless requests and JSON-ceiling routes are never counted.
            assert await answer(asgi_scope("GET", "/v1/health")) == 200
            assert await answer(asgi_scope("GET", "/")) != 503
            assert await answer(asgi_scope("POST", "/", [(b"content-length", b"0")])) != 503
            json_post = [(b"content-type", b"application/json"), length]
            assert await answer(asgi_scope("POST", "/v1/jobs", json_post), b"{}") == 401
            assert await answer(asgi_scope("POST", "/v1/uploads", [length]), b"x") == 401
        finally:
            for _task, connection in holders:
                connection.events.put_nowait({"type": "http.disconnect"})
            await asyncio.wait_for(asyncio.gather(*(task for task, _ in holders)), timeout=2)
        assert all(spool.closed for spool in body_spools)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("path", "authorization", "expected"),
    [
        ("/v1/uploads", b"Bearer test-token", "staged"),
        ("/v1/uploads", b"", "default"),
        ("/v1/uploads", b"Bearer wrong-token", "default"),
        ("/v1/audio/transcriptions", b"Bearer test-token", "sync"),
        ("/v1/audio/transcriptions", b"", "default"),
        ("/", b"", "sync"),
        ("/v1/jobs", b"Bearer test-token", "default"),
        ("/v1/internal/work/claim", b"", "default"),
        ("/v1/uploads/", b"Bearer test-token", "default"),
    ],
)
def test_route_ceilings(settings, path, authorization, expected):
    from config.body_limit import (
        ADMISSION_POOLS,
        DEFAULT_BODY_LIMIT_BYTES,
        UPLOAD_FRAMING_HEADROOM_BYTES,
        request_body_limit,
    )

    settings.VOXHELM_MAX_UPLOAD_BYTES = 25 * MIB
    settings.VOXHELM_BATCH_MAX_STAGED_UPLOAD_BYTES = 512 * MIB
    headers = [(b"authorization", authorization)] if authorization else []
    limits = {
        "staged": 512 * MIB + UPLOAD_FRAMING_HEADROOM_BYTES,
        "sync": 25 * MIB + UPLOAD_FRAMING_HEADROOM_BYTES,
        "default": DEFAULT_BODY_LIMIT_BYTES,
    }

    assert (
        request_body_limit({"path": path, "root_path": "", "headers": headers})
        == (limits[expected])
    )
    # Every route that admits a large body has its own admission pool.
    if expected != "default":
        assert path in ADMISSION_POOLS


def test_route_ceiling_strips_the_root_path(settings):
    from config.body_limit import UPLOAD_FRAMING_HEADROOM_BYTES, request_body_limit

    settings.VOXHELM_BATCH_MAX_STAGED_UPLOAD_BYTES = 512 * MIB
    scope = {
        "path": "/voxhelm/v1/uploads",
        "root_path": "/voxhelm",
        "headers": [(b"authorization", b"Bearer test-token")],
    }

    assert request_body_limit(scope) == 512 * MIB + UPLOAD_FRAMING_HEADROOM_BYTES


def test_sync_ceiling_never_drops_below_the_json_limit(settings):
    from config.body_limit import DEFAULT_BODY_LIMIT_BYTES, request_body_limit

    settings.VOXHELM_MAX_UPLOAD_BYTES = 1024
    scope = {
        "path": "/v1/audio/transcriptions",
        "root_path": "",
        "headers": [(b"authorization", b"Bearer test-token")],
    }

    assert request_body_limit(scope) >= DEFAULT_BODY_LIMIT_BYTES
