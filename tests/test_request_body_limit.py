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
from typing import Any

import pytest

MIB = 1024 * 1024


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

    scope = {
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

    async def main() -> None:
        await asyncio.wait_for(application(scope, receive, send), timeout=60)

    asyncio.run(main())
    response_headers = {key.lower(): value for key, value in start.get("headers", [])}
    return int(start.get("status", 0)), response_headers, bytes(body), pulled


def test_unauthenticated_unbounded_stream_is_cut_off_before_django_spools_it():
    status, headers, body, pulled = run_asgi(method="POST", path="/v1/health", chunks=64)

    assert status == 413
    assert pulled <= 4
    assert headers[b"connection"] == b"close"
    assert headers[b"x-frame-options"] == b"DENY"
    assert headers[b"x-content-type-options"] == b"nosniff"
    assert json.loads(body)["error"]["type"] == "invalid_request_error"


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
