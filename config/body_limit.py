"""Bound HTTP request bodies before Django reads them under ASGI.

Django's ASGI handler copies the whole request body into a temporary file before
URL routing, CSRF, authentication or any view runs, and it has no size limit for
that copy. Without this wrapper any client that reaches the port could stream an
unbounded body to disk without credentials.

The ceiling depends on the route. Only the two bearer upload routes admit large
bodies, and only when the request carries a valid producer token; the operator
form at ``/`` admits the synchronous upload limit. Every other request, and an
upload request without a valid token, gets the JSON ceiling plus headroom so the
views still answer their own precise 413/401 for small bodies. A declared
``Content-Length`` above the ceiling is refused before any body byte is read; a
streamed body is cut off at the first chunk past it. The refusal is a JSON 413
with ``Connection: close``, so the server stops reading the rest.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Final

from django.conf import settings

# The shapes Django's ASGIHandler.__call__ declares.
Scope = dict[str, Any]
Message = Mapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

MIB: Final[int] = 1024 * 1024
# JSON endpoints read at most 2 MiB; one extra MiB keeps their own 413 messages.
DEFAULT_BODY_LIMIT_BYTES: Final[int] = 3 * MIB
# Multipart boundaries, part headers and the small form fields beside the file.
UPLOAD_FRAMING_HEADROOM_BYTES: Final[int] = 1 * MIB

STAGED_UPLOAD_PATH: Final[str] = "/v1/uploads"
SYNC_TRANSCRIPTION_PATH: Final[str] = "/v1/audio/transcriptions"
OPERATOR_FORM_PATH: Final[str] = "/"


def _route_path(scope: Scope) -> str:
    path = str(scope.get("path") or "")
    root_path = str(scope.get("root_path") or "")
    if root_path and path.startswith(root_path):
        path = path[len(root_path) :]
    return path or "/"


def _header(scope: Scope, name: bytes) -> str:
    for key, value in scope.get("headers") or ():
        if key.lower() == name:
            return bytes(value).decode("latin-1")
    return ""


def _has_producer_token(scope: Scope) -> bool:
    from transcriptions.views import producer_for_authorization

    return producer_for_authorization(_header(scope, b"authorization")) is not None


def request_body_limit(scope: Scope) -> int:
    """Return the largest request body this route admits, in bytes."""
    path = _route_path(scope)
    sync_limit = max(
        settings.VOXHELM_MAX_UPLOAD_BYTES + UPLOAD_FRAMING_HEADROOM_BYTES,
        DEFAULT_BODY_LIMIT_BYTES,
    )
    if path == OPERATOR_FORM_PATH:
        return sync_limit
    if path == SYNC_TRANSCRIPTION_PATH and _has_producer_token(scope):
        return sync_limit
    if path == STAGED_UPLOAD_PATH and _has_producer_token(scope):
        return settings.VOXHELM_BATCH_MAX_STAGED_UPLOAD_BYTES + UPLOAD_FRAMING_HEADROOM_BYTES
    return DEFAULT_BODY_LIMIT_BYTES


def _declared_length_exceeds(scope: Scope, limit: int) -> bool:
    for key, value in scope.get("headers") or ():
        if key.lower() != b"content-length":
            continue
        text = bytes(value).decode("latin-1").strip()
        # isdigit() also accepts non-ASCII digits that int() refuses.
        if text.isascii() and text.isdigit() and int(text) > limit:
            return True
    return False


async def _send_too_large(send: Send, limit: int) -> None:
    body = json.dumps(
        {
            "error": {
                "message": (
                    f"Request body exceeds the {limit // MIB} MiB limit for this endpoint."
                ),
                "type": "invalid_request_error",
            }
        }
    ).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"connection", b"close"),
                (b"cache-control", b"no-store"),
                # Bypasses Django's middleware; keep its framing and sniffing policy.
                (b"x-frame-options", b"DENY"),
                (b"x-content-type-options", b"nosniff"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})


class RequestBodyLimitMiddleware:
    """ASGI wrapper that refuses request bodies above the route's ceiling."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        limit = request_body_limit(scope)
        if _declared_length_exceeds(scope, limit):
            await _send_too_large(send, limit)
            return

        received = 0
        exceeded = False
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received, exceeded
            if exceeded:
                return {"type": "http.disconnect"}
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body") or b"")
                if received > limit:
                    # Django treats a disconnect while reading the body as an
                    # aborted request and returns without a response.
                    exceeded = True
                    return {"type": "http.disconnect"}
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        await self.app(scope, limited_receive, tracking_send)
        if exceeded and not response_started:
            await _send_too_large(send, limit)
