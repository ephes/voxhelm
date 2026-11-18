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
Receipt of the body is also timed, without timing the view: each chunk must
arrive within an idle timeout, and the whole body within a total budget that
starts at a fixed base and grows with the declared size at a minimum rate, so a
multi-gigabyte producer upload over a slow link still completes while a
stalled or trickling client is cut off.

Each process also bounds how many large-body requests (a route ceiling above the
JSON ceiling, with a body announced) own a Django spool at once. Each large-body
route has its own pool, so stalled anonymous operator form posts cannot block
producer uploads and synchronous inference cannot block staged batch uploads.
Bodyless requests and every route with the JSON ceiling (health, worker control
plane, job API) are never counted, so long uploads or inference cannot starve
them.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Awaitable, Callable, Mapping
from http import HTTPStatus
from threading import BoundedSemaphore
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
# Each route whose ceiling can exceed the JSON ceiling has its own admission pool.
ADMISSION_POOLS: Final[tuple[str, ...]] = (
    OPERATOR_FORM_PATH,
    SYNC_TRANSCRIPTION_PATH,
    STAGED_UPLOAD_PATH,
)


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


def _announces_body(scope: Scope) -> bool:
    """Whether the request may carry body bytes (HTTP/1.1 framing headers)."""
    lengths = []
    for key, value in scope.get("headers") or ():
        name = key.lower()
        if name == b"transfer-encoding":
            return True
        if name == b"content-length":
            lengths.append(bytes(value).decode("latin-1").strip())
    # Without framing headers an HTTP/1.1 request has an empty body. An
    # unparsable length is counted, so it can never bypass admission.
    return any(text != "0" for text in lengths)


def _admission_pool(scope: Scope, limit: int) -> str | None:
    """Return the admission pool for a large-body request, or None if uncounted."""
    if limit <= DEFAULT_BODY_LIMIT_BYTES or not _announces_body(scope):
        return None
    # Only the ADMISSION_POOLS routes can have a ceiling above the JSON ceiling.
    return _route_path(scope)


def _declared_length(scope: Scope) -> int | None:
    """Return the largest parsable ``Content-Length``, or None without one."""
    declared = None
    for key, value in scope.get("headers") or ():
        if key.lower() != b"content-length":
            continue
        text = bytes(value).decode("latin-1").strip()
        # isdigit() also accepts non-ASCII digits that int() refuses.
        if text.isascii() and text.isdigit():
            declared = max(declared or 0, int(text))
    return declared


def _declared_length_exceeds(scope: Scope, limit: int) -> bool:
    # Any oversized length is refused early, whatever else frames the body.
    declared = _declared_length(scope)
    return declared is not None and declared > limit


def _framed_length(scope: Scope) -> int | None:
    """Return the length that frames the body, or None for chunked framing.

    ``Transfer-Encoding`` takes precedence over ``Content-Length`` in HTTP/1.1
    (h11 reads such a body as chunked), so a length sent beside it frames
    nothing and must not extend the receipt budget.
    """
    if any(key.lower() == b"transfer-encoding" for key, _value in scope.get("headers") or ()):
        return None
    return _declared_length(scope)


def receipt_budget_seconds(scope: Scope, limit: int) -> float:
    """Return the total time this request's body may take to arrive.

    The budget is the configured base, or the declared body length (capped at
    the route ceiling) at the minimum upload rate if that takes longer. A body
    without a declared length keeps the base budget, and small ceilings
    (anonymous and JSON routes) never exceed it.
    """
    declared = _framed_length(scope)
    if declared is None:
        return float(settings.VOXHELM_ASGI_BODY_TIMEOUT_SECONDS)
    scaled = math.ceil(min(declared, limit) / settings.VOXHELM_ASGI_BODY_MIN_BYTES_PER_SECOND)
    return float(max(settings.VOXHELM_ASGI_BODY_TIMEOUT_SECONDS, scaled))


async def _send_error(
    send: Send, status: HTTPStatus, message: str, *, error_type: str = "invalid_request_error"
) -> None:
    body = json.dumps({"error": {"message": message, "type": error_type}}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": int(status),
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


def _staged_upload_without_length(scope: Scope) -> bool:
    """Whether an admitted staged upload announces a body without a length.

    Django parses multipart only with a ``Content-Length``; without one the
    staging view would refuse the upload only after spooling all of it.
    """
    return (
        _route_path(scope) == STAGED_UPLOAD_PATH
        and _announces_body(scope)
        and _framed_length(scope) is None
        and _has_producer_token(scope)
    )


async def _send_too_large(send: Send, limit: int) -> None:
    await _send_error(
        send,
        HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        f"Request body exceeds the {limit // MIB} MiB limit for this endpoint.",
    )


class RequestBodyLimitMiddleware:
    """Bound request-body bytes, receipt time and concurrent Django spools."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        slots = settings.VOXHELM_ASGI_MAX_ACTIVE_UPLOADS
        self._admission = {pool: BoundedSemaphore(slots) for pool in ADMISSION_POOLS}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        limit = request_body_limit(scope)
        if _declared_length_exceeds(scope, limit):
            await _send_too_large(send, limit)
            return
        if _staged_upload_without_length(scope):
            await _send_error(
                send,
                HTTPStatus.LENGTH_REQUIRED,
                "Staged uploads require a Content-Length header.",
            )
            return
        pool = _admission_pool(scope, limit)
        admission = None if pool is None else self._admission[pool]
        if admission is not None and not admission.acquire(blocking=False):
            await _send_error(
                send, HTTPStatus.SERVICE_UNAVAILABLE, "Server is busy.", error_type="server_error"
            )
            return

        try:
            rejection = await self._serve(scope, receive, send, limit)
        finally:
            if admission is not None:
                admission.release()
        if rejection == HTTPStatus.REQUEST_ENTITY_TOO_LARGE:
            await _send_too_large(send, limit)
        elif rejection == HTTPStatus.REQUEST_TIMEOUT:
            await _send_error(send, rejection, "Request body receipt timed out.")

    async def _serve(
        self, scope: Scope, receive: Receive, send: Send, limit: int
    ) -> HTTPStatus | None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + receipt_budget_seconds(scope, limit)
        idle_timeout = settings.VOXHELM_ASGI_BODY_IDLE_TIMEOUT_SECONDS
        received = 0
        body_complete = False
        rejection: HTTPStatus | None = None
        receive_error: Exception | asyncio.CancelledError | None = None
        response_started = False
        disconnect_requested = False
        disconnect_receive: asyncio.Future[Message] | None = None

        async def receive_after_body() -> Message:
            nonlocal disconnect_receive
            if disconnect_requested:
                return {"type": "http.disconnect"}
            # Interrupt the transport wait independently of Django's listener.
            # Django may itself cancel the listener when the response finishes.
            listener = asyncio.current_task()
            assert listener is not None
            disconnect_receive = asyncio.ensure_future(receive())
            try:
                message = await disconnect_receive
            except asyncio.CancelledError:
                if not disconnect_requested or listener.cancelling():
                    raise
            else:
                if not disconnect_requested:
                    return message
            finally:
                disconnect_receive = None
            return {"type": "http.disconnect"}

        async def limited_receive() -> Message:
            nonlocal received, body_complete, rejection, receive_error
            if rejection is not None or receive_error is not None:
                return {"type": "http.disconnect"}
            if body_complete:
                return await receive_after_body()

            now = loop.time()
            # The next chunk must arrive within the idle timeout and the total budget.
            chunk_deadline = min(deadline, now + idle_timeout)
            timer = asyncio.timeout_at(chunk_deadline)
            message = None
            try:
                if now < deadline:
                    async with timer:
                        message = await receive()
            except (Exception, asyncio.CancelledError) as exc:
                if not isinstance(exc, TimeoutError) or not timer.expired():
                    receive_error = exc
                    return {"type": "http.disconnect"}

            # Django closes the spool on disconnect and returns without a response.
            # A chunk that resumes after either deadline, before the timer
            # callback ran, is late too.
            if message is None or (
                message.get("type") != "http.disconnect" and loop.time() >= chunk_deadline
            ):
                rejection = HTTPStatus.REQUEST_TIMEOUT
                return {"type": "http.disconnect"}
            if message.get("type") == "http.request":
                received += len(message.get("body") or b"")
                if received > limit:
                    rejection = HTTPStatus.REQUEST_ENTITY_TOO_LARGE
                    return {"type": "http.disconnect"}
                body_complete = not message.get("more_body", False)
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if disconnect_requested:
                return
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        app_task = asyncio.ensure_future(self.app(scope, limited_receive, tracking_send))
        try:
            await asyncio.shield(app_task)
        except asyncio.CancelledError as cancelled:
            if not app_task.done():
                if body_complete:
                    # Let Django cancel its view and close the spool via its
                    # normal disconnect path. Cancelling its handler directly
                    # after receipt can leave both running beyond admission.
                    disconnect_requested = True
                    if disconnect_receive is not None:
                        disconnect_receive.cancel()
                else:
                    # Before the last chunk Django only suspends in receive()
                    # (Django 5.2 writes spool chunks synchronously), so the
                    # cancellation lands in limited_receive, which turns it into
                    # a disconnect; Django closes the spool and it is re-raised.
                    app_task.cancel()
                while not app_task.done():
                    try:
                        await asyncio.wait((app_task,))
                    except asyncio.CancelledError:
                        continue
            if not app_task.cancelled() and (error := app_task.exception()) is not None:
                raise cancelled from error
            raise
        if receive_error is not None:
            raise receive_error
        return None if response_started else rejection
