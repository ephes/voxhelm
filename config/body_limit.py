"""Bound HTTP request bodies before Django reads them under ASGI.

Django's ASGI handler copies the whole request body into a temporary file before
URL routing, CSRF, authentication or any view runs, and it has no size limit for
that copy. Without this wrapper any client that reaches the port could stream an
unbounded body to disk without credentials.

The ceiling depends on the route and method. POSTs to the two bearer upload
routes admit large bodies with a valid producer token. POSTs to the operator
form at ``/`` admit the synchronous upload limit. Every other request, and an
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
JSON ceiling, with a body announced or received) own a Django spool at once. Each large-body
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
    if scope.get("method", "").upper() != "POST":
        return DEFAULT_BODY_LIMIT_BYTES
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
    for key, value in scope.get("headers") or ():
        name = key.lower()
        if name == b"transfer-encoding":
            return True
        if name == b"content-length":
            length = bytes(value).decode("latin-1").strip()
            # Malformed lengths also count as a body announcement.
            if not length or length.lstrip("0"):
                return True
    return False


def _declared_length(scope: Scope, limit: int) -> int | None:
    """Return the largest valid length, saturating above the ceiling at limit + 1."""
    ceiling = str(limit)
    declared = None
    for key, value in scope.get("headers") or ():
        if key.lower() != b"content-length":
            continue
        text = bytes(value).decode("latin-1").strip()
        if not (text.isascii() and text.isdigit()):
            continue
        significant = text.lstrip("0") or "0"
        if len(significant) > len(ceiling) or (
            len(significant) == len(ceiling) and significant > ceiling
        ):
            return limit + 1
        declared = max(declared or 0, int(significant))
    return declared


def _declared_length_exceeds(scope: Scope, limit: int) -> bool:
    # Any oversized length is refused early, whatever else frames the body.
    declared = _declared_length(scope, limit)
    return declared is not None and declared > limit


def _framed_length(scope: Scope, limit: int) -> int | None:
    """Return the length that frames the body, or None for chunked framing.

    ``Transfer-Encoding`` takes precedence over ``Content-Length`` in HTTP/1.1
    (h11 reads such a body as chunked), so a length sent beside it frames
    nothing and must not extend the receipt budget.
    """
    if any(key.lower() == b"transfer-encoding" for key, _value in scope.get("headers") or ()):
        return None
    return _declared_length(scope, limit)


def receipt_budget_seconds(scope: Scope, limit: int) -> float:
    """Return the total time this request's body may take to arrive.

    The budget is the configured base, or the declared body length (capped at
    the route ceiling) at the minimum upload rate if that takes longer. A body
    without a declared length keeps the base budget, and small ceilings
    (anonymous and JSON routes) never exceed it.
    """
    declared = _framed_length(scope, limit)
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


def _staged_upload_requires_length(scope: Scope, limit: int) -> bool:
    """Whether a producer's staged POST lacks a length for multipart parsing."""
    return (
        scope.get("method", "").upper() == "POST"
        and _route_path(scope) == STAGED_UPLOAD_PATH
        and not _framed_length(scope, limit)
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
        length_required = _staged_upload_requires_length(scope, limit)
        announced = _announces_body(scope)
        admission = (
            self._admission[_route_path(scope)] if limit > DEFAULT_BODY_LIMIT_BYTES else None
        )
        admitted = False

        def try_admit() -> bool:
            nonlocal admitted
            if admission is None:
                return True
            if not admitted:
                admitted = admission.acquire(blocking=False)
            return admitted

        rejection: HTTPStatus | None
        if announced and length_required:
            rejection = HTTPStatus.LENGTH_REQUIRED
        elif announced and not try_admit():
            rejection = HTTPStatus.SERVICE_UNAVAILABLE
        else:
            try:
                rejection = await self._serve(
                    scope, receive, send, limit, try_admit, length_required
                )
            finally:
                if admission is not None and admitted:
                    admission.release()

        if rejection == HTTPStatus.REQUEST_ENTITY_TOO_LARGE:
            await _send_too_large(send, limit)
        elif rejection == HTTPStatus.REQUEST_TIMEOUT:
            await _send_error(send, rejection, "Request body receipt timed out.")
        elif rejection == HTTPStatus.LENGTH_REQUIRED:
            await _send_error(send, rejection, "Staged uploads require a Content-Length header.")
        elif rejection == HTTPStatus.SERVICE_UNAVAILABLE:
            await _send_error(send, rejection, "Server is busy.", error_type="server_error")

    async def _serve(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        limit: int,
        try_admit: Callable[[], bool],
        length_required: bool,
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
                chunk_size = len(message.get("body") or b"")
                if received + chunk_size > limit:
                    rejection = HTTPStatus.REQUEST_ENTITY_TOO_LARGE
                    return {"type": "http.disconnect"}
                if received == 0 and chunk_size:
                    # An ASGI transport can supply bytes without HTTP/1.1
                    # framing. Refuse before Django writes the first chunk.
                    if length_required:
                        rejection = HTTPStatus.LENGTH_REQUIRED
                        return {"type": "http.disconnect"}
                    if not try_admit():
                        rejection = HTTPStatus.SERVICE_UNAVAILABLE
                        return {"type": "http.disconnect"}
                received += chunk_size
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
