"""Stream an open binary reader as an HTTP response under WSGI and ASGI."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import suppress
from threading import Lock
from typing import BinaryIO

from django.core.handlers.asgi import ASGIRequest
from django.http import FileResponse, HttpRequest, StreamingHttpResponse
from django.http.response import HttpResponseBase

_LOGGER = logging.getLogger(__name__)


def _close_reader(reader: BinaryIO, *, preserve_error: bool) -> None:
    try:
        reader.close()
    except BaseException:
        if not preserve_error:
            raise
        # A failing diagnostic must also leave the primary error in control.
        with suppress(BaseException):
            _LOGGER.warning("Reader cleanup failed; operator recovery may be needed.")


class AsyncReaderStream:
    """Keep ownership in the read worker after cancellation until the read settles."""

    def __init__(self, reader: BinaryIO) -> None:
        self.reader = reader
        self.lock = Lock()
        self.claimed = False
        self.reading = False
        self.closed = False

    def close(self, primary: BaseException | None = None) -> None:
        """Close once; a close error must not replace ``primary``, this stream's own failure."""
        with self.lock:
            if self.closed:
                return
            self.closed = True
            if not self.reading:
                _close_reader(self.reader, preserve_error=primary is not None)

    def _read(self) -> bytes:
        with self.lock:
            if self.closed:
                return b""
            self.reading = True
        try:
            return self.reader.read(65536)
        finally:
            with self.lock:
                self.reading = False
                if self.closed:
                    # Retirement was requested during this read (for example by
                    # cancellation). Neither a read error nor cancellation may
                    # be replaced by its secondary close failure.
                    _close_reader(self.reader, preserve_error=True)

    async def __aiter__(self) -> AsyncIterator[bytes]:
        with self.lock:
            if self.closed:
                return
            if self.claimed:
                raise RuntimeError("Reader stream is already being consumed.")
            self.claimed = True

        # Track this iterator's own failure explicitly: sys.exception() in a finally
        # block would also report an unrelated exception a caller is handling.
        primary: BaseException | None = None
        try:
            while chunk := await asyncio.to_thread(self._read):
                yield chunk
        except BaseException as exc:
            primary = exc
            raise
        finally:
            self.close(primary)


def reader_response(
    request: HttpRequest,
    reader: BinaryIO,
    *,
    content_type: str,
    content_length: int | None = None,
) -> HttpResponseBase:
    """Return a bounded streaming response that owns ``reader``.

    Under ASGI, Django would materialize a synchronous file iterator in memory, so
    reads run in worker threads instead; the response closes the reader exactly
    once, after any in-flight read settles. Both transports use the admitted
    ``content_length`` when supplied and otherwise keep Django's inferred headers.
    """
    owner = None
    try:
        file_response = FileResponse(reader, content_type=content_type)
        if content_length is not None:
            file_response["Content-Length"] = str(content_length)
        if not isinstance(request, ASGIRequest):
            return file_response
        # Only the stream owner closes the reader; FileResponse is a header
        # probe and must not race an in-flight worker after a disconnect.
        owner = AsyncReaderStream(reader)
        response = StreamingHttpResponse(owner, content_type=content_type)
        for header in ("Content-Length", "Content-Disposition"):
            if header in file_response:
                response[header] = file_response[header]
        return response
    except BaseException as exc:
        if owner is not None:
            owner.close(exc)
        else:
            _close_reader(reader, preserve_error=True)
        raise
