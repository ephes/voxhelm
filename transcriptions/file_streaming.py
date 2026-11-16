"""Stream an open binary reader as an HTTP response under WSGI and ASGI."""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import AsyncIterator
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
        _LOGGER.warning("Reader cleanup failed; operator recovery may be needed.")


class AsyncReaderStream:
    """Keep ownership in the read worker after cancellation until the read settles."""

    def __init__(self, reader: BinaryIO) -> None:
        self.reader = reader
        self.lock = Lock()
        self.reading = False
        self.closed = False

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.closed = True
            if not self.reading:
                _close_reader(self.reader, preserve_error=sys.exception() is not None)

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
        try:
            while chunk := await asyncio.to_thread(self._read):
                yield chunk
        finally:
            self.close()


def reader_response(
    request: HttpRequest, reader: BinaryIO, *, content_type: str
) -> HttpResponseBase:
    """Return a bounded streaming response that owns ``reader``.

    Under ASGI, Django would materialize a synchronous file iterator in memory, so
    reads run in worker threads instead; the response closes the reader exactly
    once, after any in-flight read settles. ``Content-Length`` and
    ``Content-Disposition`` match what ``FileResponse`` computes under WSGI.
    """
    owner = None
    try:
        file_response = FileResponse(reader, content_type=content_type)
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
    except BaseException:
        if owner is not None:
            owner.close()
        else:
            _close_reader(reader, preserve_error=True)
        raise
