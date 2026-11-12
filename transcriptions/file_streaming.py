"""Stream an open binary reader as an HTTP response under WSGI and ASGI."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from threading import Lock
from typing import BinaryIO

from django.core.handlers.asgi import ASGIRequest
from django.http import FileResponse, HttpRequest, StreamingHttpResponse
from django.http.response import HttpResponseBase


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
                self.reader.close()

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
                    self.reader.close()

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
    file_response = FileResponse(reader, content_type=content_type)
    if not isinstance(request, ASGIRequest):
        return file_response
    # Register only the owner's closer; the FileResponse would also register
    # reader.close and race an in-flight worker after a disconnect. It is
    # discarded unclosed and only supplies its computed headers.
    response = StreamingHttpResponse(AsyncReaderStream(reader), content_type=content_type)
    for header in ("Content-Length", "Content-Disposition"):
        if header in file_response:
            response[header] = file_response[header]
    return response
