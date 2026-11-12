"""Bounded artifact downloads for producer and operator endpoints."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from threading import Lock
from typing import BinaryIO

from botocore.exceptions import BotoCoreError, ClientError
from django.core.handlers.asgi import ASGIRequest
from django.http import FileResponse, HttpRequest, StreamingHttpResponse
from django.http.response import HttpResponseBase

from jobs import artifacts
from jobs.models import JobArtifact
from transcriptions.views import openai_error_response


class _AsyncDownload:
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


def artifact_response(request: HttpRequest, artifact: JobArtifact) -> HttpResponseBase:
    reader = None
    try:
        store = artifacts.get_artifact_store_for_identity(artifact.storage_identity)
        reader = store.open_reader(key=artifact.storage_key)
        file_response = FileResponse(reader, content_type=artifact.content_type)
        if not isinstance(request, ASGIRequest):
            return file_response
        # Register only the owner's closer; the FileResponse would also register
        # reader.close and race an in-flight worker after a disconnect. It is
        # discarded unclosed and only supplies its computed headers.
        response = StreamingHttpResponse(_AsyncDownload(reader), content_type=artifact.content_type)
        for header in ("Content-Length", "Content-Disposition"):
            if header in file_response:
                response[header] = file_response[header]
        return response
    except (OSError, BotoCoreError, ClientError, ValueError, RuntimeError) as exc:
        if reader is not None:
            try:
                reader.close()
            except OSError, BotoCoreError, ClientError:
                pass
        missing = isinstance(exc, FileNotFoundError) or (
            isinstance(exc, ClientError)
            and exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "NotFound", "404"}
        )
        return openai_error_response(
            "Artifact object is missing."
            if missing
            else "Artifact storage is unavailable; retry later.",
            status=404 if missing else 503,
            error_type="not_found_error" if missing else "server_error",
        )
