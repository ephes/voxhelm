"""Bounded artifact downloads for producer and operator endpoints."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import BinaryIO

from asgiref.sync import sync_to_async
from botocore.exceptions import BotoCoreError, ClientError
from django.core.handlers.asgi import ASGIRequest
from django.http import FileResponse, HttpRequest
from django.http.response import HttpResponseBase

from jobs import artifacts
from jobs.models import JobArtifact
from transcriptions.views import openai_error_response


async def read_chunks(reader: BinaryIO) -> AsyncIterator[bytes]:
    try:
        while chunk := await asyncio.to_thread(reader.read, 65536):
            yield chunk
    finally:
        await sync_to_async(reader.close)()


def artifact_response(request: HttpRequest, artifact: JobArtifact) -> HttpResponseBase:
    reader = None
    try:
        store = artifacts.get_artifact_store_for_identity(artifact.storage_identity)
        reader = store.open_reader(key=artifact.storage_key)
        response = FileResponse(reader, content_type=artifact.content_type)
        if isinstance(request, ASGIRequest):
            # Django otherwise materializes a synchronous iterator under ASGI.
            response.streaming_content = read_chunks(reader)
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
