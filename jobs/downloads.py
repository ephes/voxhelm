"""Bounded artifact downloads for producer and operator endpoints."""

from __future__ import annotations

from collections.abc import Mapping

from botocore.exceptions import BotoCoreError, ClientError
from django.http import HttpRequest
from django.http.response import HttpResponseBase

from jobs import artifacts
from jobs.models import JobArtifact
from transcriptions.file_streaming import reader_response
from transcriptions.views import openai_error_response


def artifact_response(request: HttpRequest, artifact: JobArtifact) -> HttpResponseBase:
    try:
        store = artifacts.get_artifact_store_for_identity(artifact.storage_identity)
        size_bytes = artifact.size_bytes
        reader = store.open_reader(key=artifact.storage_key, expected_size_bytes=size_bytes)
        return reader_response(
            request, reader, content_type=artifact.content_type, content_length=size_bytes
        )
    except (OSError, BotoCoreError, ClientError, ValueError, RuntimeError) as exc:
        missing = isinstance(exc, FileNotFoundError)
        if isinstance(exc, ClientError) and isinstance(exc.response, Mapping):
            error = exc.response.get("Error")
            code = error.get("Code") if isinstance(error, Mapping) else None
            missing = isinstance(code, str) and code in {"NoSuchKey", "NotFound", "404"}
        return openai_error_response(
            "Artifact object is missing."
            if missing
            else "Artifact storage is unavailable; retry later.",
            status=404 if missing else 503,
            error_type="not_found_error" if missing else "server_error",
        )
