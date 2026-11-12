"""Bounded artifact downloads for producer and operator endpoints."""

from __future__ import annotations

from botocore.exceptions import BotoCoreError, ClientError
from django.http import HttpRequest
from django.http.response import HttpResponseBase

from jobs import artifacts
from jobs.models import JobArtifact
from transcriptions.file_streaming import reader_response
from transcriptions.views import openai_error_response


def artifact_response(request: HttpRequest, artifact: JobArtifact) -> HttpResponseBase:
    reader = None
    try:
        store = artifacts.get_artifact_store_for_identity(artifact.storage_identity)
        reader = store.open_reader(key=artifact.storage_key)
        return reader_response(request, reader, content_type=artifact.content_type)
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
