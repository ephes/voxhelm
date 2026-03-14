from __future__ import annotations

import json
from uuid import UUID

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from jobs.artifacts import get_artifact_store
from jobs.models import Job, JobArtifact
from jobs.services import create_job_from_payload, serialize_job
from transcriptions.errors import ApiError
from transcriptions.views import openai_error_response, require_bearer_token


@csrf_exempt
@require_POST
def jobs_collection(request: HttpRequest) -> JsonResponse:
    try:
        producer = require_bearer_token(request)
        if not (request.content_type or "").lower().startswith("application/json"):
            raise ApiError("Batch job submission requires application/json.")
        try:
            payload = json.loads(request.body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ApiError("Request body was not valid JSON.") from exc
        if not isinstance(payload, dict):
            raise ApiError("Request body must be a JSON object.")
        job, created = create_job_from_payload(producer=producer, payload=payload)
        status = 201 if created else 200
        return JsonResponse(serialize_job(job), status=status)
    except ApiError as exc:
        return openai_error_response(exc.message, status=exc.status, error_type=exc.error_type)
    except RuntimeError as exc:
        return openai_error_response(str(exc), status=500, error_type="server_error")


@require_GET
def job_detail(request: HttpRequest, job_id: UUID) -> JsonResponse:
    try:
        producer = require_bearer_token(request)
        job = get_object_or_404(Job, id=job_id, producer=producer)
        return JsonResponse(serialize_job(job))
    except ApiError as exc:
        return openai_error_response(exc.message, status=exc.status, error_type=exc.error_type)


@require_GET
def job_artifact(request: HttpRequest, job_id: UUID, name: str) -> HttpResponse:
    try:
        producer = require_bearer_token(request)
        job = get_object_or_404(Job, id=job_id, producer=producer)
        artifact = get_object_or_404(JobArtifact, job=job, name=name, exposed=True)
        store = get_artifact_store()
        data = store.read_bytes(key=artifact.storage_key)
        return HttpResponse(data, content_type=artifact.content_type)
    except ApiError as exc:
        return openai_error_response(exc.message, status=exc.status, error_type=exc.error_type)
