from __future__ import annotations

import json
from uuid import UUID

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from jobs.models import Job, JobArtifact
from jobs.remote_workers import (
    claim_remote_work,
    complete_remote_work,
    fail_remote_work,
    heartbeat_remote_work,
    record_worker_heartbeat,
    require_worker_token,
    serialize_claim,
    validate_worker_payload_identity,
)
from jobs.services import create_job_from_payload, serialize_job
from jobs.staging import serialize_staged_media, stage_uploaded_audio
from transcriptions.errors import ApiError
from transcriptions.views import openai_error_response, require_bearer_token


@csrf_exempt
@require_POST
def uploads_collection(request: HttpRequest) -> JsonResponse:
    try:
        producer = require_bearer_token(request)
        content_type = (request.content_type or "").lower()
        if not content_type.startswith("multipart/form-data"):
            raise ApiError("Batch upload staging requires multipart/form-data.")
        upload = request.FILES.get("file")
        if upload is None:
            raise ApiError("Multipart requests must include a file field named 'file'.")
        staged_media = stage_uploaded_audio(producer=producer, upload=upload)
        return JsonResponse(serialize_staged_media(staged_media), status=201)
    except ApiError as exc:
        return openai_error_response(exc.message, status=exc.status, error_type=exc.error_type)
    except RuntimeError as exc:
        return openai_error_response(str(exc), status=500, error_type="server_error")


@csrf_exempt
@require_POST
def jobs_collection(request: HttpRequest) -> JsonResponse:
    try:
        producer = require_bearer_token(request)
        if not (request.content_type or "").lower().startswith("application/json"):
            raise ApiError("Batch job submission requires application/json.")
        try:
            payload = json.loads(request.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
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


@csrf_exempt
@require_POST
def worker_heartbeat(request: HttpRequest) -> JsonResponse:
    try:
        worker_id = require_worker_token(request)
        payload = parse_json_object_request(request, "Worker heartbeat")
        validate_worker_payload_identity(worker_id=worker_id, payload=payload)
        return JsonResponse(record_worker_heartbeat(worker_id=worker_id, payload=payload))
    except ApiError as exc:
        return openai_error_response(exc.message, status=exc.status, error_type=exc.error_type)


@csrf_exempt
@require_POST
def work_claim(request: HttpRequest) -> JsonResponse | HttpResponse:
    try:
        worker_id = require_worker_token(request)
        payload = parse_json_object_request(request, "Worker claim")
        validate_worker_payload_identity(worker_id=worker_id, payload=payload)
        claim = claim_remote_work(worker_id=worker_id, payload=payload)
        if claim is None:
            return HttpResponse(status=204)
        return JsonResponse(serialize_claim(claim))
    except ApiError as exc:
        return openai_error_response(exc.message, status=exc.status, error_type=exc.error_type)


@csrf_exempt
@require_POST
def work_heartbeat(request: HttpRequest, job_id: UUID) -> JsonResponse:
    try:
        worker_id = require_worker_token(request)
        payload = parse_json_object_request(request, "Worker job heartbeat")
        validate_worker_payload_identity(worker_id=worker_id, payload=payload)
        return JsonResponse(
            heartbeat_remote_work(worker_id=worker_id, job_id=job_id, payload=payload)
        )
    except ApiError as exc:
        return openai_error_response(exc.message, status=exc.status, error_type=exc.error_type)


@csrf_exempt
@require_POST
def work_complete(request: HttpRequest, job_id: UUID) -> JsonResponse:
    try:
        worker_id = require_worker_token(request)
        payload = parse_json_object_request(request, "Worker job completion")
        validate_worker_payload_identity(worker_id=worker_id, payload=payload)
        return JsonResponse(
            complete_remote_work(worker_id=worker_id, job_id=job_id, payload=payload)
        )
    except Job.DoesNotExist:
        return openai_error_response(
            "Job not found.",
            status=404,
            error_type="invalid_request_error",
        )
    except ApiError as exc:
        return openai_error_response(exc.message, status=exc.status, error_type=exc.error_type)


@csrf_exempt
@require_POST
def work_fail(request: HttpRequest, job_id: UUID) -> JsonResponse:
    try:
        worker_id = require_worker_token(request)
        payload = parse_json_object_request(request, "Worker job failure")
        validate_worker_payload_identity(worker_id=worker_id, payload=payload)
        return JsonResponse(
            fail_remote_work(worker_id=worker_id, job_id=job_id, payload=payload)
        )
    except Job.DoesNotExist:
        return openai_error_response(
            "Job not found.",
            status=404,
            error_type="invalid_request_error",
        )
    except ApiError as exc:
        return openai_error_response(exc.message, status=exc.status, error_type=exc.error_type)


def parse_json_object_request(request: HttpRequest, context: str) -> dict[str, object]:
    if not (request.content_type or "").lower().startswith("application/json"):
        raise ApiError(f"{context} requires application/json.")
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ApiError("Request body was not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ApiError("Request body must be a JSON object.")
    return payload


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
        from jobs.artifacts import get_artifact_store_for_identity

        store = get_artifact_store_for_identity(artifact.storage_identity)
        data = store.read_bytes(key=artifact.storage_key)
        return HttpResponse(data, content_type=artifact.content_type)
    except ApiError as exc:
        return openai_error_response(exc.message, status=exc.status, error_type=exc.error_type)
