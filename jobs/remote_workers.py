from __future__ import annotations

import hashlib
import hmac
import math
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import PurePosixPath
from typing import Any
from uuid import UUID

from django.conf import settings
from django.db import transaction
from django.db.models import Case, F, IntegerField, Q, Value, When
from django.db.models.functions import Coalesce
from django.http import HttpRequest
from django.utils import timezone

from jobs.artifacts import current_artifact_store_identity, get_artifact_store_for_identity
from jobs.media import validate_allowed_media_url
from jobs.models import Job, JobArtifact, StagedMedia, Worker
from jobs.services import (
    DEFAULT_TRANSCRIPTION_OUTPUT_FORMATS,
    isoformat_or_none,
    reconcile_remote_job_state,
    serialize_job,
    transcription_diarization_enabled,
    transcription_diarization_payload,
    transcription_diarization_strategy,
    validate_known_speaker_reference_urls,
)
from jobs.staging import delete_staged_media, release_staged_media_claims_for_job
from transcriptions.errors import ApiError
from transcriptions.known_speaker import (
    ANONYMOUS_STRATEGY,
    KNOWN_SPEAKER_STRATEGY,
    KnownSpeakerConfig,
)
from transcriptions.service import (
    resolve_backend_name_for_model,
    resolve_model_name_for_backend,
)

TRANSCRIPT_ARTIFACTS_BY_FORMAT: dict[str, tuple[str, str]] = {
    "text": ("transcript.txt", JobArtifact.Kind.TRANSCRIPT_TEXT),
    "json": ("transcript.json", JobArtifact.Kind.TRANSCRIPT_JSON),
    "vtt": ("transcript.vtt", JobArtifact.Kind.TRANSCRIPT_VTT),
    "dote": ("transcript.dote.json", JobArtifact.Kind.TRANSCRIPT_DOTE),
    "podlove": ("transcript.podlove.json", JobArtifact.Kind.TRANSCRIPT_PODLOVE),
}
TRANSCRIPT_CONTENT_TYPES_BY_FORMAT: dict[str, str] = {
    "text": "text/plain; charset=utf-8",
    "json": "application/json",
    "vtt": "text/vtt; charset=utf-8",
    "dote": "application/json",
    "podlove": "application/json",
}
TRANSCRIPT_FORMATS_BY_KIND: dict[str, str] = {
    kind: format_name for format_name, (_name, kind) in TRANSCRIPT_ARTIFACTS_BY_FORMAT.items()
}
REMOTE_SIDE_ARTIFACT_KINDS: set[str] = {
    JobArtifact.Kind.SOURCE,
    JobArtifact.Kind.EXTRACTED_AUDIO,
    JobArtifact.Kind.TRANSCRIPT_SPEAKERS,
}
REMOTE_RESULT_METADATA_KEYS: set[str] = {
    "backend",
    "requested_model",
    "model",
    "language",
    "duration_seconds",
    "processing_seconds",
    "source_url",
    "source_name",
    "source_content_type",
}
REMOTE_RESULT_METADATA_STRING_KEYS: set[str] = {
    "backend",
    "requested_model",
    "model",
    "language",
    "source_name",
    "source_content_type",
}
REMOTE_RESULT_METADATA_NUMBER_KEYS: set[str] = {
    "duration_seconds",
    "processing_seconds",
}
REMOTE_METADATA_STRING_MAX_LENGTH = 256
KNOWN_SPEAKER_SUMMARY_STRING_KEYS: set[str] = {
    "strategy",
    "embedding_model",
    "embedding_version",
}
KNOWN_SPEAKER_SUMMARY_INTEGER_KEYS: set[str] = {
    "segment_count",
    "confident_segment_count",
    "uncertain_segment_count",
}
KNOWN_SPEAKER_SUMMARY_NUMBER_KEYS: set[str] = {
    "min_segment_duration",
    "auto_accept_margin",
    "min_top_similarity",
    "margin_median",
}
SUMMARY_STRING_MAX_LENGTH = 256
SPEAKER_SIDECAR_CONTENT_TYPE = "application/json"
CLAIM_UPDATE_CANDIDATE_LIMIT = 10


@dataclass(frozen=True)
class RemoteClaim:
    job: Job
    lease_token: str


@dataclass(frozen=True)
class ResolvedSttTarget:
    backend: str
    model: str


def require_worker_token(request: HttpRequest, payload: dict[str, Any] | None = None) -> str:
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        raise ApiError(
            "Missing worker bearer token.",
            status=401,
            error_type="authentication_error",
        )

    presented = header.removeprefix("Bearer ").strip()
    matched_worker_id = ""
    for worker_id, token in settings.VOXHELM_WORKER_TOKENS.items():
        if hmac.compare_digest(presented, token):
            matched_worker_id = worker_id
            break
    if not matched_worker_id:
        raise ApiError(
            "Invalid worker bearer token.",
            status=401,
            error_type="authentication_error",
        )

    if payload is not None:
        validate_worker_payload_identity(worker_id=matched_worker_id, payload=payload)
    return matched_worker_id


def validate_worker_payload_identity(*, worker_id: str, payload: dict[str, Any]) -> None:
    if "worker_id" not in payload:
        return
    requested_worker_id = payload.get("worker_id")
    if requested_worker_id != worker_id:
        raise ApiError(
            "Worker token is not authorized for the requested worker_id.",
            status=403,
            error_type="permission_error",
        )


def record_worker_heartbeat(*, worker_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    capabilities = optional_object(payload.get("capabilities"), "capabilities")
    hostname = optional_string(payload.get("hostname")) or ""
    running_job_ids = normalize_string_list(payload.get("running_job_ids"), "running_job_ids")
    concurrency = optional_positive_int(payload.get("concurrency"), "concurrency") or 1
    now = timezone.now()

    worker, created = Worker.objects.get_or_create(
        worker_id=worker_id,
        defaults={
            "hostname": hostname,
            "capabilities": capabilities,
            "concurrency": concurrency,
            "running_job_ids": running_job_ids,
            "last_seen_at": now,
        },
    )
    if not created:
        ensure_worker_enabled(worker_id)
    if not created:
        worker.hostname = hostname
        worker.capabilities = capabilities
        worker.concurrency = concurrency
        worker.running_job_ids = running_job_ids
        worker.last_seen_at = now
        worker.save(
            update_fields=[
                "hostname",
                "capabilities",
                "concurrency",
                "running_job_ids",
                "last_seen_at",
                "updated_at",
            ]
        )

    return {
        "worker_id": worker.worker_id,
        "enabled": worker.enabled,
        "server_time": isoformat_or_none(now),
        "poll_after_seconds": settings.VOXHELM_REMOTE_WORKER_POLL_SECONDS,
    }


def claim_remote_work(*, worker_id: str, payload: dict[str, Any]) -> RemoteClaim | None:
    ensure_worker_enabled(worker_id)
    max_jobs = optional_positive_int(payload.get("max_jobs"), "max_jobs") or 1
    if max_jobs < 1:
        raise ApiError("max_jobs must be at least 1.")
    capabilities = optional_object(payload.get("capabilities"), "capabilities")
    while True:
        now = timezone.now()
        expire_exhausted_remote_leases(now)
        candidates = remote_claim_candidates(now=now, capabilities=capabilities)
        if not candidates:
            return None

        with transaction.atomic():
            lock_worker_for_claim(worker_id=worker_id, now=now)
            if not worker_has_claim_capacity(
                worker_id=worker_id,
                requested_max_jobs=max_jobs,
                now=now,
            ):
                return None
            # Fair-share: yield this round to a fresh, idle, less-loaded peer so
            # both workers converge on ~50/50 instead of the fastest poller
            # winning every race. No eligible peer => never defers.
            if worker_should_defer_for_fairness(worker_id=worker_id, now=now):
                return None

            for candidate in candidates:
                next_attempt = candidate.attempt_count + 1
                artifact_prefix = build_attempt_artifact_prefix(job=candidate, attempt=next_attempt)
                artifact_store_identity = current_artifact_store_identity()
                lease_token = secrets.token_urlsafe(32)
                lease_token_hash = hash_lease_token(lease_token)
                lease_expires_at = now + timedelta(
                    seconds=settings.VOXHELM_REMOTE_WORKER_LEASE_SECONDS
                )
                updated = (
                    Job.objects.filter(
                        id=candidate.id,
                        execution_mode=Job.ExecutionMode.REMOTE_PULL,
                        job_type=Job.JobType.TRANSCRIBE,
                        attempt_count=candidate.attempt_count,
                    )
                    .filter(claimable_job_filter(now))
                    .update(
                        state=Job.State.RUNNING,
                        assigned_worker_id=worker_id,
                        lease_token_hash=lease_token_hash,
                        lease_expires_at=lease_expires_at,
                        attempt_count=F("attempt_count") + 1,
                        leased_artifact_prefix=artifact_prefix,
                        leased_artifact_store=artifact_store_identity,
                        started_at=Coalesce("started_at", Value(now)),
                        last_worker_heartbeat_at=now,
                        worker_progress={},
                        error_detail="",
                    )
                )
                if updated == 1:
                    job = Job.objects.get(id=candidate.id)
                    return RemoteClaim(job=job, lease_token=lease_token)
    return None


def remote_claim_candidates(*, now: object, capabilities: dict[str, Any]) -> list[Job]:
    candidates = (
        Job.objects.filter(
            execution_mode=Job.ExecutionMode.REMOTE_PULL,
            job_type=Job.JobType.TRANSCRIBE,
        )
        .filter(claimable_job_filter(now))
        .annotate(
            priority_rank=Case(
                When(priority=Job.Priority.HIGH, then=Value(0)),
                When(priority=Job.Priority.NORMAL, then=Value(1)),
                When(priority=Job.Priority.LOW, then=Value(2)),
                default=Value(3),
                output_field=IntegerField(),
            )
        )
        .order_by("priority_rank", "created_at")
    )

    eligible: list[Job] = []
    for candidate in candidates.iterator(chunk_size=100):
        if not remote_payload_is_claimable(candidate):
            continue
        if not worker_can_run_job(candidate, capabilities):
            continue
        eligible.append(candidate)
        if len(eligible) >= CLAIM_UPDATE_CANDIDATE_LIMIT:
            break
    return eligible


def heartbeat_remote_work(
    *,
    worker_id: str,
    job_id: UUID,
    payload: dict[str, Any],
) -> dict[str, Any]:
    lease_token = require_lease_token(payload)
    progress = normalize_progress(payload.get("progress"))
    now = timezone.now()
    lease_expires_at = now + timedelta(seconds=settings.VOXHELM_REMOTE_WORKER_LEASE_SECONDS)
    updated = Job.objects.filter(
        id=job_id,
        execution_mode=Job.ExecutionMode.REMOTE_PULL,
        state=Job.State.RUNNING,
        assigned_worker_id=worker_id,
        lease_token_hash=hash_lease_token(lease_token),
        lease_expires_at__gt=now,
    ).update(
        lease_expires_at=lease_expires_at,
        last_worker_heartbeat_at=now,
        worker_progress=progress,
    )
    if updated != 1:
        raise lease_conflict()
    return {
        "job_id": str(job_id),
        "lease_expires_at": isoformat_or_none(lease_expires_at),
        "server_time": isoformat_or_none(now),
    }


def complete_remote_work(
    *,
    worker_id: str,
    job_id: UUID,
    payload: dict[str, Any],
) -> dict[str, Any]:
    lease_token = require_lease_token(payload)
    raw_artifacts = payload.get("artifacts")
    result_metadata_payload = optional_object(payload.get("result_metadata"), "result_metadata")
    result_text = require_result_text(payload.get("result_text"))
    current_job = Job.objects.get(id=job_id)
    if not is_same_successful_completion(
        job=current_job,
        worker_id=worker_id,
        lease_token=lease_token,
    ):
        validate_active_lease(job=current_job, worker_id=worker_id, lease_token=lease_token)
        validate_remote_artifacts(job=current_job, artifacts=raw_artifacts)

    staged_upload_id: str | None = None
    response: dict[str, Any]
    with transaction.atomic():
        lock_job_for_settlement(job_id=job_id, now=timezone.now())
        job = Job.objects.get(id=job_id)
        if is_same_successful_completion(job=job, worker_id=worker_id, lease_token=lease_token):
            artifacts = validate_remote_artifacts(
                job=job,
                artifacts=raw_artifacts,
                verify_objects=False,
            )
            metadata = normalize_remote_result_metadata(
                job=job,
                worker_id=worker_id,
                result_metadata=result_metadata_payload,
                use_stored_source_metadata=True,
            )
            if (
                job.result_text == result_text
                and job.result_metadata == metadata
                and stored_artifact_manifest(job) == artifacts
            ):
                staged_upload_id = remote_staged_upload_id(job)
                response = serialize_job(job)
            else:
                raise ApiError(
                    "Completion retry conflicts with the already committed result.",
                    status=409,
                    error_type="conflict_error",
                )
        else:
            validate_active_lease(job=job, worker_id=worker_id, lease_token=lease_token)
            artifacts = validate_remote_artifacts(
                job=job,
                artifacts=raw_artifacts,
                verify_objects=False,
            )
            metadata = normalize_remote_result_metadata(
                job=job,
                worker_id=worker_id,
                result_metadata=result_metadata_payload,
            )
            JobArtifact.objects.filter(job=job).delete()
            JobArtifact.objects.bulk_create(
                [
                    JobArtifact(
                        job=job,
                        name=artifact["name"],
                        kind=artifact["kind"],
                        format=artifact["format"],
                        storage_backend=artifact["storage_backend"],
                        storage_key=artifact["storage_key"],
                        storage_identity=artifact["storage_identity"],
                        content_type=artifact["content_type"],
                        size_bytes=artifact["size_bytes"],
                        exposed=artifact["exposed"],
                    )
                    for artifact in artifacts
                ]
            )
            finished_at = timezone.now()
            updated = active_remote_lease_queryset(
                job_id=job.id,
                worker_id=worker_id,
                lease_token=lease_token,
                now=finished_at,
            ).update(
                state=Job.State.SUCCEEDED,
                result_text=result_text,
                result_metadata=metadata,
                finished_at=finished_at,
                worker_progress={},
                updated_at=finished_at,
            )
            if updated != 1:
                raise lease_conflict()
            staged_upload_id = remote_staged_upload_id(job)
            job.refresh_from_db()
            response = serialize_job(job)
    delete_completed_remote_staged_upload(job_id=job_id, upload_id=staged_upload_id)
    return response


def fail_remote_work(*, worker_id: str, job_id: UUID, payload: dict[str, Any]) -> dict[str, Any]:
    lease_token = require_lease_token(payload)
    retryable = payload.get("retryable")
    if not isinstance(retryable, bool):
        raise ApiError("retryable must be a boolean.")
    require_string(payload.get("error_detail"), "error_detail")
    now = timezone.now()
    with transaction.atomic():
        lock_job_for_settlement(job_id=job_id, now=now)
        job = Job.objects.get(id=job_id)
        validate_active_lease(job=job, worker_id=worker_id, lease_token=lease_token)
        will_retry = retryable and job.attempt_count < job.max_attempts
        error_detail = sanitize_remote_failure_detail(retryable=will_retry)
        if will_retry:
            updated = active_remote_lease_queryset(
                job_id=job.id,
                worker_id=worker_id,
                lease_token=lease_token,
                now=now,
            ).update(
                state=Job.State.QUEUED,
                assigned_worker_id="",
                lease_token_hash="",
                lease_expires_at=None,
                leased_artifact_prefix="",
                leased_artifact_store={},
                last_worker_heartbeat_at=None,
                worker_progress={},
                error_detail=error_detail,
                updated_at=now,
            )
        else:
            updated = active_remote_lease_queryset(
                job_id=job.id,
                worker_id=worker_id,
                lease_token=lease_token,
                now=now,
            ).update(
                state=Job.State.FAILED,
                error_detail=error_detail,
                finished_at=now,
                worker_progress={},
                updated_at=now,
            )
        if updated != 1:
            raise lease_conflict()
        if not will_retry:
            release_staged_media_claims_for_job(job=job)
        job.refresh_from_db()
        return serialize_job(job)


def sanitize_remote_failure_detail(*, retryable: bool) -> str:
    if retryable:
        return "Remote worker reported a retryable failure."
    return "Remote worker reported a terminal failure."


def serialize_claim(claim: RemoteClaim) -> dict[str, Any]:
    job = claim.job
    target = resolve_remote_stt_target(job)
    return {
        "job": {
            "id": str(job.id),
            "job_type": job.job_type,
            "attempt": job.attempt_count,
            "lease_token": claim.lease_token,
            "lease_expires_at": isoformat_or_none(job.lease_expires_at),
            "backend": target.backend,
            "model": target.model,
            "requested_backend": job.backend or "auto",
            "requested_model": job.model or "auto",
            "language": job.language or None,
            "input": serialize_claim_input(job),
            "output": {
                "formats": list(
                    job.output_data.get("formats", list(DEFAULT_TRANSCRIPTION_OUTPUT_FORMATS))
                ),
                "diarization": serialize_claim_diarization(job),
            },
            "artifact_prefix": leased_artifact_prefix(job),
            "artifact_store": job.leased_artifact_store,
        }
    }


def serialize_claim_input(job: Job) -> dict[str, Any]:
    input_kind = str(job.input_data.get("kind") or "")
    if input_kind == "url":
        source_url = str(job.input_data.get("url") or "")
        validate_allowed_media_url(source_url)
        return {"kind": "url", "url": source_url}
    if input_kind == "upload":
        payload: dict[str, Any] = {
            "kind": "upload",
            "filename": job.input_data.get("filename"),
            "content_type": job.input_data.get("content_type"),
            "size_bytes": job.input_data.get("size_bytes"),
        }
        staged_backend = job.input_data.get("staged_storage_backend")
        staged_key = job.input_data.get("staged_storage_key")
        staged_identity = job.input_data.get("staged_storage_identity")
        if isinstance(staged_backend, str) and isinstance(staged_key, str) and staged_key:
            payload["staged_artifact"] = {
                "storage_backend": staged_backend,
                "storage_key": staged_key,
            }
            if isinstance(staged_identity, dict):
                payload["staged_artifact"]["storage_identity"] = staged_identity
            return payload
        staged = StagedMedia.objects.filter(
            id=job.input_data.get("upload_id"),
            claimed_by_job=job,
        ).first()
        if staged is not None:
            payload["staged_artifact"] = {
                "storage_backend": staged.storage_backend,
                "storage_key": staged.storage_key,
                "storage_identity": staged.storage_identity,
            }
        return payload
    raise ApiError(f"Unsupported remote worker input kind '{input_kind}'.")


def serialize_claim_diarization(job: Job) -> dict[str, Any]:
    raw_diarization = job.output_data.get("diarization")
    if not isinstance(raw_diarization, dict):
        return {"enabled": False}
    payload = dict(raw_diarization)
    if payload.get("enabled") is True and "strategy" not in payload:
        payload["strategy"] = ANONYMOUS_STRATEGY
    return payload


def remote_staged_upload_id(job: Job) -> str | None:
    if str(job.input_data.get("kind") or "") != "upload":
        return None
    upload_id = job.input_data.get("upload_id")
    if not isinstance(upload_id, str) or not upload_id:
        return None
    return upload_id


def lock_worker_for_claim(*, worker_id: str, now: object) -> None:
    # The write serializes concurrent claim transactions for this worker before
    # the capacity count and job lease update run.
    updated = Worker.objects.filter(worker_id=worker_id, enabled=True).update(updated_at=now)
    if updated != 1:
        ensure_worker_enabled(worker_id)


def lock_job_for_settlement(*, job_id: UUID, now: object) -> None:
    # Serialize duplicate completion/failure requests before artifact mutation.
    Job.objects.filter(id=job_id).update(updated_at=now)


def worker_recent_claim_load(*, worker_id: str, window_start: datetime) -> int:
    return Job.objects.filter(
        execution_mode=Job.ExecutionMode.REMOTE_PULL,
        assigned_worker_id=worker_id,
        started_at__gte=window_start,
    ).count()


def worker_should_defer_for_fairness(*, worker_id: str, now: datetime) -> bool:
    """Return True if this worker should yield the next claim to a less-loaded peer.

    Defers only when another worker is enabled, freshly heartbeating (really
    online), has spare capacity, and has handled strictly fewer recent claims.
    With no such peer it never defers, so a lone worker keeps claiming everything
    and the fleet self-heals when a peer goes away.
    """

    if not settings.VOXHELM_REMOTE_WORKER_BALANCE_ENABLED:
        return False

    window_start = now - timedelta(
        seconds=settings.VOXHELM_REMOTE_WORKER_BALANCE_WINDOW_SECONDS
    )
    fresh_cutoff = now - timedelta(
        seconds=settings.VOXHELM_REMOTE_WORKER_BALANCE_PEER_FRESH_SECONDS
    )
    my_load = worker_recent_claim_load(worker_id=worker_id, window_start=window_start)

    peers = (
        Worker.objects.filter(enabled=True, last_seen_at__gte=fresh_cutoff)
        .exclude(worker_id=worker_id)
    )
    for peer in peers:
        peer_running = Job.objects.filter(
            execution_mode=Job.ExecutionMode.REMOTE_PULL,
            state=Job.State.RUNNING,
            assigned_worker_id=peer.worker_id,
            lease_expires_at__gt=now,
        ).count()
        if peer_running >= peer.concurrency:
            continue
        peer_load = worker_recent_claim_load(
            worker_id=peer.worker_id, window_start=window_start
        )
        if peer_load < my_load:
            return True
    return False


def worker_has_claim_capacity(
    *,
    worker_id: str,
    requested_max_jobs: int,
    now: object,
) -> bool:
    worker = Worker.objects.get(worker_id=worker_id)
    capacity = min(requested_max_jobs, worker.concurrency)
    active_count = Job.objects.filter(
        execution_mode=Job.ExecutionMode.REMOTE_PULL,
        state=Job.State.RUNNING,
        assigned_worker_id=worker_id,
        lease_expires_at__gt=now,
    ).count()
    return active_count < capacity


def delete_completed_remote_staged_upload(*, job_id: UUID, upload_id: str | None) -> None:
    if upload_id is None:
        return
    staged = StagedMedia.objects.filter(id=upload_id, claimed_by_job_id=job_id).first()
    if staged is not None:
        delete_staged_media(staged=staged, missing_ok=True)


def claimable_job_filter(now: object) -> Q:
    attempts_remain = Q(attempt_count__lt=F("max_attempts"))
    return attempts_remain & (
        Q(state=Job.State.QUEUED)
        | Q(state=Job.State.RUNNING, lease_expires_at__lt=now)
    )


def remote_payload_is_claimable(job: Job) -> bool:
    try:
        validate_remote_claim_urls(job)
    except ApiError as exc:
        mark_remote_job_failed_before_claim(job=job, message=exc.message)
        return False
    return True


def validate_remote_claim_urls(job: Job) -> None:
    input_kind = str(job.input_data.get("kind") or "")
    if input_kind == "url":
        validate_allowed_media_url(str(job.input_data.get("url") or ""))
    validate_known_speaker_reference_urls(job.output_data.get("diarization"))
    validate_remote_known_speaker_reference_audio(job.output_data.get("diarization"))


def validate_remote_known_speaker_reference_audio(diarization: object) -> None:
    if not isinstance(diarization, dict):
        return
    if diarization.get("strategy") != KNOWN_SPEAKER_STRATEGY:
        return
    known_speakers = diarization.get("known_speakers")
    if not isinstance(known_speakers, list):
        return
    for speaker in known_speakers:
        if not isinstance(speaker, dict):
            continue
        references = speaker.get("references")
        if not isinstance(references, list):
            continue
        for reference in references:
            if not isinstance(reference, dict):
                continue
            audio = reference.get("audio")
            if isinstance(audio, dict) and audio.get("kind") == "upload":
                raise ApiError(
                    "Uploaded known-speaker reference audio is not supported for remote workers. "
                    "Use URL reference audio."
                )


def mark_remote_job_failed_before_claim(*, job: Job, message: str) -> None:
    now = timezone.now()
    updated = Job.objects.filter(id=job.id, execution_mode=Job.ExecutionMode.REMOTE_PULL).filter(
        Q(state=Job.State.QUEUED, assigned_worker_id="")
        | Q(state=Job.State.RUNNING, lease_expires_at__lt=now)
    ).update(
        state=Job.State.FAILED,
        error_detail=message,
        finished_at=now,
        worker_progress={},
        updated_at=now,
    )
    if updated:
        release_staged_media_claims_for_job(job=job)


def worker_can_run_job(job: Job, capabilities: dict[str, Any]) -> bool:
    if Job.JobType.TRANSCRIBE not in string_set(capabilities.get("job_types")):
        return False
    if not worker_supports_requested_stt(job, capabilities):
        return False
    if not worker_supports_requested_output_formats(job, capabilities):
        return False

    if not transcription_diarization_enabled(job):
        return True

    diarization_capabilities = optional_object(capabilities.get("diarization"), "diarization")
    if transcription_diarization_strategy(job) != KNOWN_SPEAKER_STRATEGY:
        return diarization_capabilities.get("anonymous") is True

    if diarization_capabilities.get("known_speaker") is not True:
        return False
    embedding_models = string_set(diarization_capabilities.get("embedding_models"))
    if known_speaker_embedding_model(job) not in embedding_models:
        return False
    output_formats = string_set(capabilities.get("output_formats"))
    return "speakers" in output_formats or "transcript_speakers" in output_formats


def worker_supports_requested_stt(job: Job, capabilities: dict[str, Any]) -> bool:
    advertised_backends = string_set(capabilities.get("backends"))
    advertised_models = string_set(capabilities.get("models"))
    if not advertised_backends or not advertised_models:
        return False

    try:
        target = resolve_remote_stt_target(job)
    except RuntimeError:
        return False
    return target.backend in advertised_backends and target.model in advertised_models


def resolve_remote_stt_target(job: Job) -> ResolvedSttTarget:
    requested_model = job.model or "auto"
    requested_backend = job.backend or "auto"
    backend_name = (
        resolve_backend_name_for_model(requested_model)
        if requested_backend == "auto"
        else requested_backend
    )
    model_name = resolve_model_name_for_backend(
        request_model=requested_model,
        backend_name=backend_name,
    )
    return ResolvedSttTarget(backend=backend_name, model=model_name)


def worker_supports_requested_output_formats(job: Job, capabilities: dict[str, Any]) -> bool:
    advertised_formats = string_set(capabilities.get("output_formats"))
    if not advertised_formats:
        return False
    raw_formats = job.output_data.get("formats", list(DEFAULT_TRANSCRIPTION_OUTPUT_FORMATS))
    requested_formats = {
        str(format_name)
        for format_name in raw_formats
    }
    return requested_formats.issubset(advertised_formats)


def known_speaker_embedding_model(job: Job) -> str:
    raw_diarization = job.output_data.get("diarization")
    if isinstance(raw_diarization, dict):
        raw_config = raw_diarization.get("known_speaker")
        if isinstance(raw_config, dict):
            embedding_model = raw_config.get("embedding_model")
            if isinstance(embedding_model, str) and embedding_model.strip():
                return embedding_model.strip()
    return KnownSpeakerConfig().embedding_model


def validate_remote_artifacts(
    *,
    job: Job,
    artifacts: object,
    verify_objects: bool = True,
) -> list[dict[str, Any]]:
    if not isinstance(artifacts, list) or not artifacts:
        raise ApiError("artifacts must be a non-empty list.")
    normalized = [normalize_remote_artifact(job=job, value=value) for value in artifacts]
    names = [artifact["name"] for artifact in normalized]
    if len(names) != len(set(names)):
        raise ApiError("Artifact names must be unique per job.")
    if not any(artifact["kind"] == JobArtifact.Kind.SOURCE for artifact in normalized):
        raise ApiError("Remote completions must include a source artifact.")

    requested_formats = set(
        job.output_data.get("formats", list(DEFAULT_TRANSCRIPTION_OUTPUT_FORMATS))
    )
    transcript_formats = {
        TRANSCRIPT_FORMATS_BY_KIND[artifact["kind"]]
        for artifact in normalized
        if artifact["kind"] in TRANSCRIPT_FORMATS_BY_KIND
    }
    missing_formats = sorted(requested_formats - transcript_formats)
    extra_formats = sorted(transcript_formats - requested_formats)
    if missing_formats:
        raise ApiError(f"Missing transcript artifact format(s): {', '.join(missing_formats)}.")
    if extra_formats:
        raise ApiError(f"Unexpected transcript artifact format(s): {', '.join(extra_formats)}.")

    has_speaker_sidecar = any(
        artifact["kind"] == JobArtifact.Kind.TRANSCRIPT_SPEAKERS for artifact in normalized
    )
    if transcription_diarization_strategy(job) == KNOWN_SPEAKER_STRATEGY:
        if not has_speaker_sidecar:
            raise ApiError("Known-speaker jobs must include transcript.speakers.json.")
    elif has_speaker_sidecar:
        raise ApiError("Speaker sidecar artifacts are only allowed for known-speaker jobs.")
    if verify_objects:
        verify_remote_artifact_objects(normalized, store_identity=job.leased_artifact_store)
    return sorted(normalized, key=lambda artifact: artifact["name"])


def normalize_remote_artifact(*, job: Job, value: object) -> dict[str, Any]:
    artifact = optional_object(value, "artifacts[]")
    name = require_string(artifact.get("name"), "artifacts[].name")
    if "/" in name or name in {".", ".."}:
        raise ApiError("Artifact names must be simple file names.")
    kind = require_string(artifact.get("kind"), "artifacts[].kind")
    if kind not in JobArtifact.Kind.values:
        raise ApiError(f"Unsupported artifact kind '{kind}'.")
    format_name = require_string(artifact.get("format"), "artifacts[].format")
    expected = TRANSCRIPT_ARTIFACTS_BY_FORMAT.get(format_name)
    if kind in TRANSCRIPT_FORMATS_BY_KIND:
        expected_format = TRANSCRIPT_FORMATS_BY_KIND[kind]
        expected_name, _expected_kind = TRANSCRIPT_ARTIFACTS_BY_FORMAT[expected_format]
        if format_name != expected_format or name != expected_name:
            raise ApiError(f"Artifact {name} does not match its transcript format.")
    elif kind == JobArtifact.Kind.TRANSCRIPT_SPEAKERS:
        if name != "transcript.speakers.json" or format_name != "speakers":
            raise ApiError("Speaker sidecar artifact must be transcript.speakers.json.")
    elif expected is not None:
        raise ApiError(f"Artifact format '{format_name}' does not match kind '{kind}'.")
    elif kind not in REMOTE_SIDE_ARTIFACT_KINDS:
        raise ApiError(f"Unsupported remote artifact kind '{kind}'.")

    storage_backend = require_string(artifact.get("storage_backend"), "artifacts[].storage_backend")
    if storage_backend != leased_artifact_store_backend(job):
        raise ApiError("Artifact storage_backend does not match Voxhelm configuration.")
    storage_key = require_string(artifact.get("storage_key"), "artifacts[].storage_key")
    prefix = leased_artifact_prefix(job)
    if not storage_key_is_under_attempt_prefix(storage_key, prefix=prefix):
        raise ApiError("Artifact storage_key must be under the claimed attempt prefix.")
    content_type = require_string(artifact.get("content_type"), "artifacts[].content_type")
    validate_remote_artifact_content_type(
        name=name,
        kind=kind,
        format_name=format_name,
        content_type=content_type,
    )
    size_bytes = artifact.get("size_bytes")
    if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
        raise ApiError("artifacts[].size_bytes must be a non-negative integer.")
    exposed = artifact.get("exposed")
    if not isinstance(exposed, bool):
        raise ApiError("artifacts[].exposed must be a boolean.")
    if kind in {JobArtifact.Kind.SOURCE, JobArtifact.Kind.EXTRACTED_AUDIO} and exposed:
        raise ApiError("Source and extracted-audio artifacts must not be exposed.")
    if kind in TRANSCRIPT_FORMATS_BY_KIND and not exposed:
        raise ApiError("Requested transcript artifacts must be exposed.")
    if kind == JobArtifact.Kind.TRANSCRIPT_SPEAKERS and not exposed:
        raise ApiError("Speaker sidecar artifact must be exposed to the producer.")
    return {
        "name": name,
        "kind": kind,
        "format": format_name,
        "storage_backend": storage_backend,
        "storage_key": storage_key,
        "storage_identity": job.leased_artifact_store,
        "content_type": content_type,
        "size_bytes": size_bytes,
        "exposed": exposed,
    }


def validate_remote_artifact_content_type(
    *,
    name: str,
    kind: str,
    format_name: str,
    content_type: str,
) -> None:
    expected_content_type = None
    if kind in TRANSCRIPT_FORMATS_BY_KIND:
        expected_content_type = TRANSCRIPT_CONTENT_TYPES_BY_FORMAT[format_name]
    elif kind == JobArtifact.Kind.TRANSCRIPT_SPEAKERS:
        expected_content_type = SPEAKER_SIDECAR_CONTENT_TYPE

    if expected_content_type is not None and content_type != expected_content_type:
        raise ApiError(f"Artifact {name} content_type must be {expected_content_type}.")


def verify_remote_artifact_objects(
    artifacts: list[dict[str, Any]],
    *,
    store_identity: dict[str, Any],
) -> None:
    try:
        store = get_artifact_store_for_identity(store_identity)
    except RuntimeError as exc:
        raise ApiError(f"Artifact store is not configured: {exc}") from exc
    for artifact in artifacts:
        try:
            stored = store.stat(key=artifact["storage_key"])
        except Exception as exc:
            raise ApiError(
                f"Artifact object '{artifact['name']}' is not available in the artifact store."
            ) from exc
        if stored.size_bytes != artifact["size_bytes"]:
            raise ApiError(
                f"Artifact object '{artifact['name']}' size_bytes does not match the manifest."
            )


def normalize_remote_result_metadata(
    *,
    job: Job,
    worker_id: str,
    result_metadata: dict[str, Any],
    use_stored_source_metadata: bool = False,
) -> dict[str, Any]:
    metadata = sanitize_remote_result_metadata(result_metadata)
    metadata["requested_model"] = job.model or "auto"
    clear_source_metadata(metadata)
    if use_stored_source_metadata:
        metadata.update(stored_remote_source_metadata(job))
    else:
        metadata.update(remote_source_metadata(job))
    if transcription_diarization_enabled(job):
        sanitized_diarization = transcription_diarization_payload(job)
        raw_diarization = result_metadata.get("diarization")
        if transcription_diarization_strategy(job) == KNOWN_SPEAKER_STRATEGY:
            if not isinstance(raw_diarization, dict) or not isinstance(
                raw_diarization.get("known_speaker_summary"),
                dict,
            ):
                raise ApiError(
                    "Known-speaker jobs must include result_metadata.diarization."
                    "known_speaker_summary."
                )
            sanitized_diarization["known_speaker_summary"] = sanitize_known_speaker_summary(
                raw_diarization["known_speaker_summary"],
                job=job,
            )
        metadata["diarization"] = sanitized_diarization
    metadata["worker_id"] = worker_id
    metadata["attempt"] = job.attempt_count
    metadata["execution_mode"] = Job.ExecutionMode.REMOTE_PULL
    return metadata


def clear_source_metadata(metadata: dict[str, Any]) -> None:
    for key in ("source_kind", "source_url", "source_name", "source_content_type"):
        metadata.pop(key, None)


def sanitize_remote_result_metadata(result_metadata: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in REMOTE_RESULT_METADATA_STRING_KEYS:
        sanitized = safe_worker_metadata_string(result_metadata.get(key))
        if sanitized is not None:
            metadata[key] = sanitized
    for key in REMOTE_RESULT_METADATA_NUMBER_KEYS:
        sanitized_number = safe_worker_metadata_number(result_metadata.get(key))
        if sanitized_number is not None:
            metadata[key] = sanitized_number
    return metadata


def safe_worker_metadata_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if (
        not normalized
        or is_url_like_metadata_string(normalized)
        or len(normalized) > REMOTE_METADATA_STRING_MAX_LENGTH
    ):
        return None
    return normalized


def safe_worker_metadata_number(value: object) -> int | float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    if value < 0 or not math.isfinite(value):
        return None
    return value


def sanitize_known_speaker_summary(value: dict[str, Any], *, job: Job) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in KNOWN_SPEAKER_SUMMARY_STRING_KEYS:
        sanitized = safe_summary_string(value.get(key))
        if sanitized is not None:
            summary[key] = sanitized
    for key in KNOWN_SPEAKER_SUMMARY_INTEGER_KEYS:
        raw = value.get(key)
        if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
            summary[key] = raw
    for key in KNOWN_SPEAKER_SUMMARY_NUMBER_KEYS:
        raw = value.get(key)
        sanitized_number = safe_worker_metadata_number(raw)
        if sanitized_number is not None:
            summary[key] = sanitized_number

    raw_known_speakers = value.get("known_speakers")
    allowed_speaker_names = configured_known_speaker_names(job)
    if isinstance(raw_known_speakers, list):
        known_speakers = [
            speaker.strip()
            for speaker in raw_known_speakers
            if (
                isinstance(speaker, str)
                and speaker.strip()
                and speaker.strip() in allowed_speaker_names
            )
        ]
        if known_speakers:
            summary["known_speakers"] = known_speakers

    raw_distribution = value.get("confident_speaker_distribution")
    if isinstance(raw_distribution, dict):
        distribution = {
            speaker.strip(): count
            for speaker, count in raw_distribution.items()
            if (
                isinstance(speaker, str)
                and speaker.strip()
                and speaker.strip() in allowed_speaker_names
                and isinstance(count, int)
                and not isinstance(count, bool)
                and count >= 0
            )
        }
        if distribution:
            summary["confident_speaker_distribution"] = distribution

    raw_turns_available = value.get("raw_diarization_available")
    if isinstance(raw_turns_available, bool):
        summary["raw_diarization_available"] = raw_turns_available
    return summary


def configured_known_speaker_names(job: Job) -> set[str]:
    raw_diarization = job.output_data.get("diarization")
    if not isinstance(raw_diarization, dict):
        return set()
    raw_speakers = raw_diarization.get("known_speakers")
    if not isinstance(raw_speakers, list):
        return set()
    return {
        speaker["name"].strip()
        for speaker in raw_speakers
        if (
            isinstance(speaker, dict)
            and isinstance(speaker.get("name"), str)
            and speaker["name"].strip()
        )
    }


def safe_summary_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if (
        not normalized
        or is_url_like_metadata_string(normalized)
        or len(normalized) > SUMMARY_STRING_MAX_LENGTH
    ):
        return None
    return normalized


def is_url_like_metadata_string(value: str) -> bool:
    if "://" in value or value.startswith("//"):
        return True
    head = value
    has_url_separator = False
    for separator in ("/", "?", "#"):
        if separator in head:
            has_url_separator = True
            head = head.split(separator, 1)[0]
    return has_url_separator and ("." in head or ":" in head)


def remote_source_metadata(job: Job) -> dict[str, Any]:
    input_kind = str(job.input_data.get("kind") or "")
    if input_kind == "url":
        return {
            "source_kind": "url",
            "source_url": remote_source_url(job),
        }
    if input_kind == "upload":
        metadata: dict[str, Any] = {"source_kind": "upload"}
        filename = job.input_data.get("filename")
        if isinstance(filename, str) and filename:
            metadata["source_name"] = filename
        content_type = job.input_data.get("content_type")
        if isinstance(content_type, str) and content_type:
            metadata["source_content_type"] = content_type
        return metadata
    return {}


def stored_remote_source_metadata(job: Job) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in ("source_kind", "source_url", "source_name", "source_content_type"):
        value = job.result_metadata.get(key)
        if isinstance(value, str):
            metadata[key] = value
    return metadata


def remote_source_url(job: Job) -> str:
    return str(job.input_data.get("url") or "")


def stored_artifact_manifest(job: Job) -> list[dict[str, Any]]:
    return sorted(
        [
            {
                "name": artifact.name,
                "kind": artifact.kind,
                "format": artifact.format,
                "storage_backend": artifact.storage_backend,
                "storage_key": artifact.storage_key,
                "storage_identity": artifact.storage_identity,
                "content_type": artifact.content_type,
                "size_bytes": artifact.size_bytes,
                "exposed": artifact.exposed,
            }
            for artifact in job.artifacts.all()
        ],
        key=lambda artifact: artifact["name"],
    )


def build_attempt_artifact_prefix(*, job: Job, attempt: int) -> str:
    configured_prefix = settings.VOXHELM_ARTIFACT_PREFIX.strip("/")
    job_prefix = f"jobs/{job.id}/attempt-{attempt}/"
    if configured_prefix:
        return f"{configured_prefix}/{job_prefix}"
    return job_prefix


def leased_artifact_prefix(job: Job) -> str:
    if job.leased_artifact_prefix:
        return job.leased_artifact_prefix
    return build_attempt_artifact_prefix(job=job, attempt=job.attempt_count)


def leased_artifact_store_backend(job: Job) -> str:
    backend = job.leased_artifact_store.get("backend")
    if isinstance(backend, str) and backend:
        return backend
    return settings.VOXHELM_ARTIFACT_BACKEND


def storage_key_is_under_attempt_prefix(storage_key: str, *, prefix: str) -> bool:
    if not storage_key.startswith(prefix):
        return False
    path = PurePosixPath(storage_key)
    if path.is_absolute() or ".." in path.parts:
        return False
    return len(storage_key) > len(prefix)


def expire_exhausted_remote_leases(now: object) -> None:
    exhausted_jobs = Job.objects.filter(
        execution_mode=Job.ExecutionMode.REMOTE_PULL,
        state=Job.State.RUNNING,
        lease_expires_at__lt=now,
        attempt_count__gte=F("max_attempts"),
    )
    for job in exhausted_jobs:
        reconcile_remote_job_state(job)


def validate_active_lease(*, job: Job, worker_id: str, lease_token: str) -> None:
    now = timezone.now()
    if (
        job.execution_mode != Job.ExecutionMode.REMOTE_PULL
        or job.state != Job.State.RUNNING
        or job.assigned_worker_id != worker_id
        or job.lease_token_hash != hash_lease_token(lease_token)
        or job.lease_expires_at is None
        or job.lease_expires_at <= now
    ):
        raise lease_conflict()


def active_remote_lease_queryset(
    *,
    job_id: UUID,
    worker_id: str,
    lease_token: str,
    now: object,
):
    return Job.objects.filter(
        id=job_id,
        execution_mode=Job.ExecutionMode.REMOTE_PULL,
        state=Job.State.RUNNING,
        assigned_worker_id=worker_id,
        lease_token_hash=hash_lease_token(lease_token),
        lease_expires_at__gt=now,
    )


def is_same_successful_completion(*, job: Job, worker_id: str, lease_token: str) -> bool:
    return (
        job.execution_mode == Job.ExecutionMode.REMOTE_PULL
        and job.state == Job.State.SUCCEEDED
        and job.assigned_worker_id == worker_id
        and job.lease_token_hash == hash_lease_token(lease_token)
    )


def ensure_worker_enabled(worker_id: str) -> None:
    worker, _created = Worker.objects.get_or_create(worker_id=worker_id)
    if not worker.enabled:
        raise ApiError(
            "Worker is disabled.",
            status=403,
            error_type="permission_error",
        )


def hash_lease_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def lease_conflict() -> ApiError:
    return ApiError(
        "The job is not leased to this worker with the current lease token.",
        status=409,
        error_type="conflict_error",
    )


def optional_object(value: object, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ApiError(f"{field_name} must be an object.")
    return value


def require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ApiError(f"{field_name} must be a non-empty string.")
    return value.strip()


def optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ApiError("Optional worker fields must be strings when provided.")
    return value.strip() or None


def optional_positive_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ApiError(f"{field_name} must be a positive integer.")
    return value


def normalize_string_list(value: object, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ApiError(f"{field_name} must be a list.")
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ApiError(f"{field_name} entries must be strings.")
        text = item.strip()
        if text:
            normalized.append(text)
    return normalized


def string_set(value: object) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {item.strip() for item in value if isinstance(item, str) and item.strip()}


def require_lease_token(payload: dict[str, Any]) -> str:
    return require_string(payload.get("lease_token"), "lease_token")


def require_result_text(value: object) -> str:
    if not isinstance(value, str):
        raise ApiError("result_text must be a string.")
    return value


def normalize_progress(value: object) -> dict[str, str]:
    if value is None:
        return {}
    progress = optional_object(value, "progress")
    normalized: dict[str, str] = {}
    for key in ("phase", "message"):
        raw = progress.get(key)
        if isinstance(raw, str) and raw.strip():
            normalized[key] = raw.strip()[:500]
    return normalized
