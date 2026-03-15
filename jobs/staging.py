from __future__ import annotations

import uuid
from datetime import timedelta
from pathlib import Path

from django.conf import settings
from django.db.models import Q
from django.utils import timezone

from jobs.artifacts import get_artifact_store
from jobs.media import (
    DownloadedMedia,
    detect_media_suffix,
    is_video_path,
    reserve_temp_media_path,
    write_uploaded_media_to_tempfile,
)
from jobs.models import Job, StagedMedia
from transcriptions.errors import ApiError


def stage_uploaded_audio(*, producer: str, upload) -> StagedMedia:
    cleanup_expired_staged_media()
    upload_size = upload.size or 0
    if upload_size > settings.VOXHELM_BATCH_MAX_STAGED_UPLOAD_BYTES:
        raise ApiError(
            "Uploaded file exceeded the configured "
            f"{settings.VOXHELM_BATCH_MAX_STAGED_UPLOAD_MIB} MiB batch staging limit."
        )

    content_type = (getattr(upload, "content_type", "") or "").lower()
    original_filename = Path(upload.name or "upload").name or "upload"
    suffix = detect_media_suffix(original_filename, content_type)
    if not suffix:
        raise ApiError("Unsupported uploaded media type for batch staging.")
    if is_video_path(Path(f"input{suffix}"), content_type=content_type):
        raise ApiError(
            "Batch staged uploads currently support audio only. "
            "Use URL input for video."
        )

    temp_path = write_uploaded_media_to_tempfile(upload.chunks(), suffix=suffix)
    staged = StagedMedia(
        id=uuid.uuid4(),
        producer=producer,
        original_filename=original_filename,
        content_type=content_type or "application/octet-stream",
        size_bytes=upload_size,
        storage_backend="",
        storage_key="",
        expires_at=timezone.now()
        + timedelta(seconds=settings.VOXHELM_STAGED_INPUT_RETENTION_SECONDS),
    )
    try:
        store = get_artifact_store()
        stored = store.put_file(
            key=build_staged_media_key(staged=staged),
            source_path=temp_path,
            content_type=staged.content_type,
        )
        staged.storage_backend = stored.backend
        staged.storage_key = stored.key
        staged.size_bytes = stored.size_bytes
        staged.save()
        return staged
    finally:
        temp_path.unlink(missing_ok=True)


def cleanup_expired_staged_media(*, exclude_upload_id: str | None = None) -> None:
    now = timezone.now()
    expired = StagedMedia.objects.filter(expires_at__lte=now).filter(
        Q(claimed_by_job__isnull=True) | Q(claimed_by_job__finished_at__isnull=False)
    )
    if exclude_upload_id:
        expired = expired.exclude(id=exclude_upload_id)
    for staged in expired:
        delete_staged_media(staged=staged, missing_ok=True)


def get_staged_media_for_submission(*, producer: str, upload_id: str) -> StagedMedia:
    try:
        staged = StagedMedia.objects.select_for_update().get(id=upload_id, producer=producer)
    except (StagedMedia.DoesNotExist, ValueError) as exc:
        raise ApiError("Unknown input.upload_id.") from exc
    if staged.claimed_by_job_id is not None:
        raise ApiError("input.upload_id has already been attached to a job.")
    if staged.expires_at <= timezone.now():
        raise ApiError("input.upload_id has expired. Stage the media again.")
    return staged


def claim_staged_media_for_job(*, staged: StagedMedia, job: Job) -> None:
    staged.claimed_by_job = job
    staged.claimed_at = timezone.now()
    staged.save(update_fields=["claimed_by_job", "claimed_at"])


def materialize_staged_media(*, staged: StagedMedia) -> DownloadedMedia:
    suffix = detect_media_suffix(staged.original_filename, staged.content_type)
    if not suffix:
        raise RuntimeError("Staged media no longer has a supported media type.")
    if is_video_path(Path(staged.original_filename), content_type=staged.content_type):
        raise RuntimeError("Uploaded video batch input is not implemented yet.")

    target_path = reserve_temp_media_path(suffix=suffix)
    try:
        get_artifact_store().download_file(key=staged.storage_key, destination_path=target_path)
    except Exception as exc:
        target_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to materialize staged input: {exc}") from exc
    return DownloadedMedia(
        path=target_path,
        content_type=staged.content_type,
        source_url="",
        source_name=staged.original_filename,
        source_kind="upload",
    )


def delete_staged_media(*, staged: StagedMedia, missing_ok: bool) -> None:
    try:
        get_artifact_store().delete(key=staged.storage_key)
    except Exception:
        if not missing_ok:
            raise
        return
    staged.delete()


def serialize_staged_media(staged: StagedMedia) -> dict[str, object]:
    return {
        "id": str(staged.id),
        "object": "staged_media",
        "filename": staged.original_filename,
        "content_type": staged.content_type,
        "size_bytes": staged.size_bytes,
        "created_at": staged.created_at.isoformat().replace("+00:00", "Z"),
        "expires_at": staged.expires_at.isoformat().replace("+00:00", "Z"),
    }


def build_staged_media_key(*, staged: StagedMedia) -> str:
    safe_name = staged.original_filename.replace("/", "-")
    prefix = settings.VOXHELM_ARTIFACT_PREFIX.strip("/")
    return f"{prefix}/staged-inputs/{staged.id}/{safe_name}"
