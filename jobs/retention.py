"""Retention cleanup for job-owned intermediate artifacts (decision D-09).

D-09 keeps transcript and speech artifacts indefinitely, keeps downloaded source
media for a limited time and drops extracted intermediate audio once the job is
done. Only non-exposed ``SOURCE`` and ``EXTRACTED_AUDIO`` artifacts of terminal
jobs are ever pruned here; exposed artifacts and every other kind are left alone,
and an object that any other row (artifact or staged upload) still references is
never deleted.

Each deletion runs in its own short transaction that first takes the database
write lock, re-checks that the candidate still exists and that nothing else
references its object, deletes the object and only then drops the row. Overlapping
prune runs and post-completion drains therefore serialize per object.
"""

from __future__ import annotations

import logging
import os
import posixpath
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import UUID

from django.conf import settings
from django.db import transaction
from django.db.models import Q, QuerySet
from django.db.models.functions import Coalesce
from django.utils import timezone

from jobs.artifacts import current_artifact_store_identity, get_artifact_store_for_identity
from jobs.models import Job, JobArtifact, PendingArtifactDeletion, StagedMedia
from jobs.services import acquire_sqlite_write_lock

logger = logging.getLogger(__name__)

PRUNABLE_ARTIFACT_KINDS: frozenset[str] = frozenset(
    {JobArtifact.Kind.SOURCE, JobArtifact.Kind.EXTRACTED_AUDIO}
)
TERMINAL_JOB_STATES: frozenset[str] = frozenset(
    {Job.State.SUCCEEDED, Job.State.FAILED, Job.State.CANCELED, Job.State.EXPIRED}
)
DEFAULT_PORTS = {"http": 80, "https": 443}

ObjectLocation = tuple[Any, ...]


@dataclass
class PruneResult:
    deleted: list[JobArtifact] = field(default_factory=list)
    failed: list[JobArtifact] = field(default_factory=list)
    pending_deleted: int = 0
    pending_failed: int = 0


@lru_cache(maxsize=64)
def resolved_filesystem_root(root: str) -> str:
    return str(Path(root).expanduser().resolve()) if root else ""


def canonical_endpoint(endpoint_url: object) -> str:
    """Canonicalize an S3 endpoint URL so equivalent spellings compare equal."""
    raw = str(endpoint_url or "").strip()
    try:
        parts = urlsplit(raw)
        port = parts.port
    except ValueError:
        return raw.lower().rstrip("/")
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    if port is not None and port != DEFAULT_PORTS.get(scheme):
        host = f"{host}:{port}"
    return f"{scheme}://{host}{parts.path.rstrip('/')}"


def object_location(*, identity: dict[str, Any] | None, key: str) -> ObjectLocation:
    """Resolve where an object physically lives.

    An empty identity means "the current store" (see
    ``get_artifact_store_for_identity``), so it is resolved the same way here.
    Filesystem keys resolve to a normalized path under the root (``a/./b`` and
    ``a/b`` are the same file); S3 keys are literal, the endpoint URL is
    canonicalized.
    """
    resolved = identity or current_artifact_store_identity()
    backend = resolved.get("backend")
    if backend == "filesystem":
        root = resolved_filesystem_root(str(resolved.get("root") or ""))
        return ("filesystem", os.path.normpath(str(Path(root) / key)))
    if backend == "s3":
        return ("s3", canonical_endpoint(resolved.get("endpoint_url")), resolved.get("bucket"), key)
    return (str(backend), repr(sorted(resolved.items())), key)


def possibly_equivalent_key_filter(key: str) -> Q:
    """Narrow DB filter that matches every key that may resolve to the same object.

    Any key normalizing to the same path ends with the same final component, or
    ends in ``.``/``/`` (``a/b/.``, ``a/c/..``, ``a/b/``). Exact matching is done
    afterwards with ``object_location``.
    """
    basename = posixpath.basename(posixpath.normpath(key)) or key
    return (
        Q(storage_key__endswith=basename)
        | Q(storage_key__endswith=".")
        | Q(storage_key__endswith="/")
    )


def object_is_referenced(
    *,
    identity: dict[str, Any] | None,
    key: str,
    exclude_artifact_pk: UUID | None = None,
) -> bool:
    """True when any artifact row or staged upload still points at this object.

    S3 keys are matched through a narrow key filter. Filesystem objects are
    compared against every row by normalized path and, when both files exist,
    by device and inode, so case-insensitive spellings and symlinked aliases of
    the same file count as references (the filesystem backend is small-scale).
    """
    location = object_location(identity=identity, key=key)
    on_filesystem = location[0] == "filesystem"
    target_stat = filesystem_stat(location) if on_filesystem else None
    key_filter = Q() if on_filesystem else possibly_equivalent_key_filter(key)
    artifacts: QuerySet[JobArtifact] = JobArtifact.objects.filter(key_filter)
    if exclude_artifact_pk is not None:
        artifacts = artifacts.exclude(pk=exclude_artifact_pk)
    references = list(artifacts.values_list("storage_identity", "storage_key"))
    references.extend(
        StagedMedia.objects.filter(key_filter).values_list("storage_identity", "storage_key")
    )
    for other_identity, other_key in references:
        if not other_key:
            continue
        other_location = object_location(identity=other_identity, key=other_key)
        if other_location == location:
            return True
        if target_stat is not None and other_location[0] == "filesystem":
            other_stat = filesystem_stat(other_location)
            if other_stat is not None and os.path.samestat(target_stat, other_stat):
                return True
    return False


def filesystem_stat(location: ObjectLocation) -> os.stat_result | None:
    try:
        return os.stat(location[1])
    except OSError:
        return None


def delete_stored_object(*, identity: dict[str, Any] | None, key: str) -> None:
    """Delete one stored object; a missing object counts as deleted."""
    if not key:
        return
    store = get_artifact_store_for_identity(identity)
    try:
        store.delete(key=key)
    except FileNotFoundError:
        return


def expired_intermediate_artifacts(*, now: datetime | None = None) -> QuerySet[JobArtifact]:
    """The prunable artifacts that are past retention, oldest job first."""
    now = now or timezone.now()
    source_cutoff = now - timedelta(seconds=settings.VOXHELM_SOURCE_ARTIFACT_RETENTION_SECONDS)
    return (
        JobArtifact.objects.filter(
            exposed=False,
            kind__in=PRUNABLE_ARTIFACT_KINDS,
            job__state__in=TERMINAL_JOB_STATES,
        )
        .annotate(job_finished_at=Coalesce("job__finished_at", "job__updated_at"))
        .filter(
            Q(kind=JobArtifact.Kind.EXTRACTED_AUDIO)
            | Q(kind=JobArtifact.Kind.SOURCE, job_finished_at__lte=source_cutoff)
        )
        .order_by("job_finished_at", "job_id", "name")
    )


def prune_artifact(*, artifact_pk: UUID, now: datetime | None) -> JobArtifact | None:
    """Delete one expired artifact under the write lock; None if it is already gone."""
    with transaction.atomic():
        acquire_sqlite_write_lock()
        artifact = (
            expired_intermediate_artifacts(now=now)
            .select_for_update()
            .filter(pk=artifact_pk)
            .first()
        )
        if artifact is None:
            return None
        if not object_is_referenced(
            identity=artifact.storage_identity,
            key=artifact.storage_key,
            exclude_artifact_pk=artifact.pk,
        ):
            delete_stored_object(identity=artifact.storage_identity, key=artifact.storage_key)
        JobArtifact.objects.filter(pk=artifact.pk).delete()
        return artifact


def prune_job_artifacts(*, now: datetime | None = None, dry_run: bool = False) -> PruneResult:
    """Delete expired intermediate artifacts: the stored object first, then the row.

    A failed object deletion keeps the row so the next run retries it. Pending
    deletions queued by remote completions are drained as well.
    """
    result = PruneResult()
    candidates = list(expired_intermediate_artifacts(now=now))
    if dry_run:
        result.deleted = candidates
        result.pending_deleted = PendingArtifactDeletion.objects.count()
        return result
    for candidate in candidates:
        try:
            pruned = prune_artifact(artifact_pk=candidate.pk, now=now)
        except Exception:
            logger.exception(
                "Could not delete %s artifact %s of job %s; keeping the row for the next run.",
                candidate.kind,
                candidate.storage_key,
                candidate.job_id,
            )
            result.failed.append(candidate)
            continue
        if pruned is not None:
            result.deleted.append(pruned)
    result.pending_deleted, result.pending_failed = drain_pending_artifact_deletions()
    return result


def queue_replaced_intermediate_objects(
    *,
    replaced: Iterable[JobArtifact],
    kept: Iterable[dict[str, Any]],
) -> list[PendingArtifactDeletion]:
    """Queue stored objects of replaced intermediate rows for deletion.

    Called inside the transaction that swaps a job's artifact rows, so the queue
    entry is durable together with the row removal. Only non-exposed
    ``SOURCE``/``EXTRACTED_AUDIO`` objects are queued, and never one that a
    replaced exposed/final artifact or a new artifact also points at.
    """
    replaced = list(replaced)
    protected: set[ObjectLocation] = {
        object_location(identity=artifact["storage_identity"], key=str(artifact["storage_key"]))
        for artifact in kept
    }
    protected.update(
        object_location(identity=artifact.storage_identity, key=artifact.storage_key)
        for artifact in replaced
        if artifact.exposed or artifact.kind not in PRUNABLE_ARTIFACT_KINDS
    )
    queued: list[PendingArtifactDeletion] = []
    for artifact in replaced:
        if artifact.exposed or artifact.kind not in PRUNABLE_ARTIFACT_KINDS:
            continue
        location = object_location(identity=artifact.storage_identity, key=artifact.storage_key)
        if location in protected:
            continue
        queued.append(
            PendingArtifactDeletion.objects.create(
                job_id=artifact.job_id,
                kind=artifact.kind,
                storage_backend=artifact.storage_backend,
                storage_key=artifact.storage_key,
                # Pin the resolved store so a later store change cannot redirect
                # the retry at a different backend or root.
                storage_identity=artifact.storage_identity or current_artifact_store_identity(),
            )
        )
    return queued


def process_pending_artifact_deletion(pending_pk: int) -> bool:
    """Delete a queued object unless it is referenced again; True when resolved."""
    try:
        with transaction.atomic():
            acquire_sqlite_write_lock()
            pending = (
                PendingArtifactDeletion.objects.select_for_update().filter(pk=pending_pk).first()
            )
            if pending is None:
                return True
            if not object_is_referenced(identity=pending.storage_identity, key=pending.storage_key):
                delete_stored_object(identity=pending.storage_identity, key=pending.storage_key)
            pending.delete()
    except Exception:
        logger.exception("Could not delete queued artifact object %s; will retry.", pending_pk)
        return False
    return True


def drain_pending_artifact_deletions(
    pending: Iterable[PendingArtifactDeletion] | None = None,
) -> tuple[int, int]:
    """Process queued deletions; returns (resolved, failed)."""
    if pending is None:
        pending_pks = list(PendingArtifactDeletion.objects.values_list("pk", flat=True))
    else:
        pending_pks = [item.pk for item in pending]
    resolved = 0
    failed = 0
    for pending_pk in pending_pks:
        if process_pending_artifact_deletion(pending_pk):
            resolved += 1
        else:
            failed += 1
    return resolved, failed
