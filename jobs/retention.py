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
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from stat import S_ISREG
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
    refused: list[JobArtifact] = field(default_factory=list)
    pending_deleted: int = 0
    pending_refused: int = 0
    pending_failed: int = 0


class RefusedDeletion(Exception):
    """The object is not one voxhelm generated for the job; it is left in place."""


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
    Filesystem keys resolve like the OS does (``realpath``: symlinks are followed
    before ``..`` is applied, so ``alias/../x`` is the real target; ``a/./b`` and
    ``a/b`` are the same file); S3 keys are literal, the endpoint URL is
    canonicalized.
    """
    resolved = identity or current_artifact_store_identity()
    backend = resolved.get("backend")
    if backend == "filesystem":
        root = resolved_filesystem_root(str(resolved.get("root") or ""))
        return ("filesystem", os.path.realpath(Path(root) / key))
    if backend == "s3":
        return ("s3", canonical_endpoint(resolved.get("endpoint_url")), resolved.get("bucket"), key)
    return (str(backend), repr(sorted(resolved.items())), key)


def job_key_is_deletable(
    *,
    identity: dict[str, Any] | None,
    key: str,
    job_id: object,
    name: str | None = None,
) -> bool:
    """Allowlist: only objects voxhelm itself generated for this job may be deleted.

    The key must end in exactly ``jobs/<job_id>/<file>`` (local execution) or
    ``jobs/<job_id>/attempt-<n>/<file>`` (remote workers) below the artifact
    prefix that was configured when it was written, with no empty, ``.`` or
    ``..`` segments anywhere; ``<file>`` must match the artifact name
    when known. On the filesystem the resolved real path must equal the literal
    path under the resolved root (no symlinks anywhere) and the file must not be
    hard-linked. Anything else is refused and left in place.
    """
    if not key or not job_id:
        return False
    parts = key.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return False
    # The configured prefix may have changed since the object was written, so
    # any leading prefix is accepted; the job-specific tail is what is checked.
    if len(parts) >= 4 and re.fullmatch(r"attempt-[0-9]+", parts[-2]):
        rest = [parts[-4], parts[-3], parts[-1]]
    else:
        rest = parts[-3:]
    if len(rest) != 3 or rest[0] != "jobs" or rest[1] != str(job_id):
        return False
    if name is not None and rest[2] != name.replace("/", "-"):
        return False
    resolved = identity or current_artifact_store_identity()
    backend = resolved.get("backend")
    if backend == "s3":
        return True
    if backend != "filesystem":
        return False
    root = resolved_filesystem_root(str(resolved.get("root") or ""))
    if not root:
        return False
    literal = os.path.join(root, *parts)
    if os.path.realpath(literal) != literal:
        return False
    try:
        stat = os.lstat(literal)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return S_ISREG(stat.st_mode) and stat.st_nlink == 1


def object_is_referenced(
    *,
    identity: dict[str, Any] | None,
    key: str,
    exclude_artifact_pk: UUID | None = None,
) -> bool:
    """True when any artifact row or staged upload still points at this object.

    Every row is resolved before comparing: S3 rows by canonical endpoint,
    bucket and exact key; filesystem rows by real path and, when both files
    exist, by device and inode (the filesystem backend is small-scale, so all
    rows are checked).
    """
    location = object_location(identity=identity, key=key)
    on_filesystem = location[0] == "filesystem"
    target_stat = filesystem_stat(location) if on_filesystem else None
    key_filter = Q() if on_filesystem else Q(storage_key=key)
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
        if not job_key_is_deletable(
            identity=artifact.storage_identity,
            key=artifact.storage_key,
            job_id=artifact.job_id,
            name=artifact.name,
        ):
            raise RefusedDeletion(artifact.storage_key)
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
        for candidate in candidates:
            if job_key_is_deletable(
                identity=candidate.storage_identity,
                key=candidate.storage_key,
                job_id=candidate.job_id,
                name=candidate.name,
            ):
                result.deleted.append(candidate)
            else:
                result.refused.append(candidate)
        for pending in PendingArtifactDeletion.objects.all():
            if job_key_is_deletable(
                identity=pending.storage_identity, key=pending.storage_key, job_id=pending.job_id
            ):
                result.pending_deleted += 1
            else:
                result.pending_refused += 1
        return result
    for candidate in candidates:
        try:
            pruned = prune_artifact(artifact_pk=candidate.pk, now=now)
        except RefusedDeletion:
            logger.warning(
                "Refusing to delete %s artifact %s of job %s: not a voxhelm-generated job key.",
                candidate.kind,
                candidate.storage_key,
                candidate.job_id,
            )
            result.refused.append(candidate)
            continue
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
    counts = drain_pending_artifact_deletions()
    result.pending_deleted = counts["resolved"]
    result.pending_refused = counts["refused"]
    result.pending_failed = counts["failed"]
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
    replaced exposed/final artifact or a new artifact also points at. Keys
    outside the job's generated layout (``job_key_is_deletable``) are never
    queued.
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
    # Filesystem aliases (case variants, symlinked directories) of a protected
    # file are the same object; compare them by device and inode as well.
    protected_stats = [
        stat
        for stat in (
            filesystem_stat(location) for location in protected if location[0] == "filesystem"
        )
        if stat is not None
    ]
    queued: list[PendingArtifactDeletion] = []
    for artifact in replaced:
        if artifact.exposed or artifact.kind not in PRUNABLE_ARTIFACT_KINDS:
            continue
        if not job_key_is_deletable(
            identity=artifact.storage_identity,
            key=artifact.storage_key,
            job_id=artifact.job_id,
            name=artifact.name,
        ):
            continue
        location = object_location(identity=artifact.storage_identity, key=artifact.storage_key)
        if location in protected:
            continue
        if location[0] == "filesystem":
            stat = filesystem_stat(location)
            if stat is not None and any(
                os.path.samestat(stat, protected_stat) for protected_stat in protected_stats
            ):
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


def process_pending_artifact_deletion(pending_pk: int) -> str:
    """Handle one queued deletion; returns "resolved", "refused" or "failed".

    A refused entry (not an allowlisted job key) is kept in the queue and the
    object is left in place.
    """
    try:
        with transaction.atomic():
            acquire_sqlite_write_lock()
            pending = (
                PendingArtifactDeletion.objects.select_for_update().filter(pk=pending_pk).first()
            )
            if pending is None:
                return "resolved"
            if not job_key_is_deletable(
                identity=pending.storage_identity, key=pending.storage_key, job_id=pending.job_id
            ):
                logger.warning(
                    "Refusing queued deletion of %s: not a voxhelm-generated job key.",
                    pending.storage_key,
                )
                return "refused"
            if not object_is_referenced(identity=pending.storage_identity, key=pending.storage_key):
                delete_stored_object(identity=pending.storage_identity, key=pending.storage_key)
            pending.delete()
    except Exception:
        logger.exception("Could not delete queued artifact object %s; will retry.", pending_pk)
        return "failed"
    return "resolved"


def drain_pending_artifact_deletions(
    pending: Iterable[PendingArtifactDeletion] | None = None,
) -> dict[str, int]:
    """Process queued deletions; returns counts per outcome."""
    if pending is None:
        pending_pks = list(PendingArtifactDeletion.objects.values_list("pk", flat=True))
    else:
        pending_pks = [item.pk for item in pending]
    counts = {"resolved": 0, "refused": 0, "failed": 0}
    for pending_pk in pending_pks:
        counts[process_pending_artifact_deletion(pending_pk)] += 1
    return counts
