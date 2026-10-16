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

Jobs of a transcription profile with an artifact backstop (``ad-analysis-v1``: 7 days)
lose every artifact, of any kind and exposed or not, and then their row once they
finished longer ago than the profile's retention, whether or not the producer deleted
the job itself (``delete_job_for_producer``, ``DELETE /v1/jobs/{id}``). The row goes
too because it carries the transcript text. Both paths also clean the job's recorded
remote attempt prefixes (``RemoteAttemptPrefix``: objects no completion registered),
delete a staged upload it still holds (allowlisted, reference-checked, under the lock)
and drop its ``DBTaskResult``.

Job metadata (the ``Job`` row) is deleted after
``VOXHELM_JOB_METADATA_RETENTION_SECONDS`` (default 90 days), but only for terminal
jobs that no longer own anything: no artifact row of any kind (intermediates are
pruned first; transcript and speech artifacts are kept indefinitely, so a job that
still has them keeps its row), no queued object deletion and no claimed staged
upload. See ``expired_job_metadata``.
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
from django.db.models import Exists, OuterRef, Q, QuerySet
from django.db.models.functions import Coalesce
from django.utils import timezone
from django_tasks_db.models import DBTaskResult

from jobs.artifacts import current_artifact_store_identity, get_artifact_store_for_identity
from jobs.models import (
    Job,
    JobArtifact,
    PendingArtifactDeletion,
    RemoteAttemptPrefix,
    StagedMedia,
)
from jobs.services import acquire_sqlite_write_lock, reconcile_job_state
from transcriptions.profiles import PROFILES

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
    attempt_prefixes_deleted: int = 0
    attempt_prefixes_refused: int = 0
    attempt_prefixes_failed: int = 0
    staged_deleted: int = 0
    staged_refused: int = 0
    staged_failed: int = 0
    jobs_deleted: list[Job] = field(default_factory=list)


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
    return stored_object_is_plain(identity=identity, parts=parts)


def staged_key_is_deletable(
    *,
    identity: dict[str, Any] | None,
    key: str,
    staged_id: object,
    filename: str,
) -> bool:
    """Allowlist for a staged upload's object, the counterpart of ``job_key_is_deletable``.

    The key must end in exactly ``staged-inputs/<upload_id>/<filename>`` (the layout of
    ``build_staged_media_key``) below any prefix, with no empty, ``.`` or ``..``
    segments, and pass the same filesystem checks.
    """
    if not key or not staged_id:
        return False
    parts = key.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return False
    if len(parts) < 3 or parts[-3:] != [
        "staged-inputs",
        str(staged_id),
        filename.replace("/", "-"),
    ]:
        return False
    return stored_object_is_plain(identity=identity, parts=parts)


def attempt_prefix_is_deletable(*, prefix: str, job_id: object, attempt: int) -> bool:
    """A recorded attempt prefix must end in exactly ``jobs/<job_id>/attempt-<n>/``."""
    parts = prefix.rstrip("/").split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return False
    return len(parts) >= 3 and parts[-3:] == ["jobs", str(job_id), f"attempt-{attempt}"]


def stored_object_is_plain(*, identity: dict[str, Any] | None, parts: list[str]) -> bool:
    """S3 objects are literal; a filesystem object must be a plain file under the root.

    On the filesystem the resolved real path must equal the literal path under the
    resolved root (no symlinks anywhere) and the file must not be hard-linked.
    """
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
    exclude_staged_pk: UUID | None = None,
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
    staged: QuerySet[StagedMedia] = StagedMedia.objects.filter(key_filter)
    if exclude_staged_pk is not None:
        staged = staged.exclude(pk=exclude_staged_pk)
    references.extend(staged.values_list("storage_identity", "storage_key"))
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


def profile_backstop_filter(*, now: datetime, field_prefix: str = "") -> Q:
    """Jobs of a profile with an artifact backstop that finished longer ago than it.

    ``field_prefix`` is ``"job__"`` when filtering artifacts by their job. The caller
    annotates ``job_finished_at``.
    """
    condition = Q(pk__in=[])
    for name, profile in PROFILES.items():
        if profile.artifact_retention_seconds <= 0:
            continue
        cutoff = now - timedelta(seconds=profile.artifact_retention_seconds)
        condition |= Q(**{f"{field_prefix}profile": name}, job_finished_at__lte=cutoff)
    return condition


def expired_intermediate_artifacts(*, now: datetime | None = None) -> QuerySet[JobArtifact]:
    """The prunable artifacts that are past retention, oldest job first.

    Non-exposed intermediates per D-09, plus every artifact of a profile job past the
    profile's backstop.
    """
    now = now or timezone.now()
    source_cutoff = now - timedelta(seconds=settings.VOXHELM_SOURCE_ARTIFACT_RETENTION_SECONDS)
    intermediate = Q(exposed=False, kind__in=PRUNABLE_ARTIFACT_KINDS) & (
        Q(kind=JobArtifact.Kind.EXTRACTED_AUDIO)
        | Q(kind=JobArtifact.Kind.SOURCE, job_finished_at__lte=source_cutoff)
    )
    return (
        JobArtifact.objects.filter(job__state__in=TERMINAL_JOB_STATES)
        .annotate(job_finished_at=Coalesce("job__finished_at", "job__updated_at"))
        .filter(intermediate | profile_backstop_filter(now=now, field_prefix="job__"))
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


def expired_profile_jobs(*, now: datetime | None = None) -> QuerySet[Job]:
    """Terminal profile jobs that finished longer ago than their profile's backstop."""
    now = now or timezone.now()
    return (
        Job.objects.filter(state__in=TERMINAL_JOB_STATES)
        .annotate(job_finished_at=Coalesce("finished_at", "updated_at"))
        .filter(profile_backstop_filter(now=now))
    )


def attempt_prefix_keys(record: RemoteAttemptPrefix) -> list[str]:
    """Every stored object below a recorded attempt prefix.

    Raises ``RefusedDeletion`` when the prefix or any object below it is outside the
    job's generated layout (``attempt_prefix_is_deletable``, ``job_key_is_deletable``).
    """
    if not attempt_prefix_is_deletable(
        prefix=record.prefix, job_id=record.job_id, attempt=record.attempt
    ):
        raise RefusedDeletion(record.prefix)
    prefix = record.prefix.rstrip("/") + "/"
    try:
        keys = get_artifact_store_for_identity(record.storage_identity).list_keys(prefix=prefix)
    except Exception as error:  # noqa: BLE001 - any listing failure (filesystem or S3)
        # A prefix that cannot be listed completely keeps its record for a retry.
        raise RefusedDeletion(prefix) from error
    for key in keys:
        if not key.startswith(prefix) or not job_key_is_deletable(
            identity=record.storage_identity, key=key, job_id=record.job_id
        ):
            raise RefusedDeletion(key)
    return keys


def clean_attempt_prefix(*, record_pk: int) -> None:
    """Delete what one finished attempt stored below its prefix, then the record.

    Covers objects no completion registered (a rejected or failed completion, a retried
    or abandoned attempt). Under the write lock like every other deletion; an object an
    artifact row or staged upload still references is kept. The record of a job that is
    not terminal is left alone.
    """
    with transaction.atomic():
        acquire_sqlite_write_lock()
        record = RemoteAttemptPrefix.objects.select_for_update().filter(pk=record_pk).first()
        if record is None:
            return
        if Job.objects.filter(id=record.job_id).exclude(state__in=TERMINAL_JOB_STATES).exists():
            return
        for key in attempt_prefix_keys(record):
            if not object_is_referenced(identity=record.storage_identity, key=key):
                delete_stored_object(identity=record.storage_identity, key=key)
        record.delete()


def delete_claimed_staged_upload(*, staged_pk: UUID, job_id: UUID) -> None:
    """Delete a staged upload a finished job still holds: object first, then the row.

    Same rules as artifacts: under the write lock, allowlisted keys only
    (``staged_key_is_deletable``, else ``RefusedDeletion``), and an object another row
    still references is kept (only the row goes).
    """
    with transaction.atomic():
        acquire_sqlite_write_lock()
        staged = (
            StagedMedia.objects.select_for_update()
            .filter(pk=staged_pk, claimed_by_job_id=job_id)
            .first()
        )
        if staged is None:
            return
        if not staged_key_is_deletable(
            identity=staged.storage_identity,
            key=staged.storage_key,
            staged_id=staged.id,
            filename=staged.original_filename,
        ):
            raise RefusedDeletion(staged.storage_key)
        if not object_is_referenced(
            identity=staged.storage_identity,
            key=staged.storage_key,
            exclude_staged_pk=staged.pk,
        ):
            delete_stored_object(identity=staged.storage_identity, key=staged.storage_key)
        staged.delete()


def delete_profile_task_results(jobs: Iterable[Job]) -> None:
    """Drop the task results of profile jobs whose rows are being deleted.

    ``django_tasks_db`` keeps a task's return value, which before the profile returned
    only its id held the complete transcript. Runs inside the caller's transaction.
    """
    task_ids: list[UUID] = []
    for job in jobs:
        if not job.profile or not job.django_task_id:
            continue
        try:
            task_ids.append(UUID(job.django_task_id))
        except ValueError:
            continue
    if task_ids:
        DBTaskResult.objects.filter(id__in=task_ids).delete()


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
        deletable_pending_pks: list[int] = []
        for pending in PendingArtifactDeletion.objects.all():
            if job_key_is_deletable(
                identity=pending.storage_identity, key=pending.storage_key, job_id=pending.job_id
            ):
                result.pending_deleted += 1
                deletable_pending_pks.append(pending.pk)
            else:
                result.pending_refused += 1
        deletable_prefix_pks: list[int] = []
        for record in RemoteAttemptPrefix.objects.filter(
            job_id__in=expired_profile_jobs(now=now).values("id")
        ):
            # The same listing and object checks the real run applies.
            try:
                attempt_prefix_keys(record)
            except RefusedDeletion:
                result.attempt_prefixes_refused += 1
            else:
                result.attempt_prefixes_deleted += 1
                deletable_prefix_pks.append(record.pk)
        deletable_staged_pks: list[UUID] = []
        for staged in StagedMedia.objects.filter(
            claimed_by_job__in=expired_profile_jobs(now=now).values("id")
        ):
            if staged_key_is_deletable(
                identity=staged.storage_identity,
                key=staged.storage_key,
                staged_id=staged.id,
                filename=staged.original_filename,
            ):
                result.staged_deleted += 1
                deletable_staged_pks.append(staged.pk)
            else:
                result.staged_refused += 1
        result.jobs_deleted = list(
            expired_job_metadata(
                now=now,
                ignore_artifact_pks=[artifact.pk for artifact in result.deleted],
                ignore_pending_pks=deletable_pending_pks,
                ignore_attempt_prefix_pks=deletable_prefix_pks,
                ignore_staged_pks=deletable_staged_pks,
            )
        )
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
    prune_profile_leftovers(result=result, now=now)
    # Job rows go last, so a job whose intermediates were just pruned is eligible
    # in the same run.
    result.jobs_deleted = prune_job_metadata(now=now)
    return result


def prune_profile_leftovers(*, result: PruneResult, now: datetime | None) -> None:
    """The profile backstop's cleanup beyond artifact rows, before the job rows go.

    Objects attempts stored without a completion registering them, and staged uploads a
    completion could not delete (``delete_completed_remote_staged_upload`` keeps the row
    when the object deletion fails), of every profile job past its backstop.
    """
    expired_jobs = expired_profile_jobs(now=now).values("id")
    for record in list(RemoteAttemptPrefix.objects.filter(job_id__in=expired_jobs)):
        try:
            clean_attempt_prefix(record_pk=record.pk)
        except RefusedDeletion as exc:
            logger.warning(
                "Refusing to clean attempt prefix %s of job %s: %s is not a "
                "voxhelm-generated job key.",
                record.prefix,
                record.job_id,
                exc,
            )
            result.attempt_prefixes_refused += 1
        except Exception:
            logger.exception(
                "Could not clean attempt prefix %s of job %s; will retry.",
                record.prefix,
                record.job_id,
            )
            result.attempt_prefixes_failed += 1
        else:
            result.attempt_prefixes_deleted += 1
    for staged in list(StagedMedia.objects.filter(claimed_by_job__in=expired_jobs)):
        assert staged.claimed_by_job_id is not None
        try:
            delete_claimed_staged_upload(staged_pk=staged.pk, job_id=staged.claimed_by_job_id)
        except RefusedDeletion:
            logger.warning(
                "Refusing to delete staged upload %s of job %s: not a voxhelm staged key.",
                staged.storage_key,
                staged.claimed_by_job_id,
            )
            result.staged_refused += 1
        except Exception:
            logger.exception(
                "Could not delete staged upload %s of job %s; will retry.",
                staged.storage_key,
                staged.claimed_by_job_id,
            )
            result.staged_failed += 1
        else:
            result.staged_deleted += 1


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


def expired_job_metadata(
    *,
    now: datetime | None = None,
    ignore_artifact_pks: Iterable[UUID] = (),
    ignore_pending_pks: Iterable[int] = (),
    ignore_attempt_prefix_pks: Iterable[int] = (),
    ignore_staged_pks: Iterable[UUID] = (),
) -> QuerySet[Job]:
    """Terminal jobs past metadata retention that own nothing anymore, oldest first.

    A job is only eligible when it finished (``finished_at``, else ``updated_at``)
    more than ``VOXHELM_JOB_METADATA_RETENTION_SECONDS`` ago (a profile job: more than
    its profile's backstop ago, even when metadata retention is disabled), is in a terminal
    state and has no remaining ``JobArtifact`` row (intermediates are pruned first;
    transcript, speech and exposed artifacts are kept indefinitely and keep their
    job), no queued ``PendingArtifactDeletion``, no ``RemoteAttemptPrefix`` left to
    clean and no ``StagedMedia`` still claimed by it. A retention of ``0`` disables job
    metadata pruning.

    ``ignore_*_pks`` let ``--dry-run`` treat artifacts, queued deletions, attempt
    prefixes and staged uploads it would remove in the same run as already gone.
    """
    retention = settings.VOXHELM_JOB_METADATA_RETENTION_SECONDS
    now = now or timezone.now()
    age_filter = profile_backstop_filter(now=now)
    if retention > 0:
        age_filter |= Q(job_finished_at__lte=now - timedelta(seconds=retention))
    artifacts = JobArtifact.objects.filter(job=OuterRef("pk")).exclude(
        pk__in=list(ignore_artifact_pks)
    )
    pending = PendingArtifactDeletion.objects.filter(job_id=OuterRef("pk")).exclude(
        pk__in=list(ignore_pending_pks)
    )
    attempt_prefixes = RemoteAttemptPrefix.objects.filter(job_id=OuterRef("pk")).exclude(
        pk__in=list(ignore_attempt_prefix_pks)
    )
    staged = StagedMedia.objects.filter(claimed_by_job=OuterRef("pk")).exclude(
        pk__in=list(ignore_staged_pks)
    )
    return (
        Job.objects.filter(state__in=TERMINAL_JOB_STATES)
        .annotate(job_finished_at=Coalesce("finished_at", "updated_at"))
        .filter(age_filter)
        .filter(~Exists(artifacts), ~Exists(pending), ~Exists(attempt_prefixes), ~Exists(staged))
        .order_by("job_finished_at", "id")
    )


def prune_job_metadata(*, now: datetime | None = None, batch_size: int = 500) -> list[Job]:
    """Delete expired job rows (D-09); returns the deleted jobs.

    Each batch re-evaluates eligibility under the database write lock inside its
    own transaction, so a job that gained an artifact, a queued deletion or a
    staged-upload claim meanwhile, or left its terminal state, is kept.
    """
    deleted: list[Job] = []
    while True:
        with transaction.atomic():
            acquire_sqlite_write_lock()
            batch = list(expired_job_metadata(now=now).select_for_update()[:batch_size])
            if not batch:
                return deleted
            delete_profile_task_results(batch)
            Job.objects.filter(pk__in=[job.pk for job in batch]).delete()
        deleted.extend(batch)
        if len(batch) < batch_size:
            return deleted


class JobNotDeletable(Exception):
    """The job cannot be deleted by its producer (not a profile job, or not finished)."""


def delete_job_artifact(*, artifact_pk: UUID) -> None:
    """Delete one artifact of a job being deleted: object first, then the row.

    Same rules as ``prune_artifact``: under the write lock, allowlisted keys only, and an
    object another row still references is kept (only the row goes).
    """
    with transaction.atomic():
        acquire_sqlite_write_lock()
        artifact = JobArtifact.objects.select_for_update().filter(pk=artifact_pk).first()
        if artifact is None:
            return
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
        artifact.delete()


def delete_job_for_producer(*, job_id: UUID, producer: str) -> None:
    """Delete a finished profile job: its artifacts and their objects, then its row.

    Only the producer that submitted the job reaches it (``Job.DoesNotExist`` otherwise).
    Deletion is all-or-nothing up front (any artifact outside the allowlist refuses the
    whole request, nothing is deleted) and resumable afterwards: each artifact goes in its
    own transaction, so a failed object deletion raises with the remaining artifacts and
    the row kept, and the next call continues. Queued object deletions of the job, the
    objects its remote attempts stored without a completion registering them, a staged
    upload it still holds (under the same allowlist, reference check and lock) and its
    task result are removed as well.
    """
    job = Job.objects.get(id=job_id, producer=producer)
    if job.profile not in PROFILES:
        raise JobNotDeletable("Only profile jobs can be deleted through the API.")
    reconcile_job_state(job)
    if job.state not in TERMINAL_JOB_STATES:
        raise JobNotDeletable("The job has not finished yet; delete it once it has.")
    artifacts = list(JobArtifact.objects.filter(job=job))
    for artifact in artifacts:
        if not job_key_is_deletable(
            identity=artifact.storage_identity,
            key=artifact.storage_key,
            job_id=artifact.job_id,
            name=artifact.name,
        ):
            raise RefusedDeletion(artifact.storage_key)
    attempt_prefixes = list(RemoteAttemptPrefix.objects.filter(job_id=job.id))
    for record in attempt_prefixes:
        attempt_prefix_keys(record)
    staged_uploads = list(StagedMedia.objects.filter(claimed_by_job=job))
    for staged in staged_uploads:
        if not staged_key_is_deletable(
            identity=staged.storage_identity,
            key=staged.storage_key,
            staged_id=staged.id,
            filename=staged.original_filename,
        ):
            raise RefusedDeletion(staged.storage_key)
    for artifact in artifacts:
        delete_job_artifact(artifact_pk=artifact.pk)
    counts = drain_pending_artifact_deletions(PendingArtifactDeletion.objects.filter(job_id=job.id))
    if counts["refused"] or counts["failed"]:
        raise RuntimeError("A queued object deletion of the job did not complete.")
    for record in attempt_prefixes:
        clean_attempt_prefix(record_pk=record.pk)
    for staged in staged_uploads:
        delete_claimed_staged_upload(staged_pk=staged.pk, job_id=job.id)
    with transaction.atomic():
        acquire_sqlite_write_lock()
        remaining = (
            JobArtifact.objects.filter(job_id=job.id).exists()
            or PendingArtifactDeletion.objects.filter(job_id=job.id).exists()
            or RemoteAttemptPrefix.objects.filter(job_id=job.id).exists()
            or StagedMedia.objects.filter(claimed_by_job_id=job.id).exists()
        )
        if remaining:
            raise RuntimeError("The job gained artifacts while it was being deleted.")
        delete_profile_task_results([job])
        Job.objects.filter(pk=job.pk, state__in=TERMINAL_JOB_STATES).delete()
