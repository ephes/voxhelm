from datetime import timedelta
from uuid import UUID

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.db import connection
from django.test.utils import CaptureQueriesContext
from django.utils import timezone

from jobs import retention, staging
from jobs.models import Job, JobArtifact, PendingArtifactDeletion, StagedMedia


@pytest.mark.django_db
@pytest.mark.parametrize("size", [1, 40])
def test_prune_has_bounded_selection_and_resumes(size, settings, django_assert_num_queries):
    settings.VOXHELM_SOURCE_ARTIFACT_RETENTION_SECONDS = 0
    job = Job.objects.create(
        producer="synthetic",
        job_type="transcribe",
        input_data={},
        state="succeeded",
        finished_at=timezone.now(),
    )
    JobArtifact.objects.bulk_create(
        [
            JobArtifact(
                job=job,
                name=f"input-{n}.mp3",
                kind="source",
                exposed=False,
                storage_backend="filesystem",
                storage_key=f"voxhelm/jobs/{job.pk}/input-{n}.mp3",
            )
            for n in range(size)
        ]
    )
    with django_assert_num_queries(5), CaptureQueriesContext(connection) as queries:
        result = retention.prune_job_artifacts(batch_size=3, dry_run=True)
    assert len(result.deleted) == min(size, 3)
    selection = [q["sql"] for q in queries if q["sql"].startswith('SELECT "jobs_jobartifact".')]
    assert selection and all("LIMIT 4" in q for q in selection)
    cursor = None
    seen: list[UUID] = []
    while True:
        result = retention.prune_job_artifacts(batch_size=3, cursor=cursor)
        assert len(result.deleted) <= 3
        seen.extend(item.pk for item in result.deleted)
        cursor = result.next_cursor
        if not cursor:
            break
    assert len(seen) == size and len(set(seen)) == size
    assert not JobArtifact.objects.exists()


@pytest.mark.django_db
def test_failed_or_refused_pending_rows_do_not_starve_later_rows(monkeypatch):
    from uuid import uuid4

    rows = [
        PendingArtifactDeletion.objects.create(
            job_id=uuid4(),
            kind="source",
            storage_backend="filesystem",
            storage_key="unsafe" if n == 0 else f"key-{n}",
        )
        for n in range(7)
    ]
    visited = []

    def process(pk):
        visited.append(pk)
        if pk == rows[0].pk:
            return "refused"
        if pk == rows[1].pk:
            return "failed"
        PendingArtifactDeletion.objects.filter(pk=pk).delete()
        return "resolved"

    monkeypatch.setattr(retention, "process_pending_artifact_deletion", process)
    cursor = None
    for _ in range(4):
        result = retention.prune_job_artifacts(batch_size=2, cursor=cursor)
        cursor = result.next_cursor
        if not cursor:
            break
    assert visited == [row.pk for row in rows]
    assert PendingArtifactDeletion.objects.count() == 2
    retention.prune_job_artifacts(batch_size=2)
    assert visited[-2:] == [row.pk for row in rows[:2]]


@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize("size", [1, 1000])
def test_admission_deletes_only_small_batch(size, monkeypatch):
    expired = timezone.now() - timedelta(days=1)
    StagedMedia.objects.bulk_create(
        [
            StagedMedia(
                producer="synthetic",
                original_filename="input.mp3",
                storage_key=f"expired-{n}",
                storage_backend="filesystem",
                expires_at=expired,
                size_bytes=1,
                content_type="audio/mpeg",
            )
            for n in range(size)
        ]
    )
    deleted = []

    def delete(*, staged, missing_ok):
        deleted.append(staged.pk)
        staged.delete()

    monkeypatch.setattr(staging, "delete_staged_media", delete)
    staged = staging.stage_uploaded_audio(
        producer="synthetic",
        upload=SimpleUploadedFile("new.mp3", b"audio", content_type="audio/mpeg"),
    )
    assert len(deleted) == min(size, 4)
    assert StagedMedia.objects.filter(expires_at=expired).count() == max(size - 4, 0)
    assert StagedMedia.objects.filter(pk=staged.pk).exists()


@pytest.mark.django_db
def test_all_maintenance_phases_are_bounded(settings):
    from jobs.models import RemoteAttemptPrefix

    job = Job.objects.create(
        producer="synthetic",
        job_type="transcribe",
        input_data={},
        profile="ad-analysis-v1",
        state="succeeded",
        finished_at=timezone.now() - timedelta(days=100),
    )
    for n in range(7):
        RemoteAttemptPrefix.objects.create(
            job_id=job.pk, attempt=n + 1, prefix=f"voxhelm/jobs/{job.pk}/attempt-{n + 1}/"
        )
        PendingArtifactDeletion.objects.create(
            job_id=job.pk,
            kind="source",
            storage_backend="filesystem",
            storage_key=f"voxhelm/jobs/{job.pk}/pending-{n}.mp3",
        )
        staged = StagedMedia(
            producer="synthetic",
            original_filename="input.mp3",
            storage_backend="filesystem",
            expires_at=timezone.now(),
            size_bytes=1,
            content_type="audio/mpeg",
        )
        staged.storage_key = staging.build_staged_media_key(staged=staged)
        staged.save()
    cursor = None
    totals = [0, 0, 0]
    for _ in range(4):
        result = retention.prune_job_artifacts(batch_size=2, cursor=cursor)
        counts = [result.pending_deleted, result.attempt_prefixes_deleted, result.staged_deleted]
        assert all(count <= 2 for count in counts)
        totals = [a + b for a, b in zip(totals, counts, strict=True)]
        cursor = result.next_cursor
    assert totals == [7, 7, 7]
    assert cursor == {}
    # The job can become eligible after its phase already exhausted this sweep.
    retention.prune_job_artifacts(batch_size=2)
    assert not Job.objects.exists()


@pytest.mark.django_db
def test_command_persists_cursor_through_failures_and_validates_before_deleting(monkeypatch):
    from io import StringIO
    from uuid import uuid4

    from django.core.management import CommandError, call_command

    from jobs.models import RetentionProgress

    rows = [
        PendingArtifactDeletion.objects.create(
            job_id=uuid4(), kind="source", storage_backend="filesystem", storage_key="synthetic"
        )
        for _ in range(5)
    ]
    visited = []

    def fail(pk):
        visited.append(pk)
        return "failed"

    monkeypatch.setattr(retention, "process_pending_artifact_deletion", fail)
    for _ in range(3):
        with pytest.raises(CommandError, match="Could not delete"):
            call_command("prune_job_artifacts", batch_size=2, stdout=StringIO())
    assert visited == [row.pk for row in rows]
    assert RetentionProgress.objects.get().cursor == {}
    with pytest.raises(CommandError, match="Invalid retention cursor for jobs"):
        call_command("prune_job_artifacts", cursor='{"jobs": "broken"}', stdout=StringIO())
    assert visited == [row.pk for row in rows]
    call_command("prune_job_artifacts", dry_run=True, batch_size=2, stdout=StringIO())
    assert RetentionProgress.objects.get().cursor == {}


@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize("failure", ["storage", "row_delete"])
def test_admission_isolates_cleanup_failures_and_keeps_retry_records(monkeypatch, caplog, failure):
    from django.db import DatabaseError

    from jobs.artifacts import FilesystemArtifactStore

    expired = timezone.now() - timedelta(days=1)
    for n in range(6):
        staging.stage_uploaded_audio(
            producer="synthetic",
            upload=SimpleUploadedFile(f"old-{n}.mp3", b"audio", content_type="audio/mpeg"),
        )
    StagedMedia.objects.update(expires_at=expired)
    candidates = list(StagedMedia.objects.order_by("expires_at", "pk"))
    failed = candidates[0]
    real_delete = FilesystemArtifactStore.delete
    real_row_delete = StagedMedia.delete
    attempted = []

    def delete_object(self, *, key):
        attempted.append(key)
        if key == failed.storage_key and failure == "storage":
            raise OSError("synthetic-private-storage-error https://private.invalid/path")
        return real_delete(self, key=key)

    def delete_row(self, *args, **kwargs):
        if self.pk == failed.pk and failure == "row_delete":
            raise DatabaseError("synthetic-private-row-error https://private.invalid/path")
        return real_row_delete(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(FilesystemArtifactStore, "delete", delete_object)
        patch.setattr(StagedMedia, "delete", delete_row)
        fresh = staging.stage_uploaded_audio(
            producer="synthetic",
            upload=SimpleUploadedFile("new.mp3", b"audio", content_type="audio/mpeg"),
        )
    assert attempted == [row.storage_key for row in candidates[:4]]
    assert StagedMedia.objects.filter(pk=fresh.pk).exists()
    assert StagedMedia.objects.filter(pk=failed.pk).exists()
    # The two unattempted rows keep their expiry; the failed one is deferred behind
    # them (still expired) so it cannot starve later bounded passes.
    assert StagedMedia.objects.filter(expires_at=expired).count() == 2
    failed.refresh_from_db()
    assert expired < failed.expires_at <= timezone.now()
    assert str(failed.pk) in caplog.text
    assert "synthetic-private" not in caplog.text
    assert "private.invalid" not in caplog.text
    assert failed.storage_key not in caplog.text
    staging.cleanup_expired_staged_media()
    assert not StagedMedia.objects.filter(expires_at=expired).exists()
