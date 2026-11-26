from __future__ import annotations

import os
from datetime import timedelta
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from django.core.management import CommandError, call_command
from django.utils import timezone

from jobs import retention
from jobs.artifacts import (
    FilesystemArtifactStore,
    current_artifact_store_identity,
    get_artifact_store,
)
from jobs.models import Job, JobArtifact, PendingArtifactDeletion, StagedMedia
from jobs.retention import (
    drain_pending_artifact_deletions,
    expired_job_metadata,
    job_key_is_deletable,
    object_is_referenced,
    object_location,
    prune_artifact,
    prune_job_artifacts,
    prune_job_metadata,
    queue_replaced_intermediate_objects,
)
from transcriptions.profiles import AD_ANALYSIS_V1


def make_job(*, state: str, finished_ago: timedelta | None) -> Job:
    finished_at = timezone.now() - finished_ago if finished_ago is not None else None
    return Job.objects.create(
        producer="archive",
        job_type=Job.JobType.TRANSCRIBE,
        input_data={"kind": "url", "url": "https://media.example.com/episode.mp3"},
        state=state,
        finished_at=finished_at,
    )


def make_artifact(
    job: Job,
    *,
    name: str,
    kind: str,
    exposed: bool = False,
    write_object: bool = True,
) -> JobArtifact:
    key = f"voxhelm/jobs/{job.id}/{name}"
    if write_object:
        stored = get_artifact_store().put_bytes(
            key=key, data=b"payload", content_type="application/octet-stream"
        )
        size = stored.size_bytes
    else:
        size = 7
    return JobArtifact.objects.create(
        job=job,
        name=name,
        kind=kind,
        format="source",
        storage_backend="filesystem",
        storage_key=key,
        storage_identity=current_artifact_store_identity(),
        content_type="application/octet-stream",
        size_bytes=size,
        exposed=exposed,
    )


def object_path(settings, artifact: JobArtifact) -> Path:
    return Path(settings.VOXHELM_ARTIFACT_ROOT) / artifact.storage_key


@pytest.mark.django_db
def test_default_pending_drain_advances_past_a_refused_page(settings):
    job = make_job(state=Job.State.SUCCEEDED, finished_ago=timedelta(hours=2))
    blocked = [
        PendingArtifactDeletion(
            job_id=job.pk,
            kind=JobArtifact.Kind.SOURCE,
            storage_backend="filesystem",
            storage_key=f"foreign/blocked-{i}.mp3",
        )
        for i in range(100)
    ]
    PendingArtifactDeletion.objects.bulk_create(blocked)
    source = make_artifact(job, name="input.mp3", kind=JobArtifact.Kind.SOURCE)
    pending = queue_replaced_intermediate_objects(replaced=[source], kept=[])[0]
    source.delete()

    assert drain_pending_artifact_deletions() == {"resolved": 0, "refused": 100, "failed": 0}
    assert object_path(settings, source).exists()

    counts = drain_pending_artifact_deletions()

    assert counts == {"resolved": 1, "refused": 99, "failed": 0}
    assert not object_path(settings, source).exists()
    assert not PendingArtifactDeletion.objects.filter(pk=pending.pk).exists()
    assert set(PendingArtifactDeletion.objects.values_list("pk", flat=True)) == {
        item.pk for item in blocked
    }


@pytest.mark.django_db
def test_pending_drain_continues_after_storage_and_logging_fail(settings, monkeypatch):
    job = make_job(state=Job.State.SUCCEEDED, finished_ago=timedelta(hours=2))
    first = make_artifact(job, name="first.mp3", kind=JobArtifact.Kind.SOURCE)
    second = make_artifact(job, name="second.mp3", kind=JobArtifact.Kind.SOURCE)
    failed, resolved = queue_replaced_intermediate_objects(replaced=[first, second], kept=[])
    first.delete()
    second.delete()
    delete_object = retention.delete_stored_object

    def delete_with_failure(*, identity, key):
        if key == failed.storage_key:
            raise OSError("storage unavailable")
        delete_object(identity=identity, key=key)

    def log_failure(*args, **kwargs):
        raise RuntimeError("logging sink unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(retention, "delete_stored_object", delete_with_failure)
        patch.setattr(retention.logger, "exception", log_failure)
        assert drain_pending_artifact_deletions() == {"resolved": 1, "refused": 0, "failed": 1}

    failed.refresh_from_db()
    assert failed.drain_attempts == 1
    assert object_path(settings, first).exists()
    assert not PendingArtifactDeletion.objects.filter(pk=resolved.pk).exists()
    assert not object_path(settings, second).exists()
    assert drain_pending_artifact_deletions() == {"resolved": 1, "refused": 0, "failed": 0}
    assert not object_path(settings, first).exists()


@pytest.mark.django_db
def test_prunes_expired_source_of_finished_job(settings):
    settings.VOXHELM_SOURCE_ARTIFACT_RETENTION_SECONDS = 3600
    job = make_job(state=Job.State.SUCCEEDED, finished_ago=timedelta(hours=2))
    source = make_artifact(job, name="input.mp3", kind=JobArtifact.Kind.SOURCE)
    transcript = make_artifact(
        job, name="transcript.txt", kind=JobArtifact.Kind.TRANSCRIPT_TEXT, exposed=True
    )

    result = prune_job_artifacts()

    assert [artifact.pk for artifact in result.deleted] == [source.pk]
    assert not object_path(settings, source).exists()
    assert not JobArtifact.objects.filter(pk=source.pk).exists()
    assert object_path(settings, transcript).exists()
    assert JobArtifact.objects.filter(pk=transcript.pk).exists()


@pytest.mark.django_db
def test_keeps_recent_source_and_running_jobs_untouched(settings):
    settings.VOXHELM_SOURCE_ARTIFACT_RETENTION_SECONDS = 3600
    recent = make_job(state=Job.State.SUCCEEDED, finished_ago=timedelta(minutes=5))
    recent_source = make_artifact(recent, name="input.mp3", kind=JobArtifact.Kind.SOURCE)
    running = make_job(state=Job.State.RUNNING, finished_ago=None)
    Job.objects.filter(pk=running.pk).update(updated_at=timezone.now() - timedelta(days=3))
    running_source = make_artifact(running, name="input.mp4", kind=JobArtifact.Kind.SOURCE)
    running_audio = make_artifact(
        running, name="extracted-audio.wav", kind=JobArtifact.Kind.EXTRACTED_AUDIO
    )

    result = prune_job_artifacts()

    assert result.deleted == []
    for artifact in (recent_source, running_source, running_audio):
        assert object_path(settings, artifact).exists()
        assert JobArtifact.objects.filter(pk=artifact.pk).exists()


@pytest.mark.django_db
def test_extracted_audio_is_pruned_once_job_is_terminal(settings):
    settings.VOXHELM_SOURCE_ARTIFACT_RETENTION_SECONDS = 3600
    job = make_job(state=Job.State.FAILED, finished_ago=timedelta(seconds=1))
    audio = make_artifact(job, name="extracted-audio.wav", kind=JobArtifact.Kind.EXTRACTED_AUDIO)
    source = make_artifact(job, name="input.mp4", kind=JobArtifact.Kind.SOURCE)

    result = prune_job_artifacts()

    assert [artifact.pk for artifact in result.deleted] == [audio.pk]
    assert not object_path(settings, audio).exists()
    assert object_path(settings, source).exists()


@pytest.mark.django_db
def test_never_touches_exposed_or_final_artifacts(settings):
    settings.VOXHELM_SOURCE_ARTIFACT_RETENTION_SECONDS = 0
    job = make_job(state=Job.State.SUCCEEDED, finished_ago=timedelta(days=30))
    kept = [
        make_artifact(job, name="transcript.json", kind=JobArtifact.Kind.TRANSCRIPT_JSON),
        make_artifact(job, name="speech.mp3", kind=JobArtifact.Kind.SPEECH_MP3, exposed=True),
        make_artifact(job, name="exposed-source.mp3", kind=JobArtifact.Kind.SOURCE, exposed=True),
    ]

    assert prune_job_artifacts().deleted == []
    for artifact in kept:
        assert object_path(settings, artifact).exists()
        assert JobArtifact.objects.filter(pk=artifact.pk).exists()


@pytest.mark.django_db
def test_dry_run_deletes_nothing(settings):
    settings.VOXHELM_SOURCE_ARTIFACT_RETENTION_SECONDS = 60
    job = make_job(state=Job.State.SUCCEEDED, finished_ago=timedelta(hours=1))
    source = make_artifact(job, name="input.mp3", kind=JobArtifact.Kind.SOURCE)
    out = StringIO()

    call_command("prune_job_artifacts", "--dry-run", stdout=out)

    assert f"Would delete source artifact {source.storage_key}" in out.getvalue()
    assert object_path(settings, source).exists()
    assert JobArtifact.objects.filter(pk=source.pk).exists()


@pytest.mark.django_db
def test_command_deletes_and_tolerates_missing_objects(settings):
    settings.VOXHELM_SOURCE_ARTIFACT_RETENTION_SECONDS = 60
    job = make_job(state=Job.State.SUCCEEDED, finished_ago=timedelta(hours=1))
    missing = make_artifact(job, name="input.mp3", kind=JobArtifact.Kind.SOURCE, write_object=False)
    out = StringIO()

    call_command("prune_job_artifacts", stdout=out)

    assert "Deleted 1 artifact(s)" in out.getvalue()
    assert not JobArtifact.objects.filter(pk=missing.pk).exists()


@pytest.mark.django_db
def test_failed_object_deletion_keeps_row_and_fails_command(settings, monkeypatch):
    settings.VOXHELM_SOURCE_ARTIFACT_RETENTION_SECONDS = 60
    job = make_job(state=Job.State.SUCCEEDED, finished_ago=timedelta(hours=1))
    source = make_artifact(job, name="input.mp3", kind=JobArtifact.Kind.SOURCE)

    def broken_delete(self, *, key: str) -> None:
        raise OSError("store unavailable")

    monkeypatch.setattr(FilesystemArtifactStore, "delete", broken_delete)

    with pytest.raises(CommandError):
        call_command("prune_job_artifacts", stdout=StringIO())

    assert JobArtifact.objects.filter(pk=source.pk).exists()
    assert object_path(settings, source).exists()


@pytest.mark.django_db
def test_shared_storage_key_is_not_deleted(settings):
    settings.VOXHELM_SOURCE_ARTIFACT_RETENTION_SECONDS = 60
    job = make_job(state=Job.State.SUCCEEDED, finished_ago=timedelta(hours=1))
    source = make_artifact(job, name="input.mp3", kind=JobArtifact.Kind.SOURCE)
    other = make_job(state=Job.State.RUNNING, finished_ago=None)
    JobArtifact.objects.create(
        job=other,
        name="input.mp3",
        kind=JobArtifact.Kind.SOURCE,
        format="source",
        storage_backend=source.storage_backend,
        storage_key=source.storage_key,
        storage_identity=source.storage_identity,
        content_type=source.content_type,
        size_bytes=source.size_bytes,
        exposed=False,
    )

    prune_job_artifacts()

    assert not JobArtifact.objects.filter(pk=source.pk).exists()
    assert object_path(settings, source).exists()


@pytest.mark.django_db
def test_legacy_empty_identity_is_recognized_as_shared(settings):
    settings.VOXHELM_SOURCE_ARTIFACT_RETENTION_SECONDS = 60
    job = make_job(state=Job.State.SUCCEEDED, finished_ago=timedelta(hours=1))
    source = make_artifact(job, name="input.mp3", kind=JobArtifact.Kind.SOURCE)
    JobArtifact.objects.filter(pk=source.pk).update(storage_identity={})
    JobArtifact.objects.create(
        job=job,
        name="speech.wav",
        kind=JobArtifact.Kind.SPEECH_WAV,
        format="wav",
        storage_backend="filesystem",
        storage_key=source.storage_key,
        storage_identity=current_artifact_store_identity(),
        content_type="audio/wav",
        size_bytes=source.size_bytes,
        exposed=True,
    )

    prune_job_artifacts()

    assert not JobArtifact.objects.filter(pk=source.pk).exists()
    assert object_path(settings, source).exists()


@pytest.mark.django_db
def test_queued_deletion_failure_is_retried_by_prune(settings, monkeypatch):
    job = make_job(state=Job.State.SUCCEEDED, finished_ago=timedelta(minutes=1))
    key = f"voxhelm/jobs/{job.id}/attempt-0/old-source.mp3"
    get_artifact_store().put_bytes(key=key, data=b"old", content_type="audio/mpeg")
    pending = PendingArtifactDeletion.objects.create(
        job_id=job.id,
        kind=JobArtifact.Kind.SOURCE,
        storage_backend="filesystem",
        storage_key=key,
        storage_identity=current_artifact_store_identity(),
    )

    def broken_delete(self, *, key: str) -> None:
        raise OSError("store unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(FilesystemArtifactStore, "delete", broken_delete)
        with pytest.raises(CommandError):
            call_command("prune_job_artifacts", stdout=StringIO())
    assert PendingArtifactDeletion.objects.filter(pk=pending.pk).exists()

    out = StringIO()
    call_command("prune_job_artifacts", stdout=out)

    assert "Deleted 1 queued replaced object(s)" in out.getvalue()
    assert not PendingArtifactDeletion.objects.exists()
    assert not (Path(settings.VOXHELM_ARTIFACT_ROOT) / key).exists()


@pytest.mark.django_db
def test_non_canonical_key_aliasing_final_output_is_refused(settings):
    settings.VOXHELM_SOURCE_ARTIFACT_RETENTION_SECONDS = 60
    job = make_job(state=Job.State.SUCCEEDED, finished_ago=timedelta(hours=1))
    transcript = make_artifact(
        job, name="transcript.txt", kind=JobArtifact.Kind.TRANSCRIPT_TEXT, exposed=True
    )
    source = JobArtifact.objects.create(
        job=job,
        name="input.mp3",
        kind=JobArtifact.Kind.SOURCE,
        format="source",
        storage_backend="filesystem",
        storage_key=f"voxhelm/jobs/{job.id}/./transcript.txt",
        storage_identity=current_artifact_store_identity(),
        content_type="audio/mpeg",
        size_bytes=transcript.size_bytes,
        exposed=False,
    )

    result = prune_job_artifacts()

    # Not a voxhelm-generated job key: refused, row and object stay.
    assert [artifact.pk for artifact in result.refused] == [source.pk]
    assert JobArtifact.objects.filter(pk=source.pk).exists()
    assert object_path(settings, transcript).exists()


@pytest.mark.django_db
def test_queued_legacy_identity_is_pinned_to_resolved_store(settings):
    job = make_job(state=Job.State.RUNNING, finished_ago=None)
    source = make_artifact(job, name="input.mp3", kind=JobArtifact.Kind.SOURCE)
    JobArtifact.objects.filter(pk=source.pk).update(storage_identity={})
    source.refresh_from_db()

    queued = queue_replaced_intermediate_objects(replaced=[source], kept=[])

    assert len(queued) == 1
    assert queued[0].storage_identity == current_artifact_store_identity()


def test_object_location_canonicalizes_s3_endpoint_but_keeps_exact_keys():
    def s3(endpoint: str) -> dict[str, object]:
        return {"backend": "s3", "endpoint_url": endpoint, "bucket": "voxhelm"}

    assert object_location(identity=s3("https://S3.example.com/"), key="a/b") == object_location(
        identity=s3("https://s3.example.com:443"), key="a/b"
    )
    assert object_location(identity=s3("https://s3.example.com"), key="a/./b") != (
        object_location(identity=s3("https://s3.example.com"), key="a/b")
    )


@pytest.mark.django_db
def test_s3_references_distinguish_ipv6_host_groups_from_ports():
    job = make_job(state=Job.State.SUCCEEDED, finished_ago=timedelta(hours=1))
    source = make_artifact(job, name="input.mp3", kind=JobArtifact.Kind.SOURCE, write_object=False)
    retained = make_artifact(
        job, name="retained.mp3", kind=JobArtifact.Kind.SOURCE, write_object=False
    )
    identity = {"backend": "s3", "endpoint_url": "https://[2001:db8::1:9000]", "bucket": "media"}
    source.storage_backend = retained.storage_backend = "s3"
    source.storage_identity = identity
    source.save(update_fields=["storage_backend", "storage_identity"])
    retained.storage_key = source.storage_key
    retained.storage_identity = {**identity, "endpoint_url": "https://[2001:db8::1:9000]:443/"}
    retained.save(update_fields=["storage_backend", "storage_identity", "storage_key"])

    def referenced():
        return object_is_referenced(
            identity=source.storage_identity, key=source.storage_key, exclude_artifact_pk=source.pk
        )

    assert referenced()
    retained.storage_identity = {**identity, "endpoint_url": "https://[2001:db8::1]:9000"}
    retained.save(update_fields=["storage_identity"])
    assert not referenced()


@pytest.mark.django_db
@pytest.mark.parametrize("retarget_during_check", [False, True])
def test_retention_keeps_admission_and_delete_under_one_root(
    settings, tmp_path, monkeypatch, retarget_during_check
):
    settings.VOXHELM_SOURCE_ARTIFACT_RETENTION_SECONDS = 60
    job = make_job(state=Job.State.SUCCEEDED, finished_ago=timedelta(hours=1))
    source = make_artifact(job, name="input.mp3", kind=JobArtifact.Kind.SOURCE)
    old_path = object_path(settings, source)
    new_root = tmp_path / "replacement"
    new_path = new_root / source.storage_key
    new_path.parent.mkdir(parents=True)
    new_path.write_bytes(b"replacement")
    os.link(new_path, new_root / "keep.mp3")

    alias = tmp_path / "store"
    alias.symlink_to(settings.VOXHELM_ARTIFACT_ROOT, target_is_directory=True)
    source.storage_identity = {"backend": "filesystem", "root": str(alias)}
    source.save(update_fields=["storage_identity"])
    assert object_location(identity=source.storage_identity, key=source.storage_key) == (
        "filesystem",
        str(old_path.resolve()),
    )

    def retarget():
        alias.unlink()
        alias.symlink_to(new_root, target_is_directory=True)

    if retarget_during_check:
        original = retention.object_is_referenced

        def retarget_then_check(**kwargs):
            retarget()
            return original(**kwargs)

        monkeypatch.setattr(retention, "object_is_referenced", retarget_then_check)
    else:
        retarget()

    result = prune_job_artifacts()

    assert new_path.exists()
    if retarget_during_check:
        assert [artifact.pk for artifact in result.deleted] == [source.pk]
        assert not old_path.exists()
    else:
        assert [artifact.pk for artifact in result.refused] == [source.pk]
        assert old_path.exists()


@pytest.mark.django_db
def test_overlapping_prune_skips_candidate_already_pruned(settings):
    settings.VOXHELM_SOURCE_ARTIFACT_RETENTION_SECONDS = 60
    job = make_job(state=Job.State.SUCCEEDED, finished_ago=timedelta(hours=1))
    source = make_artifact(job, name="input.mp3", kind=JobArtifact.Kind.SOURCE)
    # Run A selected the candidate; run B prunes it first.
    stale_candidate_pk = source.pk
    prune_job_artifacts()
    assert not JobArtifact.objects.filter(pk=source.pk).exists()
    # A file reappearing at that key (e.g. a new row) must not be touched by A.
    get_artifact_store().put_bytes(key=source.storage_key, data=b"new", content_type="audio/mpeg")

    assert prune_artifact(artifact_pk=stale_candidate_pk, now=None) is None
    assert object_path(settings, source).exists()


@pytest.mark.django_db
def test_symlinked_alias_of_final_output_is_refused(settings):
    settings.VOXHELM_SOURCE_ARTIFACT_RETENTION_SECONDS = 60
    job = make_job(state=Job.State.SUCCEEDED, finished_ago=timedelta(hours=1))
    transcript = make_artifact(
        job, name="transcript.txt", kind=JobArtifact.Kind.TRANSCRIPT_TEXT, exposed=True
    )
    root = Path(settings.VOXHELM_ARTIFACT_ROOT)
    alias_dir = root / "voxhelm" / "alias"
    alias_dir.symlink_to(object_path(settings, transcript).parent, target_is_directory=True)
    source = JobArtifact.objects.create(
        job=job,
        name="input.mp3",
        kind=JobArtifact.Kind.SOURCE,
        format="source",
        storage_backend="filesystem",
        storage_key="voxhelm/alias/transcript.txt",
        storage_identity=current_artifact_store_identity(),
        content_type="audio/mpeg",
        size_bytes=transcript.size_bytes,
        exposed=False,
    )

    result = prune_job_artifacts()

    # Not a voxhelm-generated job key: refused, row and object stay.
    assert [artifact.pk for artifact in result.refused] == [source.pk]
    assert JobArtifact.objects.filter(pk=source.pk).exists()
    assert object_path(settings, transcript).exists()


@pytest.mark.django_db
def test_case_variant_key_of_final_output_is_kept_on_case_insensitive_fs(settings):
    settings.VOXHELM_SOURCE_ARTIFACT_RETENTION_SECONDS = 60
    job = make_job(state=Job.State.SUCCEEDED, finished_ago=timedelta(hours=1))
    transcript = make_artifact(
        job, name="transcript.txt", kind=JobArtifact.Kind.TRANSCRIPT_TEXT, exposed=True
    )
    upper_key = transcript.storage_key.replace("transcript.txt", "TRANSCRIPT.txt")
    if not (Path(settings.VOXHELM_ARTIFACT_ROOT) / upper_key).exists():
        pytest.skip("filesystem is case-sensitive")
    source = JobArtifact.objects.create(
        job=job,
        name="TRANSCRIPT.txt",
        kind=JobArtifact.Kind.SOURCE,
        format="source",
        storage_backend="filesystem",
        storage_key=upper_key,
        storage_identity=current_artifact_store_identity(),
        content_type="audio/mpeg",
        size_bytes=transcript.size_bytes,
        exposed=False,
    )

    prune_job_artifacts()

    assert not JobArtifact.objects.filter(pk=source.pk).exists()
    assert object_path(settings, transcript).exists()


@pytest.mark.django_db
def test_replaced_source_aliasing_replaced_transcript_is_not_queued(settings):
    job = make_job(state=Job.State.RUNNING, finished_ago=None)
    transcript = make_artifact(
        job, name="transcript.txt", kind=JobArtifact.Kind.TRANSCRIPT_TEXT, exposed=True
    )
    root = Path(settings.VOXHELM_ARTIFACT_ROOT)
    (root / "voxhelm" / "alias").symlink_to(
        object_path(settings, transcript).parent, target_is_directory=True
    )
    source = JobArtifact.objects.create(
        job=job,
        name="input.mp3",
        kind=JobArtifact.Kind.SOURCE,
        format="source",
        storage_backend="filesystem",
        storage_key="voxhelm/alias/transcript.txt",
        storage_identity=current_artifact_store_identity(),
        content_type="audio/mpeg",
        size_bytes=transcript.size_bytes,
        exposed=False,
    )

    assert queue_replaced_intermediate_objects(replaced=[transcript, source], kept=[]) == []
    assert object_path(settings, transcript).exists()


@pytest.mark.django_db
def test_replacement_indexes_protected_inodes_and_preserves_hard_link(settings, monkeypatch):
    job = make_job(state=Job.State.RUNNING, finished_ago=None)
    protected = [
        make_artifact(job, name=f"final-{i}.txt", kind=JobArtifact.Kind.TRANSCRIPT_TEXT)
        for i in range(12)
    ]
    sources = [
        make_artifact(job, name=f"source-{i}.mp3", kind=JobArtifact.Kind.SOURCE) for i in range(12)
    ]
    alias = make_artifact(job, name="alias.mp3", kind=JobArtifact.Kind.SOURCE, write_object=False)
    object_path(settings, alias).hardlink_to(object_path(settings, protected[0]))
    key_is_deletable = retention.job_key_is_deletable
    filesystem_stat = retention.filesystem_stat
    inode_operations = 0

    def admit_alias(**kwargs):
        # Exercise inode matching independently of the hard-link allowlist check.
        return kwargs["key"] == alias.storage_key or key_is_deletable(**kwargs)

    class CountedInode(int):
        def __hash__(self):
            nonlocal inode_operations
            inode_operations += 1
            return super().__hash__()

        def __eq__(self, other):
            nonlocal inode_operations
            inode_operations += 1
            return super().__eq__(other)

    def count_inode_operations(location):
        stat = filesystem_stat(location)
        if stat is None:
            return None
        return SimpleNamespace(st_dev=stat.st_dev, st_ino=CountedInode(stat.st_ino))

    monkeypatch.setattr(retention, "job_key_is_deletable", admit_alias)
    monkeypatch.setattr(retention, "filesystem_stat", count_inode_operations)

    queued = queue_replaced_intermediate_objects(replaced=[*protected, *sources, alias], kept=[])

    assert [entry.storage_key for entry in queued] == [source.storage_key for source in sources]
    assert inode_operations <= 3 * (len(protected) + len(sources) + 1)


@pytest.mark.django_db
def test_symlink_followed_by_parent_reference_is_refused(settings):
    settings.VOXHELM_SOURCE_ARTIFACT_RETENTION_SECONDS = 60
    job = make_job(state=Job.State.SUCCEEDED, finished_ago=timedelta(hours=1))
    transcript = make_artifact(
        job, name="transcript.txt", kind=JobArtifact.Kind.TRANSCRIPT_TEXT, exposed=True
    )
    root = Path(settings.VOXHELM_ARTIFACT_ROOT)
    real_dir = object_path(settings, transcript).parent
    (real_dir / "child").mkdir()
    (root / "alias").symlink_to(real_dir / "child", target_is_directory=True)
    source_key = "alias/../transcript.txt"
    assert (root / source_key).resolve() == object_path(settings, transcript).resolve()
    source = JobArtifact.objects.create(
        job=job,
        name="input.mp3",
        kind=JobArtifact.Kind.SOURCE,
        format="source",
        storage_backend="filesystem",
        storage_key=source_key,
        storage_identity=current_artifact_store_identity(),
        content_type="audio/mpeg",
        size_bytes=transcript.size_bytes,
        exposed=False,
    )

    result = prune_job_artifacts()

    # Not a voxhelm-generated job key: refused, row and object stay.
    assert [artifact.pk for artifact in result.refused] == [source.pk]
    assert JobArtifact.objects.filter(pk=source.pk).exists()
    assert object_path(settings, transcript).exists()

    replaced_source = JobArtifact(
        job=job,
        name="input2.mp3",
        kind=JobArtifact.Kind.SOURCE,
        storage_backend="filesystem",
        storage_key=source_key,
        storage_identity=current_artifact_store_identity(),
        exposed=False,
    )
    assert (
        queue_replaced_intermediate_objects(replaced=[transcript, replaced_source], kept=[]) == []
    )


@pytest.mark.django_db
def test_hard_linked_or_foreign_job_keys_are_refused(settings):
    settings.VOXHELM_SOURCE_ARTIFACT_RETENTION_SECONDS = 60
    job = make_job(state=Job.State.SUCCEEDED, finished_ago=timedelta(hours=1))
    other = make_job(state=Job.State.SUCCEEDED, finished_ago=timedelta(hours=1))
    linked = make_artifact(job, name="input.mp3", kind=JobArtifact.Kind.SOURCE)
    keep_path = Path(settings.VOXHELM_ARTIFACT_ROOT) / "elsewhere.mp3"
    os.link(object_path(settings, linked), keep_path)
    foreign = make_artifact(other, name="input.mp3", kind=JobArtifact.Kind.SOURCE)
    JobArtifact.objects.filter(pk=foreign.pk).update(storage_key=linked.storage_key)

    result = prune_job_artifacts()

    assert {artifact.pk for artifact in result.refused} == {linked.pk, foreign.pk}
    assert keep_path.exists()
    assert object_path(settings, linked).exists()


def test_job_key_allowlist_shapes(settings):
    settings.VOXHELM_ARTIFACT_PREFIX = "voxhelm"
    s3 = {"backend": "s3", "endpoint_url": "https://s3.example.com", "bucket": "b"}
    job_id = "11111111-1111-1111-1111-111111111111"
    ok = [
        f"voxhelm/jobs/{job_id}/input.mp3",
        f"voxhelm/jobs/{job_id}/attempt-2/source.mp3",
        # Written under an earlier VOXHELM_ARTIFACT_PREFIX.
        f"old-prefix/jobs/{job_id}/input.mp3",
    ]
    bad = [
        f"voxhelm/jobs/{job_id}/./input.mp3",
        f"voxhelm/jobs/{job_id}/../x/input.mp3",
        f"voxhelm/jobs/{job_id}//input.mp3",
        f"/voxhelm/jobs/{job_id}/input.mp3",
        "voxhelm/jobs/22222222-2222-2222-2222-222222222222/input.mp3",
        f"voxhelm/jobs/{job_id}/sub/dir/input.mp3",
        f"voxhelm/staged-inputs/{job_id}/input.mp3",
    ]
    for key in ok:
        assert job_key_is_deletable(identity=s3, key=key, job_id=job_id)
    for key in bad:
        assert not job_key_is_deletable(identity=s3, key=key, job_id=job_id), key
    assert not job_key_is_deletable(identity=s3, key=ok[0], job_id=job_id, name="other.mp3")


@pytest.mark.django_db
def test_prefix_change_does_not_block_retention(settings):
    settings.VOXHELM_SOURCE_ARTIFACT_RETENTION_SECONDS = 60
    job = make_job(state=Job.State.SUCCEEDED, finished_ago=timedelta(hours=1))
    source = make_artifact(job, name="input.mp3", kind=JobArtifact.Kind.SOURCE)
    settings.VOXHELM_ARTIFACT_PREFIX = "renamed"

    result = prune_job_artifacts()

    assert [artifact.pk for artifact in result.deleted] == [source.pk]
    assert not object_path(settings, source).exists()


@pytest.mark.django_db
def test_refused_queue_entry_is_kept_and_reported(settings):
    job = make_job(state=Job.State.SUCCEEDED, finished_ago=timedelta(minutes=1))
    key = f"voxhelm/jobs/{job.id}/attempt-0/old-source.mp3"
    get_artifact_store().put_bytes(key=key, data=b"old", content_type="audio/mpeg")
    os.link(Path(settings.VOXHELM_ARTIFACT_ROOT) / key, Path(settings.VOXHELM_ARTIFACT_ROOT) / "x")
    pending = PendingArtifactDeletion.objects.create(
        job_id=job.id,
        kind=JobArtifact.Kind.SOURCE,
        storage_backend="filesystem",
        storage_key=key,
        storage_identity=current_artifact_store_identity(),
    )
    out = StringIO()

    call_command("prune_job_artifacts", stdout=out)

    assert "Refused 1 queued object(s)" in out.getvalue()
    assert PendingArtifactDeletion.objects.filter(pk=pending.pk).exists()
    assert (Path(settings.VOXHELM_ARTIFACT_ROOT) / key).exists()


@pytest.mark.django_db
def test_dry_run_reports_refusals_like_a_real_run(settings):
    settings.VOXHELM_SOURCE_ARTIFACT_RETENTION_SECONDS = 60
    job = make_job(state=Job.State.SUCCEEDED, finished_ago=timedelta(hours=1))
    good = make_artifact(job, name="input.mp3", kind=JobArtifact.Kind.SOURCE)
    foreign = make_artifact(job, name="input.mp4", kind=JobArtifact.Kind.SOURCE)
    JobArtifact.objects.filter(pk=foreign.pk).update(storage_key="voxhelm/elsewhere.mp4")

    result = prune_job_artifacts(dry_run=True)

    assert [artifact.pk for artifact in result.deleted] == [good.pk]
    assert [artifact.pk for artifact in result.refused] == [foreign.pk]
    assert object_path(settings, good).exists()


METADATA_RETENTION = timedelta(days=90)


@pytest.mark.django_db
def test_pending_deletion_lookups_use_job_index(settings, monkeypatch):
    settings.VOXHELM_JOB_METADATA_RETENTION_SECONDS = int(METADATA_RETENTION.total_seconds())
    old = make_job(state=Job.State.SUCCEEDED, finished_ago=timedelta(days=100))
    guarded = make_job(state=Job.State.SUCCEEDED, finished_ago=timedelta(days=100))
    guarded.profile = AD_ANALYSIS_V1
    guarded.save(update_fields=["profile"])
    pending = PendingArtifactDeletion.objects.create(
        job_id=guarded.pk,
        kind=JobArtifact.Kind.SOURCE,
        storage_backend="filesystem",
        storage_key=f"voxhelm/jobs/{guarded.pk}/attempt-1/old.mp3",
        storage_identity=current_artifact_store_identity(),
    )
    PendingArtifactDeletion.objects.bulk_create(
        PendingArtifactDeletion(
            job_id=uuid4(),
            kind=JobArtifact.Kind.SOURCE,
            storage_backend="filesystem",
            storage_key=f"unrelated/{i}.mp3",
        )
        for i in range(3000)
    )

    plans = {"metadata pruning": expired_job_metadata().explain()}
    assert [job.pk for job in prune_job_metadata()] == [old.pk]
    assert Job.objects.filter(pk=guarded.pk).exists()

    def explain_drain(selection):
        plans["explicit job deletion"] = selection.explain()
        assert list(selection) == [pending]
        return drain_pending_artifact_deletions(selection)

    monkeypatch.setattr(retention, "drain_pending_artifact_deletions", explain_drain)
    retention.delete_job_for_producer(job_id=guarded.pk, producer=guarded.producer)
    assert not Job.objects.filter(pk=guarded.pk).exists()
    assert not PendingArtifactDeletion.objects.filter(pk=pending.pk).exists()
    assert PendingArtifactDeletion.objects.count() == 3000

    assert all(
        any(
            "SEARCH " in line and "jobs_pending_job_id_idx (job_id=?)" in line
            for line in plan.splitlines()
        )
        for plan in plans.values()
    ), plans


@pytest.mark.django_db
def test_job_metadata_deleted_only_for_old_terminal_jobs(settings):
    settings.VOXHELM_JOB_METADATA_RETENTION_SECONDS = int(METADATA_RETENTION.total_seconds())
    old = {
        state: make_job(state=state, finished_ago=METADATA_RETENTION + timedelta(hours=1))
        for state in (
            Job.State.SUCCEEDED,
            Job.State.FAILED,
            Job.State.CANCELED,
            Job.State.EXPIRED,
        )
    }
    recent = make_job(state=Job.State.FAILED, finished_ago=METADATA_RETENTION - timedelta(hours=1))
    queued = make_job(state=Job.State.QUEUED, finished_ago=None)
    running = make_job(state=Job.State.RUNNING, finished_ago=None)
    Job.objects.filter(pk__in=[queued.pk, running.pk]).update(
        updated_at=timezone.now() - timedelta(days=365)
    )

    result = prune_job_artifacts()

    assert {job.pk for job in result.jobs_deleted} == {job.pk for job in old.values()}
    assert not Job.objects.filter(pk__in=[job.pk for job in old.values()]).exists()
    assert set(Job.objects.values_list("pk", flat=True)) == {recent.pk, queued.pk, running.pk}


@pytest.mark.django_db
def test_job_metadata_falls_back_to_updated_at(settings):
    settings.VOXHELM_JOB_METADATA_RETENTION_SECONDS = int(METADATA_RETENTION.total_seconds())
    legacy = make_job(state=Job.State.CANCELED, finished_ago=None)
    fresh = make_job(state=Job.State.CANCELED, finished_ago=None)
    Job.objects.filter(pk=legacy.pk).update(updated_at=timezone.now() - timedelta(days=91))

    assert [job.pk for job in prune_job_metadata()] == [legacy.pk]
    assert Job.objects.filter(pk=fresh.pk).exists()


@pytest.mark.django_db
def test_job_with_final_artifacts_keeps_its_metadata(settings):
    settings.VOXHELM_JOB_METADATA_RETENTION_SECONDS = int(METADATA_RETENTION.total_seconds())
    job = make_job(state=Job.State.SUCCEEDED, finished_ago=timedelta(days=200))
    transcript = make_artifact(
        job, name="transcript.txt", kind=JobArtifact.Kind.TRANSCRIPT_TEXT, exposed=True
    )

    assert prune_job_artifacts().jobs_deleted == []
    assert Job.objects.filter(pk=job.pk).exists()
    assert JobArtifact.objects.filter(pk=transcript.pk).exists()
    assert object_path(settings, transcript).exists()


@pytest.mark.django_db
def test_intermediates_are_pruned_before_job_metadata_in_one_run(settings):
    settings.VOXHELM_SOURCE_ARTIFACT_RETENTION_SECONDS = 3600
    settings.VOXHELM_JOB_METADATA_RETENTION_SECONDS = int(METADATA_RETENTION.total_seconds())
    job = make_job(state=Job.State.FAILED, finished_ago=timedelta(days=100))
    source = make_artifact(job, name="input.mp4", kind=JobArtifact.Kind.SOURCE)
    audio = make_artifact(job, name="extracted-audio.wav", kind=JobArtifact.Kind.EXTRACTED_AUDIO)
    out = StringIO()

    call_command("prune_job_artifacts", "--dry-run", stdout=out)

    assert f"Would delete failed transcribe job {job.id}" in out.getvalue()
    assert "Would delete 1 job row(s)" in out.getvalue()
    assert Job.objects.filter(pk=job.pk).exists()
    assert object_path(settings, source).exists()

    out = StringIO()
    call_command("prune_job_artifacts", stdout=out)

    assert "Deleted 1 job row(s)" in out.getvalue()
    assert not Job.objects.filter(pk=job.pk).exists()
    assert not object_path(settings, source).exists()
    assert not object_path(settings, audio).exists()


@pytest.mark.django_db
def test_job_whose_intermediates_cannot_be_deleted_keeps_its_metadata(settings, monkeypatch):
    settings.VOXHELM_SOURCE_ARTIFACT_RETENTION_SECONDS = 3600
    settings.VOXHELM_JOB_METADATA_RETENTION_SECONDS = int(METADATA_RETENTION.total_seconds())
    job = make_job(state=Job.State.SUCCEEDED, finished_ago=timedelta(days=100))
    source = make_artifact(job, name="input.mp3", kind=JobArtifact.Kind.SOURCE)

    def broken_delete(self, *, key: str) -> None:
        raise OSError("store unavailable")

    monkeypatch.setattr(FilesystemArtifactStore, "delete", broken_delete)

    with pytest.raises(CommandError):
        call_command("prune_job_artifacts", stdout=StringIO())

    assert Job.objects.filter(pk=job.pk).exists()
    assert JobArtifact.objects.filter(pk=source.pk).exists()


@pytest.mark.django_db
def test_refused_intermediate_keeps_job_metadata(settings):
    settings.VOXHELM_SOURCE_ARTIFACT_RETENTION_SECONDS = 3600
    settings.VOXHELM_JOB_METADATA_RETENTION_SECONDS = int(METADATA_RETENTION.total_seconds())
    job = make_job(state=Job.State.SUCCEEDED, finished_ago=timedelta(days=100))
    source = make_artifact(job, name="input.mp3", kind=JobArtifact.Kind.SOURCE)
    JobArtifact.objects.filter(pk=source.pk).update(storage_key="shared/input.mp3")

    result = prune_job_artifacts()

    assert [artifact.pk for artifact in result.refused] == [source.pk]
    assert result.jobs_deleted == []
    assert Job.objects.filter(pk=job.pk).exists()


@pytest.mark.django_db
def test_queued_deletion_or_staged_claim_keeps_job_metadata(settings):
    settings.VOXHELM_JOB_METADATA_RETENTION_SECONDS = int(METADATA_RETENTION.total_seconds())
    queued_job = make_job(state=Job.State.SUCCEEDED, finished_ago=timedelta(days=100))
    PendingArtifactDeletion.objects.create(
        job_id=queued_job.id,
        kind=JobArtifact.Kind.SOURCE,
        storage_backend="filesystem",
        storage_key="elsewhere/input.mp3",
        storage_identity=current_artifact_store_identity(),
    )
    staged_job = make_job(state=Job.State.FAILED, finished_ago=timedelta(days=100))
    StagedMedia.objects.create(
        producer="archive",
        original_filename="input.mp3",
        content_type="audio/mpeg",
        storage_backend="filesystem",
        storage_key="staged/input.mp3",
        storage_identity=current_artifact_store_identity(),
        claimed_by_job=staged_job,
        claimed_at=timezone.now(),
        expires_at=timezone.now() + timedelta(days=1),
    )

    result = prune_job_artifacts()

    assert result.pending_refused == 1
    assert result.jobs_deleted == []
    assert Job.objects.filter(pk__in=[queued_job.pk, staged_job.pk]).count() == 2


@pytest.mark.django_db
def test_job_metadata_retention_zero_disables_pruning(settings):
    settings.VOXHELM_JOB_METADATA_RETENTION_SECONDS = 0
    job = make_job(state=Job.State.FAILED, finished_ago=timedelta(days=1000))
    out = StringIO()

    call_command("prune_job_artifacts", stdout=out)

    assert "Deleted 0 job row(s) (job metadata retention disabled)" in out.getvalue()
    assert Job.objects.filter(pk=job.pk).exists()
    assert not expired_job_metadata().exists()


@pytest.mark.django_db
def test_job_metadata_pruning_runs_in_batches(settings):
    settings.VOXHELM_JOB_METADATA_RETENTION_SECONDS = int(METADATA_RETENTION.total_seconds())
    jobs = [make_job(state=Job.State.FAILED, finished_ago=timedelta(days=100)) for _ in range(5)]

    deleted = prune_job_metadata(batch_size=2)

    assert {job.pk for job in deleted} == {job.pk for job in jobs}
    assert not Job.objects.exists()
