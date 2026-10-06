from __future__ import annotations

import os
from datetime import timedelta
from io import StringIO
from pathlib import Path

import pytest
from django.core.management import CommandError, call_command
from django.utils import timezone

from jobs.artifacts import current_artifact_store_identity, get_artifact_store
from jobs.models import Job, JobArtifact, PendingArtifactDeletion
from jobs.retention import (
    job_key_is_deletable,
    object_location,
    prune_artifact,
    prune_job_artifacts,
    queue_replaced_intermediate_objects,
)


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

    class BrokenStore:
        def delete(self, *, key: str) -> None:
            raise RuntimeError("S3 unavailable")

    monkeypatch.setattr(
        "jobs.retention.get_artifact_store_for_identity", lambda identity: BrokenStore()
    )

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

    class BrokenStore:
        def delete(self, *, key: str) -> None:
            raise RuntimeError("S3 unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(
            "jobs.retention.get_artifact_store_for_identity", lambda identity: BrokenStore()
        )
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
    ok = [f"voxhelm/jobs/{job_id}/input.mp3", f"voxhelm/jobs/{job_id}/attempt-2/source.mp3"]
    bad = [
        f"voxhelm/jobs/{job_id}/./input.mp3",
        f"voxhelm/jobs/{job_id}/../x/input.mp3",
        f"voxhelm/jobs/{job_id}//input.mp3",
        f"/voxhelm/jobs/{job_id}/input.mp3",
        f"other/jobs/{job_id}/input.mp3",
        "voxhelm/jobs/22222222-2222-2222-2222-222222222222/input.mp3",
        f"voxhelm/jobs/{job_id}/sub/dir/input.mp3",
        f"voxhelm/staged-inputs/{job_id}/input.mp3",
    ]
    for key in ok:
        assert job_key_is_deletable(identity=s3, key=key, job_id=job_id)
    for key in bad:
        assert not job_key_is_deletable(identity=s3, key=key, job_id=job_id), key
    assert not job_key_is_deletable(identity=s3, key=ok[0], job_id=job_id, name="other.mp3")
