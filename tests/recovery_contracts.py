"""Process death and reopened-completion cleanup contracts, with synthetic bytes."""

from __future__ import annotations

import multiprocessing
import sys
from datetime import timedelta
from unittest.mock import patch

from tests.process_contracts import create_database, initialize, make_job


def delete_child(root, kind, row_pk, job_id, target, deleted, park, mutation):
    initialize(root)
    from jobs import retention

    original = retention.delete_stored_object

    def delete_then_park(**kwargs):
        original(**kwargs)
        if kwargs["key"] == target:
            deleted.set()
            assert park.wait(30), "parent did not kill deletion process"

    retention.delete_stored_object = delete_then_park
    if mutation == "ignore_references":
        retention.object_is_referenced = lambda **kwargs: False
    if kind == "prefix":
        retention.clean_attempt_prefix(record_pk=row_pk)
    else:
        retention.delete_claimed_staged_upload(staged_pk=row_pk, job_id=job_id)


def retention_scenario(root, kind, mutation):
    create_database(root)
    from django.conf import settings
    from django.db import connections
    from django.utils import timezone

    from jobs.artifacts import current_artifact_store_identity, get_artifact_store
    from jobs.models import JobArtifact, RemoteAttemptPrefix, StagedMedia
    from jobs.retention import prune_job_artifacts
    from jobs.staging import build_staged_media_key

    job = make_job(
        profile="ad-analysis-v1", state="succeeded", finished_at=timezone.now() - timedelta(days=8)
    )
    canary_job = make_job(state="succeeded", finished_at=timezone.now())
    identity = current_artifact_store_identity()
    prefix = f"voxhelm/jobs/{job.pk}/attempt-1/"
    record = RemoteAttemptPrefix.objects.create(
        job_id=job.pk, attempt=1, prefix=prefix, storage_identity=identity
    )
    orphan = prefix + "orphan.bin"
    canary = prefix + "canary.bin"
    staged = StagedMedia(
        producer="synthetic",
        original_filename="input.mp3",
        content_type="audio/mpeg",
        storage_backend="filesystem",
        storage_identity=identity,
        size_bytes=4,
        claimed_by_job=job,
        claimed_at=timezone.now(),
        expires_at=timezone.now() - timedelta(days=1),
    )
    staged.storage_key = build_staged_media_key(staged=staged)
    staged.save()
    for key in (orphan, canary, staged.storage_key):
        get_artifact_store().put_bytes(
            key=key, data=b"safe", content_type="application/octet-stream"
        )
    reference = JobArtifact.objects.create(
        job=canary_job,
        name="canary.bin",
        kind="source",
        storage_backend="filesystem",
        storage_key=canary,
        storage_identity=identity,
        content_type="application/octet-stream",
        size_bytes=4,
        exposed=False,
    )
    # An independent candidate must still converge after reopening the crashed row.
    healthy = make_job(
        profile="ad-analysis-v1", state="failed", finished_at=timezone.now() - timedelta(days=8)
    )
    healthy_prefix = f"voxhelm/jobs/{healthy.pk}/attempt-1/"
    RemoteAttemptPrefix.objects.create(
        job_id=healthy.pk, attempt=1, prefix=healthy_prefix, storage_identity=identity
    )
    healthy_key = healthy_prefix + "orphan.bin"
    get_artifact_store().put_bytes(
        key=healthy_key, data=b"other", content_type="application/octet-stream"
    )
    target = orphan if kind == "prefix" else staged.storage_key
    ctx = multiprocessing.get_context("spawn")
    deleted, park = ctx.Event(), ctx.Event()
    child = ctx.Process(
        target=delete_child,
        args=(
            root,
            kind,
            record.pk if kind == "prefix" else staged.pk,
            job.pk,
            target,
            deleted,
            park,
            mutation,
        ),
    )
    try:
        child.start()
        assert deleted.wait(15), "external deletion did not execute"
        assert not (settings.VOXHELM_ARTIFACT_ROOT / target).exists()
        assert (settings.VOXHELM_ARTIFACT_ROOT / canary).is_file(), "referenced canary removed"
        assert (settings.VOXHELM_ARTIFACT_ROOT / canary).read_bytes() == b"safe"
    finally:
        if child.is_alive():
            child.kill()  # process death, no Python finally or transaction commit
        child.join(10)
        assert not child.is_alive()
        assert child.exitcode is not None and child.exitcode != 0
    connections.close_all()
    assert RemoteAttemptPrefix.objects.filter(pk=record.pk).exists()
    assert StagedMedia.objects.filter(pk=staged.pk, claimed_by_job=job).exists()
    if mutation == "skip_missing_staged":
        from jobs import retention

        original = retention.delete_expired_staged_upload

        def skip_missing(**kwargs):
            row = StagedMedia.objects.get(pk=kwargs["staged_pk"])
            if (settings.VOXHELM_ARTIFACT_ROOT / row.storage_key).exists():
                original(**kwargs)

        retention.delete_expired_staged_upload = skip_missing
    outcome = prune_job_artifacts()
    assert (
        not outcome.failed and outcome.attempt_prefixes_failed == 0 and outcome.staged_failed == 0
    )
    assert not RemoteAttemptPrefix.objects.filter(job_id__in=[job.pk, healthy.pk]).exists()
    assert not StagedMedia.objects.filter(pk=staged.pk).exists(), (
        "staged ownership did not converge"
    )
    assert not (settings.VOXHELM_ARTIFACT_ROOT / target).exists()
    assert not (settings.VOXHELM_ARTIFACT_ROOT / healthy_key).exists()
    assert JobArtifact.objects.filter(pk=reference.pk).exists()
    assert (settings.VOXHELM_ARTIFACT_ROOT / canary).read_bytes() == b"safe"
    again = prune_job_artifacts()
    assert not again.failed and again.attempt_prefixes_failed == 0 and again.staged_failed == 0


def completion_cleanup_scenario(root, mutation):
    create_database(root)
    from django.conf import settings
    from django.db import connections
    from django.utils import timezone

    from jobs import remote_workers as rw
    from jobs.artifacts import get_artifact_store
    from jobs.models import JobArtifact, StagedMedia, Worker
    from jobs.staging import build_staged_media_key
    from tests.test_remote_workers import (
        plain_capabilities,
        store_manifest_artifacts,
        transcript_manifest,
    )
    from transcriptions.errors import ApiError

    job = make_job()
    inputs = []
    for _ in range(2):
        staged = StagedMedia(
            producer="synthetic",
            original_filename="input.mp3",
            content_type="audio/mpeg",
            storage_backend="filesystem",
            size_bytes=4,
            expires_at=timezone.now() + timedelta(days=1),
        )
        from jobs.artifacts import current_artifact_store_identity

        staged.storage_identity = current_artifact_store_identity()
        staged.storage_key = build_staged_media_key(staged=staged)
        staged.save()
        get_artifact_store().put_bytes(
            key=staged.storage_key, data=b"safe", content_type="audio/mpeg"
        )
        inputs.append(staged)
    staged, unrelated = inputs
    staged.claimed_by_job = job
    staged.claimed_at = timezone.now()
    staged.save()
    job.input_data = {
        "kind": "upload",
        "upload_id": str(staged.pk),
        "filename": "input.mp3",
        "staged_storage_key": staged.storage_key,
        "staged_storage_backend": "filesystem",
        "staged_storage_identity": staged.storage_identity,
    }
    job.save()
    Worker.objects.create(worker_id="synthetic")
    claim = rw.claim_remote_work(
        worker_id="synthetic", payload={"capabilities": plain_capabilities()}
    )
    assert claim
    manifest = transcript_manifest(str(job.pk), 1)
    store_manifest_artifacts(settings, manifest)
    payload = {
        "lease_token": claim.lease_token,
        "result_text": "synthetic transcript",
        "result_metadata": {"backend": "whispercpp", "processing_seconds": 1},
        "artifacts": manifest,
    }
    store = get_artifact_store()
    delete = store.delete
    attempts = []

    def fail_once(*, key):
        attempts.append(key)
        if key == staged.storage_key and attempts.count(key) == 1:
            raise OSError("synthetic deletion failure")
        delete(key=key)

    with patch.object(store, "delete", side_effect=fail_once):
        first = rw.complete_remote_work(worker_id="synthetic", job_id=job.pk, payload=payload)
        assert first["state"] == "succeeded"
        assert attempts == [staged.storage_key]  # the fault actually ran
        assert StagedMedia.objects.filter(pk=staged.pk, claimed_by_job=job).exists()
        assert (settings.VOXHELM_ARTIFACT_ROOT / staged.storage_key).read_bytes() == b"safe"
        job.refresh_from_db()
        committed = (
            job.finished_at,
            job.result_text,
            job.result_metadata,
            job.input_data,
            list(JobArtifact.objects.filter(job=job).order_by("pk").values()),
        )
        connections.close_all()  # retry uses a new DB connection and committed state
        changed = {**payload, "result_metadata": {"backend": "whispercpp", "processing_seconds": 2}}
        try:
            rw.complete_remote_work(worker_id="synthetic", job_id=job.pk, payload=changed)
        except ApiError as exc:
            assert exc.status == 409
        else:
            raise AssertionError("changed completion metadata accepted")
        assert attempts == [staged.storage_key]
        if mutation == "no_idempotent_completion":
            rw.is_same_successful_completion = lambda **kwargs: False
        try:
            second = rw.complete_remote_work(worker_id="synthetic", job_id=job.pk, payload=payload)
        except ApiError as exc:
            raise AssertionError("identical completion retry refused") from exc
        assert second == first
        job.refresh_from_db()
        assert (
            job.finished_at,
            job.result_text,
            job.result_metadata,
            job.input_data,
            list(JobArtifact.objects.filter(job=job).order_by("pk").values()),
        ) == committed
        assert attempts == [staged.storage_key, staged.storage_key]
        assert not StagedMedia.objects.filter(pk=staged.pk).exists()
        assert not (settings.VOXHELM_ARTIFACT_ROOT / staged.storage_key).exists()
        assert StagedMedia.objects.filter(pk=unrelated.pk).exists()
        assert (settings.VOXHELM_ARTIFACT_ROOT / unrelated.storage_key).read_bytes() == b"safe"
        assert len(committed[4]) == 3


if __name__ == "__main__":
    root, scenario, mutation = sys.argv[1:]
    if scenario == "completion":
        completion_cleanup_scenario(root, mutation)
    else:
        retention_scenario(root, scenario, mutation)
