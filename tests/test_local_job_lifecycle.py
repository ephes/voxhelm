from __future__ import annotations

from types import SimpleNamespace

import pytest
from django.utils import timezone
from django_tasks.base import TaskResultStatus

from jobs import services
from jobs.artifacts import get_artifact_store
from jobs.media import DownloadedMedia
from jobs.models import Job, JobArtifact, PendingArtifactDeletion
from tests.test_remote_worker_cli import sample_transcription_result


def make_job(**kwargs):
    return Job.objects.create(producer="synthetic", job_type="transcribe", input_data={}, **kwargs)


@pytest.mark.django_db
@pytest.mark.parametrize("state", ["running", "succeeded", "failed", "canceled", "expired"])
@pytest.mark.parametrize(
    "executor", [services.execute_transcription_job, services.execute_synthesis_job]
)
def test_duplicate_local_task_does_not_restart_or_fail(state, executor):
    finished = timezone.now()
    job = make_job(
        state=state, django_task_id="expected-task", result_text="keep", finished_at=finished
    )
    result = executor(job_id=str(job.id), task_result_id="stale-task")
    job.refresh_from_db()
    assert result["execution"] == "not_claimed"
    assert (job.state, job.result_text, job.finished_at, job.django_task_id) == (
        state,
        "keep",
        finished,
        "expected-task",
    )


@pytest.mark.django_db
def test_queued_job_rejects_other_task_identity():
    job = make_job(django_task_id="expected")
    assert services.initialize_running_job(job_id=str(job.id), task_result_id="other") is None
    job.refresh_from_db()
    assert job.state == "queued"
    assert job.django_task_id == "expected"


@pytest.mark.django_db
def test_local_reentrant_invocation_runs_inference_and_input_once(monkeypatch, tmp_path):
    job = make_job(django_task_id="expected", output_data={"formats": ["text"]})
    media = tmp_path / "synthetic.mp3"
    media.write_bytes(b"synthetic media")
    prepared, inferred, consumed = [], [], []
    staged = SimpleNamespace()

    def prepare(**kwargs):
        prepared.append(True)
        return DownloadedMedia(
            path=media, content_type="audio/mpeg", source_name="synthetic.mp3"
        ), staged

    def infer(*args, **kwargs):
        inferred.append(True)
        if len(inferred) == 1:
            duplicate = services.execute_transcription_job(
                job_id=str(job.id), task_result_id="expected"
            )
            assert duplicate["execution"] == "not_claimed"
        return sample_transcription_result()

    monkeypatch.setattr(services, "prepare_transcription_input_media", prepare)
    monkeypatch.setattr(services, "delete_staged_media", lambda **kwargs: consumed.append(True))
    monkeypatch.setattr(services, "transcribe_audio", infer)
    services.execute_transcription_job(job_id=str(job.id), task_result_id="expected")
    job.refresh_from_db()
    assert job.state == "succeeded"
    assert prepared == inferred == consumed == [True]
    assert not media.exists()
    assert job.artifacts.filter(exposed=True).count() == 1


@pytest.mark.django_db
@pytest.mark.parametrize("finish", ["success", "failure"])
def test_stale_local_task_cannot_publish_terminal_state(finish):
    job = make_job(state="running", django_task_id="old-task")
    Job.objects.filter(pk=job.pk).update(django_task_id="new-task", result_text="keep")
    if finish == "success":
        services.mark_job_succeeded(job=job, metadata={"stale": True}, result_text="stale")
    else:
        services.mark_job_failed(job=job, exc=ValueError("stale"))
    job.refresh_from_db()
    assert (job.state, job.django_task_id, job.result_text, job.finished_at) == (
        "running",
        "new-task",
        "keep",
        None,
    )


@pytest.mark.django_db
def test_committed_success_survives_running_backend(monkeypatch):
    job = make_job(state="running", django_task_id="expected")
    stale = Job.objects.get(pk=job.pk)
    services.mark_job_succeeded(job=job, metadata={"keep": True}, result_text="keep")
    monkeypatch.setattr(
        services,
        "default_task_backend",
        SimpleNamespace(
            supports_get_result=True,
            get_result=lambda _: SimpleNamespace(
                status=TaskResultStatus.RUNNING, started_at=None, finished_at=None, errors=[]
            ),
        ),
    )
    services.reconcile_job_state(stale)
    assert stale.state == "succeeded"
    assert stale.result_metadata == {"keep": True}
    assert stale.result_text == "keep"
    assert stale.finished_at == job.finished_at


@pytest.mark.django_db
@pytest.mark.parametrize("change", ["completion", "identity"])
def test_reconciliation_cannot_overwrite_newer_database_state(monkeypatch, change):
    job = make_job(state="running", django_task_id="expected")

    def backend_result(task_id):
        if change == "completion":
            Job.objects.filter(pk=job.pk).update(
                state="succeeded", result_text="keep", finished_at=timezone.now()
            )
        else:
            Job.objects.filter(pk=job.pk).update(django_task_id="new-task")
        return SimpleNamespace(
            status=TaskResultStatus.FAILED, started_at=None, finished_at=timezone.now(), errors=[]
        )

    monkeypatch.setattr(
        services,
        "default_task_backend",
        SimpleNamespace(supports_get_result=True, get_result=backend_result),
    )
    services.reconcile_job_state(job)
    assert job.state == ("succeeded" if change == "completion" else "running")
    assert job.django_task_id == ("expected" if change == "completion" else "new-task")


@pytest.mark.django_db
def test_backend_failure_resolves_unfinished_job(monkeypatch):
    job = make_job(state="running", django_task_id="expected")
    finished = timezone.now()
    monkeypatch.setattr(
        services,
        "default_task_backend",
        SimpleNamespace(
            supports_get_result=True,
            get_result=lambda _: SimpleNamespace(
                status=TaskResultStatus.FAILED, started_at=None, finished_at=finished, errors=[]
            ),
        ),
    )
    services.reconcile_job_state(job)
    assert job.state == "failed"
    assert job.finished_at == finished


def publish(job, payload=b"OLD", source_path=None):
    kwargs = dict(
        job=job,
        name="transcript.txt",
        kind="transcript_text",
        format_name="text",
        content_type="text/plain",
        exposed=True,
    )
    if source_path is not None:
        return services.create_or_replace_artifact_from_file(**kwargs, source_path=source_path)
    return services.create_or_replace_artifact(**kwargs, payload=payload)


@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize("file_upload", [False, True])
def test_artifact_database_failure_preserves_old_row_and_bytes(monkeypatch, tmp_path, file_upload):
    job = make_job()
    original = publish(job)
    original_key = original.storage_key
    source = tmp_path / "new.txt"
    source.write_bytes(b"NEW")
    called = []

    def fail(**kwargs):
        called.append(True)
        raise RuntimeError("synthetic metadata failure")

    monkeypatch.setattr(JobArtifact.objects, "update_or_create", fail)
    with pytest.raises(RuntimeError, match="metadata failure"):
        publish(job, payload=b"NEW", source_path=source if file_upload else None)
    assert called == [True]
    original.refresh_from_db()
    assert original.storage_key == original_key
    store = get_artifact_store()
    assert store.read_bytes(key=original_key) == b"OLD"
    assert store.list_keys(prefix="voxhelm/jobs/") == [original_key]
    assert not PendingArtifactDeletion.objects.exists()


@pytest.mark.django_db(transaction=True)
def test_successful_replacement_removes_only_unreferenced_previous_object():
    job = make_job()
    old = publish(job)
    old_key = old.storage_key
    new = publish(job, b"NEW")
    store = get_artifact_store()
    assert new.storage_key != old_key
    assert store.read_bytes(key=new.storage_key) == b"NEW"
    assert store.list_keys(prefix="voxhelm/jobs/") == [new.storage_key]
    # Shared references preserve the older allocation.
    other = make_job()
    shared = JobArtifact.objects.get(pk=new.pk)
    shared.pk = None
    shared.job = other
    shared.save()
    last = publish(job, b"LAST")
    assert store.read_bytes(key=new.storage_key) == b"NEW"
    assert store.read_bytes(key=last.storage_key) == b"LAST"


@pytest.mark.django_db(transaction=True)
def test_stale_publication_cleans_only_its_allocation():
    job = make_job(state="running", django_task_id="old")
    old = publish(job)
    Job.objects.filter(pk=job.pk).update(django_task_id="new")
    with pytest.raises(RuntimeError, match="no longer owns"):
        publish(job, b"STALE")
    old.refresh_from_db()
    assert get_artifact_store().read_bytes(key=old.storage_key) == b"OLD"
    assert get_artifact_store().list_keys(prefix="voxhelm/jobs/") == [old.storage_key]


@pytest.mark.django_db
def test_running_backend_cannot_consume_execution_claim(monkeypatch):
    job = make_job(django_task_id="expected")
    monkeypatch.setattr(
        services,
        "default_task_backend",
        SimpleNamespace(
            supports_get_result=True,
            get_result=lambda _: SimpleNamespace(
                status=TaskResultStatus.RUNNING,
                started_at=timezone.now(),
                finished_at=None,
                errors=[],
            ),
        ),
    )
    services.reconcile_job_state(job)
    claimed = services.initialize_running_job(job_id=str(job.id), task_result_id="expected")
    assert claimed is not None
    assert claimed.state == "running"


@pytest.mark.django_db(transaction=True)
def test_failed_cleanup_retries_against_recorded_store(settings, monkeypatch, tmp_path):
    from jobs.retention import drain_pending_artifact_deletions

    job = make_job()
    old = publish(job)
    # Legacy rows used an empty identity to mean the active store. Pin it now.
    old.storage_identity = {}
    old.save(update_fields=["storage_identity"])
    old_key = old.storage_key
    store = get_artifact_store()
    old_root = settings.VOXHELM_ARTIFACT_ROOT

    def fail_delete(**kwargs):
        raise OSError("synthetic deletion failure")

    monkeypatch.setattr(store, "delete", fail_delete)
    new = publish(job, b"NEW")
    assert PendingArtifactDeletion.objects.filter(storage_key=old_key).exists()
    assert store.read_bytes(key=old_key) == b"OLD"
    settings.VOXHELM_ARTIFACT_ROOT = tmp_path / "other-store"
    get_artifact_store.cache_clear()
    assert drain_pending_artifact_deletions() == {"resolved": 1, "refused": 0, "failed": 0}
    assert not (old_root / old_key).exists()
    assert (old_root / new.storage_key).read_bytes() == b"NEW"
