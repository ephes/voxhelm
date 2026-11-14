"""Synthetic HTTP admission and account publication regressions."""

import json
from io import StringIO
from uuid import uuid4

import pytest
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management import call_command
from django_tasks_db.models import DBTaskResult

from jobs.models import Job, JobArtifact, StagedMedia, Worker


@pytest.mark.django_db
@pytest.mark.parametrize(
    "route", ["jobs", "workers/heartbeat", "work/claim", "heartbeat", "complete"]
)
@pytest.mark.parametrize(
    "body",
    [
        b'{"task_ref":"first","task_ref":"second"}',
        b'{"nested":{"job_id":"first","job_id":"second"}}',
        b'{"n":' + b"9" * 5000 + b"}",
        b'{"n":' + b"[" * 1100 + b"0" + b"]" * 1100 + b"}",
        b'{"n":NaN}',
        b'{"n":1e999}',
    ],
)
def test_batch_and_worker_json_is_unambiguous(client, settings, route, body):
    settings.VOXHELM_WORKER_TOKENS = {"synthetic": "worker-token"}
    if route == "jobs":
        path, token = "/v1/jobs", "test-token"
    elif route in {"heartbeat", "complete"}:
        path, token = f"/v1/internal/work/{uuid4()}/{route}", "worker-token"
    else:
        path, token = f"/v1/internal/{route}", "worker-token"
    response = client.post(
        path, data=body, content_type="application/json", HTTP_AUTHORIZATION=f"Bearer {token}"
    )
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "Request body was not valid JSON."
    for model in (Job, JobArtifact, StagedMedia, Worker, DBTaskResult):
        assert model.objects.count() == 0


@pytest.mark.django_db
def test_json_byte_budget_and_auth_before_read(client, settings, monkeypatch):
    from jobs import views

    monkeypatch.setattr(views, "JSON_REQUEST_MAX_BYTES", 32, raising=False)
    body = json.dumps({"text": "x" * 33})
    for token, expected in [("wrong", 401), ("test-token", 413)]:
        response = client.post(
            "/v1/jobs",
            data=body,
            content_type="application/json",
            HTTP_AUTHORIZATION=f"Bearer {token}",
        )
        assert response.status_code == expected
    assert not Job.objects.exists()


@pytest.mark.django_db
@pytest.mark.parametrize(
    "route,seam", [("/v1/jobs", "create_job_from_payload"), ("/v1/uploads", "stage_uploaded_audio")]
)
def test_batch_backend_failure_is_private(client, monkeypatch, caplog, route, seam):
    from jobs import views

    canary = "https://secret:synthetic-token@private.invalid/private/path"

    def fail(**kwargs):
        raise RuntimeError(canary)

    monkeypatch.setattr(views, seam, fail)
    if route.endswith("uploads"):
        data = {"file": SimpleUploadedFile("synthetic.mp3", b"synthetic", "audio/mpeg")}
        response = client.post(route, data=data, HTTP_AUTHORIZATION="Bearer test-token")
    else:
        response = client.post(
            route,
            data="{}",
            content_type="application/json",
            HTTP_AUTHORIZATION="Bearer test-token",
        )
    assert response.status_code == 500
    assert response.json()["error"]["message"] == "Batch processing failed."
    assert canary not in response.content.decode() + caplog.text
    assert "batch_backend_failure" in caplog.text
    assert not StagedMedia.objects.exists()


@pytest.mark.django_db
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure", ["hash", "save"])
def test_bootstrap_failure_never_publishes_partial_account(monkeypatch, existing, failure):
    user_model = get_user_model()
    if existing:
        user_model.objects.create_user(
            username="synthetic",
            email="old@example.invalid",
            password="old",
            is_staff=False,
            is_active=False,
        )
    before = list(user_model.objects.values())
    original_save = user_model.save

    def fail_hash(self, raw):
        raise RuntimeError("synthetic hashing failure")

    def fail_save(self, *args, **kwargs):
        original_save(self, *args, **kwargs)
        raise RuntimeError("synthetic save failure")

    monkeypatch.setattr(
        user_model,
        "set_password" if failure == "hash" else "save",
        fail_hash if failure == "hash" else fail_save,
    )
    with pytest.raises(RuntimeError):
        call_command(
            "bootstrap_operator",
            username="synthetic",
            email="new@example.invalid",
            password=" new password ",
            stdout=StringIO(),
        )
    assert list(user_model.objects.values()) == before


@pytest.mark.django_db(transaction=True)
def test_upload_backend_runtime_failure_cleans_owned_staging(client, settings, monkeypatch, caplog):
    from jobs.artifacts import FilesystemArtifactStore

    canary = "https://synthetic:token@private.invalid/storage"

    def fail(self, *, key, source_path, content_type):
        target = self.root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"partial synthetic upload")
        raise RuntimeError(canary)

    monkeypatch.setattr(FilesystemArtifactStore, "put_file", fail)
    response = client.post(
        "/v1/uploads",
        data={"file": SimpleUploadedFile("synthetic.mp3", b"audio", "audio/mpeg")},
        HTTP_AUTHORIZATION="Bearer test-token",
    )
    assert response.status_code == 500
    assert canary not in response.content.decode() + caplog.text
    assert not StagedMedia.objects.exists()
    assert not [p for p in settings.VOXHELM_ARTIFACT_ROOT.rglob("*") if p.is_file()]


@pytest.mark.django_db(transaction=True)
def test_failed_staging_compensation_keeps_record_without_backend_text(
    client, settings, monkeypatch, caplog
):
    from jobs.artifacts import FilesystemArtifactStore

    canary = "https://synthetic:token@private.invalid/storage"

    def fail_put(self, *, key, source_path, content_type):
        target = self.root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"partial synthetic upload")
        raise RuntimeError(canary)

    def fail_delete(self, *, key):
        raise RuntimeError(canary)

    monkeypatch.setattr(FilesystemArtifactStore, "put_file", fail_put)
    monkeypatch.setattr(FilesystemArtifactStore, "delete", fail_delete)
    response = client.post(
        "/v1/uploads",
        data={"file": SimpleUploadedFile("synthetic.mp3", b"audio", "audio/mpeg")},
        HTTP_AUTHORIZATION="Bearer test-token",
    )
    assert response.status_code == 500
    assert canary not in response.content.decode() + caplog.text
    staged = StagedMedia.objects.get()
    assert (settings.VOXHELM_ARTIFACT_ROOT / staged.storage_key).is_file()


@pytest.mark.django_db
def test_task_backend_runtime_failure_rolls_back_submission(client, settings, monkeypatch, caplog):
    from jobs import tasks
    from tests.test_jobs import build_job_payload

    canary = "https://synthetic:token@private.invalid/task-backend"
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}

    class FailingTask:
        def using(self, **kwargs):
            return self

        def enqueue(self, job_id):
            raise RuntimeError(canary)

    monkeypatch.setattr(tasks, "run_transcription_job", FailingTask())
    response = client.post(
        "/v1/jobs",
        data=json.dumps(build_job_payload()),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )
    assert response.status_code == 500
    assert canary not in response.content.decode() + caplog.text
    assert not Job.objects.exists()
    assert not DBTaskResult.objects.exists()


@pytest.mark.django_db
def test_duplicate_task_identity_cannot_create_job(client, settings):
    from tests.test_remote_workers import build_job_payload, enable_remote_workers

    enable_remote_workers(settings)
    raw = json.dumps(build_job_payload()).removesuffix("}") + ',"task_ref":"other"}'
    response = client.post(
        "/v1/jobs",
        data=raw,
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )
    assert response.status_code == 400
    assert not Job.objects.exists()


@pytest.mark.django_db
@pytest.mark.parametrize("operation", ["heartbeat", "complete"])
def test_ambiguous_payload_cannot_mutate_owned_lease(client, settings, operation):
    from tests.test_remote_workers import (
        claim_one,
        store_manifest_artifacts,
        submit_remote_job,
        transcript_manifest,
        worker_headers,
    )

    job = submit_remote_job(client, settings)
    claim = claim_one(client).json()["job"]
    manifest = transcript_manifest(str(job.pk), claim["attempt"])
    store_manifest_artifacts(settings, manifest)
    payload = {
        "worker_id": "atlas",
        "lease_token": claim["lease_token"],
        "result_text": "synthetic transcript",
        "artifacts": manifest,
    }
    raw = (
        json.dumps(payload).removesuffix("}")
        + ',"lease_token":'
        + json.dumps(claim["lease_token"])
        + "}"
    )
    before = list(Job.objects.values()), list(Worker.objects.values())
    response = client.post(
        f"/v1/internal/work/{job.pk}/{operation}",
        data=raw,
        content_type="application/json",
        **worker_headers(),
    )
    assert response.status_code == 400
    assert (list(Job.objects.values()), list(Worker.objects.values())) == before
    assert not JobArtifact.objects.exists()
