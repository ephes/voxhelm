from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, cast

import pytest
from asgiref.local import Local
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management import call_command
from django.test import Client
from django_tasks import task_backends

from jobs.media import DownloadedMedia
from jobs.models import Job, JobArtifact
from transcriptions.service import TranscribeParams, TranscriptionResult, TranscriptionSegment


class DummyBackend:
    def transcribe(self, audio_path: Path, params: TranscribeParams) -> TranscriptionResult:
        del audio_path, params
        return TranscriptionResult(
            text="Operator transcript",
            language="en",
            segments=[
                TranscriptionSegment(id=0, start=0.0, end=1.25, text="Operator"),
                TranscriptionSegment(id=1, start=1.25, end=2.5, text="transcript"),
            ],
        )


def configure_task_backend(settings, backend: str) -> None:
    settings.TASKS = {"default": {"BACKEND": backend}}
    handler = cast(Any, task_backends)
    connections = handler._connections
    handler._connections = Local(connections._thread_critical)


@pytest.mark.django_db
def test_root_renders_login_page_for_anonymous_user(client):
    response = client.get("/")

    assert response.status_code == 200
    body = response.content.decode()
    assert "Sign In" in body
    assert "Submit Transcript" not in body


@pytest.mark.django_db
def test_authenticated_root_renders_operator_home(client):
    user = get_user_model().objects.create_user(username="jochen", password="secret", is_staff=True)
    client.force_login(user)

    response = client.get("/")

    assert response.status_code == 200
    assert "Submit Transcript" in response.content.decode()


@pytest.mark.django_db
def test_login_succeeds_behind_https_proxy_with_csrf_checks(settings):
    host = "voxhelm.home.xn--wersdrfer-47a.de"
    settings.ALLOWED_HOSTS = [host, "testserver"]
    settings.CSRF_TRUSTED_ORIGINS = [f"https://{host}"]
    settings.SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
    get_user_model().objects.create_user(username="jochen", password="secret", is_staff=True)
    client = Client(enforce_csrf_checks=True)

    response = client.get("/", HTTP_HOST=host, HTTP_X_FORWARDED_PROTO="https")

    assert response.status_code == 200
    csrf_token = response.cookies["csrftoken"].value

    response = client.post(
        "/",
        data={
            "username": "jochen",
            "password": "secret",
            "csrfmiddlewaretoken": csrf_token,
        },
        HTTP_HOST=host,
        HTTP_ORIGIN=f"https://{host}",
        HTTP_REFERER=f"https://{host}/",
        HTTP_X_FORWARDED_PROTO="https",
    )

    assert response.status_code == 302
    assert response.headers["Location"] == "/"


@pytest.mark.django_db
def test_uploaded_audio_submission_runs_sync_path(client, monkeypatch):
    user = get_user_model().objects.create_user(username="jochen", password="secret", is_staff=True)
    client.force_login(user)
    monkeypatch.setattr("transcriptions.service.get_backend_service", lambda: DummyBackend())
    upload = SimpleUploadedFile("operator.mp3", b"mp3-bytes", content_type="audio/mpeg")

    response = client.post("/", data={"audio_file": upload}, follow=True)

    assert response.status_code == 200
    job = Job.objects.get(operator=user)
    assert job.dispatch_mode == Job.DispatchMode.SYNC
    assert job.input_data["kind"] == "upload"
    body = response.content.decode()
    assert "Operator transcript" in body
    assert f"/transcripts/{job.id}/artifacts/dote" in body
    assert f"/transcripts/{job.id}/artifacts/podlove" in body


@pytest.mark.django_db
def test_audio_url_submission_runs_sync_path(client, monkeypatch):
    user = get_user_model().objects.create_user(username="jochen", password="secret", is_staff=True)
    client.force_login(user)
    monkeypatch.setattr("transcriptions.service.get_backend_service", lambda: DummyBackend())
    temp_file = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name)
    temp_file.write_bytes(b"mp3-bytes")

    def fake_download(*, source_url: str) -> Path:
        assert source_url == "https://media.example.com/episode.mp3"
        return temp_file

    monkeypatch.setattr("operators.views.download_allowed_url_to_tempfile", fake_download)

    response = client.post(
        "/",
        data={"audio_url": "https://media.example.com/episode.mp3"},
        follow=True,
    )

    assert response.status_code == 200
    job = Job.objects.get(operator=user)
    assert job.dispatch_mode == Job.DispatchMode.SYNC
    assert job.input_data["url"] == "https://media.example.com/episode.mp3"


@pytest.mark.django_db
def test_video_url_submission_runs_batch_path(client, settings, monkeypatch, tmp_path):
    user = get_user_model().objects.create_user(username="jochen", password="secret", is_staff=True)
    client.force_login(user)
    configure_task_backend(settings, "django_tasks.backends.immediate.ImmediateBackend")
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video-bytes")
    extracted_path = tmp_path / "clip.wav"
    extracted_path.write_bytes(b"wav-bytes")

    monkeypatch.setattr(
        "jobs.services.download_allowed_media",
        lambda *, source_url: DownloadedMedia(
            path=video_path,
            content_type="video/mp4",
            source_url=source_url,
        ),
    )
    monkeypatch.setattr(
        "jobs.services.extract_audio_from_video",
        lambda *, source_path: extracted_path,
    )
    monkeypatch.setattr("transcriptions.service.get_backend_service", lambda: DummyBackend())

    response = client.post(
        "/",
        data={"video_url": "https://media.example.com/clip.mp4"},
        follow=True,
    )

    assert response.status_code == 200
    job = Job.objects.get(operator=user)
    assert job.dispatch_mode == Job.DispatchMode.BATCH
    assert job.producer == "__operator_ui__"
    body = response.content.decode()
    assert "Batch route" in body
    assert "Operator transcript" in body


@pytest.mark.django_db
def test_recent_transcripts_list_is_scoped_to_authenticated_operator(client):
    user_model = get_user_model()
    jochen = user_model.objects.create_user(username="jochen", password="secret", is_staff=True)
    other = user_model.objects.create_user(username="other", password="secret", is_staff=True)
    Job.objects.create(
        producer="__operator_ui__",
        operator=jochen,
        job_type=Job.JobType.TRANSCRIBE,
        lane=Job.Lane.BATCH,
        dispatch_mode=Job.DispatchMode.SYNC,
        priority=Job.Priority.NORMAL,
        backend="auto",
        model="gpt-4o-mini-transcribe",
        input_data={"kind": "upload", "filename": "jochen.mp3"},
        output_data={"formats": ["text"]},
        context_data={},
        state=Job.State.SUCCEEDED,
        result_text="Visible transcript",
    )
    Job.objects.create(
        producer="__operator_ui__",
        operator=other,
        job_type=Job.JobType.TRANSCRIBE,
        lane=Job.Lane.BATCH,
        dispatch_mode=Job.DispatchMode.SYNC,
        priority=Job.Priority.NORMAL,
        backend="auto",
        model="gpt-4o-mini-transcribe",
        input_data={"kind": "upload", "filename": "other.mp3"},
        output_data={"formats": ["text"]},
        context_data={},
        state=Job.State.SUCCEEDED,
        result_text="Hidden transcript",
    )
    client.force_login(jochen)

    response = client.get("/")

    body = response.content.decode()
    assert response.status_code == 200
    assert "jochen.mp3" in body
    assert "other.mp3" not in body


@pytest.mark.django_db
def test_bootstrap_operator_command_creates_jochen_account():
    call_command("bootstrap_operator", username="jochen", password="secret")
    user = get_user_model().objects.get(username="jochen")

    assert user.is_staff is True
    assert user.check_password("secret") is True


@pytest.mark.django_db
def test_root_forbids_non_staff_user(client):
    user = get_user_model().objects.create_user(
        username="jochen",
        password="secret",
        is_staff=False,
    )
    client.force_login(user)

    response = client.get("/")

    assert response.status_code == 403
    assert "staff account" in response.content.decode()


@pytest.mark.django_db
def test_operator_artifact_rejects_other_operator_job(client):
    user_model = get_user_model()
    jochen = user_model.objects.create_user(username="jochen", password="secret", is_staff=True)
    other = user_model.objects.create_user(username="other", password="secret", is_staff=True)
    job = Job.objects.create(
        producer="__operator_ui__",
        operator=other,
        job_type=Job.JobType.TRANSCRIBE,
        lane=Job.Lane.BATCH,
        dispatch_mode=Job.DispatchMode.SYNC,
        priority=Job.Priority.NORMAL,
        backend="auto",
        model="gpt-4o-mini-transcribe",
        input_data={"kind": "upload", "filename": "other.mp3"},
        output_data={"formats": ["text"]},
        context_data={},
        state=Job.State.SUCCEEDED,
        result_text="Hidden transcript",
    )
    JobArtifact.objects.create(
        job=job,
        name="transcript.txt",
        kind=JobArtifact.Kind.TRANSCRIPT_TEXT,
        format="text",
        storage_backend="filesystem",
        storage_key="voxhelm/jobs/test/transcript.txt",
        content_type="text/plain; charset=utf-8",
        size_bytes=3,
        exposed=True,
    )
    client.force_login(jochen)

    response = client.get(f"/transcripts/{job.id}/artifacts/text")

    assert response.status_code == 404


@pytest.mark.django_db
def test_form_rejects_multiple_inputs(client):
    user = get_user_model().objects.create_user(username="jochen", password="secret", is_staff=True)
    client.force_login(user)

    response = client.post(
        "/",
        data={
            "audio_url": "https://media.example.com/episode.mp3",
            "video_url": "https://media.example.com/clip.mp4",
        },
    )

    assert response.status_code == 200
    assert "Submit only one input at a time." in response.content.decode()
    assert Job.objects.count() == 0


@pytest.mark.django_db
def test_bootstrap_operator_command_updates_existing_password():
    user_model = get_user_model()
    user_model.objects.create_user(username="jochen", password="old", is_staff=False)

    call_command("bootstrap_operator", username="jochen", password="new")

    user = user_model.objects.get(username="jochen")
    assert user.is_staff is True
    assert user.check_password("new") is True


@pytest.mark.django_db
def test_logout_view_logs_user_out(client):
    user = get_user_model().objects.create_user(username="jochen", password="secret", is_staff=True)
    client.force_login(user)

    response = client.post("/logout", follow=True)

    assert response.status_code == 200
    assert "_auth_user_id" not in client.session
    assert "Sign In" in response.content.decode()


@pytest.mark.django_db
def test_invalid_job_query_parameter_is_ignored(client):
    user = get_user_model().objects.create_user(username="jochen", password="secret", is_staff=True)
    client.force_login(user)

    response = client.get("/?job=not-a-uuid")

    assert response.status_code == 200
    assert "No transcripts yet." in response.content.decode()
