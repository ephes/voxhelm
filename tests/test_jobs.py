from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from asgiref.local import Local
from django_tasks import task_backends

from jobs.media import DownloadedMedia
from jobs.models import Job, JobArtifact
from transcriptions.service import TranscribeParams, TranscriptionResult, TranscriptionSegment


class DummyBackend:
    def transcribe(self, audio_path: Path, params: TranscribeParams) -> TranscriptionResult:
        del audio_path, params
        return TranscriptionResult(
            text="Batch hello world",
            language="en",
            segments=[
                TranscriptionSegment(id=0, start=0.0, end=1.0, text="Batch hello"),
                TranscriptionSegment(id=1, start=1.0, end=2.0, text="world"),
            ],
        )


def configure_task_backend(settings, backend: str) -> None:
    settings.TASKS = {"default": {"BACKEND": backend}}
    handler = cast(Any, task_backends)
    connections = handler._connections
    handler._connections = Local(connections._thread_critical)


def build_job_payload(url: str = "https://media.example.com/episode.mp3") -> dict[str, object]:
    return {
        "job_type": "transcribe",
        "priority": "normal",
        "lane": "batch",
        "backend": "auto",
        "model": "auto",
        "language": "en",
        "input": {"kind": "url", "url": url},
        "output": {"formats": ["text", "json"]},
        "context": {"producer": "archive", "item_id": 123},
        "task_ref": "archive-item-123",
    }


@pytest.mark.django_db
def test_jobs_endpoint_requires_bearer_token(client):
    response = client.post(
        "/v1/jobs",
        data=json.dumps(build_job_payload()),
        content_type="application/json",
    )

    assert response.status_code == 401
    assert response.json()["error"]["type"] == "authentication_error"


@pytest.mark.django_db
def test_create_job_queued_with_dummy_backend(client, settings):
    configure_task_backend(settings, "django_tasks.backends.dummy.DummyBackend")
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}

    response = client.post(
        "/v1/jobs",
        data=json.dumps(build_job_payload()),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    payload = response.json()
    assert response.status_code == 201
    assert payload["state"] == "queued"
    job = Job.objects.get(id=payload["id"])
    assert job.django_task_id


@pytest.mark.django_db
def test_job_submission_is_idempotent(client, settings):
    configure_task_backend(settings, "django_tasks.backends.dummy.DummyBackend")
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}
    body = json.dumps(build_job_payload())

    first = client.post(
        "/v1/jobs",
        data=body,
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )
    second = client.post(
        "/v1/jobs",
        data=body,
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert first.status_code == 201
    assert second.status_code == 200
    assert first.json()["id"] == second.json()["id"]
    assert Job.objects.count() == 1


@pytest.mark.django_db
def test_immediate_backend_executes_job_and_serves_artifacts(
    client, settings, monkeypatch, tmp_path
):
    configure_task_backend(settings, "django_tasks.backends.immediate.ImmediateBackend")
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}
    media_path = tmp_path / "episode.mp3"
    media_path.write_bytes(b"mp3-bytes")
    monkeypatch.setattr(
        "jobs.services.download_allowed_media",
        lambda *, source_url: DownloadedMedia(
            path=media_path,
            content_type="audio/mpeg",
            source_url=source_url,
        ),
    )
    monkeypatch.setattr("transcriptions.service.get_backend_service", lambda: DummyBackend())

    response = client.post(
        "/v1/jobs",
        data=json.dumps(build_job_payload()),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    payload = response.json()
    assert response.status_code == 201
    assert payload["state"] == "succeeded"
    assert payload["result"]["text"] == "Batch hello world"
    assert payload["result"]["artifacts"]["text"].endswith("/transcript.txt")
    assert payload["result"]["artifacts"]["json"].endswith("/transcript.json")

    artifact_response = client.get(
        payload["result"]["artifacts"]["text"],
        HTTP_AUTHORIZATION="Bearer test-token",
    )
    assert artifact_response.status_code == 200
    assert artifact_response.content.decode() == "Batch hello world"

    json_response = client.get(
        payload["result"]["artifacts"]["json"],
        HTTP_AUTHORIZATION="Bearer test-token",
    )
    assert json_response.status_code == 200
    assert json_response.json()["segments"][0]["text"] == "Batch hello"


@pytest.mark.django_db
def test_video_job_extracts_audio_before_transcription(client, settings, monkeypatch, tmp_path):
    configure_task_backend(settings, "django_tasks.backends.immediate.ImmediateBackend")
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video-bytes")
    extracted_path = tmp_path / "clip.wav"
    extracted_path.write_bytes(b"wav-bytes")
    extracted_calls: list[Path] = []

    monkeypatch.setattr(
        "jobs.services.download_allowed_media",
        lambda *, source_url: DownloadedMedia(
            path=video_path,
            content_type="video/mp4",
            source_url=source_url,
        ),
    )

    def fake_extract(*, source_path: Path) -> Path:
        extracted_calls.append(source_path)
        return extracted_path

    monkeypatch.setattr("jobs.services.extract_audio_from_video", fake_extract)
    monkeypatch.setattr("transcriptions.service.get_backend_service", lambda: DummyBackend())

    response = client.post(
        "/v1/jobs",
        data=json.dumps(build_job_payload(url="https://media.example.com/clip.mp4")),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    payload = response.json()
    assert response.status_code == 201
    assert payload["state"] == "succeeded"
    assert extracted_calls == [video_path]
    artifact_kinds = set(JobArtifact.objects.values_list("kind", flat=True))
    assert JobArtifact.Kind.EXTRACTED_AUDIO in artifact_kinds


@pytest.mark.django_db
def test_failed_job_records_error(client, settings, monkeypatch, tmp_path):
    configure_task_backend(settings, "django_tasks.backends.immediate.ImmediateBackend")
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}
    media_path = tmp_path / "episode.mp3"
    media_path.write_bytes(b"mp3-bytes")
    monkeypatch.setattr(
        "jobs.services.download_allowed_media",
        lambda *, source_url: DownloadedMedia(
            path=media_path,
            content_type="audio/mpeg",
            source_url=source_url,
        ),
    )

    class FailingBackend:
        def transcribe(self, audio_path: Path, params: TranscribeParams) -> TranscriptionResult:
            del audio_path, params
            raise RuntimeError("backend exploded")

    monkeypatch.setattr("transcriptions.service.get_backend_service", lambda: FailingBackend())

    response = client.post(
        "/v1/jobs",
        data=json.dumps(build_job_payload()),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    payload = response.json()
    assert response.status_code == 201
    assert payload["state"] == "failed"
    assert payload["error"]["message"] == "backend exploded"
