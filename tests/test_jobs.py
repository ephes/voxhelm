from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from asgiref.local import Local
from django.core.files.uploadedfile import SimpleUploadedFile
from django.db import IntegrityError, connection, connections, transaction
from django.db.migrations.executor import MigrationExecutor
from django.utils import timezone
from django_tasks import task_backends

from jobs import services
from jobs.artifacts import get_artifact_store
from jobs.fingerprints import stored_job_request_fingerprint
from jobs.media import DownloadedMedia
from jobs.models import Job, JobArtifact, StagedMedia
from transcriptions.diarization import DiarizationParams, SpeakerTurn
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


def build_job_payload(
    url: str = "https://media.example.com/episode.mp3",
    *,
    input_data: dict[str, object] | None = None,
    task_ref: str = "archive-item-123",
) -> dict[str, object]:
    return {
        "job_type": "transcribe",
        "priority": "normal",
        "lane": "batch",
        "backend": "auto",
        "model": "auto",
        "language": "en",
        "input": input_data or {"kind": "url", "url": url},
        "output": {"formats": ["text", "json"]},
        "context": {"producer": "archive", "item_id": 123},
        "task_ref": task_ref,
    }


def build_synthesis_payload(text: str = "Hello from Voxhelm") -> dict[str, object]:
    return {
        "job_type": "synthesize",
        "priority": "normal",
        "lane": "batch",
        "backend": "auto",
        "model": "tts-1",
        "language": "en",
        "voice": "en_US-lessac-medium",
        "input": {"kind": "text", "text": text},
        "output": {"formats": ["wav"]},
        "context": {"producer": "archive", "item_id": 456},
        "task_ref": "archive-item-456-audio-v1",
    }


def stage_upload(
    client,
    *,
    name: str = "episode.mp3",
    content: bytes = b"mp3-bytes",
    content_type: str,
):
    return client.post(
        "/v1/uploads",
        data={"file": SimpleUploadedFile(name, content, content_type=content_type)},
        HTTP_AUTHORIZATION="Bearer test-token",
    )


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
    assert job.output_data["diarization"] == {"enabled": False}


@pytest.mark.django_db
def test_task_ref_reuse_survives_transcription_execution_mode_switch(client, settings):
    configure_task_backend(settings, "django_tasks.backends.dummy.DummyBackend")
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}
    settings.VOXHELM_TRANSCRIPTION_EXECUTION_MODE = "remote_pull"
    payload = build_job_payload(task_ref="archive-item-mode-switch")

    remote_response = client.post(
        "/v1/jobs",
        data=json.dumps(payload),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )
    settings.VOXHELM_TRANSCRIPTION_EXECUTION_MODE = "django_tasks"
    local_response = client.post(
        "/v1/jobs",
        data=json.dumps(payload),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert remote_response.status_code == 201
    assert local_response.status_code == 200
    assert local_response.json()["id"] == remote_response.json()["id"]
    remote_job = Job.objects.get(id=remote_response.json()["id"])
    assert remote_job.execution_mode == Job.ExecutionMode.REMOTE_PULL
    assert not remote_job.django_task_id
    assert Job.objects.count() == 1


@pytest.mark.django_db
@pytest.mark.parametrize(
    "diarization",
    [
        True,
        [],
        {},
        {"enabled": "true"},
        {"enabled": 1},
        {"enabled": None},
        {"enabled": True, "num_speakers": 0},
        {"enabled": True, "num_speakers": True},
        {"enabled": True, "num_speaker": 2},
        {"enabled": False, "num_speakers": 2},
        {"enabled": True, "num_speakers": 2, "min_speakers": 1},
        {"enabled": True, "min_speakers": 4, "max_speakers": 2},
    ],
)
def test_transcription_job_rejects_malformed_diarization_option(
    client,
    settings,
    diarization,
):
    configure_task_backend(settings, "django_tasks.backends.dummy.DummyBackend")
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}
    payload = build_job_payload(task_ref="archive-item-diarization-malformed")
    payload["diarization"] = diarization

    response = client.post(
        "/v1/jobs",
        data=json.dumps(payload),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert response.status_code == 400
    assert "diarization" in response.json()["error"]["message"]


@pytest.mark.django_db
def test_transcription_job_accepts_disabled_diarization_option(client, settings):
    configure_task_backend(settings, "django_tasks.backends.dummy.DummyBackend")
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}
    payload = build_job_payload(task_ref="archive-item-diarization-disabled")
    payload["diarization"] = {"enabled": False}

    response = client.post(
        "/v1/jobs",
        data=json.dumps(payload),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert response.status_code == 201
    job = Job.objects.get(id=response.json()["id"])
    assert job.output_data["diarization"] == {"enabled": False}


@pytest.mark.django_db
def test_transcription_job_accepts_diarization_speaker_hints(client, settings):
    configure_task_backend(settings, "django_tasks.backends.dummy.DummyBackend")
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}
    payload = build_job_payload(task_ref="archive-item-diarization-hints")
    payload["diarization"] = {"enabled": True, "min_speakers": 2, "max_speakers": 4}

    response = client.post(
        "/v1/jobs",
        data=json.dumps(payload),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert response.status_code == 201
    job = Job.objects.get(id=response.json()["id"])
    assert job.output_data["diarization"] == {
        "enabled": True,
        "min_speakers": 2,
        "max_speakers": 4,
    }


@pytest.mark.django_db
def test_known_speaker_reference_urls_must_be_allowlisted(client, settings):
    configure_task_backend(settings, "django_tasks.backends.dummy.DummyBackend")
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}
    payload = build_job_payload(task_ref="archive-item-known-speaker-blocked-reference")
    payload["diarization"] = {
        "enabled": True,
        "strategy": "pyannote_known_speaker",
        "known_speakers": [
            {
                "id": "12",
                "name": "Johannes",
                "references": [
                    {
                        "kind": "source_range",
                        "audio": {
                            "kind": "url",
                            "url": "https://blocked.example.com/reference.m4a",
                        },
                        "start": 0.0,
                        "end": 2.0,
                    }
                ],
            }
        ],
    }

    response = client.post(
        "/v1/jobs",
        data=json.dumps(payload),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert response.status_code == 400
    assert "allowlist" in response.json()["error"]["message"]
    assert not Job.objects.exists()


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
def test_task_ref_retry_returns_existing_job_before_revalidating_url_allowlist(client, settings):
    configure_task_backend(settings, "django_tasks.backends.dummy.DummyBackend")
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}
    body = json.dumps(build_job_payload(task_ref="archive-item-allowlist-retry"))

    first = client.post(
        "/v1/jobs",
        data=body,
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"other.example.com"}
    retry = client.post(
        "/v1/jobs",
        data=body,
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert first.status_code == 201
    assert retry.status_code == 200
    assert first.json()["id"] == retry.json()["id"]
    assert Job.objects.count() == 1


@pytest.mark.django_db
def test_job_submission_deduplicates_output_formats_regardless_of_order(client, settings):
    configure_task_backend(settings, "django_tasks.backends.dummy.DummyBackend")
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}
    first_payload = build_job_payload(task_ref="archive-item-output-order-idempotency")
    first_payload["output"] = {"formats": ["json", "podlove"]}
    second_payload = build_job_payload(task_ref="archive-item-output-order-idempotency")
    second_payload["output"] = {"formats": ["podlove", "json"]}

    first = client.post(
        "/v1/jobs",
        data=json.dumps(first_payload),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )
    second = client.post(
        "/v1/jobs",
        data=json.dumps(second_payload),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert first.status_code == 201
    assert second.status_code == 200
    assert first.json()["id"] == second.json()["id"]
    assert Job.objects.count() == 1


@pytest.mark.django_db
def test_job_submission_with_same_task_ref_creates_new_job_for_different_output_formats(
    client,
    settings,
):
    configure_task_backend(settings, "django_tasks.backends.dummy.DummyBackend")
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}
    first_payload = build_job_payload(task_ref="archive-item-output-idempotency")
    first_payload["output"] = {"formats": ["json"]}
    second_payload = build_job_payload(task_ref="archive-item-output-idempotency")
    second_payload["output"] = {"formats": ["json", "podlove"]}

    first = client.post(
        "/v1/jobs",
        data=json.dumps(first_payload),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )
    second = client.post(
        "/v1/jobs",
        data=json.dumps(second_payload),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )
    third = client.post(
        "/v1/jobs",
        data=json.dumps(second_payload),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert first.status_code == 201
    assert second.status_code == 201
    assert third.status_code == 200
    assert first.json()["id"] != second.json()["id"]
    assert third.json()["id"] == second.json()["id"]
    assert Job.objects.count() == 2


@pytest.mark.django_db
def test_job_submission_with_same_task_ref_deduplicates_interleaved_payloads(client, settings):
    configure_task_backend(settings, "django_tasks.backends.dummy.DummyBackend")
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}
    first_payload = build_job_payload(task_ref="archive-item-interleaved-idempotency")
    first_payload["output"] = {"formats": ["json"]}
    second_payload = build_job_payload(task_ref="archive-item-interleaved-idempotency")
    second_payload["output"] = {"formats": ["json", "podlove"]}

    first = client.post(
        "/v1/jobs",
        data=json.dumps(first_payload),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )
    second = client.post(
        "/v1/jobs",
        data=json.dumps(second_payload),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )
    third = client.post(
        "/v1/jobs",
        data=json.dumps(first_payload),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert first.status_code == 201
    assert second.status_code == 201
    assert third.status_code == 200
    assert first.json()["id"] != second.json()["id"]
    assert third.json()["id"] == first.json()["id"]
    assert Job.objects.count() == 2


@pytest.mark.django_db
def test_job_submission_with_same_task_ref_creates_new_job_for_different_diarization(
    client,
    settings,
):
    configure_task_backend(settings, "django_tasks.backends.dummy.DummyBackend")
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}
    first_payload = build_job_payload(task_ref="archive-item-diarization-idempotency")
    second_payload = build_job_payload(task_ref="archive-item-diarization-idempotency")
    second_payload["diarization"] = {"enabled": True, "num_speakers": 4}

    first = client.post(
        "/v1/jobs",
        data=json.dumps(first_payload),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )
    second = client.post(
        "/v1/jobs",
        data=json.dumps(second_payload),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )
    third = client.post(
        "/v1/jobs",
        data=json.dumps(second_payload),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert first.status_code == 201
    assert second.status_code == 201
    assert third.status_code == 200
    assert first.json()["id"] != second.json()["id"]
    assert third.json()["id"] == second.json()["id"]
    assert Job.objects.count() == 2


@pytest.mark.django_db
def test_batch_upload_staging_contract_accepts_large_audio_file(client, settings):
    settings.VOXHELM_MAX_UPLOAD_BYTES = 4
    settings.VOXHELM_BATCH_MAX_STAGED_UPLOAD_BYTES = 8

    response = stage_upload(
        client,
        content=b"12345",
        content_type="audio/mpeg",
    )

    payload = response.json()
    assert response.status_code == 201
    assert payload["object"] == "staged_media"
    assert payload["filename"] == "episode.mp3"
    staged = StagedMedia.objects.get(id=payload["id"])
    assert staged.size_bytes == 5


@pytest.mark.django_db
def test_batch_upload_staging_contract_rejects_oversized_audio_file(client, settings):
    settings.VOXHELM_BATCH_MAX_STAGED_UPLOAD_BYTES = 4

    response = stage_upload(
        client,
        content=b"12345",
        content_type="audio/mpeg",
    )

    assert response.status_code == 400
    assert "batch staging limit" in response.json()["error"]["message"]


@pytest.mark.django_db
def test_batch_upload_contract_rejects_uploaded_video(client):
    response = stage_upload(
        client,
        name="clip.mp4",
        content=b"video-bytes",
        content_type="video/mp4",
    )

    assert response.status_code == 400
    assert "audio only" in response.json()["error"]["message"]


@pytest.mark.django_db
def test_expired_staged_upload_is_rejected(client, settings):
    configure_task_backend(settings, "django_tasks.backends.dummy.DummyBackend")
    staged_response = stage_upload(
        client,
        name="expired.mp3",
        content=b"private-audio",
        content_type="audio/mpeg",
    )
    staged = StagedMedia.objects.get(id=staged_response.json()["id"])
    staged.expires_at = timezone.now() - timedelta(seconds=1)
    staged.save(update_fields=["expires_at"])

    response = client.post(
        "/v1/jobs",
        data=json.dumps(
            build_job_payload(
                input_data={"kind": "upload", "upload_id": str(staged.id)},
                task_ref="archive-item-expired",
            )
        ),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert response.status_code == 400
    assert "has expired" in response.json()["error"]["message"]


@pytest.mark.django_db
def test_already_claimed_staged_upload_is_rejected(client, settings):
    configure_task_backend(settings, "django_tasks.backends.dummy.DummyBackend")
    staged_response = stage_upload(
        client,
        name="claimed.mp3",
        content=b"private-audio",
        content_type="audio/mpeg",
    )
    upload_id = staged_response.json()["id"]

    first = client.post(
        "/v1/jobs",
        data=json.dumps(
            build_job_payload(
                input_data={"kind": "upload", "upload_id": upload_id},
                task_ref="archive-item-claimed-1",
            )
        ),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )
    second = client.post(
        "/v1/jobs",
        data=json.dumps(
            build_job_payload(
                input_data={"kind": "upload", "upload_id": upload_id},
                task_ref="archive-item-claimed-2",
            )
        ),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert first.status_code == 201
    assert second.status_code == 400
    assert "already been attached" in second.json()["error"]["message"]


@pytest.mark.django_db
def test_wrong_producer_cannot_submit_foreign_staged_upload(client, settings):
    settings.VOXHELM_BEARER_TOKENS = {"archive": "test-token", "other": "other-token"}
    configure_task_backend(settings, "django_tasks.backends.dummy.DummyBackend")
    staged_response = stage_upload(
        client,
        name="foreign.mp3",
        content=b"private-audio",
        content_type="audio/mpeg",
    )

    response = client.post(
        "/v1/jobs",
        data=json.dumps(
            build_job_payload(
                input_data={"kind": "upload", "upload_id": staged_response.json()["id"]},
                task_ref="other-item-123",
            )
        ),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer other-token",
    )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "Unknown input.upload_id."


@pytest.mark.django_db
def test_malformed_upload_id_is_rejected(client, settings):
    configure_task_backend(settings, "django_tasks.backends.dummy.DummyBackend")

    response = client.post(
        "/v1/jobs",
        data=json.dumps(
            build_job_payload(
                input_data={"kind": "upload", "upload_id": "not-a-uuid"},
                task_ref="archive-item-malformed",
            )
        ),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "input.upload_id must be a UUID string."


@pytest.mark.django_db
def test_expired_staged_uploads_are_cleaned_on_next_stage_request(client, settings):
    first = stage_upload(
        client,
        name="first.mp3",
        content=b"first-audio",
        content_type="audio/mpeg",
    )
    expired = StagedMedia.objects.get(id=first.json()["id"])
    artifact_path = expired.storage_key
    expired.expires_at = timezone.now() - timedelta(seconds=1)
    expired.save(update_fields=["expires_at"])

    second = stage_upload(
        client,
        name="second.mp3",
        content=b"second-audio",
        content_type="audio/mpeg",
    )

    assert second.status_code == 201
    assert not StagedMedia.objects.filter(id=expired.id).exists()
    assert not (settings.VOXHELM_ARTIFACT_ROOT / artifact_path).exists()


@pytest.mark.django_db
def test_staged_audio_job_executes_and_serves_artifacts(client, settings, monkeypatch):
    configure_task_backend(settings, "django_tasks.backends.immediate.ImmediateBackend")
    settings.VOXHELM_BATCH_MAX_STAGED_UPLOAD_BYTES = 1024
    monkeypatch.setattr("transcriptions.service.get_backend_service", lambda: DummyBackend())
    staged_response = stage_upload(
        client,
        name="private-episode.mp3",
        content=b"private-audio",
        content_type="audio/mpeg",
    )
    upload_id = staged_response.json()["id"]

    response = client.post(
        "/v1/jobs",
        data=json.dumps(build_job_payload(input_data={"kind": "upload", "upload_id": upload_id})),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    payload = response.json()
    assert response.status_code == 201
    assert payload["state"] == "succeeded"
    assert payload["result"]["metadata"]["source_kind"] == "upload"
    assert payload["result"]["metadata"]["source_name"] == "private-episode.mp3"
    assert not StagedMedia.objects.filter(id=upload_id).exists()

    artifact_response = client.get(
        payload["result"]["artifacts"]["text"],
        HTTP_AUTHORIZATION="Bearer test-token",
    )
    assert artifact_response.status_code == 200
    assert artifact_response.content.decode() == "Batch hello world"


@pytest.mark.django_db
def test_staged_audio_submission_rejects_upload_from_previous_artifact_backend(client, settings):
    configure_task_backend(settings, "django_tasks.backends.dummy.DummyBackend")
    settings.VOXHELM_BATCH_MAX_STAGED_UPLOAD_BYTES = 1024
    staged_response = stage_upload(
        client,
        name="private-episode.mp3",
        content=b"private-audio",
        content_type="audio/mpeg",
    )
    upload_id = staged_response.json()["id"]
    settings.VOXHELM_ARTIFACT_BACKEND = "s3"

    response = client.post(
        "/v1/jobs",
        data=json.dumps(build_job_payload(input_data={"kind": "upload", "upload_id": upload_id})),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert response.status_code == 400
    assert "different artifact store" in response.json()["error"]["message"]
    assert not Job.objects.exists()


@pytest.mark.django_db
def test_staged_audio_submission_rejects_upload_from_previous_artifact_root(
    client,
    settings,
    tmp_path,
):
    configure_task_backend(settings, "django_tasks.backends.dummy.DummyBackend")
    settings.VOXHELM_BATCH_MAX_STAGED_UPLOAD_BYTES = 1024
    staged_response = stage_upload(
        client,
        name="private-episode.mp3",
        content=b"private-audio",
        content_type="audio/mpeg",
    )
    upload_id = staged_response.json()["id"]
    settings.VOXHELM_ARTIFACT_ROOT = tmp_path / "new-artifact-root"
    get_artifact_store.cache_clear()

    response = client.post(
        "/v1/jobs",
        data=json.dumps(build_job_payload(input_data={"kind": "upload", "upload_id": upload_id})),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert response.status_code == 400
    assert "different artifact store" in response.json()["error"]["message"]
    assert not Job.objects.exists()


@pytest.mark.django_db
def test_staged_audio_submission_accepts_pre_migration_upload_without_identity(client, settings):
    configure_task_backend(settings, "django_tasks.backends.dummy.DummyBackend")
    settings.VOXHELM_BATCH_MAX_STAGED_UPLOAD_BYTES = 1024
    staged_response = stage_upload(
        client,
        name="private-episode.mp3",
        content=b"private-audio",
        content_type="audio/mpeg",
    )
    upload_id = staged_response.json()["id"]
    StagedMedia.objects.filter(id=upload_id).update(storage_identity={})

    response = client.post(
        "/v1/jobs",
        data=json.dumps(build_job_payload(input_data={"kind": "upload", "upload_id": upload_id})),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert response.status_code == 201
    staged = StagedMedia.objects.get(id=upload_id)
    assert staged.storage_identity == {}
    assert staged.claimed_by_job_id == Job.objects.get(id=response.json()["id"]).id


@pytest.mark.django_db
def test_staged_audio_submission_is_idempotent_with_task_ref(client, settings, monkeypatch):
    configure_task_backend(settings, "django_tasks.backends.immediate.ImmediateBackend")
    settings.VOXHELM_BATCH_MAX_STAGED_UPLOAD_BYTES = 1024
    monkeypatch.setattr("transcriptions.service.get_backend_service", lambda: DummyBackend())
    staged_response = stage_upload(
        client,
        name="repeatable.mp3",
        content=b"private-audio",
        content_type="audio/mpeg",
    )
    upload_id = staged_response.json()["id"]
    body = json.dumps(build_job_payload(input_data={"kind": "upload", "upload_id": upload_id}))

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
def test_staged_audio_job_records_materialization_failure(client, settings, monkeypatch):
    configure_task_backend(settings, "django_tasks.backends.immediate.ImmediateBackend")
    settings.VOXHELM_BATCH_MAX_STAGED_UPLOAD_BYTES = 1024
    monkeypatch.setattr(
        "jobs.services.materialize_staged_media",
        lambda *, staged: (_ for _ in ()).throw(RuntimeError("staging exploded")),
    )
    staged_response = stage_upload(
        client,
        name="broken.mp3",
        content=b"private-audio",
        content_type="audio/mpeg",
    )

    response = client.post(
        "/v1/jobs",
        data=json.dumps(
            build_job_payload(
                input_data={"kind": "upload", "upload_id": staged_response.json()["id"]}
            )
        ),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert response.status_code == 201
    assert response.json()["state"] == "failed"
    assert response.json()["error"]["message"] == "staging exploded"


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
    json_payload = json_response.json()
    assert json_payload["segments"][0]["text"] == "Batch hello"
    assert "speaker" not in json_payload["segments"][0]


class StubEmbeddingBackend:
    embedding_version = "stub-v1"

    def embed(self, samples, sample_rate):
        del sample_rate
        import numpy

        tag = int(round(float(numpy.mean(samples)) * 10))
        vector = [0.0] * 6
        vector[max(0, min(tag, 5))] = 1.0
        return vector


@pytest.mark.django_db
def test_known_speaker_job_emits_speaker_suggestions(client, settings, monkeypatch, tmp_path):
    import numpy

    configure_task_backend(settings, "django_tasks.backends.immediate.ImmediateBackend")
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com", "cdn.example.com"}
    media_path = tmp_path / "episode.mp3"
    media_path.write_bytes(b"mp3-bytes")
    ref_path = tmp_path / "ref.mp3"
    ref_path.write_bytes(b"mp3-bytes")
    monkeypatch.setattr(
        "jobs.services.download_allowed_media",
        lambda *, source_url: DownloadedMedia(
            path=media_path if "episode" in source_url else ref_path,
            content_type="audio/mpeg",
            source_url=source_url,
        ),
    )
    monkeypatch.setattr("transcriptions.service.get_backend_service", lambda: DummyBackend())
    # Both job audio and reference decode to constant 0.1 -> stub tag 1 (Johannes).
    monkeypatch.setattr(
        "jobs.services.decode_mono_16k",
        lambda path: numpy.full(2 * 16000, 0.1, dtype="float32"),
    )
    monkeypatch.setattr(
        "jobs.services.get_known_speaker_backend", lambda model: StubEmbeddingBackend()
    )
    monkeypatch.setattr(
        "jobs.services.diarize_audio",
        lambda audio_path, params: [SpeakerTurn(start=0.0, end=2.0, speaker="SPEAKER_00")],
    )

    payload = build_job_payload(task_ref="archive-known-speaker")
    payload["output"] = {"formats": ["podlove"]}
    payload["diarization"] = {
        "enabled": True,
        "strategy": "pyannote_known_speaker",
        "known_speakers": [
            {
                "id": "12",
                "name": "Johannes",
                "references": [
                    {
                        "kind": "source_range",
                        "audio": {"kind": "url", "url": "https://cdn.example.com/pp_60.m4a"},
                        "start": 0.0,
                        "end": 2.0,
                    }
                ],
            }
        ],
        "known_speaker": {
            "min_segment_duration": 0.5,
            "auto_accept_margin": 0.1,
            "min_top_similarity": 0.5,
        },
    }

    response = client.post(
        "/v1/jobs",
        data=json.dumps(payload),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )
    result = response.json()["result"]
    assert response.status_code == 201
    assert "speakers" in result["artifacts"]
    # Known-speaker summary is surfaced in result metadata without leaking refs.
    summary = result["metadata"]["diarization"]["known_speaker_summary"]
    assert summary["strategy"] == "pyannote_known_speaker"
    assert summary["confident_segment_count"] == 2
    assert result["metadata"]["diarization"]["known_speakers"] == [
        {"id": "12", "name": "Johannes", "reference_count": 1}
    ]
    assert "url" not in json.dumps(result["metadata"]["diarization"])

    speakers_response = client.get(
        result["artifacts"]["speakers"], HTTP_AUTHORIZATION="Bearer test-token"
    )
    assert speakers_response.status_code == 200
    speakers_payload = speakers_response.json()
    assert speakers_payload["segments"][0]["speaker"] == "Johannes"
    assert speakers_payload["segments"][0]["raw_diarization_speaker"] == "Speaker 1"
    assert speakers_payload["segments"][0]["speaker_uncertain"] is False

    # Known-speaker mode leaves public Podlove unlabeled until review/approval;
    # the suggestion lives only in the speakers sidecar.
    podlove_response = client.get(
        result["artifacts"]["podlove"], HTTP_AUTHORIZATION="Bearer test-token"
    )
    assert podlove_response.json()["transcripts"][0]["speaker"] == ""


@pytest.mark.django_db
def test_known_speaker_job_differs_from_anonymous_for_dedup(client, settings, monkeypatch):
    configure_task_backend(settings, "django_tasks.backends.dummy.DummyBackend")
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com", "cdn.example.com"}

    anonymous = build_job_payload(task_ref="dedup-strategy")
    anonymous["diarization"] = {"enabled": True, "num_speakers": 2}
    first = client.post(
        "/v1/jobs",
        data=json.dumps(anonymous),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )
    known = build_job_payload(task_ref="dedup-strategy")
    known["diarization"] = {
        "enabled": True,
        "num_speakers": 2,
        "strategy": "pyannote_known_speaker",
        "known_speakers": [
            {
                "id": "1",
                "name": "A",
                "references": [
                    {
                        "kind": "clip_artifact",
                        "audio": {"kind": "url", "url": "https://cdn.example.com/a.wav"},
                    }
                ],
            }
        ],
    }
    second = client.post(
        "/v1/jobs",
        data=json.dumps(known),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )
    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["id"] != second.json()["id"]


@pytest.mark.django_db
def test_transcription_job_accepts_dote_and_podlove_outputs(
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
    payload = build_job_payload()
    payload["output"] = {"formats": ["text", "json", "vtt", "dote", "podlove"]}

    response = client.post(
        "/v1/jobs",
        data=json.dumps(payload),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    result = response.json()["result"]
    assert response.status_code == 201
    assert set(result["artifacts"]) == {"text", "json", "vtt", "dote", "podlove"}

    dote_response = client.get(result["artifacts"]["dote"], HTTP_AUTHORIZATION="Bearer test-token")
    assert dote_response.status_code == 200
    dote_payload = dote_response.json()
    assert dote_payload["lines"][0]["startTime"] == "00:00:00,000"
    assert dote_payload["lines"][0]["speakerDesignation"] == ""

    podlove_response = client.get(
        result["artifacts"]["podlove"],
        HTTP_AUTHORIZATION="Bearer test-token",
    )
    assert podlove_response.status_code == 200
    podlove_payload = podlove_response.json()
    assert podlove_payload["transcripts"][1]["text"] == "world"
    assert podlove_payload["transcripts"][1]["speaker"] == ""
    assert podlove_payload["transcripts"][1]["voice"] == ""


@pytest.mark.django_db
def test_diarized_transcription_job_labels_server_owned_artifacts(
    client,
    settings,
    monkeypatch,
    tmp_path,
):
    configure_task_backend(settings, "django_tasks.backends.immediate.ImmediateBackend")
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}
    media_path = tmp_path / "episode.mp3"
    media_path.write_bytes(b"mp3-bytes")
    diarization_calls: list[Path] = []
    monkeypatch.setattr(
        "jobs.services.download_allowed_media",
        lambda *, source_url: DownloadedMedia(
            path=media_path,
            content_type="audio/mpeg",
            source_url=source_url,
        ),
    )
    monkeypatch.setattr("transcriptions.service.get_backend_service", lambda: DummyBackend())

    def fake_diarize(
        audio_path: Path,
        params: DiarizationParams | None = None,
    ) -> list[SpeakerTurn]:
        del params
        diarization_calls.append(audio_path)
        return [
            SpeakerTurn(start=0.0, end=1.0, speaker="SPEAKER_00"),
            SpeakerTurn(start=1.0, end=2.0, speaker="SPEAKER_01"),
        ]

    monkeypatch.setattr("jobs.services.diarize_audio", fake_diarize)
    payload = build_job_payload(task_ref="archive-item-diarized")
    payload["diarization"] = {"enabled": True}
    payload["output"] = {"formats": ["text", "json", "vtt", "dote", "podlove"]}

    response = client.post(
        "/v1/jobs",
        data=json.dumps(payload),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    result = response.json()["result"]
    assert response.status_code == 201
    assert response.json()["state"] == "succeeded"
    assert diarization_calls == [media_path]
    job = Job.objects.get(id=response.json()["id"])
    assert job.output_data["diarization"] == {"enabled": True}
    assert result["metadata"]["diarization"] == {"enabled": True}

    json_response = client.get(
        result["artifacts"]["json"],
        HTTP_AUTHORIZATION="Bearer test-token",
    )
    assert json_response.status_code == 200
    assert json_response.json()["segments"] == [
        {
            "id": 0,
            "seek": 0,
            "start": 0.0,
            "end": 1.0,
            "text": "Batch hello",
            "speaker": "Speaker 1",
        },
        {
            "id": 1,
            "seek": 100,
            "start": 1.0,
            "end": 2.0,
            "text": "world",
            "speaker": "Speaker 2",
        },
    ]

    dote_response = client.get(result["artifacts"]["dote"], HTTP_AUTHORIZATION="Bearer test-token")
    assert dote_response.status_code == 200
    assert [line["speakerDesignation"] for line in dote_response.json()["lines"]] == [
        "Speaker 1",
        "Speaker 2",
    ]

    podlove_response = client.get(
        result["artifacts"]["podlove"],
        HTTP_AUTHORIZATION="Bearer test-token",
    )
    assert podlove_response.status_code == 200
    podlove_payload = podlove_response.json()
    assert [line["speaker"] for line in podlove_payload["transcripts"]] == [
        "Speaker 1",
        "Speaker 2",
    ]
    assert [line["voice"] for line in podlove_payload["transcripts"]] == [
        "Speaker 1",
        "Speaker 2",
    ]

    vtt_response = client.get(result["artifacts"]["vtt"], HTTP_AUTHORIZATION="Bearer test-token")
    assert vtt_response.status_code == 200
    assert "Speaker" not in vtt_response.content.decode()


@pytest.mark.django_db
def test_diarized_transcription_job_passes_speaker_count_hint_to_backend(
    client,
    settings,
    monkeypatch,
    tmp_path,
):
    configure_task_backend(settings, "django_tasks.backends.immediate.ImmediateBackend")
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}
    media_path = tmp_path / "episode.mp3"
    media_path.write_bytes(b"mp3-bytes")
    diarization_params: list[DiarizationParams | None] = []
    monkeypatch.setattr(
        "jobs.services.download_allowed_media",
        lambda *, source_url: DownloadedMedia(
            path=media_path,
            content_type="audio/mpeg",
            source_url=source_url,
        ),
    )
    monkeypatch.setattr("transcriptions.service.get_backend_service", lambda: DummyBackend())

    def fake_diarize(
        audio_path: Path,
        params: DiarizationParams | None = None,
    ) -> list[SpeakerTurn]:
        del audio_path
        diarization_params.append(params)
        return [SpeakerTurn(start=0.0, end=2.0, speaker="SPEAKER_00")]

    monkeypatch.setattr("jobs.services.diarize_audio", fake_diarize)
    payload = build_job_payload(task_ref="archive-item-diarized-speaker-hint")
    payload["diarization"] = {"enabled": True, "num_speakers": 4}

    response = client.post(
        "/v1/jobs",
        data=json.dumps(payload),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    result = response.json()["result"]
    assert response.status_code == 201
    assert response.json()["state"] == "succeeded"
    assert diarization_params == [DiarizationParams(num_speakers=4)]
    job = Job.objects.get(id=response.json()["id"])
    assert job.output_data["diarization"] == {"enabled": True, "num_speakers": 4}
    assert result["metadata"]["diarization"] == {"enabled": True, "num_speakers": 4}


@pytest.mark.django_db
def test_requested_diarization_fails_when_backend_is_unavailable(
    client,
    settings,
    monkeypatch,
    tmp_path,
):
    configure_task_backend(settings, "django_tasks.backends.immediate.ImmediateBackend")
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}
    settings.VOXHELM_DIARIZATION_BACKEND = "none"
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
    payload = build_job_payload(task_ref="archive-item-diarization-unavailable")
    payload["diarization"] = {"enabled": True}

    response = client.post(
        "/v1/jobs",
        data=json.dumps(payload),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert response.status_code == 201
    assert response.json()["state"] == "failed"
    assert "Diarization was requested" in response.json()["error"]["message"]


@pytest.mark.django_db
def test_requested_diarization_fails_when_backend_returns_no_usable_turns(
    client,
    settings,
    monkeypatch,
    tmp_path,
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
    monkeypatch.setattr("jobs.services.diarize_audio", lambda audio_path, params=None: [])
    payload = build_job_payload(task_ref="archive-item-diarization-empty")
    payload["diarization"] = {"enabled": True}

    response = client.post(
        "/v1/jobs",
        data=json.dumps(payload),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert response.status_code == 201
    assert response.json()["state"] == "failed"
    assert "no usable speaker turns" in response.json()["error"]["message"]
    assert not JobArtifact.objects.filter(kind=JobArtifact.Kind.TRANSCRIPT_JSON).exists()


@pytest.mark.django_db
def test_transcription_job_uses_non_interactive_scheduler_lane(
    client, settings, monkeypatch, tmp_path
):
    configure_task_backend(settings, "django_tasks.backends.immediate.ImmediateBackend")
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}
    lanes: list[str] = []
    media_path = tmp_path / "episode.mp3"
    media_path.write_bytes(b"mp3-bytes")

    @contextmanager
    def fake_admit(lane: str, *, cancel_event=None):
        del cancel_event
        lanes.append(lane)
        yield object()

    monkeypatch.setattr("transcriptions.service.admit_local_inference", fake_admit)
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

    assert response.status_code == 201
    assert lanes == ["non-interactive"]


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
def test_video_extraction_failure_marks_job_failed(client, settings, monkeypatch, tmp_path):
    configure_task_backend(settings, "django_tasks.backends.immediate.ImmediateBackend")
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video-bytes")

    monkeypatch.setattr(
        "jobs.services.download_allowed_media",
        lambda *, source_url: DownloadedMedia(
            path=video_path,
            content_type="video/mp4",
            source_url=source_url,
            source_name="clip.mp4",
            source_kind="url",
        ),
    )
    monkeypatch.setattr(
        "jobs.services.extract_audio_from_video",
        lambda *, source_path: (_ for _ in ()).throw(RuntimeError("ffmpeg exploded")),
    )

    response = client.post(
        "/v1/jobs",
        data=json.dumps(build_job_payload(url="https://media.example.com/clip.mp4")),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert response.status_code == 201
    assert response.json()["state"] == "failed"
    assert response.json()["error"]["message"] == "ffmpeg exploded"


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


@pytest.mark.django_db
def test_synthesize_job_serves_audio_artifact(client, settings, monkeypatch, tmp_path):
    configure_task_backend(settings, "django_tasks.backends.immediate.ImmediateBackend")
    speech_path = tmp_path / "speech.wav"
    speech_path.write_bytes(b"RIFFspeech")

    monkeypatch.setattr(
        "jobs.services.synthesize_text",
        lambda text, params: type(
            "SpeechResult",
            (),
            {
                "audio_path": speech_path,
                "backend_name": "piper",
                "model_name": "piper",
                "voice_name": "en_US-lessac-medium",
                "language": "en",
                "duration_seconds": 1.25,
            },
        )(),
    )
    monkeypatch.setattr(
        "jobs.services.export_audio",
        lambda result, output_format: type(
            "ExportedAudio",
            (),
            {"path": result.audio_path, "format_name": output_format, "content_type": "audio/wav"},
        )(),
    )

    response = client.post(
        "/v1/jobs",
        data=json.dumps(build_synthesis_payload()),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    payload = response.json()
    assert response.status_code == 201
    assert payload["state"] == "succeeded"
    assert payload["result"]["artifacts"]["wav"].endswith("/speech.wav")
    assert payload["result"]["metadata"]["voice"] == "en_US-lessac-medium"

    artifact_response = client.get(
        payload["result"]["artifacts"]["wav"],
        HTTP_AUTHORIZATION="Bearer test-token",
    )
    assert artifact_response.status_code == 200
    assert artifact_response.content == b"RIFFspeech"


@pytest.mark.django_db
def test_synthesize_job_uses_non_interactive_scheduler_lane(
    client, settings, monkeypatch, tmp_path
):
    configure_task_backend(settings, "django_tasks.backends.immediate.ImmediateBackend")
    lanes: list[str] = []
    speech_path = tmp_path / "speech.wav"
    speech_path.write_bytes(b"RIFFspeech")

    @contextmanager
    def fake_admit(lane: str):
        lanes.append(lane)
        yield object()

    monkeypatch.setattr("synthesis.service.admit_local_inference", fake_admit)
    monkeypatch.setattr(
        "synthesis.service.get_backend_service",
        lambda *args, **kwargs: type(
            "Backend",
            (),
            {
                "synthesize": lambda self, text, params: type(
                    "SpeechResult",
                    (),
                    {
                        "audio_path": speech_path,
                        "backend_name": "piper",
                        "model_name": "piper",
                        "voice_name": "en_US-lessac-medium",
                        "language": "en",
                        "duration_seconds": 1.25,
                    },
                )(),
            },
        )(),
    )
    monkeypatch.setattr(
        "jobs.services.export_audio",
        lambda result, output_format: type(
            "ExportedAudio",
            (),
            {"path": result.audio_path, "format_name": output_format, "content_type": "audio/wav"},
        )(),
    )

    response = client.post(
        "/v1/jobs",
        data=json.dumps(build_synthesis_payload()),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert response.status_code == 201
    assert lanes == ["non-interactive"]


@pytest.mark.django_db
def test_synthesize_job_failure_records_error(client, settings, monkeypatch):
    configure_task_backend(settings, "django_tasks.backends.immediate.ImmediateBackend")

    def fail_synthesis(text: str, params: object):
        del text, params
        raise RuntimeError("synthesis exploded")

    monkeypatch.setattr("jobs.services.synthesize_text", fail_synthesis)

    response = client.post(
        "/v1/jobs",
        data=json.dumps(build_synthesis_payload()),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    payload = response.json()
    assert response.status_code == 201
    assert payload["state"] == "failed"
    assert payload["error"]["message"] == "synthesis exploded"


@pytest.mark.django_db
def test_synthesize_job_rejects_out_of_range_speed(client):
    payload = build_synthesis_payload()
    payload["speed"] = 100

    response = client.post(
        "/v1/jobs",
        data=json.dumps(payload),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert response.status_code == 400
    assert "between 0.25 and 4.0" in response.json()["error"]["message"]


# --- task_ref idempotency race (request fingerprint + partial unique constraint) ---


def submit_job(client, payload: dict[str, object]):
    return client.post(
        "/v1/jobs",
        data=json.dumps(payload),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )


def stored_fingerprint(job: Job) -> str:
    return stored_job_request_fingerprint(
        job_type=job.job_type,
        backend=job.backend,
        model=job.model,
        language=job.language,
        input_data=job.input_data,
        output_data=job.output_data,
    )


def miss_first_lookups(monkeypatch, count: int) -> None:
    """Make the optimistic pre-insert dedup lookup miss, as it does when two submissions race."""
    real_lookup = services.find_reusable_job
    remaining = {"misses": count}

    def lookup(*, producer: str, request: services.JobRequest) -> Job | None:
        if remaining["misses"] > 0:
            remaining["misses"] -= 1
            return None
        return real_lookup(producer=producer, request=request)

    monkeypatch.setattr(services, "find_reusable_job", lookup)


@pytest.mark.django_db
def test_task_ref_job_stores_fingerprint_matching_persisted_request(client, settings):
    configure_task_backend(settings, "django_tasks.backends.dummy.DummyBackend")
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}

    response = submit_job(client, build_job_payload(task_ref="fingerprint-parity"))

    assert response.status_code == 201
    job = Job.objects.get(id=response.json()["id"])
    assert len(job.request_fingerprint) == 64
    assert job.request_fingerprint == stored_fingerprint(job)


@pytest.mark.django_db
def test_job_without_task_ref_has_no_fingerprint(client, settings):
    configure_task_backend(settings, "django_tasks.backends.dummy.DummyBackend")
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}
    payload = build_job_payload()
    payload.pop("task_ref")

    first = submit_job(client, payload)
    second = submit_job(client, payload)

    assert first.status_code == 201
    assert second.status_code == 201
    assert set(Job.objects.values_list("request_fingerprint", flat=True)) == {""}
    assert Job.objects.count() == 2


@pytest.mark.django_db
def test_racing_submission_that_missed_dedup_returns_existing_job(client, settings, monkeypatch):
    configure_task_backend(settings, "django_tasks.backends.dummy.DummyBackend")
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}
    payload = build_job_payload(task_ref="race-missed-lookup")

    first = submit_job(client, payload)
    miss_first_lookups(monkeypatch, 1)
    second = submit_job(client, payload)

    assert first.status_code == 201
    assert second.status_code == 200
    assert second.json()["id"] == first.json()["id"]
    assert Job.objects.count() == 1


@pytest.mark.django_db
def test_racing_synthesis_submission_that_missed_dedup_returns_existing_job(
    client, settings, monkeypatch
):
    configure_task_backend(settings, "django_tasks.backends.dummy.DummyBackend")
    first_payload = build_synthesis_payload("first text")
    second_payload = build_synthesis_payload("different text keeps loose synthesis reuse")

    first = submit_job(client, first_payload)
    miss_first_lookups(monkeypatch, 1)
    second = submit_job(client, second_payload)

    assert first.status_code == 201
    assert second.status_code == 200
    assert second.json()["id"] == first.json()["id"]
    assert Job.objects.count() == 1


@pytest.mark.django_db
def test_racing_staged_upload_submission_returns_existing_job(client, settings, monkeypatch):
    configure_task_backend(settings, "django_tasks.backends.dummy.DummyBackend")
    staged = stage_upload(client, name="race.mp3", content=b"audio", content_type="audio/mpeg")
    payload = build_job_payload(
        input_data={"kind": "upload", "upload_id": staged.json()["id"]},
        task_ref="race-staged-upload",
    )

    first = submit_job(client, payload)
    miss_first_lookups(monkeypatch, 1)
    second = submit_job(client, payload)

    assert first.status_code == 201
    assert second.status_code == 200
    assert second.json()["id"] == first.json()["id"]
    assert Job.objects.count() == 1


@pytest.mark.django_db
def test_failed_task_ref_job_allows_resubmission(client, settings):
    configure_task_backend(settings, "django_tasks.backends.dummy.DummyBackend")
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}
    payload = build_job_payload(task_ref="failed-then-retry")

    first = submit_job(client, payload)
    Job.objects.filter(id=first.json()["id"]).update(state=Job.State.FAILED)
    second = submit_job(client, payload)

    assert first.status_code == 201
    assert second.status_code == 201
    assert second.json()["id"] != first.json()["id"]
    jobs = list(Job.objects.order_by("created_at"))
    assert len(jobs) == 2
    assert jobs[0].request_fingerprint == jobs[1].request_fingerprint


@pytest.mark.django_db
def test_different_payload_with_same_task_ref_gets_distinct_fingerprint(client, settings):
    configure_task_backend(settings, "django_tasks.backends.dummy.DummyBackend")
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}
    first_payload = build_job_payload(task_ref="same-ref-different-payload")
    second_payload = build_job_payload(task_ref="same-ref-different-payload")
    second_payload["output"] = {"formats": ["vtt"]}

    first = submit_job(client, first_payload)
    second = submit_job(client, second_payload)

    assert first.status_code == 201
    assert second.status_code == 201
    fingerprints = set(Job.objects.values_list("request_fingerprint", flat=True))
    assert len(fingerprints) == 2


@pytest.mark.django_db
def test_constraint_rejects_second_active_job_with_same_fingerprint():
    common: dict[str, Any] = {
        "producer": "archive",
        "task_ref": "constraint-ref",
        "request_fingerprint": "a" * 64,
        "job_type": Job.JobType.TRANSCRIBE,
        "input_data": {"kind": "url", "url": "https://media.example.com/a.mp3"},
    }
    Job.objects.create(**common, state=Job.State.FAILED)
    Job.objects.create(**common, state=Job.State.SUCCEEDED)

    with pytest.raises(IntegrityError), transaction.atomic():
        Job.objects.create(**common, state=Job.State.QUEUED)

    Job.objects.create(**{**common, "request_fingerprint": ""}, state=Job.State.QUEUED)
    Job.objects.create(**{**common, "producer": "other"}, state=Job.State.QUEUED)


@pytest.mark.django_db(transaction=True)
def test_concurrent_identical_submissions_create_one_job(settings, monkeypatch):
    configure_task_backend(settings, "django_tasks.backends.dummy.DummyBackend")
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}
    payload = build_job_payload(task_ref="concurrent-submit")
    submitters = 4

    # Every submitter must finish its optimistic dedup lookup (and miss) before any of them
    # inserts, reproducing the race window that used to create duplicate jobs. After the barrier
    # the rest of each submission runs one at a time only because the in-memory shared-cache
    # SQLite test database raises "table is locked" instead of queueing concurrent writers; the
    # production file database serializes writers via its write lock and busy timeout.
    real_lookup = services.find_reusable_job
    barrier = threading.Barrier(submitters, timeout=10)
    write_lock = threading.Lock()
    state = threading.local()

    def lookup(*, producer: str, request: services.JobRequest) -> Job | None:
        result = real_lookup(producer=producer, request=request)
        if not getattr(state, "holds_lock", False):
            barrier.wait()
            write_lock.acquire()
            state.holds_lock = True
        return result

    monkeypatch.setattr(services, "find_reusable_job", lookup)

    results: list[tuple[str, bool]] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            job, created = services.create_job_from_payload(producer="archive", payload=payload)
            results.append((str(job.id), created))
        except BaseException as exc:
            errors.append(exc)
        finally:
            connections.close_all()
            if getattr(state, "holds_lock", False):
                write_lock.release()

    threads = [threading.Thread(target=worker) for _ in range(submitters)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert errors == []
    assert len(results) == submitters
    assert len({job_id for job_id, _ in results}) == 1
    assert sum(1 for _, created in results if created) == 1
    assert Job.objects.count() == 1


FILE_SQLITE_RACE_DRIVER = """
import json
import sys
import threading
from pathlib import Path

import django

root = Path(sys.argv[1])
django.setup()

from django.conf import settings
from django.db import connections

# Point the default connection at a file database before anything opens a connection.
settings.DATABASES["default"]["NAME"] = root / "race.db"
connections.__dict__.pop("settings", None)
settings.TASKS = {"default": {"BACKEND": "django_tasks.backends.dummy.DummyBackend"}}
settings.VOXHELM_ARTIFACT_BACKEND = "filesystem"
settings.VOXHELM_ARTIFACT_ROOT = root / "artifacts"
settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}
settings.VOXHELM_TRANSCRIPTION_EXECUTION_MODE = "django_tasks"

from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management import call_command

from jobs import services
from jobs.models import Job
from jobs.staging import stage_uploaded_audio

assert connections["default"].settings_dict["NAME"] == root / "race.db"
call_command("migrate", verbosity=0)
submitters = 4
staged = stage_uploaded_audio(
    producer="archive",
    upload=SimpleUploadedFile("race.mp3", b"audio", content_type="audio/mpeg"),
)
payloads = {
    "url": {"kind": "url", "url": "https://media.example.com/race.mp3"},
    "upload": {"kind": "upload", "upload_id": str(staged.id)},
}
report = {}
real_lookup = services.find_reusable_job
for kind, input_data in payloads.items():
    payload = {
        "job_type": "transcribe",
        "model": "auto",
        "language": "en",
        "input": input_data,
        "output": {"formats": ["text", "json"]},
        "task_ref": "file-race-" + kind,
    }
    barrier = threading.Barrier(submitters, timeout=20)
    local = threading.local()

    def lookup(*, producer, request):
        result = real_lookup(producer=producer, request=request)
        if not getattr(local, "waited", False):
            local.waited = True
            barrier.wait()
        return result

    services.find_reusable_job = lookup
    results, errors = [], []

    def worker():
        try:
            job, created = services.create_job_from_payload(producer="archive", payload=payload)
            results.append([str(job.id), created])
        except BaseException as exc:
            errors.append(repr(exc))
        finally:
            connections.close_all()

    threads = [threading.Thread(target=worker) for _ in range(submitters)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    services.find_reusable_job = real_lookup
    report[kind] = {
        "results": results,
        "errors": errors,
        "jobs": Job.objects.filter(task_ref="file-race-" + kind).count(),
    }
json.dump(report, sys.stdout)
"""


def test_concurrent_identical_submissions_on_file_sqlite_create_one_job(tmp_path):
    """Unserialized race on a real file-backed SQLite database, like production uses.

    The pytest database is in-memory shared-cache SQLite, which fails concurrent writers with
    "table is locked" instead of queueing them, so this runs in a subprocess against a file.
    """
    repo_root = Path(__file__).resolve().parent.parent
    driver = tmp_path / "race_driver.py"
    driver.write_text(FILE_SQLITE_RACE_DRIVER)
    env = {
        **os.environ,
        "DJANGO_SETTINGS_MODULE": "config.settings",
        "PYTHONPATH": str(repo_root),
    }
    completed = subprocess.run(
        [sys.executable, str(driver), str(tmp_path)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)

    for kind in ("url", "upload"):
        outcome = report[kind]
        assert outcome["errors"] == [], (kind, outcome["errors"])
        assert len(outcome["results"]) == 4, kind
        assert len({job_id for job_id, _ in outcome["results"]}) == 1, kind
        assert sum(1 for _, created in outcome["results"] if created) == 1, kind
        assert outcome["jobs"] == 1, kind


@pytest.mark.django_db(transaction=True)
def test_fingerprint_migration_backfills_and_tolerates_legacy_duplicates():
    executor = MigrationExecutor(connection)
    executor.migrate([("jobs", "0007_remote_workers")])
    old_apps = executor.loader.project_state([("jobs", "0007_remote_workers")]).apps
    OldJob = old_apps.get_model("jobs", "Job")
    common: dict[str, Any] = {
        "producer": "archive",
        "task_ref": "legacy-ref",
        "job_type": "transcribe",
        "backend": "auto",
        "model": "auto",
        "language": "en",
        "input_data": {"kind": "url", "url": "https://media.example.com/legacy.mp3"},
        "output_data": {"formats": ["text", "json"], "diarization": {"enabled": False}},
    }
    older_duplicate = OldJob.objects.create(**common, state="queued")
    newer_duplicate = OldJob.objects.create(**common, state="succeeded")
    failed = OldJob.objects.create(**common, state="failed")
    no_ref = OldJob.objects.create(**{**common, "task_ref": ""}, state="queued")

    executor = MigrationExecutor(connection)
    executor.loader.build_graph()
    executor.migrate([("jobs", "0008_job_request_fingerprint")])

    try:
        jobs = {job.id: job for job in Job.objects.all()}
        expected = stored_fingerprint(jobs[newer_duplicate.id])
        assert jobs[newer_duplicate.id].request_fingerprint == expected
        assert jobs[older_duplicate.id].request_fingerprint == ""
        assert jobs[failed.id].request_fingerprint == expected
        assert jobs[no_ref.id].request_fingerprint == ""
    finally:
        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate(executor.loader.graph.leaf_nodes())
