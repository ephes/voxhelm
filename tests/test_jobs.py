from __future__ import annotations

import json
from contextlib import contextmanager
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
    assert dote_response.json()["lines"][0]["startTime"] == "00:00:00,000"

    podlove_response = client.get(
        result["artifacts"]["podlove"],
        HTTP_AUTHORIZATION="Bearer test-token",
    )
    assert podlove_response.status_code == 200
    assert podlove_response.json()["transcripts"][1]["text"] == "world"


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
    def fake_admit(lane: str):
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
        lambda: type(
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
