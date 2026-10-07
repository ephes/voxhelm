from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.utils import timezone

from jobs.artifacts import StoredArtifact, current_artifact_store_identity, get_artifact_store
from jobs.models import Job, JobArtifact, PendingArtifactDeletion, StagedMedia, Worker
from jobs.retention import prune_job_artifacts
from jobs.services import reconcile_remote_job_state


def build_job_payload(
    *,
    url: str = "https://media.example.com/episode.mp3",
    input_data: dict[str, object] | None = None,
    model: str = "auto",
    task_ref: str = "remote-item-123",
    diarization: dict[str, Any] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "job_type": "transcribe",
        "priority": "normal",
        "lane": "batch",
        "backend": "auto",
        "model": model,
        "language": "en",
        "input": input_data or {"kind": "url", "url": url},
        "output": {"formats": ["text", "json"]},
        "context": {"producer": "archive", "item_id": 123},
        "task_ref": task_ref,
    }
    if diarization is not None:
        payload["diarization"] = diarization
    return payload


def plain_capabilities() -> dict[str, object]:
    return {
        "job_types": ["transcribe"],
        "backends": ["whispercpp"],
        "models": ["ggml-large-v3.bin"],
        "output_formats": ["text", "json", "vtt", "dote", "podlove"],
        "diarization": {"anonymous": False, "known_speaker": False, "embedding_models": []},
    }


def limited_text_capabilities() -> dict[str, object]:
    capabilities = plain_capabilities()
    capabilities["output_formats"] = ["text"]
    return capabilities


def no_stt_capabilities() -> dict[str, object]:
    return {
        "job_types": ["transcribe"],
        "output_formats": ["text", "json", "vtt", "dote", "podlove"],
        "diarization": {"anonymous": False, "known_speaker": False, "embedding_models": []},
    }


def incompatible_stt_capabilities() -> dict[str, object]:
    capabilities = plain_capabilities()
    capabilities["backends"] = ["mlx"]
    capabilities["models"] = ["mlx-community/other-model"]
    return capabilities


def known_speaker_capabilities() -> dict[str, object]:
    capabilities = plain_capabilities()
    capabilities["output_formats"] = [
        "text",
        "json",
        "vtt",
        "dote",
        "podlove",
        "speakers",
    ]
    capabilities["diarization"] = {
        "anonymous": True,
        "known_speaker": True,
        "embedding_models": ["pyannote/wespeaker-voxceleb-resnet34-LM"],
    }
    return capabilities


def worker_headers(token: str = "atlas-token") -> dict[str, str]:
    return {"HTTP_AUTHORIZATION": f"Bearer {token}"}


def producer_headers() -> dict[str, str]:
    return {"HTTP_AUTHORIZATION": "Bearer test-token"}


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
        **producer_headers(),
    )


def enable_remote_workers(settings) -> None:
    settings.VOXHELM_TRANSCRIPTION_EXECUTION_MODE = "remote_pull"
    settings.VOXHELM_WORKER_TOKENS = {"atlas": "atlas-token", "zephyr": "zephyr-token"}
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com", "cdn.example.com"}


def submit_remote_job(client, settings, **payload_kwargs) -> Job:
    enable_remote_workers(settings)
    response = client.post(
        "/v1/jobs",
        data=json.dumps(build_job_payload(**payload_kwargs)),
        content_type="application/json",
        **producer_headers(),
    )
    assert response.status_code == 201
    return Job.objects.get(id=response.json()["id"])


def submit_remote_upload_job(client, settings, *, task_ref: str) -> tuple[Job, str, str]:
    enable_remote_workers(settings)
    settings.VOXHELM_BATCH_MAX_STAGED_UPLOAD_BYTES = 1024
    staged_response = stage_upload(
        client,
        name="private-episode.mp3",
        content=b"private-audio",
        content_type="audio/mpeg",
    )
    assert staged_response.status_code == 201
    upload_id = staged_response.json()["id"]
    staged_key = StagedMedia.objects.get(id=upload_id).storage_key
    job_response = client.post(
        "/v1/jobs",
        data=json.dumps(
            build_job_payload(
                input_data={"kind": "upload", "upload_id": upload_id},
                task_ref=task_ref,
            )
        ),
        content_type="application/json",
        **producer_headers(),
    )
    assert job_response.status_code == 201
    return Job.objects.get(id=job_response.json()["id"]), upload_id, staged_key


def claim_one(client, *, worker_id: str = "atlas", capabilities: dict[str, object] | None = None):
    return client.post(
        "/v1/internal/work/claim",
        data=json.dumps(
            {
                "worker_id": worker_id,
                "max_jobs": 1,
                "capabilities": capabilities or plain_capabilities(),
            }
        ),
        content_type="application/json",
        **worker_headers(f"{worker_id}-token"),
    )


def store_remote_artifact(settings, *, key: str, content: bytes) -> None:
    path = settings.VOXHELM_ARTIFACT_ROOT / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def store_manifest_artifacts(
    settings,
    manifest: list[dict[str, object]],
    *,
    skip_names: set[str] | None = None,
) -> None:
    skip_names = skip_names or set()
    for artifact in manifest:
        if artifact["name"] in skip_names:
            continue
        content = b"remote transcript"
        if artifact["name"] != "transcript.txt":
            size_bytes = artifact["size_bytes"]
            assert isinstance(size_bytes, int)
            content = b"x" * size_bytes
        store_remote_artifact(settings, key=str(artifact["storage_key"]), content=content)


def transcript_manifest(job_id: str, attempt: int) -> list[dict[str, object]]:
    prefix = f"voxhelm/jobs/{job_id}/attempt-{attempt}/"
    return [
        {
            "name": "source.mp3",
            "kind": "source",
            "format": "source",
            "storage_backend": "filesystem",
            "storage_key": f"{prefix}source.mp3",
            "content_type": "audio/mpeg",
            "size_bytes": 9,
            "exposed": False,
        },
        {
            "name": "transcript.txt",
            "kind": "transcript_text",
            "format": "text",
            "storage_backend": "filesystem",
            "storage_key": f"{prefix}transcript.txt",
            "content_type": "text/plain; charset=utf-8",
            "size_bytes": 17,
            "exposed": True,
        },
        {
            "name": "transcript.json",
            "kind": "transcript_json",
            "format": "json",
            "storage_backend": "filesystem",
            "storage_key": f"{prefix}transcript.json",
            "content_type": "application/json",
            "size_bytes": 33,
            "exposed": True,
        },
    ]


def speakers_artifact(
    job_id: str,
    attempt: int,
    *,
    format_name: str = "speakers",
) -> dict[str, object]:
    prefix = f"voxhelm/jobs/{job_id}/attempt-{attempt}/"
    return {
        "name": "transcript.speakers.json",
        "kind": "transcript_speakers",
        "format": format_name,
        "storage_backend": "filesystem",
        "storage_key": f"{prefix}transcript.speakers.json",
        "content_type": "application/json",
        "size_bytes": 99,
        "exposed": True,
    }


@pytest.mark.django_db
def test_worker_heartbeat_uses_separate_worker_token_domain(client, settings):
    enable_remote_workers(settings)

    producer_response = client.post(
        "/v1/internal/workers/heartbeat",
        data=json.dumps({"worker_id": "atlas"}),
        content_type="application/json",
        **producer_headers(),
    )
    assert producer_response.status_code == 401

    mismatch_response = client.post(
        "/v1/internal/workers/heartbeat",
        data=json.dumps({"worker_id": "zephyr"}),
        content_type="application/json",
        **worker_headers(),
    )
    assert mismatch_response.status_code == 403

    response = client.post(
        "/v1/internal/workers/heartbeat",
        data=json.dumps(
            {
                "worker_id": "atlas",
                "hostname": "atlas.local",
                "version": "0.1.0",
                "concurrency": 1,
                "capabilities": plain_capabilities(),
                "running_job_ids": [],
            }
        ),
        content_type="application/json",
        **worker_headers(),
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["worker_id"] == "atlas"
    assert payload["enabled"] is True
    worker = Worker.objects.get(worker_id="atlas")
    assert worker.hostname == "atlas.local"
    assert worker.capabilities["job_types"] == ["transcribe"]


@pytest.mark.django_db
@pytest.mark.parametrize(
    "path",
    [
        "/v1/internal/workers/heartbeat",
        "/v1/internal/work/claim",
        "/v1/internal/work/00000000-0000-0000-0000-000000000001/heartbeat",
        "/v1/internal/work/00000000-0000-0000-0000-000000000001/complete",
        "/v1/internal/work/00000000-0000-0000-0000-000000000001/fail",
    ],
)
def test_worker_endpoints_authenticate_before_body_parse(client, settings, path):
    enable_remote_workers(settings)

    response = client.post(path, data="not-json", content_type="application/json")

    assert response.status_code == 401
    assert response.json()["error"]["type"] == "authentication_error"
    assert "worker bearer token" in response.json()["error"]["message"]


@pytest.mark.django_db
def test_worker_json_request_rejects_invalid_utf8_with_json_error(client, settings):
    enable_remote_workers(settings)

    response = client.post(
        "/v1/internal/workers/heartbeat",
        data=b"\xff",
        content_type="application/json",
        **worker_headers(),
    )

    assert response.status_code == 400
    assert response.headers["Content-Type"].startswith("application/json")
    assert response.json()["error"]["message"] == "Request body was not valid JSON."


@pytest.mark.django_db
def test_worker_heartbeat_rejects_disabled_worker_without_mutating(client, settings):
    enable_remote_workers(settings)
    Worker.objects.create(
        worker_id="atlas",
        hostname="old-host",
        enabled=False,
        capabilities={"job_types": ["transcribe"]},
        concurrency=1,
        running_job_ids=[],
        last_seen_at=timezone.now() - timedelta(seconds=60),
    )

    response = client.post(
        "/v1/internal/workers/heartbeat",
        data=json.dumps(
            {
                "worker_id": "atlas",
                "hostname": "new-host",
                "capabilities": plain_capabilities(),
                "running_job_ids": ["job-1"],
            }
        ),
        content_type="application/json",
        **worker_headers(),
    )

    assert response.status_code == 403
    worker = Worker.objects.get(worker_id="atlas")
    assert worker.hostname == "old-host"
    assert worker.capabilities == {"job_types": ["transcribe"]}
    assert worker.running_job_ids == []


@pytest.mark.django_db
def test_remote_pull_dispatch_keeps_transcription_out_of_django_tasks(client, settings):
    job = submit_remote_job(client, settings)

    assert job.execution_mode == Job.ExecutionMode.REMOTE_PULL
    assert job.state == Job.State.QUEUED
    assert job.django_task_id == ""
    assert job.assigned_worker_id == ""


@pytest.mark.django_db
def test_worker_claim_sets_studio_timed_lease_and_attempt_prefix(client, settings):
    job = submit_remote_job(client, settings)

    response = claim_one(client)

    payload = response.json()["job"]
    assert response.status_code == 200
    assert payload["id"] == str(job.id)
    assert payload["attempt"] == 1
    assert payload["lease_token"]
    assert payload["backend"] == "whispercpp"
    assert payload["model"] == "ggml-large-v3.bin"
    assert payload["requested_backend"] == "auto"
    assert payload["requested_model"] == "auto"
    assert payload["artifact_prefix"] == f"voxhelm/jobs/{job.id}/attempt-1/"
    assert payload["artifact_store"]["backend"] == "filesystem"
    assert payload["input"] == {
        "kind": "url",
        "url": "https://media.example.com/episode.mp3",
    }
    job.refresh_from_db()
    assert job.state == Job.State.RUNNING
    assert job.assigned_worker_id == "atlas"
    assert job.lease_token_hash
    assert job.lease_expires_at is not None
    assert job.attempt_count == 1
    assert job.leased_artifact_prefix == f"voxhelm/jobs/{job.id}/attempt-1/"
    assert job.leased_artifact_store["backend"] == "filesystem"


@pytest.mark.django_db
def test_worker_claim_includes_default_anonymous_diarization_strategy(client, settings):
    job = submit_remote_job(
        client,
        settings,
        task_ref="remote-anonymous-diarization-claim",
        diarization={"enabled": True, "min_speakers": 2},
    )

    response = claim_one(client, capabilities=known_speaker_capabilities())

    payload = response.json()["job"]
    assert response.status_code == 200
    assert payload["id"] == str(job.id)
    assert payload["output"]["diarization"] == {
        "enabled": True,
        "min_speakers": 2,
        "strategy": "pyannote",
    }


@pytest.mark.django_db
def test_worker_claim_respects_single_job_capacity(client, settings):
    first_job = submit_remote_job(client, settings, task_ref="remote-capacity-1")
    second_job = submit_remote_job(client, settings, task_ref="remote-capacity-2")

    first_response = claim_one(client)
    second_response = claim_one(client)

    assert first_response.status_code == 200
    assert first_response.json()["job"]["id"] == str(first_job.id)
    assert second_response.status_code == 204
    second_job.refresh_from_db()
    assert second_job.state == Job.State.QUEUED


@pytest.mark.django_db
def test_worker_claim_requires_advertised_output_formats(client, settings):
    job = submit_remote_job(client, settings)

    limited_response = claim_one(client, capabilities=limited_text_capabilities())
    assert limited_response.status_code == 204
    job.refresh_from_db()
    assert job.state == Job.State.QUEUED

    full_response = claim_one(client)
    assert full_response.status_code == 200
    assert full_response.json()["job"]["id"] == str(job.id)


@pytest.mark.django_db
def test_worker_claim_requires_advertised_stt_capability_for_auto_jobs(client, settings):
    job = submit_remote_job(client, settings, task_ref="remote-auto-needs-stt-capability")

    no_stt_response = claim_one(client, capabilities=no_stt_capabilities())
    assert no_stt_response.status_code == 204
    job.refresh_from_db()
    assert job.state == Job.State.QUEUED

    incompatible_response = claim_one(client, capabilities=incompatible_stt_capabilities())
    assert incompatible_response.status_code == 204
    job.refresh_from_db()
    assert job.state == Job.State.QUEUED

    full_response = claim_one(client)
    assert full_response.status_code == 200
    assert full_response.json()["job"]["id"] == str(job.id)


@pytest.mark.django_db
@pytest.mark.parametrize("model_alias", ["whisper-1", "gpt-4o-mini-transcribe"])
def test_worker_claim_treats_openai_model_aliases_as_default_stt_capability(
    client,
    settings,
    model_alias,
):
    settings.VOXHELM_STT_BACKEND = "whispercpp"
    settings.VOXHELM_WHISPERCPP_MODEL = "ggml-large-v3.bin"
    job = submit_remote_job(
        client,
        settings,
        model=model_alias,
        task_ref=f"remote-model-alias-{model_alias}",
    )

    response = claim_one(client)

    assert response.status_code == 200
    assert response.json()["job"]["id"] == str(job.id)


@pytest.mark.django_db
def test_remote_submission_rejects_disallowed_url_before_queueing(client, settings):
    enable_remote_workers(settings)

    response = client.post(
        "/v1/jobs",
        data=json.dumps(
            build_job_payload(
                url="https://blocked.example.com/episode.mp3",
                task_ref="remote-blocked-url",
            )
        ),
        content_type="application/json",
        **producer_headers(),
    )

    assert response.status_code == 400
    assert "allowlist" in response.json()["error"]["message"]
    assert not Job.objects.exists()


@pytest.mark.django_db
def test_worker_claim_fails_expired_running_job_when_revalidated_url_is_disallowed(
    client,
    settings,
):
    job = submit_remote_job(
        client,
        settings,
        url="https://media.example.com/episode.mp3",
        task_ref="remote-stale-blocked-url",
    )
    assert claim_one(client).status_code == 200
    job.refresh_from_db()
    job.lease_expires_at = timezone.now() - timedelta(seconds=1)
    job.save(update_fields=["lease_expires_at"])
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"cdn.example.com"}

    response = claim_one(client)

    assert response.status_code == 204
    job.refresh_from_db()
    assert job.state == Job.State.FAILED
    assert "allowlist" in job.error_detail
    assert job.worker_progress == {}


@pytest.mark.django_db
def test_worker_claim_fails_job_when_known_speaker_reference_url_is_disallowed(
    client,
    settings,
):
    job = submit_remote_job(
        client,
        settings,
        task_ref="remote-stale-blocked-known-speaker-reference",
        diarization={
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
                                "url": "https://cdn.example.com/reference.m4a",
                            },
                            "start": 0.0,
                            "end": 2.0,
                        }
                    ],
                }
            ],
        },
    )
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}

    response = claim_one(client, capabilities=known_speaker_capabilities())

    assert response.status_code == 204
    job.refresh_from_db()
    assert job.state == Job.State.FAILED
    assert "allowlist" in job.error_detail
    assert job.worker_progress == {}


@pytest.mark.django_db
def test_worker_claim_fails_job_with_uploaded_known_speaker_reference_audio(
    client,
    settings,
):
    job = submit_remote_job(
        client,
        settings,
        task_ref="remote-uploaded-known-speaker-reference",
        diarization={
            "enabled": True,
            "strategy": "pyannote_known_speaker",
            "known_speakers": [
                {
                    "id": "12",
                    "name": "Johannes",
                    "references": [
                        {
                            "kind": "clip_artifact",
                            "audio": {
                                "kind": "upload",
                                "upload_id": "00000000-0000-0000-0000-000000000123",
                            },
                        }
                    ],
                }
            ],
        },
    )

    response = claim_one(client, capabilities=known_speaker_capabilities())

    assert response.status_code == 204
    job.refresh_from_db()
    assert job.state == Job.State.FAILED
    assert "Uploaded known-speaker reference audio" in job.error_detail
    assert job.worker_progress == {}


@pytest.mark.django_db
def test_preclaim_remote_failure_releases_staged_upload_for_retry(client, settings):
    job, upload_id, _staged_key = submit_remote_upload_job(
        client,
        settings,
        task_ref="remote-upload-stale-known-speaker-reference",
    )
    job.output_data["diarization"] = {
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
    job.save(update_fields=["output_data"])

    response = claim_one(client, capabilities=known_speaker_capabilities())

    assert response.status_code == 204
    job.refresh_from_db()
    assert job.state == Job.State.FAILED
    assert "allowlist" in job.error_detail
    staged = StagedMedia.objects.get(id=upload_id)
    assert staged.claimed_by_job_id is None
    retry_response = client.post(
        "/v1/jobs",
        data=json.dumps(
            build_job_payload(
                input_data={"kind": "upload", "upload_id": upload_id},
                task_ref="remote-upload-stale-known-speaker-reference-retry",
            )
        ),
        content_type="application/json",
        **producer_headers(),
    )
    assert retry_response.status_code == 201
    assert retry_response.json()["id"] != str(job.id)


@pytest.mark.django_db
def test_known_speaker_jobs_require_matching_worker_capabilities(client, settings):
    job = submit_remote_job(
        client,
        settings,
        task_ref="remote-known-speaker",
        diarization={
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
                                "url": "https://cdn.example.com/pp_60.m4a",
                            },
                            "start": 0.0,
                            "end": 2.0,
                        }
                    ],
                }
            ],
        },
    )

    plain_response = claim_one(client)
    assert plain_response.status_code == 204
    job.refresh_from_db()
    assert job.state == Job.State.QUEUED

    capable_response = claim_one(client, capabilities=known_speaker_capabilities())
    payload = capable_response.json()["job"]
    assert capable_response.status_code == 200
    assert payload["id"] == str(job.id)
    assert payload["output"]["diarization"]["strategy"] == "pyannote_known_speaker"
    assert payload["output"]["diarization"]["known_speakers"][0]["name"] == "Johannes"


@pytest.mark.django_db
def test_worker_claim_scans_past_incompatible_candidates(client, settings):
    known_speaker = {
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
    }
    for index in range(25):
        submit_remote_job(
            client,
            settings,
            task_ref=f"remote-incompatible-{index}",
            diarization=known_speaker,
        )
    compatible = submit_remote_job(client, settings, task_ref="remote-compatible-after-head")

    response = claim_one(client)

    assert response.status_code == 200
    assert response.json()["job"]["id"] == str(compatible.id)


@pytest.mark.django_db
def test_worker_claim_revalidates_candidates_before_worker_write_lock(
    client,
    settings,
    monkeypatch,
):
    job = submit_remote_job(client, settings, task_ref="remote-claim-revalidate-outside-lock")
    worker_lock_started = False

    def claimable_without_write_lock(candidate: Job) -> bool:
        assert candidate.id == job.id
        assert not worker_lock_started
        return True

    original_lock_worker_for_claim = __import__(
        "jobs.remote_workers",
        fromlist=["lock_worker_for_claim"],
    ).lock_worker_for_claim

    def record_worker_lock(*, worker_id: str, now: object) -> None:
        nonlocal worker_lock_started
        worker_lock_started = True
        original_lock_worker_for_claim(worker_id=worker_id, now=now)

    monkeypatch.setattr(
        "jobs.remote_workers.remote_payload_is_claimable",
        claimable_without_write_lock,
    )
    monkeypatch.setattr("jobs.remote_workers.lock_worker_for_claim", record_worker_lock)

    response = claim_one(client)

    assert response.status_code == 200
    assert response.json()["job"]["id"] == str(job.id)


@pytest.mark.django_db
def test_worker_claim_rescans_after_prefiltered_candidate_loses_race(
    client,
    settings,
    monkeypatch,
):
    first = submit_remote_job(client, settings, task_ref="remote-raced-candidate-1")
    second = submit_remote_job(client, settings, task_ref="remote-raced-candidate-2")
    stale_first = Job.objects.get(id=first.id)
    Job.objects.filter(id=first.id).update(attempt_count=1)
    current_second = Job.objects.get(id=second.id)
    calls = 0

    def raced_then_next_candidate(*, now: object, capabilities: dict[str, Any]) -> list[Job]:
        del now, capabilities
        nonlocal calls
        calls += 1
        if calls == 1:
            return [stale_first]
        return [current_second]

    monkeypatch.setattr("jobs.remote_workers.remote_claim_candidates", raced_then_next_candidate)

    response = claim_one(client)

    assert response.status_code == 200
    assert response.json()["job"]["id"] == str(second.id)
    assert calls == 2


@pytest.mark.django_db
def test_remote_completion_persists_manifest_and_serves_existing_artifact_url(
    client,
    monkeypatch,
    settings,
):
    job = submit_remote_job(client, settings)
    claim = claim_one(client).json()["job"]
    lease_token = claim["lease_token"]
    manifest = transcript_manifest(str(job.id), claim["attempt"])
    store_manifest_artifacts(settings, manifest)
    body = {
        "worker_id": "atlas",
        "lease_token": lease_token,
        "result_text": "remote transcript",
        "result_metadata": {
            "backend": "whisper.cpp",
            "requested_model": "worker-lied",
            "model": "ggml-large-v3.bin",
            "language": "en",
            "source_url": "https://cdn.example.com/resolved-episode.mp3?token=secret&expires=1",
            "processing_seconds": 12.3,
        },
        "artifacts": manifest,
    }

    response = client.post(
        f"/v1/internal/work/{job.id}/complete",
        data=json.dumps(body),
        content_type="application/json",
        **worker_headers(),
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["state"] == "succeeded"
    assert "lease_token" not in json.dumps(payload)
    assert payload["result"]["metadata"]["worker_id"] == "atlas"
    assert payload["result"]["metadata"]["attempt"] == 1
    assert payload["result"]["metadata"]["execution_mode"] == "remote_pull"
    assert payload["result"]["metadata"]["requested_model"] == "auto"
    assert payload["result"]["metadata"]["source_url"] == "https://media.example.com/episode.mp3"
    assert payload["result"]["artifacts"]["text"].endswith("/transcript.txt")
    assert JobArtifact.objects.filter(job=job, name="transcript.txt").exists()

    artifact_response = client.get(payload["result"]["artifacts"]["text"], **producer_headers())
    assert artifact_response.status_code == 200
    assert artifact_response.content == b"remote transcript"

    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}
    retry_response = client.post(
        f"/v1/internal/work/{job.id}/complete",
        data=json.dumps(body),
        content_type="application/json",
        **worker_headers(),
    )
    assert retry_response.status_code == 200
    assert retry_response.json()["id"] == str(job.id)

    def unavailable_store(store_identity):
        del store_identity
        raise RuntimeError("S3 temporarily unavailable")

    monkeypatch.setattr("jobs.remote_workers.get_artifact_store_for_identity", unavailable_store)

    retry_response = client.post(
        f"/v1/internal/work/{job.id}/complete",
        data=json.dumps(body),
        content_type="application/json",
        **worker_headers(),
    )
    assert retry_response.status_code == 200

    conflicting = dict(body)
    conflicting["result_text"] = "changed"
    conflict_response = client.post(
        f"/v1/internal/work/{job.id}/complete",
        data=json.dumps(conflicting),
        content_type="application/json",
        **worker_headers(),
    )
    assert conflict_response.status_code == 409


@pytest.mark.django_db
def test_remote_completion_verifies_artifact_objects_outside_settlement_transaction(
    client,
    monkeypatch,
    settings,
):
    job = submit_remote_job(client, settings, task_ref="remote-artifact-stat-outside-txn")
    claim = claim_one(client).json()["job"]
    manifest = transcript_manifest(str(job.id), claim["attempt"])
    store_manifest_artifacts(settings, manifest)
    settlement_started = False

    class AssertingStore:
        backend_name = "filesystem"

        def stat(self, *, key: str) -> StoredArtifact:
            assert not settlement_started
            path = settings.VOXHELM_ARTIFACT_ROOT / key
            return StoredArtifact(backend="filesystem", key=key, size_bytes=path.stat().st_size)

    original_lock_job_for_settlement = __import__(
        "jobs.remote_workers",
        fromlist=["lock_job_for_settlement"],
    ).lock_job_for_settlement

    def record_settlement_lock(*, job_id, now: object) -> None:
        nonlocal settlement_started
        settlement_started = True
        original_lock_job_for_settlement(job_id=job_id, now=now)

    monkeypatch.setattr(
        "jobs.remote_workers.get_artifact_store_for_identity",
        lambda store_identity: AssertingStore(),
    )
    monkeypatch.setattr("jobs.remote_workers.lock_job_for_settlement", record_settlement_lock)

    response = client.post(
        f"/v1/internal/work/{job.id}/complete",
        data=json.dumps(
            {
                "worker_id": "atlas",
                "lease_token": claim["lease_token"],
                "result_text": "remote transcript",
                "result_metadata": {
                    "backend": "whisper.cpp",
                    "source_url": "worker-local-source-token",
                },
                "artifacts": manifest,
            }
        ),
        content_type="application/json",
        **worker_headers(),
    )

    assert response.status_code == 200


@pytest.mark.django_db
def test_remote_completion_uses_leased_artifact_snapshot_after_setting_change(
    client,
    settings,
    tmp_path,
):
    job = submit_remote_job(client, settings, task_ref="remote-claimed-prefix-survives-rollout")
    claim = claim_one(client).json()["job"]
    assert claim["artifact_prefix"] == f"voxhelm/jobs/{job.id}/attempt-1/"
    manifest = transcript_manifest(str(job.id), claim["attempt"])
    store_manifest_artifacts(settings, manifest)
    settings.VOXHELM_ARTIFACT_PREFIX = "new-prefix-after-claim"
    settings.VOXHELM_ARTIFACT_ROOT = tmp_path / "new-artifact-root"
    get_artifact_store.cache_clear()

    response = client.post(
        f"/v1/internal/work/{job.id}/complete",
        data=json.dumps(
            {
                "worker_id": "atlas",
                "lease_token": claim["lease_token"],
                "result_text": "remote transcript",
                "result_metadata": {"backend": "whisper.cpp"},
                "artifacts": manifest,
            }
        ),
        content_type="application/json",
        **worker_headers(),
    )

    assert response.status_code == 200
    assert response.json()["state"] == "succeeded"
    artifact_response = client.get(
        f"/v1/jobs/{job.id}/artifacts/transcript.txt",
        **producer_headers(),
    )
    assert artifact_response.status_code == 200
    assert artifact_response.content == b"remote transcript"


@pytest.mark.django_db
def test_disabled_worker_can_complete_existing_lease(client, settings):
    job = submit_remote_job(client, settings, task_ref="remote-disabled-worker-complete")
    claim = claim_one(client).json()["job"]
    manifest = transcript_manifest(str(job.id), claim["attempt"])
    store_manifest_artifacts(settings, manifest)
    Worker.objects.filter(worker_id="atlas").update(enabled=False)

    response = client.post(
        f"/v1/internal/work/{job.id}/complete",
        data=json.dumps(
            {
                "worker_id": "atlas",
                "lease_token": claim["lease_token"],
                "result_text": "remote transcript",
                "result_metadata": {"backend": "whisper.cpp"},
                "artifacts": manifest,
            }
        ),
        content_type="application/json",
        **worker_headers(),
    )

    assert response.status_code == 200
    assert response.json()["state"] == "succeeded"


@pytest.mark.django_db
def test_remote_completion_deletes_staged_upload_after_success(client, settings):
    job, upload_id, staged_key = submit_remote_upload_job(
        client,
        settings,
        task_ref="remote-upload-cleanup",
    )
    claim = claim_one(client).json()["job"]
    manifest = transcript_manifest(str(job.id), claim["attempt"])
    store_manifest_artifacts(settings, manifest)

    response = client.post(
        f"/v1/internal/work/{job.id}/complete",
        data=json.dumps(
            {
                "worker_id": "atlas",
                "lease_token": claim["lease_token"],
                "result_text": "remote transcript",
                "result_metadata": {"backend": "whisper.cpp"},
                "artifacts": manifest,
            }
        ),
        content_type="application/json",
        **worker_headers(),
    )

    assert response.status_code == 200
    assert "source_url" not in response.json()["result"]["metadata"]
    assert claim["input"]["staged_artifact"]["storage_key"] == staged_key
    assert claim["input"]["staged_artifact"]["storage_backend"] == "filesystem"
    assert claim["input"]["staged_artifact"]["storage_identity"]["backend"] == "filesystem"
    assert not StagedMedia.objects.filter(id=upload_id).exists()
    assert not (settings.VOXHELM_ARTIFACT_ROOT / staged_key).exists()


@pytest.mark.django_db
def test_remote_completion_rejects_missing_artifact_object_and_keeps_upload(client, settings):
    job, upload_id, staged_key = submit_remote_upload_job(
        client,
        settings,
        task_ref="remote-upload-missing-artifact",
    )
    claim = claim_one(client).json()["job"]
    manifest = transcript_manifest(str(job.id), claim["attempt"])
    store_manifest_artifacts(settings, manifest, skip_names={"transcript.txt"})

    response = client.post(
        f"/v1/internal/work/{job.id}/complete",
        data=json.dumps(
            {
                "worker_id": "atlas",
                "lease_token": claim["lease_token"],
                "result_text": "remote transcript",
                "result_metadata": {"backend": "whisper.cpp"},
                "artifacts": manifest,
            }
        ),
        content_type="application/json",
        **worker_headers(),
    )

    assert response.status_code == 400
    assert "transcript.txt" in response.json()["error"]["message"]
    assert "artifact store" in response.json()["error"]["message"]
    assert not JobArtifact.objects.filter(job=job).exists()
    job.refresh_from_db()
    assert job.state == Job.State.RUNNING
    assert StagedMedia.objects.filter(id=upload_id).exists()
    assert (settings.VOXHELM_ARTIFACT_ROOT / staged_key).exists()


@pytest.mark.django_db
def test_remote_completion_reports_artifact_store_configuration_error(
    client,
    settings,
    monkeypatch,
):
    job = submit_remote_job(client, settings, task_ref="remote-artifact-store-configuration")
    claim = claim_one(client).json()["job"]
    manifest = transcript_manifest(str(job.id), claim["attempt"])

    def unavailable_store(store_identity):
        del store_identity
        raise RuntimeError("S3 artifact backend is missing configuration: endpoint")

    monkeypatch.setattr("jobs.remote_workers.get_artifact_store_for_identity", unavailable_store)

    response = client.post(
        f"/v1/internal/work/{job.id}/complete",
        data=json.dumps(
            {
                "worker_id": "atlas",
                "lease_token": claim["lease_token"],
                "result_text": "remote transcript",
                "result_metadata": {"backend": "whisper.cpp"},
                "artifacts": manifest,
            }
        ),
        content_type="application/json",
        **worker_headers(),
    )

    assert response.status_code == 400
    assert "Artifact store is not configured" in response.json()["error"]["message"]
    job.refresh_from_db()
    assert job.state == Job.State.RUNNING
    assert not JobArtifact.objects.filter(job=job).exists()


@pytest.mark.django_db
def test_remote_completion_rejects_artifact_size_mismatch(client, settings):
    job = submit_remote_job(client, settings, task_ref="remote-artifact-size-mismatch")
    claim = claim_one(client).json()["job"]
    manifest = transcript_manifest(str(job.id), claim["attempt"])
    store_manifest_artifacts(settings, manifest)
    store_remote_artifact(
        settings,
        key=str(manifest[1]["storage_key"]),
        content=b"too short",
    )

    response = client.post(
        f"/v1/internal/work/{job.id}/complete",
        data=json.dumps(
            {
                "worker_id": "atlas",
                "lease_token": claim["lease_token"],
                "result_text": "remote transcript",
                "result_metadata": {"backend": "whisper.cpp"},
                "artifacts": manifest,
            }
        ),
        content_type="application/json",
        **worker_headers(),
    )

    assert response.status_code == 400
    assert "size_bytes" in response.json()["error"]["message"]
    assert not JobArtifact.objects.filter(job=job).exists()


@pytest.mark.django_db
def test_remote_completion_rejects_upload_without_source_artifact(client, settings):
    job, upload_id, staged_key = submit_remote_upload_job(
        client,
        settings,
        task_ref="remote-upload-requires-source",
    )
    claim = claim_one(client).json()["job"]
    manifest = [
        artifact
        for artifact in transcript_manifest(str(job.id), claim["attempt"])
        if artifact["kind"] != "source"
    ]

    response = client.post(
        f"/v1/internal/work/{job.id}/complete",
        data=json.dumps(
            {
                "worker_id": "atlas",
                "lease_token": claim["lease_token"],
                "result_text": "remote transcript",
                "result_metadata": {"backend": "whisper.cpp"},
                "artifacts": manifest,
            }
        ),
        content_type="application/json",
        **worker_headers(),
    )

    assert response.status_code == 400
    assert "source artifact" in response.json()["error"]["message"]
    assert StagedMedia.objects.filter(id=upload_id).exists()
    assert (settings.VOXHELM_ARTIFACT_ROOT / staged_key).exists()


@pytest.mark.django_db
def test_remote_completion_accepts_empty_result_text(client, settings):
    job = submit_remote_job(client, settings, task_ref="remote-empty-transcript")
    claim = claim_one(client).json()["job"]
    manifest = transcript_manifest(str(job.id), claim["attempt"])
    store_manifest_artifacts(settings, manifest)

    response = client.post(
        f"/v1/internal/work/{job.id}/complete",
        data=json.dumps(
            {
                "worker_id": "atlas",
                "lease_token": claim["lease_token"],
                "result_text": "",
                "result_metadata": {"backend": "whisper.cpp"},
                "artifacts": manifest,
            }
        ),
        content_type="application/json",
        **worker_headers(),
    )

    assert response.status_code == 200
    assert response.json()["state"] == "succeeded"


@pytest.mark.django_db
def test_remote_completion_rechecks_lease_at_commit(client, settings, monkeypatch):
    job = submit_remote_job(client, settings, task_ref="remote-expired-during-complete")
    claim = claim_one(client).json()["job"]
    manifest = transcript_manifest(str(job.id), claim["attempt"])
    store_manifest_artifacts(settings, manifest)
    from jobs import remote_workers

    original_validate = remote_workers.validate_remote_artifacts

    def expire_lease_during_validation(*, job: Job, artifacts: object):
        result = original_validate(job=job, artifacts=artifacts)
        Job.objects.filter(id=job.id).update(lease_expires_at=timezone.now() - timedelta(seconds=1))
        return result

    monkeypatch.setattr(remote_workers, "validate_remote_artifacts", expire_lease_during_validation)

    response = client.post(
        f"/v1/internal/work/{job.id}/complete",
        data=json.dumps(
            {
                "worker_id": "atlas",
                "lease_token": claim["lease_token"],
                "result_text": "remote transcript",
                "result_metadata": {"backend": "whisper.cpp"},
                "artifacts": manifest,
            }
        ),
        content_type="application/json",
        **worker_headers(),
    )

    assert response.status_code == 409
    assert not JobArtifact.objects.filter(job=job).exists()


@pytest.mark.django_db
def test_remote_completion_rejects_artifacts_outside_claimed_attempt_prefix(client, settings):
    job = submit_remote_job(client, settings)
    claim = claim_one(client).json()["job"]
    manifest = transcript_manifest(str(job.id), claim["attempt"])
    manifest[1]["storage_key"] = f"voxhelm/jobs/{job.id}/transcript.txt"

    response = client.post(
        f"/v1/internal/work/{job.id}/complete",
        data=json.dumps(
            {
                "worker_id": "atlas",
                "lease_token": claim["lease_token"],
                "result_text": "remote transcript",
                "result_metadata": {"backend": "whisper.cpp"},
                "artifacts": manifest,
            }
        ),
        content_type="application/json",
        **worker_headers(),
    )

    assert response.status_code == 400
    assert "attempt prefix" in response.json()["error"]["message"]


@pytest.mark.django_db
def test_remote_completion_rejects_traversing_storage_keys(client, settings):
    job = submit_remote_job(client, settings, task_ref="remote-traversal")
    claim = claim_one(client).json()["job"]
    manifest = transcript_manifest(str(job.id), claim["attempt"])
    manifest[1]["storage_key"] = f"voxhelm/jobs/{job.id}/attempt-{claim['attempt']}/../outside.txt"

    response = client.post(
        f"/v1/internal/work/{job.id}/complete",
        data=json.dumps(
            {
                "worker_id": "atlas",
                "lease_token": claim["lease_token"],
                "result_text": "remote transcript",
                "result_metadata": {"backend": "whisper.cpp"},
                "artifacts": manifest,
            }
        ),
        content_type="application/json",
        **worker_headers(),
    )

    assert response.status_code == 400
    assert "attempt prefix" in response.json()["error"]["message"]


@pytest.mark.django_db
def test_remote_completion_rejects_transcript_artifact_content_type(client, settings):
    job = submit_remote_job(client, settings, task_ref="remote-bad-transcript-mime")
    claim = claim_one(client).json()["job"]
    manifest = transcript_manifest(str(job.id), claim["attempt"])
    manifest[1]["content_type"] = "application/json"

    response = client.post(
        f"/v1/internal/work/{job.id}/complete",
        data=json.dumps(
            {
                "worker_id": "atlas",
                "lease_token": claim["lease_token"],
                "result_text": "remote transcript",
                "result_metadata": {"backend": "whisper.cpp"},
                "artifacts": manifest,
            }
        ),
        content_type="application/json",
        **worker_headers(),
    )

    assert response.status_code == 400
    assert "content_type" in response.json()["error"]["message"]


@pytest.mark.django_db
def test_remote_completion_rejects_unrequested_speaker_sidecar(client, settings):
    job = submit_remote_job(client, settings, task_ref="remote-unrequested-speakers")
    claim = claim_one(client).json()["job"]
    manifest = [
        *transcript_manifest(str(job.id), claim["attempt"]),
        speakers_artifact(str(job.id), claim["attempt"]),
    ]
    store_manifest_artifacts(settings, manifest)

    response = client.post(
        f"/v1/internal/work/{job.id}/complete",
        data=json.dumps(
            {
                "worker_id": "atlas",
                "lease_token": claim["lease_token"],
                "result_text": "remote transcript",
                "result_metadata": {"backend": "whisper.cpp"},
                "artifacts": manifest,
            }
        ),
        content_type="application/json",
        **worker_headers(),
    )

    assert response.status_code == 400
    assert "only allowed for known-speaker jobs" in response.json()["error"]["message"]


@pytest.mark.django_db
def test_known_speaker_completion_requires_speakers_artifact_format(client, settings):
    job = submit_remote_job(
        client,
        settings,
        task_ref="remote-known-speaker-format",
        diarization={
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
                                "url": "https://cdn.example.com/pp_60.m4a",
                            },
                            "start": 0.0,
                            "end": 2.0,
                        }
                    ],
                }
            ],
        },
    )
    claim = claim_one(client, capabilities=known_speaker_capabilities()).json()["job"]
    manifest = [
        *transcript_manifest(str(job.id), claim["attempt"]),
        speakers_artifact(str(job.id), claim["attempt"], format_name="json"),
    ]

    response = client.post(
        f"/v1/internal/work/{job.id}/complete",
        data=json.dumps(
            {
                "worker_id": "atlas",
                "lease_token": claim["lease_token"],
                "result_text": "remote transcript",
                "result_metadata": {
                    "backend": "whisper.cpp",
                    "diarization": {
                        "known_speaker_summary": {"strategy": "pyannote_known_speaker"}
                    },
                },
                "artifacts": manifest,
            }
        ),
        content_type="application/json",
        **worker_headers(),
    )

    assert response.status_code == 400
    assert "Speaker sidecar" in response.json()["error"]["message"]


@pytest.mark.django_db
def test_known_speaker_completion_rejects_speaker_artifact_content_type(client, settings):
    job = submit_remote_job(
        client,
        settings,
        task_ref="remote-known-speaker-mime",
        diarization={
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
                                "url": "https://cdn.example.com/pp_60.m4a",
                            },
                            "start": 0.0,
                            "end": 2.0,
                        }
                    ],
                }
            ],
        },
    )
    claim = claim_one(client, capabilities=known_speaker_capabilities()).json()["job"]
    speaker_manifest = speakers_artifact(str(job.id), claim["attempt"])
    speaker_manifest["content_type"] = "text/plain; charset=utf-8"
    manifest = [
        *transcript_manifest(str(job.id), claim["attempt"]),
        speaker_manifest,
    ]

    response = client.post(
        f"/v1/internal/work/{job.id}/complete",
        data=json.dumps(
            {
                "worker_id": "atlas",
                "lease_token": claim["lease_token"],
                "result_text": "remote transcript",
                "result_metadata": {
                    "backend": "whisper.cpp",
                    "diarization": {
                        "known_speaker_summary": {"strategy": "pyannote_known_speaker"}
                    },
                },
                "artifacts": manifest,
            }
        ),
        content_type="application/json",
        **worker_headers(),
    )

    assert response.status_code == 400
    assert "content_type" in response.json()["error"]["message"]


@pytest.mark.django_db
def test_known_speaker_completion_strips_private_worker_metadata(client, settings):
    private_reference_url = "https://cdn.example.com/private-reference.m4a?token=abc&expires=1"
    private_reference_url_variant = (
        "https://CDN.example.com:443/private-%72eference.m4a?expires=1&token=abc"
    )
    job = submit_remote_job(
        client,
        settings,
        task_ref="remote-known-speaker-private-metadata",
        diarization={
            "enabled": True,
            "strategy": "pyannote_known_speaker",
            "known_speakers": [
                {
                    "id": "12",
                    "name": "Johannes",
                    "references": [
                        {
                            "kind": "source_range",
                            "audio": {"kind": "url", "url": private_reference_url},
                            "start": 0.0,
                            "end": 2.0,
                        }
                    ],
                }
            ],
        },
    )
    claim = claim_one(client, capabilities=known_speaker_capabilities()).json()["job"]
    manifest = [
        *transcript_manifest(str(job.id), claim["attempt"]),
        speakers_artifact(str(job.id), claim["attempt"]),
    ]
    store_manifest_artifacts(settings, manifest)

    response = client.post(
        f"/v1/internal/work/{job.id}/complete",
        data=json.dumps(
            {
                "worker_id": "atlas",
                "lease_token": claim["lease_token"],
                "result_text": "remote transcript",
                "result_metadata": {
                    "backend": "whisper.cpp",
                    "requested_model": "worker-lied",
                    "model": "https://cdn.example.com/private-model-ref",
                    "language": "//cdn.example.com/private-language-ref?token=abc",
                    "source_name": "cdn.example.com/private-reference.m4a?token=abc",
                    "source_content_type": "//cdn.example.com/private-content-type?token=abc",
                    "duration_seconds": {"source_range": [0.0, 2.0]},
                    "processing_seconds": -1,
                    "source_url": private_reference_url_variant,
                    "claim": claim,
                    "known_speakers": [
                        {
                            "id": "12",
                            "references": [{"url": private_reference_url}],
                        }
                    ],
                    "diarization": {
                        "known_speaker_summary": {
                            "strategy": "pyannote_known_speaker",
                            "known_speakers": [
                                "Johannes",
                                private_reference_url,
                                {"references": [{"url": private_reference_url}]},
                            ],
                            "segment_count": 1,
                            "confident_speaker_distribution": {
                                "Johannes": 1,
                                private_reference_url: 1,
                            },
                            "raw_diarization_available": True,
                            "claim": claim,
                            "embedding_version": "cdn.example.com/private-embedding?token=abc",
                            "reference_url": private_reference_url,
                            "min_top_similarity": 1e999,
                            "margin_median": float("inf"),
                        }
                    },
                },
                "artifacts": manifest,
            }
        ),
        content_type="application/json",
        **worker_headers(),
    )

    assert response.status_code == 200
    metadata = response.json()["result"]["metadata"]
    assert metadata["backend"] == "whisper.cpp"
    assert metadata["requested_model"] == "auto"
    assert metadata["source_url"] == "https://media.example.com/episode.mp3"
    assert "model" not in metadata
    assert "language" not in metadata
    assert "source_name" not in metadata
    assert "source_content_type" not in metadata
    assert "duration_seconds" not in metadata
    assert "processing_seconds" not in metadata
    assert "claim" not in metadata
    assert "known_speakers" not in metadata
    assert metadata["diarization"]["known_speakers"] == [
        {"id": "12", "name": "Johannes", "reference_count": 1}
    ]
    assert metadata["diarization"]["known_speaker_summary"] == {
        "strategy": "pyannote_known_speaker",
        "segment_count": 1,
        "known_speakers": ["Johannes"],
        "confident_speaker_distribution": {"Johannes": 1},
        "raw_diarization_available": True,
    }
    assert "min_top_similarity" not in metadata["diarization"]["known_speaker_summary"]
    assert "margin_median" not in metadata["diarization"]["known_speaker_summary"]
    assert private_reference_url not in json.dumps(metadata)
    assert private_reference_url_variant not in json.dumps(metadata)


@pytest.mark.django_db
def test_retryable_remote_failure_requeues_and_next_claim_uses_new_attempt(
    client,
    settings,
):
    job = submit_remote_job(client, settings)
    first_claim = claim_one(client).json()["job"]

    fail_response = client.post(
        f"/v1/internal/work/{job.id}/fail",
        data=json.dumps(
            {
                "worker_id": "atlas",
                "lease_token": first_claim["lease_token"],
                "retryable": True,
                "error_detail": "worker lost power",
            }
        ),
        content_type="application/json",
        **worker_headers(),
    )
    assert fail_response.status_code == 200
    job.refresh_from_db()
    assert job.state == Job.State.QUEUED
    assert job.assigned_worker_id == ""
    assert job.attempt_count == 1

    second_claim = claim_one(client).json()["job"]
    assert second_claim["attempt"] == 2
    assert second_claim["artifact_prefix"] == f"voxhelm/jobs/{job.id}/attempt-2/"


@pytest.mark.django_db
def test_retryable_remote_failure_uses_terminal_summary_when_attempts_exhausted(client, settings):
    settings.VOXHELM_REMOTE_WORKER_MAX_ATTEMPTS = 1
    job = submit_remote_job(client, settings, task_ref="remote-retryable-final-failure")
    claim = claim_one(client).json()["job"]

    response = client.post(
        f"/v1/internal/work/{job.id}/fail",
        data=json.dumps(
            {
                "worker_id": "atlas",
                "lease_token": claim["lease_token"],
                "retryable": True,
                "error_detail": "temporary worker error",
            }
        ),
        content_type="application/json",
        **worker_headers(),
    )

    assert response.status_code == 200
    assert response.json()["state"] == "failed"
    assert response.json()["error"]["message"] == "Remote worker reported a terminal failure."


@pytest.mark.django_db
def test_disabled_worker_can_fail_existing_lease(client, settings):
    job = submit_remote_job(client, settings, task_ref="remote-disabled-worker-fail")
    claim = claim_one(client).json()["job"]
    Worker.objects.filter(worker_id="atlas").update(enabled=False)

    response = client.post(
        f"/v1/internal/work/{job.id}/fail",
        data=json.dumps(
            {
                "worker_id": "atlas",
                "lease_token": claim["lease_token"],
                "retryable": False,
                "error_detail": "worker disabled after lease",
            }
        ),
        content_type="application/json",
        **worker_headers(),
    )

    assert response.status_code == 200
    assert response.json()["state"] == "failed"


@pytest.mark.django_db
def test_remote_failure_sanitizes_worker_error_detail(client, settings):
    private_reference_url = "https://cdn.example.com/private-reference.m4a"
    job = submit_remote_job(
        client,
        settings,
        task_ref="remote-failure-private-detail",
        diarization={
            "enabled": True,
            "strategy": "pyannote_known_speaker",
            "known_speakers": [
                {
                    "id": "12",
                    "name": "Johannes",
                    "references": [
                        {
                            "kind": "source_range",
                            "audio": {"kind": "url", "url": private_reference_url},
                            "start": 0.0,
                            "end": 2.0,
                        }
                    ],
                }
            ],
        },
    )
    claim = claim_one(client, capabilities=known_speaker_capabilities()).json()["job"]

    response = client.post(
        f"/v1/internal/work/{job.id}/fail",
        data=json.dumps(
            {
                "worker_id": "atlas",
                "lease_token": claim["lease_token"],
                "retryable": False,
                "error_detail": f"failed while fetching {private_reference_url}",
            }
        ),
        content_type="application/json",
        **worker_headers(),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] == "failed"
    assert payload["error"]["message"] == "Remote worker reported a terminal failure."
    assert private_reference_url not in json.dumps(payload)


@pytest.mark.django_db
def test_terminal_remote_failure_releases_staged_upload_for_retry(client, settings):
    job, upload_id, _staged_key = submit_remote_upload_job(
        client,
        settings,
        task_ref="remote-upload-terminal-failure",
    )
    claim = claim_one(client).json()["job"]

    fail_response = client.post(
        f"/v1/internal/work/{job.id}/fail",
        data=json.dumps(
            {
                "worker_id": "atlas",
                "lease_token": claim["lease_token"],
                "retryable": False,
                "error_detail": "worker lost power",
            }
        ),
        content_type="application/json",
        **worker_headers(),
    )

    assert fail_response.status_code == 200
    staged = StagedMedia.objects.get(id=upload_id)
    assert staged.claimed_by_job_id is None
    retry_response = client.post(
        "/v1/jobs",
        data=json.dumps(
            build_job_payload(
                input_data={"kind": "upload", "upload_id": upload_id},
                task_ref="remote-upload-terminal-failure-retry",
            )
        ),
        content_type="application/json",
        **producer_headers(),
    )
    assert retry_response.status_code == 201
    assert retry_response.json()["id"] != str(job.id)


@pytest.mark.django_db
def test_expired_final_attempt_fails_on_producer_poll(client, settings):
    settings.VOXHELM_REMOTE_WORKER_MAX_ATTEMPTS = 1
    job = submit_remote_job(client, settings, task_ref="remote-expired-final-attempt")
    claim_one(client)
    job.refresh_from_db()
    job.lease_expires_at = timezone.now() - timedelta(seconds=1)
    job.save(update_fields=["lease_expires_at"])

    response = client.get(f"/v1/jobs/{job.id}", **producer_headers())

    assert response.status_code == 200
    assert response.json()["state"] == "failed"
    assert "maximum attempts" in response.json()["error"]["message"]


@pytest.mark.django_db
def test_expired_final_attempt_releases_staged_upload_for_retry(client, settings):
    settings.VOXHELM_REMOTE_WORKER_MAX_ATTEMPTS = 1
    job, upload_id, _staged_key = submit_remote_upload_job(
        client,
        settings,
        task_ref="remote-upload-expired-final-attempt",
    )
    claim_one(client)
    job.refresh_from_db()
    job.lease_expires_at = timezone.now() - timedelta(seconds=1)
    job.save(update_fields=["lease_expires_at"])

    poll_response = client.get(f"/v1/jobs/{job.id}", **producer_headers())

    assert poll_response.status_code == 200
    assert poll_response.json()["state"] == "failed"
    staged = StagedMedia.objects.get(id=upload_id)
    assert staged.claimed_by_job_id is None
    retry_response = client.post(
        "/v1/jobs",
        data=json.dumps(
            build_job_payload(
                input_data={"kind": "upload", "upload_id": upload_id},
                task_ref="remote-upload-expired-final-attempt-retry",
            )
        ),
        content_type="application/json",
        **producer_headers(),
    )
    assert retry_response.status_code == 201
    assert retry_response.json()["id"] != str(job.id)


@pytest.mark.django_db
def test_submission_reconciles_expired_final_attempt_before_reusing_upload(
    client,
    settings,
):
    settings.VOXHELM_REMOTE_WORKER_MAX_ATTEMPTS = 1
    job, upload_id, _staged_key = submit_remote_upload_job(
        client,
        settings,
        task_ref="remote-upload-expired-final-attempt-unpolled",
    )
    claim_one(client)
    job.refresh_from_db()
    job.lease_expires_at = timezone.now() - timedelta(seconds=1)
    job.save(update_fields=["lease_expires_at"])

    retry_response = client.post(
        "/v1/jobs",
        data=json.dumps(
            build_job_payload(
                input_data={"kind": "upload", "upload_id": upload_id},
                task_ref="remote-upload-expired-final-attempt-unpolled-retry",
            )
        ),
        content_type="application/json",
        **producer_headers(),
    )

    assert retry_response.status_code == 201
    assert retry_response.json()["id"] != str(job.id)
    job.refresh_from_db()
    assert job.state == Job.State.FAILED
    staged = StagedMedia.objects.get(id=upload_id)
    assert str(staged.claimed_by_job_id) == retry_response.json()["id"]


@pytest.mark.django_db
def test_task_ref_retry_replaces_remote_job_failed_during_reconciliation(client, settings):
    settings.VOXHELM_REMOTE_WORKER_MAX_ATTEMPTS = 1
    task_ref = "remote-expired-final-attempt-retry"
    job = submit_remote_job(client, settings, task_ref=task_ref)
    claim_one(client)
    job.refresh_from_db()
    job.lease_expires_at = timezone.now() - timedelta(seconds=1)
    job.save(update_fields=["lease_expires_at"])

    response = client.post(
        "/v1/jobs",
        data=json.dumps(build_job_payload(task_ref=task_ref)),
        content_type="application/json",
        **producer_headers(),
    )

    assert response.status_code == 201
    assert response.json()["id"] != str(job.id)
    job.refresh_from_db()
    replacement = Job.objects.get(id=response.json()["id"])
    assert job.state == Job.State.FAILED
    assert replacement.state == Job.State.QUEUED
    assert replacement.execution_mode == Job.ExecutionMode.REMOTE_PULL


@pytest.mark.django_db
def test_task_ref_retry_requeues_expired_remote_lease_with_attempts_remaining(
    client,
    settings,
):
    settings.VOXHELM_REMOTE_WORKER_MAX_ATTEMPTS = 3
    task_ref = "remote-expired-lease-idempotent-retry"
    job = submit_remote_job(client, settings, task_ref=task_ref)
    claim_one(client)
    job.refresh_from_db()
    job.lease_expires_at = timezone.now() - timedelta(seconds=1)
    job.worker_progress = {"phase": "transcribing"}
    job.save(update_fields=["lease_expires_at", "worker_progress"])

    response = client.post(
        "/v1/jobs",
        data=json.dumps(build_job_payload(task_ref=task_ref)),
        content_type="application/json",
        **producer_headers(),
    )

    assert response.status_code == 200
    assert response.json()["id"] == str(job.id)
    assert response.json()["state"] == "queued"
    job.refresh_from_db()
    assert job.state == Job.State.QUEUED
    assert job.assigned_worker_id == ""
    assert job.lease_token_hash == ""
    assert job.lease_expires_at is None
    assert job.leased_artifact_prefix == ""
    assert job.last_worker_heartbeat_at is None
    assert job.worker_progress == {}
    assert job.attempt_count == 1


@pytest.mark.django_db
def test_stale_final_attempt_reconciliation_does_not_fail_extended_lease(
    client,
    settings,
):
    settings.VOXHELM_REMOTE_WORKER_MAX_ATTEMPTS = 1
    job = submit_remote_job(client, settings, task_ref="remote-stale-expiry-reconcile")
    claim_one(client)
    stale_job = Job.objects.get(id=job.id)
    expired_at = timezone.now() - timedelta(seconds=1)
    future_expiry = timezone.now() + timedelta(minutes=5)
    stale_job.lease_expires_at = expired_at
    Job.objects.filter(id=job.id).update(lease_expires_at=future_expiry)

    reconciled = reconcile_remote_job_state(stale_job)

    assert reconciled is False
    assert stale_job.state == Job.State.RUNNING
    assert stale_job.lease_expires_at == future_expiry
    job.refresh_from_db()
    assert job.state == Job.State.RUNNING
    assert job.lease_expires_at == future_expiry
    assert not job.error_detail


@pytest.mark.django_db
def test_stale_expired_lease_reconciliation_refreshes_if_already_requeued(
    client,
    settings,
):
    settings.VOXHELM_REMOTE_WORKER_MAX_ATTEMPTS = 3
    job = submit_remote_job(client, settings, task_ref="remote-stale-requeue-reconcile")
    claim_one(client)
    stale_job = Job.objects.get(id=job.id)
    expired_at = timezone.now() - timedelta(seconds=1)
    stale_job.lease_expires_at = expired_at
    Job.objects.filter(id=job.id).update(
        state=Job.State.QUEUED,
        assigned_worker_id="",
        lease_token_hash="",
        lease_expires_at=None,
        leased_artifact_prefix="",
        last_worker_heartbeat_at=None,
        worker_progress={},
    )

    reconciled = reconcile_remote_job_state(stale_job)

    assert reconciled is False
    assert stale_job.state == Job.State.QUEUED
    assert stale_job.assigned_worker_id == ""
    assert stale_job.lease_token_hash == ""
    assert stale_job.lease_expires_at is None
    assert stale_job.leased_artifact_prefix == ""
    assert stale_job.worker_progress == {}


@pytest.mark.django_db
def test_job_heartbeat_rejects_stale_lease_token(client, settings):
    job = submit_remote_job(client, settings)
    claim = claim_one(client).json()["job"]

    ok_response = client.post(
        f"/v1/internal/work/{job.id}/heartbeat",
        data=json.dumps(
            {
                "worker_id": "atlas",
                "lease_token": claim["lease_token"],
                "progress": {"phase": "transcribing", "message": "running whisper.cpp"},
            }
        ),
        content_type="application/json",
        **worker_headers(),
    )
    assert ok_response.status_code == 200
    job.refresh_from_db()
    assert job.worker_progress["phase"] == "transcribing"

    stale_response = client.post(
        f"/v1/internal/work/{job.id}/heartbeat",
        data=json.dumps({"worker_id": "atlas", "lease_token": "stale"}),
        content_type="application/json",
        **worker_headers(),
    )
    assert stale_response.status_code == 409


# --- Least-loaded fairness balancing (control-plane claim gate) ---

from jobs.remote_workers import worker_should_defer_for_fairness  # noqa: E402


def _make_worker(worker_id, *, enabled=True, seen_seconds_ago=2, concurrency=1):
    return Worker.objects.create(
        worker_id=worker_id,
        hostname=worker_id,
        enabled=enabled,
        capabilities=plain_capabilities(),
        concurrency=concurrency,
        running_job_ids=[],
        last_seen_at=timezone.now() - timedelta(seconds=seen_seconds_ago),
    )


def _make_load_job(
    worker_id,
    *,
    started_seconds_ago=10,
    state=Job.State.SUCCEEDED,
    running_lease=False,
):
    now = timezone.now()
    return Job.objects.create(
        producer="archive",
        job_type=Job.JobType.TRANSCRIBE,
        execution_mode=Job.ExecutionMode.REMOTE_PULL,
        input_data={"kind": "url", "url": "https://media.example.com/e.mp3"},
        state=state,
        assigned_worker_id=worker_id,
        started_at=now - timedelta(seconds=started_seconds_ago),
        lease_expires_at=(now + timedelta(minutes=10)) if running_lease else None,
    )


@pytest.mark.django_db
def test_fairness_defers_when_fresh_idle_peer_is_less_loaded(settings):
    _make_worker("atlas")
    _make_worker("studio")
    _make_load_job("atlas")
    _make_load_job("atlas")  # atlas=2, studio=0
    assert worker_should_defer_for_fairness(worker_id="atlas", now=timezone.now()) is True


@pytest.mark.django_db
def test_fairness_allows_when_worker_is_not_ahead(settings):
    _make_worker("atlas")
    _make_worker("studio")
    _make_load_job("studio")
    _make_load_job("studio")  # atlas=0, studio=2 -> atlas is behind
    assert worker_should_defer_for_fairness(worker_id="atlas", now=timezone.now()) is False


@pytest.mark.django_db
def test_fairness_allows_on_tie(settings):
    _make_worker("atlas")
    _make_worker("studio")
    _make_load_job("atlas")
    _make_load_job("studio")  # 1 vs 1
    assert worker_should_defer_for_fairness(worker_id="atlas", now=timezone.now()) is False


@pytest.mark.django_db
def test_fairness_ignores_stale_peer(settings):
    _make_worker("atlas")
    _make_worker("studio", seen_seconds_ago=600)  # peer not polling
    _make_load_job("atlas")
    _make_load_job("atlas")  # atlas ahead, but peer is gone
    assert worker_should_defer_for_fairness(worker_id="atlas", now=timezone.now()) is False


@pytest.mark.django_db
def test_fairness_ignores_busy_peer_with_no_spare_capacity(settings):
    _make_worker("atlas")
    _make_worker("studio")
    _make_load_job("atlas")
    _make_load_job("atlas")  # atlas ahead
    _make_load_job("studio", state=Job.State.RUNNING, running_lease=True)  # studio busy 1/1
    assert worker_should_defer_for_fairness(worker_id="atlas", now=timezone.now()) is False


@pytest.mark.django_db
def test_fairness_disabled_never_defers(settings):
    settings.VOXHELM_REMOTE_WORKER_BALANCE_ENABLED = False
    _make_worker("atlas")
    _make_worker("studio")
    _make_load_job("atlas")
    _make_load_job("atlas")
    assert worker_should_defer_for_fairness(worker_id="atlas", now=timezone.now()) is False


@pytest.mark.django_db
def test_fairness_no_peers_never_defers(settings):
    _make_worker("atlas")
    _make_load_job("atlas")
    assert worker_should_defer_for_fairness(worker_id="atlas", now=timezone.now()) is False


@pytest.mark.django_db
def test_remote_completion_deletes_replaced_intermediate_objects(client, settings):
    job = submit_remote_job(client, settings)
    claim = claim_one(client).json()["job"]
    stale_key = f"voxhelm/jobs/{job.id}/attempt-0/stale-source.mp3"
    stale_transcript_key = f"voxhelm/jobs/{job.id}/attempt-0/stale-transcript.txt"
    store_remote_artifact(settings, key=stale_key, content=b"stale")
    exposed_source_key = f"voxhelm/jobs/{job.id}/attempt-0/exposed-source.mp3"
    store_remote_artifact(settings, key=stale_transcript_key, content=b"old text")
    store_remote_artifact(settings, key=exposed_source_key, content=b"exposed")
    for name, kind, key, exposed in (
        ("stale-source.mp3", JobArtifact.Kind.SOURCE, stale_key, False),
        ("stale-transcript.txt", JobArtifact.Kind.TRANSCRIPT_TEXT, stale_transcript_key, True),
        # Shares the transcript's object: must survive with it.
        ("shared-source.mp3", JobArtifact.Kind.SOURCE, stale_transcript_key, False),
        ("exposed-source.mp3", JobArtifact.Kind.SOURCE, exposed_source_key, True),
    ):
        JobArtifact.objects.create(
            job=job,
            name=name,
            kind=kind,
            format="source",
            storage_backend="filesystem",
            storage_key=key,
            storage_identity=current_artifact_store_identity(),
            content_type="audio/mpeg",
            size_bytes=5,
            exposed=exposed,
        )
    manifest = transcript_manifest(str(job.id), claim["attempt"])
    store_manifest_artifacts(settings, manifest)

    response = client.post(
        f"/v1/internal/work/{job.id}/complete",
        data=json.dumps(
            {
                "worker_id": "atlas",
                "lease_token": claim["lease_token"],
                "result_text": "remote transcript",
                "artifacts": manifest,
            }
        ),
        content_type="application/json",
        **worker_headers(),
    )

    assert response.status_code == 200
    root = Path(settings.VOXHELM_ARTIFACT_ROOT)
    assert not (root / stale_key).exists()
    # Only non-exposed D-09 intermediates that nothing else references are deleted.
    assert (root / stale_transcript_key).exists()
    assert (root / exposed_source_key).exists()
    assert not PendingArtifactDeletion.objects.exists()
    for artifact in manifest:
        assert (root / str(artifact["storage_key"])).exists()


@pytest.mark.django_db
def test_remote_completion_retry_still_matches_after_source_was_pruned(client, settings):
    job = submit_remote_job(client, settings)
    claim = claim_one(client).json()["job"]
    manifest = transcript_manifest(str(job.id), claim["attempt"])
    store_manifest_artifacts(settings, manifest)
    body = {
        "worker_id": "atlas",
        "lease_token": claim["lease_token"],
        "result_text": "remote transcript",
        "artifacts": manifest,
    }
    first = client.post(
        f"/v1/internal/work/{job.id}/complete",
        data=json.dumps(body),
        content_type="application/json",
        **worker_headers(),
    )
    assert first.status_code == 200

    settings.VOXHELM_SOURCE_ARTIFACT_RETENTION_SECONDS = 0
    pruned = prune_job_artifacts(now=timezone.now() + timedelta(seconds=1))
    assert [artifact.name for artifact in pruned.deleted] == ["source.mp3"]

    retry = client.post(
        f"/v1/internal/work/{job.id}/complete",
        data=json.dumps(body),
        content_type="application/json",
        **worker_headers(),
    )
    assert retry.status_code == 200

    changed = json.loads(json.dumps(body))
    changed["artifacts"][1]["size_bytes"] = 18
    conflict = client.post(
        f"/v1/internal/work/{job.id}/complete",
        data=json.dumps(changed),
        content_type="application/json",
        **worker_headers(),
    )
    assert conflict.status_code == 409
