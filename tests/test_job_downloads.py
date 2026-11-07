from __future__ import annotations

import asyncio
import io
import json
from types import SimpleNamespace

import pytest
from asgiref.sync import sync_to_async
from botocore.exceptions import ClientError, EndpointConnectionError
from django.http import StreamingHttpResponse
from django.test import AsyncClient

from jobs.artifacts import S3ArtifactStore, get_artifact_store
from jobs.models import Job, JobArtifact
from tests.test_jobs import build_job_payload, build_synthesis_payload, configure_task_backend


@pytest.mark.django_db
@pytest.mark.parametrize("field,value", [("speed", 10**400), ("speed", True), ("known", 10**400)])
def test_job_post_rejects_unrepresentable_numbers(client, settings, field, value):
    configure_task_backend(settings, "django_tasks.backends.dummy.DummyBackend")
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}
    if field == "speed":
        payload = build_synthesis_payload()
        payload["speed"] = value
    else:
        payload = build_job_payload()
        payload["diarization"] = {
            "enabled": True,
            "strategy": "pyannote_known_speaker",
            "known_speakers": [
                {
                    "id": "synthetic",
                    "name": "Synthetic",
                    "references": [
                        {
                            "kind": "clip_artifact",
                            "audio": {
                                "kind": "url",
                                "url": "https://media.example.com/reference.wav",
                            },
                        }
                    ],
                }
            ],
            "known_speaker": {"min_segment_duration": value},
        }
    response = client.post(
        "/v1/jobs",
        data=json.dumps(payload),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )
    assert response.status_code == 400
    assert "error" in response.json()
    expected_field = (
        "speed" if field == "speed" else "diarization.known_speaker.min_segment_duration"
    )
    assert expected_field in response.json()["error"]["message"]
    assert not Job.objects.exists()


class BoundedReader(io.RawIOBase):
    def __init__(self):
        super().__init__()
        self.remaining = 1024 * 1024 + 7
        self.requests = []

    def read(self, size=-1):
        assert 0 < size <= 65536
        self.requests.append(size)
        count = min(self.remaining, size)
        self.remaining -= count
        return b"x" * count


def install_store(monkeypatch, factory):
    from operators import views

    monkeypatch.setattr("jobs.artifacts.get_artifact_store_for_identity", factory)
    monkeypatch.setattr(views, "get_artifact_store_for_identity", factory, raising=False)


def unbounded_read(**kwargs):
    pytest.fail("download attempted to buffer the whole artifact")


def artifact_fixture(user=None):
    job = Job.objects.create(
        producer="archive", operator=user, job_type="transcribe", input_data={}, state="succeeded"
    )
    artifact = JobArtifact.objects.create(
        job=job,
        name="transcript.txt",
        kind="transcript_text",
        format="text",
        content_type="text/plain",
        storage_backend="filesystem",
        storage_key=f"voxhelm/jobs/{job.id}/transcript.txt",
        storage_identity={"backend": "filesystem", "root": "/synthetic/recorded-store"},
        size_bytes=1024 * 1024 + 7,
        exposed=True,
    )
    return job, artifact


def artifact_url(job, operator):
    return (
        f"/transcripts/{job.id}/artifacts/text"
        if operator
        else f"/v1/jobs/{job.id}/artifacts/transcript.txt"
    )


@pytest.mark.django_db
@pytest.mark.parametrize("operator", [False, True])
@pytest.mark.parametrize("early_close", [False, True])
def test_download_streams_bounded_chunks_and_closes(
    client, django_user_model, monkeypatch, operator, early_close
):
    user = django_user_model.objects.create_user(username="synthetic-operator", is_staff=True)
    if operator:
        client.force_login(user)
    job, artifact = artifact_fixture(user)
    reader = BoundedReader()
    identities = []

    def store(identity):
        identities.append(identity)
        return SimpleNamespace(open_reader=lambda **kwargs: reader, read_bytes=unbounded_read)

    install_store(monkeypatch, store)
    response = client.get(artifact_url(job, operator), HTTP_AUTHORIZATION="Bearer test-token")
    assert response.status_code == 200
    assert response.streaming
    assert response["Content-Type"] == "text/plain"
    assert "private" in response["Cache-Control"] and "no-store" in response["Cache-Control"]
    assert identities == [artifact.storage_identity]
    assert reader.requests == []
    if early_close:
        assert next(iter(response.streaming_content))
        response.close()
        assert reader.remaining > 0
    else:
        assert sum(map(len, response.streaming_content)) == artifact.size_bytes
    assert reader.closed


@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize("operator", [False, True])
def test_asgi_download_remains_async_and_closes_on_cancellation(
    django_user_model, monkeypatch, operator
):
    user = django_user_model.objects.create_user(username="synthetic-asgi", is_staff=True)
    job, _ = artifact_fixture(user)
    reader = BoundedReader()
    install_store(
        monkeypatch,
        lambda identity: SimpleNamespace(
            open_reader=lambda **kwargs: reader, read_bytes=unbounded_read
        ),
    )

    async def request():
        client = AsyncClient()
        if operator:
            await client.aforce_login(user)
        response = await client.get(
            artifact_url(job, operator), headers={"Authorization": "Bearer test-token"}
        )
        assert response.status_code == 200
        assert isinstance(response, StreamingHttpResponse)
        assert response.streaming and response.is_async
        assert reader.requests == []
        iterator = response.streaming_content
        assert len(await anext(iterator)) <= 65536
        await iterator.aclose()
        await sync_to_async(response.close)()
        assert reader.remaining > 0
        assert reader.closed

    asyncio.run(request())


@pytest.mark.django_db
@pytest.mark.parametrize("operator", [False, True])
@pytest.mark.parametrize(
    "error,status",
    [
        (FileNotFoundError("synthetic-private-marker"), 404),
        (OSError("synthetic-private-marker"), 503),
        (
            ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "synthetic-private-marker"}}, "GetObject"
            ),
            404,
        ),
        (
            ClientError(
                {"Error": {"Code": "AccessDenied", "Message": "synthetic-private-marker"}},
                "GetObject",
            ),
            503,
        ),
        (EndpointConnectionError(endpoint_url="https://synthetic-private-marker.invalid"), 503),
    ],
)
def test_download_storage_errors_are_bounded_and_keep_metadata(
    client, django_user_model, monkeypatch, operator, error, status
):
    user = django_user_model.objects.create_user(username="synthetic-errors", is_staff=True)
    if operator:
        client.force_login(user)
    job, artifact = artifact_fixture(user)

    def fail(**kwargs):
        raise error

    install_store(monkeypatch, lambda identity: SimpleNamespace(open_reader=fail, read_bytes=fail))
    response = client.get(artifact_url(job, operator), HTTP_AUTHORIZATION="Bearer test-token")
    assert response.status_code == status
    assert "error" in response.json()
    assert b"synthetic-private-marker" not in response.content
    assert len(response.content) < 300
    assert "no-store" in response["Cache-Control"]
    assert JobArtifact.objects.filter(pk=artifact.pk).exists()


def test_s3_reader_closes_body_on_read_bytes_error():
    reader = BoundedReader()
    store = object.__new__(S3ArtifactStore)
    store.bucket = "synthetic"
    store.client = SimpleNamespace(get_object=lambda **kwargs: {"Body": reader})
    assert not store.open_reader(key="synthetic").closed
    # read_bytes explicitly owns and closes its opened body, including exceptions.
    with pytest.raises(AssertionError):
        store.read_bytes(key="synthetic")
    assert reader.closed


@pytest.mark.django_db
def test_real_filesystem_missing_download_returns_not_found(client):
    job, artifact = artifact_fixture()
    artifact.storage_identity = {}
    artifact.save(update_fields=["storage_identity"])
    response = client.get(artifact_url(job, False), HTTP_AUTHORIZATION="Bearer test-token")
    assert response.status_code == 404
    assert JobArtifact.objects.filter(pk=artifact.pk).exists()
    assert get_artifact_store().list_keys(prefix="voxhelm/jobs/") == []


@pytest.mark.django_db
@pytest.mark.parametrize("speed", [0.25, 4.0])
def test_job_post_keeps_valid_speed_boundaries(client, settings, speed):
    configure_task_backend(settings, "django_tasks.backends.dummy.DummyBackend")
    payload = build_synthesis_payload()
    payload["speed"] = speed
    response = client.post(
        "/v1/jobs",
        data=json.dumps(payload),
        content_type="application/json",
        HTTP_AUTHORIZATION="Bearer test-token",
    )
    assert response.status_code == 201
    assert Job.objects.get(pk=response.json()["id"]).input_data["speed"] == speed


@pytest.mark.django_db
@pytest.mark.parametrize("operator", [False, True])
def test_download_denial_never_opens_storage(client, django_user_model, monkeypatch, operator):
    owner = django_user_model.objects.create_user(username="synthetic-owner", is_staff=True)
    job, _ = artifact_fixture(owner)
    if operator:
        other = django_user_model.objects.create_user(username="synthetic-other", is_staff=True)
        client.force_login(other)
    install_store(monkeypatch, lambda identity: pytest.fail("unauthorized storage access"))
    response = client.get(
        artifact_url(job, operator), HTTP_AUTHORIZATION="Bearer invalid-synthetic"
    )
    assert response.status_code in {401, 403, 404}
    assert "no-store" in response["Cache-Control"]


@pytest.mark.django_db
@pytest.mark.parametrize("operator", [False, True])
def test_midstream_error_closes_reader_and_keeps_metadata(
    client, django_user_model, monkeypatch, operator
):
    user = django_user_model.objects.create_user(username="synthetic-stream-error", is_staff=True)
    if operator:
        client.force_login(user)
    job, artifact = artifact_fixture(user)

    class FailedReader(BoundedReader):
        def read(self, size=-1):
            if self.requests:
                raise OSError("synthetic midstream failure")
            return super().read(size)

    reader = FailedReader()
    install_store(
        monkeypatch,
        lambda identity: SimpleNamespace(
            open_reader=lambda **kwargs: reader, read_bytes=unbounded_read
        ),
    )
    response = client.get(artifact_url(job, operator), HTTP_AUTHORIZATION="Bearer test-token")
    assert response.status_code == 200
    try:
        with pytest.raises(OSError, match="midstream failure"):
            list(response.streaming_content)
    finally:
        response.close()
    assert reader.closed
    assert JobArtifact.objects.filter(pk=artifact.pk).exists()


@pytest.mark.django_db
def test_header_and_close_failure_remain_a_safe_storage_response(client, monkeypatch):
    job, _ = artifact_fixture()
    attempted = []

    def fail_header():
        raise OSError("synthetic-private-marker")

    def fail_close():
        attempted.append(True)
        raise OSError("synthetic-close-private-marker")

    reader = SimpleNamespace(
        read=lambda size: b"", seek=lambda *args: None, seekable=fail_header, close=fail_close
    )
    install_store(
        monkeypatch, lambda identity: SimpleNamespace(open_reader=lambda **kwargs: reader)
    )
    response = client.get(artifact_url(job, False), HTTP_AUTHORIZATION="Bearer test-token")
    assert response.status_code == 503
    assert b"private-marker" not in response.content
    assert attempted == [True]


@pytest.mark.django_db(transaction=True)
def test_asgi_cancellation_closes_an_active_storage_read(monkeypatch):
    import threading

    job, _ = artifact_fixture()
    started, released, finished = threading.Event(), threading.Event(), threading.Event()

    class BlockedReader(BoundedReader):
        def read(self, size=-1):
            started.set()
            try:
                assert released.wait(3), "read was never closed"
                return b""
            finally:
                finished.set()

        def close(self):
            released.set()
            super().close()

    reader = BlockedReader()
    install_store(
        monkeypatch, lambda identity: SimpleNamespace(open_reader=lambda **kwargs: reader)
    )

    async def request():
        response = await AsyncClient().get(
            artifact_url(job, False), headers={"Authorization": "Bearer test-token"}
        )
        assert isinstance(response, StreamingHttpResponse)
        iterator = response.streaming_content
        consume = asyncio.create_task(anext(iterator))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            consume.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(consume, 1)
            assert reader.closed
            assert await asyncio.to_thread(finished.wait, 2)
        finally:
            released.set()
            await iterator.aclose()
            await sync_to_async(response.close)()

    asyncio.run(request())
