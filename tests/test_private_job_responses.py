from __future__ import annotations

from uuid import uuid4

import pytest
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import Client, RequestFactory
from django.utils.cache import get_cache_key

from jobs.artifacts import get_artifact_store
from jobs.models import Job, JobArtifact


@pytest.fixture
def cached_responses(settings):
    settings.MIDDLEWARE = [
        "django.middleware.cache.UpdateCacheMiddleware",
        *settings.MIDDLEWARE,
        "django.middleware.cache.FetchFromCacheMiddleware",
    ]
    settings.CACHE_MIDDLEWARE_SECONDS = 600
    # Prove this arrangement really stores ordinary cacheable responses.
    probe = Client()
    assert probe.get("/v1/health").status_code == 200
    key = get_cache_key(RequestFactory().get("/v1/health", HTTP_HOST="testserver"))
    assert key is not None and cache.get(key) is not None
    cache.clear()
    return Client()


@pytest.fixture
def private_job():
    user = get_user_model().objects.create_user(username="synthetic-operator", is_staff=True)
    job = Job.objects.create(
        producer="archive",
        operator=user,
        job_type="transcribe",
        input_data={},
        state="succeeded",
        result_text="private synthetic transcript",
    )
    key = f"voxhelm/jobs/{job.id}/transcript.txt"
    get_artifact_store().put_bytes(
        key=key, data=b"private synthetic transcript", content_type="text/plain"
    )
    artifact = JobArtifact.objects.create(
        job=job,
        name="transcript.txt",
        kind="transcript_text",
        format="text",
        storage_backend="filesystem",
        storage_key=key,
        content_type="text/plain",
        exposed=True,
    )
    return user, job, artifact


def assert_private(response, vary):
    policy = response.headers["Cache-Control"]
    assert "private" in policy and "no-store" in policy
    assert vary.lower() in response.headers["Vary"].lower()


@pytest.mark.django_db
@pytest.mark.parametrize("download", [False, True])
@pytest.mark.parametrize("revocation", ["token", "ownership"])
def test_same_bearer_cannot_reuse_cached_private_response(
    cached_responses, private_job, settings, download, revocation
):
    _, job, _ = private_job
    url = f"/v1/jobs/{job.id}" + ("/artifacts/transcript.txt" if download else "")
    headers = {"HTTP_AUTHORIZATION": "Bearer test-token"}
    first = cached_responses.get(url, **headers)
    assert first.status_code == 200
    assert b"private synthetic transcript" in first.content
    if revocation == "token":
        settings.VOXHELM_BEARER_TOKENS = {}
    else:
        Job.objects.filter(pk=job.pk).update(producer="other")
    denied = cached_responses.get(url, **headers)
    assert denied.status_code == (401 if revocation == "token" else 404)
    assert b"private synthetic transcript" not in denied.content
    assert_private(first, "Authorization")
    assert_private(denied, "Authorization")


@pytest.mark.django_db
@pytest.mark.parametrize("revocation", ["staff", "ownership", "exposed", "session"])
def test_operator_download_rechecks_access_with_same_cookie(
    cached_responses, private_job, revocation
):
    user, job, artifact = private_job
    cached_responses.force_login(user)
    url = f"/transcripts/{job.id}/artifacts/text"
    first = cached_responses.get(url)
    assert first.status_code == 200
    if revocation == "staff":
        get_user_model().objects.filter(pk=user.pk).update(is_staff=False)
    elif revocation == "ownership":
        Job.objects.filter(pk=job.pk).update(operator=None)
    elif revocation == "exposed":
        JobArtifact.objects.filter(pk=artifact.pk).update(exposed=False)
    else:
        from django.contrib.sessions.models import Session

        Session.objects.all().delete()
    denied = cached_responses.get(url)
    assert (
        denied.status_code
        == {"staff": 403, "ownership": 404, "exposed": 404, "session": 302}[revocation]
    )
    assert b"private synthetic transcript" not in denied.content
    assert_private(first, "Cookie")
    assert_private(denied, "Cookie")


@pytest.mark.django_db
@pytest.mark.parametrize(
    "path", ["/v1/jobs", "/v1/uploads", "/v1/jobs/{id}", "/v1/jobs/{id}/artifacts/transcript.txt"]
)
def test_private_api_errors_have_policy(client, path):
    response = client.get(path.format(id=uuid4()))
    assert response.status_code in {401, 405}
    assert_private(response, "Authorization")
