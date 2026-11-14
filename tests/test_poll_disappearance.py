"""Owner-scoped HTTP outcome when a job disappears after lookup."""

import pytest
from django.db import DatabaseError

from jobs import views
from jobs.models import Job


@pytest.mark.django_db
@pytest.mark.parametrize("disappear", [True, False])
def test_job_poll_disappearance_is_private_404(client, monkeypatch, disappear):
    job = Job.objects.create(
        producer="archive", job_type="transcribe", state="succeeded", input_data={}
    )
    serialize = views.serialize_job

    def after_lookup(owned):
        assert owned.pk == job.pk
        if disappear:
            Job.objects.filter(pk=owned.pk).delete()
        return serialize(owned)

    monkeypatch.setattr(views, "serialize_job", after_lookup)
    response = client.get(f"/v1/jobs/{job.pk}", HTTP_AUTHORIZATION="Bearer test-token")
    assert response.status_code == (404 if disappear else 200)
    if disappear:
        assert response.json() == {
            "error": {"message": "Job not found.", "type": "invalid_request_error"}
        }
        assert not Job.objects.exists()
    else:
        assert response.json()["id"] == str(job.pk)


@pytest.mark.django_db
def test_poll_keeps_database_errors_distinct(client, monkeypatch):
    job = Job.objects.create(
        producer="archive", job_type="transcribe", state="succeeded", input_data={}
    )

    def fail(owned):
        raise DatabaseError("synthetic storage failure")

    monkeypatch.setattr(views, "serialize_job", fail)
    with pytest.raises(DatabaseError):
        client.get(f"/v1/jobs/{job.pk}", HTTP_AUTHORIZATION="Bearer test-token")


@pytest.mark.django_db
def test_initial_missing_and_foreign_job_poll_share_private_response(client):
    from uuid import uuid4

    foreign = Job.objects.create(producer="other", job_type="transcribe", input_data={})
    for job_id in (foreign.pk, uuid4()):
        response = client.get(f"/v1/jobs/{job_id}", HTTP_AUTHORIZATION="Bearer test-token")
        assert response.status_code == 404
        assert response.json() == {
            "error": {"message": "Job not found.", "type": "invalid_request_error"}
        }
