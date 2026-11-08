"""Synthetic admission regressions through the worker claim entry point."""

from datetime import timedelta

import pytest
from django.utils import timezone

from jobs import remote_workers
from jobs.models import Job, Worker
from tests.test_remote_workers import (
    _make_load_job,
    _make_worker,
    claim_one,
    incompatible_stt_capabilities,
    limited_text_capabilities,
    plain_capabilities,
    submit_remote_job,
)


@pytest.mark.django_db
@pytest.mark.parametrize("peer_kind", ["backend", "format", "compatible", "stale"])
def test_claim_fairness_only_yields_to_peer_that_can_run_candidate(client, settings, peer_kind):
    job = submit_remote_job(client, settings)
    _make_worker("atlas")
    peer = _make_worker("zephyr", seen_seconds_ago=600 if peer_kind == "stale" else 0)
    peer.capabilities = {
        "backend": incompatible_stt_capabilities(),
        "format": limited_text_capabilities(),
    }.get(peer_kind, plain_capabilities())
    peer.save()
    _make_load_job("atlas")
    response = claim_one(client)
    if peer_kind == "compatible":
        assert response.status_code == 204
    else:
        assert response.status_code == 200
        assert response.json()["job"]["id"] == str(job.id)


@pytest.mark.django_db
@pytest.mark.parametrize("expiring_work", [False, True])
def test_claim_admission_clock_is_sampled_after_worker_lock(
    client, settings, monkeypatch, expiring_work
):
    job = submit_remote_job(client, settings)
    Worker.objects.create(worker_id="atlas")
    settings.VOXHELM_REMOTE_WORKER_LEASE_SECONDS = 2
    before = timezone.now()
    admitted = before + timedelta(seconds=5)
    clock = [before]
    monkeypatch.setattr(remote_workers.timezone, "now", lambda: clock[0])
    if expiring_work:
        old = _make_load_job("atlas", state=Job.State.RUNNING)
        old.lease_expires_at = before + timedelta(seconds=1)
        old.max_attempts = old.attempt_count
        old.save()
    original = remote_workers.lock_worker_for_claim

    def delayed_lock(**kwargs):
        original(**kwargs)
        clock[0] = admitted

    monkeypatch.setattr(remote_workers, "lock_worker_for_claim", delayed_lock)
    response = claim_one(client)
    assert response.status_code == 200
    assert response.json()["job"] is not None
    job.refresh_from_db()
    assert job.lease_expires_at == admitted + timedelta(seconds=2)
    assert job.last_worker_heartbeat_at == admitted


@pytest.mark.django_db
@pytest.mark.parametrize("malformed", [["legacy"], {"diarization": "invalid"}])
def test_claim_ignores_malformed_peer_capabilities(client, settings, malformed):
    from tests.test_remote_workers import known_speaker_capabilities

    job = submit_remote_job(client, settings, diarization={"enabled": True})
    _make_worker("atlas")
    peer = _make_worker("zephyr")
    peer.capabilities = (
        {**known_speaker_capabilities(), **malformed} if isinstance(malformed, dict) else malformed
    )
    peer.save()
    _make_load_job("atlas")
    response = claim_one(client, capabilities=known_speaker_capabilities())
    assert response.status_code == 200
    assert response.json()["job"]["id"] == str(job.id)
