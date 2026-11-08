"""Bound fleet query count and backlog hydration without starving the tail."""

from datetime import timedelta

import pytest
from django.db import connection
from django.test.utils import CaptureQueriesContext
from django.utils import timezone

from jobs import remote_workers
from jobs.models import Job, Worker
from tests.test_remote_workers import _make_worker, claim_one, plain_capabilities, submit_remote_job


@pytest.mark.django_db
@pytest.mark.parametrize("fleet_size", [1, 40])
def test_fair_share_groups_fleet_load_in_two_queries(settings, fleet_size):
    _make_worker("atlas")
    for i in range(fleet_size):
        _make_worker(f"peer-{i}")
    candidate = Job(job_type="transcribe")
    with CaptureQueriesContext(connection) as queries:
        deferred = remote_workers.worker_should_defer_for_fairness(
            worker_id="atlas", now=timezone.now(), candidates=[candidate]
        )
    assert deferred is False  # all workers have zero recent load
    assert len(queries) == 2


@pytest.mark.django_db
@pytest.mark.parametrize("busy", [False, True])
def test_grouped_fairness_matches_independent_boundary_oracle(settings, busy):
    now = timezone.now()
    window = now - timedelta(seconds=settings.VOXHELM_REMOTE_WORKER_BALANCE_WINDOW_SECONDS)
    fresh = now - timedelta(seconds=settings.VOXHELM_REMOTE_WORKER_BALANCE_PEER_FRESH_SECONDS)
    Worker.objects.bulk_create(
        [
            Worker(worker_id="atlas"),
            Worker(worker_id="peer", capabilities=plain_capabilities(), last_seen_at=fresh),
            Worker(
                worker_id="stale",
                capabilities=plain_capabilities(),
                last_seen_at=fresh - timedelta(microseconds=1),
            ),
        ]
    )
    rows = [
        ("atlas", window, "succeeded", None),
        ("atlas", window - timedelta(microseconds=1), "succeeded", None),
        (
            "peer",
            window - timedelta(seconds=1),
            "running",
            now + timedelta(microseconds=1) if busy else now,
        ),
    ]
    Job.objects.bulk_create(
        [
            Job(
                producer="synthetic",
                job_type="transcribe",
                execution_mode="remote_pull",
                input_data={},
                assigned_worker_id=worker,
                started_at=start,
                state=state,
                lease_expires_at=expiry,
            )
            for worker, start, state, expiry in rows
        ]
    )
    loads = {
        worker: sum(w == worker and start >= window for w, start, _, _ in rows)
        for worker in ("atlas", "peer")
    }
    active = sum(
        w == "peer" and state == "running" and expiry > now
        for w, _, state, expiry in rows
        if expiry is not None
    )
    expected = active < 1 and loads["peer"] < loads["atlas"]
    assert (
        remote_workers.worker_should_defer_for_fairness(
            worker_id="atlas", now=now, candidates=[Job(job_type="transcribe")]
        )
        == expected
    )


@pytest.mark.django_db
def test_claim_scan_bounds_each_poll_and_eventually_reaches_compatible_tail(
    client, settings, monkeypatch
):
    seed = submit_remote_job(client, settings)
    seed.delete()
    Worker.objects.create(worker_id="atlas")
    jobs = [
        Job(
            producer="synthetic",
            job_type="transcribe",
            execution_mode="remote_pull",
            backend="mlx",
            model="unsupported",
            input_data={"kind": "url", "url": "https://media.example.com/synthetic.mp3"},
        )
        for _ in range(2000)
    ]
    Job.objects.bulk_create(jobs)
    tail = submit_remote_job(client, settings, task_ref="compatible-tail")
    examined = []
    original = remote_workers.remote_payload_is_claimable

    def record(job):
        examined.append(job.pk)
        return original(job)

    monkeypatch.setattr(remote_workers, "remote_payload_is_claimable", record)
    for _ in range(42):
        examined.clear()
        response = claim_one(client)
        assert len(examined) <= 100
        if response.status_code == 200:
            assert response.json()["job"]["id"] == str(tail.id)
            break
        assert response.status_code == 204
    else:
        pytest.fail("compatible tail starved")
    # Completing a sweep wraps to the front and discovers newly eligible work.
    Job.objects.filter(pk=jobs[0].pk).update(backend="auto", model="auto")
    Job.objects.filter(pk=tail.pk).update(state="succeeded")
    for _ in range(42):
        response = claim_one(client)
        if response.status_code == 200:
            assert response.json()["job"]["id"] == str(jobs[0].pk)
            break
    else:
        pytest.fail("scan failed to wrap")


@pytest.mark.django_db
def test_bounded_claims_preserve_compatible_priority_and_fifo(client, settings):
    jobs = [submit_remote_job(client, settings, task_ref=f"ordered-{i}") for i in range(12)]
    for job in jobs:
        Job.objects.filter(pk=job.pk).update(priority="high")
    low = submit_remote_job(client, settings, task_ref="low-priority")
    for expected in [*jobs, low]:
        response = claim_one(client)
        assert response.status_code == 200
        assert response.json()["job"]["id"] == str(expected.pk)
        Job.objects.filter(pk=expected.pk).update(state="succeeded")


@pytest.mark.django_db
def test_bounded_scan_checks_new_high_priority_head_each_poll(client, settings):
    seed = submit_remote_job(client, settings)
    seed.delete()
    Job.objects.bulk_create(
        [
            Job(
                producer="synthetic",
                job_type="transcribe",
                execution_mode="remote_pull",
                backend="mlx",
                model="unsupported",
                input_data={"kind": "url", "url": "https://media.example.com/synthetic.mp3"},
            )
            for _ in range(200)
        ]
    )
    assert claim_one(client).status_code == 204
    high = submit_remote_job(client, settings, task_ref="new-high")
    Job.objects.filter(pk=high.pk).update(priority="high")
    response = claim_one(client)
    assert response.status_code == 200
    assert response.json()["job"]["id"] == str(high.pk)
