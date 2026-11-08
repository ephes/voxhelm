"""Grouped fair-share query cost and load-window boundary equivalence."""

from datetime import timedelta

import pytest
from django.db import connection
from django.test.utils import CaptureQueriesContext
from django.utils import timezone

from jobs import remote_workers
from jobs.models import Job, Worker
from tests.test_remote_workers import _make_worker, plain_capabilities


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
@pytest.mark.parametrize("spare_load", [1, 2])
def test_grouped_counts_stay_attached_to_each_peer(settings, spare_load):
    now = timezone.now()
    for worker_id in ("atlas", "busy", "spare"):
        _make_worker(worker_id)
    window = now - timedelta(seconds=settings.VOXHELM_REMOTE_WORKER_BALANCE_WINDOW_SECONDS)
    rows = [
        Job(
            producer="synthetic",
            job_type="transcribe",
            execution_mode="remote_pull",
            input_data={},
            assigned_worker_id=worker_id,
            started_at=now,
            state="succeeded",
        )
        for worker_id in ["atlas", "atlas", *(["spare"] * spare_load)]
    ]
    rows.append(
        Job(
            producer="synthetic",
            job_type="transcribe",
            execution_mode="remote_pull",
            input_data={},
            assigned_worker_id="busy",
            state="running",
            started_at=window - timedelta(seconds=1),
            lease_expires_at=now + timedelta(minutes=5),
        )
    )
    Job.objects.bulk_create(rows)
    assert remote_workers.worker_should_defer_for_fairness(
        worker_id="atlas", now=now, candidates=[Job(job_type="transcribe")]
    ) == (spare_load < 2)
