"""Bounded, priority-correct remote claim scans (scout round 81, slice 64)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from django.db import connection
from django.test.utils import CaptureQueriesContext
from django.utils import timezone

from jobs import remote_workers
from jobs.models import Job, Worker
from tests.test_remote_workers import claim_one, plain_capabilities, submit_remote_job

SYNTHETIC_URL = "https://media.example.com/synthetic.mp3"


@pytest.fixture(autouse=True)
def no_change_grace(monkeypatch):
    # Every job a test creates is "new" within the production grace window; tests
    # that are not about the grace itself use exact sweep timestamps.
    monkeypatch.setattr(remote_workers, "CLAIM_SCAN_CHANGE_GRACE_SECONDS", 0)


@pytest.fixture
def examined(monkeypatch):
    seen: list = []
    original = remote_workers.remote_payload_is_claimable

    def record(job):
        seen.append(job.pk)
        return original(job)

    monkeypatch.setattr(remote_workers, "remote_payload_is_claimable", record)
    return seen


def incompatible(count: int, *, priority: str = "normal") -> list[Job]:
    return Job.objects.bulk_create(
        [
            Job(
                producer="synthetic",
                job_type="transcribe",
                execution_mode="remote_pull",
                backend="mlx",
                model="unsupported",
                priority=priority,
                input_data={"kind": "url", "url": SYNTHETIC_URL},
            )
            for _ in range(count)
        ]
    )


def compatible(client, settings, *, task_ref: str, priority: str = "normal") -> Job:
    job = submit_remote_job(client, settings, task_ref=task_ref)
    if priority != "normal":
        Job.objects.filter(pk=job.pk).update(priority=priority, updated_at=timezone.now())
    return Job.objects.get(pk=job.pk)


def prepare(client, settings) -> None:
    # Enables remote workers; the seed job itself is not part of the scenario.
    submit_remote_job(client, settings, task_ref="seed").delete()
    Worker.objects.create(worker_id="atlas")


def claimed_id(response) -> str | None:
    if response.status_code == 204:
        return None
    assert response.status_code == 200, response.content
    return response.json()["job"]["id"]


def finish(job_id: str) -> None:
    Job.objects.filter(pk=job_id).update(state="succeeded", updated_at=timezone.now())


@pytest.mark.django_db
def test_scan_bounds_each_poll_and_eventually_reaches_compatible_tail(client, settings, examined):
    prepare(client, settings)
    backlog = incompatible(2000)
    tail = compatible(client, settings, task_ref="compatible-tail")
    for _ in range(25):
        examined.clear()
        job_id = claimed_id(claim_one(client))
        assert len(examined) <= remote_workers.CLAIM_SCAN_LIMIT
        if job_id is not None:
            assert job_id == str(tail.pk)
            break
    else:
        pytest.fail("compatible tail starved")
    # A finished sweep wraps to the top and finds work that became eligible there.
    finish(str(tail.pk))
    Job.objects.filter(pk=backlog[0].pk).update(backend="auto", model="auto")
    for _ in range(25):
        job_id = claimed_id(claim_one(client))
        if job_id is not None:
            assert job_id == str(backlog[0].pk)
            break
    else:
        pytest.fail("scan failed to wrap")


@pytest.mark.django_db
def test_new_high_priority_job_behind_the_cursor_wins_over_later_compatible_work(
    client, settings, examined
):
    """The parked r81 warning: a new compatible high-priority job sorting before the
    cursor (beyond any fixed head) must not wait while lower-priority compatible work
    after the cursor keeps the continuation busy."""
    prepare(client, settings)
    incompatible(250)
    # Two polls advance the cursor into the middle of the normal-priority backlog.
    for _ in range(2):
        assert claimed_id(claim_one(client)) is None
    cursor = Worker.objects.get(worker_id="atlas").claim_scan_cursor
    assert cursor and cursor["priority_rank"] == 1
    # Lower-priority compatible work lands after the cursor ...
    low = [compatible(client, settings, task_ref=f"low-{i}", priority="low") for i in range(3)]
    # ... and a high-priority burst lands before it: 150 incompatible, then one compatible.
    incompatible(150, priority="high")
    high = compatible(client, settings, task_ref="new-high", priority="high")

    claimed = []
    for _ in range(10):
        examined.clear()
        job_id = claimed_id(claim_one(client))
        assert len(examined) <= remote_workers.CLAIM_SCAN_LIMIT
        if job_id is not None:
            claimed.append(job_id)
            finish(job_id)
            if len(claimed) == 4:
                break
    assert claimed == [str(high.pk), *(str(job.pk) for job in low)]


@pytest.mark.django_db
def test_bounded_claims_preserve_compatible_priority_and_fifo(client, settings):
    prepare(client, settings)
    high = [
        compatible(client, settings, task_ref=f"ordered-{i}", priority="high") for i in range(12)
    ]
    low = compatible(client, settings, task_ref="low-priority", priority="low")
    for expected in [*high, low]:
        assert claimed_id(claim_one(client)) == str(expected.pk)
        finish(str(expected.pk))


@pytest.mark.django_db
def test_expired_lease_before_the_cursor_restarts_the_sweep(client, settings):
    prepare(client, settings)
    early = compatible(client, settings, task_ref="early-high", priority="high")
    Job.objects.filter(pk=early.pk).update(
        state="running",
        assigned_worker_id="other",
        lease_expires_at=timezone.now() + timedelta(hours=1),
        attempt_count=1,
    )
    incompatible(250)
    for _ in range(2):
        assert claimed_id(claim_one(client)) is None
    assert Worker.objects.get(worker_id="atlas").claim_scan_cursor["priority_rank"] == 1
    # The other worker's lease lapses without a row write: the job is claimable again
    # and sorts before the cursor, so the very next poll restarts and claims it.
    Job.objects.filter(pk=early.pk).update(lease_expires_at=timezone.now())
    assert claimed_id(claim_one(client)) == str(early.pk)


@pytest.mark.django_db
def test_unleased_eligible_job_is_not_skipped_by_the_cursor(client, settings):
    prepare(client, settings)
    incompatible(150)
    target = compatible(client, settings, task_ref="target")
    blocker = compatible(client, settings, task_ref="blocker", priority="high")
    Job.objects.filter(pk=blocker.pk).update(
        state="running",
        assigned_worker_id="atlas",
        lease_expires_at=timezone.now() + timedelta(hours=1),
    )
    # Capacity is full: polls find ``target`` but cannot lease it.
    for _ in range(4):
        assert claimed_id(claim_one(client)) is None
    finish(str(blocker.pk))
    assert claimed_id(claim_one(client)) == str(target.pk)


@pytest.mark.django_db
def test_changed_capabilities_restart_the_sweep(client, settings):
    prepare(client, settings)
    backlog = incompatible(250)
    for _ in range(2):
        assert claimed_id(claim_one(client)) is None
    assert Worker.objects.get(worker_id="atlas").claim_scan_cursor["priority_rank"] == 1
    capabilities = plain_capabilities()
    capabilities["backends"] = ["whispercpp", "mlx"]
    capabilities["models"] = ["ggml-large-v3.bin", "unsupported"]
    assert claimed_id(claim_one(client, capabilities=capabilities)) == str(backlog[0].pk)


@pytest.mark.django_db
@pytest.mark.parametrize(
    "stored",
    [
        {"v": 99},
        {"v": 1, "priority_rank": "x"},
        {"v": 1, "priority_rank": 1, "created_at": "nope", "id": "1", "sweep_started_at": ""},
        ["not", "a", "dict"],
    ],
    ids=["version", "rank", "fields", "type"],
)
def test_malformed_cursor_scans_from_the_top(client, settings, stored):
    prepare(client, settings)
    first = compatible(client, settings, task_ref="first")
    Worker.objects.filter(worker_id="atlas").update(claim_scan_cursor=stored)
    assert claimed_id(claim_one(client)) == str(first.pk)


@pytest.mark.django_db
def test_old_sweep_keeps_its_position_and_unrecorded_changes_wait_one_sweep(
    client, settings, examined
):
    prepare(client, settings)
    backlog = incompatible(1000)
    for _ in range(2):
        assert claimed_id(claim_one(client)) is None
    # A sweep has no age limit: hours later it still continues where it stopped.
    cursor = dict(Worker.objects.get(worker_id="atlas").claim_scan_cursor)
    cursor["sweep_started_at"] = (timezone.now() - timedelta(hours=6)).isoformat()
    Worker.objects.filter(worker_id="atlas").update(claim_scan_cursor=cursor)
    Job.objects.update(updated_at=timezone.now() - timedelta(hours=7))
    # Compatibility changes without a row timestamp (as a server-side configuration
    # change would): the running sweep cannot see it ...
    Job.objects.filter(pk=backlog[0].pk).update(backend="auto", model="auto")
    examined.clear()
    assert claimed_id(claim_one(client)) is None
    assert backlog[0].pk not in examined
    assert examined[0] == backlog[200].pk
    # ... but finds it once the sweep has reached the end and starts over.
    responses: list[str | None] = []
    while len(responses) < 12:
        responses.append(claimed_id(claim_one(client)))
        if responses[-1] is not None:
            break
    assert responses[-1] == str(backlog[0].pk)
    assert len(responses) <= 9


@pytest.mark.django_db
def test_cursor_in_the_future_scans_from_the_top(client, settings):
    prepare(client, settings)
    backlog = incompatible(250)
    for _ in range(2):
        assert claimed_id(claim_one(client)) is None
    cursor = dict(Worker.objects.get(worker_id="atlas").claim_scan_cursor)
    cursor["sweep_started_at"] = (timezone.now() + timedelta(hours=1)).isoformat()
    Worker.objects.filter(worker_id="atlas").update(claim_scan_cursor=cursor)
    Job.objects.filter(pk=backlog[0].pk).update(backend="auto", model="auto")
    assert claimed_id(claim_one(client)) == str(backlog[0].pk)


@pytest.mark.django_db
def test_idle_polls_do_not_rewrite_the_cursor(client, settings):
    prepare(client, settings)
    assert claimed_id(claim_one(client)) is None
    with CaptureQueriesContext(connection) as queries:
        assert claimed_id(claim_one(client)) is None
    assert not [
        q["sql"] for q in queries if "claim_scan_cursor" in q["sql"] and "UPDATE" in q["sql"]
    ]


@pytest.mark.django_db
def test_changes_inside_the_grace_window_restart_the_sweep(client, settings, monkeypatch):
    monkeypatch.setattr(remote_workers, "CLAIM_SCAN_CHANGE_GRACE_SECONDS", 5)
    prepare(client, settings)
    backlog = incompatible(250)
    Job.objects.update(updated_at=timezone.now() - timedelta(hours=1))
    for _ in range(2):
        assert claimed_id(claim_one(client)) is None
    cursor = Worker.objects.get(worker_id="atlas").claim_scan_cursor
    # A requeue whose transaction sampled its clock just before the sweep started.
    started = datetime.fromisoformat(cursor["sweep_started_at"])
    Job.objects.filter(pk=backlog[0].pk).update(
        backend="auto", model="auto", updated_at=started + timedelta(seconds=1)
    )
    assert claimed_id(claim_one(client)) == str(backlog[0].pk)


@pytest.mark.django_db
def test_lost_claim_races_are_bounded_per_poll(client, settings, monkeypatch):
    prepare(client, settings)
    for i in range(3):
        compatible(client, settings, task_ref=f"race-{i}")
    calls = []
    original = remote_workers.remote_claim_candidates

    def losing(**kwargs):
        candidates = original(**kwargs)
        calls.append(len(candidates))
        # Another worker wins every candidate between the scan and the claim.
        for candidate in candidates:
            Job.objects.filter(pk=candidate.pk).update(attempt_count=candidate.attempt_count + 1)
        return candidates

    monkeypatch.setattr(remote_workers, "remote_claim_candidates", losing)
    assert claimed_id(claim_one(client)) is None
    assert len(calls) == remote_workers.CLAIM_SCAN_ROUNDS
