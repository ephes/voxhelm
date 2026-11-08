"""Spawn-only, synthetic SQLite scenarios used by the remote ownership tests."""

from __future__ import annotations

import multiprocessing
import sys
from pathlib import Path
from unittest.mock import patch


class SyntheticArtifactStore:
    """Portable fake object store; only synthetic test-owned keys reach it."""

    backend_name = "filesystem"

    def __init__(self, *, root):
        self.root = Path(root)

    def put_bytes(self, *, key, data, content_type):
        from jobs.artifacts import StoredArtifact

        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return StoredArtifact(backend=self.backend_name, key=key, size_bytes=len(data))

    def delete(self, *, key):
        (self.root / key).unlink(missing_ok=True)

    def stat(self, *, key):
        from jobs.artifacts import StoredArtifact

        return StoredArtifact(
            backend=self.backend_name, key=key, size_bytes=(self.root / key).stat().st_size
        )

    def list_keys(self, *, prefix):
        return sorted(
            path.relative_to(self.root).as_posix()
            for path in (self.root / prefix).rglob("*")
            if path.is_file()
        )


def initialize(root):
    import django
    from django.conf import settings

    root = Path(root)
    settings.DATABASES["default"]["NAME"] = root / "contracts.sqlite3"
    settings.DATABASES["default"]["OPTIONS"] = {"timeout": 10}
    settings.VOXHELM_ARTIFACT_BACKEND = "filesystem"
    settings.VOXHELM_ARTIFACT_ROOT = root / "artifacts"
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}
    settings.VOXHELM_REMOTE_WORKER_BALANCE_ENABLED = False
    django.setup()
    from jobs import artifacts

    artifacts.get_artifact_store.cache_clear()
    # This fixture owns the subprocess lifetime; no production file I/O is used.
    patch.object(artifacts, "FilesystemArtifactStore", SyntheticArtifactStore).start()


def create_database(root):
    initialize(root)
    from django.core.management import call_command

    call_command("migrate", verbosity=0)


def make_job(**kwargs):
    from jobs.models import Job

    return Job.objects.create(
        producer="synthetic",
        job_type="transcribe",
        execution_mode="remote_pull",
        backend="whispercpp",
        model="ggml-large-v3.bin",
        input_data={"kind": "url", "url": "https://media.example.com/synthetic.mp3"},
        output_data={"formats": ["text", "json"]},
        **kwargs,
    )


def claim_child(
    root, role, scans_ready, first_read, second_attempt, second_read, release, results, mutation
):
    initialize(root)
    from django.db import connections

    from jobs import remote_workers as rw
    from tests.test_remote_workers import plain_capabilities

    lock = rw.lock_worker_for_claim
    capacity = rw.worker_has_claim_capacity
    scan = rw.remote_claim_candidates

    def coordinated_scan(**kwargs):
        candidates = scan(**kwargs)
        scans_ready.wait(15)
        if role == "second":
            assert first_read.wait(15), "first claim did not count capacity"
        return candidates

    def coordinated_lock(**kwargs):
        if role == "second":
            second_attempt.set()
        if mutation != "no_worker_lock":
            lock(**kwargs)

    def coordinated_capacity(**kwargs):
        admitted = capacity(**kwargs)
        if role == "first":
            first_read.set()
            assert release.wait(15), "capacity release timed out"
        else:
            second_read.set()
        return admitted

    rw.remote_claim_candidates = coordinated_scan
    rw.lock_worker_for_claim = coordinated_lock
    rw.worker_has_claim_capacity = coordinated_capacity
    capabilities = plain_capabilities()
    capabilities.update(
        profiles=["ad-analysis-v1"], backends=["parakeet"], models=["parakeet-tdt-0.6b-v3"]
    )
    try:
        claim = rw.claim_remote_work(worker_id="synthetic", payload={"capabilities": capabilities})
        results.put(
            {
                "role": role,
                "job": str(claim.job.pk) if claim else None,
                "token": claim.lease_token if claim else None,
            }
        )
    except BaseException as exc:
        results.put({"role": role, "error": repr(exc)})
    finally:
        connections.close_all()


def stop_children(children):
    errors = []
    for child in children:
        child.join(15)
        if child.is_alive():
            child.kill()
            child.join(5)
            errors.append("child did not finish")
        elif child.exitcode != 0:
            errors.append(f"child exited {child.exitcode}")
    assert not errors, errors


def capacity_scenario(root, mutation):
    create_database(root)
    from django.db import connections
    from django.utils import timezone

    from jobs.models import Job, RemoteAttemptPrefix, Worker
    from jobs.remote_workers import claim_remote_work, fail_remote_work
    from tests.test_remote_workers import plain_capabilities

    Worker.objects.create(worker_id="synthetic", concurrency=1)
    for _ in range(2):
        make_job(profile="ad-analysis-v1")
    ctx = multiprocessing.get_context("spawn")
    first_read, second_attempt, second_read, release = [ctx.Event() for _ in range(4)]
    results = ctx.Queue()
    args = (ctx.Barrier(2), first_read, second_attempt, second_read, release, results, mutation)
    children = [
        ctx.Process(target=claim_child, args=(root, role, *args)) for role in ("first", "second")
    ]
    try:
        children[0].start()
        children[1].start()
        assert first_read.wait(15), "first capacity count not reached"
        assert second_attempt.wait(15), "second claim did not contend"
        # With serialization the second count cannot finish until release. With
        # the counterfactual lock removed it reads the same empty capacity.
        second_read.wait(1)
    finally:
        release.set()
        stop_children([p for p in children if p.pid is not None])
    outcomes = [results.get(timeout=5) for _ in children]
    assert not any("error" in result for result in outcomes), outcomes
    winners = [result for result in outcomes if result["job"]]
    assert len(winners) == 1, outcomes
    connections.close_all()
    active = Job.objects.filter(state="running", lease_expires_at__gt=timezone.now())
    assert active.count() == 1
    assert sorted(Job.objects.values_list("attempt_count", flat=True)) == [0, 1]
    winner = winners[0]
    records = list(RemoteAttemptPrefix.objects.all())
    assert len(records) == 1
    assert str(records[0].job_id) == winner["job"] and records[0].attempt == 1
    assert records[0].prefix == active.get().leased_artifact_prefix
    fail_remote_work(
        worker_id="synthetic",
        job_id=active.get().pk,
        payload={"lease_token": winner["token"], "retryable": False, "error_detail": "synthetic"},
    )
    capabilities = plain_capabilities()
    capabilities.update(
        profiles=["ad-analysis-v1"], backends=["parakeet"], models=["parakeet-tdt-0.6b-v3"]
    )
    next_claim = claim_remote_work(worker_id="synthetic", payload={"capabilities": capabilities})
    assert next_claim and str(next_claim.job.pk) != winner["job"]
    assert next_claim.job.attempt_count == 1
    assert RemoteAttemptPrefix.objects.count() == 2
    assert RemoteAttemptPrefix.objects.values("job_id", "attempt", "prefix").distinct().count() == 2


def settlement_child(
    root, role, action, job_id, payload, locked, attempted, done, release, results, mutation
):
    initialize(root)
    from django.db import connections

    from jobs import remote_workers as rw
    from transcriptions.errors import ApiError

    lock = rw.lock_job_for_settlement

    def coordinated_lock(**kwargs):
        if role == "second":
            attempted.set()
        if mutation != "no_settlement_lock":
            lock(**kwargs)
        if role == "first":
            locked.set()
            assert release.wait(15), "settlement release timed out"

    rw.lock_job_for_settlement = coordinated_lock
    try:
        result = (rw.complete_remote_work if action == "complete" else rw.fail_remote_work)(
            worker_id="synthetic", job_id=job_id, payload=payload
        )
        results.put({"role": role, "state": result["state"]})
    except ApiError as exc:
        results.put({"role": role, "conflict": exc.status})
    except BaseException as exc:
        results.put({"role": role, "error": repr(exc)})
    finally:
        done.set()
        connections.close_all()


def settlement_scenario(root, mutation, first_action, retryable):
    create_database(root)
    from django.conf import settings
    from django.db import connections

    from jobs.artifacts import current_artifact_store_identity, get_artifact_store
    from jobs.models import JobArtifact, PendingArtifactDeletion, Worker
    from jobs.remote_workers import claim_remote_work, complete_remote_work, fail_remote_work
    from tests.test_remote_workers import (
        plain_capabilities,
        store_manifest_artifacts,
        transcript_manifest,
    )
    from transcriptions.errors import ApiError

    Worker.objects.create(worker_id="synthetic")
    job = make_job()
    claim = claim_remote_work(worker_id="synthetic", payload={"capabilities": plain_capabilities()})
    assert claim
    manifest = transcript_manifest(str(job.pk), 1)
    store_manifest_artifacts(settings, manifest)
    old_key = f"voxhelm/jobs/{job.pk}/input.mp3"
    get_artifact_store().put_bytes(key=old_key, data=b"old", content_type="audio/mpeg")
    old = JobArtifact.objects.create(
        job=job,
        name="input.mp3",
        kind="source",
        format="source",
        storage_backend="filesystem",
        storage_key=old_key,
        storage_identity=current_artifact_store_identity(),
        size_bytes=3,
        content_type="audio/mpeg",
        exposed=False,
    )
    completion = {
        "lease_token": claim.lease_token,
        "result_text": "synthetic transcript",
        "result_metadata": {"backend": "whispercpp"},
        "artifacts": manifest,
    }
    failure = {
        "lease_token": claim.lease_token,
        "retryable": retryable,
        "error_detail": "synthetic",
    }
    ctx = multiprocessing.get_context("spawn")
    locked, attempted, release = [ctx.Event() for _ in range(3)]
    done = [ctx.Event(), ctx.Event()]
    results = ctx.Queue()
    actions = [first_action, "fail" if first_action == "complete" else "complete"]
    children = [
        ctx.Process(
            target=settlement_child,
            args=(
                root,
                role,
                action,
                job.pk,
                completion if action == "complete" else failure,
                locked,
                attempted,
                done[i],
                release,
                results,
                mutation,
            ),
        )
        for i, (role, action) in enumerate(zip(("first", "second"), actions, strict=True))
    ]
    try:
        children[0].start()
        assert locked.wait(15), "first settlement not locked"
        children[1].start()
        assert attempted.wait(15), "second settlement did not contend"
        done[1].wait(1)
    finally:
        release.set()
        stop_children([p for p in children if p.pid is not None])
    outcomes = {r["role"]: r for r in [results.get(timeout=5) for _ in children]}
    assert "error" not in outcomes["first"] and "error" not in outcomes["second"], outcomes
    expected = "succeeded" if first_action == "complete" else ("queued" if retryable else "failed")
    assert outcomes["first"].get("state") == expected, outcomes
    assert outcomes["second"].get("conflict") == 409, outcomes
    connections.close_all()
    job.refresh_from_db()
    assert job.state == expected
    assert PendingArtifactDeletion.objects.count() == 0
    if first_action == "complete":
        assert job.result_text == completion["result_text"]
        assert job.result_metadata["backend"] == "whispercpp"
        assert not JobArtifact.objects.filter(pk=old.pk).exists()
        assert not (settings.VOXHELM_ARTIFACT_ROOT / old_key).exists()
        expected_manifest = [
            {**item, "storage_identity": current_artifact_store_identity()} for item in manifest
        ]
        actual_manifest = list(job.artifacts.values(*expected_manifest[0].keys()))
        assert sorted(actual_manifest, key=lambda item: str(item["name"])) == sorted(
            expected_manifest, key=lambda item: str(item["name"])
        )
        assert job.finished_at is not None
        assert job.result_metadata["worker_id"] == "synthetic"
        assert job.result_metadata["attempt"] == 1
    else:
        assert job.result_text == "" and job.result_metadata == {}
        assert list(job.artifacts.values_list("pk", flat=True)) == [old.pk]
        assert (settings.VOXHELM_ARTIFACT_ROOT / old_key).read_bytes() == b"old"
        if retryable:
            replacement = claim_remote_work(
                worker_id="synthetic", payload={"capabilities": plain_capabilities()}
            )
            assert replacement and replacement.job.attempt_count == 2
            before = list(job.artifacts.values())
            ownership = (
                replacement.job.lease_token_hash,
                replacement.job.lease_expires_at,
                replacement.job.leased_artifact_prefix,
            )
            for settle, payload in (
                (complete_remote_work, completion),
                (fail_remote_work, failure),
            ):
                try:
                    settle(worker_id="synthetic", job_id=job.pk, payload=payload)
                except ApiError as exc:
                    assert exc.status == 409
                else:
                    raise AssertionError("old lease settled replacement")
            job.refresh_from_db()
            assert job.state == "running" and job.attempt_count == 2
            assert (
                job.lease_token_hash,
                job.lease_expires_at,
                job.leased_artifact_prefix,
            ) == ownership
            assert list(job.artifacts.values()) == before
            assert PendingArtifactDeletion.objects.count() == 0


def clock_child(root, attempted, results):
    initialize(root)
    from django.conf import settings
    from django.db import connections

    from jobs import remote_workers as rw
    from tests.test_remote_workers import plain_capabilities

    settings.VOXHELM_REMOTE_WORKER_LEASE_SECONDS = 2
    scan = rw.remote_claim_candidates

    def record_scan(**kwargs):
        attempted.set()  # claim's initial clock was already sampled
        return scan(**kwargs)

    rw.remote_claim_candidates = record_scan
    try:
        claim = rw.claim_remote_work(
            worker_id="synthetic", payload={"capabilities": plain_capabilities()}
        )
        assert claim
        results.put((claim.job.last_worker_heartbeat_at, claim.job.lease_expires_at))
    finally:
        connections.close_all()


def clock_scenario(root):
    create_database(root)
    from datetime import timedelta

    from django.db import transaction
    from django.utils import timezone

    from jobs.models import Worker

    Worker.objects.create(worker_id="synthetic")
    make_job()
    ctx = multiprocessing.get_context("spawn")
    attempted, pause = ctx.Event(), ctx.Event()
    results = ctx.Queue()
    child = ctx.Process(target=clock_child, args=(root, attempted, results))
    try:
        with transaction.atomic():
            Worker.objects.filter(worker_id="synthetic").update(updated_at=timezone.now())
            child.start()
            assert attempted.wait(15), "claim scan not reached"
            pause.wait(2.1)  # hold a real writer reservation beyond the two-second lease
            released_at = timezone.now()
    finally:
        stop_children([child] if child.pid is not None else [])
    issued, expiry = results.get(timeout=5)
    assert issued >= released_at, (issued, released_at)
    assert expiry == issued + timedelta(seconds=2), (issued, expiry)


if __name__ == "__main__":
    root, scenario, mutation, *options = sys.argv[1:]
    if scenario == "clock":
        clock_scenario(root)
    elif scenario == "capacity":
        capacity_scenario(root, mutation)
    elif scenario == "settlement":
        settlement_scenario(root, mutation, options[0], options[1] == "True")
    else:
        raise AssertionError(scenario)
