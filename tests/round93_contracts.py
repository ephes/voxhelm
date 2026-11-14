"""Independent processes using a disposable database and synthetic bytes only."""

import multiprocessing
import sys
from io import StringIO
from unittest.mock import patch

from tests.process_contracts import create_database, initialize, stop_children


def bootstrap_child(root, role, ready, release, results):
    initialize(root)
    from django.contrib.auth import get_user_model
    from django.core.management import call_command

    user_model = get_user_model()
    hash_password = user_model.set_password

    def pause_hash(self, raw):
        hash_password(self, raw)
        ready.wait(15)
        assert release.wait(15)

    try:
        with patch.object(user_model, "set_password", pause_hash):
            call_command(
                "bootstrap_operator",
                username="synthetic",
                password=f" {role} ",
                email=f"{role}@example.invalid",
                stdout=StringIO(),
            )
        results.put(role)
    except Exception as exc:
        results.put(repr(exc))


def bootstrap_scenario(root):
    create_database(root)
    from django.contrib.auth import get_user_model
    from django.db import connections

    ctx = multiprocessing.get_context("spawn")
    ready, release, results = ctx.Barrier(3), ctx.Event(), ctx.Queue()
    children = [
        ctx.Process(target=bootstrap_child, args=(root, role, ready, release, results))
        for role in ("first", "second")
    ]
    try:
        for child in children:
            child.start()
        ready.wait(20)
        assert not get_user_model().objects.exists(), "partially published bootstrap account"
    finally:
        release.set()
        stop_children(children)
    outcomes = [results.get(timeout=5) for _ in children]
    assert sorted(outcomes) == ["first", "second"], outcomes
    connections.close_all()
    user = get_user_model().objects.get()
    role = user.email.split("@")[0]
    assert user.check_password(f" {role} ")
    assert user.is_active and user.is_staff


def retention_child(root, slow, ready, release, results):
    initialize(root)
    from django.core.management import CommandError, call_command

    from jobs import retention
    from jobs.management.commands import prune_job_artifacts as command

    original = command.prune_job_artifacts
    delete = retention.delete_stored_object

    def failing_delete(*, identity, key):
        if key.endswith("retry.mp3") and (root / "fail-delete").exists():
            raise RuntimeError("synthetic transient delete failure")
        return delete(identity=identity, key=key)

    retention.delete_stored_object = failing_delete

    def pause_after_sweep(**kwargs):
        result = original(**kwargs)
        ready.set()
        assert release.wait(20)
        return result

    if slow:
        command.prune_job_artifacts = pause_after_sweep
    outputs = []
    for _ in range(1 if slow else 2):
        out = StringIO()
        try:
            call_command("prune_job_artifacts", batch_size=1, stdout=out)
        except CommandError as exc:
            assert "Could not delete" in str(exc)
        outputs.append(out.getvalue())
    results.put(outputs)


def retention_scenario(root):
    from pathlib import Path
    from uuid import uuid4

    create_database(root)
    root = Path(root)
    from django.core.management import call_command

    from jobs.artifacts import current_artifact_store_identity, get_artifact_store
    from jobs.models import PendingArtifactDeletion, RetentionProgress

    RetentionProgress.objects.create(name="job-artifacts")
    (root / "fail-delete").touch()
    for number in range(5):
        job_id = uuid4()
        name = "retry.mp3" if number == 0 else f"source-{number}.mp3"
        key = f"voxhelm/jobs/{job_id}/attempt-0/{name}"
        get_artifact_store().put_bytes(key=key, data=b"synthetic", content_type="audio/mpeg")
        PendingArtifactDeletion.objects.create(
            job_id=job_id,
            kind="source",
            storage_backend="filesystem",
            storage_key=key,
            storage_identity=current_artifact_store_identity(),
        )
    ctx = multiprocessing.get_context("spawn")
    ready, release, results = ctx.Event(), ctx.Event(), ctx.Queue()
    slow = ctx.Process(target=retention_child, args=(root, True, ready, release, results))
    fast = ctx.Process(target=retention_child, args=(root, False, ready, release, results))
    try:
        slow.start()
        assert ready.wait(20)
        fast.start()
        stop_children([fast])
        latest = RetentionProgress.objects.get().cursor
        assert latest
    finally:
        release.set()
        stop_children([slow])
    outputs = [results.get(timeout=5) for _ in range(2)]
    assert RetentionProgress.objects.get().cursor == latest, outputs
    assert any("continuation was not saved" in text for batch in outputs for text in batch), outputs
    (root / "fail-delete").unlink()
    for _ in range(12):
        call_command("prune_job_artifacts", batch_size=1, stdout=StringIO())
        if not PendingArtifactDeletion.objects.exists():
            break
    assert not PendingArtifactDeletion.objects.exists()
    assert not [p for p in (root / "artifacts").rglob("*") if p.is_file()]


if __name__ == "__main__":
    root, scenario, *_ = sys.argv[1:]
    if scenario == "bootstrap":
        bootstrap_scenario(root)
    elif scenario == "retention":
        retention_scenario(root)
    else:
        raise AssertionError(scenario)
