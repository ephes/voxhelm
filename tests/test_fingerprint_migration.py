"""Historical upgrade contracts using isolated synthetic databases."""

import importlib
import uuid
from contextlib import contextmanager

import pytest
from django.db import IntegrityError, connections
from django.db.migrations.executor import MigrationExecutor
from django.db.models.query import QuerySet

migration = importlib.import_module("jobs.migrations.0008_job_request_fingerprint")
OLD = [("jobs", "0007_remote_workers")]
NEW = [("jobs", "0008_job_request_fingerprint")]


@contextmanager
def historical_database(tmp_path, django_db_blocker):
    alias = "fingerprint_upgrade"
    config = connections["default"].settings_dict.copy()
    config["NAME"] = str(tmp_path / "upgrade.sqlite3")
    connections.databases[alias] = config
    with django_db_blocker.unblock():
        db = connections[alias]
        try:
            executor = MigrationExecutor(db)
            executor.migrate(OLD)
            yield db, executor.loader.project_state(OLD).apps.get_model("jobs", "Job")
        finally:
            db.close()
            del connections[alias]
            del connections.databases[alias]


def seed(Job, alias, count=7):
    rows = [
        Job.objects.using(alias).create(
            id=uuid.UUID(int=i + 1),
            producer="synthetic",
            task_ref="duplicate",
            job_type="synthesize",
            input_data={"large": "x" * 100000},
            output_data={"large": "y" * 100000},
            state="failed" if i == 1 else "queued",
        )
        for i in range(count)
    ]
    # Exercise the id tie-breaker and duplicates spanning multiple batches.
    Job.objects.using(alias).update(created_at=rows[0].created_at)
    return rows


@pytest.mark.django_db(transaction=True)
def test_upgrade_only_changes_selected_database(tmp_path, django_db_blocker):
    default = connections["default"]
    executor = MigrationExecutor(default)
    executor.migrate(OLD)
    DefaultJob = executor.loader.project_state(OLD).apps.get_model("jobs", "Job")
    try:
        seed(DefaultJob, "default", count=3)
        DefaultJob.objects.update(producer="default", task_ref="default-only")
        before = list(DefaultJob.objects.order_by("id").values())
        with historical_database(tmp_path, django_db_blocker) as (db, OldJob):
            rows = seed(OldJob, db.alias)
            executor = MigrationExecutor(db)
            executor.migrate(NEW)
            NewJob = executor.loader.project_state(NEW).apps.get_model("jobs", "Job")
            values = dict(NewJob.objects.using(db.alias).values_list("id", "request_fingerprint"))
            # Independent canonical digest for {"job_type":"synthesize"}.
            import hashlib

            expected = hashlib.sha256(b'{"job_type":"synthesize"}').hexdigest()
            assert values[rows[-1].id] == values[rows[1].id] == expected
            assert all(values[row.id] == "" for row in rows[:-1] if row.state != "failed")
            with pytest.raises(IntegrityError):
                NewJob.objects.using(db.alias).create(
                    producer="synthetic",
                    task_ref="duplicate",
                    job_type="synthesize",
                    state="queued",
                    request_fingerprint=expected,
                    input_data={},
                    output_data={},
                )
        assert list(DefaultJob.objects.order_by("id").values()) == before
        with default.cursor() as cursor:
            constraints = default.introspection.get_constraints(cursor, "jobs_job")
            assert "jobs_job_active_task_ref_fingerprint_unique" not in constraints
    finally:
        executor = MigrationExecutor(default)
        executor.migrate(executor.loader.graph.leaf_nodes())


def test_upgrade_bounds_loaded_payloads(tmp_path, django_db_blocker, monkeypatch):
    with historical_database(tmp_path, django_db_blocker) as (db, Job):
        seed(Job, db.alias)
        monkeypatch.setattr(migration, "BACKFILL_BATCH_SIZE", 2, raising=False)
        original = QuerySet._fetch_all
        loaded = []

        def fetch(query):
            original(query)
            if query.model._meta.label_lower == "jobs.job" and query._result_cache:
                loaded.append(len(query._result_cache))

        monkeypatch.setattr(QuerySet, "_fetch_all", fetch)
        MigrationExecutor(db).migrate(NEW)
        assert loaded and max(loaded) <= 2
        with db.cursor() as cursor:
            cursor.execute("SELECT request_fingerprint, state FROM jobs_job")
            values = cursor.fetchall()
        assert sum(bool(fp) for fp, _ in values) == 2


def test_upgrade_failure_rolls_back_prior_batches(tmp_path, django_db_blocker, monkeypatch):
    with historical_database(tmp_path, django_db_blocker) as (db, Job):
        rows = seed(Job, db.alias)
        monkeypatch.setattr(migration, "BACKFILL_BATCH_SIZE", 2, raising=False)
        original = migration._fingerprint
        calls = []

        def fail_late(job):
            calls.append(job.pk)
            if len(calls) == 4:
                raise RuntimeError("synthetic late failure")
            return original(job)

        monkeypatch.setattr(migration, "_fingerprint", fail_late)
        with pytest.raises(RuntimeError, match="synthetic late failure"):
            MigrationExecutor(db).migrate(NEW)
        assert len(calls) == 4
        assert Job.objects.using(db.alias).count() == len(rows)
        with db.cursor() as cursor:
            columns = db.introspection.get_table_description(cursor, "jobs_job")
            assert "request_fingerprint" not in {c.name for c in columns}
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
            assert not any("backfill" in name for (name,) in cursor.fetchall())
        assert NEW[0] not in MigrationExecutor(db).loader.applied_migrations
