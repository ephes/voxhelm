import os
import subprocess
import sys
from pathlib import Path

DRIVER = """
import sys
import threading
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch
import django
django.setup()
from django.conf import settings
from django.db import connections
from django.core.management import call_command
from django.utils import timezone
from jobs.models import Job
from jobs.remote_workers import heartbeat_remote_work, hash_lease_token
from transcriptions.errors import ApiError
settings.DATABASES['default']['NAME'] = Path(sys.argv[1]) / 'heartbeats.db'
connections.__dict__.pop('settings', None)
call_command('migrate', verbosity=0)
now = timezone.now()
job = Job.objects.create(producer='synthetic', job_type='transcribe', input_data={},
    execution_mode='remote_pull', state='running', assigned_worker_id='test',
    lease_token_hash=hash_lease_token('synthetic'),
    lease_expires_at=now + timedelta(seconds=300))
started, release = threading.Event(), threading.Event()
responses, errors = {}, []

def clock():
    if threading.current_thread().name == 'older' and not started.is_set():
        started.set()
        assert release.wait(10)
        return now
    return now + timedelta(seconds=10)

def beat(name):
    try:
        responses[name] = heartbeat_remote_work(worker_id='test', job_id=job.pk,
            payload={'lease_token': 'synthetic', 'progress': {'phase': name}})
    except BaseException as exc:
        errors.append(repr(exc))
    finally:
        connections.close_all()

with patch('jobs.remote_workers.timezone.now', clock):
    older = threading.Thread(target=beat, args=('older',), name='older')
    older.start()
    try:
        assert started.wait(10)
        beat('newer')
    finally:
        release.set()
        older.join(10)
    assert not older.is_alive()
assert errors == [], errors
job.refresh_from_db()
assert job.worker_progress == {'phase': 'newer'}, job.worker_progress
assert job.last_worker_heartbeat_at == now + timedelta(seconds=10)
assert responses['older'] == responses['newer'], responses
# A shorter configured extension never shortens a stored expiry.
settings.VOXHELM_REMOTE_WORKER_LEASE_SECONDS = 1
before = job.lease_expires_at
with patch('jobs.remote_workers.timezone.now', return_value=now + timedelta(seconds=20)):
    heartbeat_remote_work(worker_id='test', job_id=job.pk,
        payload={'lease_token': 'synthetic'})
job.refresh_from_db()
assert job.last_worker_heartbeat_at == now + timedelta(seconds=20)
assert job.lease_expires_at == before
for updates in ({'lease_token_hash': hash_lease_token('rotated')},
                {'lease_token_hash': hash_lease_token('synthetic'),
                 'lease_expires_at': now - timedelta(seconds=1)}):
    Job.objects.filter(pk=job.pk).update(**updates)
    try:
        heartbeat_remote_work(worker_id='test', job_id=job.pk, payload={'lease_token': 'synthetic'})
    except ApiError:
        pass
    else:
        raise AssertionError('lost lease accepted')
"""


def test_reverse_heartbeat_publication_on_file_sqlite(tmp_path):
    root = Path(__file__).resolve().parent.parent
    driver = tmp_path / "heartbeat_driver.py"
    driver.write_text(DRIVER)
    completed = subprocess.run(
        [sys.executable, str(driver), str(tmp_path)],
        cwd=root,
        env={**os.environ, "DJANGO_SETTINGS_MODULE": "tests.settings", "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
