from typing import Any, cast

import pytest
from asgiref.local import Local
from django.core.cache import cache
from django_tasks import task_backends

from jobs.artifacts import get_artifact_store
from operators import login_throttle


def reset_task_backend_cache() -> None:
    handler = cast(Any, task_backends)
    connections = handler._connections
    handler._connections = Local(connections._thread_critical)


def reset_login_reservations() -> None:
    with login_throttle._LOCK:
        login_throttle._PENDING.clear()


@pytest.fixture(autouse=True)
def configure_settings(settings, tmp_path):
    settings.VOXHELM_BEARER_TOKENS = {"archive": "test-token"}
    settings.VOXHELM_ARTIFACT_BACKEND = "filesystem"
    settings.VOXHELM_ARTIFACT_ROOT = tmp_path / "artifacts"
    get_artifact_store.cache_clear()
    reset_task_backend_cache()
    cache.clear()
    reset_login_reservations()
    yield
    get_artifact_store.cache_clear()
    reset_task_backend_cache()
    cache.clear()
    reset_login_reservations()
