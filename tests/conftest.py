from typing import Any, cast

import pytest
from asgiref.local import Local
from django_tasks import task_backends

from jobs.artifacts import get_artifact_store


def reset_task_backend_cache() -> None:
    handler = cast(Any, task_backends)
    connections = handler._connections
    handler._connections = Local(connections._thread_critical)


@pytest.fixture(autouse=True)
def configure_settings(settings, tmp_path):
    settings.VOXHELM_BEARER_TOKENS = {"archive": "test-token"}
    settings.VOXHELM_ARTIFACT_BACKEND = "filesystem"
    settings.VOXHELM_ARTIFACT_ROOT = tmp_path / "artifacts"
    get_artifact_store.cache_clear()
    reset_task_backend_cache()
    yield
    get_artifact_store.cache_clear()
    reset_task_backend_cache()
