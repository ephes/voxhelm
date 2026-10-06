"""Django settings for the test suite and mypy's django-stubs plugin.

``config.settings`` refuses to start without ``DJANGO_SECRET_KEY`` when
``DJANGO_DEBUG`` is off, so provide a fixed test-only key (unless one is already
set) before importing it. ``setdefault`` writes to ``os.environ``, so subprocesses
started by tests with ``DJANGO_SETTINGS_MODULE=config.settings`` inherit it.
"""

import os

os.environ.setdefault("DJANGO_SECRET_KEY", "voxhelm-test-only-secret-key")

from config.settings import *  # noqa: E402, F403
