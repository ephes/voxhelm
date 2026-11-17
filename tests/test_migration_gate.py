"""The migration drift gate: models and committed migrations must agree.

``just check`` runs ``just check-migrations`` (``makemigrations --check --dry-run``
under ``tests.settings``) so a model change without its migration fails the local
gate before merge instead of surfacing at deploy time.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from django.core.management import call_command

JUSTFILE = Path(__file__).resolve().parent.parent / "justfile"


def _recipe_body(name: str) -> list[str]:
    """Return the indented command lines of one justfile recipe."""
    lines = JUSTFILE.read_text(encoding="utf-8").splitlines()
    header = re.compile(rf"^{re.escape(name)}(\s[^:]*)?:(\s|$)")
    for index, line in enumerate(lines):
        if header.match(line):
            body: list[str] = []
            for follow in lines[index + 1 :]:
                if not follow.strip():
                    break
                if not follow[0].isspace():
                    break
                body.append(follow.strip())
            return body
    raise AssertionError(f"justfile has no {name!r} recipe")


@pytest.mark.django_db
def test_models_have_no_pending_migrations() -> None:
    # ``--check`` exits non-zero (SystemExit) when a migration is missing.
    call_command("makemigrations", "--check", "--dry-run", verbosity=0)


def test_check_migrations_recipe_runs_makemigrations_check_under_test_settings() -> None:
    body = _recipe_body("check-migrations")
    command = " ".join(body)
    assert "manage.py makemigrations --check --dry-run" in command
    assert "DJANGO_SETTINGS_MODULE=tests.settings" in command


def test_check_recipe_runs_the_migration_gate() -> None:
    assert "just check-migrations" in _recipe_body("check")
