"""Real process/connection contracts with synthetic media and no external services."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


def run_contract(tmp_path, module, scenario, *options, mutation="normal", expect_failure=False):
    root = Path(__file__).resolve().parent.parent
    completed = subprocess.run(
        [sys.executable, "-m", module, str(tmp_path), scenario, mutation, *options],
        cwd=root,
        env={**os.environ, "DJANGO_SETTINGS_MODULE": "tests.settings", "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        timeout=90,
    )
    if expect_failure:
        assert completed.returncode != 0
        # Both participants must reach the ownership oracle; setup/import/timeouts
        # do not count as detecting the counterfactual lock removal.
        assert "AssertionError" in completed.stderr, completed.stderr
        assert "'role': 'first'" in completed.stderr, completed.stderr
        assert "'role': 'second'" in completed.stderr, completed.stderr
    else:
        assert completed.returncode == 0, completed.stdout + completed.stderr


def test_same_worker_capacity_across_spawned_processes(tmp_path):
    run_contract(tmp_path, "tests.process_contracts", "capacity")


def test_claim_time_after_real_worker_writer_contention(tmp_path):
    run_contract(tmp_path, "tests.process_contracts", "clock")


@pytest.mark.parametrize("first_action", ["complete", "fail"])
@pytest.mark.parametrize("retryable", [True, False])
def test_completion_failure_race_preserves_winner_manifest(tmp_path, first_action, retryable):
    run_contract(tmp_path, "tests.process_contracts", "settlement", first_action, str(retryable))


@pytest.mark.parametrize(
    "scenario,mutation,options",
    [
        ("capacity", "no_worker_lock", ()),
        ("settlement", "no_settlement_lock", ("complete", "True")),
    ],
)
def test_process_oracles_detect_counterfactual_lock_removal(tmp_path, scenario, mutation, options):
    run_contract(
        tmp_path,
        "tests.process_contracts",
        scenario,
        *options,
        mutation=mutation,
        expect_failure=True,
    )
