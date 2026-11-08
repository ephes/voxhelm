"""Durable cleanup recovery with synthetic storage and isolated SQLite."""

import pytest

from tests.test_remote_process_contracts import run_contract


@pytest.mark.parametrize("kind", ["prefix", "staged"])
def test_retention_retries_after_delete_before_commit_process_death(tmp_path, kind):
    run_contract(tmp_path, "tests.recovery_contracts", kind)


def test_identical_completion_retry_converges_after_staged_cleanup_failure(tmp_path):
    run_contract(tmp_path, "tests.recovery_contracts", "completion")


@pytest.mark.parametrize(
    "scenario,mutation,marker",
    [
        ("prefix", "ignore_references", "referenced canary removed"),
        ("staged", "skip_missing_staged", "staged ownership did not converge"),
        ("completion", "no_idempotent_completion", "identical completion retry refused"),
    ],
)
def test_recovery_oracles_detect_counterfactual_regressions(tmp_path, scenario, mutation, marker):
    with pytest.raises(AssertionError, match=marker):
        run_contract(tmp_path, "tests.recovery_contracts", scenario, mutation=mutation)
