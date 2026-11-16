"""Owned process cleanup failures with synthetic processes, never real signals."""

import subprocess
from types import SimpleNamespace

import pytest

from transcriptions import owned_process

MESSAGES = owned_process.OwnedProcessMessages("start", "deadline", "diagnostics", "failure")


@pytest.fixture
def fake_child(monkeypatch):
    events = []
    failures = set()
    primary = OSError("synthetic read failure")

    def stage(name):
        events.append(name)
        if name in failures:
            raise OSError("private cleanup diagnostic")

    child = SimpleNamespace(
        pid=123456789,
        returncode=0,
        stdout=SimpleNamespace(close=lambda: stage("stdout")),
        stderr=SimpleNamespace(close=lambda: stage("stderr")),
        wait=lambda **kwargs: stage("wait"),
    )

    class Selector:
        fail = True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def register(self, *args):
            if self.fail:
                raise primary

        def get_map(self):
            return {}

    def popen(*args, **kwargs):
        if kwargs["stdout"] == subprocess.DEVNULL:
            child.stdout = None
        return child

    monkeypatch.setattr(owned_process.subprocess, "Popen", popen)
    monkeypatch.setattr(owned_process.selectors, "DefaultSelector", Selector)
    monkeypatch.setattr(owned_process.os, "waitid", lambda *a: object())
    monkeypatch.setattr(owned_process.os, "killpg", lambda *a: stage("term"))
    monkeypatch.setattr(owned_process, "_kill_process_group", lambda p: stage("kill"))
    return SimpleNamespace(
        child=child, events=events, failures=failures, primary=primary, selector=Selector
    )


def run(kind):
    if kind == "owned":
        return owned_process.run_owned_process(["synthetic"], timeout=1, messages=MESSAGES)
    return owned_process.run_native_process(
        ["synthetic"],
        check_cancelled=lambda: None,
        timeout=1,
        poll_seconds=0.01,
        terminate_grace_seconds=0,
        env=None,
        pass_fds=(),
    )


@pytest.mark.parametrize("kind", ["owned", "native"])
@pytest.mark.parametrize("stage", ["kill", "stdout", "stderr", "wait", "term"])
def test_cleanup_attempts_all_stages_and_preserves_primary(fake_child, caplog, kind, stage):
    fake_child.failures.add(stage)
    with pytest.raises(OSError) as caught:
        run(kind)
    assert caught.value is fake_child.primary
    expected = (["term"] if kind == "native" else []) + ["kill"]
    if kind == "native":
        expected += ["stdout"]
    expected += ["stderr", "wait"]
    assert fake_child.events == expected
    assert "private cleanup diagnostic" not in caplog.text
    if stage in expected:
        assert "Process cleanup failed" in caplog.text


@pytest.mark.parametrize("kind", ["owned", "native"])
@pytest.mark.parametrize("stage", ["kill", "stderr", "wait"])
def test_cleanup_failure_never_returns_success(fake_child, kind, stage):
    fake_child.selector.fail = False
    fake_child.failures.add(stage)
    with pytest.raises(RuntimeError, match="Process cleanup failed"):
        run(kind)
    assert fake_child.events[-1] == "wait"
