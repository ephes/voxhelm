"""Owned process cleanup failures with synthetic processes, never real signals."""

import subprocess
from types import SimpleNamespace

import pytest

from transcriptions import owned_process

MESSAGES = owned_process.OwnedProcessMessages("start", "deadline", "diagnostics", "failure")


@pytest.fixture
def fake_child(monkeypatch):
    real_kill = owned_process._kill_process_group
    events = []
    failures: set[str] = set()
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
        child=child,
        events=events,
        failures=failures,
        primary=primary,
        selector=Selector,
        real_kill=real_kill,
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


@pytest.mark.parametrize("kind", ["owned", "native"])
def test_kill_refusal_has_bounded_reap_and_marks_primary(fake_child, monkeypatch, kind):
    waits = []

    def wait(*, timeout=None):
        waits.append(timeout)
        raise subprocess.TimeoutExpired(["synthetic"], timeout or 0)

    def denied(*args):
        fake_child.events.append("denied")
        raise PermissionError("private kill diagnostic")

    monkeypatch.setattr(owned_process, "_kill_process_group", fake_child.real_kill)
    monkeypatch.setattr(owned_process.os, "killpg", denied)
    monkeypatch.setattr(owned_process.os, "waitid", lambda *a: None)
    fake_child.child.wait = wait
    with pytest.raises(OSError) as caught:
        run(kind)
    assert caught.value is fake_child.primary
    assert len(waits) == 1 and waits[0] is not None and 0 < waits[0] <= 2
    assert owned_process.process_ownership_uncertain(caught.value)
    assert "stderr" in fake_child.events
    if kind == "native":
        assert "stdout" in fake_child.events


@pytest.mark.parametrize("kind", ["owned", "native"])
def test_unreaped_child_never_returns_completed_output(fake_child, kind):
    fake_child.selector.fail = False
    waits = []

    def wait(*, timeout=None):
        waits.append(timeout)
        raise subprocess.TimeoutExpired(["synthetic"], timeout or 0)

    fake_child.child.wait = wait
    with pytest.raises(RuntimeError, match="Process cleanup failed") as caught:
        run(kind)
    assert len(waits) == 1 and waits[0] is not None and 0 < waits[0] <= 2
    assert owned_process.process_ownership_uncertain(caught.value)


def test_zombie_only_group_refusal_is_confirmed_gone_after_reap(fake_child, monkeypatch):
    fake_child.selector.fail = False
    signals = []

    def signal_group(pid, sig):
        signals.append(sig)
        if sig == 0:
            assert fake_child.events[-1] == "wait"
            raise ProcessLookupError
        raise PermissionError

    monkeypatch.setattr(owned_process, "_kill_process_group", fake_child.real_kill)
    monkeypatch.setattr(owned_process.os, "killpg", signal_group)
    run("native")
    assert signals == [owned_process.signal.SIGKILL, 0]


def test_exited_leader_does_not_prove_denied_group_is_retired(fake_child, monkeypatch):
    fake_child.selector.fail = False

    def denied(*args):
        raise PermissionError

    monkeypatch.setattr(owned_process, "_kill_process_group", fake_child.real_kill)
    monkeypatch.setattr(owned_process.os, "killpg", denied)
    with pytest.raises(RuntimeError, match="Process cleanup failed") as caught:
        run("native")
    assert owned_process.process_ownership_uncertain(caught.value)


@pytest.mark.parametrize("uncertain", [False, True])
def test_temporary_output_is_retained_only_for_uncertain_process_ownership(tmp_path, uncertain):
    from pathlib import Path

    failure = OSError("synthetic primary")
    if uncertain:
        failure.add_note(owned_process.PROCESS_OWNERSHIP_UNCERTAIN)
    output = None
    with pytest.raises(RuntimeError, match="wrapper"):
        with owned_process.process_output_directory(prefix=str(tmp_path / "synthetic-")) as root:
            output = Path(root) / "output.wav"
            output.write_bytes(b"synthetic")
            try:
                raise failure
            except OSError as exc:
                raise RuntimeError("wrapper") from exc
    assert output is not None and output.exists() is uncertain


def test_whisper_scratch_survives_uncertain_retirement(monkeypatch, tmp_path):
    from pathlib import Path

    from transcriptions import service

    monkeypatch.setattr(owned_process.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(service, "resolve_whispercpp_binary", lambda _: "synthetic")
    monkeypatch.setattr(service, "resolve_whispercpp_model_path", lambda _: tmp_path / "model")
    monkeypatch.setattr(
        service,
        "normalize_audio_to_wav_16k_mono",
        lambda **kw: kw["output_path"].write_bytes(b"wav"),
    )
    failure = subprocess.TimeoutExpired(["synthetic"], 1)
    failure.add_note(owned_process.PROCESS_OWNERSHIP_UNCERTAIN)
    outputs = []

    def fail(args, **kwargs):
        output = Path(args[args.index("-of") + 1]).with_suffix(".json")
        output.write_text("{}")
        outputs.append(output)
        raise failure

    monkeypatch.setattr(service, "run_cancellable_process", fail)
    backend = service.WhisperCppBackend(binary_path="synthetic", model_name="model", processors=1)
    with pytest.raises(subprocess.TimeoutExpired):
        backend.transcribe(tmp_path / "input", service.TranscribeParams("auto", None, None))
    assert outputs[0].exists()
    assert (outputs[0].parent / "input.wav").exists()


def test_export_output_survives_uncertain_retirement(monkeypatch, tmp_path):
    from pathlib import Path

    from synthesis import service
    from tests.test_speech_bounds import _result

    monkeypatch.setattr(owned_process.tempfile, "tempdir", str(tmp_path))
    failure = RuntimeError("synthetic failure")
    failure.add_note(owned_process.PROCESS_OWNERSHIP_UNCERTAIN)
    outputs = []

    def fail(args, **kwargs):
        output = Path(args[-1])
        output.write_bytes(b"partial synthetic")
        outputs.append(output)
        raise failure

    monkeypatch.setattr(service, "run_owned_process", fail)
    with pytest.raises(RuntimeError) as caught:
        service.export_audio(_result(tmp_path), output_format="mp3")
    assert caught.value is failure
    assert outputs[0].read_bytes() == b"partial synthetic"


@pytest.mark.parametrize("kind", ["owned", "native"])
def test_caller_handled_exception_is_not_mistaken_for_run_failure(fake_child, kind):
    # A run that starts while its caller is handling an unrelated exception must
    # still fail closed when cleanup cannot reap the child, and must not tag the
    # caller's exception with the ownership note.
    fake_child.selector.fail = False

    def wait(*, timeout=None):
        raise subprocess.TimeoutExpired(["synthetic"], timeout or 0)

    fake_child.child.wait = wait
    outer = ValueError("caller is handling this")
    try:
        raise outer
    except ValueError:
        with pytest.raises(RuntimeError, match="Process cleanup failed") as caught:
            run(kind)
    assert owned_process.process_ownership_uncertain(caught.value)
    assert not owned_process.process_ownership_uncertain(outer)


def test_output_directory_ignores_caller_handled_uncertain_exception(tmp_path):
    from pathlib import Path

    outer = OSError("caller is handling this")
    outer.add_note(owned_process.PROCESS_OWNERSHIP_UNCERTAIN)
    try:
        raise outer
    except OSError:
        with owned_process.process_output_directory(prefix=str(tmp_path / "synthetic-")) as root:
            output = Path(root) / "output.wav"
            output.write_bytes(b"synthetic")
    # This block succeeded, so its own scratch is cleaned up.
    assert not output.exists()
