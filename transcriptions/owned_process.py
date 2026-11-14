"""Run a converter that owns its process group, with bounded diagnostics and one deadline.

Shared by speech export and transcription audio normalization. Only the standard
library is imported, so any app can use it without import cycles.
"""

from __future__ import annotations

import logging
import os
import selectors
import signal
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import monotonic, sleep

_LOGGER = logging.getLogger(__name__)

OWNED_PROCESS_DIAGNOSTIC_MAX_BYTES = 64 * 1024
OWNED_PROCESS_DIAGNOSTIC_LOG_BYTES = 2048


@dataclass(frozen=True)
class OwnedProcessMessages:
    """Fixed public error texts; tool output never goes into them."""

    start: str
    deadline: str
    diagnostics: str
    failure: str


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    # EPERM: on macOS a group whose only member is the unreaped leader refuses signals.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError, PermissionError:
        pass


def run_owned_process(
    args: list[str],
    *,
    timeout: float,
    messages: OwnedProcessMessages,
    fail_on_diagnostic_overflow: bool = True,
) -> None:
    """Run ``args`` in its own process group under one elapsed deadline.

    stdout is discarded and stderr is drained with a fixed byte ceiling, so a noisy
    tool cannot grow memory. Past the ceiling the run fails, or, with
    ``fail_on_diagnostic_overflow=False`` (tools fed arbitrary user media), further
    output is drained and discarded so a noisy but successful run still succeeds.
    The deadline covers the whole run, including pipes held
    open by descendants; on any exit path the whole process group is killed and the
    child reaped. Failures raise ``RuntimeError`` with the given fixed messages
    (timeouts chain ``subprocess.TimeoutExpired``); a bounded prefix of the tool's
    output is logged for operators.
    """
    deadline = monotonic() + timeout
    try:
        process = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise RuntimeError(messages.start) from exc
    diagnostic = bytearray()
    received = 0
    try:
        assert process.stderr is not None
        with selectors.DefaultSelector() as selector:
            selector.register(process.stderr, selectors.EVENT_READ)
            while selector.get_map():
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise RuntimeError(messages.deadline) from subprocess.TimeoutExpired(
                        args, timeout
                    )
                for key, _ in selector.select(remaining):
                    chunk = os.read(key.fd, 8192)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    received += len(chunk)
                    room = OWNED_PROCESS_DIAGNOSTIC_LOG_BYTES - len(diagnostic)
                    if room > 0:
                        diagnostic.extend(chunk[:room])
                    if (
                        fail_on_diagnostic_overflow
                        and received > OWNED_PROCESS_DIAGNOSTIC_MAX_BYTES
                    ):
                        raise RuntimeError(messages.diagnostics)
        # Wait for the child without reaping it: while it is unreaped its PID (and so
        # its process group ID) cannot be reused, so the group kill below can never
        # reach an unrelated process.
        while os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
            if monotonic() >= deadline:
                raise RuntimeError(messages.deadline) from subprocess.TimeoutExpired(args, timeout)
            sleep(0.01)
    finally:
        _kill_process_group(process)
        if process.stderr is not None:
            process.stderr.close()
        process.wait()
    if process.returncode != 0:
        _LOGGER.warning(
            "%s Exit status %s: %s",
            messages.failure,
            process.returncode,
            diagnostic.decode("utf-8", errors="replace").strip() or "(no output)",
        )
        raise RuntimeError(messages.failure)


def run_native_process(
    args: list[str],
    *,
    check_cancelled: Callable[[], None],
    timeout: float | None,
    poll_seconds: float,
    terminate_grace_seconds: float,
    env: Mapping[str, str] | None,
    pass_fds: tuple[int, ...],
) -> subprocess.CompletedProcess[str]:
    """Capture bounded diagnostic tails, preserving return codes and process ownership."""
    deadline = None if timeout is None else monotonic() + timeout
    check_cancelled()
    process = subprocess.Popen(
        args,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        env=None if env is None else dict(env),
        pass_fds=pass_fds,
    )
    tails = {"stdout": bytearray(), "stderr": bytearray()}
    received = 0
    try:
        assert process.stdout is not None and process.stderr is not None
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
            while True:
                check_cancelled()
                if deadline is not None and monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(args, timeout or 0)
                wait = (
                    poll_seconds if deadline is None else min(poll_seconds, deadline - monotonic())
                )
                if selector.get_map():
                    for key, _ in selector.select(max(0, wait)):
                        chunk = os.read(key.fd, 8192)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        received += len(chunk)
                        tail = tails[key.data]
                        tail.extend(chunk)
                        del tail[:-OWNED_PROCESS_DIAGNOSTIC_LOG_BYTES]
                        if received > OWNED_PROCESS_DIAGNOSTIC_MAX_BYTES:
                            raise RuntimeError(
                                "Native inference exceeded the diagnostic output limit."
                            )
                else:
                    if os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT):
                        break
                    sleep(max(0, wait))
    except BaseException:
        # Cooperate first, then retire the entire group, including pipe-owning descendants.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError, PermissionError:
            pass
        grace = monotonic() + terminate_grace_seconds
        while os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
            if monotonic() >= grace:
                break
            sleep(0.01)
        raise
    finally:
        _kill_process_group(process)
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
        process.wait()
    check_cancelled()
    return subprocess.CompletedProcess(
        args,
        process.returncode,
        tails["stdout"].decode("utf-8", errors="replace"),
        tails["stderr"].decode("utf-8", errors="replace"),
    )
