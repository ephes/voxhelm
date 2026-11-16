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
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import monotonic, sleep

_LOGGER = logging.getLogger(__name__)

OWNED_PROCESS_DIAGNOSTIC_MAX_BYTES = 64 * 1024
# run_native_process retains at most OWNED_PROCESS_DIAGNOSTIC_MAX_BYTES across both pipes.
_STDERR_HEAD_BYTES = 52 * 1024
_STDERR_TAIL_BYTES = 8 * 1024
_STDOUT_HEAD_BYTES = 2 * 1024
_STDOUT_TAIL_BYTES = 2 * 1024
# After the direct child exits, pipes still held by descendants are drained this long.
_EXITED_DRAIN_SECONDS = 2.0
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


def _cleanup_process(process: subprocess.Popen[bytes], *, preserve_error: bool) -> None:
    """Attempt every owned resource even when an earlier cleanup stage fails."""
    failed = False
    stages: list[tuple[str, Callable[[], object]]] = [
        ("kill", lambda: _kill_process_group(process)),
    ]
    for name in ("stdout", "stderr"):
        pipe = getattr(process, name)
        if pipe is not None:
            stages.append((name, pipe.close))
    stages.append(("reap", process.wait))
    for name, action in stages:
        try:
            action()
        except BaseException:
            failed = True
            _LOGGER.warning("Process cleanup failed at %s; operator recovery may be needed.", name)
    if failed and not preserve_error:
        raise RuntimeError("Process cleanup failed; operator recovery may be needed.")


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
        _cleanup_process(process, preserve_error=sys.exception() is not None)
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
    """Drain both pipes keeping a bounded head and tail of each (64 KiB in total).

    stderr keeps its first 52 KiB and last 8 KiB, stdout its first and last 2 KiB.

    Output beyond that is discarded, not refused: whisper-cli prints every
    transcript segment to stdout, so long jobs exceed any small cap. Return
    codes and process-group ownership are preserved.
    """
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
    # stderr carries parsed evidence (whisper-cli's "auto-detected language:" line
    # follows its model-load log) and the error text, so it gets most of the budget;
    # stdout is only read for short answers (ffprobe, --version).
    streams = {
        "stdout": _HeadTail(head=_STDOUT_HEAD_BYTES, tail=_STDOUT_TAIL_BYTES),
        "stderr": _HeadTail(head=_STDERR_HEAD_BYTES, tail=_STDERR_TAIL_BYTES),
    }
    exited_drain_until: float | None = None
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
                if exited_drain_until is None and os.waitid(
                    os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT
                ):
                    # A descendant may still hold a pipe open after the child exited:
                    # drain what is buffered briefly, then let the group cleanup run.
                    exited_drain_until = monotonic() + _EXITED_DRAIN_SECONDS
                if exited_drain_until is not None and monotonic() >= exited_drain_until:
                    break
                if exited_drain_until is not None:
                    wait = min(wait, max(0.0, exited_drain_until - monotonic()))
                if selector.get_map():
                    for key, _ in selector.select(max(0, wait)):
                        chunk = os.read(key.fd, 8192)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        # whisper-cli prints every transcript segment to stdout, so
                        # long jobs legitimately exceed the cap: keep a bounded head
                        # (language/version evidence) and tail (errors), drop the rest.
                        streams[key.data].add(chunk)
                else:
                    if os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT):
                        break
                    sleep(max(0, wait))
    except BaseException:
        # A secondary signal/probe error must not replace cancellation/read failure.
        try:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError, PermissionError:
                pass
            grace = monotonic() + terminate_grace_seconds
            while os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
                if monotonic() >= grace:
                    break
                sleep(0.01)
        except BaseException:
            _LOGGER.warning("Process cleanup failed at terminate; operator recovery may be needed.")
        raise
    finally:
        _cleanup_process(process, preserve_error=sys.exception() is not None)
    check_cancelled()
    return subprocess.CompletedProcess(
        args,
        process.returncode,
        streams["stdout"].text(),
        streams["stderr"].text(),
    )


class _HeadTail:
    """A bounded copy of one stream: its first ``head`` and last ``tail`` bytes."""

    def __init__(self, *, head: int, tail: int) -> None:
        self.head_limit = head
        self.tail_limit = tail
        self.head = bytearray()
        self.tail = bytearray()
        self.omitted = 0

    def add(self, chunk: bytes) -> None:
        room = self.head_limit - len(self.head)
        if room > 0:
            self.head.extend(chunk[:room])
            chunk = chunk[room:]
        if not chunk:
            return
        self.tail.extend(chunk)
        excess = len(self.tail) - self.tail_limit
        if excess > 0:
            del self.tail[:excess]
            self.omitted += excess

    def text(self) -> str:
        head = self.head.decode("utf-8", errors="replace")
        tail = self.tail.decode("utf-8", errors="replace")
        if not self.omitted:
            return head + tail
        return f"{head}\n[{self.omitted} bytes of output omitted]\n{tail}"
