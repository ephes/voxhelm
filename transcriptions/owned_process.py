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
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
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
_REAP_TIMEOUT_SECONDS = 2.0
PROCESS_OWNERSHIP_UNCERTAIN = "Native process ownership is uncertain; retain writable outputs."


def process_ownership_uncertain(error: BaseException | None) -> bool:
    """Recognize uncertain retirement even through a caller's exception wrapper."""
    seen: set[int] = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if PROCESS_OWNERSHIP_UNCERTAIN in getattr(error, "__notes__", ()):
            return True
        error = error.__cause__ or error.__context__
    return False


@contextmanager
def process_output_directory(*, prefix: str) -> Iterator[str]:
    """Retain native writable scratch when retirement could not be confirmed."""
    temporary = tempfile.TemporaryDirectory(prefix=prefix, delete=False)
    with temporary as directory:
        # Only this block's own failure counts; sys.exception() in finally would also
        # see an unrelated exception a caller is handling.
        primary: BaseException | None = None
        try:
            yield directory
        except BaseException as exc:
            primary = exc
            raise
        finally:
            if process_ownership_uncertain(primary):
                _LOGGER.warning("Native writable scratch retained; operator recovery is required.")
            else:
                # delete=False disables automatic cleanup, including on GC; explicit
                # cleanup is safe only after the process has retired.
                temporary.cleanup()


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
    except ProcessLookupError:
        pass


def _cleanup_process(process: subprocess.Popen[bytes], *, primary: BaseException | None) -> None:
    """Attempt every owned resource even when an earlier cleanup stage fails.

    ``primary`` is the run's own failure (None after a normal exit). It is passed
    explicitly because sys.exception() in a finally block also reports exceptions
    a caller is still handling, which would hide a cleanup failure after success.
    """
    preserve_error = primary is not None
    failed = False
    uncertain = False
    kill_denied = False
    reaped = False
    cleanup_deadline = monotonic() + _REAP_TIMEOUT_SECONDS
    stages: list[tuple[str, Callable[[], object]]] = [
        ("kill", lambda: _kill_process_group(process)),
    ]
    for name in ("stdout", "stderr"):
        pipe = getattr(process, name)
        if pipe is not None:
            stages.append((name, pipe.close))
    stages.append(("reap", lambda: process.wait(timeout=max(0.0, cleanup_deadline - monotonic()))))
    for name, action in stages:
        try:
            action()
            if name == "reap":
                reaped = True
        except PermissionError:
            if name == "kill":
                # macOS refuses a signal to a zombie-only group. Defer judgment
                # until after reaping, then probe existence without signaling.
                kill_denied = True
                continue
            failed = True
            uncertain |= name == "reap"
            _LOGGER.warning("Process cleanup failed at %s; operator recovery may be needed.", name)
        except BaseException:
            failed = True
            uncertain |= name in {"kill", "reap"}
            _LOGGER.warning("Process cleanup failed at %s; operator recovery may be needed.", name)
    if kill_denied:
        group_gone = False
        if reaped:
            try:
                # Signal 0 is safe even if the former group ID has been reused.
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                group_gone = True
            except BaseException:
                pass
        if not group_gone:
            failed = uncertain = True
            _LOGGER.warning("Process cleanup failed at kill; operator recovery may be needed.")
    if uncertain and primary is not None:
        primary.add_note(PROCESS_OWNERSHIP_UNCERTAIN)
    if failed and not preserve_error:
        error = RuntimeError("Process cleanup failed; operator recovery may be needed.")
        if uncertain:
            error.add_note(PROCESS_OWNERSHIP_UNCERTAIN)
        raise error


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
    open by descendants; every exit attempts group retirement and bounded child
    reaping. Unconfirmed retirement marks the failure for writable-output recovery.
    Failures raise ``RuntimeError`` with the given fixed messages
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
    primary: BaseException | None = None
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
    except BaseException as exc:
        primary = exc
        raise
    finally:
        _cleanup_process(process, primary=primary)
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
    primary: BaseException | None = None
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
    except BaseException as exc:
        primary = exc
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
        _cleanup_process(process, primary=primary)
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
