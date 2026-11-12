"""Bounded diagnostic handling for decoders whose output goes to a file."""

import os
import selectors
import subprocess
from time import monotonic

DECODE_DIAGNOSTIC_MAX_BYTES = 64 * 1024


class AudioDecodeError(RuntimeError):
    """A decoder exceeded its diagnostic allowance."""


def run_audio_decode(argv: list[str], *, timeout: float) -> None:
    """Discard stdout and diagnostics; kill/reap on byte or elapsed deadline limits."""
    deadline = monotonic() + timeout
    with subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE) as process:
        assert process.stderr is not None
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stderr, selectors.EVENT_READ)
                received = 0
                while selector.get_map():
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(argv, timeout)
                    for key, _ in selector.select(remaining):
                        chunk = os.read(
                            key.fd, min(8192, DECODE_DIAGNOSTIC_MAX_BYTES - received + 1)
                        )
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        received += len(chunk)
                        if received > DECODE_DIAGNOSTIC_MAX_BYTES:
                            raise AudioDecodeError(
                                "Audio decode exceeded the diagnostic byte limit."
                            )
                code = process.wait(timeout=max(0, deadline - monotonic()))
                if code:
                    raise subprocess.CalledProcessError(code, argv)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
