"""Bounded regular-file admission for native transcript output."""

import errno
import os
from pathlib import Path
from stat import S_ISREG


def read_native_json(path: Path, *, limit: int) -> bytes:
    """Read whisper.cpp's native JSON output, refusing files over ``limit`` bytes.

    The file is opened without following a symlink and without blocking (a FIFO with
    no writer cannot stall the open), and must be a regular file on the open
    descriptor; anything else raises ``RuntimeError``. Reads at most ``limit + 1``
    bytes from that descriptor, so a file that grows after a size check cannot slip
    past the cap. Raises ``RuntimeError`` when over the cap and ``OSError`` when the
    file cannot be opened or read. Descriptor identity, size, mtime and ctime must
    stay unchanged across the read, and the read must return the admitted size;
    a changing or incomplete generation raises ``RuntimeError``.
    """
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0) | os.O_CLOEXEC
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise RuntimeError("whisper.cpp JSON output is not a regular file.") from exc
        raise
    with os.fdopen(descriptor, "rb") as handle:
        metadata = os.fstat(handle.fileno())
        if not S_ISREG(metadata.st_mode):
            raise RuntimeError("whisper.cpp JSON output is not a regular file.")
        if metadata.st_size > limit:
            raise RuntimeError(
                f"whisper.cpp JSON output exceeded the {limit // (1024 * 1024)} MiB window limit."
            )
        os.set_blocking(handle.fileno(), True)
        raw = handle.read(limit + 1)
        after = os.fstat(handle.fileno())
        if len(raw) > limit:
            raise RuntimeError(
                f"whisper.cpp JSON output exceeded the {limit // (1024 * 1024)} MiB window limit."
            )
        if (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or len(raw) != metadata.st_size:
            raise RuntimeError("whisper.cpp JSON output changed while being read.")
    return raw
