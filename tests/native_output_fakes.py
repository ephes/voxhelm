"""Pause a native JSON read while a separate writer changes the admitted file."""

import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path


def install_native_read_barrier(monkeypatch, change):
    original_open = os.open
    original_fdopen = os.fdopen
    paths: dict[int, Path] = {}
    changed: list[Path] = []

    def open_file(path, *args, **kwargs):
        descriptor = original_open(path, *args, **kwargs)
        if str(path).endswith(".json"):
            paths[descriptor] = Path(path)
        return descriptor

    def fdopen(descriptor, *args, **kwargs):
        handle = original_fdopen(descriptor, *args, **kwargs)
        path = paths.pop(descriptor, None)
        if path is None:
            return handle
        native_path: Path = path

        @contextmanager
        def paused():
            with handle:
                yield PausedReader()

        class PausedReader:
            def fileno(self):
                return handle.fileno()

            def read(self, size):
                metadata = os.fstat(handle.fileno())
                if change == "short":
                    return handle.read(size)[:-1]
                first = handle.read(max(1, metadata.st_size // 2))

                def write():
                    raw = native_path.read_bytes()
                    if change == "rewrite":
                        raw = raw.replace(b"before", b"after!")
                    elif change == "truncate":
                        raw = raw[: len(raw) // 2]
                    elif change == "grow":
                        raw += b" "
                    else:
                        raise AssertionError(change)
                    native_path.write_bytes(raw)
                    if change == "rewrite":
                        # Restoring mtime must not hide an in-place rewrite.
                        os.utime(native_path, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
                    changed.append(native_path)

                # The first bytes have been consumed; the rest cannot be read
                # until the writer completes. No timing-dependent sleeps.
                with ThreadPoolExecutor(max_workers=1) as writer:
                    writer.submit(write).result(timeout=5)
                return first + handle.read(size - len(first))

        return paused()

    monkeypatch.setattr(os, "open", open_file)
    monkeypatch.setattr(os, "fdopen", fdopen)
    return changed
