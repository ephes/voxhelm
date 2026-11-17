"""Pause a native JSON read while a separate writer changes the admitted file."""

import os
import time
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
                        # Restoring mtime must not hide an in-place rewrite. Keep
                        # restoring until ctime has visibly advanced, so the test
                        # does not depend on the filesystem's timestamp granularity.
                        deadline = time.monotonic() + 2
                        while True:
                            os.utime(native_path, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
                            if os.stat(native_path).st_ctime_ns != metadata.st_ctime_ns:
                                break
                            if time.monotonic() > deadline:
                                raise AssertionError("ctime did not advance")
                            time.sleep(0.001)
                    changed.append(native_path)

                # The writer finishes before the remaining read call returns. The
                # first read may already have buffered the whole small file, so a
                # same-size rewrite is caught by the changed descriptor metadata.
                with ThreadPoolExecutor(max_workers=1) as writer:
                    writer.submit(write).result(timeout=5)
                return first + handle.read(size - len(first))

        return paused()

    monkeypatch.setattr(os, "open", open_file)
    monkeypatch.setattr(os, "fdopen", fdopen)
    return changed
