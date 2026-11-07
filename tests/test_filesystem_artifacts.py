from __future__ import annotations

import os
import stat
from typing import Any

import pytest

from jobs.artifacts import FilesystemArtifactStore


@pytest.mark.parametrize(
    "operation",
    ["put_bytes", "put_file", "read_bytes", "download_file", "stat", "delete", "list_keys"],
)
@pytest.mark.parametrize("alias", ["absolute", "parent", "parent_alias", "leaf_alias"])
def test_every_operation_refuses_escaping_keys(tmp_path, operation, alias):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    canary = outside / "canary"
    canary.write_bytes(b"private")
    (root / "parent_alias").symlink_to(outside, target_is_directory=True)
    (root / "leaf_alias").symlink_to(canary)
    key = {
        "absolute": str(canary),
        "parent": "../outside/canary",
        "parent_alias": "parent_alias/canary",
        "leaf_alias": "leaf_alias",
    }[alias]
    store = FilesystemArtifactStore(root=root)
    kwargs: dict[str, Any] = {"key": key}
    if operation == "list_keys":
        kwargs = {"prefix": key}
    elif operation == "put_bytes":
        kwargs.update(data=b"replacement", content_type="text/plain")
    elif operation == "put_file":
        kwargs.update(source_path=canary, content_type="text/plain")
    elif operation == "download_file":
        kwargs.update(destination_path=tmp_path / "download")
    with pytest.raises((ValueError, OSError)):
        getattr(store, operation)(**kwargs)
    assert canary.read_bytes() == b"private"
    assert not (tmp_path / "download").exists()


@pytest.mark.parametrize("method", ["put_bytes", "put_file"])
def test_failed_publish_preserves_previous_bytes_and_cleans_temp(tmp_path, monkeypatch, method):
    store = FilesystemArtifactStore(root=tmp_path / "root")
    store.put_bytes(key="job/transcript", data=b"previous", content_type="text/plain")
    source = tmp_path / "source"
    source.write_bytes(b"replacement")
    if method == "put_file":

        def fail_copy(source, target):
            target.write(b"partial")
            assert store.read_bytes(key="job/transcript") == b"previous"
            raise OSError("synthetic copy failure")

        monkeypatch.setattr("shutil.copyfileobj", fail_copy)
    else:

        def fail_flush(fd):
            assert store.read_bytes(key="job/transcript") == b"previous"
            raise OSError("synthetic flush failure")

        monkeypatch.setattr(os, "fsync", fail_flush)
    kwargs = {"source_path": source} if method == "put_file" else {"data": b"replacement"}
    with pytest.raises(OSError):
        getattr(store, method)(key="job/transcript", content_type="text/plain", **kwargs)
    assert store.read_bytes(key="job/transcript") == b"previous"
    assert store.list_keys(prefix="job/") == ["job/transcript"]


def test_publication_is_private_and_readers_see_only_complete_objects(tmp_path, monkeypatch):
    root = tmp_path / "root"
    store = FilesystemArtifactStore(root=root)
    store.put_bytes(key="job/transcript", data=b"previous", content_type="text/plain")
    old_replace = os.replace
    observed = []

    def replace(source, target, **kwargs):
        observed.append(store.read_bytes(key="job/transcript"))
        assert stat.S_IMODE(os.stat(source, dir_fd=kwargs["src_dir_fd"]).st_mode) == 0o600
        old_replace(source, target, **kwargs)

    monkeypatch.setattr(os, "replace", replace)
    store.put_bytes(key="job/transcript", data=b"complete replacement", content_type="text/plain")
    assert observed == [b"previous"]
    assert store.read_bytes(key="job/transcript") == b"complete replacement"
    for path, mode in [(root, 0o700), (root / "job", 0o700), (root / "job/transcript", 0o600)]:
        assert stat.S_IMODE(path.stat().st_mode) == mode
    destination = tmp_path / "download"
    store.download_file(key="job/transcript", destination_path=destination)
    assert destination.read_bytes() == b"complete replacement"
    assert store.stat(key="job/transcript").size_bytes == len(destination.read_bytes())
    store.delete(key="job/transcript")
    store.delete(key="job/transcript")
    assert store.list_keys(prefix="job/") == []


def test_trusted_root_alias_works(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    store = FilesystemArtifactStore(root=alias)
    store.put_bytes(key="job/file", data=b"private", content_type="x")
    assert store.read_bytes(key="job/file") == b"private"
    assert store.list_keys(prefix="job/") == ["job/file"]
    store.delete(key="job/file")


def test_disappearing_entry_during_listing_is_not_reported_as_empty(tmp_path, monkeypatch):
    store = FilesystemArtifactStore(root=tmp_path)
    store.put_bytes(key="job/retained", data=b"keep", content_type="x")

    def disappearing_walk(*args, **kwargs):
        kwargs["onerror"](FileNotFoundError("synthetic disappearing child"))
        return iter(())

    monkeypatch.setattr(os, "fwalk", disappearing_walk)
    with pytest.raises(FileNotFoundError):
        store.list_keys(prefix="job/")
    assert store.read_bytes(key="job/retained") == b"keep"


@pytest.mark.parametrize("failure", ["raw", "buffer", "buffer_after_close", "fstat"])
def test_reader_factory_closes_descriptor_on_initialization_failure(monkeypatch, tmp_path, failure):
    import io
    import os

    from jobs.artifacts import FilesystemArtifactStore

    store = FilesystemArtifactStore(root=tmp_path)
    store.put_bytes(key="synthetic.bin", data=b"synthetic", content_type="application/octet-stream")
    real_raw, real_buffer, real_fstat = io.FileIO, io.BufferedReader, os.fstat
    descriptors, readers, replacements = [], [], []

    def raw_open(fd, *args, **kwargs):
        descriptors.append(fd)
        if failure == "raw":
            raise OSError("synthetic raw failure")
        raw = real_raw(fd, *args, **kwargs)
        readers.append(raw)
        return raw

    def buffered(raw, *args, **kwargs):
        if failure == "buffer_after_close":
            raw.close()
            # A constructor can close the raw object before it raises. A later
            # descriptor allocation must not be closed by numeric fd cleanup.
            replacements.append(real_raw(tmp_path / "synthetic.bin", "rb"))
            raise OSError("synthetic buffer_after_close failure")
        if failure == "buffer":
            raise OSError("synthetic buffer failure")
        return real_buffer(raw, *args, **kwargs)

    def fstat(fd):
        raise OSError("synthetic fstat failure")

    monkeypatch.setattr(io, "FileIO", raw_open)
    monkeypatch.setattr(io, "BufferedReader", buffered)
    if failure == "fstat":
        monkeypatch.setattr(os, "fstat", fstat)
    try:
        with pytest.raises(OSError, match=failure):
            store.open_reader(key="synthetic.bin")
        assert len(descriptors) == 1
        assert all(reader.closed for reader in readers)
        if replacements:
            assert real_fstat(replacements[0].fileno()).st_size == len(b"synthetic")
        else:
            with pytest.raises(OSError):
                real_fstat(descriptors[0])
    finally:
        for replacement in replacements:
            replacement.close()
