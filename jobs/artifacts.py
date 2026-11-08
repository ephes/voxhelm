from __future__ import annotations

import io
import logging
import os
import shutil
import stat
import uuid
from collections.abc import Iterator
from contextlib import ExitStack, closing, contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, BinaryIO, Protocol

import boto3
from botocore.config import Config
from django.conf import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StoredArtifact:
    backend: str
    key: str
    size_bytes: int


class ArtifactStore(Protocol):
    backend_name: str

    def put_file(self, *, key: str, source_path: Path, content_type: str) -> StoredArtifact: ...

    def put_bytes(self, *, key: str, data: bytes, content_type: str) -> StoredArtifact: ...

    def open_reader(self, *, key: str) -> BinaryIO: ...

    def read_bytes(self, *, key: str) -> bytes: ...

    def download_file(self, *, key: str, destination_path: Path) -> None: ...

    def stat(self, *, key: str) -> StoredArtifact: ...

    def delete(self, *, key: str) -> None: ...

    def list_keys(self, *, prefix: str) -> list[str]: ...


class FilesystemArtifactStore:
    backend_name = "filesystem"

    def __init__(self, *, root: Path) -> None:
        self.root = root.expanduser().resolve()

    @staticmethod
    def _parts(key: str) -> list[str]:
        parts = key.split("/")
        if any(part in {"", ".", ".."} for part in parts) or "\x00" in key:
            raise ValueError("Artifact keys must be nonempty relative paths without dot segments.")
        return parts

    @contextmanager
    def _parent(self, key: str, *, create: bool = False) -> Iterator[tuple[int, str]]:
        parts = self._parts(key)
        # The configured root and its ancestors are trusted administrator-owned paths.
        if create:
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for part in parts[:-1]:
                if create:
                    try:
                        os.mkdir(part, mode=0o700, dir_fd=directory)
                    except FileExistsError:
                        pass
                child = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
                )
                os.close(directory)
                directory = child
            yield directory, parts[-1]
        finally:
            os.close(directory)

    @staticmethod
    def _check_leaf(directory: int, name: str) -> None:
        try:
            mode = os.stat(name, dir_fd=directory, follow_symlinks=False).st_mode
        except FileNotFoundError:
            return
        if not stat.S_ISREG(mode):
            raise ValueError("Artifact objects must be regular files, not aliases or directories.")

    def open_reader(self, *, key: str) -> BinaryIO:
        with self._parent(key) as (directory, name):
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            try:
                raw = io.FileIO(fd, "rb", closefd=True)
            except BaseException:
                os.close(fd)
                raise
            try:
                source = io.BufferedReader(raw)
                if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                    raise ValueError("Artifact objects must be regular files.")
            except BaseException:
                raw.close()
                raise
            return source

    @contextmanager
    def _reader(self, key: str) -> Iterator[BinaryIO]:
        with self.open_reader(key=key) as source:
            yield source

    @contextmanager
    def _writer(self, key: str) -> Iterator[BinaryIO]:
        with self._parent(key, create=True) as (directory, name):
            self._check_leaf(directory, name)
            temporary = f".artifact-{uuid.uuid4().hex}.tmp"
            fd = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=directory,
            )
            primary_error = False
            try:
                with os.fdopen(fd, "wb") as target:
                    yield target
                    target.flush()
                    os.fsync(target.fileno())
                self._check_leaf(directory, name)
                os.replace(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
            except BaseException:
                primary_error = True
                raise
            finally:
                try:
                    os.unlink(temporary, dir_fd=directory)
                except FileNotFoundError:
                    pass
                except OSError:
                    if not primary_error:
                        raise
                    logger.warning("Artifact temporary cleanup failed after publication failure.")

    def put_file(self, *, key: str, source_path: Path, content_type: str) -> StoredArtifact:
        del content_type
        with source_path.open("rb") as source, self._writer(key) as target:
            shutil.copyfileobj(source, target)
            size = target.tell()
        return StoredArtifact(backend=self.backend_name, key=key, size_bytes=size)

    def put_bytes(self, *, key: str, data: bytes, content_type: str) -> StoredArtifact:
        del content_type
        with self._writer(key) as target:
            target.write(data)
        return StoredArtifact(backend=self.backend_name, key=key, size_bytes=len(data))

    def read_bytes(self, *, key: str) -> bytes:
        with self._reader(key) as source:
            return source.read()

    def download_file(self, *, key: str, destination_path: Path) -> None:
        with self._reader(key) as source:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            with destination_path.open("wb") as destination:
                shutil.copyfileobj(source, destination)

    def stat(self, *, key: str) -> StoredArtifact:
        with self._reader(key) as source:
            size = os.fstat(source.fileno()).st_size
        return StoredArtifact(backend=self.backend_name, key=key, size_bytes=size)

    def delete(self, *, key: str) -> None:
        try:
            with self._parent(key) as (directory, name):
                self._check_leaf(directory, name)
                os.unlink(name, dir_fd=directory)
        except FileNotFoundError:
            pass

    def list_keys(self, *, prefix: str) -> list[str]:
        """List aliases for retention to refuse, without following them."""
        base = prefix.removesuffix("/")
        self._parts(base)
        keys: list[str] = []

        def raise_listing_error(error: OSError) -> None:
            raise error

        with ExitStack() as stack:
            try:
                directory, _ = stack.enter_context(self._parent(f"{base}/placeholder"))
            except FileNotFoundError:
                return []
            for current, dirnames, filenames, current_fd in os.fwalk(
                ".", dir_fd=directory, follow_symlinks=False, onerror=raise_listing_error
            ):
                relative = Path(current).parts
                names = [
                    *filenames,
                    *(
                        name
                        for name in dirnames
                        if stat.S_ISLNK(
                            os.stat(name, dir_fd=current_fd, follow_symlinks=False).st_mode
                        )
                    ),
                ]
                for name in names:
                    keys.append("/".join((base, *relative, name)))
        return sorted(keys)


class S3ArtifactStore:
    backend_name = "s3"

    def __init__(
        self,
        *,
        endpoint_url: str,
        region_name: str,
        access_key_id: str,
        secret_access_key: str,
        bucket: str,
        force_path_style: bool,
    ) -> None:
        self.bucket = bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region_name,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            config=Config(
                read_timeout=60,
                s3={"addressing_style": "path" if force_path_style else "auto"},
            ),
        )

    def put_file(self, *, key: str, source_path: Path, content_type: str) -> StoredArtifact:
        self.client.upload_file(
            str(source_path),
            self.bucket,
            key,
            ExtraArgs={"ContentType": content_type},
        )
        return StoredArtifact(
            backend=self.backend_name,
            key=key,
            size_bytes=source_path.stat().st_size,
        )

    def put_bytes(self, *, key: str, data: bytes, content_type: str) -> StoredArtifact:
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        return StoredArtifact(
            backend=self.backend_name,
            key=key,
            size_bytes=len(data),
        )

    def open_reader(self, *, key: str) -> BinaryIO:
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        return response["Body"]

    def read_bytes(self, *, key: str) -> bytes:
        with closing(self.open_reader(key=key)) as source:
            return source.read()

    def download_file(self, *, key: str, destination_path: Path) -> None:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(destination_path))

    def stat(self, *, key: str) -> StoredArtifact:
        response = self.client.head_object(Bucket=self.bucket, Key=key)
        return StoredArtifact(
            backend=self.backend_name,
            key=key,
            size_bytes=int(response["ContentLength"]),
        )

    def delete(self, *, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def list_keys(self, *, prefix: str) -> list[str]:
        keys: list[str] = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            keys.extend(str(item["Key"]) for item in page.get("Contents", []))
        return sorted(keys)


@lru_cache(maxsize=1)
def get_artifact_store() -> ArtifactStore:
    backend = settings.VOXHELM_ARTIFACT_BACKEND
    if backend == "filesystem":
        return FilesystemArtifactStore(root=settings.VOXHELM_ARTIFACT_ROOT)
    if backend == "s3":
        required = {
            "VOXHELM_ARTIFACT_S3_ENDPOINT_URL": settings.VOXHELM_ARTIFACT_S3_ENDPOINT_URL,
            "VOXHELM_ARTIFACT_S3_ACCESS_KEY_ID": settings.VOXHELM_ARTIFACT_S3_ACCESS_KEY_ID,
            "VOXHELM_ARTIFACT_S3_SECRET_ACCESS_KEY": settings.VOXHELM_ARTIFACT_S3_SECRET_ACCESS_KEY,
            "VOXHELM_ARTIFACT_BUCKET": settings.VOXHELM_ARTIFACT_BUCKET,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            joined = ", ".join(sorted(missing))
            raise RuntimeError(f"S3 artifact backend is missing configuration: {joined}.")
        return S3ArtifactStore(
            endpoint_url=settings.VOXHELM_ARTIFACT_S3_ENDPOINT_URL,
            region_name=settings.VOXHELM_ARTIFACT_S3_REGION,
            access_key_id=settings.VOXHELM_ARTIFACT_S3_ACCESS_KEY_ID,
            secret_access_key=settings.VOXHELM_ARTIFACT_S3_SECRET_ACCESS_KEY,
            bucket=settings.VOXHELM_ARTIFACT_BUCKET,
            force_path_style=settings.VOXHELM_ARTIFACT_S3_FORCE_PATH_STYLE,
        )
    raise RuntimeError(f"Unsupported artifact backend '{backend}'.")


def current_artifact_store_identity() -> dict[str, Any]:
    backend = settings.VOXHELM_ARTIFACT_BACKEND
    if backend == "filesystem":
        return {
            "backend": backend,
            "root": str(settings.VOXHELM_ARTIFACT_ROOT.expanduser().resolve()),
        }
    if backend == "s3":
        return {
            "backend": backend,
            "endpoint_url": settings.VOXHELM_ARTIFACT_S3_ENDPOINT_URL,
            "region": settings.VOXHELM_ARTIFACT_S3_REGION,
            "bucket": settings.VOXHELM_ARTIFACT_BUCKET,
            "force_path_style": settings.VOXHELM_ARTIFACT_S3_FORCE_PATH_STYLE,
        }
    raise RuntimeError(f"Unsupported artifact backend '{backend}'.")


def get_artifact_store_for_identity(identity: dict[str, Any] | None) -> ArtifactStore:
    if not identity or identity == current_artifact_store_identity():
        return get_artifact_store()
    backend = identity.get("backend")
    if backend == "filesystem":
        root = identity.get("root")
        if not isinstance(root, str) or not root:
            raise RuntimeError("Filesystem artifact store identity is missing root.")
        return FilesystemArtifactStore(root=Path(root))
    if backend == "s3":
        required = {
            "VOXHELM_ARTIFACT_S3_ACCESS_KEY_ID": settings.VOXHELM_ARTIFACT_S3_ACCESS_KEY_ID,
            "VOXHELM_ARTIFACT_S3_SECRET_ACCESS_KEY": settings.VOXHELM_ARTIFACT_S3_SECRET_ACCESS_KEY,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            joined = ", ".join(sorted(missing))
            raise RuntimeError(f"S3 artifact backend is missing configuration: {joined}.")
        endpoint_url = identity.get("endpoint_url")
        bucket = identity.get("bucket")
        region = identity.get("region") or settings.VOXHELM_ARTIFACT_S3_REGION
        if not isinstance(endpoint_url, str) or not endpoint_url:
            raise RuntimeError("S3 artifact store identity is missing endpoint_url.")
        if not isinstance(bucket, str) or not bucket:
            raise RuntimeError("S3 artifact store identity is missing bucket.")
        return S3ArtifactStore(
            endpoint_url=endpoint_url,
            region_name=str(region),
            access_key_id=settings.VOXHELM_ARTIFACT_S3_ACCESS_KEY_ID,
            secret_access_key=settings.VOXHELM_ARTIFACT_S3_SECRET_ACCESS_KEY,
            bucket=bucket,
            force_path_style=identity.get("force_path_style") is not False,
        )
    raise RuntimeError(f"Unsupported artifact backend '{backend}'.")
