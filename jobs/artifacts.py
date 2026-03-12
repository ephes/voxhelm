from __future__ import annotations

import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import boto3
from botocore.config import Config
from django.conf import settings


@dataclass(frozen=True)
class StoredArtifact:
    backend: str
    key: str
    size_bytes: int


class ArtifactStore(Protocol):
    backend_name: str

    def put_file(self, *, key: str, source_path: Path, content_type: str) -> StoredArtifact: ...

    def put_bytes(self, *, key: str, data: bytes, content_type: str) -> StoredArtifact: ...

    def read_bytes(self, *, key: str) -> bytes: ...


class FilesystemArtifactStore:
    backend_name = "filesystem"

    def __init__(self, *, root: Path) -> None:
        self.root = root

    def put_file(self, *, key: str, source_path: Path, content_type: str) -> StoredArtifact:
        del content_type
        target = self.root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target)
        return StoredArtifact(
            backend=self.backend_name,
            key=key,
            size_bytes=target.stat().st_size,
        )

    def put_bytes(self, *, key: str, data: bytes, content_type: str) -> StoredArtifact:
        del content_type
        target = self.root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return StoredArtifact(
            backend=self.backend_name,
            key=key,
            size_bytes=len(data),
        )

    def read_bytes(self, *, key: str) -> bytes:
        return (self.root / key).read_bytes()


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
            config=Config(s3={"addressing_style": "path" if force_path_style else "auto"}),
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

    def read_bytes(self, *, key: str) -> bytes:
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()


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
