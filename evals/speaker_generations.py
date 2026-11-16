"""Content identities and cooperating local publishers for offline evidence."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from evals.speaker_inputs import file_generation, items, read_json, require, text, validate_manifest

GENERATION_SETTINGS = (
    "VOXHELM_STT_BACKEND",
    "VOXHELM_STT_FALLBACK_BACKEND",
    "VOXHELM_KNOWN_SPEAKER_STT_BACKEND",
    "VOXHELM_MLX_MODEL",
    "VOXHELM_WHISPERCPP_MODEL",
    "VOXHELM_PARAKEET_MODEL_DIR",
    "VOXHELM_PARAKEET_MODEL_NAME",
    "VOXHELM_WHISPERKIT_MODEL",
    "VOXHELM_PYANNOTE_MODEL",
    "VOXHELM_PYANNOTE_DEVICE",
    "VOXHELM_FFMPEG_BIN",
    "VOXHELM_AUDIO_DECODE_TIMEOUT_SECONDS",
    "VOXHELM_MLX_CONDITION_ON_PREVIOUS_TEXT",
    "VOXHELM_WHISPERCPP_BIN",
    "VOXHELM_WHISPERCPP_PROCESSORS",
    "VOXHELM_WHISPERCPP_MAX_CONTEXT",
    "VOXHELM_WHISPERCPP_SUPPRESS_NST",
    "VOXHELM_WHISPERCPP_MAX_GAP_SECONDS",
    "VOXHELM_WHISPERKIT_ENABLED",
    "VOXHELM_WHISPERKIT_BASE_URL",
    "VOXHELM_WHISPERKIT_TIMEOUT_SECONDS",
    "VOXHELM_PARAKEET_BIN",
    "VOXHELM_PARAKEET_CHUNK_SECONDS",
    "VOXHELM_PARAKEET_OVERLAP_SECONDS",
    "VOXHELM_PARAKEET_MAX_SENTENCE_SECONDS",
    "VOXHELM_PARAKEET_TIMEOUT_SECONDS",
    "VOXHELM_SANITIZE_TRANSCRIPT",
    "VOXHELM_SANITIZE_REPEAT_THRESHOLD",
)

# Capture-path modules (ASR, diarization, decode, sanitizer). known_speaker.py is
# the postprocessing under comparison: its reference/job decode outputs are frozen
# as sample-sealed arrays, so editing it must still replay the same bundles.
CAPTURE_CODE = (
    "audio_decode.py",
    "coverage.py",
    "diarization.py",
    "errors.py",
    "formats.py",
    "json_input.py",
    "native_output.py",
    "owned_process.py",
    "sanitizer.py",
    "service.py",
)


def json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def file_digest(path: Path) -> str:
    """Hash a stable opened regular file without following the final symlink."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(fd)
            require(stat.S_ISREG(before.st_mode), "identity.file")
            digest = hashlib.sha256()
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
            require(
                file_generation(os.fstat(fd)) == file_generation(before)
                and file_generation(path.stat()) == file_generation(before),
                "identity.changed",
            )
            return digest.hexdigest()
    except OSError as exc:
        raise ValueError("Invalid evaluation identity file") from exc


def source_identity(manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Explicit local model artifacts are required; never resolve/download models."""
    from django.conf import settings

    validate_manifest(manifest)
    generation = manifest.get("generation_files")
    require(
        isinstance(generation, dict) and set(generation) == {"asr", "diarization"},
        "identity.generation_files",
    )
    assert isinstance(generation, dict)
    models: dict[str, Any] = {}
    for role in ("asr", "diarization"):
        files = items(generation[role], "identity.generation_files." + role, 128)
        require(bool(files), "identity.generation_files." + role)
        for name in files:
            text(name, "identity.generation_files[]")
        models[role] = {name: file_digest(manifest_path.parent / name) for name in files}
    names = {
        manifest["audio"],
        *(ref.get("audio", manifest["audio"]) for ref in manifest["references"]),
    }
    sources = {name: file_digest(manifest_path.parent / name) for name in sorted(names)}
    packages: dict[str, str | None] = {}
    for name in ("numpy", "torch", "pyannote.audio", "mlx-whisper", "transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    code = Path(__file__).resolve().parent.parent / "transcriptions"
    return {
        "manifest_sha256": json_digest(manifest),
        "sources": sources,
        "models": models,
        "settings_sha256": json_digest(
            {name: str(getattr(settings, name, "")) for name in GENERATION_SETTINGS}
        ),
        "packages": packages,
        "code": {name: file_digest(code / name) for name in CAPTURE_CODE},
    }


def verify_bundle_identity(manifest_path: Path, bundle: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    frozen = read_json(bundle / "bundle.json")
    require(
        frozen.get("version") == 2
        and frozen.get("manifest") == manifest
        and frozen.get("identity") == source_identity(manifest_path, manifest),
        "bundle.identity (use a new bundle directory on mismatch)",
    )
    return frozen["identity"]


@contextmanager
def publisher(parent: Path, name: str):
    """One nonblocking publisher per name; OS releases ownership on process death."""
    parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(
        parent / f".{name}.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600
    )
    try:
        require(stat.S_ISREG(os.fstat(fd).st_mode), "publisher.lock")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Evaluation publisher already active") from None
        yield
    finally:
        os.close(fd)
