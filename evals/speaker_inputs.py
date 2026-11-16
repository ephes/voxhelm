"""Bounded offline speaker-evaluation controls and frozen array admission.

Errors identify schema locations, never input values or transcript contents.
"""

from __future__ import annotations

import hashlib
import math
import os
import stat
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from typing import Any

from transcriptions.json_input import decode_json_object
from transcriptions.native_output import read_native_json

JSON_MAX_BYTES = 64 * 1024 * 1024
MAX_FRAMES = 1_000_000
MAX_ITEMS = 50_000
MAX_FRAME_VISITS = 5_000_000
MAX_SECONDS = 7200
MAX_WINDOWS = 1000
MAX_ARRAY_BYTES = 512 * 1024 * 1024


def require(ok: bool, location: str) -> None:
    if not ok:
        raise ValueError(f"Invalid evaluation input at {location}")


def file_generation(metadata: os.stat_result) -> tuple[int, ...]:
    # Reads may update access time; only identity, geometry and mutations matter.
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def read_json(path: Path) -> dict[str, Any]:
    try:
        return decode_json_object(read_native_json(path, limit=JSON_MAX_BYTES))
    except (OSError, RuntimeError, ValueError, RecursionError) as exc:
        raise ValueError("Invalid evaluation control file") from exc


def text(value: Any, location: str, *, nullable: bool = False) -> None:
    require(
        (nullable and value is None) or (isinstance(value, str) and 0 < len(value) <= 4096),
        location,
    )


def items(value: Any, location: str, limit: int = MAX_ITEMS) -> list[Any]:
    require(isinstance(value, list) and len(value) <= limit, location)
    return value


def number(value: Any, location: str, maximum: float = MAX_SECONDS) -> float:
    require(type(value) in (int, float), location)
    try:
        result = float(value)
    except OverflowError, ValueError:
        raise ValueError(f"Invalid evaluation input at {location}") from None
    require(math.isfinite(result) and 0 <= result <= maximum, location)
    return result


def interval(value: Any, location: str) -> tuple[float, float]:
    require(isinstance(value, dict), location)
    start = number(value.get("start"), location + ".start")
    end = number(value.get("end"), location + ".end")
    require(end >= start, location)
    return start, end


def validate_truth(value: Any) -> tuple[list[Any], float]:
    require(isinstance(value, dict), "truth")
    fps = number(value.get("fps"), "truth.fps", 1000)
    require(fps > 0, "truth.fps")
    frames = items(value.get("frames"), "truth.frames", MAX_FRAMES)
    for frame in frames:
        text(frame, "truth.frames[]", nullable=True)
    return frames, fps


def validate_sidecar(value: Any, frames: list[Any], fps: float) -> None:
    require(isinstance(value, dict), "sidecar")
    summary = value.get("summary")
    require(isinstance(summary, dict), "sidecar.summary")
    labels = summary.get("raw_diarization_labels", {})
    require(isinstance(labels, dict) and len(labels) <= MAX_ITEMS, "sidecar.summary.labels")
    for label, entry in labels.items():
        text(label, "sidecar.summary.labels[]")
        require(isinstance(entry, dict), "sidecar.summary.labels[]")
        text(entry.get("speaker"), "sidecar.summary.labels[].speaker", nullable=True)
    for name in items(summary.get("known_speakers", []), "sidecar.summary.known_speakers"):
        text(name, "sidecar.summary.known_speakers[]")
    work = 0
    for segment in items(value.get("segments"), "sidecar.segments"):
        start, end = interval(segment, "sidecar.segments[]")
        require(type(segment.get("speaker_uncertain")) is bool, "sidecar.segments[].uncertain")
        for key in (
            "speaker",
            "raw_diarization_speaker",
            "speaker_uncertain_reason",
            "speaker_accept_reason",
        ):
            text(segment.get(key), "sidecar.segments[]." + key, nullable=True)
        require(isinstance(segment.get("text", ""), str), "sidecar.segments[].text")
        work += max(
            0, min(len(frames), math.ceil(end * fps)) - min(len(frames), math.ceil(start * fps))
        )
        require(work <= MAX_FRAME_VISITS, "sidecar.frame_work")


def validate_manifest(value: Any) -> None:
    require(isinstance(value, dict), "manifest")
    text(value.get("audio"), "manifest.audio")
    for key in ("model", "prompt", "language"):
        text(value.get(key), "manifest." + key, nullable=True)
    for name in items(value.get("unreferenced_speakers", []), "manifest.guests", 128):
        text(name, "manifest.guests[]")
    for ref in items(value.get("references"), "manifest.references", 128):
        require(isinstance(ref, dict), "manifest.references[]")
        for key in ("id", "name"):
            text(ref.get(key), "manifest.references[]." + key)
        if "audio" in ref:
            text(ref["audio"], "manifest.references[].audio")
        require(("start" in ref) == ("end" in ref), "manifest.references[].range")
        if "start" in ref:
            start, end = interval(ref, "manifest.references[]")
            require(0 < end - start <= 600, "manifest.references[].range")
    # Reuse the production count contract, without starting a backend.
    from evals.speaker_naming_eval import manifest_diarization_params

    manifest_diarization_params(value)


def validate_bundle(value: Any) -> list[tuple[str, int]]:
    require(
        isinstance(value, dict)
        and type(value.get("version")) is int
        and value["version"] in (1, 2),
        "bundle.version",
    )
    validate_manifest(value.get("manifest"))
    result = value.get("result")
    require(isinstance(result, dict), "bundle.result")
    require(
        set(result) <= {"text", "language", "segments", "backend_name", "model_name", "coverage"},
        "bundle.result.fields",
    )
    for key in ("backend_name", "model_name"):
        require(isinstance(result.get(key, ""), str), "bundle.result." + key)
    coverage = result.get("coverage")
    if coverage is not None:
        keys = {"media_duration_seconds", "covered_seconds", "largest_gap_start", "largest_gap_end"}
        require(isinstance(coverage, dict) and set(coverage) == keys, "bundle.result.coverage")
        for key in keys:
            number(coverage[key], "bundle.result.coverage." + key)
    require(isinstance(result.get("text"), str), "bundle.result.text")
    text(result.get("language"), "bundle.result.language", nullable=True)
    words = 0
    for segment in items(result.get("segments"), "bundle.result.segments"):
        interval(segment, "bundle.result.segments[]")
        require(
            set(segment) <= {"id", "start", "end", "text", "speaker", "words"},
            "bundle.result.segments[].fields",
        )
        require(type(segment.get("id")) is int, "bundle.result.segments[].id")
        require(isinstance(segment.get("text"), str), "bundle.result.segments[].text")
        text(segment.get("speaker"), "bundle.result.segments[].speaker", nullable=True)
        for word in items(segment.get("words", []), "bundle.words"):
            interval(word, "bundle.words[]")
            require(set(word) == {"start", "end", "text"}, "bundle.words[].fields")
            require(isinstance(word.get("text"), str), "bundle.words[].text")
            words += 1
            require(words <= MAX_ITEMS, "bundle.words")
    turns = value.get("turns")
    require(isinstance(turns, dict), "bundle.turns")
    for kind in ("regular", "exclusive"):
        for turn in items(turns.get(kind), "bundle.turns." + kind):
            interval(turn, "bundle.turns[]")
            require(set(turn) == {"start", "end", "speaker"}, "bundle.turns[].fields")
            text(turn.get("speaker"), "bundle.turns[].speaker")
    members = [(value.get("audio"), MAX_SECONDS * 16000)]
    for ref in items(value.get("references"), "bundle.references", 128):
        require(isinstance(ref, dict), "bundle.references[]")
        text(ref.get("id"), "bundle.references[].id")
        text(ref.get("name"), "bundle.references[].name")
        windows = items(ref.get("windows"), "bundle.windows", MAX_WINDOWS)
        require(bool(windows), "bundle.windows")
        members.extend((name, 5 * 16000) for name in windows)
    require(len(members) <= MAX_WINDOWS + 1, "bundle.windows")
    for name, _ in members:
        text(name, "bundle.member")
    return members


def open_member(stack: ExitStack, root_fd: int, name: str) -> Any:
    path = PurePosixPath(name)
    require(
        not path.is_absolute()
        and "\\" not in name
        and all(part not in ("", ".", "..") for part in name.split("/"))
        and path.suffix == ".npy",
        "bundle.member",
    )
    directory = root_fd
    for part in path.parts[:-1]:
        directory = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
        stack.callback(os.close, directory)
    fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    handle = stack.enter_context(os.fdopen(fd, "rb"))
    metadata = os.fstat(fd)
    require(stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1, "bundle.member")
    return handle


def sample_digests(members: list[tuple[str, int]], arrays: list[Any]) -> dict[str, str]:
    return {
        name: hashlib.sha256(array).hexdigest()
        for (name, _), array in zip(members, arrays, strict=True)
    }


def load_arrays(
    root: Path, members: list[tuple[str, int]], *, expected: dict[str, str] | None = None
) -> list[Any]:
    """Inspect every header/aggregate before any allocation, then load the same FDs."""
    import numpy as np

    try:
        with ExitStack() as stack:
            root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            stack.callback(os.close, root_fd)
            handles = []
            total = 0
            for name, max_frames in members:
                handle = open_member(stack, root_fd, name)
                version = np.lib.format.read_magic(handle)
                require(version in ((1, 0), (2, 0)), "bundle.array.version")
                reader = (
                    np.lib.format.read_array_header_1_0
                    if version == (1, 0)
                    else np.lib.format.read_array_header_2_0
                )
                shape, fortran, dtype = reader(handle, max_header_size=4096)
                require(
                    len(shape) == 1
                    and 0 < shape[0] <= max_frames
                    and dtype == np.dtype("float32")
                    and not fortran,
                    "bundle.array.geometry",
                )
                size = shape[0] * dtype.itemsize
                total += size
                require(total <= MAX_ARRAY_BYTES, "bundle.array.aggregate")
                require(
                    os.fstat(handle.fileno()).st_size == handle.tell() + size, "bundle.array.bytes"
                )
                handles.append((handle, size, os.fstat(handle.fileno())))
            arrays = []
            for handle, size, metadata in handles:
                # Fixed-count raw float32 read: never reparse a mutable header during load,
                # and never invoke pickle or allocate from untrusted shape metadata.
                raw = handle.read(size + 1)
                require(
                    len(raw) == size
                    and file_generation(os.fstat(handle.fileno())) == file_generation(metadata),
                    "bundle.array.changed",
                )
                array = np.frombuffer(raw, dtype=np.float32)
                for start in range(0, len(array), 65536):
                    require(
                        bool(np.isfinite(array[start : start + 65536]).all()),
                        "bundle.array.samples",
                    )
                arrays.append(array)
            if expected is not None:
                require(expected == sample_digests(members, arrays), "bundle.sample.identity")
            return arrays
    except (OSError, EOFError, TypeError, ValueError) as exc:
        raise ValueError("Invalid evaluation input at bundle.array") from exc
