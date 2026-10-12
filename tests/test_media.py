from __future__ import annotations

import sys
from pathlib import Path

import pytest

from jobs import media
from jobs.media import extract_audio_from_video, write_uploaded_media_to_tempfile


def test_extract_audio_from_video_handles_non_utf8_ffmpeg_stderr(tmp_path: Path, settings) -> None:
    fake_ffmpeg = tmp_path / "fake-ffmpeg.py"
    fake_ffmpeg.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "sys.stderr.buffer.write(b'bad byte: \\xf0')\n"
        "raise SystemExit(1)\n",
        encoding="utf-8",
    )
    fake_ffmpeg.chmod(0o755)
    source_path = tmp_path / "sample.mov"
    source_path.write_bytes(b"not-really-video")
    settings.VOXHELM_FFMPEG_BIN = str(fake_ffmpeg)

    try:
        extract_audio_from_video(source_path=source_path)
    except RuntimeError as exc:
        assert str(exc) == "ffmpeg audio extraction failed: bad byte: �"
    else:  # pragma: no cover
        raise AssertionError("Expected ffmpeg extraction failure to raise RuntimeError.")


@pytest.fixture
def reserved_paths(monkeypatch, tmp_path: Path) -> list[Path]:
    reserved: list[Path] = []

    def reserve(*, suffix: str) -> Path:
        path = tmp_path / f"upload-{len(reserved)}{suffix}"
        path.touch()
        reserved.append(path)
        return path

    monkeypatch.setattr(media, "reserve_temp_media_path", reserve)
    return reserved


def test_write_uploaded_media_keeps_tempfile_for_caller_on_success(
    reserved_paths: list[Path],
) -> None:
    path = write_uploaded_media_to_tempfile(iter([b"abc", b"def"]), suffix=".mp3")

    assert reserved_paths == [path]
    assert path.read_bytes() == b"abcdef"


def test_write_uploaded_media_removes_tempfile_when_chunk_stream_fails(
    reserved_paths: list[Path],
) -> None:
    def chunks():
        yield b"private-audio"
        raise ConnectionResetError("client went away")

    with pytest.raises(ConnectionResetError, match="client went away"):
        write_uploaded_media_to_tempfile(chunks(), suffix=".mp3")

    assert len(reserved_paths) == 1
    assert not reserved_paths[0].exists()


def test_write_uploaded_media_removes_tempfile_when_write_fails(
    monkeypatch, reserved_paths: list[Path]
) -> None:
    handles = []
    real_open = Path.open

    class FailingHandle:
        def __init__(self, handle) -> None:
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *exc_info) -> None:
            self.close()

        def close(self) -> None:
            self.handle.close()

        @property
        def name(self) -> str:
            return self.handle.name

        def write(self, data: bytes) -> int:
            raise OSError(28, "No space left on device")

    def failing_open(self, *args, **kwargs):
        handle = real_open(self, *args, **kwargs)
        handles.append(handle)
        return FailingHandle(handle)

    monkeypatch.setattr(Path, "open", failing_open)

    with pytest.raises(OSError, match="No space left on device"):
        write_uploaded_media_to_tempfile(iter([b"private-audio"]), suffix=".mp3")

    monkeypatch.undo()
    assert len(handles) == 1
    assert handles[0].closed
    assert not (handles[0].name and Path(handles[0].name).exists())
