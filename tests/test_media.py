from __future__ import annotations

import sys
from pathlib import Path

from jobs.media import extract_audio_from_video


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
