"""A fake window runtime for profile tests: ``whisper-cli``, ffmpeg and a pinned model.

The fake ffmpeg writes the clip start (``-ss``) into the clip file; the fake whisper-cli
reads it back and answers with the language and probability configured for that clip
start (``languages``, keyed by the start formatted as ffmpeg got it, ``"default"``
otherwise) and with the configured tokens (offsets relative to the clip). Each call is
appended to a log, so a test sees which windows were retried.

Called as the chapter windows call it (``-dtw``, the clip as the last argument), the fake
whisper-cli logs ``chapter <clip start>`` and its arguments, exits with the configured
status for that clip start (``chapter_fail``) or writes ``-oj`` JSON with one segment per
configured word (``chapter_segments``, offsets relative to the clip; ``REPLACEMENT``
becomes a cut UTF-8 sequence, ``CONTROL`` a raw control character); ``chapter_sleep``
makes it wait first and ``chapter_remove_model`` deletes the model before failing. The
fake ffprobe prints the configured duration (``duration``) in seconds.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

from transcriptions import windows

TURBO_WEIGHTS = b"turbo weights"
FAKE_WHISPER = """#!{python}
import json, sys
from pathlib import Path

args = sys.argv[1:]
if args == ["--version"]:
    print("whisper.cpp version: 1.9.5")
    sys.exit(0)
config = json.loads(Path({config!r}).read_text())
if "-dtw" in args:
    import time
    start = Path(args[-1]).read_text().strip()
    with open({chapter_calls!r}, "a") as log:
        log.write(json.dumps([start, args]) + "\\n")
    time.sleep(config["chapter_sleep"])
    if config["chapter_remove_model"]:
        Path(args[args.index("-m") + 1]).unlink()
        sys.exit(1)
    status = config["chapter_fail"].get(start, 0)
    if status:
        sys.exit(status)
    segments = [
        {{"timestamps": {{}}, "offsets": {{"from": a, "to": b}}, "text": text}}
        for text, a, b in config["chapter_segments"]
    ]
    base = args[args.index("-of") + 1]
    Path(base + ".json").write_bytes(
        json.dumps({{"result": {{"language": "en"}}, "transcription": segments}}).encode()
        .replace(b"REPLACEMENT", b"\\xe2\\x82")
        .replace(b"CONTROL", b"\\x01")
    )
    sys.exit(0)
start = Path(args[args.index("-f") + 1]).read_text().strip()
with open({calls!r}, "a") as log:
    log.write(start + "\\n")
code, p = config["languages"].get(start, config["languages"]["default"])
if code == "fail":
    print("whisper_full: failed to decode", file=sys.stderr)
    sys.exit(3)
print(f"whisper_full_with_state: auto-detected language: {{code}} (p = {{p:.6f}})", file=sys.stderr)
tokens = [
    {{"text": text, "offsets": {{"from": a, "to": b}}, "id": 1000, "p": 0.9}}
    for text, a, b in config["tokens"]
]
base = args[args.index("-of") + 1]
Path(base + ".json").write_text(
    json.dumps({{"result": {{"language": code}}, "transcription": [{{"tokens": tokens}}]}})
)
"""
FAKE_FFMPEG = """#!{python}
import sys
from pathlib import Path

args = sys.argv[1:]
Path(args[-1]).write_text(args[args.index("-ss") + 1])
"""
FAKE_FFPROBE = """#!{python}
import json, sys
from pathlib import Path

config = json.loads(Path({config!r}).read_text())
with open({calls!r}, "a") as log:
    log.write(json.dumps(sys.argv[1:]) + "\\n")
print(config["duration"])
"""
# Relative to the clip start (t - 1 for t >= 1): file times t + 0.0 ... t + 1.4.
DEFAULT_TOKENS: list[tuple[str, int, int]] = [
    (" Das", 1000, 1300),
    (" ist", 1400, 1600),
    (" gut.", 2000, 2400),
]
# One word per segment (``-ml 1 -sow``), offsets relative to the clip start: a blank
# segment (dropped), a word split inside a multi-byte character (decoded with the
# replacement character) and whitespace around words (stripped).
DEFAULT_CHAPTER_SEGMENTS: list[tuple[str, int, int]] = [
    (" Thanks", 59100, 59380),
    (" ", 59380, 59400),
    (" to", 59400, 59520),
    (" REPLACEMENT", 59520, 59600),
    (" Linode.", 59600, 60250),
]


class FakeWindowRuntime:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.model = root / "models" / "ggml-large-v3-turbo.bin"
        self.config_path = root / "whisper-config.json"
        self.calls_path = root / "whisper-calls.txt"
        self.chapter_calls_path = root / "chapter-calls.txt"
        self.ffprobe_calls_path = root / "ffprobe-calls.txt"
        self.temp_root = root / "tmp"
        self.languages: dict[str, list[Any]] = {"default": ["de", 0.999549]}
        self.tokens: list[tuple[str, int, int]] = list(DEFAULT_TOKENS)
        self.duration = "7200.000000"
        self.chapter_fail: dict[str, int] = {}
        self.chapter_segments: list[tuple[str, int, int]] = list(DEFAULT_CHAPTER_SEGMENTS)
        self.chapter_sleep = 0.0
        self.chapter_remove_model = False
        self.write_config()

    def write_config(self) -> None:
        self.config_path.write_text(
            json.dumps(
                {
                    "languages": self.languages,
                    "tokens": self.tokens,
                    "duration": self.duration,
                    "chapter_fail": self.chapter_fail,
                    "chapter_segments": self.chapter_segments,
                    "chapter_sleep": self.chapter_sleep,
                    "chapter_remove_model": self.chapter_remove_model,
                }
            ),
            encoding="utf-8",
        )

    def chapter_calls(self) -> list[tuple[str, list[str]]]:
        """(clip start, arguments) of every chapter window whisper-cli run, in order."""
        if not self.chapter_calls_path.exists():
            return []
        lines = self.chapter_calls_path.read_text(encoding="utf-8").splitlines()
        calls: list[tuple[str, list[str]]] = []
        for line in lines:
            start, args = json.loads(line)
            calls.append((start, args))
        return calls

    def ffprobe_calls(self) -> list[list[str]]:
        if not self.ffprobe_calls_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.ffprobe_calls_path.read_text(encoding="utf-8").splitlines()
        ]

    def set_language(self, clip_start: str, code: str, probability: float) -> None:
        """Answer ``code`` with ``probability`` for the clip starting at ``clip_start``."""
        self.languages[clip_start] = [code, probability]
        self.write_config()

    def clip_starts(self) -> list[str]:
        """The clip starts whisper-cli was run on, in order."""
        if not self.calls_path.exists():
            return []
        return self.calls_path.read_text(encoding="utf-8").split()


def install_fake_window_runtime(
    settings: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> FakeWindowRuntime:
    root = tmp_path / "window-runtime"
    (root / "models").mkdir(parents=True)
    runtime = FakeWindowRuntime(root)
    runtime.model.write_bytes(TURBO_WEIGHTS)
    # The deployment's batch whisper.cpp model sits next to it and must never be used.
    (root / "models" / "ggml-large-v3.bin").write_bytes(b"large-v3 weights")
    monkeypatch.setattr(windows, "WINDOW_MODEL_SHA256", hashlib.sha256(TURBO_WEIGHTS).hexdigest())
    cli = root / "whisper-cli"
    cli.write_text(
        FAKE_WHISPER.format(
            python=sys.executable,
            calls=str(runtime.calls_path),
            chapter_calls=str(runtime.chapter_calls_path),
            config=str(runtime.config_path),
        ),
        encoding="utf-8",
    )
    cli.chmod(0o755)
    ffprobe = root / "ffprobe"
    ffprobe.write_text(
        FAKE_FFPROBE.format(
            python=sys.executable,
            calls=str(runtime.ffprobe_calls_path),
            config=str(runtime.config_path),
        ),
        encoding="utf-8",
    )
    ffprobe.chmod(0o755)
    ffmpeg = root / "ffmpeg"
    ffmpeg.write_text(FAKE_FFMPEG.format(python=sys.executable), encoding="utf-8")
    ffmpeg.chmod(0o755)
    runtime.temp_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(runtime.temp_root))
    settings.VOXHELM_WHISPERCPP_BIN = str(cli)
    settings.VOXHELM_WHISPERCPP_MODEL = "ggml-large-v3.bin"
    settings.VOXHELM_MODEL_CACHE_DIR = root / "models"
    settings.VOXHELM_WINDOW_MODEL = "ggml-large-v3-turbo.bin"
    settings.VOXHELM_FFMPEG_BIN = str(ffmpeg)
    settings.VOXHELM_FFPROBE_BIN = str(ffprobe)
    windows._model_sha256.cache_clear()
    windows._whispercpp_version.cache_clear()
    return runtime


def clear_window_caches() -> None:
    windows._model_sha256.cache_clear()
    windows._whispercpp_version.cache_clear()
