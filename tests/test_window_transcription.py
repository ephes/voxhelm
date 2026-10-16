"""Window transcription (whisper large-v3-turbo on one clip): pinned model and identity,
no fallback, nothing left behind, and word assembly identical to the research runner."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

import pytest
from django_tasks_db.models import DBTaskResult

from jobs.models import Job, JobArtifact, StagedMedia
from lane_scheduler import InferenceCancelled
from transcriptions import windows
from transcriptions.service import BackendUnavailableError
from transcriptions.windows import (
    WINDOW_SETTINGS,
    detected_language,
    transcribe_window,
    window_runtime_available,
    window_words,
)

TRANSCRIPT_MARKER = "Geheimer Transkripttext"
FAKE_WHISPER = """#!{python}
import json, sys
from pathlib import Path

args = sys.argv[1:]
if args == ["--version"]:
    print("whisper.cpp version: 1.9.5")
    sys.exit(0)
Path({record!r}).write_text(json.dumps(args))
mode = Path({mode!r}).read_text().strip()
print("[00:00:00.000 --> 00:00:02.000]   {marker}")
if mode == "fail":
    print("whisper_init: failed to load model", file=sys.stderr)
    sys.exit(3)
if mode != "no-language":
    print("whisper_full_with_state: auto-detected language: de (p = 0.999549)", file=sys.stderr)
if mode == "no-json":
    sys.exit(0)
base = args[args.index("-of") + 1]
text = "{{not json" if mode == "bad-json" else Path({payload!r}).read_text()
Path(base + ".json").write_text(text)
"""
FAKE_FFMPEG = """#!{python}
import json, sys
from pathlib import Path

args = sys.argv[1:]
Path({record!r}).write_text(json.dumps(args))
if Path({mode!r}).read_text().strip() == "fail":
    print("Invalid data found when processing input", file=sys.stderr)
    sys.exit(1)
Path(args[-1]).write_bytes(b"RIFF" + bytes(64))
"""


def token(text: str, start_ms: int, end_ms: int, token_id: int = 1000) -> dict[str, Any]:
    return {
        "text": text,
        "offsets": {"from": start_ms, "to": end_ms},
        "id": token_id,
        "p": 0.9,
        "t_dtw": -1,
    }


def segment(*tokens: dict[str, Any]) -> dict[str, Any]:
    return {"offsets": {"from": 0, "to": 0}, "text": "", "tokens": list(tokens)}


# Clip start 1199.0 for the window t = 1200: file time = 1199 + offset.
FIXTURE_PAYLOAD: dict[str, Any] = {
    "result": {"language": "de"},
    "transcription": [
        segment(
            token("[_BEG_]", 0, 0, 50365),
            token(",", 0, 40),  # punctuation before any word: dropped
            token(" vorher", 100, 400),  # starts before t: assembled, then filtered out
            token(" Hallo", 1000, 1300),  # starts exactly at t
            token(",", 1300, 1340),
            token(" Welt", 1400, 1700),
            token(" ,", 1700, 1720),  # space-led punctuation joins the word
            token(" ", 1720, 1730),  # whitespace only: skipped
            token(" Office", 2000, 2300),
            token("-", 2300, 2320),
            token("Lizenz", 2320, 2600),
            token("[_TT_130]", 2600, 2600, 50495),
        ),
        segment(
            token("[_BEG_]", 2600, 2600, 50365),
            token("en", 2600, 2700),  # no leading space across a segment boundary: joins
            token("<|endoftext|>", 2700, 2700, 50257),  # special by id, whatever its text
            token("[_extra]", 2700, 2710),  # skipped by its text
            token(" wirklich", 3000, 3300),
            token(" !?", 3300, 3340),  # space-led, but a substring of .,!?;:, so it joins
            token(" ...", 3400, 3500),  # not a substring of .,!?;: starts a word
            token(" Ende", 30900, 30990),  # starts at 1229.9 < 1230: kept
            token(" danach", 31000, 31500),  # starts at 1230 = t + 30: filtered out
            token(".", 31500, 31600),
        ),
    ],
}


def retry_py_words(payload: dict[str, Any], st: float, t: float) -> list[dict[str, Any]]:
    """``retry.py``'s word loop, verbatim, on what ``whisper-server`` returned for the same
    tokens: its verbose_json leaves out tokens with id >= end-of-text and reports token
    times as centiseconds * 0.01."""
    server_segments = []
    for raw in payload["transcription"]:
        server_words = []
        for tk in raw["tokens"]:
            if tk["id"] >= 50257:
                continue
            server_words.append(
                {
                    "word": tk["text"],
                    "start": (tk["offsets"]["from"] // 10) * 0.01,
                    "end": (tk["offsets"]["to"] // 10) * 0.01,
                }
            )
        server_segments.append({"words": server_words})
    r = {"segments": server_segments}
    WIN = 30.0
    words: list[dict[str, Any]] = []
    for s in r.get("segments", []):
        for tk in s.get("words", []):
            w = tk["word"]
            if w.startswith("[_") or not w.strip():
                continue
            a, e = st + tk["start"], st + tk["end"]
            if (w.startswith(" ") or not words) and w.strip() not in ".,!?;:":
                words.append({"w": w.strip(), "s": round(a, 3), "e": round(e, 3)})
            elif words:
                words[-1]["w"] += w.strip()
                words[-1]["e"] = round(e, 3)
    return [w for w in words if t <= w["s"] < t + WIN]


@pytest.fixture
def window_runtime(settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Fake ``whisper-cli`` and ffmpeg, a pinned fake model and a private temp directory."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model = model_dir / "ggml-large-v3-turbo.bin"
    model.write_bytes(b"turbo weights")
    # The deployment's batch whisper.cpp model sits next to it and must never be used.
    (model_dir / "ggml-large-v3.bin").write_bytes(b"large-v3 weights")
    monkeypatch.setattr(
        windows, "WINDOW_MODEL_SHA256", hashlib.sha256(b"turbo weights").hexdigest()
    )

    payload = tmp_path / "payload.json"
    payload.write_text(json.dumps(FIXTURE_PAYLOAD), encoding="utf-8")
    whisper_mode = tmp_path / "whisper-mode"
    whisper_mode.write_text("ok", encoding="utf-8")
    ffmpeg_mode = tmp_path / "ffmpeg-mode"
    ffmpeg_mode.write_text("ok", encoding="utf-8")
    whisper_args = tmp_path / "whisper-args.json"
    ffmpeg_args = tmp_path / "ffmpeg-args.json"
    cli = tmp_path / "whisper-cli"
    cli.write_text(
        FAKE_WHISPER.format(
            python=sys.executable,
            record=str(whisper_args),
            mode=str(whisper_mode),
            payload=str(payload),
            marker=TRANSCRIPT_MARKER,
        ),
        encoding="utf-8",
    )
    cli.chmod(0o755)
    ffmpeg = tmp_path / "ffmpeg"
    ffmpeg.write_text(
        FAKE_FFMPEG.format(python=sys.executable, record=str(ffmpeg_args), mode=str(ffmpeg_mode)),
        encoding="utf-8",
    )
    ffmpeg.chmod(0o755)
    temp_root = tmp_path / "tmp"
    temp_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp_root))

    settings.VOXHELM_WHISPERCPP_BIN = str(cli)
    settings.VOXHELM_WHISPERCPP_MODEL = "ggml-large-v3.bin"
    settings.VOXHELM_MODEL_CACHE_DIR = model_dir
    settings.VOXHELM_WINDOW_MODEL = "ggml-large-v3-turbo.bin"
    settings.VOXHELM_FFMPEG_BIN = str(ffmpeg)
    windows._model_sha256.cache_clear()
    windows._whispercpp_version.cache_clear()

    def no_other_engine(*, backend_name: str, model_name: str):
        raise AssertionError(f"window transcription built the {backend_name} backend")

    monkeypatch.setattr("transcriptions.service.build_backend_service", no_other_engine)

    source = tmp_path / "episode.mp3"
    source.write_bytes(b"ID3 fake audio")

    class Runtime:
        pass

    runtime: Any = Runtime()
    runtime.model = model
    runtime.source = source
    runtime.temp_root = temp_root
    runtime.whisper_mode = whisper_mode
    runtime.ffmpeg_mode = ffmpeg_mode
    runtime.payload = payload
    runtime.whisper_args = lambda: json.loads(whisper_args.read_text(encoding="utf-8"))
    runtime.whisper_ran = whisper_args.exists
    runtime.ffmpeg_args = lambda: json.loads(ffmpeg_args.read_text(encoding="utf-8"))
    yield runtime
    windows._model_sha256.cache_clear()
    windows._whispercpp_version.cache_clear()


def assert_nothing_left(runtime: Any) -> None:
    assert list(runtime.temp_root.iterdir()) == []
    assert Job.objects.count() == 0
    assert JobArtifact.objects.count() == 0
    assert StagedMedia.objects.count() == 0
    assert DBTaskResult.objects.count() == 0


@pytest.mark.django_db
def test_window_returns_words_language_probability_and_identity(window_runtime) -> None:
    result = transcribe_window(window_runtime.source, 1200)

    assert result.t == 1200.0
    assert result.clip_start == 1199.0
    assert result.language == "de"
    assert result.language_probability == 0.999549
    assert result.words == retry_py_words(FIXTURE_PAYLOAD, 1199.0, 1200.0)
    assert result.identity == {
        "engine": "whisper.cpp",
        "cli": "whisper-cli",
        "cli_version": "1.9.5",
        "model": "ggml-large-v3-turbo",
        "model_sha256": hashlib.sha256(b"turbo weights").hexdigest(),
        "settings": {
            "pad_seconds": 1.0,
            "window_seconds": 30.0,
            "clip_seconds": 32.0,
            "sample_rate": 16000,
            "language": "auto",
            "decoding": "greedy",
            "best_of": 2,
            "temperature": 0.0,
            "temperature_inc": 0.2,
            "threads": 4,
            "processors": 1,
            "max_len": 60,
            "token_timestamps": True,
        },
    }
    assert result.as_record()["identity"] == result.identity

    whisper_args = window_runtime.whisper_args()
    assert whisper_args[whisper_args.index("-m") + 1] == str(window_runtime.model)
    flags = {
        flag: whisper_args[whisper_args.index(flag) + 1]
        for flag in ("-l", "-t", "-p", "-bs", "-bo", "-tp", "-tpi", "-ml")
    }
    assert flags == {
        "-l": "auto",
        "-t": "4",
        "-p": "1",
        "-bs": "1",
        "-bo": "2",
        "-tp": "0.00",
        "-tpi": "0.20",
        "-ml": "60",
    }
    assert "-ojf" in whisper_args
    # The probability is read from whisper.cpp's log, so the log must not be silenced.
    assert "-np" not in whisper_args
    ffmpeg_args = window_runtime.ffmpeg_args()
    assert ffmpeg_args[:-1] == [
        "-nostdin",
        "-v",
        "error",
        "-ss",
        "1199.000",
        "-t",
        "32.000",
        "-i",
        str(window_runtime.source),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
    ]
    assert_nothing_left(window_runtime)


@pytest.mark.django_db
def test_window_at_file_start_clamps_the_clip(window_runtime) -> None:
    result = transcribe_window(window_runtime.source, 0.5)

    assert result.clip_start == 0.0
    assert window_runtime.ffmpeg_args()[4:7] == ["0.000", "-t", "32.000"]


@pytest.mark.django_db
def test_wrong_model_hash_is_refused_without_running_anything(
    window_runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(windows, "WINDOW_MODEL_SHA256", "0" * 64)

    with pytest.raises(BackendUnavailableError, match="not the pinned"):
        transcribe_window(window_runtime.source, 1200)
    assert not window_runtime.whisper_ran()
    assert window_runtime_available() is False
    assert_nothing_left(window_runtime)


@pytest.mark.django_db
def test_missing_turbo_model_never_uses_the_batch_model(window_runtime) -> None:
    window_runtime.model.unlink()

    with pytest.raises(BackendUnavailableError):
        transcribe_window(window_runtime.source, 1200)
    assert not window_runtime.whisper_ran()
    assert window_runtime_available() is False
    assert_nothing_left(window_runtime)


@pytest.mark.django_db
def test_missing_binary_is_unavailable(window_runtime, settings, tmp_path: Path) -> None:
    settings.VOXHELM_WHISPERCPP_BIN = str(tmp_path / "missing" / "whisper-cli")

    with pytest.raises(BackendUnavailableError):
        transcribe_window(window_runtime.source, 1200)
    assert window_runtime_available() is False


def test_runtime_available_with_the_pinned_model(window_runtime) -> None:
    assert window_runtime_available() is True


def test_replaced_model_file_is_hashed_again(window_runtime) -> None:
    assert window_runtime_available() is True
    window_runtime.model.write_bytes(b"other weights, same name")

    assert window_runtime_available() is False


@pytest.mark.django_db
@pytest.mark.parametrize("mode", ["fail", "no-json", "bad-json", "no-language"])
def test_whisper_failure_leaves_nothing_and_no_transcript_in_the_error(
    window_runtime, mode: str
) -> None:
    window_runtime.whisper_mode.write_text(mode, encoding="utf-8")

    with pytest.raises(RuntimeError) as excinfo:
        transcribe_window(window_runtime.source, 1200)
    assert TRANSCRIPT_MARKER not in str(excinfo.value)
    if mode == "fail":
        assert "failed to load model" in str(excinfo.value)
    assert_nothing_left(window_runtime)


@pytest.mark.django_db
def test_ffmpeg_failure_leaves_nothing(window_runtime) -> None:
    window_runtime.ffmpeg_mode.write_text("fail", encoding="utf-8")

    with pytest.raises(RuntimeError, match="ffmpeg clip decoding failed"):
        transcribe_window(window_runtime.source, 1200)
    assert not window_runtime.whisper_ran()
    assert_nothing_left(window_runtime)


@pytest.mark.django_db
def test_malformed_tokens_fail_and_leave_nothing(window_runtime) -> None:
    broken = json.loads(json.dumps(FIXTURE_PAYLOAD))
    del broken["transcription"][0]["tokens"][3]["offsets"]
    window_runtime.payload.write_text(json.dumps(broken), encoding="utf-8")

    with pytest.raises(RuntimeError, match="without offsets"):
        transcribe_window(window_runtime.source, 1200)
    assert_nothing_left(window_runtime)


@pytest.mark.django_db
def test_cancelled_before_the_call_runs_nothing(window_runtime) -> None:
    cancel = threading.Event()
    cancel.set()

    with pytest.raises(InferenceCancelled):
        transcribe_window(window_runtime.source, 1200, cancel_event=cancel)
    assert not window_runtime.whisper_ran()
    assert_nothing_left(window_runtime)


@pytest.mark.django_db
@pytest.mark.parametrize("step", ["decode_clip", "window_words", "window_identity"])
def test_cancelled_after_a_fast_step_returns_no_result(
    window_runtime, monkeypatch: pytest.MonkeyPatch, step: str
) -> None:
    # The children finish before the process runner looks at the event again.
    cancel = threading.Event()
    original = getattr(windows, step)

    def then_cancel(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        cancel.set()
        return result

    monkeypatch.setattr(windows, step, then_cancel)

    with pytest.raises(InferenceCancelled):
        transcribe_window(window_runtime.source, 1200, cancel_event=cancel)
    assert window_runtime.whisper_ran() == (step != "decode_clip")
    assert_nothing_left(window_runtime)


@pytest.mark.parametrize("t", [-1, math.nan, math.inf, True, "30"])
def test_window_start_must_be_a_non_negative_number(window_runtime, t: Any) -> None:
    with pytest.raises(ValueError):
        transcribe_window(window_runtime.source, t)
    assert not window_runtime.whisper_ran()


def test_word_assembly_and_window_filter_match_retry_py() -> None:
    words = window_words(FIXTURE_PAYLOAD, clip_start=1199.0, t=1200.0)

    assert words == retry_py_words(FIXTURE_PAYLOAD, 1199.0, 1200.0)
    assert words == [
        {"w": "Hallo,", "s": 1200.0, "e": 1200.34},
        {"w": "Welt,", "s": 1200.4, "e": 1200.72},
        {"w": "Office-Lizenzen", "s": 1201.0, "e": 1201.7},
        {"w": "wirklich!?", "s": 1202.0, "e": 1202.34},
        {"w": "...", "s": 1202.4, "e": 1202.5},
        {"w": "Ende", "s": 1229.9, "e": 1229.99},
    ]


@pytest.mark.parametrize(
    ("clip_start", "t"), [(0.0, 0.0), (0.0, 0.5), (12.345, 13.345), (5909.0, 5910.0)]
)
def test_word_assembly_matches_retry_py_at_other_offsets(clip_start: float, t: float) -> None:
    assert window_words(FIXTURE_PAYLOAD, clip_start=clip_start, t=t) == retry_py_words(
        FIXTURE_PAYLOAD, clip_start, t
    )


def test_window_filter_uses_the_window_not_the_clip() -> None:
    assert WINDOW_SETTINGS.clip_seconds == 32.0
    payload = {"transcription": [segment(token(" a", 500, 600), token(" b", 31_200, 31_500))]}
    # Clip [99, 131): "a" at 99.5 is padding before t, "b" at 130.2 padding after t + 30.
    assert window_words(payload, clip_start=99.0, t=100.0) == []


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"transcription": [{"text": "no tokens"}]},
        {"transcription": [segment({"offsets": {"from": 0, "to": 1}, "id": 5})]},
        {"transcription": [segment({"text": " a", "offsets": {"from": 0, "to": 1}})]},
        {"transcription": [segment({"text": " a", "id": 5, "offsets": {"from": 0}})]},
        {"transcription": [segment({"text": " a", "id": 5, "offsets": {"from": 0, "to": 1.5}})]},
    ],
)
def test_malformed_output_raises(payload: dict[str, Any]) -> None:
    with pytest.raises(RuntimeError):
        window_words(payload, clip_start=0.0, t=0.0)


def test_detected_language_reads_the_log_and_checks_the_result() -> None:
    log = "whisper_full_with_state: auto-detected language: en (p = 0.999908)\n"

    assert detected_language(log, {"result": {"language": "en"}}) == ("en", 0.999908)
    with pytest.raises(RuntimeError, match="inconsistent"):
        detected_language(log, {"result": {"language": "de"}})
    with pytest.raises(RuntimeError, match="exactly one"):
        detected_language("", {"result": {"language": "en"}})
    with pytest.raises(RuntimeError, match="exactly one"):
        detected_language(log + log, {"result": {"language": "en"}})
