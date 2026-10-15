from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from lane_scheduler import InferenceCancelled
from transcriptions.service import (
    BackendUnavailableError,
    ChildProcessTimeout,
    ParakeetBackend,
    TranscribeParams,
    TranscriptionWord,
    build_backend_service,
    normalize_parakeet_payload,
    resolve_model_name_for_backend,
    run_cancellable_process,
)

FIXTURE = Path(__file__).parent / "fixtures" / "parakeet_excerpt.json"
PARAMS = TranscribeParams(request_model="auto", prompt=None, language="en", word_timestamps=True)


def fixture_payload() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def word_texts(words: tuple[TranscriptionWord, ...]) -> list[str]:
    return [word.text for word in words]


# --- adapter -------------------------------------------------------------------


def test_normalize_parakeet_payload_builds_sentence_segments_with_words() -> None:
    # Recorded parakeet-mlx 0.5.3 output: the opening of Django Chat episode 219
    # and three sentences around the 600 s chunk seam of episode 206.
    result = normalize_parakeet_payload(fixture_payload(), model_name="parakeet-tdt-0.6b-v3")

    assert result.backend_name == "parakeet"
    assert result.model_name == "parakeet-tdt-0.6b-v3"
    assert result.language is None
    assert result.text.startswith("Hi, welcome to another episode of Django Chat.")
    assert [segment.text for segment in result.segments][:4] == [
        "Hi, welcome to another episode of Django Chat.",
        "I'm Will Vincent with Carlton Gibson.",
        "Hey Carlton.",
        "Hello, Will.",
    ]
    assert [segment.id for segment in result.segments] == list(range(7))
    first = result.segments[0]
    assert (first.start, first.end) == (5.84, 8.32)
    # Sub-word pieces ("wel" "c" "ome") and punctuation join the current word.
    assert word_texts(first.words) == [
        "Hi,",
        "welcome",
        "to",
        "another",
        "episode",
        "of",
        "Django",
        "Chat.",
    ]
    assert first.words[1] == TranscriptionWord(start=6.24, end=6.56, text="welcome")
    assert word_texts(result.segments[3].words) == ["Hello,", "Will."]
    # The re-cut rebuilds piece text from words, so words must cover the text.
    for segment in result.segments:
        assert " ".join(word_texts(segment.words)) == segment.text
        assert segment.start <= segment.words[0].start
        assert segment.words[-1].end <= segment.end
    seam = [segment for segment in result.segments if segment.start < 600 < segment.end]
    assert [segment.text for segment in seam] == ["Um, so maybe I put it back."]


def test_normalize_parakeet_payload_starts_a_word_at_a_sentence_start_without_space() -> None:
    payload = {
        "text": "ing on. Next",
        "sentences": [
            {
                "text": "ing on.",
                "start": 1.0,
                "end": 2.0,
                "tokens": [
                    {"text": "ing", "start": 1.0, "end": 1.2},
                    {"text": " on", "start": 1.2, "end": 1.6},
                    {"text": ".", "start": 1.6, "end": 1.7},
                ],
            }
        ],
    }
    result = normalize_parakeet_payload(payload, model_name="m")
    assert word_texts(result.segments[0].words) == ["ing", "on."]
    assert result.segments[0].words[1] == TranscriptionWord(start=1.2, end=1.7, text="on.")
    assert result.text == "ing on. Next"


def test_normalize_parakeet_payload_clamps_word_times_into_the_sentence() -> None:
    payload = {
        "sentences": [
            {
                "text": "a b",
                "start": 1.0,
                "end": 2.0,
                "tokens": [
                    {"text": " a", "start": 0.5, "end": 1.4},
                    {"text": " b", "start": 1.2, "end": 2.5},
                ],
            }
        ],
    }
    words = normalize_parakeet_payload(payload, model_name="m").segments[0].words
    assert words == (
        TranscriptionWord(start=1.0, end=1.4, text="a"),
        TranscriptionWord(start=1.2, end=2.0, text="b"),
    )


@pytest.mark.parametrize(
    "tokens",
    [
        "not a list",
        [{"text": " a", "start": 1.0, "end": 1.1}, "not a token"],
        [{"text": " a", "start": "1.0", "end": 1.1}],
        [{"text": " a", "start": True, "end": 1.1}],
        [{"text": " a", "start": float("nan"), "end": 1.1}],
        [{"start": 1.0, "end": 1.1}],
    ],
)
def test_normalize_parakeet_payload_drops_words_of_a_sentence_with_malformed_tokens(
    tokens: object,
) -> None:
    good = {
        "text": "fine",
        "start": 3.0,
        "end": 4.0,
        "tokens": [{"text": " fine", "start": 3.0, "end": 3.5}],
    }
    bad = {"text": "a", "start": 1.0, "end": 2.0, "tokens": tokens}
    result = normalize_parakeet_payload({"sentences": [bad, good]}, model_name="m")
    assert [segment.text for segment in result.segments] == ["a", "fine"]
    assert result.segments[0].words == ()
    assert word_texts(result.segments[1].words) == ["fine"]


def test_normalize_parakeet_payload_skips_malformed_and_empty_sentences() -> None:
    payload = {
        "text": "",
        "sentences": [
            "not a sentence",
            {"text": "   ", "start": 0.0, "end": 1.0, "tokens": []},
            {"text": "no times", "tokens": []},
            {"text": "kept", "start": 2.0, "end": 1.5, "tokens": []},
        ],
    }
    result = normalize_parakeet_payload(payload, model_name="m")
    assert [(segment.text, segment.start, segment.end) for segment in result.segments] == [
        ("kept", 2.0, 2.0)
    ]
    # Without a payload text the segment texts are joined.
    assert result.text == "kept"


def test_normalize_parakeet_payload_handles_empty_and_rejects_non_objects() -> None:
    empty = normalize_parakeet_payload({"text": "", "sentences": []}, model_name="m")
    assert (empty.text, empty.segments) == ("", [])
    assert normalize_parakeet_payload({}, model_name="m").segments == []
    with pytest.raises(RuntimeError, match="unexpected payload"):
        normalize_parakeet_payload(["not", "an", "object"], model_name="m")


# --- backend with a fake CLI ---------------------------------------------------


FAKE_CLI = """#!{python}
import json, os, sys, time
from pathlib import Path

args = sys.argv[1:]
record = Path({record!r})
record.write_text(json.dumps({{"args": args, "env": dict(os.environ)}}))
mode = {mode!r}
if mode == "sleep":
    time.sleep(60)
if mode == "fail":
    print("model exploded", file=sys.stderr)
    sys.exit(3)
if mode == "garbage":
    out = Path(args[args.index("--output-dir") + 1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "transcript.json").write_text("not json")
if mode == "ok":
    out = Path(args[args.index("--output-dir") + 1])
    out.mkdir(parents=True, exist_ok=True)
    template = args[args.index("--output-template") + 1]
    (out / (template + ".json")).write_text(Path({fixture!r}).read_text())
"""


@pytest.fixture
def parakeet_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights")
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"not-really-audio")
    record = tmp_path / "record.json"
    normalized: list[tuple[Path, Path]] = []

    def fake_normalize(*, input_path: Path, output_path: Path) -> None:
        normalized.append((input_path, output_path))
        output_path.write_bytes(b"RIFFfakewav")

    monkeypatch.setattr("transcriptions.service.normalize_audio_to_wav_16k_mono", fake_normalize)

    def make(mode: str = "ok", **overrides: object) -> ParakeetBackend:
        cli = tmp_path / f"parakeet-mlx-{mode}"
        cli.write_text(
            FAKE_CLI.format(
                python=sys.executable, record=str(record), mode=mode, fixture=str(FIXTURE)
            ),
            encoding="utf-8",
        )
        cli.chmod(0o755)
        options: dict[str, object] = {
            "binary_path": str(cli),
            "model_dir": str(model_dir),
            "model_name": "parakeet-tdt-0.6b-v3",
            "chunk_seconds": 600,
            "overlap_seconds": 30,
            "max_sentence_seconds": None,
            "timeout_seconds": 30,
        }
        options.update(overrides)
        return ParakeetBackend(**options)  # type: ignore[arg-type]

    def recorded() -> dict[str, object]:
        return json.loads(record.read_text(encoding="utf-8"))

    make.audio = audio  # type: ignore[attr-defined]
    make.model_dir = model_dir  # type: ignore[attr-defined]
    make.recorded = recorded  # type: ignore[attr-defined]
    make.normalized = normalized  # type: ignore[attr-defined]
    return make


def test_parakeet_backend_runs_the_cli_offline_with_pinned_arguments(
    parakeet_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PARAKEET_MAX_DURATION", "3")
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    backend = parakeet_env()

    result = backend.transcribe(parakeet_env.audio, PARAMS)

    assert result.backend_name == "parakeet"
    assert len(result.segments) == 7
    assert word_texts(result.segments[3].words) == ["Hello,", "Will."]
    recorded = parakeet_env.recorded()
    args = recorded["args"]
    wav = parakeet_env.normalized[0][1]
    assert parakeet_env.normalized[0][0] == parakeet_env.audio
    assert args[:-1] == [
        "--model",
        str(parakeet_env.model_dir),
        "--output-format",
        "json",
        "--output-dir",
        str(wav.parent / "out"),
        "--output-template",
        "transcript",
        "--chunk-duration",
        "600",
        "--overlap-duration",
        "30",
        "--local-attention",
    ]
    assert args[-1] == str(wav)
    env = recorded["env"]
    assert env["HF_HUB_OFFLINE"] == "1"
    assert not any(key.startswith("PARAKEET_") for key in env)
    # The temporary directory is gone afterwards.
    assert not wav.parent.exists()


def test_parakeet_backend_passes_max_sentence_seconds_when_configured(parakeet_env) -> None:
    parakeet_env(max_sentence_seconds=12.5).transcribe(parakeet_env.audio, PARAMS)
    args = parakeet_env.recorded()["args"]
    assert args[args.index("--max-duration") + 1] == "12.5"


@pytest.mark.parametrize("language", [None, "", "de", "en-US", "pt_BR"])
def test_parakeet_backend_accepts_supported_or_missing_languages(
    parakeet_env, language: str | None
) -> None:
    params = TranscribeParams(request_model="auto", prompt=None, language=language)
    assert parakeet_env().transcribe(parakeet_env.audio, params).segments


@pytest.mark.parametrize("language", ["ja", "zh", "ar"])
def test_parakeet_backend_is_unavailable_for_unsupported_languages(
    parakeet_env, language: str
) -> None:
    params = TranscribeParams(request_model="auto", prompt=None, language=language)
    with pytest.raises(BackendUnavailableError, match="does not support language"):
        parakeet_env().transcribe(parakeet_env.audio, params)
    assert parakeet_env.normalized == []


def test_parakeet_backend_is_unavailable_without_cli(parakeet_env, tmp_path: Path) -> None:
    with pytest.raises(BackendUnavailableError, match="was not found at"):
        parakeet_env(binary_path=str(tmp_path / "missing")).transcribe(parakeet_env.audio, PARAMS)
    with pytest.raises(BackendUnavailableError, match="not found in PATH"):
        parakeet_env(binary_path="voxhelm-no-such-parakeet").transcribe(parakeet_env.audio, PARAMS)


def test_parakeet_backend_is_unavailable_without_model(parakeet_env, tmp_path: Path) -> None:
    with pytest.raises(BackendUnavailableError, match="not configured"):
        parakeet_env(model_dir="").transcribe(parakeet_env.audio, PARAMS)
    with pytest.raises(BackendUnavailableError, match="config.json, model.safetensors"):
        parakeet_env(model_dir=str(tmp_path / "gone")).transcribe(parakeet_env.audio, PARAMS)
    (parakeet_env.model_dir / "model.safetensors").unlink()
    with pytest.raises(BackendUnavailableError, match="is missing model.safetensors"):
        parakeet_env().transcribe(parakeet_env.audio, PARAMS)


def test_parakeet_backend_fails_on_non_zero_exit(parakeet_env) -> None:
    with pytest.raises(RuntimeError, match="Parakeet transcription failed: model exploded") as info:
        parakeet_env("fail").transcribe(parakeet_env.audio, PARAMS)
    assert not isinstance(info.value, BackendUnavailableError)


def test_parakeet_backend_fails_on_missing_or_invalid_output(parakeet_env) -> None:
    with pytest.raises(RuntimeError, match="transcript.json was not produced"):
        parakeet_env("silent").transcribe(parakeet_env.audio, PARAMS)
    with pytest.raises(RuntimeError, match="invalid JSON"):
        parakeet_env("garbage").transcribe(parakeet_env.audio, PARAMS)


def test_parakeet_backend_stops_the_child_after_the_timeout(parakeet_env) -> None:
    started = time.monotonic()
    with pytest.raises(ChildProcessTimeout):
        parakeet_env("sleep", timeout_seconds=1).transcribe(parakeet_env.audio, PARAMS)
    assert time.monotonic() - started < 10


def test_parakeet_backend_stops_the_child_on_cancellation(parakeet_env) -> None:
    cancel_event = threading.Event()
    params = TranscribeParams(
        request_model="auto", prompt=None, language="en", cancel_event=cancel_event
    )
    timer = threading.Timer(0.5, cancel_event.set)
    timer.start()
    try:
        with pytest.raises(InferenceCancelled):
            parakeet_env("sleep").transcribe(parakeet_env.audio, params)
    finally:
        timer.cancel()


def test_build_backend_service_configures_parakeet_from_settings(settings) -> None:
    settings.VOXHELM_PARAKEET_BIN = "/opt/parakeet-mlx"
    settings.VOXHELM_PARAKEET_MODEL_DIR = "/srv/models/parakeet"
    settings.VOXHELM_PARAKEET_CHUNK_SECONDS = 300
    settings.VOXHELM_PARAKEET_OVERLAP_SECONDS = 15
    settings.VOXHELM_PARAKEET_MAX_SENTENCE_SECONDS = 20.0
    settings.VOXHELM_PARAKEET_TIMEOUT_SECONDS = 99
    settings.VOXHELM_PARAKEET_MODEL_NAME = "parakeet-test"

    model_name = resolve_model_name_for_backend(request_model="auto", backend_name="parakeet")
    backend = build_backend_service(backend_name="parakeet", model_name=model_name)

    assert isinstance(backend, ParakeetBackend)
    assert (
        backend.binary_path,
        backend.model_dir,
        backend.model_name,
        backend.chunk_seconds,
        backend.overlap_seconds,
        backend.max_sentence_seconds,
        backend.timeout_seconds,
    ) == ("/opt/parakeet-mlx", "/srv/models/parakeet", "parakeet-test", 300, 15, 20.0, 99)


# --- run_cancellable_process timeout -------------------------------------------


def test_run_cancellable_process_times_out_without_cancel_event() -> None:
    started = time.monotonic()
    with pytest.raises(ChildProcessTimeout, match="0.5 s time limit"):
        run_cancellable_process(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            cancel_event=None,
            timeout_seconds=0.5,
        )
    assert time.monotonic() - started < 10


def test_run_cancellable_process_passes_env_and_returns_output() -> None:
    completed = run_cancellable_process(
        [sys.executable, "-c", "import os; print(os.environ['VOXHELM_PROBE'])"],
        cancel_event=None,
        timeout_seconds=30,
        env={"VOXHELM_PROBE": "hello"},
    )
    assert isinstance(completed, subprocess.CompletedProcess)
    assert (completed.returncode, completed.stdout.strip()) == (0, "hello")


def test_parakeet_backend_passes_a_relative_model_dir_as_an_absolute_path(
    parakeet_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    # parakeet-mlx tries the Hub first, so a relative path that looks like a repo
    # id must never reach it.
    monkeypatch.chdir(parakeet_env.model_dir.parent)
    parakeet_env(model_dir="model").transcribe(parakeet_env.audio, PARAMS)
    args = parakeet_env.recorded()["args"]
    assert args[args.index("--model") + 1] == str(parakeet_env.model_dir.resolve())


def settings_import_error(**env: str) -> str:
    repo_root = Path(__file__).resolve().parent.parent
    completed = subprocess.run(
        [sys.executable, "-c", "import config.settings"],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root), "DJANGO_SECRET_KEY": "test", **env},
        capture_output=True,
        text=True,
        timeout=60,
    )
    return completed.stderr if completed.returncode else ""


@pytest.mark.parametrize("overlap", ["600", "900"])
def test_settings_reject_a_parakeet_overlap_not_below_the_chunk(overlap: str) -> None:
    error = settings_import_error(
        VOXHELM_PARAKEET_CHUNK_SECONDS="600", VOXHELM_PARAKEET_OVERLAP_SECONDS=overlap
    )
    assert "must be smaller than VOXHELM_PARAKEET_CHUNK_SECONDS" in error


@pytest.mark.parametrize("value", ["-1", "nan", "inf"])
def test_settings_reject_an_invalid_parakeet_max_sentence_duration(value: str) -> None:
    error = settings_import_error(VOXHELM_PARAKEET_MAX_SENTENCE_SECONDS=value)
    assert "VOXHELM_PARAKEET_MAX_SENTENCE_SECONDS must be a non-negative number" in error
