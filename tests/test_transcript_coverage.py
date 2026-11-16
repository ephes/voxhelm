"""Incomplete whisper.cpp transcripts are reported as errors (archive item 408)."""

from __future__ import annotations

import json
import logging
import math
import subprocess
import wave
from pathlib import Path
from typing import Any

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile

from jobs import remote_worker_cli as worker_cli
from jobs.remote_workers import sanitize_remote_result_metadata
from jobs.services import build_transcription_metadata
from tests.test_remote_worker_cli import (
    RecordingClient,
    base_claim,
    configure_filesystem_store,
    install_fake_download,
)
from tests.test_service import owned_process_from
from transcriptions.coverage import (
    IncompleteTranscriptError,
    TranscriptCoverage,
    check_transcript_coverage,
    coverage_metadata,
    measure_transcript_coverage,
    wav_duration_seconds,
)
from transcriptions.sanitizer import sanitize_result
from transcriptions.service import (
    BackendInvocation,
    TranscribeParams,
    TranscriptionResult,
    TranscriptionSegment,
    WhisperCppBackend,
    build_backend_service,
    transcribe_audio,
)

# Archive item 408: 3:09:03 of audio; with ``-p 4`` parts 2-4 each stopped
# about 700 s after their start, and whisper-cli still exited 0.
ITEM_408_DURATION = 11343.0
ITEM_408_RUNS = [(0.0, 3536.0), (5672.0, 6378.0), (8508.0, 9208.0)]
# A low sample rate keeps a 3 h fixture WAV at about 2 MB; only frames / rate matters.
FIXTURE_RATE = 100
ITEM_408_GAP = "no text from 3536 s to 5672 s of 11343 s"


def seg(start: float, end: float, index: int = 0, text: str = "Text") -> TranscriptionSegment:
    return TranscriptionSegment(id=index, start=start, end=end, text=text)


def run_segments(runs: list[tuple[float, float]], step: float = 10.0) -> list[TranscriptionSegment]:
    segments: list[TranscriptionSegment] = []
    for run_start, run_end in runs:
        cursor = run_start
        while cursor < run_end:
            end = min(cursor + step, run_end)
            segments.append(seg(cursor, end, len(segments), f"Satz {len(segments)}"))
            cursor = end
    return segments


def write_wav(path: Path, seconds: float, rate: int = FIXTURE_RATE) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(b"\0\0" * round(seconds * rate))


def timestamp(seconds: float) -> str:
    whole = int(seconds)
    millis = round((seconds - whole) * 1000)
    return f"{whole // 3600:02d}:{whole % 3600 // 60:02d}:{whole % 60:02d},{millis:03d}"


def whispercpp_json(segments: list[TranscriptionSegment]) -> dict[str, Any]:
    return {
        "result": {"language": "de"},
        "transcription": [
            {
                "text": segment.text,
                "timestamps": {"from": timestamp(segment.start), "to": timestamp(segment.end)},
            }
            for segment in segments
        ],
    }


# --- Coverage contract -------------------------------------------------------


def test_coverage_is_the_union_of_overlapping_segments() -> None:
    coverage = measure_transcript_coverage(
        [seg(0, 10), seg(5, 15), seg(15, 20), seg(30, 40)], media_duration_seconds=50
    )
    assert coverage.covered_seconds == pytest.approx(30)
    # Two 10 s gaps (20-30 and 40-50): the first one found is kept.
    assert (coverage.largest_gap_start, coverage.largest_gap_end) == (20, 30)


def test_coverage_counts_a_leading_gap() -> None:
    coverage = measure_transcript_coverage([seg(700, 900)], media_duration_seconds=900)
    assert (coverage.largest_gap_start, coverage.largest_gap_end) == (0, 700)


def test_coverage_counts_a_missing_tail() -> None:
    coverage = measure_transcript_coverage([seg(0, 100)], media_duration_seconds=1000)
    assert (coverage.largest_gap_start, coverage.largest_gap_end) == (100, 1000)
    assert coverage.largest_gap_seconds == 900


def test_empty_transcript_is_one_gap_over_the_whole_media() -> None:
    coverage = measure_transcript_coverage([], media_duration_seconds=1200)
    assert coverage == TranscriptCoverage(
        media_duration_seconds=1200,
        covered_seconds=0,
        largest_gap_start=0,
        largest_gap_end=1200,
    )


def test_segments_are_sorted_and_clamped_to_the_media() -> None:
    coverage = measure_transcript_coverage(
        [seg(90, 130), seg(-5, 10), seg(10, 90)], media_duration_seconds=100
    )
    assert coverage.covered_seconds == pytest.approx(100)
    assert coverage.largest_gap_seconds == 0


def test_a_zero_length_segment_splits_a_gap_but_covers_nothing() -> None:
    coverage = measure_transcript_coverage(
        [seg(0, 10), seg(400, 400), seg(990, 1000)], media_duration_seconds=1000
    )
    assert coverage.covered_seconds == pytest.approx(20)
    assert (coverage.largest_gap_start, coverage.largest_gap_end) == (400, 990)


def test_non_finite_segment_times_are_ignored() -> None:
    coverage = measure_transcript_coverage(
        [seg(0, math.inf), seg(math.nan, 5), seg(0, 50)], media_duration_seconds=100
    )
    assert coverage.covered_seconds == pytest.approx(50)


def test_a_gap_exactly_at_the_threshold_passes_and_above_it_fails() -> None:
    check_transcript_coverage(
        [seg(0, 100)], media_duration_seconds=700, max_gap_seconds=600, backend_name="x"
    )
    with pytest.raises(IncompleteTranscriptError, match="no text from 100 s to 701 s of 701 s"):
        check_transcript_coverage(
            [seg(0, 100)], media_duration_seconds=701, max_gap_seconds=600, backend_name="x"
        )


def test_threshold_zero_measures_without_rejecting() -> None:
    coverage = check_transcript_coverage(
        [], media_duration_seconds=5000, max_gap_seconds=0, backend_name="x"
    )
    assert coverage.largest_gap_seconds == 5000


def test_short_silent_media_passes() -> None:
    # Legitimate silence shorter than the threshold, e.g. a 5 min clip without speech.
    check_transcript_coverage([], media_duration_seconds=300, max_gap_seconds=600, backend_name="x")


def test_item_408_segments_are_rejected() -> None:
    with pytest.raises(IncompleteTranscriptError) as excinfo:
        check_transcript_coverage(
            run_segments(ITEM_408_RUNS),
            media_duration_seconds=ITEM_408_DURATION,
            max_gap_seconds=600,
            backend_name="whisper.cpp",
        )
    message = str(excinfo.value)
    # Three holes of about 2130 s each (one of them the missing tail); the first
    # largest one is named.
    assert "no text from 3536 s to 5672 s of 11343 s" in message
    assert "covered 4942 s" in message


def test_wav_duration_is_frames_over_rate(tmp_path: Path) -> None:
    path = tmp_path / "input.wav"
    write_wav(path, 12.5)
    assert wav_duration_seconds(path) == pytest.approx(12.5)


def test_an_unreadable_wav_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "input.wav"
    path.write_bytes(b"RIFFfakewav")
    with pytest.raises(RuntimeError, match="duration could not be measured"):
        wav_duration_seconds(path)


# --- whisper.cpp backend with a zero-exit CLI ----------------------------------


def run_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    settings: Any,
    *,
    media_seconds: float,
    segments: list[TranscriptionSegment],
    returncode: int = 0,
    stderr: str = "",
    stdout: str = "",
    max_gap_seconds: float = 600,
    processors: int = 4,
) -> tuple[TranscriptionResult, list[str]]:
    model_path = tmp_path / "ggml-large-v3.bin"
    model_path.write_text("model", encoding="utf-8")
    input_path = tmp_path / "episode.mp3"
    input_path.write_bytes(b"audio")
    settings.VOXHELM_FFMPEG_BIN = "/tmp/fake-ffmpeg"
    whisper_args: list[str] = []
    monkeypatch.setattr(
        "transcriptions.service.resolve_whispercpp_binary", lambda _path: "/tmp/whisper-cli"
    )
    monkeypatch.setattr(
        "transcriptions.service.resolve_whispercpp_model_path", lambda _name: model_path
    )

    def fake_ffmpeg(args, **kwargs):
        del kwargs
        write_wav(Path(str(args[-1])), media_seconds)
        return type("Completed", (), {"returncode": 0, "stderr": "", "stdout": ""})()

    def fake_whisper_cli(args, **kwargs):
        del kwargs
        call = [str(part) for part in args]
        whisper_args.extend(call)
        output_base = Path(call[call.index("-of") + 1])
        output_base.with_suffix(".json").write_text(
            json.dumps(whispercpp_json(segments)), encoding="utf-8"
        )
        return subprocess.CompletedProcess(call, returncode, stdout, stderr)

    monkeypatch.setattr("transcriptions.service.run_owned_process", owned_process_from(fake_ffmpeg))
    monkeypatch.setattr("transcriptions.service.run_cancellable_process", fake_whisper_cli)
    backend = WhisperCppBackend(
        binary_path="/tmp/whisper-cli",
        model_name="ggml-large-v3.bin",
        processors=processors,
        max_gap_seconds=max_gap_seconds,
    )
    result = backend.transcribe(
        input_path, TranscribeParams(request_model="whisper-1", prompt=None, language="de")
    )
    return result, whisper_args


def test_zero_exit_partial_whispercpp_output_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, settings: Any, caplog: Any
) -> None:
    stderr = (
        "whisper_full_with_state: failed to encode\n"
        "ggml_metal_device_init: GPU name:   MTL0\n"
        "whisper_print_timings:     total time = 1.0 ms\n"
    )
    with caplog.at_level(logging.WARNING, logger="transcriptions.service"):
        with pytest.raises(IncompleteTranscriptError, match=ITEM_408_GAP):
            run_backend(
                monkeypatch,
                tmp_path,
                settings,
                media_seconds=ITEM_408_DURATION,
                segments=run_segments(ITEM_408_RUNS, step=60.0),
                stderr=stderr,
            )
    # whisper.cpp's error line is kept in the log; init and timing noise is not.
    assert "whisper_full_with_state: failed to encode" in caplog.text
    assert "GPU name" not in caplog.text
    assert "total time" not in caplog.text


def test_complete_whispercpp_output_carries_its_coverage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, settings: Any
) -> None:
    result, args = run_backend(
        monkeypatch,
        tmp_path,
        settings,
        media_seconds=1000,
        segments=run_segments([(5.0, 990.0)], step=60.0),
    )
    assert result.coverage is not None
    assert result.coverage.media_duration_seconds == pytest.approx(1000)
    assert result.coverage.covered_seconds == pytest.approx(985)
    # whisper.cpp's library log (and so its errors) must not be silenced.
    assert "-np" not in args


def test_exit_zero_with_error_lines_is_logged_even_when_coverage_passes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, settings: Any, caplog: Any
) -> None:
    with caplog.at_level(logging.WARNING, logger="transcriptions.service"):
        run_backend(
            monkeypatch,
            tmp_path,
            settings,
            media_seconds=100,
            segments=[seg(0, 100)],
            stderr="whisper_full_with_state: failed to decode\n",
        )
    assert "exited 0 but logged errors" in caplog.text
    assert "failed to decode" in caplog.text


def test_the_threshold_setting_reaches_the_backend(settings: Any) -> None:
    settings.VOXHELM_WHISPERCPP_MAX_GAP_SECONDS = 1234
    backend = build_backend_service(backend_name="whispercpp", model_name="ggml-large-v3.bin")
    assert isinstance(backend, WhisperCppBackend)
    assert backend.max_gap_seconds == 1234


def test_threshold_zero_lets_a_partial_transcript_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, settings: Any
) -> None:
    result, _ = run_backend(
        monkeypatch,
        tmp_path,
        settings,
        media_seconds=2000,
        segments=[seg(0, 100)],
        max_gap_seconds=0,
    )
    assert result.coverage is not None
    assert result.coverage.largest_gap_seconds == pytest.approx(1900)


def test_failed_run_reports_error_lines_not_transcript_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, settings: Any
) -> None:
    stderr = "ggml_metal_device_init: GPU name:   MTL0\nwhisper_model_load: failed to load model\n"
    with pytest.raises(RuntimeError) as excinfo:
        run_backend(
            monkeypatch,
            tmp_path,
            settings,
            media_seconds=10,
            segments=[seg(0, 10)],
            returncode=1,
            stderr=stderr,
            stdout="[00:00:00.000 --> 00:00:10.000]  private transcript text\n",
        )
    message = str(excinfo.value)
    assert message == "whisper.cpp transcription failed: whisper_model_load: failed to load model"
    assert not isinstance(excinfo.value, IncompleteTranscriptError)


def test_failed_run_without_error_lines_reports_the_stderr_tail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, settings: Any
) -> None:
    stderr = "\n".join(f"line {index}" for index in range(10)) + "\n"
    with pytest.raises(RuntimeError) as excinfo:
        run_backend(
            monkeypatch,
            tmp_path,
            settings,
            media_seconds=10,
            segments=[seg(0, 10)],
            returncode=3,
            stderr=stderr,
        )
    assert str(excinfo.value) == (
        "whisper.cpp transcription failed: line 5; line 6; line 7; line 8; line 9"
    )


# --- Failure semantics on the three paths -------------------------------------


class IncompleteBackend:
    def transcribe(self, audio_path: Path, params: TranscribeParams) -> TranscriptionResult:
        del audio_path, params
        raise IncompleteTranscriptError("whisper.cpp transcript is incomplete: synthetic")


class MustNotRunBackend:
    def transcribe(self, audio_path: Path, params: TranscribeParams) -> TranscriptionResult:
        raise AssertionError("an incomplete transcript must not fall back to another backend")


def test_local_and_sync_path_fail_closed_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "transcriptions.service.get_backend_services_for_model",
        lambda _request_model: [
            BackendInvocation("whispercpp", IncompleteBackend()),
            BackendInvocation("mlx", MustNotRunBackend()),
        ],
    )
    with pytest.raises(IncompleteTranscriptError):
        transcribe_audio(
            Path("/tmp/episode.mp3"),
            TranscribeParams(request_model="whisper-1", prompt=None, language="de"),
        )


def test_sync_endpoint_returns_a_server_error_for_an_incomplete_transcript(
    client: Any, monkeypatch: pytest.MonkeyPatch, settings: Any
) -> None:
    del settings
    monkeypatch.setattr(
        "transcriptions.service.get_backend_services_for_model",
        lambda _request_model: [BackendInvocation("whispercpp", IncompleteBackend())],
    )
    upload = SimpleUploadedFile("episode.mp3", b"mp3-bytes", content_type="audio/mpeg")
    response = client.post(
        "/v1/audio/transcriptions",
        data={"file": upload, "model": "whisper-1"},
        HTTP_AUTHORIZATION="Bearer test-token",
    )
    assert response.status_code == 500
    assert "incomplete" in response.json()["error"]["message"]


def test_remote_worker_fails_the_attempt_as_retryable_without_publishing(
    monkeypatch: pytest.MonkeyPatch, settings: Any, tmp_path: Path
) -> None:
    configure_filesystem_store(settings, tmp_path / "artifacts")
    install_fake_download(monkeypatch, tmp_path)
    targets: list[str] = []

    def backend_for(*, backend_name: str, model_name: str) -> Any:
        del model_name
        targets.append(backend_name)
        return IncompleteBackend() if backend_name == "whispercpp" else MustNotRunBackend()

    monkeypatch.setattr(worker_cli, "build_backend_service", backend_for)
    claim = base_claim()
    claim["fallbacks"] = [{"backend": "mlx", "model": "whisper-large-v3"}]
    client = RecordingClient()
    failures: list[dict[str, Any]] = []
    monkeypatch.setattr(client, "heartbeat_worker", lambda **kwargs: {})
    monkeypatch.setattr(client, "claim", lambda **kwargs: claim)

    def fail_job(**kwargs: Any) -> dict[str, Any]:
        failures.append(kwargs)
        return {}

    monkeypatch.setattr(client, "fail_job", fail_job)

    outcome = worker_cli.run_worker_once(config=client.config, client=client, capabilities={})

    assert outcome == worker_cli.WorkerOutcome.FAILED
    assert targets == ["whispercpp"]
    assert client.completed is None
    assert len(failures) == 1
    assert failures[0]["retryable"] is True
    assert "incomplete" in failures[0]["error_detail"]


# --- Coverage in result metadata ---------------------------------------------


def covered_result() -> TranscriptionResult:
    return TranscriptionResult(
        text="hallo",
        language="de",
        segments=[seg(0, 10)],
        backend_name="whisper.cpp",
        model_name="ggml-large-v3.bin",
        coverage=TranscriptCoverage(
            media_duration_seconds=12.3456,
            covered_seconds=10.0,
            largest_gap_start=10.0,
            largest_gap_end=12.3456,
        ),
    )


def test_coverage_metadata_is_empty_without_coverage() -> None:
    assert coverage_metadata(None) == {}


def test_local_job_metadata_records_coverage() -> None:
    metadata = build_transcription_metadata(
        requested_model="auto",
        requested_language="de",
        result=covered_result(),
        processing_seconds=1.0,
        source_kind="upload",
        source_name="episode.mp3",
        source_url="",
        source_content_type="audio/mpeg",
    )
    assert metadata["media_duration_seconds"] == 12.346
    assert metadata["covered_seconds"] == 10.0
    assert metadata["duration_seconds"] == 10.0


def test_remote_metadata_records_coverage_and_the_server_keeps_it(
    monkeypatch: pytest.MonkeyPatch, settings: Any, tmp_path: Path
) -> None:
    configure_filesystem_store(settings, tmp_path / "artifacts")
    install_fake_download(monkeypatch, tmp_path)

    class CoveredBackend:
        def transcribe(self, audio_path: Path, params: object) -> TranscriptionResult:
            del audio_path, params
            return covered_result()

    monkeypatch.setattr(worker_cli, "build_backend_service", lambda **_: CoveredBackend())
    client = RecordingClient()
    worker_cli.process_claim(base_claim(), config=client.config, client=client)
    assert client.completed is not None
    sent = client.completed["result_metadata"]
    assert sent["media_duration_seconds"] == 12.346
    assert sent["covered_seconds"] == 10.0
    kept = sanitize_remote_result_metadata(sent)
    assert kept["media_duration_seconds"] == 12.346
    assert kept["covered_seconds"] == 10.0
    for bad in (-1, math.nan, "12", True):
        assert "covered_seconds" not in sanitize_remote_result_metadata({"covered_seconds": bad})


def test_sanitizing_keeps_the_backend_coverage() -> None:
    loop = [seg(float(index), float(index + 1), index, "Immer wieder.") for index in range(8)]
    result = TranscriptionResult(
        text=" ".join(segment.text for segment in loop),
        language="de",
        segments=loop,
        backend_name="whisper.cpp",
        model_name="ggml-large-v3.bin",
        coverage=covered_result().coverage,
    )
    sanitized = sanitize_result(result, enabled=True, repeat_threshold=4)
    assert len(sanitized.segments) == 1
    assert sanitized.coverage == result.coverage
