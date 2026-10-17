"""Chapter windows of ``ad-analysis-v1`` (D-29): request validation, the research's clip,
decoding and word assembly, the artifact and its identity, no fallback, the duration
check, cancellation, no residue, remote workers and retention."""

from __future__ import annotations

import json
import threading
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from django.utils import timezone

from jobs import remote_worker_cli as worker_cli
from jobs.artifacts import current_artifact_store_identity, get_artifact_store
from jobs.fingerprints import job_request_fingerprint_from_fields
from jobs.models import Job, JobArtifact
from jobs.retention import prune_job_artifacts
from lane_scheduler import InferenceCancelled
from tests.test_ad_analysis_profile import (  # noqa: F401  (fixture)
    delete_job,
    parakeet_cli,
    parakeet_worker_capabilities,
    post_job,
    profile_payload,
    run_local_profile_job,
    words_manifest_entry,
    worker_identity,
)
from tests.test_remote_workers import (
    claim_one,
    enable_remote_workers,
    producer_headers,
    store_manifest_artifacts,
    transcript_manifest,
    worker_headers,
)
from transcriptions import chapter_windows, windows
from transcriptions.chapter_windows import (
    CHAPTER_WINDOW_SETTINGS,
    MAX_CHAPTER_MARK_MS,
    MAX_CHAPTER_MARKS,
    ChapterMarkError,
    chapter_window_words,
    parse_chapter_marks,
    run_chapter_windows,
)
from transcriptions.profiles import AD_ANALYSIS_V1, PROFILES, run_profile_transcription

MARKS = [1591000, 1701741]
RESEARCH_PROMPT = "Hello, and welcome back. Thanks!"


@pytest.fixture
def runtime(parakeet_cli):  # noqa: F811
    """The fake Parakeet CLI with the fake window runtime (``runtime.window``)."""
    return parakeet_cli


def chapter_document(client, payload: dict[str, Any]) -> dict[str, Any]:
    url = payload["result"]["artifacts"]["chapter_windows"]
    assert url.endswith("/chapter_windows.json")
    response = client.get(url, **producer_headers())
    assert response.status_code == 200
    assert response["Content-Type"] == "application/json"
    return json.loads(response.content)


def research_words(raw: bytes, t0: float) -> list[dict[str, Any]]:
    """``asrwin120.py``'s word list, verbatim, in the ``{w, s, e}`` shape Delve reads."""
    j = json.loads(raw.decode("utf-8", errors="replace"))
    words = [
        [
            round(t0 + x["offsets"]["from"] / 1000, 3),
            round(t0 + x["offsets"]["to"] / 1000, 3),
            x["text"].strip(),
        ]  # noqa: E501
        for x in j["transcription"]
        if x["text"].strip()
    ]
    return [{"w": w[2], "s": float(w[0]), "e": float(w[1])} for w in words]


def expected_identity(cli_version: str = "1.9.5") -> dict[str, Any]:
    return {
        "rule": "turbo-window-v1",
        "engine": "whisper.cpp",
        "cli": "whisper-cli",
        "cli_version": cli_version,
        "model": "ggml-large-v3-turbo",
        "model_sha256": windows.WINDOW_MODEL_SHA256,
        "settings": {
            "half_window_seconds": 60.0,
            "clip_seconds": 120.0,
            "sample_rate": 16000,
            "language": "en",
            "dtw": "large.v3.turbo",
            "max_len": 1,
            "split_on_word": True,
            "prompt": RESEARCH_PROMPT,
            "prompt_sha256": "9cb9a7d3694f8a08f1f6d8c22d72413ae80e7d612b58da5653321473e8c97654",
            "shifts_seconds": [0.0, 0.7, -0.7, 1.9, -1.9],
            "decoding": "beam_search",
            "beam_size": 5,
            "best_of": 5,
            "temperature": 0.0,
            "temperature_inc": 0.2,
            "threads": 4,
            "processors": 1,
        },
    }


# --- request validation ------------------------------------------------------------------


def test_marks_are_parsed_strictly():
    assert parse_chapter_marks(None) is None
    assert parse_chapter_marks([0, 1, MAX_CHAPTER_MARK_MS]) == [0, 1, MAX_CHAPTER_MARK_MS]
    assert len(parse_chapter_marks(list(range(MAX_CHAPTER_MARKS))) or []) == MAX_CHAPTER_MARKS


@pytest.mark.parametrize(
    ("marks", "message"),
    [
        ("1000", "must be a list"),
        ({"start": 1000}, "must be a list"),
        ([], "must not be empty"),
        (list(range(MAX_CHAPTER_MARKS + 1)), "at most 256"),
        ([1000.0], "whole milliseconds"),
        ([1000.5], "whole milliseconds"),
        ([True], "whole milliseconds"),
        (["1000"], "whole milliseconds"),
        ([None], "whole milliseconds"),
        ([-1], "must lie in"),
        ([MAX_CHAPTER_MARK_MS + 1], "must lie in"),
        ([2000, 1000], "strictly increasing"),
        ([1000, 1000], "strictly increasing"),
    ],
)
def test_bad_marks_are_refused(marks: Any, message: str):
    with pytest.raises(ChapterMarkError, match=message):
        parse_chapter_marks(marks)


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"chapter_marks_ms": []}, "must not be empty"),
        ({"chapter_marks_ms": [1000, 1000]}, "strictly increasing"),
        ({"chapter_marks_ms": [1.5]}, "whole milliseconds"),
        ({"chapter_marks_ms": list(range(300))}, "at most 256"),
        ({"chapter_marks_ms": [MAX_CHAPTER_MARK_MS + 1]}, "must lie in"),
        (
            {"chapter_marks_ms": [1000], "profile": None, "language_hint": None},
            "chapter_marks_ms requires a profile",
        ),
    ],
)
def test_submission_refuses_bad_marks(client, settings, overrides, message):
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}
    response = post_job(client, profile_payload(**overrides))

    assert response.status_code == 400
    assert message in response.json()["error"]["message"]
    assert not Job.objects.exists()


@pytest.mark.django_db
def test_marks_are_stored_and_part_of_the_task_ref_identity(client, settings):
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}
    first = post_job(client, profile_payload(chapter_marks_ms=MARKS))
    again = post_job(client, profile_payload(chapter_marks_ms=MARKS))
    other = post_job(client, profile_payload(chapter_marks_ms=[1591000]))
    none = post_job(client, profile_payload())

    assert [r.status_code for r in (first, again, other, none)] == [201, 200, 201, 201]
    assert again.json()["id"] == first.json()["id"]
    assert len({first.json()["id"], other.json()["id"], none.json()["id"]}) == 3
    job = Job.objects.get(id=first.json()["id"])
    assert job.output_data["chapter_marks_ms"] == MARKS
    assert "chapter_marks_ms" not in Job.objects.get(id=none.json()["id"]).output_data


def test_fingerprints_without_marks_are_unchanged():
    fields: dict[str, Any] = {
        "job_type": "transcribe",
        "backend": "auto",
        "model": "auto",
        "language": "de",
        "input_data": {"kind": "upload", "upload_id": "u1"},
        "output_formats": ["json"],
        "diarization": {"enabled": False},
        "profile": AD_ANALYSIS_V1,
    }
    plain = job_request_fingerprint_from_fields(**fields)
    assert job_request_fingerprint_from_fields(**fields, chapter_marks_ms=None) == plain
    assert job_request_fingerprint_from_fields(**fields, chapter_marks_ms=[]) == plain
    assert job_request_fingerprint_from_fields(**fields, chapter_marks_ms=MARKS) != plain


# --- the research's clip, decoding and words ------------------------------------------------


@pytest.mark.django_db
def test_local_job_writes_the_chapter_windows_artifact(client, settings, runtime):
    payload = run_local_profile_job(
        client, settings, language_hint="en", chapter_marks_ms=[30000, *MARKS]
    )

    assert payload["state"] == "succeeded"
    window = runtime.window
    calls = window.chapter_calls()
    # One clip per mark, [t - 60, t + 60), clamped at the start of the file.
    assert [start for start, _args in calls] == ["0.000", "1531.000", "1641.741"]
    args = calls[1][1]
    assert args[args.index("-m") + 1] == str(window.model)  # the pinned turbo file
    assert args[args.index("-l") + 1] == "en"
    assert args[args.index("-dtw") + 1] == "large.v3.turbo"
    assert args[args.index("-ml") + 1] == "1"
    assert "-sow" in args and "-oj" in args and "-np" in args and "-ojf" not in args
    assert args[args.index("--prompt") + 1] == RESEARCH_PROMPT
    assert [args[args.index(flag) + 1] for flag in ("-t", "-p", "-bs", "-bo", "-tp", "-tpi")] == [
        "4",
        "1",
        "5",
        "5",
        "0.00",
        "0.20",
    ]
    assert args[-1].endswith("/x.wav")
    # The duration was measured once, before Parakeet ran.
    assert len(window.ffprobe_calls()) == 1

    document = chapter_document(client, payload)
    assert document["schema_version"] == 1
    assert document["profile"] == AD_ANALYSIS_V1
    assert document["rule"] == "turbo-window-v1"
    assert document["identity"] == expected_identity()
    assert document["duration_ms"] == 7200000
    assert document["summary"]["windows"] == 3
    assert document["summary"]["ok"] == 3
    assert [w["mark_ms"] for w in document["windows"]] == [30000, *MARKS]
    second = document["windows"][1]
    assert {k: v for k, v in second.items() if k != "seconds"} == {
        "mark_ms": 1591000,
        "ok": True,
        "clip_start": 1531.0,
        "shift": 0.0,
        "exit_codes": [0],
        "words": [
            {"w": "Thanks", "s": 1590.1, "e": 1590.38},
            {"w": "to", "s": 1590.4, "e": 1590.52},
            {"w": "�", "s": 1590.52, "e": 1590.6},
            {"w": "Linode.", "s": 1590.6, "e": 1591.25},
        ],
    }
    assert document["windows"][2]["words"][0] == {"w": "Thanks", "s": 1700.841, "e": 1701.121}
    assert document["windows"][0]["words"][0] == {"w": "Thanks", "s": 59.1, "e": 59.38}

    metadata = payload["result"]["metadata"]
    assert metadata["chapter_windows"]["identity"] == expected_identity()
    assert (metadata["chapter_windows"]["windows"], metadata["chapter_windows"]["ok"]) == (3, 3)
    # words.json and its identity are what they are without marks.
    assert "chapter_windows" not in metadata["profile"]
    assert list(window.temp_root.iterdir()) == []
    assert runtime.other_engine_calls == []


@pytest.mark.django_db
def test_a_job_without_marks_has_no_chapter_windows(client, settings, runtime):
    payload = run_local_profile_job(client, settings, language_hint="en")

    assert payload["state"] == "succeeded"
    assert "chapter_windows" not in payload["result"]["artifacts"]
    assert "chapter_windows" not in payload["result"]["metadata"]
    assert runtime.window.chapter_calls() == []
    assert runtime.window.ffprobe_calls() == []
    assert not JobArtifact.objects.filter(kind=JobArtifact.Kind.CHAPTER_WINDOWS).exists()


def test_words_follow_the_research_loop_verbatim():
    raw = json.dumps(
        {
            "transcription": [
                {"offsets": {"from": 0, "to": 120}, "text": " Thanks"},
                {"offsets": {"from": 120, "to": 120}, "text": "   "},
                {"offsets": {"from": 120, "to": 381}, "text": "\tto "},
                {"offsets": {"from": 381, "to": 999}, "text": " Linode"},
                {"offsets": {"from": 999, "to": 1000}, "text": "."},
                {"offsets": {"from": 1000, "to": 1000}, "text": ""},
                {"offsets": {"from": 1001, "to": 1500}, "text": " café"},
            ]
        },
        ensure_ascii=False,
    ).encode("utf-8")
    broken = raw.replace("café".encode(), b"caf\xc3")
    for clip_start in (0.0, 1641.741, 1641.0410000000002, 59.9995):
        for data in (raw, broken):
            assert chapter_window_words(data, clip_start=clip_start) == research_words(
                data, clip_start
            )
    assert chapter_window_words(broken, clip_start=0.0)[-1]["w"] == "caf�"


@pytest.mark.parametrize(
    "raw",
    [
        b"not json",
        b"{}",
        b'{"transcription": [1]}',
        b'{"transcription": [{"text": " a"}]}',
        b'{"transcription": [{"text": " a", "offsets": {"from": "0", "to": 1}}]}',
        b'{"transcription": [{"text": 3, "offsets": {"from": 0, "to": 1}}]}',
    ],
)
def test_malformed_whisper_output_fails_the_window(raw: bytes):
    with pytest.raises(RuntimeError):
        chapter_window_words(raw, clip_start=0.0)


@pytest.mark.django_db
def test_a_failing_clip_is_shifted_as_the_research_did(client, settings, runtime):
    window = runtime.window
    # DTW aborts on the first two clips of the first mark: 0 and +0.7 s.
    window.chapter_fail = {"1531.000": 134, "1531.700": 134}
    window.write_config()

    payload = run_local_profile_job(client, settings, language_hint="en", chapter_marks_ms=MARKS)

    assert payload["state"] == "succeeded"
    assert [start for start, _ in window.chapter_calls()] == [
        "1531.000",
        "1531.700",
        "1530.300",
        "1641.741",
    ]
    first = chapter_document(client, payload)["windows"][0]
    assert (first["ok"], first["shift"], first["clip_start"], first["exit_codes"]) == (
        True,
        -0.7,
        1530.3,
        [134, 134, 0],
    )
    assert first["words"][0] == {"w": "Thanks", "s": 1589.4, "e": 1589.68}


@pytest.mark.django_db
def test_a_window_no_clip_transcribes_keeps_the_mark(client, settings, runtime):
    window = runtime.window
    window.chapter_fail = {
        f"{1531.0 + shift:.3f}": 134 for shift in CHAPTER_WINDOW_SETTINGS.shifts_seconds
    }
    window.write_config()

    payload = run_local_profile_job(client, settings, language_hint="en", chapter_marks_ms=MARKS)

    assert payload["state"] == "succeeded"
    document = chapter_document(client, payload)
    first = document["windows"][0]
    assert (first["ok"], first["words"], first["exit_codes"], first["shift"]) == (
        False,
        [],
        [134] * 5,
        -1.9,
    )
    assert document["windows"][1]["ok"] is True
    assert document["summary"]["ok"] == 1
    assert payload["result"]["metadata"]["chapter_windows"]["ok"] == 1


# --- no fallback, the duration check ---------------------------------------------------------


@pytest.mark.django_db
@pytest.mark.parametrize("breakage", ["missing", "other-hash", "no-binary", "no-ffprobe"])
def test_missing_runtime_fails_the_job_before_parakeet_runs(client, settings, runtime, breakage):
    window = runtime.window
    if breakage == "missing":
        window.model.unlink()
    elif breakage == "other-hash":
        window.model.write_bytes(b"another turbo")
    elif breakage == "no-binary":
        settings.VOXHELM_WHISPERCPP_BIN = str(window.root / "no-such-whisper-cli")
    else:
        settings.VOXHELM_FFPROBE_BIN = str(window.root / "no-such-ffprobe")

    payload = run_local_profile_job(client, settings, language_hint="en", chapter_marks_ms=MARKS)

    assert payload["state"] == "failed"
    with pytest.raises(FileNotFoundError):
        runtime.args()  # Parakeet never ran
    assert window.chapter_calls() == []
    assert not JobArtifact.objects.filter(
        kind__in=[JobArtifact.Kind.WORDS, JobArtifact.Kind.CHAPTER_WINDOWS]
    ).exists()
    assert runtime.other_engine_calls == []


@pytest.mark.django_db
def test_the_model_lost_during_the_windows_fails_the_job(client, settings, runtime):
    runtime.window.chapter_remove_model = True
    runtime.window.write_config()

    payload = run_local_profile_job(client, settings, language_hint="en", chapter_marks_ms=MARKS)

    assert payload["state"] == "failed"
    assert "is missing" in payload["error"]["message"]
    # One clip, then the model check: never five "untranscribed" clips.
    assert len(runtime.window.chapter_calls()) == 1
    assert not JobArtifact.objects.filter(kind=JobArtifact.Kind.CHAPTER_WINDOWS).exists()
    assert runtime.other_engine_calls == []


@pytest.mark.django_db
def test_a_mark_after_the_media_end_fails_the_job_before_parakeet_runs(client, settings, runtime):
    runtime.window.duration = "1701.7404"  # 1701740 ms, one before the last mark
    runtime.window.write_config()

    payload = run_local_profile_job(client, settings, language_hint="en", chapter_marks_ms=MARKS)

    assert payload["state"] == "failed"
    assert "after the media's end (1701740 ms)" in payload["error"]["message"]
    with pytest.raises(FileNotFoundError):
        runtime.args()
    assert runtime.window.chapter_calls() == []


@pytest.mark.django_db
def test_a_mark_on_the_last_millisecond_is_within_the_media(client, settings, runtime):
    runtime.window.duration = "1701.741"  # 1701741 ms: the last mark is the last millisecond
    runtime.window.write_config()

    payload = run_local_profile_job(client, settings, language_hint="en", chapter_marks_ms=MARKS)

    assert payload["state"] == "succeeded"
    assert chapter_document(client, payload)["duration_ms"] == 1701741


# --- cancellation and residue -----------------------------------------------------------------


def test_cancellation_while_a_window_runs(runtime, tmp_path: Path):
    runtime.window.chapter_sleep = 2.0
    runtime.window.write_config()
    executable, model = windows.resolve_window_runtime()
    cancel = threading.Event()
    timer = threading.Timer(0.5, cancel.set)
    timer.start()
    try:
        with pytest.raises(InferenceCancelled):
            run_chapter_windows(
                tmp_path / "episode.mp3",
                MARKS,
                executable=executable,
                model_path=model,
                duration_ms=7200000,
                cancel_event=cancel,
            )
    finally:
        timer.cancel()
    # The first window was stopped; the second was never started.
    assert len(runtime.window.chapter_calls()) == 1
    assert list(runtime.window.temp_root.iterdir()) == []


def test_profile_run_honours_cancellation_before_the_windows(
    runtime, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cancel = threading.Event()
    real = chapter_windows.transcribe_chapter_window

    def cancel_after_first(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = real(*args, **kwargs)
        cancel.set()
        return result

    monkeypatch.setattr(chapter_windows, "transcribe_chapter_window", cancel_after_first)
    audio = tmp_path / "episode.wav"
    audio.write_bytes(b"RIFF")

    with pytest.raises(InferenceCancelled):
        run_profile_transcription(
            PROFILES[AD_ANALYSIS_V1],
            audio,
            language_hint="en",
            chapter_marks_ms=MARKS,
            cancel_event=cancel,
        )
    assert len(runtime.window.chapter_calls()) == 1
    assert list(runtime.window.temp_root.iterdir()) == []

    # Cancelled before it starts: no duration probe, no Parakeet, no window.
    with pytest.raises(InferenceCancelled):
        run_profile_transcription(
            PROFILES[AD_ANALYSIS_V1],
            audio,
            language_hint="en",
            chapter_marks_ms=MARKS,
            cancel_event=cancel,
        )
    assert len(runtime.window.chapter_calls()) == 1
    assert len(runtime.window.ffprobe_calls()) == 1


@pytest.mark.django_db
@pytest.mark.parametrize("failure", ["ffmpeg", "invalid-json"])
def test_failures_leave_nothing_and_carry_no_transcript(client, settings, runtime, failure):
    window = runtime.window
    if failure == "ffmpeg":
        ffmpeg = window.root / "ffmpeg-broken"
        ffmpeg.write_text("#!/bin/sh\necho 'broken input' >&2\nexit 1\n", encoding="utf-8")
        ffmpeg.chmod(0o755)
        settings.VOXHELM_FFMPEG_BIN = str(ffmpeg)
    else:
        window.chapter_segments = [(" SecretCONTROLword", 0, 10)]  # a raw control character
        window.write_config()

    payload = run_local_profile_job(client, settings, language_hint="en", chapter_marks_ms=MARKS)

    assert payload["state"] == "failed"
    assert "Secret" not in payload["error"]["message"]
    assert list(window.temp_root.iterdir()) == []
    assert not JobArtifact.objects.filter(kind=JobArtifact.Kind.CHAPTER_WINDOWS).exists()


# --- retention -----------------------------------------------------------------------------------


@pytest.mark.django_db
def test_producer_delete_removes_the_chapter_windows(client, settings, runtime):
    payload = run_local_profile_job(client, settings, language_hint="en", chapter_marks_ms=MARKS)
    assert chapter_document(client, payload)["windows"]
    job = Job.objects.get(id=payload["id"])

    assert delete_job(client, job).status_code == 204

    assert not Job.objects.filter(id=job.id).exists()
    assert not JobArtifact.objects.exists()
    root = Path(settings.VOXHELM_ARTIFACT_ROOT)
    assert [path for path in root.rglob("*") if path.is_file()] == []


@pytest.mark.django_db
def test_backstop_removes_the_chapter_windows_after_seven_days(client, settings, runtime):
    payload = run_local_profile_job(client, settings, language_hint="en", chapter_marks_ms=MARKS)
    Job.objects.filter(id=payload["id"]).update(
        finished_at=timezone.now() - timedelta(days=6, hours=23)
    )
    prune_job_artifacts()
    assert JobArtifact.objects.filter(kind=JobArtifact.Kind.CHAPTER_WINDOWS).exists()

    Job.objects.filter(id=payload["id"]).update(
        finished_at=timezone.now() - timedelta(days=7, minutes=1)
    )
    prune_job_artifacts()

    assert not Job.objects.filter(id=payload["id"]).exists()
    assert not JobArtifact.objects.exists()
    root = Path(settings.VOXHELM_ARTIFACT_ROOT)
    assert [path for path in root.rglob("*") if path.is_file()] == []


# --- remote workers ------------------------------------------------------------------------------


def submit_remote_job_with_marks(client, settings, marks: list[int] | None = MARKS) -> Job:
    enable_remote_workers(settings)
    settings.VOXHELM_PARAKEET_MODEL_NAME = "parakeet-tdt-0.6b-v3"
    response = post_job(client, profile_payload(chapter_marks_ms=marks))
    assert response.status_code == 201
    return Job.objects.get(id=response.json()["id"])


def chapter_manifest_entry(job_id: str, attempt: int, **overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": "chapter_windows.json",
        "kind": "chapter_windows",
        "format": "chapter_windows",
        "storage_backend": "filesystem",
        "storage_key": f"voxhelm/jobs/{job_id}/attempt-{attempt}/chapter_windows.json",
        "content_type": "application/json",
        "size_bytes": 12,
        "exposed": True,
    }
    entry.update(overrides)
    return entry


def chapter_metadata(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "identity": expected_identity(cli_version="1.9.6"),
        "windows": len(MARKS),
        "ok": len(MARKS),
        "seconds": 9.3,
    }
    record.update(overrides)
    return record


def complete(client, settings, job: Job, *, entries, chapter: Any):
    claim = claim_one(
        client, capabilities=parakeet_worker_capabilities(profiles_advertised=True)
    ).json()["job"]
    manifest = [
        entry
        for entry in transcript_manifest(str(job.id), claim["attempt"])
        if entry["format"] != "text"
    ]
    manifest.extend(entries(str(job.id), claim["attempt"]))
    store_manifest_artifacts(settings, manifest)
    result_metadata: dict[str, Any] = {"backend": "parakeet", "profile": worker_identity()}
    if chapter is not None:
        result_metadata["chapter_windows"] = chapter
    response = client.post(
        f"/v1/internal/work/{job.id}/complete",
        data=json.dumps(
            {
                "worker_id": "atlas",
                "lease_token": claim["lease_token"],
                "result_text": "remote transcript",
                "result_metadata": result_metadata,
                "artifacts": manifest,
            }
        ),
        content_type="application/json",
        **worker_headers(),
    )
    return claim, response


def both(job_id: str, attempt: int) -> list[dict[str, Any]]:
    return [words_manifest_entry(job_id, attempt), chapter_manifest_entry(job_id, attempt)]


@pytest.mark.django_db
def test_the_claim_carries_the_marks_and_the_completion_keeps_the_identity(client, settings):
    job = submit_remote_job_with_marks(client, settings)

    claim, response = complete(client, settings, job, entries=both, chapter=chapter_metadata())

    assert claim["chapter_marks_ms"] == MARKS
    assert "fallbacks" not in claim
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["metadata"]["chapter_windows"] == chapter_metadata()
    assert result["artifacts"]["chapter_windows"].endswith("/chapter_windows.json")
    assert client.get(result["artifacts"]["chapter_windows"], **producer_headers()).status_code == (
        200
    )


@pytest.mark.django_db
def test_a_claim_without_marks_carries_none(client, settings):
    submit_remote_job_with_marks(client, settings, marks=None)
    claim = claim_one(
        client, capabilities=parakeet_worker_capabilities(profiles_advertised=True)
    ).json()["job"]
    assert "chapter_marks_ms" not in claim


def other_settings(**changes: Any) -> dict[str, Any]:
    return {**expected_identity()["settings"], **changes}


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("entries", "chapter", "message"),
    [
        (
            lambda job_id, attempt: [words_manifest_entry(job_id, attempt)],
            chapter_metadata(),
            "must include chapter_windows.json",
        ),
        (
            lambda job_id, attempt: [
                words_manifest_entry(job_id, attempt),
                chapter_manifest_entry(job_id, attempt, exposed=False),
            ],
            chapter_metadata(),
            "must be exposed",
        ),
        (
            lambda job_id, attempt: [
                words_manifest_entry(job_id, attempt),
                chapter_manifest_entry(job_id, attempt, name="windows.json"),
            ],
            chapter_metadata(),
            "must be chapter_windows.json",
        ),
        (
            lambda job_id, attempt: [
                words_manifest_entry(job_id, attempt),
                chapter_manifest_entry(job_id, attempt, content_type="text/plain"),
            ],
            chapter_metadata(),
            "content_type must be application/json",
        ),
        (both, None, "must include result_metadata.chapter_windows"),
        (
            both,
            chapter_metadata(identity={**expected_identity(), "model_sha256": "0" * 64}),
            "pinned chapter windows",
        ),
        (
            both,
            chapter_metadata(identity={**expected_identity(), "model": "ggml-large-v3"}),
            "pinned chapter windows",
        ),
        (
            both,
            chapter_metadata(identity={**expected_identity(), "rule": "turbo-window-v2"}),
            "pinned chapter windows",
        ),
        (
            both,
            chapter_metadata(
                identity={**expected_identity(), "settings": other_settings(prompt="Hi.")}
            ),
            "pinned chapter windows",
        ),
        (
            both,
            chapter_metadata(
                identity={**expected_identity(), "settings": other_settings(beam_size=1)}
            ),
            "pinned chapter windows",
        ),
        (
            both,
            chapter_metadata(
                identity={**expected_identity(), "settings": other_settings(shifts_seconds=[0.0])}
            ),
            "pinned chapter windows",
        ),
        (both, chapter_metadata(windows=1), "one window per chapter mark"),
        (both, chapter_metadata(ok=3), "one window per chapter mark"),
        (both, chapter_metadata(ok=True), "one window per chapter mark"),
        (both, chapter_metadata(seconds=-1), "one window per chapter mark"),
    ],
)
def test_remote_completions_with_other_windows_are_refused(
    client, settings, entries, chapter, message
):
    job = submit_remote_job_with_marks(client, settings)

    _claim, response = complete(client, settings, job, entries=entries, chapter=chapter)

    assert response.status_code == 400
    assert message in response.json()["error"]["message"]
    job.refresh_from_db()
    assert job.state != Job.State.SUCCEEDED


@pytest.mark.django_db
def test_chapter_windows_are_refused_for_a_job_without_marks(client, settings):
    job = submit_remote_job_with_marks(client, settings, marks=None)

    _claim, response = complete(client, settings, job, entries=both, chapter=chapter_metadata())

    assert response.status_code == 400
    assert "only allowed for jobs with chapter marks" in response.json()["error"]["message"]


def test_worker_advertises_the_profile_only_with_ffprobe(runtime, settings):
    settings.VOXHELM_DIARIZATION_BACKEND = "none"
    settings.VOXHELM_WHISPERKIT_ENABLED = False
    capabilities = worker_cli.build_capabilities()
    assert capabilities["profiles"] == [AD_ANALYSIS_V1]

    real = settings.VOXHELM_FFPROBE_BIN
    settings.VOXHELM_FFPROBE_BIN = str(runtime.window.root / "no-such-ffprobe")
    assert worker_cli.with_current_stt_runtimes(capabilities)["profiles"] == []
    settings.VOXHELM_FFPROBE_BIN = real
    assert worker_cli.with_current_stt_runtimes(capabilities)["profiles"] == [AD_ANALYSIS_V1]


def test_worker_refuses_bad_marks_for_good(runtime):
    claim = {"id": "job-1", "profile": AD_ANALYSIS_V1, "language": "en", "chapter_marks_ms": [5, 5]}

    with pytest.raises(ChapterMarkError):
        worker_cli.transcribe_claim_profile(claim=claim, audio_path=Path("/tmp/a.mp3"))
    assert not worker_cli.retryable_worker_error(ChapterMarkError("after the end"))
    with pytest.raises(FileNotFoundError):
        runtime.args()


@pytest.mark.django_db
def test_worker_uploads_the_chapter_windows_for_a_claim_with_marks(runtime, tmp_path: Path):
    staged_key = "voxhelm/staged-inputs/u1/episode.mp3"
    get_artifact_store().put_bytes(key=staged_key, data=b"mp3", content_type="audio/mpeg")
    claim = {
        "id": "job-1",
        "lease_token": "lease",
        "backend": "parakeet",
        "model": "parakeet-tdt-0.6b-v3",
        "profile": AD_ANALYSIS_V1,
        "language": "en",
        "chapter_marks_ms": MARKS,
        "input": {
            "kind": "upload",
            "filename": "episode.mp3",
            "content_type": "audio/mpeg",
            "staged_artifact": {"storage_backend": "filesystem", "storage_key": staged_key},
        },
        "output": {"formats": ["json"], "diarization": {"enabled": False}},
        "artifact_prefix": "voxhelm/jobs/job-1/attempt-1",
        "artifact_store": current_artifact_store_identity(),
    }
    completed: dict[str, Any] = {}

    class FakeClient:
        def heartbeat_job(self, **kwargs: Any) -> dict[str, Any]:
            return {}

        def complete_job(self, **kwargs: Any) -> dict[str, Any]:
            completed.update(kwargs)
            return {}

    config = worker_cli.WorkerConfig(
        base_url="http://voxhelm.local",
        worker_id="atlas",
        token="t",
        hostname="atlas",
        poll_seconds=1,
        http_timeout_seconds=5,
        heartbeat_interval_seconds=60,
    )
    worker_cli.process_claim(claim, config=config, client=FakeClient())  # type: ignore[arg-type]

    names = [artifact["name"] for artifact in completed["artifacts"]]
    assert names == ["source.mp3", "transcript.json", "words.json", "chapter_windows.json"]
    entry = completed["artifacts"][-1]
    assert (entry["kind"], entry["format"], entry["exposed"]) == (
        "chapter_windows",
        "chapter_windows",
        True,
    )
    document = json.loads(
        get_artifact_store().read_bytes(key="voxhelm/jobs/job-1/attempt-1/chapter_windows.json")
    )
    assert [w["mark_ms"] for w in document["windows"]] == MARKS
    assert document["identity"] == expected_identity()
    record = completed["result_metadata"]["chapter_windows"]
    assert record["identity"] == expected_identity()
    assert (record["windows"], record["ok"]) == (2, 2)
    assert list(runtime.window.temp_root.iterdir()) == []
    assert runtime.other_engine_calls == []


@pytest.mark.parametrize("marks", [[1000], [], "invalid", None])
def test_a_synthesis_job_refuses_chapter_marks(marks: Any) -> None:
    from jobs.services import ApiError, parse_job_request
    from tests.test_jobs import build_synthesis_payload

    payload = build_synthesis_payload()
    payload["chapter_marks_ms"] = marks
    with pytest.raises(ApiError, match="chapter_marks_ms requires a profile"):
        parse_job_request(payload)
