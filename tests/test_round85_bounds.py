"""Speaker-turn domain and alignment budget, remote manifest bounds, window JSON,
normalizer process ownership and staged-cleanup fairness."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import timedelta
from types import SimpleNamespace

import pytest
from django.utils import timezone

from jobs import staging as jobs_staging
from jobs.models import Job, JobArtifact, StagedMedia
from tests.test_remote_workers import (
    claim_one,
    store_manifest_artifacts,
    submit_remote_job,
    transcript_manifest,
    worker_headers,
)
from transcriptions import chapter_windows, windows
from transcriptions.diarization import (
    SpeakerTurn,
    SpeakerTurnIndex,
    apply_speaker_labels,
    choose_speaker_for_segment,
    normalize_speaker_turns,
)
from transcriptions.known_speaker import raw_turn_overlaps
from transcriptions.service import (
    TranscriptionResult,
    TranscriptionSegment,
    normalize_audio_to_wav_16k_mono,
)

# --- B17: diarization turn finite domain -------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        SpeakerTurn(start=math.nan, end=5.0, speaker="X"),
        SpeakerTurn(start=1.0, end=math.nan, speaker="X"),
        SpeakerTurn(start=0.0, end=math.inf, speaker="X"),
        SpeakerTurn(start=-math.inf, end=2.0, speaker="X"),
    ],
    ids=["nan-start", "nan-end", "inf-end", "neg-inf-start"],
)
def test_non_finite_turns_are_dropped(bad) -> None:
    good = SpeakerTurn(start=1.0, end=2.0, speaker="A")
    normalized = normalize_speaker_turns([bad, good])
    assert normalized == [SpeakerTurn(start=1.0, end=2.0, speaker="Speaker 1")]


def test_infinite_turn_cannot_claim_every_segment() -> None:
    result = TranscriptionResult(
        text="a b",
        language="en",
        segments=[
            TranscriptionSegment(id=0, start=0.0, end=1.0, text="a"),
            TranscriptionSegment(id=1, start=10.0, end=11.0, text="b"),
        ],
    )
    turns = [
        SpeakerTurn(start=0.0, end=math.inf, speaker="BOGUS"),
        SpeakerTurn(start=0.0, end=1.0, speaker="A"),
        SpeakerTurn(start=10.0, end=11.0, speaker="B"),
    ]
    labeled = apply_speaker_labels(result, turns)
    assert [segment.speaker for segment in labeled.segments] == ["Speaker 1", "Speaker 2"]


# --- B17: speaker alignment interval work budget ------------------------------------------


def _scan_choice(start, end, turns):
    """The previous full-scan selection, kept as an oracle."""
    candidates = []
    for index, turn in enumerate(turns):
        overlap = min(end, turn.end) - max(start, turn.start)
        if overlap > 0:
            candidates.append((overlap, turn.start, turn.end, turn.speaker, index))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[3], item[4]))
    return candidates[0][3]


def _scan_overlaps(start, end, turns):
    overlaps: dict[str, float] = {}
    for turn in turns:
        overlap = min(end, turn.end) - max(start, turn.start)
        if overlap > 0:
            overlaps[turn.speaker] = overlaps.get(turn.speaker, 0.0) + overlap
    return overlaps


def test_non_finite_segment_times_align_to_nothing() -> None:
    turns = normalize_speaker_turns([SpeakerTurn(start=0.0, end=5.0, speaker="A")])
    index = SpeakerTurnIndex(turns)
    for start, end in ((math.nan, 1.0), (0.0, math.nan), (0.0, math.inf)):
        assert choose_speaker_for_segment(segment_start=start, segment_end=end, turns=index) is None
        assert choose_speaker_for_segment(segment_start=start, segment_end=end, turns=turns) is None
        assert raw_turn_overlaps(start, end, index) == {}


def test_turn_index_matches_full_scan() -> None:
    import random

    rng = random.Random(85)
    raw = []
    for _ in range(400):
        start = rng.uniform(0, 600)
        raw.append(
            SpeakerTurn(start=start, end=start + rng.uniform(0.01, 40), speaker=rng.choice("ABCDE"))
        )
    raw.append(SpeakerTurn(start=0.0, end=599.0, speaker="L"))  # one long overlap
    turns = normalize_speaker_turns(raw)
    index = SpeakerTurnIndex(turns)
    for _ in range(500):
        start = rng.uniform(-5, 650)
        end = start + rng.uniform(0.001, 30)
        assert choose_speaker_for_segment(
            segment_start=start, segment_end=end, turns=index
        ) == _scan_choice(start, end, turns)
        assert raw_turn_overlaps(start, end, index) == _scan_overlaps(start, end, turns)
    # Touching boundaries do not overlap, exactly as before.
    edge = turns[0]
    assert choose_speaker_for_segment(
        segment_start=edge.end, segment_end=edge.end + 0.0, turns=index
    ) == _scan_choice(edge.end, edge.end, turns)


def test_alignment_work_is_bounded_by_overlaps(monkeypatch) -> None:
    """Many segments against many disjoint turns visit only nearby turns."""
    count = 4000
    turns = normalize_speaker_turns(
        [SpeakerTurn(start=float(i), end=i + 0.9, speaker=f"S{i % 3}") for i in range(count)]
    )
    segments = [
        TranscriptionSegment(id=i, start=i + 0.1, end=i + 0.5, text="w") for i in range(count)
    ]
    visited = 0
    import transcriptions.diarization as diarization

    original = diarization.overlapping_turns

    def counting(turns_arg, start, end):
        nonlocal visited
        found = original(turns_arg, start, end)
        visited += len(found)
        return found

    monkeypatch.setattr(diarization, "overlapping_turns", counting)
    labeled = apply_speaker_labels(
        TranscriptionResult(text="", language="en", segments=segments), turns
    )
    assert all(segment.speaker for segment in labeled.segments)
    # A full scan would visit count * count = 16,000,000 turns.
    assert visited == count


# --- B17: chapter word time domain ------------------------------------------------------------


def _words(offsets: list[tuple[int, int]]) -> bytes:
    return json.dumps(
        {
            "transcription": [
                {"offsets": {"from": a, "to": b}, "text": f" w{i}"}
                for i, (a, b) in enumerate(offsets)
            ]
        }
    ).encode()


@pytest.mark.parametrize(
    "offsets",
    [[(-1, 10)], [(50, 10)], [(0, 10**18)], [(0, 10**400)], [(0, 121_001)], [(120_500, 240_000)]],
    ids=["negative", "inverted", "huge", "overflow", "past-tolerance", "past-clip"],
)
def test_chapter_word_offsets_outside_the_clip_fail_the_window(offsets) -> None:
    with pytest.raises(RuntimeError, match="outside the clip"):
        chapter_windows.chapter_window_words(_words(offsets), clip_start=10.0, clip_seconds=120.0)


def test_chapter_word_offsets_at_clip_edges_are_kept() -> None:
    # 120,100 ms: the real G6 fixture's overshoot (atp-t7#2s, "felt."); 121,000 ms: the
    # documented tolerance edge.
    words = chapter_windows.chapter_window_words(
        _words([(0, 0), (119_550, 120_100), (120_000, 121_000)]),
        clip_start=10.0,
        clip_seconds=120.0,
    )
    assert [(w["s"], w["e"]) for w in words] == [(10.0, 10.0), (129.55, 130.1), (130.0, 131.0)]


def test_rejected_offsets_leave_the_source_bytes_intact() -> None:
    raw = _words([(0, 10), (0, 10**400)])
    before = bytes(raw)
    with pytest.raises(RuntimeError, match="outside the clip"):
        chapter_windows.chapter_window_words(raw, clip_start=0.0, clip_seconds=120.0)
    assert raw == before


def test_blank_segments_are_skipped_before_the_offset_check() -> None:
    raw = json.dumps(
        {
            "transcription": [
                {"offsets": {"from": 500, "to": -3}, "text": "  "},
                {"offsets": {"from": 0, "to": 10}, "text": " ok"},
            ]
        }
    ).encode()
    words = chapter_windows.chapter_window_words(raw, clip_start=0.0, clip_seconds=120.0)
    assert [w["w"] for w in words] == ["ok"]


@pytest.mark.parametrize(
    "raw", [b'{"transcription": [], "x": ' + b"9" * 5000 + b"}"], ids=["digit-limit"]
)
def test_chapter_window_parser_limits_fail_the_window(raw) -> None:
    with pytest.raises(RuntimeError):
        chapter_windows.chapter_window_words(raw, clip_start=0.0)


# --- B18: remote manifest preflight and artifact field domain ---------------------------------


def _complete(client, job, claim, manifest):
    return client.post(
        f"/v1/internal/work/{job.id}/complete",
        data=json.dumps(
            {
                "worker_id": "atlas",
                "lease_token": claim["lease_token"],
                "result_text": "remote transcript",
                "result_metadata": {"backend": "whisper.cpp"},
                "artifacts": manifest,
            }
        ),
        content_type="application/json",
        **worker_headers(),
    )


@pytest.mark.django_db
def test_oversized_manifest_is_refused_before_store_access(client, settings, monkeypatch):
    job = submit_remote_job(client, settings, task_ref="manifest-budget")
    claim = claim_one(client).json()["job"]
    manifest = transcript_manifest(str(job.id), claim["attempt"])
    filler = dict(manifest[0])
    manifest = manifest + [
        {**filler, "name": f"extra-{i}.mp3", "storage_key": f"{filler['storage_key']}-{i}"}
        for i in range(10_000)
    ]
    from jobs import remote_workers

    def no_store(*args, **kwargs):
        pytest.fail("store must not be consulted for an oversized manifest")

    monkeypatch.setattr(remote_workers, "get_artifact_store_for_identity", no_store)
    response = _complete(client, job, claim, manifest)
    assert response.status_code == 400
    assert "at most 32" in response.json()["error"]["message"]
    assert not JobArtifact.objects.filter(job=job).exists()


@pytest.mark.django_db
def test_slow_store_exhausts_the_cumulative_verification_budget(client, settings, monkeypatch):
    from jobs import remote_workers

    job = submit_remote_job(client, settings, task_ref="manifest-slow-store")
    claim = claim_one(client).json()["job"]
    manifest = transcript_manifest(str(job.id), claim["attempt"])
    store_manifest_artifacts(settings, manifest)
    assert len(manifest) >= 2
    clock = [1000.0]
    # Only remote_workers sees the fake clock; the process-wide time module is untouched.
    monkeypatch.setattr(remote_workers, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    original = remote_workers.get_artifact_store_for_identity
    stats: list[str] = []

    def slow_store(identity):
        store = original(identity)

        class Slow:
            def stat(self, *, key):
                stats.append(key)
                clock[0] += remote_workers.REMOTE_ARTIFACT_VERIFY_BUDGET_SECONDS
                return store.stat(key=key)

        return Slow()

    monkeypatch.setattr(remote_workers, "get_artifact_store_for_identity", slow_store)
    response = _complete(client, job, claim, manifest)
    assert response.status_code == 400
    assert "budget" in response.json()["error"]["message"]
    assert len(stats) == 1
    job.refresh_from_db()
    assert job.state == "running"
    assert not JobArtifact.objects.filter(job=job).exists()

    # The same manifest completes when the store answers within the budget.
    monkeypatch.setattr(remote_workers, "get_artifact_store_for_identity", original)
    response = _complete(client, job, claim, manifest)
    assert response.status_code == 200, response.content
    assert JobArtifact.objects.filter(job=job).count() == len(manifest)


@pytest.mark.django_db
@pytest.mark.parametrize(
    "field,value,message",
    [
        ("size_bytes", 2**63, "64-bit"),
        ("content_type", "audio/" + "x" * 300, "at most 255"),
        ("name", "a" * 256, "at most 255"),
        ("content_type", "audio/mpeg\x00", "NUL"),
    ],
    ids=["size-overflow", "content-type-length", "name-length", "nul"],
)
def test_artifact_fields_must_fit_the_model(client, settings, field, value, message):
    job = submit_remote_job(client, settings, task_ref=f"field-domain-{field}-{len(str(value))}")
    claim = claim_one(client).json()["job"]
    manifest = transcript_manifest(str(job.id), claim["attempt"])
    store_manifest_artifacts(settings, manifest)
    manifest[0][field] = value
    response = _complete(client, job, claim, manifest)
    assert response.status_code == 400
    assert message in response.json()["error"]["message"]
    assert Job.objects.get(id=job.id).state == Job.State.RUNNING


@pytest.mark.django_db
@pytest.mark.parametrize("variant", ["identical", "double-slash", "dot-segment"])
def test_artifacts_cannot_share_a_storage_object(client, settings, variant):
    job = submit_remote_job(client, settings, task_ref=f"key-unique-{variant}")
    claim = claim_one(client).json()["job"]
    manifest = transcript_manifest(str(job.id), claim["attempt"])
    store_manifest_artifacts(settings, manifest)
    first, second = manifest[0], manifest[1]
    head, _, tail = str(first["storage_key"]).rpartition("/")
    alias = {
        "identical": first["storage_key"],
        "double-slash": f"{head}//{tail}",
        "dot-segment": f"{head}/./{tail}",
    }[variant]
    second["storage_key"] = alias
    second["size_bytes"] = first["size_bytes"]
    response = _complete(client, job, claim, manifest)
    assert response.status_code == 400
    assert "storage_key values must be unique" in response.json()["error"]["message"]
    assert not JobArtifact.objects.filter(job=job).exists()


# --- B19: window native JSON admission -----------------------------------------------------------


def test_window_json_over_the_cap_is_refused(tmp_path) -> None:
    path = tmp_path / "out.json"
    path.write_bytes(b" " * 2049)
    with pytest.raises(RuntimeError, match="window limit"):
        windows.read_window_json(path, limit=2048)
    path.write_bytes(b" " * 2048)
    assert len(windows.read_window_json(path, limit=2048)) == 2048


def test_window_json_fifo_is_refused_without_blocking(tmp_path) -> None:
    path = tmp_path / "out.json"
    os.mkfifo(path)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="not a regular file"):
        windows.read_window_json(path, limit=2048)
    assert time.monotonic() - started < 5


def test_window_json_symlink_and_directory_are_refused(tmp_path) -> None:
    target = tmp_path / "elsewhere.json"
    target.write_bytes(b"{}")
    link = tmp_path / "out.json"
    link.symlink_to(target)
    with pytest.raises(RuntimeError, match="not a regular file"):
        windows.read_window_json(link, limit=2048)
    directory = tmp_path / "dir.json"
    directory.mkdir()
    with pytest.raises((RuntimeError, OSError)):
        windows.read_window_json(directory, limit=2048)


def test_missing_window_json_is_still_an_os_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        windows.read_window_json(tmp_path / "missing.json", limit=2048)


# --- B19: normalizer owned process diagnostic budget ---


def _pid_gone(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def test_noisy_successful_normalization_drains_bounded_diagnostics(
    tmp_path, settings, caplog
) -> None:
    """Damaged but decodable media may log a lot; the decode still succeeds."""
    argv_file = tmp_path / "argv"
    fake = tmp_path / "ffmpeg"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys\n"
        f"pathlib.Path({str(argv_file)!r}).write_text(json.dumps(sys.argv))\n"
        "for _ in range(200):\n"
        "    sys.stderr.write('/private/secret ' + 'x' * 8192 + '\\n')\n"
        "pathlib.Path(sys.argv[-1]).write_bytes(b'RIFF')\n"
    )
    fake.chmod(0o755)
    settings.VOXHELM_FFMPEG_BIN = str(fake)
    output = tmp_path / "out.wav"
    normalize_audio_to_wav_16k_mono(input_path=tmp_path / "in.mp3", output_path=output)
    assert output.read_bytes() == b"RIFF"
    argv = json.loads(argv_file.read_text())
    assert argv[argv.index("-loglevel") + 1] == "error"
    assert "-hide_banner" in argv and "-nostats" in argv


def test_failed_normalization_keeps_diagnostics_out_of_the_error(
    tmp_path, settings, caplog
) -> None:
    fake = tmp_path / "ffmpeg"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "sys.stderr.write('/private/secret ' + 'x' * (1024 * 1024))\n"
        "raise SystemExit(1)\n"
    )
    fake.chmod(0o755)
    settings.VOXHELM_FFMPEG_BIN = str(fake)
    with caplog.at_level("WARNING", logger="transcriptions.owned_process"):
        with pytest.raises(RuntimeError) as caught:
            normalize_audio_to_wav_16k_mono(
                input_path=tmp_path / "in.mp3", output_path=tmp_path / "out.wav"
            )
    assert str(caught.value) == "ffmpeg audio normalization failed."
    # Only a bounded prefix reaches the operator log.
    assert "/private/secret" in caplog.text
    assert len(caplog.text) < 8192


def test_stalled_noisy_normalization_is_killed_at_the_deadline(tmp_path, settings) -> None:
    pid_file = tmp_path / "pid"
    fake = tmp_path / "ffmpeg"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, sys, time\n"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
        "sys.stderr.write('x' * (1024 * 1024)); sys.stderr.flush()\n"
        "time.sleep(30)\n"
    )
    fake.chmod(0o755)
    settings.VOXHELM_FFMPEG_BIN = str(fake)
    settings.VOXHELM_AUDIO_DECODE_TIMEOUT_SECONDS = 1
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="configured decode deadline"):
        normalize_audio_to_wav_16k_mono(
            input_path=tmp_path / "in.mp3", output_path=tmp_path / "out.wav"
        )
    assert time.monotonic() - started < 10
    assert _pid_gone(int(pid_file.read_text()))


def test_normalizer_kills_descendants_holding_its_pipes(tmp_path, settings) -> None:
    child_pid = tmp_path / "child"
    fake = tmp_path / "ffmpeg"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, subprocess, sys\n"
        "pathlib.Path(sys.argv[-1]).write_bytes(b'RIFF')\n"
        f"c = subprocess.Popen([{sys.executable!r}, '-c', 'import time; time.sleep(60)'])\n"
        f"pathlib.Path({str(child_pid)!r}).write_text(str(c.pid))\n"
    )
    fake.chmod(0o755)
    settings.VOXHELM_FFMPEG_BIN = str(fake)
    settings.VOXHELM_AUDIO_DECODE_TIMEOUT_SECONDS = 1
    with pytest.raises(RuntimeError, match="configured decode deadline"):
        normalize_audio_to_wav_16k_mono(
            input_path=tmp_path / "in.mp3", output_path=tmp_path / "out.wav"
        )
    assert _pid_gone(int(child_pid.read_text()))


# --- B19: staged cleanup failure fairness ---


def _expired_upload(index: int, now) -> StagedMedia:
    return StagedMedia.objects.create(
        producer="synthetic",
        original_filename=f"u{index}.mp3",
        content_type="audio/mpeg",
        storage_backend="filesystem",
        storage_key=f"voxhelm/staged/synthetic/u{index}.mp3",
        expires_at=now - timedelta(hours=10 - index),
    )


@pytest.mark.django_db
def test_failing_cleanup_rows_do_not_starve_newer_expired_uploads(monkeypatch) -> None:
    now = timezone.now()
    failing = [_expired_upload(i, now) for i in range(2)]
    healthy = [_expired_upload(i, now) for i in range(2, 6)]
    failing_ids = {row.pk for row in failing}

    def delete(*, staged, missing_ok):
        if staged.pk in failing_ids:
            raise OSError("synthetic storage failure")
        staged.delete()

    monkeypatch.setattr(jobs_staging, "delete_staged_media", delete)
    for _ in range(4):
        jobs_staging.cleanup_expired_staged_media(limit=2)
    remaining = set(StagedMedia.objects.values_list("pk", flat=True))
    assert remaining == failing_ids
    # Deferred rows stay expired, so a later sweep still retries them.
    assert all(row.expires_at <= timezone.now() for row in StagedMedia.objects.all())
    assert not any(row.pk in remaining for row in healthy)
