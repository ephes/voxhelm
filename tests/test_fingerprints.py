from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any

import pytest

from jobs.fingerprints import (
    job_request_fingerprint_from_fields,
    stored_job_request_fingerprint,
)
from jobs.models import Job
from jobs.services import JobRequest, existing_job_matches_request, job_request_fingerprint

URL = "https://media.example.test/Grüße/声.wav"
UPLOAD_ID = "00000000-0000-4000-8000-000000000001"

# Handwritten persisted wire identities. Do not derive these strings from production
# canonicalization: sorted keys, compact separators and literal UTF-8 are the contract.
LEGACY_JSON = (
    '{"backend":"auto","diarization":{"enabled":false},'
    '"input":{"kind":"url","url":"https://media.example.test/Grüße/声.wav"},'
    '"job_type":"transcribe","language":null,"model":"whisper-1",'
    '"output_formats":["json","text"]}'
)
PROFILE_JSON = (
    '{"backend":"auto","diarization":{"enabled":false},'
    '"input":{"kind":"url","url":"https://media.example.test/Grüße/声.wav"},'
    '"job_type":"transcribe","language":null,"model":"auto",'
    '"output_formats":["json","text"],"profile":"ad-analysis-v1"}'
)
CHAPTER_JSON = (
    '{"backend":"auto","chapter_marks_ms":[0,1234,5678],"diarization":{"enabled":false},'
    '"input":{"kind":"url","url":"https://media.example.test/Grüße/声.wav"},'
    '"job_type":"transcribe","language":null,"model":"auto",'
    '"output_formats":["json","text"],"profile":"ad-analysis-v1"}'
)


def request_fixture(**changes: Any) -> JobRequest:
    return replace(
        JobRequest(
            job_type="transcribe",
            priority="normal",
            lane="batch",
            backend="auto",
            model="whisper-1",
            language=None,
            input_data={"url": URL, "kind": "url"},
            output_formats=["text", "json"],
            diarization={"enabled": False},
            context={},
            task_ref="synthetic-ref",
        ),
        **changes,
    )


def stored_fixture(request: JobRequest) -> Job:
    output_data: dict[str, Any] = {"formats": list(request.output_formats)}
    # Legacy rows omit default diarization and optional chapter marks.
    if request.diarization != {"enabled": False}:
        output_data["diarization"] = request.diarization
    if request.chapter_marks_ms is not None:
        output_data["chapter_marks_ms"] = request.chapter_marks_ms
    return Job(
        job_type=request.job_type,
        backend=request.backend,
        model=request.model,
        language=request.language or "",
        input_data=request.input_data,
        output_data=output_data,
        profile=request.profile,
    )


def stored_digest(job: Job) -> str:
    return stored_job_request_fingerprint(
        job_type=job.job_type,
        backend=job.backend,
        model=job.model,
        language=job.language,
        input_data=job.input_data,
        output_data=job.output_data,
        profile=job.profile,
    )


@pytest.mark.parametrize(
    "changes,canonical,expected",
    [
        ({}, LEGACY_JSON, "120bfff07a6304856f9370d044e6e51822b7ad4672da01556b36424c3e6fc0e6"),
        (
            {"language": "", "diarization": None, "chapter_marks_ms": []},
            LEGACY_JSON,
            "120bfff07a6304856f9370d044e6e51822b7ad4672da01556b36424c3e6fc0e6",
        ),
        (
            {"model": "auto", "profile": "ad-analysis-v1"},
            PROFILE_JSON,
            "4f1e6806491c56b8c563d8a15411107507779edf9b4bcc30ae0264ac4ff689fa",
        ),
        (
            {"model": "auto", "profile": "ad-analysis-v1", "chapter_marks_ms": [0, 1234, 5678]},
            CHAPTER_JSON,
            "f05776f05b3c7b5365e8e050d158cff65dae09724262165e6fb1685da0b7a5ce",
        ),
        (
            {"job_type": "synthesize", "input_data": {"text": "声🌍"}, "output_formats": ["wav"]},
            '{"job_type":"synthesize"}',
            "0221a94a0ad8df69934d9cc624cffc797b2f3ce12e386fe02f224d7221aa709d",
        ),
    ],
    ids=["legacy", "empty-defaults", "profile", "chapters", "synthesis"],
)
def test_persisted_fingerprint_golden_bytes(changes, canonical, expected):
    fields = {
        "job_type": "transcribe",
        "backend": "auto",
        "model": "whisper-1",
        "language": None,
        "input_data": {"url": URL, "kind": "url"},
        "output_formats": ["text", "json"],
        "diarization": {"enabled": False},
        "profile": "",
        "chapter_marks_ms": None,
        **changes,
    }
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == expected
    assert job_request_fingerprint_from_fields(**fields) == expected
    # Exercise the persisted-row adapter independently of the request adapter too.
    output_data = {"formats": fields["output_formats"]}
    if fields["diarization"] is not None:
        output_data["diarization"] = fields["diarization"]
    if fields["chapter_marks_ms"] is not None:
        output_data["chapter_marks_ms"] = fields["chapter_marks_ms"]
    assert (
        stored_job_request_fingerprint(
            job_type=fields["job_type"],
            backend=fields["backend"],
            model=fields["model"],
            language=fields["language"] or "",
            input_data=fields["input_data"],
            output_data=output_data,
            profile=fields["profile"],
        )
        == expected
    )


@pytest.mark.parametrize(
    "changes,equivalent",
    [
        ({}, True),
        ({"backend": "whispercpp"}, False),
        ({"model": "ggml-large-v3.bin"}, False),
        ({"language": "de"}, False),
        ({"input_data": {"kind": "url", "url": URL + "?revision=2"}}, False),
        ({"input_data": {"kind": "upload", "upload_id": UPLOAD_ID}}, False),
        ({"input_data": {"kind": "url", "url": URL}}, True),
        ({"output_formats": ["json", "text"]}, True),
        ({"output_formats": ["json", "text", "vtt"]}, False),
        ({"diarization": {"enabled": True}}, False),
        ({"profile": "ad-analysis-v1", "model": "auto"}, False),
        ({"priority": "high"}, True),
        ({"context": {"synthetic": "nonidentity"}}, True),
        ({"task_ref": "another-ref"}, True),
        ({"job_type": "synthesize"}, False),
    ],
    ids=[
        "same",
        "backend",
        "model",
        "language",
        "url",
        "input-kind",
        "input-key-order",
        "format-order",
        "formats",
        "diarization",
        "profile",
        "priority",
        "context",
        "task-ref",
        "job-type",
    ],
)
def test_transcription_matcher_and_fingerprint_equivalence(changes, equivalent):
    baseline = request_fixture()
    existing = stored_fixture(baseline)
    request = replace(baseline, **changes)

    assert existing_job_matches_request(existing, request) is equivalent
    assert (stored_digest(existing) == job_request_fingerprint(request)) is equivalent


@pytest.mark.parametrize(
    "marks,equivalent",
    [(None, False), ([0, 1234, 5678], True), ([0, 1235, 5678], False)],
)
def test_ordered_chapter_identity(marks, equivalent):
    baseline = request_fixture(
        model="auto", profile="ad-analysis-v1", chapter_marks_ms=[0, 1234, 5678]
    )
    existing = stored_fixture(baseline)
    request = replace(baseline, chapter_marks_ms=marks)

    assert existing_job_matches_request(existing, request) is equivalent
    assert (stored_digest(existing) == job_request_fingerprint(request)) is equivalent


def test_profile_identity_with_other_fields_unchanged():
    baseline = request_fixture(model="auto")
    existing = stored_fixture(baseline)
    request = replace(baseline, profile="ad-analysis-v1")

    assert existing_job_matches_request(existing, request) is False
    assert stored_digest(existing) != job_request_fingerprint(request)


@pytest.mark.parametrize("speakers", [2, 3])
def test_diarization_options_are_identity(speakers):
    baseline = request_fixture(diarization={"enabled": True, "num_speakers": 2})
    existing = stored_fixture(baseline)
    request = replace(baseline, diarization={"enabled": True, "num_speakers": speakers})

    assert existing_job_matches_request(existing, request) is (speakers == 2)
    assert (stored_digest(existing) == job_request_fingerprint(request)) is (speakers == 2)


@pytest.mark.parametrize("same_upload", [False, True])
def test_upload_staging_metadata_is_not_identity(same_upload):
    baseline = request_fixture(input_data={"kind": "upload", "upload_id": UPLOAD_ID})
    existing = stored_fixture(baseline)
    existing.input_data = {
        **existing.input_data,
        "filename": "private-synthetic.wav",
        "size_bytes": 123,
        "storage_key": "synthetic/staging-key",
        "content_type": "audio/wav",
    }
    existing.output_data["artifact_metadata"] = {"bytes": 999}
    request = replace(
        baseline,
        input_data={
            "kind": "upload",
            "upload_id": UPLOAD_ID if same_upload else "00000000-0000-4000-8000-000000000002",
            "filename": "renamed.wav",
        },
    )

    assert existing_job_matches_request(existing, request) is same_upload
    assert (stored_digest(existing) == job_request_fingerprint(request)) is same_upload


@pytest.mark.parametrize(
    "changes",
    [
        {"backend": "other"},
        {"model": "other"},
        {"language": "de"},
        {"input_data": {"text": "A different synthetic utterance", "voice": "other"}},
        {"output_formats": ["mp3"]},
    ],
)
def test_synthesis_retains_looser_job_type_identity(changes):
    baseline = request_fixture(
        job_type="synthesize",
        backend="piper",
        model="",
        input_data={"text": "Synthetic", "voice": "synthetic"},
        output_formats=["wav"],
    )
    existing = stored_fixture(baseline)
    request = replace(baseline, **changes)

    assert existing_job_matches_request(existing, request) is True
    assert (
        job_request_fingerprint(request)
        == stored_digest(existing)
        == ("0221a94a0ad8df69934d9cc624cffc797b2f3ce12e386fe02f224d7221aa709d")
    )
