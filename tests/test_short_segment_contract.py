"""The short-segment threshold has the same domain and value on both executors."""

import pytest

from jobs.models import Job
from jobs.remote_worker_cli import known_speaker_config_from_claim
from jobs.remote_workers import sanitize_known_speaker_summary, serialize_claim_diarization
from jobs.services import known_speaker_config_for_job, parse_known_speaker_config
from tests.test_known_speaker import known_speaker_payload
from tests.test_remote_workers import claim_one, known_speaker_capabilities, submit_remote_job
from transcriptions.errors import ApiError
from transcriptions.known_speaker import KnownSpeakerConfig, known_speaker_config_from_mapping


@pytest.mark.parametrize("margin", [0, 0.15, 0.3, 3])
def test_threshold_api_storage_and_claim_share_the_same_config(margin):
    parsed = parse_known_speaker_config({"short_segment_margin": margin})
    job = Job(output_data={"diarization": {"known_speaker": parsed}})
    local = known_speaker_config_for_job(job)
    remote = known_speaker_config_from_claim(serialize_claim_diarization(job))
    assert local == remote == KnownSpeakerConfig(short_segment_margin=margin)
    assert parsed == {"short_segment_margin": float(margin)}


@pytest.mark.parametrize("value", [True, "0.15", None, -1, float("inf"), float("nan"), 10**400])
def test_threshold_rejected_at_api_boundary(value):
    with pytest.raises(ApiError, match="short_segment_margin"):
        parse_known_speaker_config({"short_segment_margin": value})


def test_omitted_and_explicit_default_execute_identically():
    assert known_speaker_config_from_mapping({}) == known_speaker_config_from_mapping(
        {"short_segment_margin": 0.15}
    )


@pytest.mark.parametrize(
    "skip", [None, "disabled", "no_raw_turns", "unresolved_label", "too_few_centroids"]
)
def test_new_summary_fields_survive_remote_completion_allowlist(skip):
    summary = {
        "short_segment_margin": 0.15,
        "episode_voiceprint_segment_count": 4,
        "episode_voiceprint_relabelled_count": 2,
        "episode_voiceprint_skipped": skip,
    }
    assert sanitize_known_speaker_summary(summary, job=Job(output_data={})) == summary


def test_remote_summary_rejects_unrecognized_skip_reason():
    assert (
        sanitize_known_speaker_summary(
            {"episode_voiceprint_skipped": "private diagnostic"}, job=Job(output_data={})
        )
        == {}
    )


@pytest.mark.django_db
def test_submission_persists_claims_and_fingerprints_threshold(client, settings):
    def submit(margin):
        return submit_remote_job(
            client,
            settings,
            task_ref="same-threshold-task",
            diarization=known_speaker_payload(known_speaker={"short_segment_margin": margin})[
                "diarization"
            ],
        )

    first, second = submit(0), submit(0.3)
    assert first.id != second.id
    assert first.request_fingerprint != second.request_fingerprint
    assert first.output_data["diarization"]["known_speaker"]["short_segment_margin"] == 0
    assert second.output_data["diarization"]["known_speaker"]["short_segment_margin"] == 0.3
    claim = claim_one(client, capabilities=known_speaker_capabilities()).json()["job"]
    stored = Job.objects.get(id=claim["id"])
    claimed = claim["output"]["diarization"]
    assert known_speaker_config_from_claim(claimed) == known_speaker_config_for_job(stored)
