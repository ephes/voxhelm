"""The ``ad-analysis-v1`` transcription profile: submission, routing without fallback,
words artifact and identity, remote workers, producer DELETE and the 7-day backstop."""

from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from django.utils import timezone

from jobs import remote_worker_cli as worker_cli
from jobs import retention
from jobs.artifacts import current_artifact_store_identity, get_artifact_store
from jobs.models import (
    Job,
    JobArtifact,
    PendingArtifactDeletion,
    RemoteAttemptPrefix,
    StagedMedia,
)
from jobs.retention import prune_job_artifacts
from tests.test_jobs import configure_task_backend
from tests.test_remote_workers import (
    claim_one,
    enable_remote_workers,
    plain_capabilities,
    producer_headers,
    stage_upload,
    store_manifest_artifacts,
    transcript_manifest,
    worker_headers,
)
from transcriptions import profiles
from transcriptions.profiles import (
    AD_ANALYSIS_V1,
    PROFILES,
    model_revision,
    parakeet_payload_words,
)
from transcriptions.service import BackendUnavailableError

FIXTURE = Path(__file__).parent / "fixtures" / "parakeet_excerpt.json"
FAKE_CLI = """#!{python}
import json, sys
from pathlib import Path

args = sys.argv[1:]
if args == ["--version"]:
    print("parakeet-mlx 0.5.3")
    sys.exit(0)
Path({record!r}).write_text(json.dumps(args))
out = Path(args[args.index("--output-dir") + 1])
out.mkdir(parents=True, exist_ok=True)
(out / "transcript.json").write_text(Path({fixture!r}).read_text())
"""


@pytest.fixture
def parakeet_cli(settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A fake ``parakeet-mlx`` that records its arguments and writes the recorded fixture."""
    model_dir = tmp_path / "parakeet" / "ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights")
    (model_dir / "VOXHELM_MODEL_REVISION").write_text(
        "repo=mlx-community/parakeet-tdt-0.6b-v3\n"
        "revision=ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15\n"
        "model.safetensors sha256=05e01c7f\n",
        encoding="utf-8",
    )
    record = tmp_path / "args.json"
    cli = tmp_path / "parakeet-mlx"
    cli.write_text(
        FAKE_CLI.format(python=sys.executable, record=str(record), fixture=str(FIXTURE)),
        encoding="utf-8",
    )
    cli.chmod(0o755)

    def fake_normalize(*, input_path: Path, output_path: Path) -> None:
        del input_path
        output_path.write_bytes(b"RIFFfakewav")

    monkeypatch.setattr("transcriptions.service.normalize_audio_to_wav_16k_mono", fake_normalize)
    # The deployment's known-speaker settings must not leak into the profile.
    settings.VOXHELM_PARAKEET_BIN = str(cli)
    settings.VOXHELM_PARAKEET_MODEL_DIR = str(model_dir)
    settings.VOXHELM_PARAKEET_MODEL_NAME = "parakeet-tdt-0.6b-v3"
    settings.VOXHELM_PARAKEET_CHUNK_SECONDS = 600
    settings.VOXHELM_PARAKEET_OVERLAP_SECONDS = 30
    settings.VOXHELM_PARAKEET_MAX_SENTENCE_SECONDS = 30.0
    settings.VOXHELM_STT_BACKEND = "whispercpp"
    settings.VOXHELM_STT_FALLBACK_BACKEND = "mlx"
    settings.VOXHELM_WHISPERCPP_MODEL = "ggml-large-v3.bin"
    settings.VOXHELM_KNOWN_SPEAKER_STT_BACKEND = "parakeet"
    settings.VOXHELM_BATCH_MAX_STAGED_UPLOAD_BYTES = 1024
    profiles._cli_versions.cache_clear()
    calls: list[str] = []

    def no_other_engine(*, backend_name: str, model_name: str):
        calls.append(backend_name)
        raise AssertionError(f"profile job built the {backend_name} backend")

    monkeypatch.setattr("transcriptions.service.build_backend_service", no_other_engine)
    monkeypatch.setattr(worker_cli, "build_backend_service", no_other_engine)

    class Runtime:
        pass

    runtime = Runtime()
    runtime.model_dir = model_dir  # type: ignore[attr-defined]
    runtime.args = lambda: json.loads(record.read_text(encoding="utf-8"))  # type: ignore[attr-defined]
    runtime.other_engine_calls = calls  # type: ignore[attr-defined]
    yield runtime
    profiles._cli_versions.cache_clear()


def profile_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "job_type": "transcribe",
        "profile": AD_ANALYSIS_V1,
        "language_hint": "de",
        "input": {"kind": "url", "url": "https://media.example.com/episode.mp3"},
        "output": {"formats": ["json"]},
        "task_ref": "smart-skip-abc",
    }
    payload.update(overrides)
    return {key: value for key, value in payload.items() if value is not None}


def post_job(client, payload: dict[str, Any], token: str = "test-token"):
    return client.post(
        "/v1/jobs",
        data=json.dumps(payload),
        content_type="application/json",
        HTTP_AUTHORIZATION=f"Bearer {token}",
    )


# --- submission ------------------------------------------------------------------


@pytest.mark.django_db
def test_profile_job_is_stored_with_the_normalized_language_hint(client, settings):
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}
    response = post_job(client, profile_payload(language_hint="de-DE"))

    assert response.status_code == 201
    assert response.json()["profile"] == AD_ANALYSIS_V1
    job = Job.objects.get(id=response.json()["id"])
    assert (job.profile, job.language, job.model, job.backend) == (
        AD_ANALYSIS_V1,
        "de",
        "auto",
        "auto",
    )


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"profile": "ad-analysis-v0"}, "Unsupported profile"),
        ({"language": "de", "language_hint": None}, "language_hint instead of language"),
        ({"model": "whisper-1"}, "choose their own model"),
        ({"language_hint": "ja"}, "not supported by the profile"),
        ({"diarization": {"enabled": True}}, "do not support diarization"),
        ({"profile": None}, "language_hint requires a profile"),
    ],
)
def test_profile_submission_rejects_unsupported_combinations(
    client, settings, overrides: dict[str, Any], message: str
):
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}
    response = post_job(client, profile_payload(**overrides))

    assert response.status_code == 400
    assert message in response.json()["error"]["message"]
    assert not Job.objects.exists()


@pytest.mark.django_db
def test_profile_is_part_of_task_ref_identity(client, settings):
    settings.VOXHELM_ALLOWED_URL_HOSTS = {"media.example.com"}
    first = post_job(client, profile_payload())
    again = post_job(client, profile_payload())
    plain = post_job(client, profile_payload(profile=None, language_hint=None))

    assert (first.status_code, again.status_code, plain.status_code) == (201, 200, 201)
    assert again.json()["id"] == first.json()["id"]
    assert plain.json()["id"] != first.json()["id"]
    assert "profile" not in plain.json()


# --- local execution ---------------------------------------------------------------


def run_local_profile_job(client, settings, **overrides: Any) -> dict[str, Any]:
    configure_task_backend(settings, "django_tasks.backends.immediate.ImmediateBackend")
    upload_id = stage_upload(client, content=b"episode-bytes", content_type="audio/mpeg").json()[
        "id"
    ]
    response = post_job(
        client, profile_payload(input={"kind": "upload", "upload_id": upload_id}, **overrides)
    )
    assert response.status_code == 201
    return response.json()


@pytest.mark.django_db
def test_local_profile_job_runs_parakeet_with_the_profile_settings_only(
    client, settings, parakeet_cli
):
    payload = run_local_profile_job(client, settings)

    assert payload["state"] == "succeeded"
    args = parakeet_cli.args()
    # 30 s chunks, 5 s overlap, full attention, and no --max-duration even though the
    # deployment's own Parakeet settings say 600 / 30 / 30 s.
    assert args[args.index("--chunk-duration") + 1] == "30"
    assert args[args.index("--overlap-duration") + 1] == "5"
    assert "--full-attention" in args
    assert "--local-attention" not in args
    assert "--max-duration" not in args
    assert parakeet_cli.other_engine_calls == []

    metadata = payload["result"]["metadata"]
    assert metadata["backend"] == "parakeet"
    identity = metadata["profile"]
    assert identity["profile"] == AD_ANALYSIS_V1
    assert identity["model"] == "parakeet-tdt-0.6b-v3"
    assert identity["model_repo"] == "mlx-community/parakeet-tdt-0.6b-v3"
    assert identity["model_revision"] == "ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15"
    assert identity["model_sha256"] == "05e01c7f"
    assert identity["cli_version"] == "0.5.3"
    assert identity["language_hint"] == "de"
    assert identity["settings"] == {
        "chunk_seconds": 30,
        "overlap_seconds": 5,
        "local_attention": False,
        "max_sentence_seconds": None,
        "decoding": "greedy",
        "precision": "bf16",
    }

    words_url = payload["result"]["artifacts"]["words"]
    assert words_url.endswith("/words.json")
    document = json.loads(client.get(words_url, **producer_headers()).content)
    assert document["schema_version"] == 1
    assert document["profile"] == AD_ANALYSIS_V1
    assert document["identity"] == identity
    assert document["words"][:2] == [
        # " Hi" (0.987) and "," (0.959): mean confidence, end of the last token.
        {"w": "Hi,", "s": 5.84, "e": 6.24, "c": 0.973},
        {"w": "welcome", "s": 6.24, "e": 6.56, "c": 1.0},
    ]
    assert payload["result"]["artifacts"]["json"].endswith("/transcript.json")


@pytest.mark.django_db
def test_local_profile_job_without_the_model_fails_and_never_runs_another_engine(
    client, settings, parakeet_cli
):
    (parakeet_cli.model_dir / "model.safetensors").unlink()

    payload = run_local_profile_job(client, settings)

    assert payload["state"] == "failed"
    assert "model.safetensors" in payload["error"]["message"]
    assert parakeet_cli.other_engine_calls == []
    assert not JobArtifact.objects.filter(kind=JobArtifact.Kind.WORDS).exists()


# --- words and identity ------------------------------------------------------------


def token(text: str, start: float, end: float, confidence: float) -> dict[str, Any]:
    return {"text": text, "start": start, "end": end, "confidence": confidence}


def test_words_follow_the_research_rule_across_sentences():
    payload = {
        "sentences": [
            {"tokens": [token(" Wel", 0.0, 0.2, 0.9), token("come", 0.2, 0.4, 0.6)]},
            # A sentence starting with a sub-word piece continues the previous word.
            {"tokens": [token("s.", 0.4, 0.5, 0.3), token(" Hi", 1.0, 1.2345, 0.98765)]},
        ]
    }

    assert parakeet_payload_words(payload) == [
        {"w": "Welcomes.", "s": 0.0, "e": 0.5, "c": 0.6},
        {"w": "Hi", "s": 1.0, "e": 1.234, "c": 0.988},
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"sentences": [{"text": "no tokens"}]},
        {"sentences": [{"tokens": [{"text": " a", "start": 0.0, "end": 1.0}]}]},
        {"sentences": [{"tokens": [token(" a", 0.0, 1.0, 0.5) | {"start": "0"}]}]},
    ],
)
def test_malformed_parakeet_output_fails_the_profile(payload: dict[str, Any]):
    with pytest.raises(RuntimeError):
        parakeet_payload_words(payload)


def test_model_revision_falls_back_to_a_hugging_face_snapshot_directory(tmp_path: Path):
    snapshot = (
        tmp_path
        / "models--mlx-community--parakeet-tdt-0.6b-v3"
        / "snapshots"
        / "ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15"
    )
    snapshot.mkdir(parents=True)
    assert model_revision(snapshot) == {
        "model_repo": "mlx-community/parakeet-tdt-0.6b-v3",
        "model_revision": "ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15",
        "model_sha256": "",
    }
    assert model_revision(tmp_path) == {"model_repo": "", "model_revision": "", "model_sha256": ""}


# --- remote workers ----------------------------------------------------------------


def parakeet_worker_capabilities(*, profiles_advertised: bool) -> dict[str, Any]:
    capabilities = plain_capabilities()
    capabilities["backends"] = ["parakeet", "whispercpp"]
    capabilities["models"] = ["ggml-large-v3.bin", "parakeet-tdt-0.6b-v3"]
    if profiles_advertised:
        capabilities["profiles"] = [AD_ANALYSIS_V1]
    return capabilities


def submit_remote_profile_job(client, settings) -> Job:
    enable_remote_workers(settings)
    settings.VOXHELM_PARAKEET_MODEL_NAME = "parakeet-tdt-0.6b-v3"
    response = post_job(client, profile_payload())
    assert response.status_code == 201
    return Job.objects.get(id=response.json()["id"])


@pytest.mark.django_db
def test_profile_job_waits_for_a_worker_that_advertises_the_profile(client, settings):
    job = submit_remote_profile_job(client, settings)

    # Parakeet and whisper.cpp alone are not enough, and nothing falls back.
    without = claim_one(
        client, capabilities=parakeet_worker_capabilities(profiles_advertised=False)
    )
    assert without.status_code == 204
    assert claim_one(client).status_code == 204
    job.refresh_from_db()
    assert job.state == Job.State.QUEUED

    claim = claim_one(
        client, capabilities=parakeet_worker_capabilities(profiles_advertised=True)
    ).json()["job"]
    assert claim["id"] == str(job.id)
    assert (claim["backend"], claim["model"]) == ("parakeet", "parakeet-tdt-0.6b-v3")
    assert claim["profile"] == AD_ANALYSIS_V1
    assert claim["language"] == "de"
    assert "fallbacks" not in claim


def words_manifest_entry(job_id: str, attempt: int, **overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": "words.json",
        "kind": "words",
        "format": "words",
        "storage_backend": "filesystem",
        "storage_key": f"voxhelm/jobs/{job_id}/attempt-{attempt}/words.json",
        "content_type": "application/json",
        "size_bytes": 12,
        "exposed": True,
    }
    entry.update(overrides)
    return entry


def worker_identity(**overrides: Any) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "profile": AD_ANALYSIS_V1,
        "backend": "parakeet",
        "model": "parakeet-tdt-0.6b-v3",
        "model_repo": "mlx-community/parakeet-tdt-0.6b-v3",
        "model_revision": "ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15",
        "model_sha256": "05e01c7f",
        "cli": "parakeet-mlx",
        "cli_version": "0.5.3",
        "mlx_version": "0.32.4",
        "settings": PROFILES[AD_ANALYSIS_V1].settings_record(),
        "language_hint": "de",
        "unexpected": "dropped",
    }
    identity.update(overrides)
    return identity


def complete_profile_job(client, settings, job: Job, *, artifacts, metadata):
    claim = claim_one(
        client, capabilities=parakeet_worker_capabilities(profiles_advertised=True)
    ).json()["job"]
    manifest = [
        entry
        for entry in transcript_manifest(str(job.id), claim["attempt"])
        if entry["format"] != "text"
    ]
    manifest.extend(artifacts(str(job.id), claim["attempt"]))
    store_manifest_artifacts(settings, manifest)
    return client.post(
        f"/v1/internal/work/{job.id}/complete",
        data=json.dumps(
            {
                "worker_id": "atlas",
                "lease_token": claim["lease_token"],
                "result_text": "remote transcript",
                "result_metadata": {"backend": "parakeet", "profile": metadata},
                "artifacts": manifest,
            }
        ),
        content_type="application/json",
        **worker_headers(),
    )


@pytest.mark.django_db
def test_remote_profile_completion_records_identity_and_serves_words(client, settings):
    job = submit_remote_profile_job(client, settings)

    response = complete_profile_job(
        client,
        settings,
        job,
        artifacts=lambda job_id, attempt: [words_manifest_entry(job_id, attempt)],
        metadata=worker_identity(language_hint="en"),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    identity = result["metadata"]["profile"]
    assert identity["model_revision"] == "ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15"
    assert identity["settings"]["chunk_seconds"] == 30
    assert identity["language_hint"] == "de"  # the job's hint, not the worker's claim
    assert "unexpected" not in identity
    words = client.get(result["artifacts"]["words"], **producer_headers())
    assert words.status_code == 200


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("artifacts", "metadata", "message"),
    [
        (lambda job_id, attempt: [], worker_identity(), "must include words.json"),
        (
            lambda job_id, attempt: [words_manifest_entry(job_id, attempt, exposed=False)],
            worker_identity(),
            "must be exposed",
        ),
        (
            lambda job_id, attempt: [words_manifest_entry(job_id, attempt)],
            worker_identity(profile="ad-analysis-v0"),
            "result_metadata.profile",
        ),
    ],
)
def test_remote_profile_completion_is_validated(client, settings, artifacts, metadata, message):
    job = submit_remote_profile_job(client, settings)

    response = complete_profile_job(client, settings, job, artifacts=artifacts, metadata=metadata)

    assert response.status_code == 400
    assert message in response.json()["error"]["message"]


def test_worker_advertises_the_profile_only_while_parakeet_resolves(parakeet_cli, settings):
    settings.VOXHELM_DIARIZATION_BACKEND = "none"
    settings.VOXHELM_WHISPERKIT_ENABLED = False
    capabilities = worker_cli.build_capabilities()
    assert capabilities["profiles"] == [AD_ANALYSIS_V1]

    (parakeet_cli.model_dir / "model.safetensors").unlink()
    assert worker_cli.with_current_stt_runtimes(capabilities)["profiles"] == []


def test_worker_profile_claim_never_falls_back(parakeet_cli):
    (parakeet_cli.model_dir / "config.json").unlink()
    claim = {
        "id": "job-1",
        "backend": "parakeet",
        "model": "parakeet-tdt-0.6b-v3",
        "profile": AD_ANALYSIS_V1,
        "language": "de",
        "fallbacks": [{"backend": "whispercpp", "model": "ggml-large-v3.bin"}],
    }

    with pytest.raises(BackendUnavailableError):
        worker_cli.transcribe_claim_profile(claim=claim, audio_path=Path("/tmp/a.mp3"))
    assert parakeet_cli.other_engine_calls == []
    # Reported as retryable: the job goes back to the queue for a capable worker.
    assert worker_cli.retryable_worker_error(BackendUnavailableError("gone"))


@pytest.mark.django_db
def test_worker_uploads_words_and_identity_for_a_profile_claim(parakeet_cli, tmp_path: Path):
    staged_key = "voxhelm/staged-inputs/u1/episode.mp3"
    get_artifact_store().put_bytes(key=staged_key, data=b"mp3", content_type="audio/mpeg")
    claim = {
        "id": "job-1",
        "lease_token": "lease",
        "backend": "parakeet",
        "model": "parakeet-tdt-0.6b-v3",
        "profile": AD_ANALYSIS_V1,
        "language": "de",
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
    assert names == ["source.mp3", "transcript.json", "words.json"]
    words = json.loads(
        get_artifact_store().read_bytes(key="voxhelm/jobs/job-1/attempt-1/words.json")
    )
    assert words["identity"]["profile"] == AD_ANALYSIS_V1
    assert completed["result_metadata"]["profile"] == words["identity"]
    assert parakeet_cli.other_engine_calls == []


# --- producer DELETE ---------------------------------------------------------------


def make_profile_job(*, finished_ago: timedelta, profile: str = AD_ANALYSIS_V1, **fields) -> Job:
    return Job.objects.create(
        producer=fields.pop("producer", "archive"),
        job_type=Job.JobType.TRANSCRIBE,
        profile=profile,
        input_data={"kind": "url", "url": "https://media.example.com/episode.mp3"},
        state=fields.pop("state", Job.State.SUCCEEDED),
        finished_at=timezone.now() - finished_ago,
        result_text="the transcript",
        **fields,
    )


def add_artifact(job: Job, name: str, kind: str, *, exposed: bool, key: str = "") -> JobArtifact:
    key = key or f"voxhelm/jobs/{job.id}/{name}"
    stored = get_artifact_store().put_bytes(key=key, data=b"payload", content_type="text/plain")
    return JobArtifact.objects.create(
        job=job,
        name=name,
        kind=kind,
        format=kind,
        storage_backend="filesystem",
        storage_key=key,
        storage_identity=current_artifact_store_identity(),
        content_type="text/plain",
        size_bytes=stored.size_bytes,
        exposed=exposed,
    )


def add_profile_artifacts(job: Job) -> list[JobArtifact]:
    return [
        add_artifact(job, "episode.mp3", JobArtifact.Kind.SOURCE, exposed=False),
        add_artifact(job, "transcript.json", JobArtifact.Kind.TRANSCRIPT_JSON, exposed=True),
        add_artifact(job, "words.json", JobArtifact.Kind.WORDS, exposed=True),
    ]


def object_exists(settings, artifact: JobArtifact) -> bool:
    return (Path(settings.VOXHELM_ARTIFACT_ROOT) / artifact.storage_key).exists()


def delete_job(client, job: Job, token: str = "test-token"):
    return client.delete(f"/v1/jobs/{job.id}", HTTP_AUTHORIZATION=f"Bearer {token}")


@pytest.mark.django_db
def test_producer_delete_removes_artifacts_objects_and_row(client, settings):
    job = make_profile_job(finished_ago=timedelta(minutes=1))
    artifacts = add_profile_artifacts(job)
    PendingArtifactDeletion.objects.create(
        job_id=job.id,
        kind=JobArtifact.Kind.SOURCE,
        storage_backend="filesystem",
        storage_key=f"voxhelm/jobs/{job.id}/attempt-1/old.mp3",
        storage_identity=current_artifact_store_identity(),
    )
    staged = add_claimed_upload(job)

    response = delete_job(client, job)

    assert response.status_code == 204
    assert not Job.objects.filter(id=job.id).exists()
    assert not JobArtifact.objects.filter(job_id=job.id).exists()
    assert not PendingArtifactDeletion.objects.exists()
    assert not StagedMedia.objects.exists()
    assert not any(object_exists(settings, artifact) for artifact in artifacts)
    assert not (Path(settings.VOXHELM_ARTIFACT_ROOT) / staged.storage_key).exists()
    assert delete_job(client, job).status_code == 404


@pytest.mark.django_db
def test_only_the_producer_may_delete(client, settings):
    settings.VOXHELM_BEARER_TOKENS = {"archive": "test-token", "cast": "cast-token"}
    job = make_profile_job(finished_ago=timedelta(minutes=1))
    artifacts = add_profile_artifacts(job)

    response = delete_job(client, job, token="cast-token")

    assert response.status_code == 404
    assert Job.objects.filter(id=job.id).exists()
    assert all(object_exists(settings, artifact) for artifact in artifacts)
    assert client.delete(f"/v1/jobs/{job.id}").status_code == 401


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"profile": ""}, "Only profile jobs"),
        ({"state": Job.State.RUNNING}, "has not finished"),
        ({"state": Job.State.QUEUED}, "has not finished"),
    ],
)
def test_delete_refuses_other_jobs_and_unfinished_ones(client, settings, fields, message):
    job = make_profile_job(finished_ago=timedelta(minutes=1), **fields)
    artifacts = add_profile_artifacts(job)

    response = delete_job(client, job)

    assert response.status_code == 409
    assert message in response.json()["error"]["message"]
    assert Job.objects.filter(id=job.id).exists()
    assert all(object_exists(settings, artifact) for artifact in artifacts)


@pytest.mark.django_db
def test_delete_refuses_an_artifact_outside_the_allowlist_and_deletes_nothing(client, settings):
    job = make_profile_job(finished_ago=timedelta(minutes=1))
    artifacts = add_profile_artifacts(job)
    foreign = add_artifact(
        job, "words-copy.json", JobArtifact.Kind.WORDS, exposed=True, key="elsewhere/words.json"
    )

    response = delete_job(client, job)

    assert response.status_code == 409
    assert JobArtifact.objects.filter(job=job).count() == 4
    assert all(object_exists(settings, artifact) for artifact in [*artifacts, foreign])


@pytest.mark.django_db
def test_failed_object_deletion_keeps_the_rest_for_a_retry(client, settings, monkeypatch):
    job = make_profile_job(finished_ago=timedelta(minutes=1))
    add_profile_artifacts(job)
    original = retention.delete_stored_object
    failures = {"left": 1}

    def flaky(*, identity, key):
        if key.endswith("transcript.json") and failures["left"]:
            failures["left"] -= 1
            raise OSError("store unavailable")
        original(identity=identity, key=key)

    monkeypatch.setattr(retention, "delete_stored_object", flaky)

    first = delete_job(client, job)
    assert first.status_code == 503
    assert Job.objects.filter(id=job.id).exists()
    assert set(JobArtifact.objects.filter(job=job).values_list("name", flat=True)) == {
        "transcript.json",
        "words.json",
    }

    assert delete_job(client, job).status_code == 204
    assert not Job.objects.filter(id=job.id).exists()


# --- 7-day backstop ----------------------------------------------------------------


@pytest.mark.django_db
def test_backstop_removes_every_artifact_and_the_row_of_an_undeleted_profile_job(settings):
    settings.VOXHELM_SOURCE_ARTIFACT_RETENTION_SECONDS = 86400
    old = make_profile_job(finished_ago=timedelta(days=7, minutes=1))
    old_artifacts = add_profile_artifacts(old)
    recent = make_profile_job(finished_ago=timedelta(days=6, hours=23))
    recent_artifacts = add_profile_artifacts(recent)
    plain = make_profile_job(finished_ago=timedelta(days=30), profile="")
    plain_source, *plain_artifacts = add_profile_artifacts(plain)

    dry_run = prune_job_artifacts(dry_run=True)
    assert {artifact.pk for artifact in dry_run.deleted} == {
        *(artifact.pk for artifact in old_artifacts),
        recent_artifacts[0].pk,  # sources past the 24 h D-09 retention
        plain_source.pk,
    }
    assert [job.id for job in dry_run.jobs_deleted] == [old.id]

    result = prune_job_artifacts()

    assert not Job.objects.filter(id=old.id).exists()
    assert not any(object_exists(settings, artifact) for artifact in old_artifacts)
    assert [job.id for job in result.jobs_deleted] == [old.id]
    # Inside the backstop only D-09 applies: the source goes, transcript and words stay.
    assert set(JobArtifact.objects.filter(job=recent).values_list("name", flat=True)) == {
        "transcript.json",
        "words.json",
    }
    # Other jobs are unchanged: final artifacts and the row are kept.
    assert all(object_exists(settings, artifact) for artifact in plain_artifacts)
    assert JobArtifact.objects.filter(job=plain).count() == 2
    assert Job.objects.filter(id=plain.id).exists()


@pytest.mark.django_db
def test_backstop_applies_with_job_metadata_retention_disabled(settings):
    settings.VOXHELM_JOB_METADATA_RETENTION_SECONDS = 0
    failed = make_profile_job(finished_ago=timedelta(days=8), state=Job.State.FAILED)
    add_profile_artifacts(failed)

    prune_job_artifacts()

    assert not Job.objects.filter(id=failed.id).exists()
    assert not JobArtifact.objects.exists()


# --- review remediation: nothing of a profile job outlives DELETE or the backstop -----


def job_objects(settings, job: Job) -> list[Path]:
    root = Path(settings.VOXHELM_ARTIFACT_ROOT) / "voxhelm" / "jobs" / str(job.id)
    return sorted(path for path in root.rglob("*") if path.is_file()) if root.exists() else []


def profile_claim(client) -> dict[str, Any]:
    response = claim_one(
        client, capabilities=parakeet_worker_capabilities(profiles_advertised=True)
    )
    assert response.status_code == 200
    return response.json()["job"]


def upload_attempt_objects(settings, job: Job, claim: dict[str, Any]) -> list[dict[str, Any]]:
    """What a worker uploads before it calls complete: source, transcript and words."""
    manifest = [
        entry
        for entry in transcript_manifest(str(job.id), claim["attempt"])
        if entry["format"] != "text"
    ]
    manifest.append(words_manifest_entry(str(job.id), claim["attempt"]))
    store_manifest_artifacts(settings, manifest)
    return manifest


def post_completion(client, job: Job, claim: dict[str, Any], manifest, metadata):
    return client.post(
        f"/v1/internal/work/{job.id}/complete",
        data=json.dumps(
            {
                "worker_id": "atlas",
                "lease_token": claim["lease_token"],
                "result_text": "remote transcript",
                "result_metadata": {"backend": "parakeet", "profile": metadata},
                "artifacts": manifest,
            }
        ),
        content_type="application/json",
        **worker_headers(),
    )


def post_failure(client, job: Job, claim: dict[str, Any], *, retryable: bool):
    return client.post(
        f"/v1/internal/work/{job.id}/fail",
        data=json.dumps(
            {
                "worker_id": "atlas",
                "lease_token": claim["lease_token"],
                "error_detail": "completion failed",
                "retryable": retryable,
            }
        ),
        content_type="application/json",
        **worker_headers(),
    )


@pytest.mark.django_db
def test_delete_removes_objects_of_a_failed_final_attempt_without_a_manifest(client, settings):
    settings.VOXHELM_REMOTE_WORKER_MAX_ATTEMPTS = 1
    job = submit_remote_profile_job(client, settings)
    claim = profile_claim(client)
    manifest = upload_attempt_objects(settings, job, claim)
    # The completion is rejected after the upload, the worker reports a terminal failure.
    rejected = post_completion(
        client, job, claim, manifest, worker_identity(profile="ad-analysis-v0")
    )
    assert rejected.status_code == 400
    assert post_failure(client, job, claim, retryable=False).status_code == 200
    job.refresh_from_db()
    assert job.state == Job.State.FAILED
    assert len(job_objects(settings, job)) == 3

    assert delete_job(client, job).status_code == 204

    assert job_objects(settings, job) == []
    assert not Job.objects.filter(id=job.id).exists()


@pytest.mark.django_db
def test_backstop_removes_objects_of_a_retried_attempt_without_a_manifest(client, settings):
    job = submit_remote_profile_job(client, settings)
    first = profile_claim(client)
    upload_attempt_objects(settings, job, first)
    assert post_failure(client, job, first, retryable=True).status_code == 200
    second = profile_claim(client)
    manifest = upload_attempt_objects(settings, job, second)
    completed = post_completion(client, job, second, manifest, worker_identity())
    assert completed.status_code == 200, completed.json()
    Job.objects.filter(id=job.id).update(finished_at=timezone.now() - timedelta(days=8))
    assert len(job_objects(settings, job)) == 6

    dry_run = prune_job_artifacts(dry_run=True)
    assert [listed.id for listed in dry_run.jobs_deleted] == [job.id]
    prune_job_artifacts()

    assert job_objects(settings, job) == []
    assert not Job.objects.filter(id=job.id).exists()


@pytest.mark.django_db(transaction=True)
def test_a_local_profile_task_result_never_holds_the_transcript(client, settings, parakeet_cli):
    from django.core.management import call_command
    from django_tasks_db.models import DBTaskResult

    configure_task_backend(settings, "django_tasks_db.backend.DatabaseBackend")
    upload_id = stage_upload(client, content=b"episode-bytes", content_type="audio/mpeg").json()[
        "id"
    ]
    response = post_job(client, profile_payload(input={"kind": "upload", "upload_id": upload_id}))
    assert response.status_code == 201
    call_command("db_worker", "--batch", "--no-startup-delay")
    job = Job.objects.get(id=response.json()["id"])
    assert job.state == Job.State.SUCCEEDED
    task_result = DBTaskResult.objects.get(id=job.django_task_id)
    assert task_result.status == "SUCCESSFUL"
    assert job.result_text
    assert job.result_text not in json.dumps(task_result.return_value)
    assert task_result.return_value == {"job_id": str(job.id), "profile": AD_ANALYSIS_V1}

    assert delete_job(client, job).status_code == 204
    assert not DBTaskResult.objects.filter(id=job.django_task_id).exists()


def add_task_result(job: Job) -> Any:
    """A task result written before the fix: the return value carried the transcript."""
    from django_tasks_db.models import DBTaskResult

    task_result = DBTaskResult.objects.create(
        status="SUCCESSFUL",
        args_kwargs={"args": [str(job.id)], "kwargs": {}},
        task_path="jobs.tasks.run_transcription_job",
        backend_name="default",
        queue_name="default",
        run_after=timezone.now(),
        return_value={"job_id": str(job.id), "text": "the transcript", "metadata": {}},
    )
    Job.objects.filter(id=job.id).update(django_task_id=str(task_result.id))
    job.refresh_from_db()
    return task_result


@pytest.mark.django_db
def test_delete_removes_an_existing_task_result_with_the_transcript(client, settings):
    from django_tasks_db.models import DBTaskResult

    job = make_profile_job(finished_ago=timedelta(minutes=1))
    add_profile_artifacts(job)
    task_result = add_task_result(job)
    other = add_task_result(make_profile_job(finished_ago=timedelta(minutes=1)))

    assert delete_job(client, job).status_code == 204

    assert not DBTaskResult.objects.filter(id=task_result.id).exists()
    assert DBTaskResult.objects.filter(id=other.id).exists()


@pytest.mark.django_db
def test_backstop_removes_an_existing_task_result_with_the_transcript(settings):
    from django_tasks_db.models import DBTaskResult

    job = make_profile_job(finished_ago=timedelta(days=8))
    add_profile_artifacts(job)
    task_result = add_task_result(job)
    recent = add_task_result(make_profile_job(finished_ago=timedelta(days=1)))

    prune_job_artifacts()

    assert not Job.objects.filter(id=job.id).exists()
    assert not DBTaskResult.objects.filter(id=task_result.id).exists()
    assert DBTaskResult.objects.filter(id=recent.id).exists()


def add_claimed_upload(job: Job, *, key: str = "") -> StagedMedia:
    staged = StagedMedia(
        producer="archive",
        original_filename="episode.mp3",
        content_type="audio/mpeg",
        storage_backend="filesystem",
        storage_identity=current_artifact_store_identity(),
        claimed_by_job=job,
        claimed_at=job.created_at,
        expires_at=timezone.now() + timedelta(days=30),
    )
    staged.storage_key = key or f"voxhelm/staged-inputs/{staged.id}/episode.mp3"
    staged.save()
    get_artifact_store().put_bytes(key=staged.storage_key, data=b"mp3", content_type="audio/mpeg")
    return staged


def staged_object_exists(settings, staged: StagedMedia) -> bool:
    return (Path(settings.VOXHELM_ARTIFACT_ROOT) / staged.storage_key).exists()


@pytest.mark.django_db
def test_backstop_retries_a_staged_upload_the_completion_could_not_delete(settings):
    job = make_profile_job(finished_ago=timedelta(days=7, minutes=1))
    add_profile_artifacts(job)
    staged = add_claimed_upload(job)

    dry_run = prune_job_artifacts(dry_run=True)
    assert [listed.id for listed in dry_run.jobs_deleted] == [job.id]
    prune_job_artifacts()

    assert not StagedMedia.objects.filter(id=staged.id).exists()
    assert not staged_object_exists(settings, staged)
    assert not Job.objects.filter(id=job.id).exists()


@pytest.mark.django_db
def test_backstop_keeps_a_staged_upload_of_a_recent_profile_job(settings):
    job = make_profile_job(finished_ago=timedelta(days=6, hours=23))
    staged = add_claimed_upload(job)

    prune_job_artifacts()

    assert StagedMedia.objects.filter(id=staged.id).exists()
    assert staged_object_exists(settings, staged)


@pytest.mark.django_db
def test_delete_keeps_a_staged_object_another_job_still_references(client, settings):
    job = make_profile_job(finished_ago=timedelta(minutes=1))
    add_profile_artifacts(job)
    staged = add_claimed_upload(job)
    other = make_profile_job(finished_ago=timedelta(minutes=1), profile="")
    shared = JobArtifact.objects.create(
        job=other,
        name="episode.mp3",
        kind=JobArtifact.Kind.SOURCE,
        format="source",
        storage_backend="filesystem",
        storage_key=staged.storage_key,
        storage_identity=current_artifact_store_identity(),
        content_type="audio/mpeg",
        size_bytes=3,
        exposed=False,
    )

    assert delete_job(client, job).status_code == 204

    assert not StagedMedia.objects.filter(id=staged.id).exists()
    assert JobArtifact.objects.filter(pk=shared.pk).exists()
    assert staged_object_exists(settings, staged)


@pytest.mark.django_db
def test_delete_refuses_a_staged_upload_outside_its_key_layout(client, settings):
    job = make_profile_job(finished_ago=timedelta(minutes=1))
    artifacts = add_profile_artifacts(job)
    staged = add_claimed_upload(job, key="voxhelm/elsewhere/episode.mp3")

    response = delete_job(client, job)

    assert response.status_code == 409
    assert StagedMedia.objects.filter(id=staged.id).exists()
    assert staged_object_exists(settings, staged)
    assert all(object_exists(settings, artifact) for artifact in artifacts)
    assert Job.objects.filter(id=job.id).exists()


def test_stores_list_every_object_below_an_attempt_prefix(tmp_path: Path):
    from jobs.artifacts import FilesystemArtifactStore, S3ArtifactStore

    store = FilesystemArtifactStore(root=tmp_path)
    store.put_bytes(key="voxhelm/jobs/j/attempt-1/words.json", data=b"{}", content_type="x")
    store.put_bytes(key="voxhelm/jobs/j/attempt-10/words.json", data=b"{}", content_type="x")
    (tmp_path / "voxhelm/jobs/j/attempt-1/alias").symlink_to(tmp_path)
    assert store.list_keys(prefix="voxhelm/jobs/j/attempt-1/") == [
        "voxhelm/jobs/j/attempt-1/alias",
        "voxhelm/jobs/j/attempt-1/words.json",
    ]
    assert store.list_keys(prefix="voxhelm/jobs/other/attempt-1/") == []

    pages = [{"Contents": [{"Key": "p/b.json"}]}, {"Contents": [{"Key": "p/a.json"}]}, {}]
    requested: dict[str, Any] = {}

    class Paginator:
        def paginate(self, **kwargs: Any):
            requested.update(kwargs)
            return iter(pages)

    s3 = S3ArtifactStore.__new__(S3ArtifactStore)
    s3.bucket = "bucket"
    s3.client = type("Client", (), {"get_paginator": lambda self, name: Paginator()})()
    assert s3.list_keys(prefix="p/") == ["p/a.json", "p/b.json"]
    assert requested == {"Bucket": "bucket", "Prefix": "p/"}


@pytest.mark.django_db
def test_delete_refuses_an_attempt_prefix_holding_a_foreign_object(client, settings):
    job = make_profile_job(finished_ago=timedelta(minutes=1))
    artifacts = add_profile_artifacts(job)
    prefix = f"voxhelm/jobs/{job.id}/attempt-1/"
    RemoteAttemptPrefix.objects.create(
        job_id=job.id, attempt=1, prefix=prefix, storage_identity=current_artifact_store_identity()
    )
    get_artifact_store().put_bytes(key=f"{prefix}words.json", data=b"{}", content_type="x")
    outside = Path(settings.VOXHELM_ARTIFACT_ROOT) / "outside.txt"
    outside.write_text("keep")
    (Path(settings.VOXHELM_ARTIFACT_ROOT) / prefix / "alias.txt").symlink_to(outside)

    assert delete_job(client, job).status_code == 409

    assert outside.exists()
    # Nothing deleted: three artifacts, the attempt's words and the alias.
    assert len(job_objects(settings, job)) == 5
    assert all(object_exists(settings, artifact) for artifact in artifacts)
    assert RemoteAttemptPrefix.objects.filter(job_id=job.id).exists()


@pytest.mark.django_db
def test_delete_keeps_the_record_of_an_attempt_directory_it_cannot_read(client, settings):
    job = make_profile_job(finished_ago=timedelta(minutes=1))
    prefix = f"voxhelm/jobs/{job.id}/attempt-1/"
    RemoteAttemptPrefix.objects.create(
        job_id=job.id, attempt=1, prefix=prefix, storage_identity=current_artifact_store_identity()
    )
    get_artifact_store().put_bytes(key=f"{prefix}words.json", data=b"{}", content_type="x")
    hidden = Path(settings.VOXHELM_ARTIFACT_ROOT) / prefix / "sub"
    hidden.mkdir()
    (hidden / "transcript.json").write_text("{}")
    hidden.chmod(0)
    try:
        # An unreadable directory must not look empty: nothing is deleted, the record stays.
        assert delete_job(client, job).status_code == 409
        assert RemoteAttemptPrefix.objects.filter(job_id=job.id).exists()
        assert (Path(settings.VOXHELM_ARTIFACT_ROOT) / prefix / "words.json").exists()
    finally:
        hidden.chmod(0o755)
    assert (hidden / "transcript.json").exists()


@pytest.mark.django_db
def test_delete_keeps_the_record_when_the_job_directory_cannot_be_searched(client, settings):
    job = make_profile_job(finished_ago=timedelta(minutes=1))
    prefix = f"voxhelm/jobs/{job.id}/attempt-1/"
    RemoteAttemptPrefix.objects.create(
        job_id=job.id, attempt=1, prefix=prefix, storage_identity=current_artifact_store_identity()
    )
    get_artifact_store().put_bytes(key=f"{prefix}words.json", data=b"{}", content_type="x")
    parent = Path(settings.VOXHELM_ARTIFACT_ROOT) / f"voxhelm/jobs/{job.id}"
    parent.chmod(0)
    try:
        # An unsearchable parent must not make the attempt directory look missing.
        assert delete_job(client, job).status_code == 409
        assert RemoteAttemptPrefix.objects.filter(job_id=job.id).exists()
        assert Job.objects.filter(id=job.id).exists()
    finally:
        parent.chmod(0o755)
    assert (parent / "attempt-1" / "words.json").exists()


@pytest.mark.django_db
def test_dry_run_refuses_an_attempt_prefix_the_real_run_refuses(settings):
    job = make_profile_job(finished_ago=timedelta(days=7, minutes=1))
    prefix = f"voxhelm/jobs/{job.id}/attempt-1/"
    RemoteAttemptPrefix.objects.create(
        job_id=job.id, attempt=1, prefix=prefix, storage_identity=current_artifact_store_identity()
    )
    get_artifact_store().put_bytes(key=f"{prefix}nested/words.json", data=b"{}", content_type="x")

    planned = prune_job_artifacts(dry_run=True)
    assert planned.attempt_prefixes_refused == 1
    assert planned.attempt_prefixes_deleted == 0
    assert job.id not in {planned_job.id for planned_job in planned.jobs_deleted}

    done = prune_job_artifacts()
    assert done.attempt_prefixes_refused == 1
    assert RemoteAttemptPrefix.objects.filter(job_id=job.id).exists()
