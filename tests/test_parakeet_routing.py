"""Routing of known-speaker jobs to Parakeet, locally and on remote workers."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from jobs import remote_worker_cli as worker_cli
from jobs.models import Job
from lane_scheduler import LANE_NON_INTERACTIVE, admit_local_inference
from tests.test_known_speaker import StubEmbeddingBackend, make_reference, two_speaker_audio
from tests.test_remote_workers import claim_one, known_speaker_capabilities, submit_remote_job
from transcriptions.known_speaker import KnownSpeakerConfig, run_known_speaker_postprocess
from transcriptions.service import (
    BackendUnavailableError,
    TranscribeParams,
    TranscriptionResult,
    TranscriptionSegment,
    get_backend_services_for_params,
    normalize_parakeet_payload,
    transcribe_audio,
)

KNOWN_SPEAKER_DIARIZATION: dict[str, Any] = {
    "enabled": True,
    "strategy": "pyannote_known_speaker",
    "known_speakers": [
        {
            "id": "12",
            "name": "Johannes",
            "references": [
                {
                    "kind": "source_range",
                    "audio": {"kind": "url", "url": "https://cdn.example.com/pp_60.m4a"},
                    "start": 0.0,
                    "end": 2.0,
                }
            ],
        }
    ],
}


@pytest.fixture
def stt_settings(settings):
    settings.VOXHELM_STT_BACKEND = "whispercpp"
    settings.VOXHELM_STT_FALLBACK_BACKEND = "mlx"
    settings.VOXHELM_WHISPERCPP_MODEL = "ggml-large-v3.bin"
    settings.VOXHELM_MLX_MODEL = "mlx-community/whisper-large-v3-mlx"
    settings.VOXHELM_WHISPERKIT_MODEL = "large-v3-v20240930"
    settings.VOXHELM_PARAKEET_MODEL_NAME = "parakeet-tdt-0.6b-v3"
    settings.VOXHELM_KNOWN_SPEAKER_STT_BACKEND = "parakeet"
    return settings


def params(*, request_model: str = "auto", word_timestamps: bool = True) -> TranscribeParams:
    return TranscribeParams(
        request_model=request_model, prompt=None, language="en", word_timestamps=word_timestamps
    )


def chain(**kwargs: Any) -> list[str]:
    return [invocation.name for invocation in get_backend_services_for_params(params(**kwargs))]


def result_from(backend_name: str) -> TranscriptionResult:
    return TranscriptionResult(
        text="hello",
        language="en",
        segments=[TranscriptionSegment(id=0, start=0.0, end=1.0, text="hello")],
        backend_name=backend_name,
    )


class ScriptedBackend:
    """Raises the scripted error or returns a result named after the backend."""

    def __init__(self, name: str, calls: list[tuple[str, TranscribeParams]], error=None) -> None:
        self.name = name
        self.calls = calls
        self.error = error

    def transcribe(self, audio_path: Path, params: TranscribeParams) -> TranscriptionResult:
        del audio_path
        self.calls.append((self.name, params))
        if self.error is not None:
            raise self.error
        return result_from(self.name)


# --- local chain ---------------------------------------------------------------


def test_known_speaker_chain_puts_parakeet_then_whispercpp_first(stt_settings) -> None:
    assert chain() == ["parakeet", "whispercpp", "mlx"]
    assert chain(request_model="whisper-1") == ["parakeet", "whispercpp", "mlx"]
    assert chain(request_model="gpt-4o-mini-transcribe") == ["parakeet", "whispercpp", "mlx"]


def test_known_speaker_chain_is_off_without_the_setting_or_word_timings(stt_settings) -> None:
    assert chain(word_timestamps=False) == ["whispercpp", "mlx"]
    stt_settings.VOXHELM_KNOWN_SPEAKER_STT_BACKEND = ""
    assert chain() == ["whispercpp", "mlx"]


def test_known_speaker_chain_never_reroutes_an_explicit_model(stt_settings) -> None:
    assert chain(request_model="mlx-community/whisper-large-v3-mlx") == ["mlx"]
    assert chain(request_model="ggml-large-v3.bin") == ["whispercpp"]


@pytest.mark.parametrize(
    ("primary", "fallback", "expected"),
    [
        ("mlx", "whispercpp", ["parakeet", "whispercpp", "mlx"]),
        ("mlx", "", ["parakeet", "whispercpp", "mlx"]),
        ("whisperkit", "mlx", ["parakeet", "whispercpp", "whisperkit", "mlx"]),
        ("whispercpp", "", ["parakeet", "whispercpp"]),
    ],
)
def test_known_speaker_chain_falls_back_to_whispercpp_before_the_configured_chain(
    stt_settings, primary: str, fallback: str, expected: list[str]
) -> None:
    stt_settings.VOXHELM_STT_BACKEND = primary
    stt_settings.VOXHELM_STT_FALLBACK_BACKEND = fallback
    assert chain() == expected


def install_backends(monkeypatch, errors: dict[str, Exception]) -> list[tuple[str, Any]]:
    calls: list[tuple[str, Any]] = []
    monkeypatch.setattr(
        "transcriptions.service.service_for_backend_name",
        lambda name, *, request_model: ScriptedBackend(name, calls, errors.get(name)),
    )
    return calls


def test_transcribe_audio_falls_back_to_whispercpp_with_words(stt_settings, monkeypatch) -> None:
    calls = install_backends(monkeypatch, {"parakeet": BackendUnavailableError("no model")})

    result = transcribe_audio(Path("/tmp/episode.wav"), params())

    assert result.backend_name == "whispercpp"
    assert [(name, p.word_timestamps) for name, p in calls] == [
        ("parakeet", True),
        ("whispercpp", True),
    ]


def test_transcribe_audio_runs_parakeet_for_known_speaker_jobs(stt_settings, monkeypatch) -> None:
    calls = install_backends(monkeypatch, {})
    assert transcribe_audio(Path("/tmp/episode.wav"), params()).backend_name == "parakeet"
    assert [name for name, _ in calls] == ["parakeet"]


def test_transcribe_audio_fails_on_a_parakeet_error_other_than_unavailable(
    stt_settings, monkeypatch
) -> None:
    calls = install_backends(monkeypatch, {"parakeet": RuntimeError("Parakeet crashed")})
    with pytest.raises(RuntimeError, match="Parakeet crashed"):
        transcribe_audio(Path("/tmp/episode.wav"), params())
    assert [name for name, _ in calls] == ["parakeet"]


def test_transcribe_audio_continues_down_the_configured_chain(stt_settings, monkeypatch) -> None:
    calls = install_backends(
        monkeypatch,
        {
            "parakeet": BackendUnavailableError("no model"),
            "whispercpp": BackendUnavailableError("no binary"),
        },
    )
    assert transcribe_audio(Path("/tmp/episode.wav"), params()).backend_name == "mlx"
    assert [name for name, _ in calls] == ["parakeet", "whispercpp", "mlx"]


# --- remote target per worker --------------------------------------------------


def parakeet_capabilities() -> dict[str, Any]:
    capabilities = known_speaker_capabilities()
    capabilities["backends"] = ["parakeet", "whispercpp"]
    capabilities["models"] = ["ggml-large-v3.bin", "parakeet-tdt-0.6b-v3"]
    return capabilities


def submit_known_speaker_job(client, settings, **kwargs: Any) -> Job:
    return submit_remote_job(
        client,
        settings,
        task_ref=kwargs.pop("task_ref", "remote-known-speaker"),
        diarization=kwargs.pop("diarization", KNOWN_SPEAKER_DIARIZATION),
        **kwargs,
    )


@pytest.mark.django_db
def test_claim_routes_known_speaker_jobs_to_parakeet_with_fallbacks(client, stt_settings):
    submit_known_speaker_job(client, stt_settings)

    payload = claim_one(client, capabilities=parakeet_capabilities()).json()["job"]

    assert (payload["backend"], payload["model"]) == ("parakeet", "parakeet-tdt-0.6b-v3")
    # mlx is in the chain but this worker does not advertise it.
    assert payload["fallbacks"] == [{"backend": "whispercpp", "model": "ggml-large-v3.bin"}]


@pytest.mark.django_db
def test_claim_lists_every_advertised_fallback_in_order(client, stt_settings):
    submit_known_speaker_job(client, stt_settings)
    capabilities = parakeet_capabilities()
    capabilities["backends"] = ["mlx", "parakeet", "whispercpp"]
    capabilities["models"] = [
        "ggml-large-v3.bin",
        "mlx-community/whisper-large-v3-mlx",
        "parakeet-tdt-0.6b-v3",
    ]

    payload = claim_one(client, capabilities=capabilities).json()["job"]

    assert payload["backend"] == "parakeet"
    assert payload["fallbacks"] == [
        {"backend": "whispercpp", "model": "ggml-large-v3.bin"},
        {"backend": "mlx", "model": "mlx-community/whisper-large-v3-mlx"},
    ]


@pytest.mark.django_db
def test_worker_without_parakeet_takes_the_job_with_whispercpp(client, stt_settings):
    # E.g. a worker whose model directory disappeared before the claim: the job
    # does not wait for a Parakeet worker.
    job = submit_known_speaker_job(client, stt_settings)

    response = claim_one(client, capabilities=known_speaker_capabilities())
    payload = response.json()["job"]

    assert payload["id"] == str(job.id)
    assert (payload["backend"], payload["model"]) == ("whispercpp", "ggml-large-v3.bin")
    assert "fallbacks" not in payload


@pytest.mark.django_db
def test_worker_with_only_a_later_chain_entry_is_eligible(client, stt_settings):
    submit_known_speaker_job(client, stt_settings)
    capabilities = known_speaker_capabilities()
    capabilities["backends"] = ["mlx"]
    capabilities["models"] = ["mlx-community/whisper-large-v3-mlx"]

    payload = claim_one(client, capabilities=capabilities).json()["job"]

    assert payload["backend"] == "mlx"
    assert "fallbacks" not in payload


@pytest.mark.django_db
def test_claim_ignores_parakeet_with_the_setting_off(client, stt_settings):
    stt_settings.VOXHELM_KNOWN_SPEAKER_STT_BACKEND = ""
    submit_known_speaker_job(client, stt_settings)

    payload = claim_one(client, capabilities=parakeet_capabilities()).json()["job"]

    assert payload["backend"] == "whispercpp"
    assert "fallbacks" not in payload


@pytest.mark.django_db
def test_claim_does_not_route_jobs_without_the_recut(client, stt_settings):
    anonymous = {"enabled": True, "strategy": "pyannote"}
    submit_remote_job(client, stt_settings, task_ref="plain")
    submit_known_speaker_job(client, stt_settings, task_ref="anon", diarization=anonymous)
    no_recut = {**KNOWN_SPEAKER_DIARIZATION, "known_speaker": {"resegment_hop": 0}}
    submit_known_speaker_job(client, stt_settings, task_ref="no-recut", diarization=no_recut)
    only_parakeet = parakeet_capabilities()
    only_parakeet["backends"] = ["parakeet"]
    only_parakeet["models"] = ["parakeet-tdt-0.6b-v3"]

    assert claim_one(client, capabilities=only_parakeet).status_code == 204

    for _ in range(3):
        payload = claim_one(client, worker_id="atlas", capabilities=parakeet_capabilities())
        job = payload.json()["job"]
        assert job["backend"] == "whispercpp"
        assert "fallbacks" not in job
        Job.objects.filter(id=job["id"]).update(state=Job.State.SUCCEEDED)


@pytest.mark.django_db
def test_claim_never_reroutes_an_explicit_model(client, stt_settings):
    submit_known_speaker_job(client, stt_settings, model="ggml-large-v3.bin")

    payload = claim_one(client, capabilities=parakeet_capabilities()).json()["job"]

    assert payload["backend"] == "whispercpp"
    assert "fallbacks" not in payload


# --- worker: claim-time fallback -----------------------------------------------


def known_speaker_claim(**extra: Any) -> dict[str, Any]:
    return {
        "id": "job-1",
        "backend": "parakeet",
        "model": "parakeet-tdt-0.6b-v3",
        "language": "en",
        "output": {"formats": ["text"], "diarization": dict(KNOWN_SPEAKER_DIARIZATION)},
        **extra,
    }


WHISPERCPP_AND_MLX = [
    {"backend": "whispercpp", "model": "ggml-large-v3.bin"},
    {"backend": "mlx", "model": "mlx-community/whisper-large-v3-mlx"},
]


def install_worker_backends(monkeypatch, errors: dict[str, Exception]) -> list[tuple[str, Any]]:
    calls: list[tuple[str, Any]] = []

    def build(*, backend_name: str, model_name: str):
        del model_name
        return ScriptedBackend(backend_name, calls, errors.get(backend_name))

    monkeypatch.setattr(worker_cli, "build_backend_service", build)
    return calls


def test_worker_falls_back_to_whispercpp_when_parakeet_is_unavailable(monkeypatch) -> None:
    calls = install_worker_backends(monkeypatch, {"parakeet": BackendUnavailableError("gone")})

    result = worker_cli.transcribe_claim_audio(
        claim=known_speaker_claim(fallbacks=WHISPERCPP_AND_MLX), audio_path=Path("/tmp/a.wav")
    )

    assert result.backend_name == "whispercpp"
    assert [(name, p.word_timestamps, p.request_model) for name, p in calls] == [
        ("parakeet", True, "parakeet-tdt-0.6b-v3"),
        ("whispercpp", True, "ggml-large-v3.bin"),
    ]


def test_worker_records_the_backend_that_actually_ran(monkeypatch) -> None:
    install_worker_backends(monkeypatch, {"parakeet": BackendUnavailableError("gone")})
    claim = known_speaker_claim(fallbacks=WHISPERCPP_AND_MLX)
    result = worker_cli.transcribe_claim_audio(claim=claim, audio_path=Path("/tmp/a.wav"))
    media = type("Media", (), {"source_name": "a.wav", "content_type": "audio/wav"})()

    metadata = worker_cli.build_result_metadata(
        claim=claim, media=media, result=result, processing_seconds=1.0, speakers_artifact=None
    )

    assert metadata["backend"] == "whispercpp"


def test_worker_continues_to_the_next_advertised_entry(monkeypatch) -> None:
    calls = install_worker_backends(
        monkeypatch,
        {
            "parakeet": BackendUnavailableError("gone"),
            "whispercpp": BackendUnavailableError("no binary"),
        },
    )
    result = worker_cli.transcribe_claim_audio(
        claim=known_speaker_claim(fallbacks=WHISPERCPP_AND_MLX), audio_path=Path("/tmp/a.wav")
    )
    assert result.backend_name == "mlx"
    assert [name for name, _ in calls] == ["parakeet", "whispercpp", "mlx"]


def test_worker_fails_with_every_reason_when_nothing_is_available(monkeypatch) -> None:
    install_worker_backends(
        monkeypatch,
        {
            "parakeet": BackendUnavailableError("gone"),
            "whispercpp": BackendUnavailableError("no binary"),
            "mlx": BackendUnavailableError("not installed"),
        },
    )
    with pytest.raises(worker_cli.WorkerError) as info:
        worker_cli.transcribe_claim_audio(
            claim=known_speaker_claim(fallbacks=WHISPERCPP_AND_MLX), audio_path=Path("/tmp/a.wav")
        )
    message = str(info.value)
    assert "parakeet: gone" in message
    assert "whispercpp: no binary" in message
    assert "mlx: not installed" in message


def test_worker_fails_at_once_on_any_other_error(monkeypatch) -> None:
    calls = install_worker_backends(monkeypatch, {"parakeet": RuntimeError("Parakeet crashed")})
    with pytest.raises(RuntimeError, match="Parakeet crashed"):
        worker_cli.transcribe_claim_audio(
            claim=known_speaker_claim(fallbacks=WHISPERCPP_AND_MLX), audio_path=Path("/tmp/a.wav")
        )
    assert [name for name, _ in calls] == ["parakeet"]


def test_worker_without_fallbacks_behaves_as_before(monkeypatch) -> None:
    calls = install_worker_backends(monkeypatch, {"parakeet": BackendUnavailableError("gone")})
    with pytest.raises(worker_cli.WorkerError, match="parakeet: gone"):
        worker_cli.transcribe_claim_audio(
            claim=known_speaker_claim(), audio_path=Path("/tmp/a.wav")
        )
    assert [name for name, _ in calls] == ["parakeet"]


@pytest.mark.parametrize(
    "fallbacks",
    ["whispercpp", ["whispercpp"], [{"backend": "whispercpp"}], [{"model": "m", "backend": ""}]],
)
def test_worker_rejects_malformed_fallbacks(monkeypatch, fallbacks: object) -> None:
    calls = install_worker_backends(monkeypatch, {})
    with pytest.raises(worker_cli.WorkerError):
        worker_cli.transcribe_claim_audio(
            claim=known_speaker_claim(fallbacks=fallbacks), audio_path=Path("/tmp/a.wav")
        )
    assert calls == []


def test_remote_stt_waits_while_the_non_interactive_lane_is_occupied(
    monkeypatch, settings, tmp_path: Path
) -> None:
    settings.VOXHELM_LANE_SCHEDULER_ENABLED = True
    settings.VOXHELM_LANE_SCHEDULER_DIR = tmp_path / "lanes"
    settings.VOXHELM_LANE_SCHEDULER_NON_INTERACTIVE_SLOTS = 1
    calls = install_worker_backends(monkeypatch, {})
    finished = threading.Event()

    def run() -> None:
        worker_cli.transcribe_claim_audio(
            claim=known_speaker_claim(), audio_path=Path("/tmp/a.wav")
        )
        finished.set()

    with admit_local_inference(LANE_NON_INTERACTIVE):
        thread = threading.Thread(target=run)
        thread.start()
        time.sleep(0.6)
        assert calls == []
    thread.join(timeout=10)
    assert finished.is_set()
    assert [name for name, _ in calls] == ["parakeet"]


# --- worker: capability advertising ----------------------------------------------


@pytest.fixture
def parakeet_runtime(settings, tmp_path: Path):
    model_dir = tmp_path / "parakeet-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights")
    cli = tmp_path / "parakeet-mlx"
    cli.write_text("#!/bin/sh\necho parakeet-mlx 0.5.3\n", encoding="utf-8")
    cli.chmod(0o755)
    settings.VOXHELM_STT_BACKEND = "whispercpp"
    settings.VOXHELM_STT_FALLBACK_BACKEND = ""
    settings.VOXHELM_WHISPERCPP_MODEL = "ggml-large-v3.bin"
    settings.VOXHELM_WHISPERKIT_ENABLED = False
    settings.VOXHELM_DIARIZATION_BACKEND = "none"
    settings.VOXHELM_PARAKEET_BIN = str(cli)
    settings.VOXHELM_PARAKEET_MODEL_DIR = str(model_dir)
    settings.VOXHELM_PARAKEET_MODEL_NAME = "parakeet-tdt-0.6b-v3"
    return model_dir


def test_capabilities_advertise_parakeet_only_with_cli_and_complete_model(
    parakeet_runtime: Path, settings
) -> None:
    capabilities = worker_cli.build_capabilities()
    assert capabilities["backends"] == ["parakeet", "whispercpp"]
    assert "parakeet-tdt-0.6b-v3" in capabilities["models"]

    (parakeet_runtime / "model.safetensors").unlink()
    refreshed = worker_cli.with_current_stt_runtimes(capabilities)
    assert refreshed["backends"] == ["whispercpp"]
    assert "parakeet-tdt-0.6b-v3" not in refreshed["models"]
    assert refreshed["output_formats"] == capabilities["output_formats"]

    (parakeet_runtime / "model.safetensors").write_bytes(b"weights")
    settings.VOXHELM_PARAKEET_BIN = "voxhelm-no-such-parakeet"
    assert "parakeet" not in worker_cli.with_current_stt_runtimes(capabilities)["backends"]
    settings.VOXHELM_PARAKEET_BIN = str(parakeet_runtime.parent / "parakeet-mlx")
    settings.VOXHELM_PARAKEET_MODEL_DIR = ""
    assert "parakeet" not in worker_cli.with_current_stt_runtimes(capabilities)["backends"]


def test_worker_rechecks_parakeet_on_every_heartbeat(parakeet_runtime: Path, monkeypatch) -> None:
    monkeypatch.setattr(worker_cli, "setup_django_for_worker", lambda: None)
    monkeypatch.setattr(worker_cli, "log_startup", lambda config: None)
    seen: list[list[str]] = []

    def run_once(*, config, client, capabilities):
        del config, client
        seen.append(capabilities["backends"])
        if len(seen) == 1:
            (parakeet_runtime / "model.safetensors").unlink()
            return True
        raise KeyboardInterrupt

    monkeypatch.setattr(worker_cli, "run_worker_once", run_once)
    monkeypatch.setenv("VOXHELM_WORKER_TOKEN", "token")

    exit_code = worker_cli.main(["--base-url", "http://voxhelm.local", "--worker-id", "atlas"])

    assert exit_code == 130
    assert seen == [["parakeet", "whispercpp"], ["whispercpp"]]


def test_worker_logs_the_parakeet_cli_version(parakeet_runtime: Path, caplog) -> None:
    with caplog.at_level("INFO", logger="voxhelm.remote_worker"):
        worker_cli.log_parakeet_version()
    assert "parakeet cli version=parakeet-mlx 0.5.3" in caplog.text


# --- end to end with the re-cut ------------------------------------------------


def test_parakeet_words_drive_the_speaker_change_recut() -> None:
    # One Parakeet sentence spanning both speakers of the stub audio (Johannes
    # until 5 s, Dominik after), with sub-word and punctuation tokens.
    tokens = [
        (" ein", 0.5, 1.0),
        ("s", 1.0, 1.5),
        (" zwei", 1.5, 3.0),
        (" drei", 3.0, 4.1),
        (",", 4.1, 4.2),
        (" vier", 5.6, 6.5),
        (" fuenf", 6.5, 8.0),
        (" sechs", 8.0, 9.4),
        (".", 9.4, 9.5),
    ]
    payload = {
        "text": "eins zwei drei, vier fuenf sechs.",
        "sentences": [
            {
                "text": " eins zwei drei, vier fuenf sechs.",
                "start": 0.0,
                "end": 10.0,
                "tokens": [
                    {"text": text, "start": start, "end": end} for text, start, end in tokens
                ],
            }
        ],
    }
    result = normalize_parakeet_payload(payload, model_name="parakeet-tdt-0.6b-v3")

    outcome = run_known_speaker_postprocess(
        result,
        references=[make_reference("Johannes", 0.1), make_reference("Dominik", 0.2)],
        job_audio_samples=two_speaker_audio(),
        raw_turns=[],
        config=KnownSpeakerConfig(),
        backend=StubEmbeddingBackend(),
    )

    assert [(s.start, s.end, s.text) for s in outcome.result.segments] == [
        (0.0, 5.6, "eins zwei drei,"),
        (5.6, 10.0, "vier fuenf sechs."),
    ]
    assert [s.speaker for s in outcome.segments] == ["Johannes", "Dominik"]
    assert outcome.result.backend_name == "parakeet"


# --- review fixes: language and whisper.cpp advertising --------------------------


def test_known_speaker_chain_leaves_parakeet_out_for_unsupported_languages(stt_settings) -> None:
    services = get_backend_services_for_params(
        TranscribeParams(request_model="auto", prompt=None, language="ja", word_timestamps=True)
    )
    assert [invocation.name for invocation in services] == ["whispercpp", "mlx"]


@pytest.mark.django_db
def test_claim_skips_parakeet_for_a_language_it_does_not_support(client, stt_settings):
    submit_known_speaker_job(client, stt_settings)
    Job.objects.update(language="ja")
    only_parakeet = parakeet_capabilities()
    only_parakeet["backends"] = ["parakeet"]
    only_parakeet["models"] = ["parakeet-tdt-0.6b-v3"]

    assert claim_one(client, capabilities=only_parakeet).status_code == 204
    payload = claim_one(client, capabilities=parakeet_capabilities()).json()["job"]
    assert payload["backend"] == "whispercpp"
    assert "fallbacks" not in payload


@pytest.mark.parametrize("primary", ["mlx", "whisperkit"])
def test_capabilities_advertise_an_available_whispercpp_outside_the_chain(
    settings, tmp_path: Path, primary: str
) -> None:
    binary = tmp_path / "whisper-cli"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    model = tmp_path / "ggml-large-v3.bin"
    model.write_bytes(b"model")
    settings.VOXHELM_STT_BACKEND = primary
    settings.VOXHELM_STT_FALLBACK_BACKEND = ""
    settings.VOXHELM_WHISPERKIT_ENABLED = primary == "whisperkit"
    settings.VOXHELM_DIARIZATION_BACKEND = "none"
    settings.VOXHELM_PARAKEET_MODEL_DIR = ""
    settings.VOXHELM_WHISPERCPP_BIN = str(binary)
    settings.VOXHELM_WHISPERCPP_MODEL = str(model)

    assert worker_cli.build_capabilities()["backends"] == sorted({primary, "whispercpp"})

    model.unlink()
    assert worker_cli.build_capabilities()["backends"] == [primary]
