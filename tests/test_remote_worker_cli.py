from __future__ import annotations

import importlib.metadata
import json
import logging
import threading
from pathlib import Path
from typing import Any

import pytest

from jobs import remote_worker_cli as worker_cli
from jobs.artifacts import current_artifact_store_identity, get_artifact_store
from jobs.media import DownloadedMedia
from transcriptions.service import TranscriptionResult, TranscriptionSegment


class RecordingClient(worker_cli.WorkerClient):
    def __init__(self) -> None:
        super().__init__(
            config=worker_cli.WorkerConfig(
                base_url="http://voxhelm.local",
                worker_id="atlas",
                token="token",
                hostname="atlas.local",
            )
        )
        self.completed: dict[str, Any] | None = None
        self.heartbeats: list[dict[str, Any]] = []

    def heartbeat_job(
        self,
        *,
        job_id: str,
        lease_token: str,
        progress: dict[str, Any],
        stop_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        self.heartbeats.append({"job_id": job_id, "lease_token": lease_token, "progress": progress})
        return {}

    def complete_job(
        self,
        *,
        job_id: str,
        lease_token: str,
        result_text: str,
        result_metadata: dict[str, Any],
        artifacts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self.completed = {
            "job_id": job_id,
            "lease_token": lease_token,
            "result_text": result_text,
            "result_metadata": result_metadata,
            "artifacts": artifacts,
        }
        return {}


class FakeSttService:
    def transcribe(self, audio_path: Path, params: object) -> TranscriptionResult:
        del audio_path, params
        return sample_transcription_result()


def sample_transcription_result() -> TranscriptionResult:
    return TranscriptionResult(
        text="hello from atlas",
        language="en",
        segments=[TranscriptionSegment(id=0, start=0.0, end=1.25, text="hello from atlas")],
        backend_name="whisper.cpp",
        model_name="ggml-large-v3.bin",
    )


class LoopSttService:
    def transcribe(self, audio_path: Path, params: object) -> TranscriptionResult:
        del audio_path, params
        loop = [
            TranscriptionSegment(
                id=index,
                start=float(index),
                end=float(index + 1),
                text="Das ist auch sehr subjektiv.",
            )
            for index in range(18)
        ]
        return TranscriptionResult(
            text=" ".join(segment.text for segment in loop),
            language="de",
            segments=loop,
            backend_name="whisper.cpp",
            model_name="ggml-large-v3.bin",
        )


def test_transcribe_claim_audio_sanitizes_repeated_loop(monkeypatch: Any) -> None:
    monkeypatch.setattr(worker_cli, "build_backend_service", lambda **_: LoopSttService())

    result = worker_cli.transcribe_claim_audio(
        claim=base_claim(),
        audio_path=Path("/tmp/episode.wav"),
    )

    assert len(result.segments) == 1
    assert result.segments[0].text == "Das ist auch sehr subjektiv."
    assert result.text == "Das ist auch sehr subjektiv."


class RecordingSttService:
    def __init__(self) -> None:
        self.params: list[Any] = []

    def transcribe(self, audio_path: Path, params: object) -> TranscriptionResult:
        del audio_path
        self.params.append(params)
        return sample_transcription_result()


def test_transcribe_claim_audio_requests_word_timestamps_for_known_speaker_claims(
    monkeypatch: Any,
) -> None:
    service = RecordingSttService()
    monkeypatch.setattr(worker_cli, "build_backend_service", lambda **_: service)
    known_speaker_claim = base_claim()
    known_speaker_claim["output"]["diarization"] = {
        "enabled": True,
        "strategy": "pyannote_known_speaker",
    }

    worker_cli.transcribe_claim_audio(claim=base_claim(), audio_path=Path("/tmp/episode.wav"))
    worker_cli.transcribe_claim_audio(
        claim=known_speaker_claim, audio_path=Path("/tmp/episode.wav")
    )

    assert [params.word_timestamps for params in service.params] == [False, True]


def test_worker_version_reports_installed_package_version() -> None:
    try:
        expected = importlib.metadata.version("voxhelm")
    except importlib.metadata.PackageNotFoundError:
        expected = "unknown"
    assert worker_cli.worker_version() == expected


def test_log_startup_emits_version_and_worker_id(caplog: Any) -> None:
    config = worker_cli.WorkerConfig(
        base_url="http://voxhelm.local",
        worker_id="studio",
        token="token",
        hostname="studio.tailde2ec.ts.net",
    )
    with caplog.at_level(logging.INFO, logger="voxhelm.remote_worker"):
        worker_cli.log_startup(config)

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        f"version={worker_cli.worker_version()}" in message and "worker_id=studio" in message
        for message in messages
    )


def base_claim() -> dict[str, Any]:
    return {
        "id": "job-1",
        "attempt": 1,
        "lease_token": "lease-token",
        "backend": "whispercpp",
        "model": "ggml-large-v3.bin",
        "requested_backend": "auto",
        "requested_model": "auto",
        "language": "en",
        "input": {"kind": "url", "url": "https://media.example.com/episode.mp3"},
        "output": {"formats": ["text", "json"], "diarization": {"enabled": False}},
        "artifact_prefix": "voxhelm/jobs/job-1/attempt-1/",
        "artifact_store": current_artifact_store_identity(),
    }


def configure_filesystem_store(settings: Any, root: Path) -> None:
    settings.VOXHELM_ARTIFACT_BACKEND = "filesystem"
    settings.VOXHELM_ARTIFACT_ROOT = root
    get_artifact_store.cache_clear()


def install_fake_download(monkeypatch: Any, tmp_path: Path) -> None:
    def fake_download_allowed_media(*, source_url: str) -> DownloadedMedia:
        source_path = tmp_path / "downloaded-episode.mp3"
        source_path.write_bytes(b"audio-bytes")
        return DownloadedMedia(
            path=source_path,
            content_type="audio/mpeg",
            source_url=source_url,
            source_name="episode.mp3",
            source_kind="url",
        )

    monkeypatch.setattr(worker_cli, "download_allowed_media", fake_download_allowed_media)


def completed_payload(client: RecordingClient) -> dict[str, Any]:
    assert client.completed is not None
    return client.completed


def artifact_by_name(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {artifact["name"]: artifact for artifact in payload["artifacts"]}


def test_process_claim_uploads_source_and_requested_transcripts(
    monkeypatch: Any,
    settings: Any,
    tmp_path: Path,
) -> None:
    configure_filesystem_store(settings, tmp_path / "artifacts")
    install_fake_download(monkeypatch, tmp_path)
    monkeypatch.setattr(worker_cli, "build_backend_service", lambda **_: FakeSttService())

    client = RecordingClient()
    worker_cli.process_claim(
        base_claim(),
        config=worker_cli.WorkerConfig(
            base_url="http://voxhelm.local",
            worker_id="atlas",
            token="token",
            hostname="atlas.local",
            heartbeat_interval_seconds=999,
        ),
        client=client,
    )

    payload = completed_payload(client)
    artifacts = artifact_by_name(payload)
    assert payload["result_text"] == "hello from atlas"
    assert set(artifacts) == {"source.mp3", "transcript.txt", "transcript.json"}
    assert artifacts["source.mp3"]["exposed"] is False
    assert artifacts["transcript.txt"]["exposed"] is True
    assert artifacts["transcript.json"]["content_type"] == "application/json"
    assert (settings.VOXHELM_ARTIFACT_ROOT / artifacts["source.mp3"]["storage_key"]).read_bytes()
    assert (
        settings.VOXHELM_ARTIFACT_ROOT / artifacts["transcript.txt"]["storage_key"]
    ).read_text() == "hello from atlas"
    transcript_json = json.loads(
        (settings.VOXHELM_ARTIFACT_ROOT / artifacts["transcript.json"]["storage_key"]).read_text()
    )
    assert transcript_json["text"] == "hello from atlas"
    assert payload["result_metadata"]["duration_seconds"] == 1.25
    assert payload["result_metadata"]["processing_seconds"] >= 0
    assert "source_url" not in payload["result_metadata"]
    assert client.heartbeats[0]["progress"]["phase"] == "materialize_input"


def test_process_claim_uploads_known_speaker_sidecar(
    monkeypatch: Any,
    settings: Any,
    tmp_path: Path,
) -> None:
    configure_filesystem_store(settings, tmp_path / "artifacts")
    install_fake_download(monkeypatch, tmp_path)
    monkeypatch.setattr(worker_cli, "build_backend_service", lambda **_: FakeSttService())

    def fake_known_speaker_for_claim(
        *,
        diarization: dict[str, Any],
        audio_path: Path,
        result: TranscriptionResult,
        source_media: DownloadedMedia | None = None,
        cancel_event: Any = None,
    ) -> tuple[TranscriptionResult, dict[str, Any]]:
        assert cancel_event is not None
        assert not cancel_event.is_set()
        assert source_media is not None
        assert source_media.path == audio_path
        assert source_media.source_url == "https://media.example.com/episode.mp3"
        assert diarization["strategy"] == "pyannote_known_speaker"
        return result, {
            "version": 1,
            "summary": {
                "strategy": "pyannote_known_speaker",
                "embedding_model": "pyannote/wespeaker-voxceleb-resnet34-LM",
                "known_speakers": ["Johannes"],
                "segment_count": 1,
                "confident_segment_count": 1,
                "uncertain_segment_count": 0,
            },
            "segments": [],
        }

    monkeypatch.setattr(worker_cli, "run_known_speaker_for_claim", fake_known_speaker_for_claim)
    claim = base_claim()
    claim["output"] = {
        "formats": ["text"],
        "diarization": {
            "enabled": True,
            "strategy": "pyannote_known_speaker",
            "known_speakers": [{"id": "12", "name": "Johannes", "references": []}],
        },
    }

    client = RecordingClient()
    worker_cli.process_claim(
        claim,
        config=worker_cli.WorkerConfig(
            base_url="http://voxhelm.local",
            worker_id="atlas",
            token="token",
            hostname="atlas.local",
            heartbeat_interval_seconds=999,
        ),
        client=client,
    )

    payload = completed_payload(client)
    artifacts = artifact_by_name(payload)
    assert set(artifacts) == {"source.mp3", "transcript.txt", "transcript.speakers.json"}
    assert artifacts["transcript.speakers.json"]["kind"] == "transcript_speakers"
    assert artifacts["transcript.speakers.json"]["format"] == "speakers"
    assert artifacts["transcript.speakers.json"]["exposed"] is True
    sidecar = json.loads(
        (
            settings.VOXHELM_ARTIFACT_ROOT / artifacts["transcript.speakers.json"]["storage_key"]
        ).read_text()
    )
    assert sidecar["summary"]["known_speakers"] == ["Johannes"]
    assert (
        payload["result_metadata"]["diarization"]["known_speaker_summary"]["strategy"]
        == "pyannote_known_speaker"
    )


def test_build_capabilities_advertises_known_speaker_when_diarization_enabled(
    settings: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings.VOXHELM_STT_BACKEND = "whispercpp"
    settings.VOXHELM_STT_FALLBACK_BACKEND = ""
    settings.VOXHELM_WHISPERCPP_MODEL = "ggml-large-v3.bin"
    settings.VOXHELM_WHISPERKIT_ENABLED = False
    settings.VOXHELM_DIARIZATION_BACKEND = "pyannote"
    settings.VOXHELM_HUGGINGFACE_TOKEN = "hf-token"
    monkeypatch.setattr(worker_cli, "module_available", lambda module_name: True)

    capabilities = worker_cli.build_capabilities()

    assert capabilities["backends"] == ["whispercpp"]
    assert "ggml-large-v3.bin" in capabilities["models"]
    assert "speakers" in capabilities["output_formats"]
    assert capabilities["diarization"] == {
        "anonymous": True,
        "known_speaker": True,
        "embedding_models": ["pyannote/wespeaker-voxceleb-resnet34-LM"],
    }


def test_build_capabilities_does_not_advertise_missing_diarization_runtime(
    settings: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings.VOXHELM_STT_BACKEND = "whispercpp"
    settings.VOXHELM_STT_FALLBACK_BACKEND = ""
    settings.VOXHELM_WHISPERCPP_MODEL = "ggml-large-v3.bin"
    settings.VOXHELM_WHISPERKIT_ENABLED = False
    settings.VOXHELM_DIARIZATION_BACKEND = "pyannote"
    settings.VOXHELM_HUGGINGFACE_TOKEN = "hf-token"
    monkeypatch.setattr(
        worker_cli,
        "module_available",
        lambda module_name: module_name != "pyannote.audio",
    )

    capabilities = worker_cli.build_capabilities()

    assert "speakers" not in capabilities["output_formats"]
    assert capabilities["diarization"] == {
        "anonymous": False,
        "known_speaker": False,
        "embedding_models": [],
    }


def test_parse_worker_config_uses_django_token_map(settings: Any) -> None:
    settings.VOXHELM_WORKER_TOKENS = {"atlas": "json-style-token"}
    args = worker_cli.build_arg_parser().parse_args(
        ["--base-url", "http://voxhelm.local", "--worker-id", "atlas"]
    )

    config = worker_cli.parse_worker_config(args)

    assert config.token == "json-style-token"


def test_materialize_staged_upload_removes_reserved_file_on_download_failure(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "reserved.mp3"

    def fake_reserve_temp_media_path(*, suffix: str) -> Path:
        assert suffix == ".mp3"
        destination.write_bytes(b"")
        return destination

    class FailingStore:
        def download_file(self, *, key: str, destination_path: Path) -> None:
            assert key == "staged/input.mp3"
            assert destination_path == destination
            raise RuntimeError("download failed")

    monkeypatch.setattr(worker_cli, "reserve_temp_media_path", fake_reserve_temp_media_path)
    monkeypatch.setattr(
        worker_cli,
        "get_artifact_store_for_identity",
        lambda _identity: FailingStore(),
    )

    with pytest.raises(RuntimeError, match="download failed"):
        worker_cli.materialize_staged_upload(
            {
                "kind": "upload",
                "filename": "input.mp3",
                "content_type": "audio/mpeg",
                "staged_artifact": {"storage_key": "staged/input.mp3"},
            }
        )

    assert not destination.exists()


@pytest.mark.parametrize("status", [200, 301, 302, 303, 307, 308])
@pytest.mark.parametrize("operation", ["heartbeat", "claim"])
def test_worker_api_refuses_credential_redirects(status, operation):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    received: list[str | None] = []
    initial: list[str | None] = []

    class Destination(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):  # noqa: N802
            received.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"job": {}}')

        do_POST = do_GET

    destination = ThreadingHTTPServer(("127.0.0.1", 0), Destination)
    destination_thread = threading.Thread(target=destination.serve_forever, daemon=True)
    destination_thread.start()

    class Origin(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):  # noqa: N802
            # Drain the POST body before closing; unread bytes can cause a TCP
            # reset that hides an otherwise-successful response on macOS.
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            initial.append(self.headers.get("Authorization"))
            self.send_response(status)
            self.send_header(
                "Location", f"http://127.0.0.1:{destination.server_port}/credential-destination"
            )
            # A redirect body can reflect a credential; it must never be displayed.
            body = b'{"job": {}, "echo": "synthetic-worker-credential"}'
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    origin = ThreadingHTTPServer(("127.0.0.1", 0), Origin)
    origin_thread = threading.Thread(target=origin.serve_forever, daemon=True)
    origin_thread.start()
    client = worker_cli.WorkerClient(
        config=worker_cli.WorkerConfig(
            base_url=f"http://127.0.0.1:{origin.server_port}",
            worker_id="test-worker",
            token="synthetic-worker-credential",
            hostname="test-host",
            http_timeout_seconds=2,
        )
    )

    def request():
        if operation == "heartbeat":
            return client.heartbeat_worker(capabilities={}, running_job_ids=[])
        return client.claim(capabilities={})

    try:
        if status == 200:
            assert isinstance(request(), dict)
        else:
            with pytest.raises(worker_cli.WorkerError, match="redirects are refused") as error:
                request()
            assert "synthetic-worker-credential" not in str(error.value)
            assert "credential-destination" not in str(error.value)
            assert len(str(error.value)) < 200
        assert initial == ["Bearer synthetic-worker-credential"]
        assert received == []
    finally:
        origin.shutdown()
        destination.shutdown()
        origin.server_close()
        destination.server_close()
        origin_thread.join(timeout=3)
        destination_thread.join(timeout=3)


@pytest.mark.parametrize("log_level", ["INFO", "CRITICAL"])
def test_main_warns_for_token_flag_without_disclosing_value(monkeypatch, capsys, log_level):
    monkeypatch.setenv("VOXHELM_WORKER_TOKEN", "environment-secret")
    monkeypatch.setattr(worker_cli, "setup_django_for_worker", lambda: None)
    monkeypatch.setattr(worker_cli, "build_capabilities", lambda: {})
    monkeypatch.setattr(worker_cli, "log_startup", lambda config: None)
    seen = []

    def run_once(*, config, client, capabilities):
        seen.append(config.token)
        return False

    monkeypatch.setattr(worker_cli, "run_worker_once", run_once)
    assert (
        worker_cli.main(
            [
                "--base-url",
                "http://voxhelm.local",
                "--worker-id",
                "atlas",
                "--token",
                "argv-secret",
                "--once",
                "--log-level",
                log_level,
            ]
        )
        == 0
    )
    assert seen == ["argv-secret"]
    output = capsys.readouterr()
    assert "--token is deprecated" in output.err
    assert "VOXHELM_WORKER_TOKEN" in output.err
    assert "--env-file" in output.err
    assert "argv-secret" not in output.out + output.err
    assert "environment-secret" not in output.out + output.err


@pytest.mark.parametrize("use_env_file", [False, True])
def test_main_environment_token_does_not_warn(monkeypatch, tmp_path, capsys, use_env_file):
    # Register restoration even when load_env_file writes an initially absent key.
    monkeypatch.setenv("VOXHELM_WORKER_TOKEN", "placeholder")
    monkeypatch.delenv("VOXHELM_WORKER_TOKEN")
    args = ["--base-url", "http://voxhelm.local", "--worker-id", "atlas", "--once"]
    if use_env_file:
        env_file = tmp_path / "worker.env"
        env_file.write_text("VOXHELM_WORKER_TOKEN=environment-secret\n")
        args.extend(["--env-file", str(env_file)])
    else:
        monkeypatch.setenv("VOXHELM_WORKER_TOKEN", "environment-secret")
    monkeypatch.setattr(worker_cli, "setup_django_for_worker", lambda: None)
    monkeypatch.setattr(worker_cli, "build_capabilities", lambda: {})
    monkeypatch.setattr(worker_cli, "log_startup", lambda config: None)
    seen = []

    def run_once(*, config, client, capabilities):
        seen.append(config.token)
        return False

    monkeypatch.setattr(worker_cli, "run_worker_once", run_once)
    assert worker_cli.main(args) == 0
    assert seen == ["environment-secret"]
    output = capsys.readouterr()
    assert output.err == ""
    assert "environment-secret" not in output.out


@pytest.mark.parametrize("body_error", [False, True])
def test_heartbeater_exit_cancels_active_http(monkeypatch, body_error):
    import asyncio

    started = threading.Event()
    closed = threading.Event()
    client = worker_cli.WorkerClient(config=RecordingClient().config)

    import httpx

    class BlockedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            started.set()
            await asyncio.sleep(8)
            yield b"{}"

        async def aclose(self):
            closed.set()

    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        worker_cli.httpx,
        "AsyncClient",
        lambda **kwargs: original_client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, stream=BlockedStream())
            ),
            **kwargs,
        ),
    )
    heartbeater = worker_cli.LeaseHeartbeater(
        client=client, job_id="synthetic-job", lease_token="synthetic-lease", interval_seconds=0.001
    )
    # Avoid sleeping for a full production interval in this synchronization test.
    heartbeater.interval_seconds = 0.001
    try:
        with heartbeater:
            assert started.wait(2)
            if body_error:
                raise ValueError("synthetic body failure")
    except ValueError:
        assert body_error
    assert closed.is_set()
    assert not heartbeater._thread.is_alive()
    heartbeater._beat_once()


@pytest.mark.parametrize("status", [401, 403, 404, 409, 503])
@pytest.mark.parametrize("loss_phase", ["transcribe", "during_inference"])
def test_claim_stops_after_lease_loss(monkeypatch, settings, tmp_path, status, loss_phase):
    configure_filesystem_store(settings, tmp_path / "artifacts")
    install_fake_download(monkeypatch, tmp_path)
    client = RecordingClient()
    inferred = []
    uploaded = []
    revoked = threading.Event()

    def beat(**kwargs):
        if kwargs["progress"]["phase"] == loss_phase or revoked.is_set():
            raise worker_cli.WorkerHttpError(status=status, detail="synthetic")
        return {}

    def infer(**kwargs):
        inferred.append(True)
        if loss_phase == "during_inference":
            revoked.set()
            # Simulate the periodic beat observing loss while the model call runs.
            active[0]._beat_once()
        return sample_transcription_result()

    active = []
    original_enter = worker_cli.LeaseHeartbeater.__enter__

    def enter(heartbeater):
        active.append(heartbeater)
        return original_enter(heartbeater)

    monkeypatch.setattr(worker_cli.LeaseHeartbeater, "__enter__", enter)
    monkeypatch.setattr(client, "heartbeat_job", beat)
    monkeypatch.setattr(worker_cli, "transcribe_claim_audio", infer)
    original_upload = worker_cli.upload_transcript_artifacts

    def upload(**kwargs):
        uploaded.append(True)
        return original_upload(**kwargs)

    monkeypatch.setattr(worker_cli, "upload_transcript_artifacts", upload)
    if status == 503:
        worker_cli.process_claim(base_claim(), config=client.config, client=client)
        assert client.completed is not None
        assert inferred and uploaded
    else:
        with pytest.raises(worker_cli.WorkerError, match="revoked"):
            worker_cli.process_claim(base_claim(), config=client.config, client=client)
        assert bool(inferred) == (loss_phase == "during_inference")
        assert not uploaded
        assert client.completed is None


def test_lost_lease_is_not_reported_as_failure(monkeypatch):
    client = RecordingClient()
    monkeypatch.setattr(client, "heartbeat_worker", lambda **kwargs: {})
    monkeypatch.setattr(client, "claim", lambda **kwargs: base_claim())
    monkeypatch.setattr(client, "fail_job", lambda **kwargs: pytest.fail("stale failure report"))

    def process(*args, **kwargs):
        raise worker_cli.WorkerLeaseLost("synthetic revoked lease")

    monkeypatch.setattr(worker_cli, "process_claim", process)
    with pytest.raises(worker_cli.WorkerLeaseLost):
        worker_cli.run_worker_once(config=client.config, client=client, capabilities={})


@pytest.mark.parametrize(
    "original_error", [None, ValueError("synthetic"), worker_cli.WorkerLeaseLost("revoked")]
)
def test_heartbeat_join_timeout_preserves_original_outcome(original_error):
    from unittest.mock import Mock

    beat = worker_cli.LeaseHeartbeater(
        client=RecordingClient(), job_id="synthetic", lease_token="lease", interval_seconds=60
    )
    beat._thread = Mock()
    beat._thread.is_alive.return_value = True
    if original_error is None:
        with pytest.raises(worker_cli.WorkerHeartbeatShutdownError):
            beat.__exit__(None, None, None)
    else:
        # Context manager propagation preserves the body's existing exception.
        beat.__exit__(type(original_error), original_error, None)
    beat._thread.join.assert_called_once_with(timeout=5)


def test_lease_loss_during_failed_inference_does_not_report_failure(
    monkeypatch, settings, tmp_path
):
    configure_filesystem_store(settings, tmp_path / "artifacts")
    install_fake_download(monkeypatch, tmp_path)
    client = RecordingClient()
    active = []
    original_enter = worker_cli.LeaseHeartbeater.__enter__

    def enter(beat):
        active.append(beat)
        return original_enter(beat)

    def infer(**kwargs):
        active[0]._lost.set()
        raise ValueError("synthetic model failure after revocation")

    monkeypatch.setattr(worker_cli.LeaseHeartbeater, "__enter__", enter)
    monkeypatch.setattr(worker_cli, "transcribe_claim_audio", infer)
    monkeypatch.setattr(client, "heartbeat_worker", lambda **kwargs: {})
    monkeypatch.setattr(client, "claim", lambda **kwargs: base_claim())
    monkeypatch.setattr(client, "fail_job", lambda **kwargs: pytest.fail("stale failure report"))
    with pytest.raises(worker_cli.WorkerLeaseLost):
        worker_cli.run_worker_once(config=client.config, client=client, capabilities={})


def test_heartbeat_shutdown_error_does_not_report_failure(monkeypatch):
    client = RecordingClient()
    monkeypatch.setattr(client, "heartbeat_worker", lambda **kwargs: {})
    monkeypatch.setattr(client, "claim", lambda **kwargs: base_claim())
    monkeypatch.setattr(
        client, "fail_job", lambda **kwargs: pytest.fail("duplicate failure report")
    )

    def process(*args, **kwargs):
        raise worker_cli.WorkerHeartbeatShutdownError("synthetic")

    monkeypatch.setattr(worker_cli, "process_claim", process)
    with pytest.raises(worker_cli.WorkerHeartbeatShutdownError):
        worker_cli.run_worker_once(config=client.config, client=client, capabilities={})


def test_late_heartbeat_conflict_does_not_override_completion(monkeypatch, settings, tmp_path):
    configure_filesystem_store(settings, tmp_path / "artifacts")
    install_fake_download(monkeypatch, tmp_path)
    client = RecordingClient()
    active = []
    original_enter = worker_cli.LeaseHeartbeater.__enter__
    original_complete = client.complete_job

    def enter(beat):
        active.append(beat)
        return original_enter(beat)

    def complete(**kwargs):
        result = original_complete(**kwargs)
        # The server can reject an in-flight heartbeat after committing completion.
        active[0]._lost.set()
        return result

    monkeypatch.setattr(worker_cli.LeaseHeartbeater, "__enter__", enter)
    monkeypatch.setattr(worker_cli, "build_backend_service", lambda **kwargs: FakeSttService())
    monkeypatch.setattr(client, "complete_job", complete)
    monkeypatch.setattr(client, "heartbeat_worker", lambda **kwargs: {})
    monkeypatch.setattr(client, "claim", lambda **kwargs: base_claim())
    monkeypatch.setattr(client, "fail_job", lambda **kwargs: pytest.fail("duplicate report"))
    assert worker_cli.run_worker_once(config=client.config, client=client, capabilities={})
    assert client.completed is not None


@pytest.mark.parametrize("outcome", ["no_work", "success", "failure", "report_failure"])
def test_once_exit_status_tracks_processing_outcome(monkeypatch, outcome):
    client = RecordingClient()
    reports = []
    monkeypatch.setenv("VOXHELM_WORKER_TOKEN", "synthetic-credential")
    monkeypatch.setattr(worker_cli, "setup_django_for_worker", lambda: None)
    monkeypatch.setattr(worker_cli, "log_parakeet_version", lambda: None)
    monkeypatch.setattr(worker_cli, "build_capabilities", lambda: {})
    monkeypatch.setattr(worker_cli, "with_current_stt_runtimes", lambda caps: caps)
    monkeypatch.setattr(worker_cli, "WorkerClient", lambda **kwargs: client)
    monkeypatch.setattr(client, "heartbeat_worker", lambda **kwargs: {})
    monkeypatch.setattr(
        client, "claim", lambda **kwargs: None if outcome == "no_work" else base_claim()
    )

    def process(*args, **kwargs):
        if outcome in {"failure", "report_failure"}:
            raise ValueError("synthetic inference failed")

    def report(**kwargs):
        reports.append(kwargs)
        if outcome == "report_failure":
            raise worker_cli.WorkerError("synthetic report failed")
        return {}

    monkeypatch.setattr(worker_cli, "process_claim", process)
    monkeypatch.setattr(client, "fail_job", report)
    result = worker_cli.main(
        ["--base-url", "http://synthetic.invalid", "--worker-id", "synthetic", "--once"]
    )
    assert result == (1 if outcome in {"failure", "report_failure"} else 0)
    assert bool(reports) == (outcome in {"failure", "report_failure"})


@pytest.mark.parametrize(
    "option", ["--poll-seconds", "--http-timeout-seconds", "--heartbeat-interval-seconds"]
)
@pytest.mark.parametrize("value", ["0", "-1", "3601", str(10**400)])
def test_startup_rejects_bad_timing_before_runtime_or_network(monkeypatch, option, value):
    monkeypatch.setenv("VOXHELM_WORKER_TOKEN", "synthetic-credential")
    monkeypatch.setattr(worker_cli, "setup_django_for_worker", lambda: None)
    monkeypatch.setattr(worker_cli, "log_parakeet_version", lambda: pytest.fail("runtime probe"))
    monkeypatch.setattr(
        worker_cli, "WorkerClient", lambda **kwargs: pytest.fail("client constructed")
    )
    with pytest.raises(SystemExit) as error:
        worker_cli.main(
            [
                "--base-url",
                "http://synthetic.invalid",
                "--worker-id",
                "synthetic",
                option,
                value,
                "--once",
            ]
        )

    assert error.value.code == 2


@pytest.mark.parametrize("hint", [True, False, 0, -1, 3601, 10**400, 1.5, "5"])
def test_invalid_server_poll_hint_never_claims(monkeypatch, hint):
    client = RecordingClient()
    monkeypatch.setattr(client, "heartbeat_worker", lambda **kwargs: {"poll_after_seconds": hint})
    monkeypatch.setattr(client, "claim", lambda **kwargs: pytest.fail("invalid hint reached claim"))
    with pytest.raises(worker_cli.WorkerError, match="poll_after_seconds"):
        worker_cli.run_worker_once(config=client.config, client=client, capabilities={})


@pytest.mark.parametrize("hint", [1, 5, 3600])
def test_valid_server_poll_hint_is_used(monkeypatch, hint):
    client = RecordingClient()
    monkeypatch.setattr(client, "heartbeat_worker", lambda **kwargs: {"poll_after_seconds": hint})
    monkeypatch.setattr(client, "claim", lambda **kwargs: None)
    assert not worker_cli.run_worker_once(config=client.config, client=client, capabilities={})
    assert client.config.poll_seconds == hint


def test_continuous_worker_keeps_polling_after_reported_failure(monkeypatch):
    client = RecordingClient()
    calls = []
    original_run = worker_cli.run_worker_once
    monkeypatch.setenv("VOXHELM_WORKER_TOKEN", "synthetic-credential")
    monkeypatch.setattr(worker_cli, "setup_django_for_worker", lambda: None)
    monkeypatch.setattr(worker_cli, "log_parakeet_version", lambda: None)
    monkeypatch.setattr(worker_cli, "build_capabilities", lambda: {})
    monkeypatch.setattr(worker_cli, "with_current_stt_runtimes", lambda caps: caps)
    monkeypatch.setattr(worker_cli, "WorkerClient", lambda **kwargs: client)
    monkeypatch.setattr(client, "heartbeat_worker", lambda **kwargs: {})
    monkeypatch.setattr(client, "claim", lambda **kwargs: base_claim())
    monkeypatch.setattr(client, "fail_job", lambda **kwargs: {})
    monkeypatch.setattr(worker_cli, "sleep", lambda seconds: pytest.fail("unexpected idle wait"))

    def process(*args, **kwargs):
        raise ValueError("synthetic inference error")

    def run(**kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return original_run(**kwargs)

    monkeypatch.setattr(worker_cli, "process_claim", process)
    monkeypatch.setattr(worker_cli, "run_worker_once", run)
    assert (
        worker_cli.main(["--base-url", "http://synthetic.invalid", "--worker-id", "synthetic"])
        == 130
    )
    assert len(calls) == 2


def run_loop_with_scripted_polls(monkeypatch, script, *, poll_seconds=5):
    """Run the continuous loop over scripted poll results; stop after the script."""
    client = RecordingClient()
    object.__setattr__(client.config, "poll_seconds", poll_seconds)
    remaining = list(script)
    delays: list[float] = []

    def run(**kwargs):
        if not remaining:
            raise KeyboardInterrupt
        step = remaining.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step

    monkeypatch.setattr(worker_cli, "with_current_stt_runtimes", lambda caps: caps)
    monkeypatch.setattr(worker_cli, "run_worker_once", run)
    monkeypatch.setattr(worker_cli, "sleep", lambda seconds: pytest.fail("real sleep used"))
    result = worker_cli.run_worker_loop(
        config=client.config, client=client, capabilities={}, sleep_fn=delays.append
    )
    assert result == 130
    return delays


def forbidden() -> worker_cli.WorkerHttpError:
    return worker_cli.WorkerRejected(status=403, detail="Forbidden")


def test_rejected_worker_logs_one_concise_line_without_traceback(monkeypatch, caplog):
    with caplog.at_level(logging.DEBUG, logger="voxhelm.remote_worker"):
        delays = run_loop_with_scripted_polls(monkeypatch, [forbidden()])

    assert delays == [5]
    records = [record for record in caplog.records if record.name == "voxhelm.remote_worker"]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert records[0].exc_info is None
    assert records[0].getMessage() == (
        "control plane rejected worker (HTTP 403, worker disabled?); retrying in 5s worker_id=atlas"
    )
    assert "Traceback" not in caplog.text


def test_rejected_worker_backs_off_exponentially_up_to_cap(monkeypatch, caplog):
    with caplog.at_level(logging.INFO, logger="voxhelm.remote_worker"):
        delays = run_loop_with_scripted_polls(monkeypatch, [forbidden() for _ in range(8)])

    assert delays == [5, 10, 20, 40, 60, 60, 60, 60]
    assert len(caplog.records) == 8
    assert all(record.exc_info is None for record in caplog.records)
    assert "Traceback" not in caplog.text


def test_backoff_cap_never_polls_faster_than_poll_interval(monkeypatch):
    delays = run_loop_with_scripted_polls(monkeypatch, [forbidden(), forbidden()], poll_seconds=90)
    assert delays == [90, 90]


def test_backoff_resets_after_successful_poll(monkeypatch, caplog):
    no_work = worker_cli.WorkerOutcome.NO_WORK
    with caplog.at_level(logging.INFO, logger="voxhelm.remote_worker"):
        delays = run_loop_with_scripted_polls(
            monkeypatch,
            [forbidden(), forbidden(), forbidden(), no_work, forbidden(), no_work, no_work],
        )

    # Backoff 5, 10, 20; idle poll 5; the next rejection starts again at 5.
    assert delays == [5, 10, 20, 5, 5, 5, 5]
    accepted = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("control plane accepted worker again")
    ]
    assert accepted == [
        "control plane accepted worker again after 3 failed poll(s) worker_id=atlas",
        "control plane accepted worker again after 1 failed poll(s) worker_id=atlas",
    ]


@pytest.mark.parametrize(
    "error",
    [
        worker_cli.WorkerHttpError(status=503, detail="Service Unavailable"),
        worker_cli.WorkerError("Could not reach Voxhelm worker API: ConnectError."),
        # A 403 from a job report after a claim is not a disabled worker.
        worker_cli.WorkerHttpError(status=403, detail="Forbidden"),
        worker_cli.WorkerLeaseLost("lease lost"),
    ],
)
def test_other_poll_errors_keep_traceback_and_back_off(monkeypatch, caplog, error):
    with caplog.at_level(logging.INFO, logger="voxhelm.remote_worker"):
        delays = run_loop_with_scripted_polls(
            monkeypatch, [error, error, error, worker_cli.WorkerOutcome.NO_WORK]
        )

    assert delays == [5, 10, 20, 5]
    failures = [r for r in caplog.records if r.getMessage().startswith("worker poll failed")]
    assert len(failures) == 3
    assert all(record.exc_info is not None for record in failures)
    assert "retrying in 5s" in failures[0].getMessage()
    assert not any("rejected worker" in r.getMessage() for r in caplog.records)
    # Only a rejection logs a recovery line.
    assert len(caplog.records) == 3


def test_403_from_heartbeat_or_claim_is_a_rejection(monkeypatch):
    client = RecordingClient()

    def reject(**kwargs):
        raise worker_cli.WorkerHttpError(status=403, detail="Forbidden")

    monkeypatch.setattr(client, "heartbeat_worker", reject)
    with pytest.raises(worker_cli.WorkerRejected):
        worker_cli.run_worker_once(config=client.config, client=client, capabilities={})

    client = RecordingClient()
    monkeypatch.setattr(client, "heartbeat_worker", lambda **kwargs: {})
    monkeypatch.setattr(client, "claim", reject)
    with pytest.raises(worker_cli.WorkerRejected):
        worker_cli.run_worker_once(config=client.config, client=client, capabilities={})

    client = RecordingClient()

    def unavailable(**kwargs):
        raise worker_cli.WorkerHttpError(status=503, detail="Service Unavailable")

    monkeypatch.setattr(client, "heartbeat_worker", lambda **kwargs: {})
    monkeypatch.setattr(client, "claim", unavailable)
    with pytest.raises(worker_cli.WorkerHttpError) as error:
        worker_cli.run_worker_once(config=client.config, client=client, capabilities={})
    assert not isinstance(error.value, worker_cli.WorkerRejected)


def test_unexpected_poll_error_keeps_traceback_and_backs_off(monkeypatch, caplog):
    with caplog.at_level(logging.INFO, logger="voxhelm.remote_worker"):
        delays = run_loop_with_scripted_polls(
            monkeypatch, [ValueError("synthetic bug"), ValueError("synthetic bug")]
        )

    assert delays == [5, 10]
    assert all(record.exc_info is not None for record in caplog.records)


def test_once_rejection_logs_one_line_and_exits_nonzero(monkeypatch, caplog):
    client = RecordingClient()

    def reject(**kwargs):
        raise worker_cli.WorkerHttpError(status=403, detail="Forbidden")

    monkeypatch.setattr(worker_cli, "with_current_stt_runtimes", lambda caps: caps)
    monkeypatch.setattr(client, "heartbeat_worker", reject)
    with caplog.at_level(logging.DEBUG, logger="voxhelm.remote_worker"):
        result = worker_cli.run_worker_loop(
            config=client.config,
            client=client,
            capabilities={},
            once=True,
            sleep_fn=lambda seconds: pytest.fail("--once must not wait"),
        )

    assert result == 1
    assert [record.getMessage() for record in caplog.records] == [
        "control plane rejected worker (HTTP 403, worker disabled?) worker_id=atlas"
    ]
    assert caplog.records[0].exc_info is None


def test_interrupt_during_backoff_exits_like_shutdown(monkeypatch):
    client = RecordingClient()

    def reject(**kwargs):
        raise worker_cli.WorkerHttpError(status=403, detail="Forbidden")

    def interrupted_sleep(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(worker_cli, "with_current_stt_runtimes", lambda caps: caps)
    monkeypatch.setattr(client, "heartbeat_worker", reject)
    assert (
        worker_cli.run_worker_loop(
            config=client.config, client=client, capabilities={}, sleep_fn=interrupted_sleep
        )
        == 130
    )


def test_accepted_again_after_mixed_failure_streak(monkeypatch, caplog):
    unavailable = worker_cli.WorkerHttpError(status=503, detail="Service Unavailable")
    with caplog.at_level(logging.INFO, logger="voxhelm.remote_worker"):
        run_loop_with_scripted_polls(
            monkeypatch, [forbidden(), unavailable, worker_cli.WorkerOutcome.NO_WORK]
        )
    assert caplog.records[-1].getMessage() == (
        "control plane accepted worker again after 2 failed poll(s) worker_id=atlas"
    )
