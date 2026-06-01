from __future__ import annotations

import json
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
    ) -> dict[str, Any]:
        self.heartbeats.append(
            {"job_id": job_id, "lease_token": lease_token, "progress": progress}
        )
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
            TranscriptionSegment(id=index, start=float(index), end=float(index + 1),
                                 text="Das ist auch sehr subjektiv.")
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
    ) -> tuple[TranscriptionResult, dict[str, Any]]:
        del audio_path
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
            settings.VOXHELM_ARTIFACT_ROOT
            / artifacts["transcript.speakers.json"]["storage_key"]
        ).read_text()
    )
    assert sidecar["summary"]["known_speakers"] == ["Johannes"]
    assert payload["result_metadata"]["diarization"]["known_speaker_summary"][
        "strategy"
    ] == "pyannote_known_speaker"


def test_build_capabilities_advertises_known_speaker_when_diarization_enabled(
    settings: Any,
) -> None:
    settings.VOXHELM_STT_BACKEND = "whispercpp"
    settings.VOXHELM_STT_FALLBACK_BACKEND = ""
    settings.VOXHELM_WHISPERCPP_MODEL = "ggml-large-v3.bin"
    settings.VOXHELM_WHISPERKIT_ENABLED = False
    settings.VOXHELM_DIARIZATION_BACKEND = "pyannote"

    capabilities = worker_cli.build_capabilities()

    assert capabilities["backends"] == ["whispercpp"]
    assert "ggml-large-v3.bin" in capabilities["models"]
    assert "speakers" in capabilities["output_formats"]
    assert capabilities["diarization"] == {
        "anonymous": True,
        "known_speaker": True,
        "embedding_models": ["pyannote/wespeaker-voxceleb-resnet34-LM"],
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
