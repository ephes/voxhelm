from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import logging
import os
import platform
import shlex
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, sleep
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from django.conf import settings

from jobs.artifacts import get_artifact_store_for_identity
from jobs.media import (
    DownloadedMedia,
    detect_media_suffix,
    download_allowed_media,
    extract_audio_from_video,
    is_video_path,
    reserve_temp_media_path,
)
from transcriptions import formats as transcript_formats
from transcriptions.diarization import (
    DiarizationError,
    DiarizationParams,
    apply_speaker_labels,
    diarize_audio,
)
from transcriptions.known_speaker import (
    ANONYMOUS_STRATEGY,
    KNOWN_SPEAKER_STRATEGY,
    SAMPLE_RATE,
    KnownSpeakerConfig,
    ReferenceAudio,
    build_speakers_artifact,
    decode_mono_16k,
    extract_reference_windows,
    get_known_speaker_backend,
    run_known_speaker_postprocess,
    slice_samples,
)
from transcriptions.sanitizer import sanitize_result
from transcriptions.service import (
    TranscribeParams,
    TranscriptionResult,
    build_backend_service,
    resolve_model_name_for_backend,
)

LOGGER = logging.getLogger("voxhelm.remote_worker")

TRANSCRIPT_ARTIFACTS_BY_FORMAT: dict[str, tuple[str, str, str]] = {
    "text": ("transcript.txt", "transcript_text", "text/plain; charset=utf-8"),
    "json": ("transcript.json", "transcript_json", "application/json"),
    "vtt": ("transcript.vtt", "transcript_vtt", "text/vtt; charset=utf-8"),
    "dote": ("transcript.dote.json", "transcript_dote", "application/json"),
    "podlove": ("transcript.podlove.json", "transcript_podlove", "application/json"),
}
SPEAKER_SIDECAR_NAME = "transcript.speakers.json"
SPEAKER_SIDECAR_CONTENT_TYPE = "application/json"
DEFAULT_POLL_SECONDS = 5
DEFAULT_HTTP_TIMEOUT_SECONDS = 60
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 60


class WorkerError(RuntimeError):
    """Raised for worker-side job execution failures."""


class WorkerHttpError(RuntimeError):
    def __init__(self, *, status: int, detail: str) -> None:
        self.status = status
        self.detail = detail
        super().__init__(f"Voxhelm worker API returned HTTP {status}: {detail}")


@dataclass(frozen=True)
class WorkerConfig:
    base_url: str
    worker_id: str
    token: str
    hostname: str
    poll_seconds: int = DEFAULT_POLL_SECONDS
    http_timeout_seconds: int = DEFAULT_HTTP_TIMEOUT_SECONDS
    heartbeat_interval_seconds: int = DEFAULT_HEARTBEAT_INTERVAL_SECONDS


class WorkerClient:
    def __init__(self, *, config: WorkerConfig) -> None:
        self.config = config

    def heartbeat_worker(
        self,
        *,
        capabilities: dict[str, Any],
        running_job_ids: list[str],
    ) -> dict[str, Any]:
        payload = {
            "worker_id": self.config.worker_id,
            "hostname": self.config.hostname,
            "concurrency": 1,
            "running_job_ids": running_job_ids,
            "capabilities": capabilities,
        }
        return self._request_json("POST", "/v1/internal/workers/heartbeat", payload)

    def claim(self, *, capabilities: dict[str, Any]) -> dict[str, Any] | None:
        payload = {
            "worker_id": self.config.worker_id,
            "max_jobs": 1,
            "capabilities": capabilities,
        }
        status, body = self._request("POST", "/v1/internal/work/claim", payload)
        if status == 204:
            return None
        if not isinstance(body, dict) or not isinstance(body.get("job"), dict):
            raise WorkerHttpError(status=status, detail="claim response did not include a job")
        return body["job"]

    def heartbeat_job(
        self,
        *,
        job_id: str,
        lease_token: str,
        progress: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "worker_id": self.config.worker_id,
            "lease_token": lease_token,
            "progress": progress,
        }
        return self._request_json("POST", f"/v1/internal/work/{job_id}/heartbeat", payload)

    def complete_job(
        self,
        *,
        job_id: str,
        lease_token: str,
        result_text: str,
        result_metadata: dict[str, Any],
        artifacts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        payload = {
            "worker_id": self.config.worker_id,
            "lease_token": lease_token,
            "result_text": result_text,
            "result_metadata": result_metadata,
            "artifacts": artifacts,
        }
        return self._request_json("POST", f"/v1/internal/work/{job_id}/complete", payload)

    def fail_job(
        self,
        *,
        job_id: str,
        lease_token: str,
        error_detail: str,
        retryable: bool,
    ) -> dict[str, Any]:
        payload = {
            "worker_id": self.config.worker_id,
            "lease_token": lease_token,
            "retryable": retryable,
            "error_detail": error_detail[:2000],
        }
        return self._request_json("POST", f"/v1/internal/work/{job_id}/fail", payload)

    def _request_json(self, method: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        status, body = self._request(method, path, payload)
        if not isinstance(body, dict):
            raise WorkerHttpError(status=status, detail="response was not a JSON object")
        return body

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any],
    ) -> tuple[int, Any]:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = Request(
            url=f"{self.config.base_url.rstrip('/')}{path}",
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self.config.token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "voxhelm-remote-worker/0.1",
            },
        )
        try:
            with urlopen(request, timeout=self.config.http_timeout_seconds) as response:
                body = response.read()
                return response.status, parse_json_response(body)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise WorkerHttpError(status=exc.code, detail=detail or exc.reason) from exc
        except URLError as exc:
            raise WorkerError(f"Could not reach Voxhelm worker API: {exc.reason}") from exc
        except OSError as exc:
            raise WorkerError(f"Could not reach Voxhelm worker API: {exc}") from exc


class LeaseHeartbeater:
    def __init__(
        self,
        *,
        client: WorkerClient,
        job_id: str,
        lease_token: str,
        interval_seconds: int,
    ) -> None:
        self.client = client
        self.job_id = job_id
        self.lease_token = lease_token
        self.interval_seconds = max(interval_seconds, 1)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._progress: dict[str, Any] = {}
        self._thread = threading.Thread(target=self._run, name=f"voxhelm-lease-{job_id}")

    def __enter__(self) -> LeaseHeartbeater:
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def update(self, *, phase: str, message: str = "") -> None:
        progress = {"phase": phase}
        if message:
            progress["message"] = message
        with self._lock:
            self._progress = progress
        self._beat_once()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._beat_once()

    def _beat_once(self) -> None:
        with self._lock:
            progress = dict(self._progress)
        try:
            self.client.heartbeat_job(
                job_id=self.job_id,
                lease_token=self.lease_token,
                progress=progress,
            )
        except Exception as exc:
            LOGGER.warning("job heartbeat failed job_id=%s detail=%s", self.job_id, exc)


def parse_json_response(body: bytes) -> Any:
    if not body:
        return None
    try:
        return json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise WorkerError("Voxhelm worker API returned invalid JSON.") from exc


def process_claim(
    claim: dict[str, Any],
    *,
    config: WorkerConfig,
    client: WorkerClient,
) -> None:
    job_id = require_string(claim.get("id"), "claim.id")
    lease_token = require_string(claim.get("lease_token"), "claim.lease_token")
    started = monotonic()
    cleanup_paths: list[Path] = []

    with LeaseHeartbeater(
        client=client,
        job_id=job_id,
        lease_token=lease_token,
        interval_seconds=config.heartbeat_interval_seconds,
    ) as heartbeater:
        try:
            heartbeater.update(phase="materialize_input")
            media = materialize_input(claim)
            cleanup_paths.append(media.path)

            store = get_artifact_store_for_identity(claim_artifact_store_identity(claim))
            artifact_prefix = require_string(claim.get("artifact_prefix"), "claim.artifact_prefix")
            artifacts = [
                upload_file_artifact(
                    store=store,
                    artifact_prefix=artifact_prefix,
                    source_path=media.path,
                    name=source_artifact_name(media),
                    kind="source",
                    format_name="source",
                    content_type=media.content_type,
                    exposed=False,
                )
            ]

            audio_path = media.path
            if is_video_path(media.path, content_type=media.content_type):
                heartbeater.update(phase="extract_audio")
                audio_path = extract_audio_from_video(source_path=media.path)
                cleanup_paths.append(audio_path)
                artifacts.append(
                    upload_file_artifact(
                        store=store,
                        artifact_prefix=artifact_prefix,
                        source_path=audio_path,
                        name="extracted-audio.wav",
                        kind="extracted_audio",
                        format_name="wav",
                        content_type="audio/wav",
                        exposed=False,
                    )
                )

            heartbeater.update(phase="transcribe")
            result = transcribe_claim_audio(claim=claim, audio_path=audio_path)

            speakers_artifact_payload: dict[str, Any] | None = None
            diarization = claim_diarization(claim)
            if diarization.get("enabled") is True:
                strategy = str(diarization.get("strategy") or ANONYMOUS_STRATEGY)
                if strategy == KNOWN_SPEAKER_STRATEGY:
                    heartbeater.update(phase="known_speaker_diarization")
                    result, speakers_artifact_payload = run_known_speaker_for_claim(
                        diarization=diarization,
                        audio_path=audio_path,
                        result=result,
                    )
                elif strategy == ANONYMOUS_STRATEGY:
                    heartbeater.update(phase="diarization")
                    result = run_anonymous_diarization_for_claim(
                        diarization=diarization,
                        audio_path=audio_path,
                        result=result,
                    )
                else:
                    raise WorkerError(f"Unsupported diarization strategy '{strategy}'.")

            heartbeater.update(phase="render_artifacts")
            artifacts.extend(
                upload_transcript_artifacts(
                    store=store,
                    artifact_prefix=artifact_prefix,
                    formats=claim_output_formats(claim),
                    result=result,
                )
            )
            if speakers_artifact_payload is not None:
                artifacts.append(
                    upload_bytes_artifact(
                        store=store,
                        artifact_prefix=artifact_prefix,
                        data=json_bytes(speakers_artifact_payload),
                        name=SPEAKER_SIDECAR_NAME,
                        kind="transcript_speakers",
                        format_name="speakers",
                        content_type=SPEAKER_SIDECAR_CONTENT_TYPE,
                        exposed=True,
                    )
                )

            heartbeater.update(phase="complete")
            client.complete_job(
                job_id=job_id,
                lease_token=lease_token,
                result_text=result.text,
                result_metadata=build_result_metadata(
                    claim=claim,
                    media=media,
                    result=result,
                    processing_seconds=round(monotonic() - started, 3),
                    speakers_artifact=speakers_artifact_payload,
                ),
                artifacts=artifacts,
            )
            LOGGER.info("completed remote job job_id=%s worker_id=%s", job_id, config.worker_id)
        finally:
            cleanup_temp_paths(cleanup_paths)


def run_worker_once(
    *,
    config: WorkerConfig,
    client: WorkerClient,
    capabilities: dict[str, Any],
) -> bool:
    heartbeat = client.heartbeat_worker(capabilities=capabilities, running_job_ids=[])
    poll_after = heartbeat.get("poll_after_seconds")
    if isinstance(poll_after, int) and poll_after > 0:
        object.__setattr__(config, "poll_seconds", poll_after)
    claim = client.claim(capabilities=capabilities)
    if claim is None:
        return False

    job_id = require_string(claim.get("id"), "claim.id")
    LOGGER.info("claimed remote job job_id=%s worker_id=%s", job_id, config.worker_id)
    try:
        process_claim(claim, config=config, client=client)
    except Exception as exc:
        LOGGER.exception("remote job failed job_id=%s worker_id=%s", job_id, config.worker_id)
        try:
            client.fail_job(
                job_id=job_id,
                lease_token=require_string(claim.get("lease_token"), "claim.lease_token"),
                error_detail=str(exc),
                retryable=retryable_worker_error(exc),
            )
        except Exception as report_exc:
            LOGGER.error(
                "failed to report remote job failure job_id=%s detail=%s",
                job_id,
                report_exc,
            )
            raise
    return True


def retryable_worker_error(exc: Exception) -> bool:
    if isinstance(exc, WorkerHttpError) and exc.status in {401, 403, 404, 409}:
        return False
    return True


def materialize_input(claim: dict[str, Any]) -> DownloadedMedia:
    raw_input = claim.get("input")
    if not isinstance(raw_input, dict):
        raise WorkerError("Claim did not include an input object.")
    input_kind = raw_input.get("kind")
    if input_kind == "url":
        source_url = require_string(raw_input.get("url"), "claim.input.url")
        return download_allowed_media(source_url=source_url)
    if input_kind == "upload":
        return materialize_staged_upload(raw_input)
    raise WorkerError(f"Unsupported remote input kind '{input_kind}'.")


def materialize_staged_upload(raw_input: dict[str, Any]) -> DownloadedMedia:
    staged = raw_input.get("staged_artifact")
    if not isinstance(staged, dict):
        raise WorkerError("Upload claims must include staged_artifact.")
    storage_key = require_string(
        staged.get("storage_key"),
        "claim.input.staged_artifact.storage_key",
    )
    filename = safe_source_name(str(raw_input.get("filename") or "input"))
    content_type = str(raw_input.get("content_type") or "application/octet-stream")
    suffix = detect_media_suffix(filename, content_type)
    if not suffix:
        raise WorkerError("Staged upload has an unsupported media type.")
    destination = reserve_temp_media_path(suffix=suffix)
    store = get_artifact_store_for_identity(optional_dict(staged.get("storage_identity")))
    try:
        store.download_file(key=storage_key, destination_path=destination)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return DownloadedMedia(
        path=destination,
        content_type=content_type,
        source_name=filename,
        source_kind="upload",
    )


def transcribe_claim_audio(*, claim: dict[str, Any], audio_path: Path) -> TranscriptionResult:
    backend_name = require_string(claim.get("backend"), "claim.backend")
    model_name = require_string(claim.get("model"), "claim.model")
    service = build_backend_service(backend_name=backend_name, model_name=model_name)
    result = service.transcribe(
        audio_path,
        TranscribeParams(
            request_model=model_name,
            prompt=None,
            language=optional_string(claim.get("language")),
        ),
    )
    # Mirror the local transcribe_audio path: sanitize before diarization and
    # artifact rendering so remote_pull workers emit the same clean transcripts.
    return sanitize_result(
        result,
        enabled=settings.VOXHELM_SANITIZE_TRANSCRIPT,
        repeat_threshold=settings.VOXHELM_SANITIZE_REPEAT_THRESHOLD,
    )


def run_anonymous_diarization_for_claim(
    *,
    diarization: dict[str, Any],
    audio_path: Path,
    result: TranscriptionResult,
) -> TranscriptionResult:
    turns = diarize_audio(audio_path, diarization_params(diarization))
    return apply_speaker_labels(result, turns)


def run_known_speaker_for_claim(
    *,
    diarization: dict[str, Any],
    audio_path: Path,
    result: TranscriptionResult,
) -> tuple[TranscriptionResult, dict[str, Any]]:
    config = known_speaker_config_from_claim(diarization)
    references = known_speaker_references_from_claim(diarization)
    backend = get_known_speaker_backend(config.embedding_model)
    job_audio_samples = decode_mono_16k(audio_path)
    raw_turns = collect_anonymous_diarization_turns(diarization=diarization, audio_path=audio_path)
    outcome = run_known_speaker_postprocess(
        result,
        references=references,
        job_audio_samples=job_audio_samples,
        raw_turns=raw_turns,
        config=config,
        backend=backend,
    )
    return outcome.result, build_speakers_artifact(outcome)


def collect_anonymous_diarization_turns(
    *,
    diarization: dict[str, Any],
    audio_path: Path,
) -> list[Any]:
    try:
        return diarize_audio(audio_path, diarization_params(diarization))
    except DiarizationError:
        return []


def known_speaker_references_from_claim(diarization: dict[str, Any]) -> list[ReferenceAudio]:
    known_speakers = diarization.get("known_speakers")
    if not isinstance(known_speakers, list):
        return []
    references: list[ReferenceAudio] = []
    for speaker in known_speakers:
        if not isinstance(speaker, dict):
            continue
        speaker_id = require_string(speaker.get("id"), "diarization.known_speakers[].id")
        name = require_string(speaker.get("name"), "diarization.known_speakers[].name")
        raw_references = speaker.get("references")
        if not isinstance(raw_references, list):
            continue
        for raw_reference in raw_references:
            if not isinstance(raw_reference, dict):
                continue
            windows = load_reference_windows(raw_reference)
            if windows:
                references.append(ReferenceAudio(speaker_id=speaker_id, name=name, windows=windows))
    return references


def load_reference_windows(reference: dict[str, Any]) -> list[Any]:
    audio = reference.get("audio")
    if not isinstance(audio, dict) or audio.get("kind") != "url":
        raise WorkerError("Remote known-speaker references must use URL audio.")
    media = download_allowed_media(
        source_url=require_string(audio.get("url"), "reference.audio.url")
    )
    try:
        samples = decode_mono_16k(media.path)
    finally:
        media.path.unlink(missing_ok=True)
    if reference.get("kind") == "source_range":
        samples = slice_samples(
            samples,
            SAMPLE_RATE,
            float(reference.get("start", 0.0)),
            float(reference.get("end", 0.0)),
        )
    return extract_reference_windows(samples, SAMPLE_RATE)


def known_speaker_config_from_claim(diarization: dict[str, Any]) -> KnownSpeakerConfig:
    raw_config = diarization.get("known_speaker")
    config_data = raw_config if isinstance(raw_config, dict) else {}
    defaults = KnownSpeakerConfig()
    return KnownSpeakerConfig(
        embedding_model=str(config_data.get("embedding_model") or defaults.embedding_model),
        min_segment_duration=float(
            config_data.get("min_segment_duration", defaults.min_segment_duration)
        ),
        auto_accept_margin=float(
            config_data.get("auto_accept_margin", defaults.auto_accept_margin)
        ),
        min_top_similarity=float(
            config_data.get("min_top_similarity", defaults.min_top_similarity)
        ),
    )


def diarization_params(diarization: dict[str, Any]) -> DiarizationParams:
    return DiarizationParams(
        num_speakers=positive_int_or_none(diarization.get("num_speakers")),
        min_speakers=positive_int_or_none(diarization.get("min_speakers")),
        max_speakers=positive_int_or_none(diarization.get("max_speakers")),
    )


def upload_transcript_artifacts(
    *,
    store: Any,
    artifact_prefix: str,
    formats: list[str],
    result: TranscriptionResult,
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for output_format in formats:
        try:
            name, kind, content_type = TRANSCRIPT_ARTIFACTS_BY_FORMAT[output_format]
        except KeyError as exc:
            raise WorkerError(f"Unsupported transcript output format '{output_format}'.") from exc
        artifacts.append(
            upload_bytes_artifact(
                store=store,
                artifact_prefix=artifact_prefix,
                data=render_transcript_artifact(output_format, result),
                name=name,
                kind=kind,
                format_name=output_format,
                content_type=content_type,
                exposed=True,
            )
        )
    return artifacts


def render_transcript_artifact(output_format: str, result: TranscriptionResult) -> bytes:
    if output_format == "text":
        return transcript_formats.render_text(result).encode("utf-8")
    if output_format == "json":
        return json_bytes(transcript_formats.render_verbose_json(result))
    if output_format == "vtt":
        return transcript_formats.render_vtt(result).encode("utf-8")
    if output_format == "dote":
        return json_bytes(transcript_formats.render_dote(result))
    if output_format == "podlove":
        return json_bytes(transcript_formats.render_podlove(result))
    raise WorkerError(f"Unsupported transcript output format '{output_format}'.")


def upload_file_artifact(
    *,
    store: Any,
    artifact_prefix: str,
    source_path: Path,
    name: str,
    kind: str,
    format_name: str,
    content_type: str,
    exposed: bool,
) -> dict[str, Any]:
    key = artifact_key(artifact_prefix, name)
    stored = store.put_file(key=key, source_path=source_path, content_type=content_type)
    return manifest_entry(
        stored=stored,
        name=name,
        kind=kind,
        format_name=format_name,
        content_type=content_type,
        exposed=exposed,
    )


def upload_bytes_artifact(
    *,
    store: Any,
    artifact_prefix: str,
    data: bytes,
    name: str,
    kind: str,
    format_name: str,
    content_type: str,
    exposed: bool,
) -> dict[str, Any]:
    key = artifact_key(artifact_prefix, name)
    stored = store.put_bytes(key=key, data=data, content_type=content_type)
    return manifest_entry(
        stored=stored,
        name=name,
        kind=kind,
        format_name=format_name,
        content_type=content_type,
        exposed=exposed,
    )


def manifest_entry(
    *,
    stored: Any,
    name: str,
    kind: str,
    format_name: str,
    content_type: str,
    exposed: bool,
) -> dict[str, Any]:
    return {
        "name": name,
        "kind": kind,
        "format": format_name,
        "storage_backend": stored.backend,
        "storage_key": stored.key,
        "content_type": content_type,
        "size_bytes": stored.size_bytes,
        "exposed": exposed,
    }


def build_result_metadata(
    *,
    claim: dict[str, Any],
    media: DownloadedMedia,
    result: TranscriptionResult,
    processing_seconds: float,
    speakers_artifact: dict[str, Any] | None,
) -> dict[str, Any]:
    duration_seconds = max((segment.end for segment in result.segments), default=0.0)
    metadata: dict[str, Any] = {
        "backend": result.backend_name or claim.get("backend") or "",
        "requested_model": claim.get("requested_model") or "auto",
        "model": result.model_name or claim.get("model") or "",
        "language": result.language or claim.get("language") or "",
        "duration_seconds": duration_seconds,
        "processing_seconds": processing_seconds,
        "source_name": media.source_name,
        "source_content_type": media.content_type,
    }
    if speakers_artifact is not None:
        metadata["diarization"] = {
            "known_speaker_summary": speakers_artifact.get("summary", {}),
        }
    return metadata


def build_capabilities() -> dict[str, Any]:
    backends = {settings.VOXHELM_STT_BACKEND}
    fallback_backend = settings.VOXHELM_STT_FALLBACK_BACKEND.strip()
    if fallback_backend:
        backends.add(fallback_backend)
    if settings.VOXHELM_WHISPERKIT_ENABLED:
        backends.add("whisperkit")

    models = set()
    for backend in sorted(backends):
        try:
            models.add(resolve_model_name_for_backend(request_model="auto", backend_name=backend))
        except RuntimeError:
            continue
    models.update(
        model
        for model in (
            settings.VOXHELM_WHISPERCPP_MODEL,
            settings.VOXHELM_MLX_MODEL,
            settings.VOXHELM_WHISPERKIT_MODEL if settings.VOXHELM_WHISPERKIT_ENABLED else "",
        )
        if model
    )

    diarization_available = diarization_runtime_available()
    output_formats = ["text", "json", "vtt", "dote", "podlove"]
    embedding_models: list[str] = []
    if diarization_available:
        output_formats.append("speakers")
        embedding_models.append(KnownSpeakerConfig().embedding_model)

    return {
        "job_types": ["transcribe"],
        "backends": sorted(backends),
        "models": sorted(models),
        "output_formats": output_formats,
        "diarization": {
            "anonymous": diarization_available,
            "known_speaker": diarization_available,
            "embedding_models": embedding_models,
        },
    }


def diarization_runtime_available() -> bool:
    backend = settings.VOXHELM_DIARIZATION_BACKEND.strip().lower()
    if backend in {"", "none"}:
        return False
    if backend != "pyannote":
        LOGGER.warning("unsupported diarization backend for remote worker backend=%s", backend)
        return False
    if not settings.VOXHELM_HUGGINGFACE_TOKEN:
        LOGGER.warning("diarization disabled for worker capabilities: missing HF token")
        return False
    missing = [
        module_name
        for module_name in ("pyannote.audio", "torch", "numpy")
        if not module_available(module_name)
    ]
    if missing:
        LOGGER.warning(
            "diarization disabled for worker capabilities: missing modules=%s",
            ",".join(missing),
        )
        return False
    return True


def module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def claim_artifact_store_identity(claim: dict[str, Any]) -> dict[str, Any] | None:
    value = claim.get("artifact_store") or claim.get("artifact_store_identity")
    return optional_dict(value)


def claim_output_formats(claim: dict[str, Any]) -> list[str]:
    output = claim.get("output")
    if not isinstance(output, dict):
        return ["text", "json"]
    formats = output.get("formats")
    if not isinstance(formats, list):
        return ["text", "json"]
    return [str(output_format) for output_format in formats]


def claim_diarization(claim: dict[str, Any]) -> dict[str, Any]:
    output = claim.get("output")
    if not isinstance(output, dict):
        return {"enabled": False}
    diarization = output.get("diarization")
    return diarization if isinstance(diarization, dict) else {"enabled": False}


def source_artifact_name(media: DownloadedMedia) -> str:
    suffix = media.path.suffix.lower() or detect_media_suffix(media.source_name, media.content_type)
    if not suffix:
        suffix = ".bin"
    return f"source{suffix}"


def safe_source_name(value: str) -> str:
    return Path(value).name or "input"


def artifact_key(artifact_prefix: str, name: str) -> str:
    return f"{artifact_prefix.rstrip('/')}/{name}"


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def cleanup_temp_paths(paths: Sequence[Path]) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            LOGGER.warning("could not remove temporary file path=%s", path)


def positive_int_or_none(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkerError(f"{field_name} must be a non-empty string.")
    return value.strip()


def optional_string(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def optional_dict(value: object) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def load_env_file(path: Path) -> None:
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            tokens = shlex.split(line, comments=True, posix=True)
        except ValueError as exc:
            raise SystemExit(f"{path}:{line_number}: could not parse env line: {exc}") from exc
        if not tokens:
            continue
        if tokens[0] == "export":
            tokens = tokens[1:]
        if len(tokens) != 1 or "=" not in tokens[0]:
            raise SystemExit(f"{path}:{line_number}: expected KEY=VALUE")
        key, value = tokens[0].split("=", 1)
        if not key:
            raise SystemExit(f"{path}:{line_number}: expected non-empty env key")
        os.environ[key] = value


def setup_django_for_worker() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    # The worker loads Django settings to reuse the local STT backends; it serves no
    # HTTP sessions and signs nothing, so a fixed local key satisfies the settings
    # check when the worker env file does not set one.
    os.environ.setdefault("DJANGO_SECRET_KEY", "voxhelm-remote-worker-local-settings")
    # This process is a worker client, not the Voxhelm control plane. Force the
    # server-only remote-pull startup checks out of the worker runtime.
    os.environ["VOXHELM_TRANSCRIPTION_EXECUTION_MODE"] = "django_tasks"
    import django

    django.setup()


def parse_worker_config(args: argparse.Namespace) -> WorkerConfig:
    worker_id = args.worker_id or os.getenv("VOXHELM_WORKER_ID", "")
    token = args.token or os.getenv("VOXHELM_WORKER_TOKEN", "")
    if not token and worker_id:
        token = worker_token_from_map(worker_id)
    base_url = (
        args.base_url
        or os.getenv("VOXHELM_WORKER_BASE_URL", "")
        or os.getenv("VOXHELM_BASE_URL", "")
    )

    missing = [
        name
        for name, value in (
            ("base URL", base_url),
            ("worker id", worker_id),
            ("worker token", token),
        )
        if not value
    ]
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(f"Missing required worker configuration: {joined}.")

    return WorkerConfig(
        base_url=base_url,
        worker_id=worker_id,
        token=token,
        hostname=args.hostname or platform.node() or "unknown",
        poll_seconds=args.poll_seconds,
        http_timeout_seconds=args.http_timeout_seconds,
        heartbeat_interval_seconds=args.heartbeat_interval_seconds,
    )


def worker_token_from_map(worker_id: str) -> str:
    token = settings.VOXHELM_WORKER_TOKENS.get(worker_id, "")
    return token.strip()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voxhelm-remote-worker")
    parser.add_argument("--env-file", type=Path, help="Load worker environment from KEY=VALUE file")
    parser.add_argument("--base-url", help="Voxhelm base URL, e.g. http://studio.local:8000")
    parser.add_argument("--worker-id", help="Worker id configured in VOXHELM_WORKER_TOKENS")
    parser.add_argument("--token", help="Worker bearer token")
    parser.add_argument("--hostname", help="Hostname reported in worker heartbeats")
    parser.add_argument("--once", action="store_true", help="Claim at most one job and exit")
    parser.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--http-timeout-seconds", type=int, default=DEFAULT_HTTP_TIMEOUT_SECONDS)
    parser.add_argument(
        "--heartbeat-interval-seconds",
        type=int,
        default=DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser


def worker_version() -> str:
    try:
        return importlib.metadata.version("voxhelm")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def log_startup(config: WorkerConfig) -> None:
    LOGGER.info(
        "voxhelm-remote-worker starting version=%s worker_id=%s hostname=%s base_url=%s",
        worker_version(),
        config.worker_id,
        config.hostname,
        config.base_url,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.env_file is not None:
        load_env_file(args.env_file)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    setup_django_for_worker()
    config = parse_worker_config(args)
    log_startup(config)
    capabilities = build_capabilities()
    client = WorkerClient(config=config)

    while True:
        try:
            claimed = run_worker_once(config=config, client=client, capabilities=capabilities)
        except KeyboardInterrupt:
            return 130
        except Exception:
            LOGGER.exception("worker poll failed worker_id=%s", config.worker_id)
            if args.once:
                return 1
            sleep(config.poll_seconds)
            continue
        if args.once:
            return 0
        if not claimed:
            sleep(config.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
