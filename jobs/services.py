from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any
from urllib.parse import urlparse

from django.conf import settings
from django.db import transaction
from django.utils import timezone
from django_tasks import default_task_backend
from django_tasks.base import TaskResultStatus
from django_tasks.exceptions import TaskResultDoesNotExist

from jobs.artifacts import get_artifact_store
from jobs.media import (
    DownloadedMedia,
    download_allowed_media,
    extract_audio_from_video,
    is_video_path,
)
from jobs.models import Job, JobArtifact
from synthesis.service import (
    AUDIO_OUTPUT_FORMATS,
    MAX_TTS_SPEED,
    MIN_TTS_SPEED,
    ExportedAudio,
    SynthesisResult,
    SynthesizeParams,
    cleanup_paths,
    export_audio,
    synthesize_text,
)
from transcriptions.service import (
    TranscribeParams,
    TranscriptionResult,
    render_verbose_json,
    render_vtt,
    transcribe_audio,
)
from transcriptions.views import ApiError

PRIORITY_TO_TASK_PRIORITY = {
    Job.Priority.LOW: -10,
    Job.Priority.NORMAL: 0,
    Job.Priority.HIGH: 10,
}
TRANSCRIPTION_OUTPUT_FORMATS = {"json", "text", "vtt", "webvtt"}
DEFAULT_TRANSCRIPTION_OUTPUT_FORMATS = ("text", "json")
DEFAULT_SPEECH_OUTPUT_FORMATS = ("wav",)


@dataclass(frozen=True)
class JobRequest:
    job_type: str
    priority: str
    lane: str
    backend: str
    model: str
    language: str | None
    input_data: dict[str, Any]
    output_formats: list[str]
    context: dict[str, Any]
    task_ref: str


def create_job_from_payload(*, producer: str, payload: dict[str, Any]) -> tuple[Job, bool]:
    request = parse_job_request(payload)
    from jobs.tasks import run_synthesis_job, run_transcription_job

    if request.task_ref:
        existing = (
            Job.objects.filter(producer=producer, task_ref=request.task_ref)
            .exclude(state=Job.State.FAILED)
            .order_by("-created_at")
            .first()
        )
        if existing is not None:
            reconcile_job_state(existing)
            return existing, False

    task_callable = (
        run_synthesis_job if request.job_type == Job.JobType.SYNTHESIZE else run_transcription_job
    )

    with transaction.atomic():
        job = Job.objects.create(
            producer=producer,
            task_ref=request.task_ref,
            job_type=request.job_type,
            lane=request.lane,
            priority=request.priority,
            backend=request.backend,
            model=request.model,
            language=request.language or "",
            input_data=request.input_data,
            output_data={"formats": request.output_formats},
            context_data=request.context,
            state=Job.State.QUEUED,
        )
        task_result = task_callable.using(
            priority=PRIORITY_TO_TASK_PRIORITY[Job.Priority(request.priority)],
            queue_name=settings.VOXHELM_TASK_QUEUE,
        ).enqueue(str(job.id))
        job.django_task_id = str(task_result.id)
        job.save(update_fields=["django_task_id", "updated_at"])

    job.refresh_from_db()
    reconcile_job_state(job)
    return job, True


def parse_job_request(payload: dict[str, Any]) -> JobRequest:
    job_type = ensure_choice(payload.get("job_type"), Job.JobType.values, "job_type")
    if job_type == Job.JobType.TRANSCRIBE:
        return parse_transcription_job_request(payload)
    if job_type == Job.JobType.SYNTHESIZE:
        return parse_synthesis_job_request(payload)
    raise ApiError(f"Unsupported job_type '{job_type}'.")


def parse_transcription_job_request(payload: dict[str, Any]) -> JobRequest:
    priority, lane, backend = parse_common_job_fields(payload)
    model = ensure_transcription_model(payload.get("model", "auto"))
    language = optional_string(payload.get("language"))

    input_data = ensure_object(payload.get("input"), "input")
    input_kind = optional_string(input_data.get("kind"))
    if input_kind != "url":
        raise ApiError("M1b currently supports only input.kind=url for transcribe jobs.")
    source_url = optional_string(input_data.get("url"))
    if not source_url:
        raise ApiError("Batch transcription requires input.url.")
    parsed = urlparse(source_url)
    if not parsed.scheme or not parsed.netloc:
        raise ApiError("input.url must be an absolute URL.")

    output_formats = validate_transcription_output_formats(
        ensure_object(payload.get("output", {}), "output")
    )
    context = ensure_object(payload.get("context", {}), "context")
    task_ref = optional_string(payload.get("task_ref")) or ""

    return JobRequest(
        job_type=Job.JobType.TRANSCRIBE,
        priority=priority,
        lane=lane,
        backend=backend,
        model=model,
        language=language,
        input_data={"kind": "url", "url": source_url},
        output_formats=output_formats,
        context=context,
        task_ref=task_ref,
    )


def parse_synthesis_job_request(payload: dict[str, Any]) -> JobRequest:
    priority, lane, backend = parse_common_job_fields(payload)
    model = ensure_synthesis_model(payload.get("model", "auto"))
    language = optional_string(payload.get("language"))
    voice = optional_string(payload.get("voice"))
    speed = validate_speed(payload.get("speed"))

    input_data = ensure_object(payload.get("input"), "input")
    input_kind = optional_string(input_data.get("kind"))
    if input_kind != "text":
        raise ApiError("Batch synthesis currently supports only input.kind=text.")
    text = optional_string(input_data.get("text"))
    if not text:
        raise ApiError("Batch synthesis requires input.text.")
    if len(text) > settings.VOXHELM_TTS_MAX_INPUT_CHARS:
        raise ApiError(
            "input.text exceeded the configured "
            f"{settings.VOXHELM_TTS_MAX_INPUT_CHARS} character limit."
        )

    output = ensure_object(payload.get("output", {}), "output")
    output_formats = validate_speech_output_formats(output)
    context = ensure_object(payload.get("context", {}), "context")
    task_ref = optional_string(payload.get("task_ref")) or ""

    synthesis_input = {
        "kind": "text",
        "text": text,
        "speed": speed,
    }
    if voice:
        synthesis_input["voice"] = voice

    return JobRequest(
        job_type=Job.JobType.SYNTHESIZE,
        priority=priority,
        lane=lane,
        backend=backend,
        model=model,
        language=language,
        input_data=synthesis_input,
        output_formats=output_formats,
        context=context,
        task_ref=task_ref,
    )


def parse_common_job_fields(payload: dict[str, Any]) -> tuple[str, str, str]:
    priority = ensure_choice(
        payload.get("priority", Job.Priority.NORMAL),
        Job.Priority.values,
        "priority",
    )
    lane = ensure_choice(payload.get("lane", Job.Lane.BATCH), Job.Lane.values, "lane")
    if lane != Job.Lane.BATCH:
        raise ApiError("Only the batch lane is supported in this slice.")

    backend = optional_string(payload.get("backend")) or "auto"
    if backend != "auto":
        raise ApiError("Only backend=auto is supported in this slice.")

    return priority, lane, backend


def validate_transcription_output_formats(output: dict[str, Any]) -> list[str]:
    raw_formats = output.get("formats", list(DEFAULT_TRANSCRIPTION_OUTPUT_FORMATS))
    if not isinstance(raw_formats, list) or not raw_formats:
        raise ApiError("output.formats must be a non-empty list.")
    output_formats: list[str] = []
    for raw in raw_formats:
        if not isinstance(raw, str):
            raise ApiError("output.formats entries must be strings.")
        normalized = raw.strip().lower()
        if normalized not in TRANSCRIPTION_OUTPUT_FORMATS:
            raise ApiError("Unsupported output format. Use text, json, or vtt.")
        if normalized == "webvtt":
            normalized = "vtt"
        if normalized not in output_formats:
            output_formats.append(normalized)
    return output_formats


def validate_speech_output_formats(output: dict[str, Any]) -> list[str]:
    raw_formats = output.get("formats", list(DEFAULT_SPEECH_OUTPUT_FORMATS))
    if not isinstance(raw_formats, list) or not raw_formats:
        raise ApiError("output.formats must be a non-empty list.")
    output_formats: list[str] = []
    for raw in raw_formats:
        if not isinstance(raw, str):
            raise ApiError("output.formats entries must be strings.")
        normalized = raw.strip().lower()
        if normalized not in AUDIO_OUTPUT_FORMATS:
            accepted = ", ".join(sorted(AUDIO_OUTPUT_FORMATS))
            raise ApiError(f"Unsupported output format. Use one of: {accepted}.")
        if normalized not in output_formats:
            output_formats.append(normalized)
    return output_formats


def ensure_choice(value: object, allowed: list[str], field_name: str) -> str:
    if not isinstance(value, str):
        raise ApiError(f"{field_name} must be a string.")
    normalized = value.strip()
    if normalized not in allowed:
        accepted = ", ".join(sorted(allowed))
        raise ApiError(f"Unsupported {field_name} '{normalized}'. Accepted values: {accepted}.")
    return normalized


def ensure_transcription_model(value: object) -> str:
    if not isinstance(value, str):
        raise ApiError("model must be a string.")
    normalized = value.strip()
    if normalized not in settings.VOXHELM_BATCH_ACCEPTED_MODELS:
        accepted = ", ".join(sorted(settings.VOXHELM_BATCH_ACCEPTED_MODELS))
        raise ApiError(f"Unsupported model '{normalized}'. Accepted values: {accepted}.")
    return normalized


def ensure_synthesis_model(value: object) -> str:
    if not isinstance(value, str):
        raise ApiError("model must be a string.")
    normalized = value.strip()
    if normalized not in settings.VOXHELM_ACCEPTED_SPEECH_MODELS:
        accepted = ", ".join(sorted(settings.VOXHELM_ACCEPTED_SPEECH_MODELS))
        raise ApiError(f"Unsupported model '{normalized}'. Accepted values: {accepted}.")
    return normalized


def validate_speed(value: object) -> float:
    if value is None:
        return 1.0
    if not isinstance(value, (int, float)):
        raise ApiError("speed must be a number.")
    normalized = float(value)
    if not MIN_TTS_SPEED <= normalized <= MAX_TTS_SPEED:
        raise ApiError(f"speed must be between {MIN_TTS_SPEED} and {MAX_TTS_SPEED}.")
    return normalized


def optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ApiError("Optional request fields must be strings when provided.")
    normalized = value.strip()
    return normalized or None


def ensure_object(value: object, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ApiError(f"{field_name} must be an object.")
    return value


def execute_transcription_job(*, job_id: str, task_result_id: str) -> dict[str, Any]:
    job = initialize_running_job(job_id=job_id, task_result_id=task_result_id)
    media: DownloadedMedia | None = None
    extracted_audio_path: Path | None = None
    started = monotonic()
    try:
        media = download_allowed_media(source_url=str(job.input_data["url"]))
        store_job_input_artifact(job=job, media=media)

        audio_path = media.path
        if is_video_path(media.path, content_type=media.content_type):
            extracted_audio_path = extract_audio_from_video(source_path=media.path)
            store_extracted_audio_artifact(job=job, audio_path=extracted_audio_path)
            audio_path = extracted_audio_path

        result = transcribe_audio(
            audio_path,
            TranscribeParams(
                request_model=job.model or "auto",
                prompt=None,
                language=job.language or None,
            ),
        )
        processing_seconds = round(monotonic() - started, 3)
        metadata = build_transcription_result_metadata(
            job=job,
            media=media,
            result=result,
            processing_seconds=processing_seconds,
        )
        persist_transcription_output_artifacts(job=job, result=result)
        mark_job_succeeded(job=job, metadata=metadata, result_text=result.text)
        return {"job_id": str(job.id), "text": result.text, "metadata": metadata}
    except Exception as exc:
        mark_job_failed(job=job, exc=exc)
        raise
    finally:
        if media is not None:
            media.path.unlink(missing_ok=True)
        if extracted_audio_path is not None:
            extracted_audio_path.unlink(missing_ok=True)


def execute_synthesis_job(*, job_id: str, task_result_id: str) -> dict[str, Any]:
    job = initialize_running_job(job_id=job_id, task_result_id=task_result_id)
    result = None
    exports: list[ExportedAudio] = []
    started = monotonic()
    try:
        result = synthesize_text(
            str(job.input_data["text"]),
            SynthesizeParams(
                request_model=job.model or "auto",
                voice=optional_string(job.input_data.get("voice")),
                language=job.language or None,
                speed=float(job.input_data.get("speed") or 1.0),
            ),
        )
        for output_format in job.output_data.get("formats", list(DEFAULT_SPEECH_OUTPUT_FORMATS)):
            exports.append(export_audio(result, output_format=output_format))

        metadata = build_synthesis_result_metadata(
            job=job,
            result=result,
            processing_seconds=round(monotonic() - started, 3),
        )
        persist_synthesis_output_artifacts(job=job, exports=exports)
        mark_job_succeeded(job=job, metadata=metadata, result_text="")
        return {"job_id": str(job.id), "metadata": metadata}
    except Exception as exc:
        mark_job_failed(job=job, exc=exc)
        raise
    finally:
        cleanup_targets = [
            export.path
            for export in exports
            if result is None or export.path != result.audio_path
        ]
        if result is not None:
            cleanup_targets.append(result.audio_path)
        cleanup_paths(*cleanup_targets)


def initialize_running_job(*, job_id: str, task_result_id: str) -> Job:
    job = Job.objects.get(id=job_id)
    job.state = Job.State.RUNNING
    job.started_at = timezone.now()
    if not job.django_task_id:
        job.django_task_id = task_result_id
    job.error_detail = ""
    job.save(
        update_fields=[
            "state",
            "started_at",
            "django_task_id",
            "error_detail",
            "updated_at",
        ]
    )
    return job


def mark_job_succeeded(*, job: Job, metadata: dict[str, Any], result_text: str) -> None:
    job.state = Job.State.SUCCEEDED
    job.result_text = result_text
    job.result_metadata = metadata
    job.finished_at = timezone.now()
    job.save(
        update_fields=[
            "state",
            "result_text",
            "result_metadata",
            "finished_at",
            "updated_at",
        ]
    )


def mark_job_failed(*, job: Job, exc: Exception) -> None:
    job.state = Job.State.FAILED
    job.error_detail = str(exc)
    job.finished_at = timezone.now()
    job.save(update_fields=["state", "error_detail", "finished_at", "updated_at"])


def build_transcription_result_metadata(
    *,
    job: Job,
    media: DownloadedMedia,
    result: TranscriptionResult,
    processing_seconds: float,
) -> dict[str, Any]:
    duration_seconds = 0.0
    if result.segments:
        duration_seconds = max(segment.end for segment in result.segments)
    return {
        "backend": result.backend_name or settings.VOXHELM_STT_BACKEND,
        "requested_model": job.model or "auto",
        "model": result.model_name or job.model or "auto",
        "language": result.language or job.language or "",
        "duration_seconds": duration_seconds,
        "processing_seconds": processing_seconds,
        "source_url": media.source_url,
        "source_content_type": media.content_type,
    }


def build_synthesis_result_metadata(
    *,
    job: Job,
    result: SynthesisResult,
    processing_seconds: float,
) -> dict[str, Any]:
    return {
        "backend": result.backend_name or settings.VOXHELM_TTS_BACKEND,
        "requested_model": job.model or "auto",
        "model": result.model_name or job.model or "auto",
        "voice": result.voice_name,
        "language": result.language or job.language or "",
        "duration_seconds": result.duration_seconds,
        "processing_seconds": processing_seconds,
    }


def persist_transcription_output_artifacts(*, job: Job, result: TranscriptionResult) -> None:
    requested_formats = set(
        job.output_data.get("formats", list(DEFAULT_TRANSCRIPTION_OUTPUT_FORMATS))
    )
    if "text" in requested_formats:
        create_or_replace_artifact(
            job=job,
            name="transcript.txt",
            kind=JobArtifact.Kind.TRANSCRIPT_TEXT,
            format_name="text",
            content_type="text/plain; charset=utf-8",
            payload=result.text.encode("utf-8"),
            exposed=True,
        )
    if "json" in requested_formats:
        json_payload = json.dumps(render_verbose_json(result), ensure_ascii=True, indent=2)
        create_or_replace_artifact(
            job=job,
            name="transcript.json",
            kind=JobArtifact.Kind.TRANSCRIPT_JSON,
            format_name="json",
            content_type="application/json",
            payload=json_payload.encode("utf-8"),
            exposed=True,
        )
    if "vtt" in requested_formats:
        create_or_replace_artifact(
            job=job,
            name="transcript.vtt",
            kind=JobArtifact.Kind.TRANSCRIPT_VTT,
            format_name="vtt",
            content_type="text/vtt; charset=utf-8",
            payload=render_vtt(result).encode("utf-8"),
            exposed=True,
        )


def persist_synthesis_output_artifacts(*, job: Job, exports: list[ExportedAudio]) -> None:
    kind_map = {
        "wav": JobArtifact.Kind.SPEECH_WAV,
        "mp3": JobArtifact.Kind.SPEECH_MP3,
        "ogg": JobArtifact.Kind.SPEECH_OGG,
    }
    for exported in exports:
        create_or_replace_artifact_from_file(
            job=job,
            name=f"speech.{exported.format_name}",
            kind=kind_map[exported.format_name],
            format_name=exported.format_name,
            content_type=exported.content_type,
            source_path=exported.path,
            exposed=True,
        )


def store_job_input_artifact(*, job: Job, media: DownloadedMedia) -> None:
    source_name = Path(urlparse(media.source_url).path or "input").name or "input"
    create_or_replace_artifact_from_file(
        job=job,
        name=source_name,
        kind=JobArtifact.Kind.SOURCE,
        format_name="source",
        content_type=media.content_type or "application/octet-stream",
        source_path=media.path,
        exposed=False,
    )


def store_extracted_audio_artifact(*, job: Job, audio_path: Path) -> None:
    create_or_replace_artifact_from_file(
        job=job,
        name="extracted-audio.wav",
        kind=JobArtifact.Kind.EXTRACTED_AUDIO,
        format_name="wav",
        content_type="audio/wav",
        source_path=audio_path,
        exposed=False,
    )


def create_or_replace_artifact(
    *,
    job: Job,
    name: str,
    kind: str,
    format_name: str,
    content_type: str,
    payload: bytes,
    exposed: bool,
) -> JobArtifact:
    store = get_artifact_store()
    key = build_artifact_key(job=job, name=name)
    stored = store.put_bytes(key=key, data=payload, content_type=content_type)
    artifact, _created = JobArtifact.objects.update_or_create(
        job=job,
        name=name,
        defaults={
            "kind": kind,
            "format": format_name,
            "storage_backend": stored.backend,
            "storage_key": stored.key,
            "content_type": content_type,
            "size_bytes": stored.size_bytes,
            "exposed": exposed,
        },
    )
    return artifact


def create_or_replace_artifact_from_file(
    *,
    job: Job,
    name: str,
    kind: str,
    format_name: str,
    content_type: str,
    source_path: Path,
    exposed: bool,
) -> JobArtifact:
    store = get_artifact_store()
    key = build_artifact_key(job=job, name=name)
    stored = store.put_file(key=key, source_path=source_path, content_type=content_type)
    artifact, _created = JobArtifact.objects.update_or_create(
        job=job,
        name=name,
        defaults={
            "kind": kind,
            "format": format_name,
            "storage_backend": stored.backend,
            "storage_key": stored.key,
            "content_type": content_type,
            "size_bytes": stored.size_bytes,
            "exposed": exposed,
        },
    )
    return artifact


def build_artifact_key(*, job: Job, name: str) -> str:
    safe_name = name.replace("/", "-")
    prefix = settings.VOXHELM_ARTIFACT_PREFIX.strip("/")
    return f"{prefix}/jobs/{job.id}/{safe_name}"


def artifact_proxy_path(*, job: Job, artifact: JobArtifact) -> str:
    return f"/v1/jobs/{job.id}/artifacts/{artifact.name}"


def serialize_job(job: Job) -> dict[str, Any]:
    reconcile_job_state(job)
    payload: dict[str, Any] = {
        "id": str(job.id),
        "state": job.state,
        "job_type": job.job_type,
        "created_at": job.created_at.isoformat().replace("+00:00", "Z"),
        "started_at": isoformat_or_none(job.started_at),
        "finished_at": isoformat_or_none(job.finished_at),
        "task_ref": job.task_ref or None,
    }
    if job.state == Job.State.SUCCEEDED:
        artifacts = {
            artifact.format: artifact_proxy_path(job=job, artifact=artifact)
            for artifact in job.artifacts.filter(exposed=True)
        }
        payload["result"] = {
            "artifacts": artifacts,
            "metadata": job.result_metadata,
        }
        if job.result_text:
            payload["result"]["text"] = job.result_text
    elif job.state == Job.State.FAILED:
        payload["error"] = {"message": job.error_detail}
    return payload


def isoformat_or_none(value) -> str | None:
    if value is None:
        return None
    return value.isoformat().replace("+00:00", "Z")


def reconcile_job_state(job: Job) -> Job:
    if not job.django_task_id or not default_task_backend.supports_get_result:
        return job
    try:
        task_result = default_task_backend.get_result(job.django_task_id)
    except TaskResultDoesNotExist:
        return job

    mapped_state = map_task_status(task_result.status)
    updates: list[str] = []
    if mapped_state and mapped_state != job.state:
        job.state = mapped_state
        updates.append("state")
    if task_result.started_at and job.started_at is None:
        job.started_at = task_result.started_at
        updates.append("started_at")
    if task_result.finished_at and job.finished_at is None:
        job.finished_at = task_result.finished_at
        updates.append("finished_at")
    if mapped_state == Job.State.FAILED and not job.error_detail and task_result.errors:
        job.error_detail = task_result.errors[0].exception_class.__name__
        updates.append("error_detail")
    if updates:
        updates.append("updated_at")
        job.save(update_fields=updates)
    return job


def map_task_status(status: TaskResultStatus) -> str | None:
    if status == TaskResultStatus.READY:
        return Job.State.QUEUED
    if status == TaskResultStatus.RUNNING:
        return Job.State.RUNNING
    if status == TaskResultStatus.SUCCESSFUL:
        return Job.State.SUCCEEDED
    if status == TaskResultStatus.FAILED:
        return Job.State.FAILED
    return None
