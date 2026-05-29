from __future__ import annotations

import json
import uuid
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

from config.settings import get_batch_accepted_stt_models
from jobs.artifacts import get_artifact_store
from jobs.media import (
    DownloadedMedia,
    download_allowed_media,
    extract_audio_from_video,
    is_video_path,
)
from jobs.models import Job, JobArtifact, StagedMedia
from jobs.staging import (
    claim_staged_media_for_job,
    cleanup_expired_staged_media,
    delete_staged_media,
    get_staged_media_for_submission,
    materialize_staged_media,
)
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
from transcriptions.diarization import (
    DiarizationError,
    DiarizationParams,
    SpeakerTurn,
    apply_speaker_labels,
    diarize_audio,
)
from transcriptions.errors import ApiError
from transcriptions.formats import render_dote, render_podlove, render_text
from transcriptions.known_speaker import (
    ANONYMOUS_STRATEGY,
    DIARIZATION_STRATEGIES,
    KNOWN_SPEAKER_STRATEGY,
    KnownSpeakerConfig,
    ReferenceAudio,
    build_speakers_artifact,
    decode_mono_16k,
    extract_reference_windows,
    get_known_speaker_backend,
    run_known_speaker_postprocess,
    slice_samples,
)
from transcriptions.service import (
    TranscribeParams,
    TranscriptionResult,
    render_verbose_json,
    render_vtt,
    transcribe_audio,
)

PRIORITY_TO_TASK_PRIORITY = {
    Job.Priority.LOW: -10,
    Job.Priority.NORMAL: 0,
    Job.Priority.HIGH: 10,
}
TRANSCRIPTION_OUTPUT_FORMATS = {"json", "text", "vtt", "webvtt", "dote", "podlove"}
DEFAULT_TRANSCRIPTION_OUTPUT_FORMATS = ("text", "json")
OPERATOR_TRANSCRIPTION_OUTPUT_FORMATS = ("text", "json", "vtt", "dote", "podlove")
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
    diarization: dict[str, Any]
    context: dict[str, Any]
    task_ref: str


def create_job_from_payload(*, producer: str, payload: dict[str, Any]) -> tuple[Job, bool]:
    return create_job_from_payload_for_actor(producer=producer, payload=payload, operator=None)


def create_job_from_payload_for_actor(
    *,
    producer: str,
    payload: dict[str, Any],
    operator,
) -> tuple[Job, bool]:
    request = parse_job_request(payload)
    from jobs.tasks import run_synthesis_job, run_transcription_job

    if request.task_ref:
        existing_jobs = (
            Job.objects.filter(producer=producer, task_ref=request.task_ref)
            .exclude(state=Job.State.FAILED)
            .order_by("-created_at")
        )
        for existing in existing_jobs.iterator():
            if existing_job_matches_request(existing, request):
                reconcile_job_state(existing)
                return existing, False

    task_callable = (
        run_synthesis_job if request.job_type == Job.JobType.SYNTHESIZE else run_transcription_job
    )
    if request.job_type == Job.JobType.TRANSCRIBE and request.input_data.get("kind") == "upload":
        cleanup_expired_staged_media(exclude_upload_id=str(request.input_data["upload_id"]))

    with transaction.atomic():
        staged_media: StagedMedia | None = None
        input_data = request.input_data
        if request.job_type == Job.JobType.TRANSCRIBE and input_data.get("kind") == "upload":
            staged_media = get_staged_media_for_submission(
                producer=producer,
                upload_id=str(input_data["upload_id"]),
            )
            input_data = {
                "kind": "upload",
                "upload_id": str(staged_media.id),
                "filename": staged_media.original_filename,
                "content_type": staged_media.content_type,
                "size_bytes": staged_media.size_bytes,
            }
        output_data: dict[str, Any] = {"formats": request.output_formats}
        if request.job_type == Job.JobType.TRANSCRIBE:
            output_data["diarization"] = request.diarization

        job = Job.objects.create(
            producer=producer,
            operator=operator,
            task_ref=request.task_ref,
            job_type=request.job_type,
            lane=request.lane,
            dispatch_mode=Job.DispatchMode.BATCH,
            priority=request.priority,
            backend=request.backend,
            model=request.model,
            language=request.language or "",
            input_data=input_data,
            output_data=output_data,
            context_data=request.context,
            state=Job.State.QUEUED,
        )
        if staged_media is not None:
            claim_staged_media_for_job(staged=staged_media, job=job)
        task_result = task_callable.using(
            priority=PRIORITY_TO_TASK_PRIORITY[Job.Priority(request.priority)],
            queue_name=settings.VOXHELM_TASK_QUEUE,
        ).enqueue(str(job.id))
        job.django_task_id = str(task_result.id)
        job.save(update_fields=["django_task_id", "updated_at"])

    job.refresh_from_db()
    reconcile_job_state(job)
    return job, True


def create_operator_sync_transcription(
    *,
    operator,
    source_path: Path,
    source_name: str,
    input_data: dict[str, Any],
    language: str | None,
    prompt: str | None = None,
    request_model: str = "gpt-4o-mini-transcribe",
) -> Job:
    job = Job.objects.create(
        producer=settings.VOXHELM_OPERATOR_PRODUCER_LABEL,
        operator=operator,
        task_ref="",
        job_type=Job.JobType.TRANSCRIBE,
        lane=Job.Lane.BATCH,
        dispatch_mode=Job.DispatchMode.SYNC,
        priority=Job.Priority.NORMAL,
        backend="auto",
        model=request_model,
        language=language or "",
        input_data=input_data,
        output_data={"formats": list(OPERATOR_TRANSCRIPTION_OUTPUT_FORMATS)},
        context_data={"source_name": source_name},
        state=Job.State.RUNNING,
        started_at=timezone.now(),
    )
    started = monotonic()
    try:
        result = transcribe_audio(
            source_path,
            TranscribeParams(
                request_model=request_model,
                prompt=prompt,
                language=language,
            ),
        )
        metadata = build_transcription_metadata(
            requested_model=request_model,
            requested_language=language,
            result=result,
            processing_seconds=round(monotonic() - started, 3),
            source_kind=str(input_data.get("kind") or ""),
            source_name=source_name,
            source_url=str(input_data.get("url") or ""),
            source_content_type=str(input_data.get("content_type") or ""),
        )
        persist_transcription_output_artifacts(job=job, result=result)
        mark_job_succeeded(job=job, metadata=metadata, result_text=result.text)
    except Exception as exc:
        mark_job_failed(job=job, exc=exc)
        raise
    return job


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
    if input_kind == "url":
        source_url = optional_string(input_data.get("url"))
        if not source_url:
            raise ApiError("Batch transcription requires input.url.")
        parsed = urlparse(source_url)
        if not parsed.scheme or not parsed.netloc:
            raise ApiError("input.url must be an absolute URL.")
        normalized_input_data = {"kind": "url", "url": source_url}
    elif input_kind == "upload":
        upload_id = optional_string(input_data.get("upload_id"))
        if not upload_id:
            raise ApiError("Batch upload transcription requires input.upload_id.")
        normalized_input_data = {
            "kind": "upload",
            "upload_id": ensure_uuid_string(upload_id, "input.upload_id"),
        }
    else:
        raise ApiError("Transcribe jobs support input.kind=url or input.kind=upload.")

    output_formats = validate_transcription_output_formats(
        ensure_object(payload.get("output", {}), "output")
    )
    diarization = parse_diarization_option(payload)
    context = ensure_object(payload.get("context", {}), "context")
    task_ref = optional_string(payload.get("task_ref")) or ""

    return JobRequest(
        job_type=Job.JobType.TRANSCRIBE,
        priority=priority,
        lane=lane,
        backend=backend,
        model=model,
        language=language,
        input_data=normalized_input_data,
        output_formats=output_formats,
        diarization=diarization,
        context=context,
        task_ref=task_ref,
    )


def existing_job_matches_request(existing: Job, request: JobRequest) -> bool:
    if existing.job_type != request.job_type:
        return False
    if request.job_type != Job.JobType.TRANSCRIBE:
        return True
    return (
        existing.backend == request.backend
        and existing.model == request.model
        and (existing.language or None) == request.language
        and existing_transcription_input_matches_request(existing.input_data, request.input_data)
        and sorted(existing.output_data.get("formats", [])) == sorted(request.output_formats)
        and existing.output_data.get("diarization", {"enabled": False}) == request.diarization
    )


def existing_transcription_input_matches_request(
    existing_input: dict[str, Any],
    request_input: dict[str, Any],
) -> bool:
    existing_kind = existing_input.get("kind")
    request_kind = request_input.get("kind")
    if existing_kind != request_kind:
        return False
    if request_kind == "upload":
        return str(existing_input.get("upload_id") or "") == str(
            request_input.get("upload_id") or ""
        )
    return existing_input == request_input


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
        diarization={"enabled": False},
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
            raise ApiError("Unsupported output format. Use text, json, vtt, dote, or podlove.")
        if normalized == "webvtt":
            normalized = "vtt"
        if normalized not in output_formats:
            output_formats.append(normalized)
    return output_formats


def parse_diarization_option(payload: dict[str, Any]) -> dict[str, Any]:
    raw_diarization = payload.get("diarization")
    if raw_diarization is None:
        return {"enabled": False}
    diarization = ensure_object(raw_diarization, "diarization")
    allowed_keys = {
        "enabled",
        "num_speakers",
        "min_speakers",
        "max_speakers",
        "strategy",
        "known_speakers",
        "known_speaker",
    }
    unknown_keys = sorted(set(diarization) - allowed_keys)
    if unknown_keys:
        raise ApiError(f"Unsupported diarization option: {', '.join(unknown_keys)}.")
    if "enabled" not in diarization:
        raise ApiError("diarization.enabled must be provided when diarization is set.")
    raw_enabled = diarization["enabled"]
    if not isinstance(raw_enabled, bool):
        raise ApiError("diarization.enabled must be a boolean.")

    option_keys = allowed_keys - {"enabled"}
    if not raw_enabled and any(key in diarization for key in option_keys):
        raise ApiError("diarization options require diarization.enabled=true.")

    normalized: dict[str, Any] = {"enabled": raw_enabled}
    if not raw_enabled:
        return normalized

    num_speakers = optional_positive_int(
        diarization.get("num_speakers"),
        "diarization.num_speakers",
    )
    min_speakers = optional_positive_int(
        diarization.get("min_speakers"),
        "diarization.min_speakers",
    )
    max_speakers = optional_positive_int(
        diarization.get("max_speakers"),
        "diarization.max_speakers",
    )
    if num_speakers is not None and (min_speakers is not None or max_speakers is not None):
        raise ApiError(
            "diarization.num_speakers cannot be combined with min_speakers or max_speakers."
        )
    if min_speakers is not None and max_speakers is not None and min_speakers > max_speakers:
        raise ApiError(
            "diarization.min_speakers must be less than or equal to diarization.max_speakers."
        )
    for key, value in (
        ("num_speakers", num_speakers),
        ("min_speakers", min_speakers),
        ("max_speakers", max_speakers),
    ):
        if value is not None:
            normalized[key] = value

    strategy = parse_diarization_strategy(diarization.get("strategy"))
    if strategy == KNOWN_SPEAKER_STRATEGY:
        normalized["strategy"] = KNOWN_SPEAKER_STRATEGY
        normalized["known_speakers"] = parse_known_speakers(diarization.get("known_speakers"))
        known_speaker_config = parse_known_speaker_config(diarization.get("known_speaker"))
        if known_speaker_config:
            normalized["known_speaker"] = known_speaker_config
    elif "known_speakers" in diarization or "known_speaker" in diarization:
        raise ApiError(
            "diarization.known_speakers and diarization.known_speaker require "
            f"strategy={KNOWN_SPEAKER_STRATEGY}."
        )
    return normalized


def parse_diarization_strategy(value: object) -> str:
    if value is None:
        return ANONYMOUS_STRATEGY
    if not isinstance(value, str):
        raise ApiError("diarization.strategy must be a string.")
    normalized = value.strip()
    if normalized not in DIARIZATION_STRATEGIES:
        accepted = ", ".join(sorted(DIARIZATION_STRATEGIES))
        raise ApiError(f"Unsupported diarization.strategy. Accepted values: {accepted}.")
    return normalized


def parse_known_speakers(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ApiError(
            "diarization.known_speakers must be a non-empty list when "
            f"strategy={KNOWN_SPEAKER_STRATEGY}."
        )
    known_speakers: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw_speaker in value:
        speaker = ensure_object(raw_speaker, "diarization.known_speakers[]")
        speaker_id = optional_string(speaker.get("id"))
        if not speaker_id:
            raise ApiError("Each diarization.known_speakers entry requires a non-empty id.")
        if speaker_id in seen_ids:
            raise ApiError(f"Duplicate diarization.known_speakers id '{speaker_id}'.")
        seen_ids.add(speaker_id)
        name = optional_string(speaker.get("name"))
        if not name:
            raise ApiError("Each diarization.known_speakers entry requires a non-empty name.")
        references = parse_known_speaker_references(speaker.get("references"))
        known_speakers.append({"id": speaker_id, "name": name, "references": references})
    return known_speakers


def parse_known_speaker_references(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ApiError(
            "Each diarization.known_speakers entry requires a non-empty references list."
        )
    references: list[dict[str, Any]] = []
    for raw_reference in value:
        reference = ensure_object(raw_reference, "diarization.known_speakers[].references[]")
        kind = optional_string(reference.get("kind"))
        if kind not in {"clip_artifact", "source_range"}:
            raise ApiError(
                "diarization reference kind must be 'clip_artifact' or 'source_range'."
            )
        audio = parse_reference_audio(reference.get("audio"))
        normalized: dict[str, Any] = {"kind": kind, "audio": audio}
        if kind == "source_range":
            start = require_non_negative_number(reference.get("start"), "reference.start")
            end = require_non_negative_number(reference.get("end"), "reference.end")
            if start >= end:
                raise ApiError("A source_range reference needs start before end.")
            normalized["start"] = start
            normalized["end"] = end
        references.append(normalized)
    return references


def parse_reference_audio(value: object) -> dict[str, Any]:
    audio = ensure_object(value, "diarization.known_speakers[].references[].audio")
    kind = optional_string(audio.get("kind"))
    if kind == "url":
        url = optional_string(audio.get("url"))
        if not url:
            raise ApiError("A url reference audio requires audio.url.")
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ApiError("reference audio.url must be an absolute URL.")
        return {"kind": "url", "url": url}
    if kind == "upload":
        upload_id = optional_string(audio.get("upload_id"))
        if not upload_id:
            raise ApiError("An upload reference audio requires audio.upload_id.")
        return {"kind": "upload", "upload_id": ensure_uuid_string(upload_id, "audio.upload_id")}
    raise ApiError("reference audio.kind must be 'url' or 'upload'.")


def parse_known_speaker_config(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    config = ensure_object(value, "diarization.known_speaker")
    allowed_keys = {
        "embedding_model",
        "min_segment_duration",
        "auto_accept_margin",
        "min_top_similarity",
    }
    unknown_keys = sorted(set(config) - allowed_keys)
    if unknown_keys:
        raise ApiError(f"Unsupported diarization.known_speaker option: {', '.join(unknown_keys)}.")
    normalized: dict[str, Any] = {}
    embedding_model = optional_string(config.get("embedding_model"))
    if embedding_model:
        normalized["embedding_model"] = embedding_model
    for key in ("min_segment_duration", "auto_accept_margin", "min_top_similarity"):
        if key in config:
            normalized[key] = require_non_negative_number(
                config.get(key), f"diarization.known_speaker.{key}"
            )
    return normalized


def optional_positive_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ApiError(f"{field_name} must be a positive integer.")
    return value


def require_non_negative_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ApiError(f"{field_name} must be a number.")
    number = float(value)
    if number < 0:
        raise ApiError(f"{field_name} must not be negative.")
    return number


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
    accepted_models = get_batch_accepted_stt_models()
    if normalized not in accepted_models:
        accepted = ", ".join(sorted(accepted_models))
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


def ensure_uuid_string(value: str, field_name: str) -> str:
    try:
        return str(uuid.UUID(value))
    except ValueError as exc:
        raise ApiError(f"{field_name} must be a UUID string.") from exc


def execute_transcription_job(*, job_id: str, task_result_id: str) -> dict[str, Any]:
    job = initialize_running_job(job_id=job_id, task_result_id=task_result_id)
    media: DownloadedMedia | None = None
    staged_media: StagedMedia | None = None
    extracted_audio_path: Path | None = None
    started = monotonic()
    try:
        media, staged_media = prepare_transcription_input_media(job=job)
        store_job_input_artifact(job=job, media=media)
        if staged_media is not None:
            delete_staged_media(staged=staged_media, missing_ok=True)
            staged_media = None

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
        speakers_artifact: dict[str, Any] | None = None
        if transcription_diarization_enabled(job):
            if transcription_diarization_strategy(job) == KNOWN_SPEAKER_STRATEGY:
                result, speakers_artifact = run_known_speaker_for_job(
                    job=job, audio_path=audio_path, result=result
                )
            else:
                result = apply_speaker_labels(
                    result,
                    diarize_audio(audio_path, transcription_diarization_params(job)),
                )
        processing_seconds = round(monotonic() - started, 3)
        metadata = build_transcription_result_metadata(
            job=job,
            media=media,
            result=result,
            processing_seconds=processing_seconds,
            speakers_artifact=speakers_artifact,
        )
        persist_transcription_output_artifacts(
            job=job, result=result, speakers_artifact=speakers_artifact
        )
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
        if staged_media is not None:
            try:
                delete_staged_media(staged=staged_media, missing_ok=True)
            except Exception:
                pass


def prepare_transcription_input_media(*, job: Job) -> tuple[DownloadedMedia, StagedMedia | None]:
    input_kind = str(job.input_data.get("kind") or "")
    if input_kind == "url":
        return download_allowed_media(source_url=str(job.input_data["url"])), None
    if input_kind == "upload":
        try:
            staged_media = StagedMedia.objects.get(
                id=str(job.input_data["upload_id"]),
                producer=job.producer,
                claimed_by_job=job,
            )
        except StagedMedia.DoesNotExist as exc:
            raise RuntimeError(
                "Failed to materialize staged input: upload handle is missing."
            ) from exc
        return materialize_staged_media(staged=staged_media), staged_media
    raise RuntimeError(f"Unsupported transcription input kind '{input_kind}'.")


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
            export.path for export in exports if result is None or export.path != result.audio_path
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
    speakers_artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = build_transcription_metadata(
        requested_model=job.model or "auto",
        requested_language=job.language or None,
        result=result,
        processing_seconds=processing_seconds,
        source_kind=media.source_kind,
        source_name=media.source_name,
        source_url=media.source_url,
        source_content_type=media.content_type,
    )
    if transcription_diarization_enabled(job):
        metadata["diarization"] = transcription_diarization_payload(job)
        if speakers_artifact is not None:
            metadata["diarization"]["known_speaker_summary"] = speakers_artifact["summary"]
    return metadata


def build_transcription_metadata(
    *,
    requested_model: str,
    requested_language: str | None,
    result: TranscriptionResult,
    processing_seconds: float,
    source_kind: str,
    source_name: str,
    source_url: str,
    source_content_type: str,
) -> dict[str, Any]:
    duration_seconds = 0.0
    if result.segments:
        duration_seconds = max(segment.end for segment in result.segments)
    return {
        "backend": result.backend_name or settings.VOXHELM_STT_BACKEND,
        "requested_model": requested_model or "auto",
        "model": result.model_name or requested_model or "auto",
        "language": result.language or requested_language or "",
        "duration_seconds": duration_seconds,
        "processing_seconds": processing_seconds,
        "source_kind": source_kind,
        "source_name": source_name,
        "source_url": source_url,
        "source_content_type": source_content_type,
    }


def transcription_diarization_enabled(job: Job) -> bool:
    raw_diarization = job.output_data.get("diarization")
    return isinstance(raw_diarization, dict) and raw_diarization.get("enabled") is True


def transcription_diarization_payload(job: Job) -> dict[str, Any]:
    raw_diarization = job.output_data.get("diarization")
    if not isinstance(raw_diarization, dict):
        return {"enabled": False}
    payload: dict[str, Any] = {"enabled": raw_diarization.get("enabled") is True}
    for key in ("num_speakers", "min_speakers", "max_speakers"):
        value = raw_diarization.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            payload[key] = value
    strategy = raw_diarization.get("strategy")
    if strategy == KNOWN_SPEAKER_STRATEGY:
        payload["strategy"] = KNOWN_SPEAKER_STRATEGY
        # Surface only non-sensitive reference shape in result metadata, never the
        # private reference URLs/ranges django-cast sent.
        known_speakers = raw_diarization.get("known_speakers")
        if isinstance(known_speakers, list):
            payload["known_speakers"] = [
                {
                    "id": speaker.get("id"),
                    "name": speaker.get("name"),
                    "reference_count": len(speaker.get("references", [])),
                }
                for speaker in known_speakers
                if isinstance(speaker, dict)
            ]
        known_speaker_config = raw_diarization.get("known_speaker")
        if isinstance(known_speaker_config, dict):
            payload["known_speaker"] = dict(known_speaker_config)
    return payload


def transcription_diarization_strategy(job: Job) -> str:
    raw_diarization = job.output_data.get("diarization")
    if (
        isinstance(raw_diarization, dict)
        and raw_diarization.get("strategy") == KNOWN_SPEAKER_STRATEGY
    ):
        return KNOWN_SPEAKER_STRATEGY
    return ANONYMOUS_STRATEGY


def transcription_diarization_params(job: Job) -> DiarizationParams:
    raw_diarization = job.output_data.get("diarization")
    counts: dict[str, Any] = raw_diarization if isinstance(raw_diarization, dict) else {}
    return DiarizationParams(
        num_speakers=_positive_int_or_none(counts.get("num_speakers")),
        min_speakers=_positive_int_or_none(counts.get("min_speakers")),
        max_speakers=_positive_int_or_none(counts.get("max_speakers")),
    )


def _positive_int_or_none(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def known_speaker_config_for_job(job: Job) -> KnownSpeakerConfig:
    raw_diarization = job.output_data.get("diarization")
    config_data: dict[str, Any] = {}
    if isinstance(raw_diarization, dict):
        raw_config = raw_diarization.get("known_speaker")
        if isinstance(raw_config, dict):
            config_data = raw_config
    defaults = KnownSpeakerConfig()
    return KnownSpeakerConfig(
        embedding_model=config_data.get("embedding_model") or defaults.embedding_model,
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


def build_known_speaker_references(job: Job) -> list[ReferenceAudio]:
    raw_diarization = job.output_data.get("diarization")
    known_speakers = None
    if isinstance(raw_diarization, dict):
        known_speakers = raw_diarization.get("known_speakers")
    if not isinstance(known_speakers, list):
        return []
    references: list[ReferenceAudio] = []
    for speaker in known_speakers:
        if not isinstance(speaker, dict):
            continue
        for raw_reference in speaker.get("references", []):
            windows = load_reference_windows(raw_reference)
            if windows:
                references.append(
                    ReferenceAudio(
                        speaker_id=str(speaker.get("id")),
                        name=str(speaker.get("name")),
                        windows=windows,
                    )
                )
    return references


def load_reference_windows(reference: dict[str, Any]) -> list[Any]:
    from transcriptions.known_speaker import SAMPLE_RATE

    audio = reference.get("audio", {})
    audio_kind = audio.get("kind")
    if audio_kind != "url":
        # Uploaded-clip references are accepted by the contract but require signed
        # private delivery; production references use source ranges into already
        # public mastered audio. Document this as a follow-up rather than guess.
        raise ApiError(
            "Only url-based known-speaker reference audio is supported for execution; "
            "deliver reference clips as source ranges into allowlisted audio."
        )
    media = download_allowed_media(source_url=str(audio.get("url")))
    try:
        samples = decode_mono_16k(media.path)
    finally:
        media.path.unlink(missing_ok=True)
    if reference.get("kind") == "source_range":
        samples = slice_samples(
            samples, SAMPLE_RATE, float(reference["start"]), float(reference["end"])
        )
    return extract_reference_windows(samples, SAMPLE_RATE)


def run_known_speaker_for_job(
    *,
    job: Job,
    audio_path: Path,
    result: TranscriptionResult,
) -> tuple[TranscriptionResult, dict[str, Any]]:
    config = known_speaker_config_for_job(job)
    references = build_known_speaker_references(job)
    backend = get_known_speaker_backend(config.embedding_model)
    job_audio_samples = decode_mono_16k(audio_path)
    raw_turns = collect_anonymous_diarization_turns(job, audio_path)
    outcome = run_known_speaker_postprocess(
        result,
        references=references,
        job_audio_samples=job_audio_samples,
        raw_turns=raw_turns,
        config=config,
        backend=backend,
    )
    return outcome.result, build_speakers_artifact(outcome)


def collect_anonymous_diarization_turns(job: Job, audio_path: Path) -> list[SpeakerTurn]:
    """Run anonymous pyannote as a fallback/debug signal; tolerate its failure.

    Known-speaker classification does not depend on these turns, so a diarization
    backend failure must not fail the known-speaker job. The raw labels are kept
    per segment for audit when available.
    """
    try:
        return diarize_audio(audio_path, transcription_diarization_params(job))
    except DiarizationError:
        return []


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


def persist_transcription_output_artifacts(
    *,
    job: Job,
    result: TranscriptionResult,
    speakers_artifact: dict[str, Any] | None = None,
) -> None:
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
            payload=render_text(result).encode("utf-8"),
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
    if "dote" in requested_formats:
        dote_payload = json.dumps(render_dote(result), ensure_ascii=True, indent=2)
        create_or_replace_artifact(
            job=job,
            name="transcript.dote.json",
            kind=JobArtifact.Kind.TRANSCRIPT_DOTE,
            format_name="dote",
            content_type="application/json",
            payload=dote_payload.encode("utf-8"),
            exposed=True,
        )
    if "podlove" in requested_formats:
        podlove_payload = json.dumps(render_podlove(result), ensure_ascii=True, indent=2)
        create_or_replace_artifact(
            job=job,
            name="transcript.podlove.json",
            kind=JobArtifact.Kind.TRANSCRIPT_PODLOVE,
            format_name="podlove",
            content_type="application/json",
            payload=podlove_payload.encode("utf-8"),
            exposed=True,
        )
    if speakers_artifact is not None:
        speakers_payload = json.dumps(speakers_artifact, ensure_ascii=True, indent=2)
        create_or_replace_artifact(
            job=job,
            name="transcript.speakers.json",
            kind=JobArtifact.Kind.TRANSCRIPT_SPEAKERS,
            format_name="speakers",
            content_type="application/json",
            payload=speakers_payload.encode("utf-8"),
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
    create_or_replace_artifact_from_file(
        job=job,
        name=media.source_name or "input",
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
