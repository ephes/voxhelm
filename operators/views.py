from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse
from uuid import UUID

from django.conf import settings
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.core.exceptions import ValidationError
from django.http import HttpRequest, HttpResponse, HttpResponseForbidden, HttpResponseRedirect
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from jobs.artifacts import get_artifact_store
from jobs.models import Job, JobArtifact
from jobs.services import (
    OPERATOR_TRANSCRIPTION_OUTPUT_FORMATS,
    create_job_from_payload_for_actor,
    create_operator_sync_transcription,
)
from transcriptions.errors import ApiError
from transcriptions.input_media import (
    detect_suffix,
    download_allowed_url_to_tempfile,
    write_upload_to_tempfile,
)

from .forms import LoginForm, TranscriptSubmissionForm


@require_http_methods(["GET", "POST"])
def root(request: HttpRequest) -> HttpResponse:
    if not request.user.is_authenticated:
        return login_page(request)
    if not request.user.is_staff:
        return operator_access_forbidden()
    return operator_home(request)


def login_page(request: HttpRequest) -> HttpResponse:
    form = LoginForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        user = authenticate(
            request,
            username=form.cleaned_data["username"],
            password=form.cleaned_data["password"],
        )
        if user is None:
            form.add_error(None, "Invalid username or password.")
        elif not user.is_active:
            form.add_error(None, "This operator account is inactive.")
        elif not user.is_staff:
            form.add_error(None, "This account is not allowed to access the operator UI.")
        else:
            login(request, user)
            return redirect("root")
    return render(request, "operators/root.html", {"login_form": form})


@login_required(login_url="/")
@require_http_methods(["GET", "POST"])
def operator_home(request: HttpRequest) -> HttpResponse:
    if not request.user.is_staff:
        return operator_access_forbidden()
    operator = cast(Any, request.user)
    selected_job = get_selected_job(request)
    form = TranscriptSubmissionForm()

    if request.method == "POST":
        form = TranscriptSubmissionForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                selected_job = handle_submission(request=request, form=form)
            except (ApiError, RuntimeError) as exc:
                form.add_error(None, str(exc))
            else:
                return HttpResponseRedirect(f"{reverse('root')}?job={selected_job.id}")

    recent_jobs = list(
        Job.objects.filter(operator=operator, job_type=Job.JobType.TRANSCRIBE)
        .prefetch_related("artifacts")
        .order_by("-created_at")[:10]
    )
    if selected_job is None and recent_jobs:
        selected_job = recent_jobs[0]

    context = {
        "form": form,
        "selected_job": selected_job,
        "selected_job_artifacts": build_download_links(selected_job),
        "recent_jobs": recent_jobs,
    }
    return render(request, "operators/root.html", context)


@login_required(login_url="/")
@require_POST
def logout_view(request: HttpRequest) -> HttpResponse:
    logout(request)
    return redirect("root")


@login_required(login_url="/")
@require_GET
def operator_artifact(request: HttpRequest, job_id: UUID, format_name: str) -> HttpResponse:
    if not request.user.is_staff:
        return operator_access_forbidden()
    operator = cast(Any, request.user)
    job = get_object_or_404(Job, id=job_id, operator=operator, job_type=Job.JobType.TRANSCRIBE)
    artifact = get_object_or_404(JobArtifact, job=job, format=format_name, exposed=True)
    store = get_artifact_store()
    return HttpResponse(
        store.read_bytes(key=artifact.storage_key),
        content_type=artifact.content_type,
    )


def get_selected_job(request: HttpRequest) -> Job | None:
    operator = cast(Any, request.user)
    raw_job_id = (request.GET.get("job") or "").strip()
    if not raw_job_id:
        return None
    try:
        return Job.objects.prefetch_related("artifacts").get(
            id=raw_job_id,
            operator=operator,
            job_type=Job.JobType.TRANSCRIBE,
        )
    except (Job.DoesNotExist, ValidationError, ValueError):
        return None


def handle_submission(*, request: HttpRequest, form: TranscriptSubmissionForm) -> Job:
    submission_type = form.cleaned_data["submission_type"]
    language = cleaned_optional_string(form.cleaned_data.get("language"))
    if submission_type == "audio_url":
        source_url = str(form.cleaned_data["audio_url"])
        temp_path = download_allowed_url_to_tempfile(source_url=source_url)
        try:
            return create_operator_sync_transcription(
                operator=request.user,
                source_path=temp_path,
                source_name=Path(urlparse(source_url).path or "audio").name or "audio",
                input_data={"kind": "url", "url": source_url},
                language=language,
            )
        finally:
            temp_path.unlink(missing_ok=True)
    if submission_type == "video_url":
        source_url = str(form.cleaned_data["video_url"])
        job, _created = create_job_from_payload_for_actor(
            producer=settings.VOXHELM_OPERATOR_PRODUCER_LABEL,
            operator=request.user,
            payload={
                "job_type": "transcribe",
                "priority": "normal",
                "lane": "batch",
                "backend": "auto",
                "model": "auto",
                "language": language,
                "input": {"kind": "url", "url": source_url},
                "output": {"formats": list(OPERATOR_TRANSCRIPTION_OUTPUT_FORMATS)},
                "context": {"submission_type": "video_url"},
                "task_ref": "",
            },
        )
        return job
    audio_file = form.cleaned_data["audio_file"]
    suffix = detect_suffix(audio_file.name or "", getattr(audio_file, "content_type", "") or "")
    temp_path = write_upload_to_tempfile(audio_file.chunks(), suffix=suffix)
    try:
        return create_operator_sync_transcription(
            operator=request.user,
            source_path=temp_path,
            source_name=audio_file.name or "upload",
            input_data={
                "kind": "upload",
                "filename": audio_file.name or "upload",
                "content_type": getattr(audio_file, "content_type", "") or "",
            },
            language=language,
        )
    finally:
        temp_path.unlink(missing_ok=True)


def build_download_links(job: Job | None) -> list[dict[str, str]]:
    if job is None or job.state != Job.State.SUCCEEDED:
        return []
    order = ["text", "json", "vtt", "dote", "podlove"]
    artifacts_by_format = {
        artifact.format: artifact
        for artifact in job.artifacts.filter(exposed=True)
    }
    downloads: list[dict[str, str]] = []
    for format_name in order:
        artifact = artifacts_by_format.get(format_name)
        if artifact is None:
            continue
        downloads.append(
            {
                "format": format_name,
                "label": format_name.upper() if format_name != "podlove" else "Podlove",
                "url": reverse(
                    "operator-artifact",
                    kwargs={"job_id": job.id, "format_name": artifact.format},
                ),
            }
        )
    return downloads


def cleaned_optional_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def operator_access_forbidden() -> HttpResponseForbidden:
    return HttpResponseForbidden("Operator access requires a staff account.")
