from __future__ import annotations

import uuid

from django.conf import settings
from django.db import models


class Job(models.Model):
    class JobType(models.TextChoices):
        TRANSCRIBE = "transcribe", "Transcribe"
        SYNTHESIZE = "synthesize", "Synthesize"

    class Lane(models.TextChoices):
        BATCH = "batch", "Batch"

    class DispatchMode(models.TextChoices):
        SYNC = "sync", "Sync"
        BATCH = "batch", "Batch"

    class Priority(models.TextChoices):
        LOW = "low", "Low"
        NORMAL = "normal", "Normal"
        HIGH = "high", "High"

    class State(models.TextChoices):
        QUEUED = "queued", "Queued"
        RUNNING = "running", "Running"
        SUCCEEDED = "succeeded", "Succeeded"
        FAILED = "failed", "Failed"
        CANCELED = "canceled", "Canceled"
        EXPIRED = "expired", "Expired"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    producer = models.CharField(max_length=64)
    operator = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="voxhelm_jobs",
    )
    task_ref = models.CharField(max_length=255, blank=True)
    job_type = models.CharField(max_length=32, choices=JobType.choices)
    lane = models.CharField(max_length=32, choices=Lane.choices, default=Lane.BATCH)
    dispatch_mode = models.CharField(
        max_length=16,
        choices=DispatchMode.choices,
        default=DispatchMode.BATCH,
    )
    priority = models.CharField(
        max_length=32,
        choices=Priority.choices,
        default=Priority.NORMAL,
    )
    backend = models.CharField(max_length=128, blank=True)
    model = models.CharField(max_length=255, blank=True)
    language = models.CharField(max_length=32, blank=True)
    input_data = models.JSONField()
    output_data = models.JSONField(default=dict)
    context_data = models.JSONField(default=dict)
    state = models.CharField(max_length=32, choices=State.choices, default=State.QUEUED)
    django_task_id = models.CharField(max_length=64, blank=True)
    error_detail = models.TextField(blank=True)
    result_text = models.TextField(blank=True)
    result_metadata = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["producer", "task_ref"]),
            models.Index(fields=["operator", "created_at"]),
            models.Index(fields=["state"]),
        ]


class JobArtifact(models.Model):
    class Kind(models.TextChoices):
        SOURCE = "source", "Source"
        EXTRACTED_AUDIO = "extracted_audio", "Extracted audio"
        TRANSCRIPT_TEXT = "transcript_text", "Transcript text"
        TRANSCRIPT_JSON = "transcript_json", "Transcript JSON"
        TRANSCRIPT_VTT = "transcript_vtt", "Transcript VTT"
        TRANSCRIPT_DOTE = "transcript_dote", "Transcript DOTe"
        TRANSCRIPT_PODLOVE = "transcript_podlove", "Transcript Podlove"
        SPEECH_WAV = "speech_wav", "Speech WAV"
        SPEECH_MP3 = "speech_mp3", "Speech MP3"
        SPEECH_OGG = "speech_ogg", "Speech OGG"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    job = models.ForeignKey(Job, on_delete=models.CASCADE, related_name="artifacts")
    name = models.CharField(max_length=255)
    kind = models.CharField(max_length=64, choices=Kind.choices)
    format = models.CharField(max_length=32, blank=True)
    storage_backend = models.CharField(max_length=32)
    storage_key = models.CharField(max_length=512)
    content_type = models.CharField(max_length=255)
    size_bytes = models.PositiveBigIntegerField(default=0)
    exposed = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]
        constraints = [
            models.UniqueConstraint(fields=["job", "name"], name="jobs_artifact_job_name_unique")
        ]
