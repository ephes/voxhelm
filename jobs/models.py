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

    class ExecutionMode(models.TextChoices):
        DJANGO_TASKS = "django_tasks", "Django Tasks"
        REMOTE_PULL = "remote_pull", "Remote pull"

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
    # SHA-256 of the normalized, result-affecting request fields (see jobs.fingerprints). Empty
    # for jobs without a task_ref and for legacy duplicates left behind by the pre-constraint race.
    request_fingerprint = models.CharField(max_length=64, blank=True, default="")
    job_type = models.CharField(max_length=32, choices=JobType.choices)
    lane = models.CharField(max_length=32, choices=Lane.choices, default=Lane.BATCH)
    dispatch_mode = models.CharField(
        max_length=16,
        choices=DispatchMode.choices,
        default=DispatchMode.BATCH,
    )
    execution_mode = models.CharField(
        max_length=32,
        choices=ExecutionMode.choices,
        default=ExecutionMode.DJANGO_TASKS,
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
    assigned_worker_id = models.CharField(max_length=64, blank=True)
    lease_token_hash = models.CharField(max_length=64, blank=True)
    lease_expires_at = models.DateTimeField(null=True, blank=True)
    attempt_count = models.PositiveIntegerField(default=0)
    max_attempts = models.PositiveIntegerField(default=3)
    leased_artifact_prefix = models.CharField(max_length=512, blank=True)
    leased_artifact_store = models.JSONField(default=dict)
    last_worker_heartbeat_at = models.DateTimeField(null=True, blank=True)
    worker_progress = models.JSONField(default=dict)
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
            models.Index(fields=["execution_mode", "state", "priority", "created_at"]),
            models.Index(fields=["assigned_worker_id", "state"]),
            models.Index(fields=["lease_expires_at"]),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=["producer", "task_ref", "request_fingerprint"],
                condition=(
                    ~models.Q(task_ref="")
                    & ~models.Q(request_fingerprint="")
                    & ~models.Q(state="failed")
                ),
                name="jobs_job_active_task_ref_fingerprint_unique",
            )
        ]


class Worker(models.Model):
    worker_id = models.CharField(max_length=64, primary_key=True)
    hostname = models.CharField(max_length=255, blank=True)
    enabled = models.BooleanField(default=True)
    capabilities = models.JSONField(default=dict)
    concurrency = models.PositiveIntegerField(default=1)
    running_job_ids = models.JSONField(default=list)
    last_seen_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["worker_id"]


class JobArtifact(models.Model):
    class Kind(models.TextChoices):
        SOURCE = "source", "Source"
        EXTRACTED_AUDIO = "extracted_audio", "Extracted audio"
        TRANSCRIPT_TEXT = "transcript_text", "Transcript text"
        TRANSCRIPT_JSON = "transcript_json", "Transcript JSON"
        TRANSCRIPT_VTT = "transcript_vtt", "Transcript VTT"
        TRANSCRIPT_DOTE = "transcript_dote", "Transcript DOTe"
        TRANSCRIPT_PODLOVE = "transcript_podlove", "Transcript Podlove"
        TRANSCRIPT_SPEAKERS = "transcript_speakers", "Transcript speaker suggestions"
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
    storage_identity = models.JSONField(default=dict)
    content_type = models.CharField(max_length=255)
    size_bytes = models.PositiveBigIntegerField(default=0)
    exposed = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]
        constraints = [
            models.UniqueConstraint(fields=["job", "name"], name="jobs_artifact_job_name_unique")
        ]


class StagedMedia(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    producer = models.CharField(max_length=64)
    original_filename = models.CharField(max_length=255)
    content_type = models.CharField(max_length=255)
    size_bytes = models.PositiveBigIntegerField(default=0)
    storage_backend = models.CharField(max_length=32)
    storage_key = models.CharField(max_length=512)
    storage_identity = models.JSONField(default=dict)
    claimed_by_job = models.ForeignKey(
        Job,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="staged_inputs",
    )
    claimed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["producer", "created_at"]),
            models.Index(fields=["expires_at"]),
        ]


class PendingArtifactDeletion(models.Model):
    """A stored object that lost its artifact row and still has to be deleted (D-09).

    Written in the same transaction that drops the row, so a failed or skipped
    object deletion is retried by ``manage.py prune_job_artifacts``.
    """

    job_id = models.UUIDField(null=True, blank=True)
    kind = models.CharField(max_length=64, choices=JobArtifact.Kind.choices)
    storage_backend = models.CharField(max_length=32)
    storage_key = models.CharField(max_length=512)
    storage_identity = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]
