from __future__ import annotations

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from jobs.retention import prune_job_artifacts


class Command(BaseCommand):
    help = (
        "Delete expired intermediate job artifacts (D-09): non-exposed source media of "
        "finished jobs older than VOXHELM_SOURCE_ARTIFACT_RETENTION_SECONDS and extracted "
        "audio of finished jobs. Transcript and speech artifacts are never touched, except "
        "that jobs of a transcription profile with a backstop (ad-analysis-v1: 7 days) lose "
        "every artifact, the objects of remote attempts and a staged upload they still hold once "
        "they finished longer ago than it. Then delete job rows of "
        "terminal jobs finished more than VOXHELM_JOB_METADATA_RETENTION_SECONDS ago (or "
        "past their profile's backstop) that no longer own any artifact, queued deletion "
        "or claimed staged upload."
    )

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="List what would be deleted without deleting anything.",
        )

    def handle(self, *args, **options) -> None:
        del args
        dry_run = bool(options["dry_run"])
        result = prune_job_artifacts(dry_run=dry_run)
        verb = "Would delete" if dry_run else "Deleted"
        for artifact in result.deleted:
            self.stdout.write(
                f"{verb} {artifact.kind} artifact {artifact.storage_key} "
                f"({artifact.size_bytes} bytes) of job {artifact.job_id}"
            )
        total_bytes = sum(artifact.size_bytes for artifact in result.deleted)
        self.stdout.write(
            f"{verb} {len(result.deleted)} artifact(s), {total_bytes} bytes "
            f"(source retention {settings.VOXHELM_SOURCE_ARTIFACT_RETENTION_SECONDS}s)."
        )
        for artifact in result.refused:
            self.stdout.write(
                f"Refused {artifact.kind} artifact {artifact.storage_key} of job "
                f"{artifact.job_id}: not a voxhelm-generated job key; left in place."
            )
        if result.pending_refused:
            self.stdout.write(
                f"Refused {result.pending_refused} queued object(s): not voxhelm-generated "
                "job keys; left in place and kept queued."
            )
        if result.pending_deleted or result.pending_failed:
            self.stdout.write(
                f"{verb} {result.pending_deleted} queued replaced object(s)"
                + (f", {result.pending_failed} still pending" if result.pending_failed else "")
                + "."
            )
        if result.attempt_prefixes_deleted or result.attempt_prefixes_failed:
            self.stdout.write(
                f"{verb} {result.attempt_prefixes_deleted} remote attempt prefix(es) of "
                "profile jobs past their backstop"
                + (
                    f", {result.attempt_prefixes_failed} still pending"
                    if result.attempt_prefixes_failed
                    else ""
                )
                + "."
            )
        if result.attempt_prefixes_refused:
            self.stdout.write(
                f"Refused {result.attempt_prefixes_refused} remote attempt prefix(es): they "
                "hold objects outside the job's generated layout; left in place."
            )
        if result.staged_deleted or result.staged_failed:
            self.stdout.write(
                f"{verb} {result.staged_deleted} staged upload(s) of profile jobs past their "
                "backstop"
                + (f", {result.staged_failed} still pending" if result.staged_failed else "")
                + "."
            )
        if result.staged_refused:
            self.stdout.write(
                f"Refused {result.staged_refused} staged upload(s): not voxhelm staged keys; "
                "left in place."
            )
        for job in result.jobs_deleted:
            self.stdout.write(
                f"{verb} {job.state} {job.job_type} job {job.id} of producer {job.producer} "
                f"(finished {(job.finished_at or job.updated_at).isoformat()})"
            )
        retention = settings.VOXHELM_JOB_METADATA_RETENTION_SECONDS
        self.stdout.write(
            f"{verb} {len(result.jobs_deleted)} job row(s) (job metadata retention "
            + (f"{retention}s" if retention else "disabled")
            + ")."
        )
        failed = (
            len(result.failed)
            + result.pending_failed
            + result.attempt_prefixes_failed
            + result.staged_failed
        )
        if failed:
            raise CommandError(
                f"Could not delete {failed} object(s) or attempt prefix(es); "
                "they stay recorded for the next run."
            )
