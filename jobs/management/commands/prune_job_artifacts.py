from __future__ import annotations

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from jobs.retention import prune_job_artifacts


class Command(BaseCommand):
    help = (
        "Delete expired intermediate job artifacts (D-09): non-exposed source media of "
        "finished jobs older than VOXHELM_SOURCE_ARTIFACT_RETENTION_SECONDS and extracted "
        "audio of finished jobs. Transcript and speech artifacts are never touched."
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
        if result.failed or result.pending_failed:
            raise CommandError(
                f"Could not delete {len(result.failed) + result.pending_failed} object(s); "
                "they stay recorded for the next run."
            )
