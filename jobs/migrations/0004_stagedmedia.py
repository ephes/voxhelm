from django.db import migrations, models
import django.db.models.deletion
import uuid


class Migration(migrations.Migration):
    dependencies = [
        ("jobs", "0003_job_dispatch_mode_job_operator_and_more"),
    ]

    operations = [
        migrations.CreateModel(
            name="StagedMedia",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("producer", models.CharField(max_length=64)),
                ("original_filename", models.CharField(max_length=255)),
                ("content_type", models.CharField(max_length=255)),
                ("size_bytes", models.PositiveBigIntegerField(default=0)),
                ("storage_backend", models.CharField(max_length=32)),
                ("storage_key", models.CharField(max_length=512)),
                ("claimed_at", models.DateTimeField(blank=True, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("expires_at", models.DateTimeField()),
                (
                    "claimed_by_job",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="staged_inputs",
                        to="jobs.job",
                    ),
                ),
            ],
            options={
                "ordering": ["-created_at"],
            },
        ),
        migrations.AddIndex(
            model_name="stagedmedia",
            index=models.Index(fields=["producer", "created_at"], name="jobs_staged_producer_76b546_idx"),
        ),
        migrations.AddIndex(
            model_name="stagedmedia",
            index=models.Index(fields=["expires_at"], name="jobs_staged_expires_617032_idx"),
        ),
    ]
