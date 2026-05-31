from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('jobs', '0006_alter_jobartifact_kind'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Worker',
            fields=[
                ('worker_id', models.CharField(max_length=64, primary_key=True, serialize=False)),
                ('hostname', models.CharField(blank=True, max_length=255)),
                ('enabled', models.BooleanField(default=True)),
                ('capabilities', models.JSONField(default=dict)),
                ('concurrency', models.PositiveIntegerField(default=1)),
                ('running_job_ids', models.JSONField(default=list)),
                ('last_seen_at', models.DateTimeField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'ordering': ['worker_id'],
            },
        ),
        migrations.AddField(
            model_name='job',
            name='assigned_worker_id',
            field=models.CharField(blank=True, max_length=64),
        ),
        migrations.AddField(
            model_name='job',
            name='attempt_count',
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.AddField(
            model_name='job',
            name='execution_mode',
            field=models.CharField(choices=[('django_tasks', 'Django Tasks'), ('remote_pull', 'Remote pull')], default='django_tasks', max_length=32),
        ),
        migrations.AddField(
            model_name='job',
            name='last_worker_heartbeat_at',
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='job',
            name='lease_expires_at',
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='job',
            name='lease_token_hash',
            field=models.CharField(blank=True, max_length=64),
        ),
        migrations.AddField(
            model_name='job',
            name='leased_artifact_prefix',
            field=models.CharField(blank=True, max_length=512),
        ),
        migrations.AddField(
            model_name='job',
            name='leased_artifact_store',
            field=models.JSONField(default=dict),
        ),
        migrations.AddField(
            model_name='job',
            name='max_attempts',
            field=models.PositiveIntegerField(default=3),
        ),
        migrations.AddField(
            model_name='job',
            name='worker_progress',
            field=models.JSONField(default=dict),
        ),
        migrations.AddField(
            model_name='jobartifact',
            name='storage_identity',
            field=models.JSONField(default=dict),
        ),
        migrations.AddIndex(
            model_name='job',
            index=models.Index(fields=['execution_mode', 'state', 'priority', 'created_at'], name='jobs_job_executi_e6b691_idx'),
        ),
        migrations.AddIndex(
            model_name='job',
            index=models.Index(fields=['assigned_worker_id', 'state'], name='jobs_job_assigne_d0bd36_idx'),
        ),
        migrations.AddIndex(
            model_name='job',
            index=models.Index(fields=['lease_expires_at'], name='jobs_job_lease_e_be8a89_idx'),
        ),
        migrations.AddField(
            model_name='stagedmedia',
            name='storage_identity',
            field=models.JSONField(default=dict),
        ),
    ]
