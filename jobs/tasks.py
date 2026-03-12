from __future__ import annotations

from django_tasks import TaskContext, task

from jobs.services import execute_transcription_job


@task(takes_context=True)
def run_transcription_job(context: TaskContext, job_id: str) -> dict[str, object]:
    return execute_transcription_job(job_id=job_id, task_result_id=str(context.task_result.id))
