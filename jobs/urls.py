from django.urls import path

from jobs.views import (
    job_artifact,
    job_detail,
    jobs_collection,
    uploads_collection,
    work_claim,
    work_complete,
    work_fail,
    work_heartbeat,
    worker_heartbeat,
)

urlpatterns = [
    path("internal/workers/heartbeat", worker_heartbeat, name="worker-heartbeat"),
    path("internal/work/claim", work_claim, name="work-claim"),
    path("internal/work/<uuid:job_id>/heartbeat", work_heartbeat, name="work-heartbeat"),
    path("internal/work/<uuid:job_id>/complete", work_complete, name="work-complete"),
    path("internal/work/<uuid:job_id>/fail", work_fail, name="work-fail"),
    path("uploads", uploads_collection, name="uploads-collection"),
    path("jobs", jobs_collection, name="jobs-collection"),
    path("jobs/<uuid:job_id>", job_detail, name="job-detail"),
    path("jobs/<uuid:job_id>/artifacts/<path:name>", job_artifact, name="job-artifact"),
]
