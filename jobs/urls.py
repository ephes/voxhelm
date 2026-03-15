from django.urls import path

from jobs.views import job_artifact, job_detail, jobs_collection, uploads_collection

urlpatterns = [
    path("uploads", uploads_collection, name="uploads-collection"),
    path("jobs", jobs_collection, name="jobs-collection"),
    path("jobs/<uuid:job_id>", job_detail, name="job-detail"),
    path("jobs/<uuid:job_id>/artifacts/<path:name>", job_artifact, name="job-artifact"),
]
