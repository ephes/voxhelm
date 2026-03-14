from django.urls import include, path

from operators.views import logout_view, operator_artifact, root
from synthesis.views import audio_speech
from transcriptions.views import audio_transcriptions, health

urlpatterns = [
    path("", root, name="root"),
    path("logout", logout_view, name="logout"),
    path(
        "transcripts/<uuid:job_id>/artifacts/<str:format_name>",
        operator_artifact,
        name="operator-artifact",
    ),
    path("v1/health", health, name="health"),
    path("v1/audio/speech", audio_speech, name="audio-speech"),
    path("v1/audio/transcriptions", audio_transcriptions, name="audio-transcriptions"),
    path("v1/", include("jobs.urls")),
]
