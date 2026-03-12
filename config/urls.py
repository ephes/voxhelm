from django.urls import include, path

from transcriptions.views import audio_transcriptions, health

urlpatterns = [
    path("v1/health", health, name="health"),
    path("v1/audio/transcriptions", audio_transcriptions, name="audio-transcriptions"),
    path("v1/", include("jobs.urls")),
]
