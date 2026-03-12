from django.urls import path

from voxhelm_api.views import audio_transcriptions, health

urlpatterns = [
    path("v1/health", health, name="health"),
    path("v1/audio/transcriptions", audio_transcriptions, name="audio-transcriptions"),
]

