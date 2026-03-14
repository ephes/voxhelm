from __future__ import annotations

from typing import Any, cast
from urllib.parse import urlparse

from django import forms
from django.conf import settings

from transcriptions.input_media import detect_suffix


class LoginForm(forms.Form):
    username = forms.CharField(max_length=150)
    password = forms.CharField(widget=forms.PasswordInput)


class TranscriptSubmissionForm(forms.Form):
    audio_url = forms.URLField(required=False, label="Audio URL", assume_scheme="https")
    video_url = forms.URLField(required=False, label="Video URL", assume_scheme="https")
    audio_file = forms.FileField(required=False, label="Audio file")
    language = forms.CharField(required=False, max_length=32)

    def clean(self) -> dict[str, Any]:
        cleaned_data = cast(dict[str, Any], super().clean())
        filled = [
            field_name
            for field_name in ("audio_url", "video_url", "audio_file")
            if cleaned_data.get(field_name)
        ]
        if not filled:
            raise forms.ValidationError(
                "Provide exactly one input: audio URL, video URL, or uploaded audio file."
            )
        if len(filled) > 1:
            raise forms.ValidationError("Submit only one input at a time.")
        cleaned_data["submission_type"] = filled[0]
        return cleaned_data

    def clean_audio_url(self) -> str:
        return self._clean_url_field("audio_url")

    def clean_video_url(self) -> str:
        return self._clean_url_field("video_url")

    def clean_audio_file(self):
        audio_file = self.cleaned_data.get("audio_file")
        if not audio_file:
            return audio_file
        if (audio_file.size or 0) > settings.VOXHELM_MAX_UPLOAD_BYTES:
            raise forms.ValidationError(
                f"Uploaded file exceeded {settings.VOXHELM_MAX_UPLOAD_MIB} MiB transcription limit."
            )
        suffix = detect_suffix(audio_file.name or "", getattr(audio_file, "content_type", "") or "")
        if not suffix:
            raise forms.ValidationError("Unsupported uploaded media type for transcription.")
        return audio_file

    def _clean_url_field(self, field_name: str) -> str:
        value = self.cleaned_data.get(field_name)
        if not value:
            return ""
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise forms.ValidationError("Enter a valid absolute HTTP(S) URL.")
        return value
