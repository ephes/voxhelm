from __future__ import annotations

import json
from dataclasses import dataclass

from django.conf import settings
from django.http import HttpRequest, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from synthesis.service import MAX_TTS_SPEED, MIN_TTS_SPEED, export_audio, synthesize_text
from transcriptions.views import (
    ApiError,
    openai_error_response,
    optional_string,
    require_bearer_token,
)

from .service import SynthesizeParams

RESPONSE_FORMATS = {"wav", "mp3", "ogg"}


@dataclass(frozen=True)
class ParsedSpeechRequest:
    text: str
    request_model: str
    response_format: str
    voice: str | None
    language: str | None
    speed: float
    routing: bool


@csrf_exempt
@require_POST
def audio_speech(request: HttpRequest) -> HttpResponse:
    result = None
    exported = None
    try:
        require_bearer_token(request)
        parsed_request = parse_speech_request(request)
        result = synthesize_text(
            parsed_request.text,
            SynthesizeParams(
                request_model=parsed_request.request_model,
                voice=parsed_request.voice,
                language=parsed_request.language,
                speed=parsed_request.speed,
                routing=parsed_request.routing,
            ),
        )
        exported = export_audio(result, output_format=parsed_request.response_format)
        response = HttpResponse(
            exported.path.read_bytes(),
            content_type=exported.content_type,
        )
        response["Content-Disposition"] = f'inline; filename="speech.{exported.format_name}"'
        # Expose the effective backend/voice/language (post-routing) for debugging
        # and Slice 6's metadata assertions.
        response["X-Voxhelm-Backend"] = result.backend_name
        response["X-Voxhelm-Voice"] = result.voice_name
        response["X-Voxhelm-Language"] = result.language or ""
        return response
    except ApiError as exc:
        return openai_error_response(exc.message, status=exc.status, error_type=exc.error_type)
    except RuntimeError as exc:
        return openai_error_response(str(exc), status=500, error_type="server_error")
    finally:
        if exported is not None and result is not None and exported.path != result.audio_path:
            exported.path.unlink(missing_ok=True)
        if result is not None:
            result.audio_path.unlink(missing_ok=True)


def parse_speech_request(request: HttpRequest) -> ParsedSpeechRequest:
    if not (request.content_type or "").lower().startswith("application/json"):
        raise ApiError("Speech synthesis requires application/json.")
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ApiError("Request body was not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ApiError("JSON request body must be an object.")

    text = payload.get("input")
    if not isinstance(text, str) or not text.strip():
        raise ApiError("The 'input' field is required and must be a non-empty string.")
    normalized_text = text.strip()
    if len(normalized_text) > settings.VOXHELM_TTS_MAX_INPUT_CHARS:
        raise ApiError(
            "Input text exceeded the configured "
            f"{settings.VOXHELM_TTS_MAX_INPUT_CHARS} character limit."
        )

    request_model = validate_model(payload.get("model"))
    response_format = validate_response_format(payload.get("response_format"))
    voice = optional_string(payload.get("voice"))
    language = optional_string(payload.get("language"))
    speed = validate_speed(payload.get("speed"))
    routing = validate_routing(payload.get("routing"))
    return ParsedSpeechRequest(
        text=normalized_text,
        request_model=request_model,
        response_format=response_format,
        voice=voice,
        language=language,
        speed=speed,
        routing=routing,
    )


def validate_model(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ApiError("The 'model' field is required.")
    normalized = value.strip()
    if normalized not in settings.VOXHELM_ACCEPTED_SPEECH_MODELS:
        accepted = ", ".join(sorted(settings.VOXHELM_ACCEPTED_SPEECH_MODELS))
        raise ApiError(f"Unsupported model '{normalized}'. Accepted values: {accepted}.")
    return normalized


def validate_response_format(value: object) -> str:
    if value is None:
        return "wav"
    if not isinstance(value, str):
        raise ApiError("The 'response_format' field must be a string.")
    normalized = value.strip().lower()
    if normalized not in RESPONSE_FORMATS:
        accepted = ", ".join(sorted(RESPONSE_FORMATS))
        raise ApiError(f"Unsupported response_format '{normalized}'. Accepted values: {accepted}.")
    return normalized


def validate_routing(value: object) -> bool:
    if value is None:
        return True
    if not isinstance(value, bool):
        raise ApiError("The 'routing' field must be a boolean.")
    return value


def validate_speed(value: object) -> float:
    if value is None:
        return 1.0
    if not isinstance(value, (int, float)):
        raise ApiError("The 'speed' field must be a number.")
    normalized = float(value)
    if not MIN_TTS_SPEED <= normalized <= MAX_TTS_SPEED:
        raise ApiError(f"The 'speed' field must be between {MIN_TTS_SPEED} and {MAX_TTS_SPEED}.")
    return normalized
