from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass

from django.conf import settings
from django.http import HttpRequest
from django.http.response import HttpResponseBase
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from synthesis.service import (
    MAX_TTS_SPEED,
    MIN_TTS_SPEED,
    SynthesisInputError,
    export_audio,
    synthesize_text,
)
from transcriptions.file_streaming import reader_response
from transcriptions.views import (
    ApiError,
    openai_error_response,
    optional_string,
    require_bearer_token,
)

from .service import SynthesizeParams

RESPONSE_FORMATS = {"wav", "mp3", "ogg"}
_LOGGER = logging.getLogger(__name__)


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
def audio_speech(request: HttpRequest) -> HttpResponseBase:
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
        # Stream from an open handle. The files are unlinked in ``finally`` below; on
        # POSIX the open handle keeps the data readable until the response closes it.
        handle = exported.path.open("rb")
        try:
            response = reader_response(request, handle, content_type=exported.content_type)
        except BaseException:
            handle.close()
            raise
        response["Content-Disposition"] = f'inline; filename="speech.{exported.format_name}"'
        # Expose the effective backend/voice/language (post-routing) for debugging
        # and Slice 6's metadata assertions.
        response["X-Voxhelm-Backend"] = result.backend_name
        response["X-Voxhelm-Voice"] = result.voice_name
        response["X-Voxhelm-Language"] = result.language or ""
        return response
    except ApiError as exc:
        return openai_error_response(exc.message, status=exc.status, error_type=exc.error_type)
    except SynthesisInputError as exc:
        return openai_error_response(str(exc), status=400, error_type="invalid_request_error")
    except RuntimeError, OSError:
        # Backend, converter and filesystem diagnostics stay in the server log.
        _LOGGER.exception("Speech synthesis failed")
        return openai_error_response(
            "Speech synthesis failed.", status=500, error_type="server_error"
        )
    finally:
        if exported is not None and result is not None and exported.path != result.audio_path:
            exported.path.unlink(missing_ok=True)
        if result is not None:
            result.audio_path.unlink(missing_ok=True)


def parse_speech_request(request: HttpRequest) -> ParsedSpeechRequest:
    if not (request.content_type or "").lower().startswith("application/json"):
        raise ApiError("Speech synthesis requires application/json.")
    payload = load_speech_json(request.body)

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


def _reject_constant(name: str) -> object:
    raise ValueError(f"non-finite constant {name}")


def load_speech_json(body: bytes) -> dict[str, object]:
    """Parse a speech request, refusing every parser limit as invalid JSON.

    Deep nesting (the parser's recursion limit, which varies between Python
    builds), NaN/Infinity and integers beyond Python's digit limit are reported
    as invalid JSON instead of escaping as server errors.
    """
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ApiError("Request body was not valid JSON.") from exc
    try:
        payload = json.loads(text, parse_constant=_reject_constant)
    except (ValueError, RecursionError) as exc:
        raise ApiError("Request body was not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ApiError("JSON request body must be an object.")
    return payload


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
    # bool is an int subclass; JSON true/false is not a speed.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ApiError("The 'speed' field must be a number.")
    try:
        normalized = float(value)
    except OverflowError:
        normalized = math.inf
    if not math.isfinite(normalized) or not MIN_TTS_SPEED <= normalized <= MAX_TTS_SPEED:
        raise ApiError(f"The 'speed' field must be between {MIN_TTS_SPEED} and {MAX_TTS_SPEED}.")
    return normalized
