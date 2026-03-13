from __future__ import annotations

import hmac
import json
import mimetypes
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from django.conf import settings
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from .observability import emit_transcription_debug_log, summarize_audio_file
from .service import (
    TranscribeParams,
    TranscriptionResult,
    render_verbose_json,
    render_vtt,
    transcribe_audio,
)

SUPPORTED_SUFFIXES: Final[dict[str, str]] = {
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".mpeg": "audio/mpeg",
    ".mpga": "audio/mpeg",
    ".oga": "audio/ogg",
    ".ogg": "audio/ogg",
    ".wav": "audio/wav",
}
CONTENT_TYPE_SUFFIXES: Final[dict[str, str]] = {
    "audio/flac": ".flac",
    "audio/mp4": ".m4a",
    "audio/mpeg": ".mp3",
    "audio/mpga": ".mpga",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
}
RESPONSE_FORMATS: Final[set[str]] = {"json", "text", "verbose_json", "vtt"}


class ApiError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status: int = 400,
        error_type: str = "invalid_request_error",
    ):
        super().__init__(message)
        self.message = message
        self.status = status
        self.error_type = error_type


@dataclass(frozen=True)
class ParsedRequest:
    input_path: Path
    request_model: str
    prompt: str | None
    language: str | None
    response_format: str


@require_GET
def health(request: HttpRequest) -> JsonResponse:
    del request
    return JsonResponse({"status": "ok"})


@csrf_exempt
@require_POST
def audio_transcriptions(request: HttpRequest) -> HttpResponse:
    temp_path: Path | None = None
    try:
        require_bearer_token(request)
        parsed_request = parse_transcription_request(request)
        temp_path = parsed_request.input_path
        started_at = time.monotonic()
        result = transcribe_audio(
            parsed_request.input_path,
            TranscribeParams(
                request_model=parsed_request.request_model,
                prompt=parsed_request.prompt,
                language=parsed_request.language,
            ),
        )
        emit_transcription_debug_log(
            source="http.audio_transcriptions",
            audio_shape=summarize_audio_file(parsed_request.input_path),
            request_model=parsed_request.request_model,
            request_language=parsed_request.language,
            prompt=parsed_request.prompt,
            result=result,
            duration_ms=int((time.monotonic() - started_at) * 1000),
        )
        return render_response(result=result, response_format=parsed_request.response_format)
    except ApiError as exc:
        return openai_error_response(exc.message, status=exc.status, error_type=exc.error_type)
    except RuntimeError as exc:
        return openai_error_response(str(exc), status=500, error_type="server_error")
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def require_bearer_token(request: HttpRequest) -> str:
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        raise ApiError(
            "Missing bearer token.",
            status=401,
            error_type="authentication_error",
        )

    presented = header.removeprefix("Bearer ").strip()
    for label, token in settings.VOXHELM_BEARER_TOKENS.items():
        if hmac.compare_digest(presented, token):
            return label

    raise ApiError(
        "Invalid bearer token.",
        status=401,
        error_type="authentication_error",
    )


def parse_transcription_request(request: HttpRequest) -> ParsedRequest:
    content_type = (request.content_type or "").lower()
    if content_type.startswith("multipart/form-data"):
        return parse_multipart_request(request)
    if content_type.startswith("application/json"):
        return parse_json_request(request)
    raise ApiError(
        "Unsupported content type. Use multipart/form-data or application/json.",
    )


def parse_multipart_request(request: HttpRequest) -> ParsedRequest:
    upload = request.FILES.get("file")
    if upload is None:
        raise ApiError("Multipart requests must include a file field named 'file'.")
    upload_size = upload.size or 0
    if upload_size > settings.VOXHELM_MAX_UPLOAD_BYTES:
        raise ApiError(
            f"Uploaded file exceeded {settings.VOXHELM_MAX_UPLOAD_MIB} MiB transcription limit."
        )

    request_model = validate_model(request.POST.get("model"))
    response_format = validate_response_format(request.POST.get("response_format"))
    suffix = detect_suffix(upload.name or "", upload.content_type or "")
    if not suffix:
        raise ApiError("Unsupported uploaded media type for transcription.")
    temp_path = write_upload_to_tempfile(upload.chunks(), suffix=suffix)
    return ParsedRequest(
        input_path=temp_path,
        request_model=request_model,
        prompt=optional_string(request.POST.get("prompt")),
        language=optional_string(request.POST.get("language")),
        response_format=response_format,
    )


def parse_json_request(request: HttpRequest) -> ParsedRequest:
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ApiError("Request body was not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ApiError("JSON request body must be an object.")

    source_url = payload.get("url")
    if not isinstance(source_url, str) or not source_url.strip():
        raise ApiError("JSON requests must include a non-empty 'url' field.")

    temp_path = download_allowed_url_to_tempfile(source_url=source_url.strip())
    return ParsedRequest(
        input_path=temp_path,
        request_model=validate_model(payload.get("model")),
        prompt=optional_string(payload.get("prompt")),
        language=optional_string(payload.get("language")),
        response_format=validate_response_format(payload.get("response_format")),
    )


def validate_model(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ApiError("The 'model' field is required.")
    normalized = value.strip()
    if normalized not in settings.VOXHELM_ACCEPTED_MODELS:
        accepted = ", ".join(sorted(settings.VOXHELM_ACCEPTED_MODELS))
        raise ApiError(f"Unsupported model '{normalized}'. Accepted values: {accepted}.")
    return normalized


def validate_response_format(value: object) -> str:
    if value is None:
        return "json"
    if not isinstance(value, str):
        raise ApiError("The 'response_format' field must be a string.")
    normalized = value.strip()
    if normalized not in RESPONSE_FORMATS:
        raise ApiError("Unsupported response_format. Use json, text, verbose_json, or vtt.")
    return normalized


def optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ApiError("Optional request fields must be strings when provided.")
    normalized = value.strip()
    return normalized or None


def write_upload_to_tempfile(chunks: Iterable[bytes], *, suffix: str) -> Path:
    file_handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        for chunk in chunks:
            file_handle.write(chunk)
    finally:
        file_handle.close()
    return Path(file_handle.name)


def download_allowed_url_to_tempfile(*, source_url: str) -> Path:
    parsed = urlparse(source_url)
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise ApiError("URL input must include a hostname.")
    if hostname not in settings.VOXHELM_ALLOWED_URL_HOSTS:
        raise ApiError("URL host is not in the configured allowlist.")
    if parsed.scheme == "https":
        pass
    elif parsed.scheme == "http":
        if hostname not in settings.VOXHELM_TRUSTED_HTTP_HOSTS:
            raise ApiError("Plain HTTP URLs are only allowed for trusted internal hosts.")
    else:
        raise ApiError("Only https URLs are allowed by default.")

    request = Request(
        source_url,
        headers={"User-Agent": "voxhelm/0.1", "Accept": "audio/*;q=1.0,*/*;q=0.1"},
    )
    temp_path: Path | None = None
    try:
        with urlopen(request, timeout=settings.VOXHELM_URL_FETCH_TIMEOUT_SECONDS) as response:
            content_type = (response.headers.get_content_type() or "").lower()
            final_url = response.geturl() or source_url
            suffix = detect_suffix(final_url, content_type)
            if not suffix:
                raise ApiError("Unsupported remote media type for transcription.")
            temp_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=suffix).name)
            total = 0
            with temp_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > settings.VOXHELM_MAX_URL_DOWNLOAD_BYTES:
                        raise ApiError("Remote media exceeded the configured download limit.")
                    handle.write(chunk)
    except HTTPError as exc:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise ApiError(f"URL fetch failed with HTTP {exc.code}.") from exc
    except URLError as exc:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise ApiError(f"URL fetch failed: {exc.reason}.") from exc
    except OSError as exc:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise ApiError(f"URL fetch failed: {exc}.") from exc
    except ApiError:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
    assert temp_path is not None
    return temp_path


def detect_suffix(filename_or_url: str, content_type: str) -> str:
    lower_name = filename_or_url.lower()
    for suffix in SUPPORTED_SUFFIXES:
        if lower_name.endswith(suffix):
            return suffix
    if content_type in CONTENT_TYPE_SUFFIXES:
        return CONTENT_TYPE_SUFFIXES[content_type]
    guessed = mimetypes.guess_extension(content_type, strict=False) or ""
    return guessed if guessed in SUPPORTED_SUFFIXES else ""


def render_response(*, result: TranscriptionResult, response_format: str) -> HttpResponse:
    if response_format == "json":
        return JsonResponse({"text": result.text})
    if response_format == "text":
        return HttpResponse(result.text, content_type="text/plain; charset=utf-8")
    if response_format == "verbose_json":
        return JsonResponse(render_verbose_json(result))
    if response_format == "vtt":
        return HttpResponse(render_vtt(result), content_type="text/vtt; charset=utf-8")
    raise AssertionError(f"Unhandled response format: {response_format}")


def openai_error_response(message: str, *, status: int, error_type: str) -> JsonResponse:
    response = JsonResponse(
        {
            "error": {
                "message": message,
                "type": error_type,
            }
        },
        status=status,
    )
    if status == 401:
        response["WWW-Authenticate"] = "Bearer"
    return response
