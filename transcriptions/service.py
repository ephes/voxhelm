from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from django.conf import settings

AUTO_BACKEND_MODEL_NAMES = {"auto", "gpt-4o-mini-transcribe", "whisper-1"}
LEADING_FILLER_WORDS = {
    "de": ("okay", "ok", "und", "also", "äh", "ah", "hm", "hmm"),
    "en": ("okay", "ok", "and", "so", "well"),
}
LEADING_FILLER_PATTERNS = {
    language: re.compile(
        r"^(?:"
        + "|".join(re.escape(word) for word in words)
        + r")(?:[\s,.;:!?\-…]+|$)",
        re.IGNORECASE,
    )
    for language, words in LEADING_FILLER_WORDS.items()
}
GERMAN_DISCOURSE_PARTICLE_PATTERN = re.compile(
    r"^(?P<prefix>(?:wie|was)\s+ist)\s+(?P<particle>denn|eigentlich|mal)\s+",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TranscribeParams:
    request_model: str
    prompt: str | None
    language: str | None


@dataclass(frozen=True)
class TranscriptionSegment:
    id: int
    start: float
    end: float
    text: str

    def as_verbose_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "seek": int(self.start * 100),
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str | None
    segments: list[TranscriptionSegment]
    backend_name: str = ""
    model_name: str = ""


@dataclass(frozen=True)
class BackendInvocation:
    name: str
    service: BackendProtocol


class BackendUnavailableError(RuntimeError):
    """Raised when a backend is not available on the current host."""


class BackendProtocol(Protocol):
    def transcribe(self, audio_path: Path, params: TranscribeParams) -> TranscriptionResult: ...


def normalize_interactive_transcript(text: str, *, language: str | None) -> str:
    normalized = " ".join(text.split()).strip()
    if not normalized:
        return normalized

    language_key = normalize_language_code(language)
    pattern = LEADING_FILLER_PATTERNS.get(language_key)
    if pattern is None:
        return normalized

    candidate = normalized
    while True:
        stripped = pattern.sub("", candidate, count=1).lstrip(" ,.;:!?-…")
        if not stripped or stripped == candidate:
            candidate = normalized if not stripped else candidate
            break
        candidate = stripped

    if language_key == "de":
        candidate = GERMAN_DISCOURSE_PARTICLE_PATTERN.sub(r"\g<prefix> ", candidate)

    return candidate


def normalize_language_code(language: str | None) -> str:
    if not language:
        return ""
    return language.strip().lower().replace("_", "-").split("-", 1)[0]


class MlxWhisperBackend:
    def __init__(self, *, model_name: str) -> None:
        self.model_name = model_name

    def transcribe(self, audio_path: Path, params: TranscribeParams) -> TranscriptionResult:
        try:
            import mlx_whisper
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise BackendUnavailableError(
                "mlx-whisper is not installed. Install the project dependencies first."
            ) from exc

        payload = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=self.model_name,
            word_timestamps=False,
            initial_prompt=params.prompt,
            language=params.language,
        )
        return normalize_transcription_payload(
            payload,
            backend_name="mlx-whisper",
            model_name=self.model_name,
        )


class WhisperCppBackend:
    def __init__(self, *, binary_path: str, model_name: str, processors: int) -> None:
        self.binary_path = binary_path
        self.model_name = model_name
        self.processors = processors

    def transcribe(self, audio_path: Path, params: TranscribeParams) -> TranscriptionResult:
        executable = resolve_whispercpp_binary(self.binary_path)
        model_path = resolve_whispercpp_model_path(self.model_name)

        with tempfile.TemporaryDirectory(prefix="voxhelm-whispercpp-") as temp_dir:
            output_base = Path(temp_dir) / "transcript"
            args = [
                executable,
                "-m",
                str(model_path),
                "-f",
                str(audio_path),
                "-oj",
                "-of",
                str(output_base),
                "-p",
                str(self.processors),
                "-l",
                params.language or "auto",
                "-np",
            ]
            if params.prompt:
                args.extend(["--prompt", params.prompt])

            completed = subprocess.run(
                args,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                detail = "\n".join(
                    part.strip()
                    for part in (completed.stderr, completed.stdout)
                    if isinstance(part, str) and part.strip()
                )
                if detail:
                    raise RuntimeError(f"whisper.cpp transcription failed: {detail}")
                raise RuntimeError("whisper.cpp transcription failed.")

            json_path = output_base.with_suffix(".json")
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        return normalize_whispercpp_payload(payload, model_name=self.model_name)


_TRANSCRIPTION_LOCK = Lock()


def normalize_transcription_payload(
    payload: dict[str, Any],
    *,
    backend_name: str = "",
    model_name: str = "",
) -> TranscriptionResult:
    text = str(payload.get("text", "")).strip()
    language_value = payload.get("language")
    language = str(language_value).strip() if isinstance(language_value, str) else None
    raw_segments = payload.get("segments")
    segments: list[TranscriptionSegment] = []

    if isinstance(raw_segments, list):
        for index, raw_segment in enumerate(raw_segments):
            if not isinstance(raw_segment, dict):
                continue
            segment_text = str(raw_segment.get("text", "")).strip()
            if not segment_text:
                continue
            segments.append(
                TranscriptionSegment(
                    id=int(raw_segment.get("id", index)),
                    start=float(raw_segment.get("start", 0.0)),
                    end=float(raw_segment.get("end") or raw_segment.get("start") or 0.0),
                    text=segment_text,
                )
            )

    if not segments and text:
        segments = [TranscriptionSegment(id=0, start=0.0, end=0.0, text=text)]

    return TranscriptionResult(
        text=text,
        language=language,
        segments=segments,
        backend_name=backend_name,
        model_name=model_name,
    )


def normalize_whispercpp_payload(
    payload: dict[str, Any],
    *,
    model_name: str,
) -> TranscriptionResult:
    raw_segments = payload.get("transcription")
    segments: list[TranscriptionSegment] = []

    if isinstance(raw_segments, list):
        for index, raw_segment in enumerate(raw_segments):
            if not isinstance(raw_segment, dict):
                continue
            timestamps = raw_segment.get("timestamps")
            if not isinstance(timestamps, dict):
                continue
            segment_text = str(raw_segment.get("text", "")).strip()
            if not segment_text:
                continue
            segments.append(
                TranscriptionSegment(
                    id=index,
                    start=timestamp_to_seconds(str(timestamps.get("from", "00:00:00,000"))),
                    end=timestamp_to_seconds(str(timestamps.get("to", "00:00:00,000"))),
                    text=segment_text,
                )
            )

    text = " ".join(segment.text for segment in segments).strip()
    return TranscriptionResult(
        text=text,
        language=str(payload.get("result", {}).get("language", "")).strip() or None,
        segments=segments,
        backend_name="whisper.cpp",
        model_name=model_name,
    )


def timestamp_to_seconds(timestamp: str) -> float:
    normalized = timestamp.replace(".", ",")
    try:
        hours, minutes, seconds = normalized.split(":")
        whole_seconds, milliseconds = seconds.split(",")
        return (
            float(hours) * 3600
            + float(minutes) * 60
            + float(whole_seconds)
            + float(milliseconds) / 1000
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid whisper.cpp timestamp '{timestamp}'.") from exc


def render_verbose_json(result: TranscriptionResult) -> dict[str, Any]:
    return {
        "task": "transcribe",
        "language": result.language,
        "text": result.text,
        "segments": [segment.as_verbose_json() for segment in result.segments],
    }


def render_vtt(result: TranscriptionResult) -> str:
    lines = ["WEBVTT", ""]
    for segment in result.segments:
        timestamp_line = (
            f"{format_vtt_timestamp(segment.start)} --> {format_vtt_timestamp(segment.end)}"
        )
        lines.append(timestamp_line)
        lines.append(segment.text)
        lines.append("")
    if len(lines) == 2 and result.text:
        lines.extend(["00:00:00.000 --> 00:00:00.000", result.text, ""])
    return "\n".join(lines).rstrip() + "\n"


def format_vtt_timestamp(seconds: float) -> str:
    milliseconds = max(int(round(seconds * 1000)), 0)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"


def get_backend_service() -> BackendProtocol:
    return build_backend_service(
        backend_name=settings.VOXHELM_STT_BACKEND,
        model_name=resolve_model_name_for_backend(
            request_model="auto",
            backend_name=settings.VOXHELM_STT_BACKEND,
        ),
    )


def get_backend_services_for_model(request_model: str) -> list[BackendInvocation]:
    primary_backend = resolve_backend_name_for_model(request_model)
    services = [
        BackendInvocation(
            primary_backend,
            service_for_backend_name(primary_backend, request_model=request_model),
        )
    ]
    fallback_backend = settings.VOXHELM_STT_FALLBACK_BACKEND.strip()
    if (
        is_auto_backend_model_request(request_model)
        and fallback_backend
        and fallback_backend != primary_backend
    ):
        services.append(
            BackendInvocation(
                fallback_backend,
                service_for_backend_name(fallback_backend, request_model=request_model),
            )
        )
    return services


def service_for_backend_name(backend_name: str, *, request_model: str) -> BackendProtocol:
    if (
        backend_name == settings.VOXHELM_STT_BACKEND
        and is_auto_backend_model_request(request_model)
    ):
        return get_backend_service()
    return build_backend_service(
        backend_name=backend_name,
        model_name=resolve_model_name_for_backend(
            request_model=request_model,
            backend_name=backend_name,
        ),
    )


def build_backend_service(*, backend_name: str, model_name: str) -> BackendProtocol:
    if backend_name == "mlx":
        return MlxWhisperBackend(model_name=model_name)
    if backend_name == "whispercpp":
        return WhisperCppBackend(
            binary_path=settings.VOXHELM_WHISPERCPP_BIN,
            model_name=model_name,
            processors=settings.VOXHELM_WHISPERCPP_PROCESSORS,
        )
    raise RuntimeError(f"Unsupported STT backend '{backend_name}'.")


def resolve_backend_name_for_model(request_model: str) -> str:
    if is_auto_backend_model_request(request_model):
        return settings.VOXHELM_STT_BACKEND
    if request_model == settings.VOXHELM_WHISPERCPP_MODEL:
        return "whispercpp"
    if request_model == settings.VOXHELM_MLX_MODEL:
        return "mlx"
    return settings.VOXHELM_STT_BACKEND


def resolve_model_name_for_backend(*, request_model: str, backend_name: str) -> str:
    if not is_auto_backend_model_request(request_model):
        return request_model
    if backend_name == "whispercpp":
        return settings.VOXHELM_WHISPERCPP_MODEL
    if backend_name == "mlx":
        return settings.VOXHELM_MLX_MODEL
    raise RuntimeError(f"Unsupported STT backend '{backend_name}'.")


def is_auto_backend_model_request(request_model: str) -> bool:
    return request_model in AUTO_BACKEND_MODEL_NAMES


def resolve_whispercpp_binary(binary_path: str) -> str:
    if "/" in binary_path:
        candidate = Path(binary_path)
        if not candidate.exists():
            raise BackendUnavailableError(
                f"whisper.cpp binary was not found at '{binary_path}'."
            )
        return str(candidate)
    resolved = shutil.which(binary_path)
    if resolved is None:
        raise BackendUnavailableError(f"whisper.cpp binary '{binary_path}' was not found in PATH.")
    return resolved


def resolve_whispercpp_model_path(model_name: str) -> Path:
    configured = Path(model_name).expanduser()
    if configured.is_absolute() or configured.parent != Path("."):
        candidate = configured
    else:
        candidate = settings.VOXHELM_MODEL_CACHE_DIR / model_name
    if not candidate.exists():
        raise BackendUnavailableError(
            "whisper.cpp model "
            f"'{candidate}' is missing. Deploy the model before selecting this backend."
        )
    return candidate


def transcribe_audio(audio_path: Path, params: TranscribeParams) -> TranscriptionResult:
    # Local STT backends are not safe to run concurrently inside this long-lived process.
    with _TRANSCRIPTION_LOCK:
        unavailable_errors: list[str] = []
        for invocation in get_backend_services_for_model(params.request_model):
            try:
                return invocation.service.transcribe(audio_path, params)
            except BackendUnavailableError as exc:
                unavailable_errors.append(f"{invocation.name}: {exc}")

        joined = "; ".join(unavailable_errors)
        raise RuntimeError(f"No configured STT backend is available. {joined}")


def serialize_health() -> str:
    return json.dumps({"status": "ok"})
