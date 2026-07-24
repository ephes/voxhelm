from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import wave
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
from typing import Any, NamedTuple, Protocol

from django.conf import settings

from lane_scheduler import LANE_NON_INTERACTIVE, admit_local_inference

_LOGGER = logging.getLogger(__name__)

AUTO_BACKEND_MODEL_NAMES = {"auto", "piper", "tts-1", "tts-1-hd"}
AUDIO_OUTPUT_FORMATS = {"wav", "mp3", "ogg"}
MIN_TTS_SPEED = 0.25
MAX_TTS_SPEED = 4.0
CONTENT_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "ogg": "audio/ogg",
}


@dataclass(frozen=True)
class SynthesizeParams:
    request_model: str
    voice: str | None
    language: str | None
    speed: float
    scheduler_lane: str = LANE_NON_INTERACTIVE
    # Per-request switch for automatic language routing. Defaults to True so
    # callers that do not opt out (Wyoming, batch) always route when routing is
    # enabled; /v1/audio/speech sets it from the request's `routing` field.
    routing: bool = True


@dataclass(frozen=True)
class InstalledVoice:
    key: str
    name: str
    backend: str
    languages: tuple[str, ...]
    artifacts: dict[str, Path]
    speakers: tuple[str, ...] = ()

    @property
    def model_path(self) -> Path:
        return self.artifacts["model"]

    @property
    def config_path(self) -> Path:
        return self.artifacts["config"]


@dataclass(frozen=True)
class VoiceRegistry:
    """Backend-agnostic view of every installed voice across all backends."""

    voices: tuple[InstalledVoice, ...]
    default_backend: str

    def by_key(self) -> dict[str, InstalledVoice]:
        return {voice.key: voice for voice in self.voices}

    def lookup(self, voice_key: str | None) -> InstalledVoice | None:
        key = (voice_key or "").strip()
        if not key:
            return None
        by_key = self.by_key()
        exact = by_key.get(key)
        if exact is not None:
            return exact
        lowered = {existing.lower(): existing for existing in by_key}
        alias = lowered.get(key.lower())
        return by_key[alias] if alias is not None else None

    def resolve_backend(self, *, voice: str | None, request_model: str) -> str:
        record = self.lookup(voice)
        if record is not None:
            return record.backend
        if request_model in AUTO_BACKEND_MODEL_NAMES:
            return self.default_backend
        return request_model


@dataclass(frozen=True)
class SynthesisResult:
    audio_path: Path
    backend_name: str
    model_name: str
    voice_name: str
    language: str | None
    sample_rate: int
    sample_width: int
    channels: int
    duration_seconds: float


@dataclass(frozen=True)
class ExportedAudio:
    path: Path
    format_name: str
    content_type: str


class BackendUnavailableError(RuntimeError):
    """Raised when a synthesis backend is not available on the current host."""


class BackendProtocol(Protocol):
    def synthesize(self, text: str, params: SynthesizeParams) -> SynthesisResult: ...


_PIPER_LOCK = Lock()
_PIPER_VOICE_CACHE: dict[str, object] = {}


class PiperBackend:
    def __init__(
        self,
        *,
        voice_dir: Path,
        configured_voices: list[str],
        default_voice: str,
        language_voices: dict[str, str],
    ) -> None:
        self.voice_dir = voice_dir
        self.configured_voices = configured_voices
        self.default_voice = default_voice
        self.language_voices = {
            normalize_language_key(language): voice
            for language, voice in language_voices.items()
            if voice.strip()
        }

    def synthesize(self, text: str, params: SynthesizeParams) -> SynthesisResult:
        resolved_voice = self.resolve_voice(voice=params.voice, language=params.language)
        voice = load_piper_voice(resolved_voice)

        try:
            from piper import SynthesisConfig
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise BackendUnavailableError(
                "piper-tts is not installed. Install the project dependencies first."
            ) from exc

        temp_wav = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name)
        with _PIPER_LOCK:
            synthesis_config = SynthesisConfig()
            if params.speed != 1.0:
                synthesis_config.length_scale = max(
                    1.0 / MAX_TTS_SPEED,
                    min(1.0 / MIN_TTS_SPEED, 1.0 / params.speed),
                )

            with wave.open(str(temp_wav), "wb") as wav_writer:
                voice.synthesize_wav(text, wav_writer, synthesis_config)

        with wave.open(str(temp_wav), "rb") as wav_reader:
            frame_rate = wav_reader.getframerate()
            frame_width = wav_reader.getsampwidth()
            channels = wav_reader.getnchannels()
            frame_count = wav_reader.getnframes()

        duration_seconds = round(frame_count / frame_rate, 3) if frame_rate else 0.0
        return SynthesisResult(
            audio_path=temp_wav,
            backend_name="piper",
            model_name="piper",
            voice_name=resolved_voice.key,
            language=resolve_result_language(resolved_voice, params.language),
            sample_rate=frame_rate,
            sample_width=frame_width,
            channels=channels,
            duration_seconds=duration_seconds,
        )

    def resolve_voice(self, *, voice: str | None, language: str | None) -> InstalledVoice:
        installed = discover_installed_voices(
            voice_dir=self.voice_dir,
            configured_voices=self.configured_voices,
        )
        if not installed:
            raise BackendUnavailableError(
                f"No Piper voices were found in '{self.voice_dir}'."
            )

        requested_voice = (voice or "").strip()
        if requested_voice:
            exact = installed.get(requested_voice)
            if exact is not None:
                return exact
            lowered = {item.lower(): item for item in installed}
            alias = lowered.get(requested_voice.lower())
            if alias is not None:
                return installed[alias]
            resolved_language = self.language_voices.get(normalize_language_key(requested_voice))
            if resolved_language and resolved_language in installed:
                return installed[resolved_language]
            raise RuntimeError(
                f"Requested voice '{requested_voice}' is not installed. "
                f"Available voices: {', '.join(sorted(installed))}."
            )

        if language:
            mapped_voice = self.language_voices.get(normalize_language_key(language))
            if mapped_voice and mapped_voice in installed:
                return installed[mapped_voice]

        default_voice = self.default_voice.strip()
        if default_voice and default_voice in installed:
            return installed[default_voice]

        return next(iter(installed.values()))


def get_backend_service(params: SynthesizeParams | None = None) -> BackendProtocol:
    if params is None:
        return build_backend_service(settings.VOXHELM_TTS_BACKEND)
    backend_name = build_voice_registry().resolve_backend(
        voice=params.voice,
        request_model=params.request_model,
    )
    return build_backend_service(backend_name)


def build_backend_service(backend_name: str) -> BackendProtocol:
    resolved = resolve_backend_name_for_model(backend_name)
    if resolved == "piper":
        return build_piper_backend()
    if resolved == "kokoro":
        return build_kokoro_backend()
    raise RuntimeError(f"Unsupported TTS backend '{backend_name}'.")


def build_kokoro_backend() -> BackendProtocol:
    from synthesis.kokoro import KokoroBackend, kokoro_model_configs

    return KokoroBackend(
        models=kokoro_model_configs(
            models=dict(settings.VOXHELM_KOKORO_MODELS),
            model_dir=settings.VOXHELM_KOKORO_MODEL_DIR,
        ),
        default_voice=settings.VOXHELM_KOKORO_DEFAULT_VOICE,
        espeak_library=settings.VOXHELM_ESPEAK_LIBRARY,
    )


def build_piper_backend() -> PiperBackend:
    return PiperBackend(
        voice_dir=settings.VOXHELM_PIPER_VOICE_DIR,
        configured_voices=list(settings.VOXHELM_PIPER_VOICES),
        default_voice=settings.VOXHELM_PIPER_DEFAULT_VOICE,
        language_voices=dict(settings.VOXHELM_PIPER_LANGUAGE_VOICES),
    )


def resolve_backend_name_for_model(request_model: str) -> str:
    if request_model in AUTO_BACKEND_MODEL_NAMES:
        return settings.VOXHELM_TTS_BACKEND
    return request_model


def build_voice_registry() -> VoiceRegistry:
    """Collect installed voices from every backend into one dispatch table.

    Each backend appends its own records here and the requested/resolved voice
    key selects the backend. A deploy with no Kokoro models configured simply
    contributes no Kokoro voices, leaving Piper behavior untouched.
    """
    voices: list[InstalledVoice] = list(piper_registry_voices().values())
    voices.extend(kokoro_registry_voices().values())
    return VoiceRegistry(
        voices=tuple(voices),
        default_backend=settings.VOXHELM_TTS_BACKEND,
    )


def kokoro_registry_voices() -> dict[str, InstalledVoice]:
    if not settings.VOXHELM_KOKORO_MODELS:
        return {}
    from synthesis.kokoro import kokoro_registry_voices as _kokoro_registry_voices

    return _kokoro_registry_voices(
        models=dict(settings.VOXHELM_KOKORO_MODELS),
        model_dir=settings.VOXHELM_KOKORO_MODEL_DIR,
    )


def piper_registry_voices() -> dict[str, InstalledVoice]:
    return discover_installed_voices(
        voice_dir=settings.VOXHELM_PIPER_VOICE_DIR,
        configured_voices=list(settings.VOXHELM_PIPER_VOICES),
    )


def synthesize_text(text: str, params: SynthesizeParams) -> SynthesisResult:
    with admit_local_inference(params.scheduler_lane):
        routed = apply_language_routing(text, params)
        backend = get_backend_service(routed)
        return backend.synthesize(text, routed)


# Deterministic routing floor (spec §Architecture / Language routing). Routing
# applies only when ALL criteria hold; any failure keeps the pinned voice/language.
ROUTING_MIN_CHARS = 20
ROUTING_MIN_CONFIDENCE = 0.90

# lingua is restricted to these two languages; the detector is a lazy singleton so
# importing this module never requires the optional `routing` extra.
_ROUTING_DETECTOR: Any = None
_ROUTING_DETECTOR_LOCK = Lock()


class RoutingDetection(NamedTuple):
    language: str
    confidence: float


def _routing_detector() -> Any:
    global _ROUTING_DETECTOR
    if _ROUTING_DETECTOR is None:
        with _ROUTING_DETECTOR_LOCK:
            if _ROUTING_DETECTOR is None:
                from lingua import Language, LanguageDetectorBuilder

                _ROUTING_DETECTOR = LanguageDetectorBuilder.from_languages(
                    Language.GERMAN, Language.ENGLISH
                ).build()
    return _ROUTING_DETECTOR


def detect_routing_language(text: str) -> RoutingDetection | None:
    """Detect ``text``'s language with the de/en-restricted lingua detector.

    Returns the top language's ISO-639-1 code (``de``/``en``) and its confidence
    via ``compute_language_confidence_values`` so the routing floor can gate on
    the confidence, or ``None`` when the detector yields no candidate.
    """
    values = _routing_detector().compute_language_confidence_values(text)
    if not values:
        return None
    top = values[0]
    return RoutingDetection(
        language=top.language.iso_code_639_1.name.lower(),
        confidence=float(top.value),
    )


def apply_language_routing(text: str, params: SynthesizeParams) -> SynthesizeParams:
    """Route ``params`` to the detected language's voice, or return it unchanged.

    Routing applies only when it is enabled, not bypassed for this request, and
    every deterministic floor criterion holds: (a) stripped text >= 20 chars,
    (b) top-language confidence >= 0.90, (c) the detected language is mapped in
    VOXHELM_TTS_LANGUAGE_VOICES, and (d) the mapped voice exists in the registry
    ((d) failing additionally logs a warning naming the voice). When it applies,
    the mapped voice AND the detected language become effective end-to-end, and an
    explicitly pinned ``request_model`` is released to ``auto`` so registry backend
    dispatch follows the mapped voice's backend rather than the pinned model name.
    Any failure keeps the pinned voice/language; this never raises for routing reasons.
    """
    if not params.routing or not settings.VOXHELM_TTS_LANGUAGE_ROUTING:
        return params
    language_voices = {
        normalize_language_key(language): voice
        for language, voice in settings.VOXHELM_TTS_LANGUAGE_VOICES.items()
        if voice.strip()
    }
    if not language_voices:
        return params

    stripped = text.strip()
    if len(stripped) < ROUTING_MIN_CHARS:  # (a)
        return params

    try:
        detection = detect_routing_language(stripped)
    except Exception:  # pragma: no cover - defensive; routing must never raise
        _LOGGER.warning("Language routing detection failed; keeping pinned voice.", exc_info=True)
        return params
    if detection is None or detection.confidence < ROUTING_MIN_CONFIDENCE:  # (b)
        return params

    mapped_voice = language_voices.get(normalize_language_key(detection.language))  # (c)
    if not mapped_voice:
        return params

    try:
        registered = build_voice_registry().lookup(mapped_voice) is not None  # (d)
    except Exception:
        # Registry discovery reads voice artifacts from disk and can raise on
        # unreadable/malformed files; routing must never fail the request.
        _LOGGER.warning(
            "Language routing registry lookup failed; keeping pinned voice.",
            exc_info=True,
        )
        return params
    if not registered:
        _LOGGER.warning(
            "Language routing mapped language '%s' to voice '%s', which is not in "
            "the voice registry; keeping the pinned voice.",
            detection.language,
            mapped_voice,
        )
        return params

    return replace(
        params,
        voice=mapped_voice,
        language=detection.language,
        request_model="auto",
    )


def export_audio(result: SynthesisResult, *, output_format: str) -> ExportedAudio:
    normalized = output_format.strip().lower()
    if normalized not in AUDIO_OUTPUT_FORMATS:
        accepted = ", ".join(sorted(AUDIO_OUTPUT_FORMATS))
        raise RuntimeError(f"Unsupported audio format '{normalized}'. Accepted values: {accepted}.")

    if normalized == "wav":
        return ExportedAudio(
            path=result.audio_path,
            format_name="wav",
            content_type=CONTENT_TYPES["wav"],
        )

    target_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=f".{normalized}").name)
    args = [
        settings.VOXHELM_FFMPEG_BIN,
        "-y",
        "-i",
        str(result.audio_path),
    ]
    if normalized == "mp3":
        args.extend(["-vn", "-codec:a", "libmp3lame", "-q:a", "2"])
    elif normalized == "ogg":
        args.extend(["-vn", "-codec:a", "libvorbis", "-q:a", "4"])
    args.append(str(target_path))

    completed = subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        target_path.unlink(missing_ok=True)
        detail = "\n".join(
            part.strip()
            for part in (completed.stderr, completed.stdout)
            if isinstance(part, str) and part.strip()
        )
        raise RuntimeError(f"Audio conversion failed: {detail or 'ffmpeg exited unsuccessfully.'}")

    return ExportedAudio(
        path=target_path,
        format_name=normalized,
        content_type=CONTENT_TYPES[normalized],
    )


def discover_installed_voices(
    *, voice_dir: Path, configured_voices: list[str]
) -> dict[str, InstalledVoice]:
    voice_dir.mkdir(parents=True, exist_ok=True)
    voice_names = list(dict.fromkeys(configured_voices))
    if not voice_names:
        voice_names = sorted(path.stem for path in voice_dir.glob("*.onnx"))

    installed: dict[str, InstalledVoice] = {}
    for voice_name in voice_names:
        model_path = voice_dir / f"{voice_name}.onnx"
        config_path = voice_dir / f"{voice_name}.onnx.json"
        if not model_path.exists() or not config_path.exists():
            continue
        installed[voice_name] = build_voice_metadata(
            voice_name=voice_name,
            model_path=model_path,
            config_path=config_path,
        )
    return installed


def build_voice_metadata(*, voice_name: str, model_path: Path, config_path: Path) -> InstalledVoice:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    languages = parse_voice_languages(voice_name)
    speaker_id_map = config.get("speaker_id_map", {})
    speakers = tuple(sorted(str(name) for name in speaker_id_map))
    return InstalledVoice(
        key=voice_name,
        name=voice_name,
        backend="piper",
        languages=languages,
        artifacts={"model": model_path, "config": config_path},
        speakers=speakers,
    )


def parse_voice_languages(voice_name: str) -> tuple[str, ...]:
    language_code = voice_name.split("-", 1)[0]
    family = language_code.split("_", 1)[0]
    return tuple(dict.fromkeys([family, language_code]))


def normalize_language_key(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def resolve_result_language(voice: InstalledVoice, requested_language: str | None) -> str | None:
    if requested_language:
        return requested_language
    return voice.languages[0] if voice.languages else None


def load_piper_voice(voice: InstalledVoice):
    try:
        from piper import PiperVoice
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise BackendUnavailableError(
            "piper-tts is not installed. Install the project dependencies first."
        ) from exc

    cached = _PIPER_VOICE_CACHE.get(voice.key)
    if cached is not None:
        return cached

    loaded_voice = PiperVoice.load(voice.model_path, config_path=voice.config_path)
    _PIPER_VOICE_CACHE[voice.key] = loaded_voice
    return loaded_voice


def cleanup_paths(*paths: Path) -> None:
    for path in paths:
        path.unlink(missing_ok=True)
