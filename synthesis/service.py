from __future__ import annotations

import json
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Protocol

from django.conf import settings

from lane_scheduler import LANE_NON_INTERACTIVE, admit_local_inference

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
    if resolve_backend_name_for_model(backend_name) == "piper":
        return build_piper_backend()
    raise RuntimeError(f"Unsupported TTS backend '{backend_name}'.")


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

    Only Piper contributes voices today; additional backends append their own
    records here and the requested/resolved voice key selects the backend.
    """
    voices: list[InstalledVoice] = list(piper_registry_voices().values())
    return VoiceRegistry(
        voices=tuple(voices),
        default_backend=settings.VOXHELM_TTS_BACKEND,
    )


def piper_registry_voices() -> dict[str, InstalledVoice]:
    return discover_installed_voices(
        voice_dir=settings.VOXHELM_PIPER_VOICE_DIR,
        configured_voices=list(settings.VOXHELM_PIPER_VOICES),
    )


def synthesize_text(text: str, params: SynthesizeParams) -> SynthesisResult:
    with admit_local_inference(params.scheduler_lane):
        backend = get_backend_service(params)
        return backend.synthesize(text, params)


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
