"""Kokoro ONNX text-to-speech backend.

The upstream ``kokoro-onnx`` and ``misaki`` packages cap their Python support
below 3.14, so neither is installable here. This backend therefore talks to
``onnxruntime`` directly and phonemizes with espeak-ng (via ``phonemizer-fork``
and ``espeakng-loader``), mirroring the inference path of the community Martin
reference implementation:

    https://huggingface.co/Godelaune/Kokoro-82M-ONNX-German-Martin
    (onnx-docker/main.py, revision a1cba7fbf0e72fbae38f0a3a48ce0dc8e6077804)

Heavy dependencies (numpy, onnxruntime, phonemizer, espeakng-loader) are
imported lazily so importing this module — and enumerating configured Kokoro
voices in the registry — never requires the optional ``kokoro`` extra. Actual
synthesis raises ``BackendUnavailableError`` with a clear message when the extra
or espeak-ng is missing.
"""

from __future__ import annotations

import tempfile
import wave
import zipfile
import zlib
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, BinaryIO

from synthesis.service import (
    ArtifactIdentity,
    BackendUnavailableError,
    InstalledVoice,
    SynthesisInputError,
    SynthesisResult,
    SynthesizeParams,
    VoiceNotInstalledError,
    piper_artifact_identity,
    require_artifact_generation,
)

if TYPE_CHECKING:
    import numpy as np

# Kokoro tokenizer vocabulary, vendored from thewh1teagle/kokoro-onnx
# (src/kokoro_onnx/config.json, MIT). Maps IPA phoneme characters (with stress
# marks) to integer token ids; characters absent from the map are dropped, so
# this is authoritative for both the official and Martin ONNX exports.
KOKORO_VOCAB: dict[str, int] = {
    ";": 1,
    ":": 2,
    ",": 3,
    ".": 4,
    "!": 5,
    "?": 6,
    "—": 9,
    "…": 10,
    '"': 11,
    "(": 12,
    ")": 13,
    "“": 14,
    "”": 15,
    " ": 16,
    "̃": 17,
    "ʣ": 18,
    "ʥ": 19,
    "ʦ": 20,
    "ʨ": 21,
    "ᵝ": 22,
    "ꭧ": 23,
    "A": 24,
    "I": 25,
    "O": 31,
    "Q": 33,
    "S": 35,
    "T": 36,
    "W": 39,
    "Y": 41,
    "ᵊ": 42,
    "a": 43,
    "b": 44,
    "c": 45,
    "d": 46,
    "e": 47,
    "f": 48,
    "h": 50,
    "i": 51,
    "j": 52,
    "k": 53,
    "l": 54,
    "m": 55,
    "n": 56,
    "o": 57,
    "p": 58,
    "q": 59,
    "r": 60,
    "s": 61,
    "t": 62,
    "u": 63,
    "v": 64,
    "w": 65,
    "x": 66,
    "y": 67,
    "z": 68,
    "ɑ": 69,
    "ɐ": 70,
    "ɒ": 71,
    "æ": 72,
    "β": 75,
    "ɔ": 76,
    "ɕ": 77,
    "ç": 78,
    "ɖ": 80,
    "ð": 81,
    "ʤ": 82,
    "ə": 83,
    "ɚ": 85,
    "ɛ": 86,
    "ɜ": 87,
    "ɟ": 90,
    "ɡ": 92,
    "ɥ": 99,
    "ɨ": 101,
    "ɪ": 102,
    "ʝ": 103,
    "ɯ": 110,
    "ɰ": 111,
    "ŋ": 112,
    "ɳ": 113,
    "ɲ": 114,
    "ɴ": 115,
    "ø": 116,
    "ɸ": 118,
    "θ": 119,
    "œ": 120,
    "ɹ": 123,
    "ɾ": 125,
    "ɻ": 126,
    "ʁ": 128,
    "ɽ": 129,
    "ʂ": 130,
    "ʃ": 131,
    "ʈ": 132,
    "ʧ": 133,
    "ʊ": 135,
    "ʋ": 136,
    "ʌ": 138,
    "ɣ": 139,
    "ɤ": 140,
    "χ": 142,
    "ʎ": 143,
    "ʒ": 147,
    "ʔ": 148,
    "ˈ": 156,
    "ˌ": 157,
    "ː": 158,
    "ʰ": 162,
    "ʲ": 164,
    "↓": 169,
    "→": 171,
    "↗": 172,
    "↘": 173,
    "ᵻ": 177,
}

# Kokoro models accept up to 510 phoneme tokens per inference call (before the
# leading/trailing pad tokens). The style pack is length-indexed by that count.
KOKORO_MAX_TOKENS = 510
KOKORO_SAMPLE_RATE = 24000
# 30 minutes of mono 16-bit PCM: 86,400,000 payload bytes, below RIFF limits.
KOKORO_MAX_OUTPUT_FRAMES = KOKORO_SAMPLE_RATE * 30 * 60
_SPACE_TOKEN = KOKORO_VOCAB[" "]
_MIN_SPEED = 0.5
_MAX_SPEED = 2.0

_EXTRA_HINT = (
    "The 'kokoro' extra is not installed. Install it with "
    "`uv sync --extra kokoro` (onnxruntime, phonemizer-fork, espeakng-loader, numpy)."
)

_CACHE_LIMIT = 8
_CACHE_LOCK = Lock()
_SESSION_CACHE: OrderedDict[ArtifactIdentity, Any] = OrderedDict()
_VOICEPACK_CACHE: OrderedDict[tuple[ArtifactIdentity, str], Any] = OrderedDict()
# Serializes ONNX inference (see synthesize()).
_KOKORO_LOCK = Lock()
# espeak-ng, reached through phonemizer, keeps process-global state (the loaded
# voice/language). Concurrent requests for different languages (e.g. DE and EN)
# can otherwise interleave espeak's language switch and phonemize calls, yielding
# phonemes from the wrong language. This lock serializes BOTH one-time espeak
# initialization and every phonemize call so a request phonemizes atomically.
_PHONEMIZER_LOCK = Lock()
_ESPEAK_READY = False


@dataclass(frozen=True)
class KokoroModelConfig:
    voice_key: str
    model_path: Path
    voicepack_path: Path
    voicepack_key: str
    phoneme_language: str


def kokoro_model_configs(
    *, models: dict[str, dict[str, str]], model_dir: Path
) -> dict[str, KokoroModelConfig]:
    """Resolve the parsed ``VOXHELM_KOKORO_MODELS`` map into config records."""
    configs: dict[str, KokoroModelConfig] = {}
    for voice_key, entry in models.items():
        configs[voice_key] = KokoroModelConfig(
            voice_key=voice_key,
            model_path=_resolve_path(entry["model"], model_dir),
            voicepack_path=_resolve_path(entry["voicepack"], model_dir),
            voicepack_key=entry["voicepack_key"],
            phoneme_language=entry["language"],
        )
    return configs


def _resolve_path(value: str, model_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else model_dir / path


def kokoro_registry_voices(
    *, models: dict[str, dict[str, str]], model_dir: Path
) -> dict[str, InstalledVoice]:
    """Return registry records for every configured Kokoro voice whose files exist.

    A voice missing its model or voicepack file is skipped so ``describe`` never
    advertises a broken voice; an unconfigured deploy simply yields no records.
    """
    installed: dict[str, InstalledVoice] = {}
    for voice_key, config in kokoro_model_configs(models=models, model_dir=model_dir).items():
        if not config.model_path.exists() or not config.voicepack_path.exists():
            continue
        installed[voice_key] = InstalledVoice(
            key=voice_key,
            name=voice_key,
            backend="kokoro",
            languages=_voice_languages(config.phoneme_language),
            artifacts={"model": config.model_path, "voicepack": config.voicepack_path},
        )
    return installed


def _voice_languages(phoneme_language: str) -> tuple[str, ...]:
    code = phoneme_language.strip().lower()
    family = code.split("-", 1)[0]
    return tuple(dict.fromkeys([family, code])) if code else ()


class KokoroBackend:
    def __init__(
        self,
        *,
        models: dict[str, KokoroModelConfig],
        default_voice: str = "",
        espeak_library: str = "",
    ) -> None:
        self.models = models
        self.default_voice = default_voice.strip()
        self.espeak_library = espeak_library.strip()

    def synthesize(self, text: str, params: SynthesizeParams) -> SynthesisResult:
        config = self._resolve_model(params.voice)

        prepared = text
        if _is_german(config.phoneme_language):
            from synthesis.german_text_rules import normalize_german_text

            prepared = normalize_german_text(prepared)

        sentences = split_sentences(prepared)
        token_lists = [self._phonemize_to_tokens(s, config.phoneme_language) for s in sentences]
        chunks = build_token_chunks(token_lists, budget=KOKORO_MAX_TOKENS)
        if not chunks:
            raise SynthesisInputError(
                "Kokoro found no supported speech tokens. Try text in the voice's language."
            )

        session = self._session(config.model_path)
        voicepack = self._voicepack(config.voicepack_path, config.voicepack_key)

        speed = float(min(_MAX_SPEED, max(_MIN_SPEED, params.speed)))
        with _KOKORO_LOCK:
            audio_path, frames = _write_wav(
                _run_inference(session, chunk, voicepack, speed) for chunk in chunks
            )
        duration_seconds = round(frames / KOKORO_SAMPLE_RATE, 3)
        return SynthesisResult(
            audio_path=audio_path,
            backend_name="kokoro",
            model_name="kokoro",
            voice_name=config.voice_key,
            language=params.language or config.phoneme_language,
            sample_rate=KOKORO_SAMPLE_RATE,
            sample_width=2,
            channels=1,
            duration_seconds=duration_seconds,
        )

    def _resolve_model(self, voice: str | None) -> KokoroModelConfig:
        requested = (voice or "").strip()
        if not requested:
            return self._default_model()
        exact = self.models.get(requested)
        if exact is not None:
            return exact
        lowered = {key.lower(): key for key in self.models}
        alias = lowered.get(requested.lower())
        if alias is not None:
            return self.models[alias]
        available = ", ".join(sorted(self.models)) or "none configured"
        raise VoiceNotInstalledError(
            f"Requested Kokoro voice '{requested}' is not configured. "
            f"Available Kokoro voices: {available}."
        )

    def _default_model(self) -> KokoroModelConfig:
        """Resolve the voice used when a request omits one.

        Mirrors ``PiperBackend.resolve_voice``: prefer the configured default
        (``VOXHELM_KOKORO_DEFAULT_VOICE``) when it names a known voice, else fall
        back deterministically to the first configured voice in sorted-key order.
        """
        if not self.models:
            raise RuntimeError(
                "No Kokoro voices are configured. Set VOXHELM_KOKORO_MODELS "
                "to configure at least one voice."
            )
        default_voice = self.default_voice.strip()
        if default_voice and default_voice in self.models:
            return self.models[default_voice]
        return self.models[sorted(self.models)[0]]

    def _phonemize_to_tokens(self, text: str, language: str) -> list[int]:
        # Hold _PHONEMIZER_LOCK across init + phonemize: espeak-ng state is
        # process-global, so this must be atomic per language to avoid races.
        with _PHONEMIZER_LOCK:
            self._ensure_espeak()
            phonemes = _phonemize(text, language)
        return tokenize(phonemes)

    def _ensure_espeak(self) -> None:
        global _ESPEAK_READY
        if _ESPEAK_READY:
            return
        try:
            import espeakng_loader
            from phonemizer.backend.espeak.wrapper import EspeakWrapper
        except ImportError as exc:
            raise BackendUnavailableError(_EXTRA_HINT) from exc

        library = self.espeak_library or espeakng_loader.get_library_path()
        data_path = espeakng_loader.get_data_path()
        try:
            EspeakWrapper.set_data_path(data_path)
            EspeakWrapper.set_library(library)
        except Exception as exc:  # pragma: no cover - defensive
            raise BackendUnavailableError(
                "espeak-ng could not be initialized. Install espeak-ng (e.g. "
                "`brew install espeak-ng`) or set VOXHELM_ESPEAK_LIBRARY to the "
                f"libespeak-ng shared library. Original error: {exc}"
            ) from exc
        _ESPEAK_READY = True

    def _session(self, model_path: str | Path) -> Any:
        with _CACHE_LOCK:
            identity = piper_artifact_identity(Path(model_path))
            cached = _SESSION_CACHE.get(identity)
            if cached is not None:
                _SESSION_CACHE.move_to_end(identity)
                return cached
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise BackendUnavailableError(_EXTRA_HINT) from exc
            session = ort.InferenceSession(str(identity[0]), providers=["CPUExecutionProvider"])
            require_artifact_generation((Path(model_path),), (identity,))
            for existing in list(_SESSION_CACHE):
                if existing[0] == identity[0]:
                    del _SESSION_CACHE[existing]
            _SESSION_CACHE[identity] = session
            if len(_SESSION_CACHE) > _CACHE_LIMIT:
                _SESSION_CACHE.popitem(last=False)
            return session

    def _voicepack(self, voicepack_path: str | Path, voicepack_key: str) -> Any:
        np = _require_numpy()
        with _CACHE_LOCK:
            identity = piper_artifact_identity(Path(voicepack_path))
            cache_key = (identity, voicepack_key)
            cached = _VOICEPACK_CACHE.get(cache_key)
            if cached is not None:
                _VOICEPACK_CACHE.move_to_end(cache_key)
                return cached
            try:
                with identity[0].open("rb") as source:
                    _admit_style_member(source, voicepack_key)
                    pack = np.load(source, allow_pickle=False)
                    with pack:
                        if voicepack_key not in pack:
                            raise BackendUnavailableError(
                                "Voicepack is missing the configured style key."
                            )
                        raw_style = pack[voicepack_key]
                        if (
                            raw_style.ndim != 3
                            or raw_style.shape[0] < 1
                            or raw_style.shape[1:] != (1, 256)
                            or raw_style.dtype.kind not in "fiu"
                        ):
                            raise BackendUnavailableError(
                                "Voicepack must contain numeric styles shaped (rows, 1, 256)."
                            )
                        style = np.array(raw_style, dtype=np.float32, copy=True)
                        if not np.isfinite(style).all():
                            raise BackendUnavailableError("Voicepack styles must be finite.")
            except (
                OSError,
                ValueError,
                TypeError,
                EOFError,
                zipfile.BadZipFile,
                zlib.error,
            ) as exc:
                raise BackendUnavailableError(
                    "Voicepack could not be loaded as numeric styles."
                ) from exc
            require_artifact_generation((Path(voicepack_path),), (identity,))
            for existing in list(_VOICEPACK_CACHE):
                if existing[0][0] == identity[0] and existing[0] != identity:
                    del _VOICEPACK_CACHE[existing]
            _VOICEPACK_CACHE[cache_key] = style
            if len(_VOICEPACK_CACHE) > _CACHE_LIMIT:
                _VOICEPACK_CACHE.popitem(last=False)
            return style


# Standard packs retain 512 rows (510 tokens plus model boundary slots).
_STYLE_MAX_ROWS = 512
_STYLE_HEADER_BYTES = 10_000
_STYLE_MAX_BYTES = _STYLE_MAX_ROWS * 256 * 16


def _admit_style_member(source: BinaryIO, key: str) -> None:
    """Bound the selected expanded NPY member before NumPy allocates its array."""
    np = _require_numpy()
    with zipfile.ZipFile(source) as archive:
        members = [entry for entry in archive.infolist() if entry.filename == f"{key}.npy"]
        if len(members) != 1:
            raise BackendUnavailableError("Voicepack must contain one configured style member.")
        member = members[0]
        if member.file_size > _STYLE_MAX_BYTES + _STYLE_HEADER_BYTES:
            raise BackendUnavailableError("Voicepack style exceeds the expanded byte limit.")
        with archive.open(member) as data:
            version = np.lib.format.read_magic(data)
            if version not in {(1, 0), (2, 0)}:
                raise BackendUnavailableError("Voicepack style uses an unsupported NPY version.")
            length_bytes = 2 if version == (1, 0) else 4
            encoded_length = data.read(length_bytes)
            if (
                len(encoded_length) != length_bytes
                or int.from_bytes(encoded_length, "little") > _STYLE_HEADER_BYTES
            ):
                raise BackendUnavailableError("Voicepack style header exceeds the byte limit.")
            # NumPy checks max_header_size only after reading the header itself.
            data.seek(8)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(
                    data, max_header_size=_STYLE_HEADER_BYTES
                )
            else:
                shape, _, dtype = np.lib.format.read_array_header_2_0(
                    data, max_header_size=_STYLE_HEADER_BYTES
                )
            if (
                len(shape) != 3
                or not 1 <= shape[0] <= _STYLE_MAX_ROWS
                or shape[1:] != (1, 256)
                or dtype.kind not in "fiu"
                or dtype.itemsize > 16
            ):
                raise BackendUnavailableError(
                    "Voicepack styles must be numeric with shape (1..512, 1, 256)."
                )
            size = shape[0] * 256 * dtype.itemsize
            if size > _STYLE_MAX_BYTES or member.file_size != data.tell() + size:
                raise BackendUnavailableError("Voicepack style payload size is invalid.")
    source.seek(0)


def split_sentences(text: str) -> list[str]:
    """Split text at real sentence boundaries (. ! ? and line breaks).

    Mirrors the Martin reference implementation: colons and quotation marks are
    deliberately not treated as separators.
    """
    import re

    text = re.sub(r"\n\s*\n", "\n\n", text)
    segments = re.split(r"(?<=[.!?])\s+(?=\S)", text)
    result: list[str] = []
    for segment in segments:
        for line in segment.split("\n"):
            stripped = line.strip()
            # Keep every segment carrying non-whitespace content; only genuinely
            # empty/whitespace lines are dropped. A one-character sentence such as
            # "I" is real speech and must never be silently discarded.
            if stripped:
                result.append(stripped)
    return result


def tokenize(phonemes: str) -> list[int]:
    return [KOKORO_VOCAB[char] for char in phonemes if char in KOKORO_VOCAB]


def build_token_chunks(token_lists: list[list[int]], *, budget: int) -> list[list[int]]:
    """Pack per-sentence token lists into chunks that stay within ``budget``.

    Sentences are joined with a single space token; a sentence longer than the
    budget on its own is hard-split into budget-sized pieces as a last resort.
    """
    chunks: list[list[int]] = []
    current: list[int] = []

    for tokens in token_lists:
        if not tokens:
            continue
        if len(tokens) > budget:
            if current:
                chunks.append(current)
                current = []
            for start in range(0, len(tokens), budget):
                chunks.append(tokens[start : start + budget])
            continue

        separator = 1 if current else 0
        if current and len(current) + separator + len(tokens) > budget:
            chunks.append(current)
            current = []
            separator = 0
        if separator:
            current = current + [_SPACE_TOKEN] + tokens
        else:
            current = list(tokens)

    if current:
        chunks.append(current)
    return chunks


def select_style_row(voicepack: np.ndarray, token_count: int) -> np.ndarray:
    """Select the style row for a chunk by its token count, clamped to the pack.

    Follows the Martin reference exactly: the row is chosen by the token count
    BEFORE the boundary/pad tokens are added, with an index clamp guarding the
    final row.
    """
    max_index = voicepack.shape[0] - 1
    index = max(0, min(token_count, max_index))
    return voicepack[index]


def _run_inference(
    session: Any, tokens: list[int], voicepack: np.ndarray, speed: float
) -> np.ndarray:
    np = _require_numpy()
    style = select_style_row(voicepack, len(tokens))
    padded = np.array([[0, *tokens, 0]], dtype=np.int64)
    inputs = {
        "tokens": padded,
        "style": np.asarray(style, dtype=np.float32),
        "speed": np.ones(1, dtype=np.float32) * speed,
    }
    outputs = session.run(None, inputs)
    if not outputs:
        raise RuntimeError("Kokoro returned no audio output.")
    audio = np.asarray(outputs[0])
    # ONNX exports return a mono vector or a single batch of mono samples.
    if (
        audio.dtype.kind not in "fiu"
        or audio.size == 0
        or not (audio.ndim == 1 or (audio.ndim == 2 and audio.shape[0] == 1))
    ):
        raise RuntimeError("Kokoro audio must be a nonempty mono vector or single mono batch.")
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not np.isfinite(samples).all():
        raise RuntimeError("Kokoro audio samples must be finite.")
    return samples


def _write_wav(audio_parts: Iterable[np.ndarray]) -> tuple[Path, int]:
    np = _require_numpy()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temporary:
        audio_path = Path(temporary.name)
    frames = 0
    try:
        with wave.open(str(audio_path), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(KOKORO_SAMPLE_RATE)
            for audio in audio_parts:
                if frames + len(audio) > KOKORO_MAX_OUTPUT_FRAMES:
                    raise RuntimeError("Kokoro audio exceeded the 30-minute output limit.")
                clipped = np.clip(audio, -1.0, 1.0)
                pcm = (clipped * 32767.0).astype("<i2")
                writer.writeframes(pcm.tobytes())
                frames += len(audio)
                del audio, clipped, pcm
            if not frames:
                raise RuntimeError("Kokoro returned no audio samples.")
    except BaseException:
        # The caller never receives the path on inference, cancellation or write failure.
        audio_path.unlink(missing_ok=True)
        raise
    return audio_path, frames


def _phonemize(text: str, language: str) -> str:
    try:
        import phonemizer
    except ImportError as exc:
        raise BackendUnavailableError(_EXTRA_HINT) from exc
    phonemes = phonemizer.phonemize(text, language, preserve_punctuation=True, with_stress=True)
    return "".join(char for char in phonemes if char in KOKORO_VOCAB).strip()


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise BackendUnavailableError(_EXTRA_HINT) from exc
    return np


def _is_german(phoneme_language: str) -> bool:
    return phoneme_language.strip().lower().split("-", 1)[0] == "de"


def reset_caches() -> None:
    """Clear session/voicepack caches. Intended for tests."""
    with _CACHE_LOCK:
        _SESSION_CACHE.clear()
        _VOICEPACK_CACHE.clear()
