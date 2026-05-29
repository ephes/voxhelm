"""Known-speaker voiceprint postprocessor.

Anonymous pyannote diarization clusters voices but cannot reliably recover a
specific known speaker in mono, similar-voice, crosstalk-heavy podcast audio.
Diarization-quality research on real Python Podcast material showed that
classifying transcript segments directly against contributor reference
voiceprints is much stronger: about 95% accuracy over all segments and 99%+
over segments long enough for a stable embedding, compared with about 77-87%
for tuned anonymous pyannote.

This module implements that path. It computes one normalized centroid per known
speaker from reviewed reference audio, embeds each mastered-mono transcript
segment, and assigns the nearest speaker by cosine similarity. It emits the
top candidate, full candidate list, confidence, margin, an uncertainty flag,
and the raw anonymous diarization label so the consumer can review uncertain
segments instead of silently trusting them.

The heavy embedding model is loaded lazily through ``importlib`` and hidden
behind :class:`KnownSpeakerEmbeddingBackendProtocol`, mirroring the diarization
backend, so the classification logic stays testable without torch/pyannote.
"""

from __future__ import annotations

import importlib
import math
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, Protocol, Sequence

from django.conf import settings

from lane_scheduler import LANE_NON_INTERACTIVE, admit_local_inference
from transcriptions.diarization import (
    SpeakerTurn,
    choose_speaker_for_segment,
    normalize_speaker_turns,
    resolve_pyannote_device,
)
from transcriptions.service import TranscriptionResult

KNOWN_SPEAKER_STRATEGY = "pyannote_known_speaker"
ANONYMOUS_STRATEGY = "pyannote"
DIARIZATION_STRATEGIES = frozenset({ANONYMOUS_STRATEGY, KNOWN_SPEAKER_STRATEGY})

DEFAULT_KNOWN_SPEAKER_EMBEDDING_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"
DEFAULT_MIN_SEGMENT_DURATION = 1.5
DEFAULT_AUTO_ACCEPT_MARGIN = 0.15
DEFAULT_MIN_TOP_SIMILARITY = 0.55

# Reference centroids are built from a handful of clean, spaced windows rather
# than the whole clip, matching the research spike's 16 high-energy 5s windows.
REFERENCE_WINDOW_SECONDS = 5.0
REFERENCE_MAX_WINDOWS = 16
SAMPLE_RATE = 16_000

KNOWN_SPEAKER_SOURCE = "known_speaker_voiceprint"


class KnownSpeakerError(RuntimeError):
    """Base error for known-speaker postprocessing failures."""


class KnownSpeakerConfigurationError(KnownSpeakerError):
    """Raised when known-speaker settings or references are invalid."""


class KnownSpeakerBackendUnavailableError(KnownSpeakerError):
    """Raised when the configured embedding backend cannot be used."""


@dataclass(frozen=True)
class KnownSpeakerConfig:
    """Tunable thresholds for known-speaker classification.

    The defaults come from the diarization-quality research: WeSpeaker had the
    best margin separation, and duration >=1.5s, margin >=0.15, top similarity
    >=0.55 was the recommended conservative auto-accept policy.
    """

    embedding_model: str = DEFAULT_KNOWN_SPEAKER_EMBEDDING_MODEL
    min_segment_duration: float = DEFAULT_MIN_SEGMENT_DURATION
    auto_accept_margin: float = DEFAULT_AUTO_ACCEPT_MARGIN
    min_top_similarity: float = DEFAULT_MIN_TOP_SIMILARITY


@dataclass(frozen=True)
class ReferenceAudio:
    """A decoded reference window set for one known speaker.

    ``windows`` is a list of mono float32 sample arrays, already sliced and
    selected by the caller. The classification core only needs decoded windows
    and never touches the filesystem.
    """

    speaker_id: str
    name: str
    windows: list[Any]


@dataclass(frozen=True)
class SpeakerCandidate:
    speaker: str
    similarity: float


@dataclass(frozen=True)
class SegmentClassification:
    """Per-segment known-speaker classification, before the accept decision."""

    top_speaker: str
    top_similarity: float
    margin: float
    candidates: list[SpeakerCandidate]


@dataclass(frozen=True)
class KnownSpeakerSegment:
    """Full reviewable per-segment result emitted in the sidecar artifact."""

    index: int
    start: float
    end: float
    speaker: str | None
    speaker_source: str
    speaker_confidence: float
    speaker_margin: float
    speaker_candidates: list[SpeakerCandidate]
    speaker_uncertain: bool
    raw_diarization_speaker: str | None


@dataclass(frozen=True)
class KnownSpeakerOutcome:
    result: TranscriptionResult
    segments: list[KnownSpeakerSegment]
    summary: dict[str, Any]


class KnownSpeakerEmbeddingBackendProtocol(Protocol):
    @property
    def embedding_version(self) -> str: ...

    def embed(self, samples: Any, sample_rate: int) -> list[float]: ...


class UnavailableKnownSpeakerEmbeddingBackend:
    embedding_version = "unavailable"

    def embed(self, samples: Any, sample_rate: int) -> list[float]:
        del samples, sample_rate
        raise KnownSpeakerBackendUnavailableError(
            "Known-speaker diarization was requested, but no embedding backend is "
            "configured. Set VOXHELM_DIARIZATION_BACKEND=pyannote and configure the "
            "Hugging Face token first."
        )


# --------------------------------------------------------------------------- #
# Pure embedding math (no numpy/torch dependency, fully unit-testable).
# --------------------------------------------------------------------------- #


def l2_normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return [0.0 for _ in vector]
    return [value / norm for value in vector]


def average_vectors(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        raise KnownSpeakerConfigurationError("Cannot average an empty set of embeddings.")
    length = len(vectors[0])
    totals = [0.0] * length
    for vector in vectors:
        if len(vector) != length:
            raise KnownSpeakerConfigurationError("Embeddings have inconsistent dimensions.")
        for index, value in enumerate(vector):
            totals[index] += value
    count = float(len(vectors))
    return [total / count for total in totals]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise KnownSpeakerConfigurationError("Cannot compare embeddings of different dimensions.")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def build_centroid(embeddings: Sequence[Sequence[float]]) -> list[float]:
    """Average then L2-normalize per-window embeddings into one centroid."""
    return l2_normalize(average_vectors(embeddings))


def classify_embedding(
    embedding: Sequence[float],
    centroids: Mapping[str, Sequence[float]],
) -> SegmentClassification:
    if not centroids:
        raise KnownSpeakerConfigurationError("Cannot classify without speaker centroids.")
    scored = [
        SpeakerCandidate(speaker=name, similarity=cosine_similarity(embedding, centroid))
        for name, centroid in centroids.items()
    ]
    # Stable: highest similarity first, then speaker name for deterministic ties.
    scored.sort(key=lambda candidate: (-candidate.similarity, candidate.speaker))
    top = scored[0]
    second_similarity = scored[1].similarity if len(scored) > 1 else 0.0
    margin = top.similarity - second_similarity
    return SegmentClassification(
        top_speaker=top.speaker,
        top_similarity=top.similarity,
        margin=margin,
        candidates=scored,
    )


def is_confident(
    classification: SegmentClassification,
    *,
    duration: float,
    config: KnownSpeakerConfig,
) -> bool:
    """Conservative auto-accept policy from the research recommendation."""
    return (
        duration >= config.min_segment_duration
        and classification.top_similarity >= config.min_top_similarity
        and classification.margin >= config.auto_accept_margin
    )


def select_reference_window_bounds(
    num_samples: int,
    sample_rate: int,
    *,
    window_seconds: float = REFERENCE_WINDOW_SECONDS,
    max_windows: int = REFERENCE_MAX_WINDOWS,
) -> list[tuple[int, int]]:
    """Pick up to ``max_windows`` evenly spaced sample windows over the clip.

    Returns half-open ``(start_sample, end_sample)`` bounds. A clip shorter than
    one window yields a single window covering the whole clip.
    """
    if num_samples <= 0:
        return []
    window_samples = max(int(window_seconds * sample_rate), 1)
    if num_samples <= window_samples:
        return [(0, num_samples)]
    count = min(max_windows, num_samples // window_samples)
    if count <= 1:
        return [(0, window_samples)]
    last_start = num_samples - window_samples
    step = last_start / (count - 1)
    bounds: list[tuple[int, int]] = []
    seen: set[int] = set()
    for index in range(count):
        start = int(round(index * step))
        start = max(0, min(start, last_start))
        if start in seen:
            continue
        seen.add(start)
        bounds.append((start, start + window_samples))
    return bounds


# --------------------------------------------------------------------------- #
# Audio decoding + embedding extraction (numpy/torch via the backend).
# --------------------------------------------------------------------------- #


def decode_mono_16k(audio_path: Path) -> Any:
    """Decode any audio file to a mono 16 kHz float32 numpy array in [-1, 1]."""
    numpy = importlib.import_module("numpy")
    with tempfile.TemporaryDirectory(prefix="voxhelm-known-speaker-") as temp_dir:
        wav_path = Path(temp_dir) / "audio.wav"
        try:
            subprocess.run(
                [
                    settings.VOXHELM_FFMPEG_BIN,
                    "-nostdin",
                    "-y",
                    "-i",
                    str(audio_path),
                    "-ac",
                    "1",
                    "-ar",
                    str(SAMPLE_RATE),
                    "-f",
                    "wav",
                    str(wav_path),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise KnownSpeakerError("Could not decode reference or segment audio.") from exc
        with wave.open(str(wav_path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frames = wav_file.readframes(wav_file.getnframes())
    if channels != 1 or sample_width != 2:
        raise KnownSpeakerError("Decoded known-speaker audio was not 16-bit mono PCM WAV.")
    return numpy.frombuffer(frames, dtype=numpy.dtype("<i2")).astype("float32") / 32768.0


def slice_samples(samples: Any, sample_rate: int, start: float, end: float) -> Any:
    start_index = max(int(round(start * sample_rate)), 0)
    end_index = min(int(round(end * sample_rate)), len(samples))
    if end_index <= start_index:
        return samples[0:0]
    return samples[start_index:end_index]


def pad_to_min_duration(samples: Any, sample_rate: int, min_duration: float) -> Any:
    numpy = importlib.import_module("numpy")
    min_samples = max(int(round(min_duration * sample_rate)), 1)
    if len(samples) >= min_samples:
        return samples
    if len(samples) == 0:
        return numpy.zeros(min_samples, dtype="float32")
    pad = min_samples - len(samples)
    return numpy.concatenate([samples, numpy.zeros(pad, dtype="float32")])


def extract_reference_windows(samples: Any, sample_rate: int) -> list[Any]:
    """Slice decoded reference audio into spaced, non-empty embedding windows."""
    bounds = select_reference_window_bounds(len(samples), sample_rate)
    windows: list[Any] = []
    for start_index, end_index in bounds:
        window = samples[start_index:end_index]
        if len(window) > 0:
            windows.append(window)
    return windows


def build_speaker_centroids(
    references: Sequence[ReferenceAudio],
    backend: KnownSpeakerEmbeddingBackendProtocol,
) -> dict[str, list[float]]:
    """Compute one normalized centroid per named speaker from reference windows.

    References for the same speaker name are merged so a contributor can supply
    multiple clips or source ranges.
    """
    embeddings_by_name: dict[str, list[list[float]]] = {}
    for reference in references:
        for window in reference.windows:
            if len(window) == 0:
                continue
            embeddings_by_name.setdefault(reference.name, []).append(
                backend.embed(window, SAMPLE_RATE)
            )
    centroids: dict[str, list[float]] = {}
    for name, embeddings in embeddings_by_name.items():
        if embeddings:
            centroids[name] = build_centroid(embeddings)
    if not centroids:
        raise KnownSpeakerConfigurationError(
            "No usable known-speaker reference audio produced an embedding."
        )
    return centroids


def classify_segments(
    result: TranscriptionResult,
    *,
    job_audio_samples: Any,
    centroids: dict[str, list[float]],
    raw_turns: list[SpeakerTurn],
    config: KnownSpeakerConfig,
    backend: KnownSpeakerEmbeddingBackendProtocol,
) -> list[KnownSpeakerSegment]:
    """Classify every transcript segment against the known-speaker centroids."""
    normalized_raw_turns = normalize_speaker_turns(raw_turns) if raw_turns else []
    segments: list[KnownSpeakerSegment] = []
    for index, segment in enumerate(result.segments):
        duration = max(segment.end - segment.start, 0.0)
        window = slice_samples(job_audio_samples, SAMPLE_RATE, segment.start, segment.end)
        window = pad_to_min_duration(window, SAMPLE_RATE, config.min_segment_duration)
        embedding = backend.embed(window, SAMPLE_RATE)
        classification = classify_embedding(embedding, centroids)
        confident = is_confident(classification, duration=duration, config=config)
        raw_label = None
        if normalized_raw_turns:
            raw_label = choose_speaker_for_segment(
                segment_start=segment.start,
                segment_end=segment.end,
                turns=normalized_raw_turns,
            )
        segments.append(
            KnownSpeakerSegment(
                index=index,
                start=segment.start,
                end=segment.end,
                speaker=classification.top_speaker if confident else None,
                speaker_source=KNOWN_SPEAKER_SOURCE,
                speaker_confidence=round(classification.top_similarity, 6),
                speaker_margin=round(classification.margin, 6),
                speaker_candidates=classification.candidates,
                speaker_uncertain=not confident,
                raw_diarization_speaker=raw_label,
            )
        )
    return segments


def summarize_known_speaker(
    segments: list[KnownSpeakerSegment],
    *,
    centroids: dict[str, list[float]],
    config: KnownSpeakerConfig,
    embedding_version: str,
    raw_turns: list[SpeakerTurn],
) -> dict[str, Any]:
    total = len(segments)
    confident = sum(1 for segment in segments if not segment.speaker_uncertain)
    per_speaker: dict[str, int] = {}
    for segment in segments:
        if segment.speaker:
            per_speaker[segment.speaker] = per_speaker.get(segment.speaker, 0) + 1
    margins = sorted(segment.speaker_margin for segment in segments)
    return {
        "strategy": KNOWN_SPEAKER_STRATEGY,
        "embedding_model": config.embedding_model,
        "embedding_version": embedding_version,
        "known_speakers": sorted(centroids),
        "min_segment_duration": config.min_segment_duration,
        "auto_accept_margin": config.auto_accept_margin,
        "min_top_similarity": config.min_top_similarity,
        "segment_count": total,
        "confident_segment_count": confident,
        "uncertain_segment_count": total - confident,
        "confident_speaker_distribution": per_speaker,
        "margin_median": _median(margins),
        "raw_diarization_available": bool(raw_turns),
    }


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    middle = len(values) // 2
    if len(values) % 2 == 1:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2.0


def serialize_known_speaker_segments(segments: list[KnownSpeakerSegment]) -> list[dict[str, Any]]:
    return [
        {
            "index": segment.index,
            "start": segment.start,
            "end": segment.end,
            "speaker": segment.speaker,
            "speaker_source": segment.speaker_source,
            "speaker_confidence": segment.speaker_confidence,
            "speaker_margin": segment.speaker_margin,
            "speaker_candidates": [
                {"speaker": candidate.speaker, "similarity": round(candidate.similarity, 6)}
                for candidate in segment.speaker_candidates
            ],
            "speaker_uncertain": segment.speaker_uncertain,
            "raw_diarization_speaker": segment.raw_diarization_speaker,
        }
        for segment in segments
    ]


def build_speakers_artifact(outcome: KnownSpeakerOutcome) -> dict[str, Any]:
    return {
        "version": 1,
        "summary": outcome.summary,
        "segments": serialize_known_speaker_segments(outcome.segments),
    }


def run_known_speaker_postprocess(
    result: TranscriptionResult,
    *,
    references: Sequence[ReferenceAudio],
    job_audio_samples: Any,
    raw_turns: list[SpeakerTurn],
    config: KnownSpeakerConfig,
    backend: KnownSpeakerEmbeddingBackendProtocol,
) -> KnownSpeakerOutcome:
    """Classify transcript segments against known-speaker references.

    ``references`` carry already-windowed reference audio; ``job_audio_samples``
    is the decoded mastered-mono job audio. ``raw_turns`` are the anonymous
    pyannote turns kept as a fallback/debug signal and recorded per segment.

    Known-speaker results are *suggestions*: the public Podlove/DOTe/VTT
    artifacts are intentionally left unlabeled so the consumer applies speaker
    identity only after review/approval. The per-segment candidates, confidence,
    margin, uncertainty, and raw labels live in the ``speakers`` sidecar.
    """
    if not references:
        raise KnownSpeakerConfigurationError(
            "Known-speaker diarization requires at least one speaker reference."
        )
    centroids = build_speaker_centroids(references, backend)
    segments = classify_segments(
        result,
        job_audio_samples=job_audio_samples,
        centroids=centroids,
        raw_turns=raw_turns,
        config=config,
        backend=backend,
    )
    summary = summarize_known_speaker(
        segments,
        centroids=centroids,
        config=config,
        embedding_version=backend.embedding_version,
        raw_turns=raw_turns,
    )
    # Public transcript artifacts stay unlabeled; suggestions ride the sidecar.
    return KnownSpeakerOutcome(result=result, segments=segments, summary=summary)


# --------------------------------------------------------------------------- #
# pyannote embedding backend (lazy heavy dependency, mirrors diarization).
# --------------------------------------------------------------------------- #


class PyannoteEmbeddingBackend:
    def __init__(self, *, model_name: str, auth_token: str, device_name: str = "auto") -> None:
        self.model_name = model_name
        self.auth_token = auth_token
        self.device_name = device_name
        self._inference: Any | None = None

    @property
    def embedding_version(self) -> str:
        return self.model_name

    def embed(self, samples: Any, sample_rate: int) -> list[float]:
        torch = importlib.import_module("torch")
        numpy = importlib.import_module("numpy")
        inference = self._load_inference()
        waveform = torch.from_numpy(numpy.ascontiguousarray(samples)).unsqueeze(0)
        vector = inference({"waveform": waveform, "sample_rate": sample_rate})
        return [float(value) for value in numpy.asarray(vector).reshape(-1)]

    def _load_inference(self) -> Any:
        if self._inference is not None:
            return self._inference
        if not self.auth_token:
            raise KnownSpeakerConfigurationError(
                "VOXHELM_HUGGINGFACE_TOKEN must be set to extract known-speaker embeddings."
            )
        try:
            pyannote_audio = importlib.import_module("pyannote.audio")
        except ModuleNotFoundError as exc:
            missing = exc.name or "pyannote.audio"
            raise KnownSpeakerBackendUnavailableError(
                f"{missing} is not installed. Install the pyannote.audio stack before "
                "requesting known-speaker diarization."
            ) from exc
        model_cls = getattr(pyannote_audio, "Model", None)
        inference_cls = getattr(pyannote_audio, "Inference", None)
        if model_cls is None or inference_cls is None:
            raise KnownSpeakerBackendUnavailableError(
                "pyannote.audio does not expose Model/Inference; check the installed version."
            )
        try:
            model = model_cls.from_pretrained(self.model_name, token=self.auth_token)
        except TypeError:
            model = model_cls.from_pretrained(self.model_name, use_auth_token=self.auth_token)
        if model is None:
            raise KnownSpeakerBackendUnavailableError(
                f"pyannote.audio could not load embedding model '{self.model_name}'."
            )
        inference = inference_cls(model, window="whole")
        inference.to(resolve_pyannote_device(self.device_name))
        self._inference = inference
        return inference


_EMBEDDING_LOCK = Lock()


@lru_cache(maxsize=4)
def get_pyannote_embedding_backend(
    *,
    model_name: str,
    auth_token: str,
    device_name: str,
) -> PyannoteEmbeddingBackend:
    return PyannoteEmbeddingBackend(
        model_name=model_name,
        auth_token=auth_token,
        device_name=device_name,
    )


def build_known_speaker_backend(
    *,
    backend_name: str,
    embedding_model: str,
) -> KnownSpeakerEmbeddingBackendProtocol:
    normalized = backend_name.strip().lower()
    if normalized in {"", "none"}:
        return UnavailableKnownSpeakerEmbeddingBackend()
    if normalized == "pyannote":
        return get_pyannote_embedding_backend(
            model_name=embedding_model,
            auth_token=settings.VOXHELM_HUGGINGFACE_TOKEN,
            device_name=settings.VOXHELM_PYANNOTE_DEVICE,
        )
    raise KnownSpeakerConfigurationError(
        f"Unsupported diarization backend '{backend_name}'. Supported values: none, pyannote."
    )


def get_known_speaker_backend(embedding_model: str) -> KnownSpeakerEmbeddingBackendProtocol:
    return build_known_speaker_backend(
        backend_name=settings.VOXHELM_DIARIZATION_BACKEND,
        embedding_model=embedding_model,
    )


def embed_under_inference_lane(
    fn: Any,
) -> Any:
    """Run an embedding-heavy callable under the non-interactive inference lane."""
    with admit_local_inference(LANE_NON_INTERACTIVE):
        with _EMBEDDING_LOCK:
            return fn()


__all__ = [
    "ANONYMOUS_STRATEGY",
    "DIARIZATION_STRATEGIES",
    "KNOWN_SPEAKER_STRATEGY",
    "KnownSpeakerBackendUnavailableError",
    "KnownSpeakerConfig",
    "KnownSpeakerConfigurationError",
    "KnownSpeakerError",
    "KnownSpeakerOutcome",
    "KnownSpeakerSegment",
    "ReferenceAudio",
    "SpeakerCandidate",
    "build_known_speaker_backend",
    "build_speakers_artifact",
    "classify_embedding",
    "decode_mono_16k",
    "extract_reference_windows",
    "get_known_speaker_backend",
    "is_confident",
    "run_known_speaker_postprocess",
    "select_reference_window_bounds",
    "slice_samples",
]
