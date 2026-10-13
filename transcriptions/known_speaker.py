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

One embedding per segment hides a second voice: a segment with a short
interjection from another speaker still classifies confidently as the main
speaker. Confident segments are therefore re-checked with sliding sub-windows,
and a segment in which a window clearly belongs to a different known speaker is
downgraded to uncertain (see ``find_mixed_windows``).

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
from dataclasses import dataclass, field
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
# Mixed-segment veto. Measured on Django Chat episode 206 (see
# specs/2026-10-12_mixed_speaker_segments.md): 1.5s windows separate a 1.5s
# interjection from the surrounding speaker, 1.0s windows are too noisy. Short
# windows score lower than whole segments, so the veto has its own similarity
# floor instead of reusing ``min_top_similarity``; the floor also keeps
# music/silence windows, which score near zero for everyone, from counting.
DEFAULT_MIXED_WINDOW_DURATION = 1.5
DEFAULT_MIXED_WINDOW_HOP = 0.75
DEFAULT_MIXED_WINDOW_MARGIN = 0.15
DEFAULT_MIXED_WINDOW_MIN_SIMILARITY = 0.30
# Lower bound for the hop so a tiny configured value cannot turn one segment
# into an unbounded number of embeddings.
MIN_MIXED_WINDOW_HOP = 0.1
# Shortest sub-window worth embedding. The embedding model rejects slices of a
# few milliseconds, and well under half a second it has no speaker evidence.
MIN_MIXED_WINDOW_DURATION = 0.5
MIXED_SPEAKERS_REASON = "mixed_speakers"
# Turn corroboration. A segment that misses only ``min_top_similarity`` is still
# accepted when the anonymous diarization independently puts one speaker on it
# and that diarization label belongs to the same known speaker in this job.
# Measured on Django Chat episodes 205 and 206 (see
# specs/2026-10-12_mixed_speaker_segments.md).
DEFAULT_TURN_CORROBORATED_MIN_SIMILARITY = 0.35
# A second diarization label overlapping the segment this long means the
# diarization does not see one speaker, so it corroborates nothing.
TURN_CORROBORATION_MAX_OTHER_SECONDS = 0.3
# A diarization label stands for a known speaker only when enough confident
# single-speaker segments agree on it.
TURN_CORROBORATION_MIN_VOTES = 5
TURN_CORROBORATION_MIN_SHARE = 0.9
TURN_CORROBORATED_REASON = "turn_corroborated"

KNOWN_SPEAKER_NUMBER_OPTIONS = (
    "min_segment_duration",
    "auto_accept_margin",
    "min_top_similarity",
    "mixed_window_duration",
    "mixed_window_hop",
    "mixed_window_margin",
    "mixed_window_min_similarity",
    "turn_corroborated_min_similarity",
)

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

    The ``mixed_window_*`` options drive the mixed-segment veto;
    ``mixed_window_duration=0`` turns it off. ``turn_corroborated_min_similarity``
    is the lower similarity floor for segments the anonymous diarization
    corroborates; ``0`` turns that off.
    """

    embedding_model: str = DEFAULT_KNOWN_SPEAKER_EMBEDDING_MODEL
    min_segment_duration: float = DEFAULT_MIN_SEGMENT_DURATION
    auto_accept_margin: float = DEFAULT_AUTO_ACCEPT_MARGIN
    min_top_similarity: float = DEFAULT_MIN_TOP_SIMILARITY
    mixed_window_duration: float = DEFAULT_MIXED_WINDOW_DURATION
    mixed_window_hop: float = DEFAULT_MIXED_WINDOW_HOP
    mixed_window_margin: float = DEFAULT_MIXED_WINDOW_MARGIN
    mixed_window_min_similarity: float = DEFAULT_MIXED_WINDOW_MIN_SIMILARITY
    turn_corroborated_min_similarity: float = DEFAULT_TURN_CORROBORATED_MIN_SIMILARITY


def known_speaker_config_from_mapping(config_data: Mapping[str, Any]) -> KnownSpeakerConfig:
    """Build a config from a stored/claimed ``diarization.known_speaker`` mapping."""
    defaults = KnownSpeakerConfig()

    def number(name: str) -> float:
        return float(config_data.get(name, getattr(defaults, name)))

    return KnownSpeakerConfig(
        embedding_model=str(config_data.get("embedding_model") or defaults.embedding_model),
        min_segment_duration=number("min_segment_duration"),
        auto_accept_margin=number("auto_accept_margin"),
        min_top_similarity=number("min_top_similarity"),
        mixed_window_duration=number("mixed_window_duration"),
        mixed_window_hop=number("mixed_window_hop"),
        mixed_window_margin=number("mixed_window_margin"),
        mixed_window_min_similarity=number("mixed_window_min_similarity"),
        turn_corroborated_min_similarity=number("turn_corroborated_min_similarity"),
    )


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
class MixedWindow:
    """A sub-window that clearly belongs to another speaker than its segment."""

    start: float
    end: float
    speaker: str
    similarity: float
    margin: float


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
    speaker_uncertain_reason: str | None = None
    speaker_mixed_windows: list[MixedWindow] = field(default_factory=list)
    speaker_accept_reason: str | None = None


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


def validate_embedding(vector: Sequence[float]) -> None:
    if len(vector) == 0 or any(not math.isfinite(value) for value in vector):
        raise KnownSpeakerError("Embedding must be nonempty and contain only finite coordinates.")


def l2_normalize(vector: Sequence[float]) -> list[float]:
    validate_embedding(vector)
    scale = max(abs(value) for value in vector)
    if scale == 0.0:
        return [0.0 for _ in vector]
    scaled = [value / scale for value in vector]
    norm = math.hypot(*scaled)
    return [value / norm for value in scaled]


def average_vectors(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        raise KnownSpeakerConfigurationError("Cannot average an empty set of embeddings.")
    length = len(vectors[0])
    for vector in vectors:
        validate_embedding(vector)
        if len(vector) != length:
            raise KnownSpeakerConfigurationError("Embeddings have inconsistent dimensions.")
    # Scale before summing to keep finite backend coordinates from overflowing.
    scale = max(abs(value) for vector in vectors for value in vector)
    if scale == 0.0:
        return [0.0] * length
    count = len(vectors)
    return [
        (math.fsum(vector[index] / scale for vector in vectors) / count) * scale
        for index in range(length)
    ]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise KnownSpeakerConfigurationError("Cannot compare embeddings of different dimensions.")
    left_unit = l2_normalize(left)
    right_unit = l2_normalize(right)
    return max(-1.0, min(1.0, math.fsum(a * b for a, b in zip(left_unit, right_unit, strict=True))))


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


def mixed_window_bounds(
    start: float,
    end: float,
    *,
    config: KnownSpeakerConfig,
) -> list[tuple[float, float]]:
    """Sliding sub-window bounds inside a segment for the mixed-segment veto.

    Empty when the veto is off or the segment is not longer than one window,
    since a single window would only repeat the whole-segment classification.
    A last window is aligned to the segment end so the tail is always covered.
    Window duration and hop are raised to their minimums, because claimed and
    stored configs do not pass through the API validation.
    """
    if config.mixed_window_duration <= 0:
        return []
    size = max(config.mixed_window_duration, MIN_MIXED_WINDOW_DURATION)
    if end - start <= size:
        return []
    hop = max(config.mixed_window_hop, MIN_MIXED_WINDOW_HOP)
    last_start = end - size
    bounds: list[tuple[float, float]] = []
    index = 0
    while start + index * hop < last_start:
        window_start = start + index * hop
        bounds.append((window_start, window_start + size))
        index += 1
    bounds.append((last_start, end))
    return bounds


def find_mixed_windows(
    *,
    segment_start: float,
    segment_end: float,
    segment_speaker: str,
    job_audio_samples: Any,
    centroids: Mapping[str, Sequence[float]],
    config: KnownSpeakerConfig,
    backend: KnownSpeakerEmbeddingBackendProtocol,
) -> list[MixedWindow]:
    """Return the sub-windows that clearly belong to a different known speaker.

    A window counts only when its top speaker differs from the segment's, its
    margin reaches ``mixed_window_margin`` and its top similarity reaches
    ``mixed_window_min_similarity``. Ambiguous or non-speech windows are
    ignored, so they neither confirm nor veto the segment.
    """
    mixed: list[MixedWindow] = []
    min_samples = int(MIN_MIXED_WINDOW_DURATION * SAMPLE_RATE)
    for window_start, window_end in mixed_window_bounds(segment_start, segment_end, config=config):
        window = slice_samples(job_audio_samples, SAMPLE_RATE, window_start, window_end)
        # ASR timestamps can run past the end of the audio; a clipped sliver is
        # no evidence and too short for the embedding model.
        if len(window) < min_samples:
            continue
        classification = classify_embedding(backend.embed(window, SAMPLE_RATE), centroids)
        if (
            classification.top_speaker != segment_speaker
            and classification.margin >= config.mixed_window_margin
            and classification.top_similarity >= config.mixed_window_min_similarity
        ):
            mixed.append(
                MixedWindow(
                    start=round(window_start, 3),
                    end=round(window_end, 3),
                    speaker=classification.top_speaker,
                    similarity=round(classification.top_similarity, 6),
                    margin=round(classification.margin, 6),
                )
            )
    return mixed


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
                timeout=settings.VOXHELM_AUDIO_DECODE_TIMEOUT_SECONDS,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.TimeoutExpired as exc:
            raise KnownSpeakerError(
                "Reference or segment audio decode exceeded the configured decode deadline."
            ) from exc
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


def raw_turn_overlaps(start: float, end: float, turns: list[SpeakerTurn]) -> dict[str, float]:
    """Seconds of each diarization label inside ``start``-``end``."""
    overlaps: dict[str, float] = {}
    for turn in turns:
        overlap = min(end, turn.end) - max(start, turn.start)
        if overlap > 0:
            overlaps[turn.speaker] = overlaps.get(turn.speaker, 0.0) + overlap
    return overlaps


def single_raw_label(overlaps: Mapping[str, float]) -> str | None:
    """The diarization label of a segment, or None unless it is the only one.

    Other labels may overlap for less than ``TURN_CORROBORATION_MAX_OTHER_SECONDS``
    in total; more than that and the diarization sees a second speaker.
    """
    if not overlaps:
        return None
    # Stable: longest overlap first, then label for deterministic ties.
    label, seconds = min(overlaps.items(), key=lambda item: (-item[1], item[0]))
    if sum(overlaps.values()) - seconds >= TURN_CORROBORATION_MAX_OTHER_SECONDS:
        return None
    return label


def map_raw_labels_to_speakers(votes: Mapping[str, Mapping[str, int]]) -> dict[str, str]:
    """Map diarization labels to known speakers from confident-segment votes.

    A label is mapped only when at least ``TURN_CORROBORATION_MIN_VOTES``
    confident single-speaker segments carry it and
    ``TURN_CORROBORATION_MIN_SHARE`` of them name the same known speaker.
    """
    mapping: dict[str, str] = {}
    for label, speaker_votes in votes.items():
        total = sum(speaker_votes.values())
        speaker, count = min(speaker_votes.items(), key=lambda item: (-item[1], item[0]))
        if total >= TURN_CORROBORATION_MIN_VOTES and count >= TURN_CORROBORATION_MIN_SHARE * total:
            mapping[label] = speaker
    return mapping


def is_turn_corroboration_candidate(
    classification: SegmentClassification,
    *,
    duration: float,
    config: KnownSpeakerConfig,
) -> bool:
    """True when the segment misses only ``min_top_similarity``, above the floor."""
    return (
        config.turn_corroborated_min_similarity > 0
        and duration >= config.min_segment_duration
        and classification.margin >= config.auto_accept_margin
        and config.turn_corroborated_min_similarity
        <= classification.top_similarity
        < config.min_top_similarity
    )


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

    def mixed_windows_for(segment: Any, speaker: str) -> list[MixedWindow]:
        return find_mixed_windows(
            segment_start=segment.start,
            segment_end=segment.end,
            segment_speaker=speaker,
            job_audio_samples=job_audio_samples,
            centroids=centroids,
            config=config,
            backend=backend,
        )

    classifications: list[SegmentClassification] = []
    confident_flags: list[bool] = []
    mixed_by_index: list[list[MixedWindow]] = []
    single_labels: list[str | None] = []
    label_votes: dict[str, dict[str, int]] = {}
    for segment in result.segments:
        duration = max(segment.end - segment.start, 0.0)
        window = slice_samples(job_audio_samples, SAMPLE_RATE, segment.start, segment.end)
        window = pad_to_min_duration(window, SAMPLE_RATE, config.min_segment_duration)
        embedding = backend.embed(window, SAMPLE_RATE)
        classification = classify_embedding(embedding, centroids)
        confident = is_confident(classification, duration=duration, config=config)
        mixed_windows: list[MixedWindow] = []
        if confident:
            # Only segments that would be auto-accepted need the extra embeddings.
            mixed_windows = mixed_windows_for(segment, classification.top_speaker)
            confident = not mixed_windows
        single_label = single_raw_label(
            raw_turn_overlaps(segment.start, segment.end, normalized_raw_turns)
        )
        if confident and single_label is not None:
            speaker_votes = label_votes.setdefault(single_label, {})
            speaker_votes[classification.top_speaker] = (
                speaker_votes.get(classification.top_speaker, 0) + 1
            )
        classifications.append(classification)
        confident_flags.append(confident)
        mixed_by_index.append(mixed_windows)
        single_labels.append(single_label)

    label_speakers = map_raw_labels_to_speakers(label_votes)
    segments: list[KnownSpeakerSegment] = []
    for index, segment in enumerate(result.segments):
        classification = classifications[index]
        confident = confident_flags[index]
        mixed_windows = mixed_by_index[index]
        single_label = single_labels[index]
        corroborated = False
        if (
            not confident
            and single_label is not None
            and label_speakers.get(single_label) == classification.top_speaker
            and is_turn_corroboration_candidate(
                classification,
                duration=max(segment.end - segment.start, 0.0),
                config=config,
            )
        ):
            # The mixed-segment veto applies to these as to any accepted segment.
            mixed_windows = mixed_windows_for(segment, classification.top_speaker)
            corroborated = not mixed_windows
            confident = corroborated
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
                speaker_uncertain_reason=MIXED_SPEAKERS_REASON if mixed_windows else None,
                speaker_mixed_windows=mixed_windows,
                speaker_accept_reason=TURN_CORROBORATED_REASON if corroborated else None,
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
    mixed = sum(1 for segment in segments if segment.speaker_mixed_windows)
    corroborated = sum(1 for segment in segments if segment.speaker_accept_reason)
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
        "mixed_window_duration": config.mixed_window_duration,
        "mixed_window_hop": config.mixed_window_hop,
        "mixed_window_margin": config.mixed_window_margin,
        "mixed_window_min_similarity": config.mixed_window_min_similarity,
        "turn_corroborated_min_similarity": config.turn_corroborated_min_similarity,
        "segment_count": total,
        "confident_segment_count": confident,
        "uncertain_segment_count": total - confident,
        "mixed_segment_count": mixed,
        "turn_corroborated_segment_count": corroborated,
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
    return [_serialize_known_speaker_segment(segment) for segment in segments]


def _serialize_known_speaker_segment(segment: KnownSpeakerSegment) -> dict[str, Any]:
    payload: dict[str, Any] = {
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
    # Additive, optional fields: only present on segments the mixed veto caught.
    if segment.speaker_mixed_windows:
        payload["speaker_uncertain_reason"] = segment.speaker_uncertain_reason
        payload["speaker_mixed_windows"] = [
            {
                "start": window.start,
                "end": window.end,
                "speaker": window.speaker,
                "similarity": window.similarity,
                "margin": window.margin,
            }
            for window in segment.speaker_mixed_windows
        ]
    # Also additive: only present on segments accepted below ``min_top_similarity``.
    if segment.speaker_accept_reason:
        payload["speaker_accept_reason"] = segment.speaker_accept_reason
    return payload


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
    "KNOWN_SPEAKER_NUMBER_OPTIONS",
    "KNOWN_SPEAKER_STRATEGY",
    "MIN_MIXED_WINDOW_DURATION",
    "MIN_MIXED_WINDOW_HOP",
    "MIXED_SPEAKERS_REASON",
    "KnownSpeakerBackendUnavailableError",
    "KnownSpeakerConfig",
    "KnownSpeakerConfigurationError",
    "KnownSpeakerError",
    "KnownSpeakerOutcome",
    "KnownSpeakerSegment",
    "MixedWindow",
    "ReferenceAudio",
    "SpeakerCandidate",
    "build_known_speaker_backend",
    "build_speakers_artifact",
    "classify_embedding",
    "decode_mono_16k",
    "extract_reference_windows",
    "find_mixed_windows",
    "get_known_speaker_backend",
    "is_confident",
    "known_speaker_config_from_mapping",
    "mixed_window_bounds",
    "run_known_speaker_postprocess",
    "select_reference_window_bounds",
    "slice_samples",
]
