from __future__ import annotations

import math

import numpy
import pytest

from jobs.models import Job
from jobs.remote_worker_cli import claim_wants_word_timestamps, known_speaker_config_from_claim
from jobs.remote_workers import (
    KNOWN_SPEAKER_SUMMARY_INTEGER_KEYS,
    KNOWN_SPEAKER_SUMMARY_NUMBER_KEYS,
    KNOWN_SPEAKER_SUMMARY_STRING_KEYS,
)
from jobs.services import job_wants_word_timestamps, parse_diarization_option
from transcriptions.diarization import SpeakerTurn
from transcriptions.errors import ApiError
from transcriptions.known_speaker import (
    ANONYMOUS_STRATEGY,
    KNOWN_SPEAKER_STRATEGY,
    MIXED_SPEAKERS_REASON,
    SPEAKER_TRACK_REASON,
    TURN_CORROBORATED_REASON,
    KnownSpeakerBackendUnavailableError,
    KnownSpeakerConfig,
    KnownSpeakerConfigurationError,
    KnownSpeakerSegment,
    PyannoteEmbeddingBackend,
    ReferenceAudio,
    SegmentClassification,
    SpeakerCandidate,
    TrackFrame,
    UnavailableKnownSpeakerEmbeddingBackend,
    average_vectors,
    build_centroid,
    build_known_speaker_backend,
    build_speaker_track,
    build_speakers_artifact,
    classify_embedding,
    classify_segments,
    cosine_similarity,
    find_mixed_windows,
    get_pyannote_embedding_backend,
    is_confident,
    known_speaker_config_from_mapping,
    l2_normalize,
    mixed_window_bounds,
    overlay_unmapped_turns,
    pad_to_min_duration,
    raw_labels_are_known,
    resegment_result,
    resegmentation_enabled,
    run_known_speaker_postprocess,
    select_reference_window_bounds,
    serialize_known_speaker_segments,
    slice_samples,
    split_segment_at_turns,
    summarize_known_speaker,
    track_speaker_for_segment,
    track_turns,
)
from transcriptions.service import TranscriptionResult, TranscriptionSegment, TranscriptionWord

# --------------------------------------------------------------------------- #
# Pure embedding math
# --------------------------------------------------------------------------- #


def test_l2_normalize_scales_to_unit_length() -> None:
    assert l2_normalize([3.0, 4.0]) == [0.6, 0.8]


def test_l2_normalize_handles_zero_vector() -> None:
    assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]


def test_average_vectors_computes_componentwise_mean() -> None:
    assert average_vectors([[1.0, 2.0], [3.0, 4.0]]) == [2.0, 3.0]


def test_average_vectors_rejects_empty_input() -> None:
    with pytest.raises(KnownSpeakerConfigurationError):
        average_vectors([])


def test_average_vectors_rejects_inconsistent_dimensions() -> None:
    with pytest.raises(KnownSpeakerConfigurationError):
        average_vectors([[1.0, 2.0], [3.0]])


def test_cosine_similarity_identical_vectors_is_one() -> None:
    assert cosine_similarity([1.0, 0.0], [2.0, 0.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_is_zero() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_similarity_rejects_mismatched_dimensions() -> None:
    with pytest.raises(KnownSpeakerConfigurationError):
        cosine_similarity([1.0], [1.0, 2.0])


def test_build_centroid_normalizes_average() -> None:
    centroid = build_centroid([[2.0, 0.0], [0.0, 0.0]])
    assert centroid == pytest.approx([1.0, 0.0])


def test_classify_embedding_picks_nearest_centroid_with_margin() -> None:
    centroids = {"Johannes": [1.0, 0.0], "Dominik": [0.0, 1.0]}
    classification = classify_embedding([0.9, 0.1], centroids)
    assert classification.top_speaker == "Johannes"
    assert classification.candidates[0].speaker == "Johannes"
    assert classification.candidates[1].speaker == "Dominik"
    assert classification.margin > 0


def test_classify_embedding_ties_break_on_speaker_name() -> None:
    centroids = {"Bravo": [1.0, 0.0], "Alpha": [1.0, 0.0]}
    classification = classify_embedding([1.0, 0.0], centroids)
    assert classification.top_speaker == "Alpha"
    assert classification.margin == pytest.approx(0.0)


def test_classify_embedding_single_centroid_has_zero_second_similarity() -> None:
    classification = classify_embedding([1.0, 0.0], {"Solo": [1.0, 0.0]})
    assert classification.margin == pytest.approx(1.0)


def test_classify_embedding_rejects_empty_centroids() -> None:
    with pytest.raises(KnownSpeakerConfigurationError):
        classify_embedding([1.0], {})


@pytest.mark.parametrize(
    "duration,top,margin,expected",
    [
        (2.0, 0.8, 0.3, True),
        (1.0, 0.8, 0.3, False),  # too short
        (2.0, 0.4, 0.3, False),  # similarity too low
        (2.0, 0.8, 0.05, False),  # margin too small
    ],
)
def test_is_confident_applies_thresholds(duration, top, margin, expected) -> None:
    classification = SegmentClassification(
        top_speaker="Johannes",
        top_similarity=top,
        margin=margin,
        candidates=[SpeakerCandidate("Johannes", top)],
    )
    config = KnownSpeakerConfig(
        min_segment_duration=1.5, auto_accept_margin=0.15, min_top_similarity=0.55
    )
    assert is_confident(classification, duration=duration, config=config) is expected


# --------------------------------------------------------------------------- #
# Window selection / slicing / padding
# --------------------------------------------------------------------------- #


def test_select_reference_window_bounds_empty_clip() -> None:
    assert select_reference_window_bounds(0, 16000) == []


def test_select_reference_window_bounds_short_clip_is_single_window() -> None:
    bounds = select_reference_window_bounds(8000, 16000, window_seconds=5.0)
    assert bounds == [(0, 8000)]


def test_select_reference_window_bounds_spaces_windows() -> None:
    bounds = select_reference_window_bounds(16000 * 60, 16000, window_seconds=5.0, max_windows=4)
    window_samples = 5 * 16000
    assert len(bounds) == 4
    assert bounds[0][0] == 0
    assert all(end - start == window_samples for start, end in bounds)
    assert bounds[-1][1] == 16000 * 60


def test_select_reference_window_bounds_single_window_when_two_fit() -> None:
    # Exactly two windows fit; count // window forces a single early window.
    bounds = select_reference_window_bounds(16000 * 9, 16000, window_seconds=5.0)
    assert bounds == [(0, 5 * 16000)]


def test_slice_samples_returns_requested_range() -> None:
    samples = numpy.arange(16000, dtype="float32")
    sliced = slice_samples(samples, 16000, 0.0, 0.5)
    assert len(sliced) == 8000


def test_slice_samples_empty_for_inverted_range() -> None:
    samples = numpy.arange(100, dtype="float32")
    assert len(slice_samples(samples, 16000, 0.01, 0.0)) == 0


def test_pad_to_min_duration_pads_short_window() -> None:
    samples = numpy.ones(8000, dtype="float32")
    padded = pad_to_min_duration(samples, 16000, 1.5)
    assert len(padded) == int(1.5 * 16000)


def test_pad_to_min_duration_keeps_long_window() -> None:
    samples = numpy.ones(int(2 * 16000), dtype="float32")
    assert len(pad_to_min_duration(samples, 16000, 1.5)) == int(2 * 16000)


def test_pad_to_min_duration_zero_window_is_zeros() -> None:
    samples = numpy.zeros(0, dtype="float32")
    padded = pad_to_min_duration(samples, 16000, 1.5)
    assert len(padded) == int(1.5 * 16000)


# --------------------------------------------------------------------------- #
# End-to-end postprocessing with a deterministic stub embedding backend
# --------------------------------------------------------------------------- #


class StubEmbeddingBackend:
    """Maps a window to a one-hot vector keyed by its mean amplitude.

    A reference/segment filled with constant value ``0.1 * k`` classifies as
    speaker ``k``, giving deterministic, separable centroids in tests.
    """

    embedding_version = "stub-v1"

    def embed(self, samples, sample_rate):
        del sample_rate
        tag = int(round(float(numpy.mean(samples)) * 10))
        vector = [0.0] * 6
        vector[max(0, min(tag, 5))] = 1.0
        return vector


def constant_window(value: float, seconds: float = 5.0, sample_rate: int = 16000):
    return numpy.full(int(seconds * sample_rate), value, dtype="float32")


def make_reference(name: str, value: float) -> ReferenceAudio:
    return ReferenceAudio(speaker_id=name.lower(), name=name, windows=[constant_window(value)])


def test_run_known_speaker_postprocess_labels_confident_segments() -> None:
    # Johannes -> amplitude 0.1 (tag 1), Dominik -> amplitude 0.2 (tag 2).
    references = [make_reference("Johannes", 0.1), make_reference("Dominik", 0.2)]
    sample_rate = 16000
    # Two long segments of clean solo audio, one per speaker.
    johannes = numpy.full(2 * sample_rate, 0.1, dtype="float32")
    dominik = numpy.full(2 * sample_rate, 0.2, dtype="float32")
    job_audio = numpy.concatenate([johannes, dominik])
    result = TranscriptionResult(
        text="a b",
        language="de",
        segments=[
            TranscriptionSegment(id=0, start=0.0, end=2.0, text="a"),
            TranscriptionSegment(id=1, start=2.0, end=4.0, text="b"),
        ],
    )
    raw_turns = [
        SpeakerTurn(start=0.0, end=2.0, speaker="SPEAKER_00"),
        SpeakerTurn(start=2.0, end=4.0, speaker="SPEAKER_01"),
    ]

    outcome = run_known_speaker_postprocess(
        result,
        references=references,
        job_audio_samples=job_audio,
        raw_turns=raw_turns,
        config=KnownSpeakerConfig(),
        backend=StubEmbeddingBackend(),
    )

    # Suggestions ride the sidecar; public transcript segments stay unlabeled.
    assert [segment.speaker for segment in outcome.result.segments] == [None, None]
    assert [seg.speaker for seg in outcome.segments] == ["Johannes", "Dominik"]
    assert [seg.speaker_uncertain for seg in outcome.segments] == [False, False]
    assert outcome.segments[0].raw_diarization_speaker == "Speaker 1"
    assert outcome.segments[1].raw_diarization_speaker == "Speaker 2"
    assert outcome.summary["confident_segment_count"] == 2
    assert outcome.summary["confident_speaker_distribution"] == {"Johannes": 1, "Dominik": 1}


def test_run_known_speaker_postprocess_marks_short_segment_uncertain() -> None:
    references = [make_reference("Johannes", 0.1), make_reference("Dominik", 0.2)]
    sample_rate = 16000
    job_audio = numpy.full(1 * sample_rate, 0.1, dtype="float32")
    result = TranscriptionResult(
        text="a",
        language="de",
        segments=[TranscriptionSegment(id=0, start=0.0, end=0.4, text="a")],
    )

    outcome = run_known_speaker_postprocess(
        result,
        references=references,
        job_audio_samples=job_audio,
        raw_turns=[],
        config=KnownSpeakerConfig(),
        backend=StubEmbeddingBackend(),
    )

    segment = outcome.segments[0]
    assert segment.speaker_uncertain is True
    assert segment.speaker is None  # uncertain: no public speaker auto-applied
    assert outcome.result.segments[0].speaker is None
    assert segment.raw_diarization_speaker is None  # no raw turns supplied
    assert outcome.summary["uncertain_segment_count"] == 1


def test_run_known_speaker_postprocess_requires_references() -> None:
    result = TranscriptionResult(text="", language="de", segments=[])
    with pytest.raises(KnownSpeakerConfigurationError):
        run_known_speaker_postprocess(
            result,
            references=[],
            job_audio_samples=numpy.zeros(10, dtype="float32"),
            raw_turns=[],
            config=KnownSpeakerConfig(),
            backend=StubEmbeddingBackend(),
        )


def test_build_speaker_centroids_requires_usable_windows() -> None:
    empty_reference = ReferenceAudio(speaker_id="x", name="X", windows=[])
    result = TranscriptionResult(text="", language="de", segments=[])
    with pytest.raises(KnownSpeakerConfigurationError):
        run_known_speaker_postprocess(
            result,
            references=[empty_reference],
            job_audio_samples=numpy.zeros(10, dtype="float32"),
            raw_turns=[],
            config=KnownSpeakerConfig(),
            backend=StubEmbeddingBackend(),
        )


def test_build_speakers_artifact_serializes_segments_and_summary() -> None:
    references = [make_reference("Johannes", 0.1)]
    job_audio = numpy.full(2 * 16000, 0.1, dtype="float32")
    result = TranscriptionResult(
        text="a",
        language="de",
        segments=[TranscriptionSegment(id=0, start=0.0, end=2.0, text="a")],
    )
    outcome = run_known_speaker_postprocess(
        result,
        references=references,
        job_audio_samples=job_audio,
        raw_turns=[],
        config=KnownSpeakerConfig(),
        backend=StubEmbeddingBackend(),
    )
    artifact = build_speakers_artifact(outcome)
    assert artifact["version"] == 1
    assert artifact["summary"]["strategy"] == KNOWN_SPEAKER_STRATEGY
    assert artifact["summary"]["embedding_version"] == "stub-v1"
    segment = artifact["segments"][0]
    assert segment["speaker"] == "Johannes"
    assert segment["speaker_source"] == "known_speaker_voiceprint"
    assert segment["speaker_candidates"][0]["speaker"] == "Johannes"
    assert math.isclose(segment["speaker_confidence"], 1.0)


# --------------------------------------------------------------------------- #
# Mixed-segment veto (sub-window classification)
# --------------------------------------------------------------------------- #


def test_mixed_window_bounds_slide_and_cover_the_tail() -> None:
    bounds = mixed_window_bounds(10.0, 14.0, config=KnownSpeakerConfig())
    assert bounds == [(10.0, 11.5), (10.75, 12.25), (11.5, 13.0), (12.25, 13.75), (12.5, 14.0)]


def test_mixed_window_bounds_empty_when_disabled_or_segment_fits_one_window() -> None:
    assert mixed_window_bounds(0.0, 10.0, config=KnownSpeakerConfig(mixed_window_duration=0)) == []
    assert mixed_window_bounds(0.0, 1.5, config=KnownSpeakerConfig()) == []


def test_mixed_window_bounds_clamps_tiny_hop() -> None:
    bounds = mixed_window_bounds(0.0, 2.0, config=KnownSpeakerConfig(mixed_window_hop=0.0))
    # 0.1s minimum hop: starts 0.0 .. 0.4, then the end-aligned window.
    assert len(bounds) == 6
    assert bounds[-1] == (0.5, 2.0)


def test_mixed_window_bounds_raises_tiny_duration_to_minimum() -> None:
    bounds = mixed_window_bounds(0.0, 1.0, config=KnownSpeakerConfig(mixed_window_duration=0.01))
    assert bounds[0] == (0.0, 0.5)
    assert bounds[-1] == (0.5, 1.0)


def test_find_mixed_windows_skips_windows_clipped_by_the_audio_end() -> None:
    # The segment claims 8.5-11.49s but the audio ends at 10s, so the
    # end-aligned window 9.99-11.49 is clipped to a 10ms sliver the embedding
    # model cannot take.
    backend = ScriptedEmbeddingBackend([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    windows = find_mixed_windows(
        segment_start=8.5,
        segment_end=11.49,
        segment_speaker="Johannes",
        job_audio_samples=numpy.zeros(10 * 16000, dtype="float32"),
        centroids={"Johannes": [1.0, 0.0, 0.0], "Dominik": [0.0, 1.0, 0.0]},
        config=KnownSpeakerConfig(),
        backend=backend,
    )

    # 8.5-10.0 is a full window and 9.25-10.75 keeps 0.75s; the sliver is skipped.
    assert windows == []
    assert backend.calls == 2


def mixed_segment_audio(sample_rate: int = 16000):
    # 5s of Johannes (0.1) followed by 1.5s of Dominik (0.2). The whole-segment
    # mean still rounds to Johannes, like a segment with a short interjection.
    return numpy.concatenate(
        [
            numpy.full(5 * sample_rate, 0.1, dtype="float32"),
            numpy.full(int(1.5 * sample_rate), 0.2, dtype="float32"),
        ]
    )


def run_mixed_segment(config: KnownSpeakerConfig):
    references = [make_reference("Johannes", 0.1), make_reference("Dominik", 0.2)]
    result = TranscriptionResult(
        text="a",
        language="de",
        segments=[TranscriptionSegment(id=0, start=0.0, end=6.5, text="a")],
    )
    return run_known_speaker_postprocess(
        result,
        references=references,
        job_audio_samples=mixed_segment_audio(),
        raw_turns=[SpeakerTurn(start=0.0, end=6.5, speaker="SPEAKER_00")],
        config=config,
        backend=StubEmbeddingBackend(),
    )


def test_mixed_segment_is_downgraded_to_uncertain() -> None:
    outcome = run_mixed_segment(KnownSpeakerConfig())

    segment = outcome.segments[0]
    # The whole segment still classifies as Johannes with full confidence ...
    assert segment.speaker_candidates[0].speaker == "Johannes"
    assert math.isclose(segment.speaker_confidence, 1.0)
    # ... but a sub-window is clearly Dominik, so no speaker is auto-applied.
    assert segment.speaker is None
    assert segment.speaker_uncertain is True
    assert segment.speaker_uncertain_reason == MIXED_SPEAKERS_REASON
    assert {window.speaker for window in segment.speaker_mixed_windows} == {"Dominik"}
    assert segment.speaker_mixed_windows[-1].end == 6.5
    assert outcome.summary["mixed_segment_count"] == 1
    assert outcome.summary["confident_segment_count"] == 0
    assert outcome.summary["uncertain_segment_count"] == 1
    assert outcome.summary["confident_speaker_distribution"] == {}


def test_mixed_segment_veto_can_be_disabled() -> None:
    outcome = run_mixed_segment(KnownSpeakerConfig(mixed_window_duration=0))

    segment = outcome.segments[0]
    assert segment.speaker == "Johannes"
    assert segment.speaker_uncertain is False
    assert segment.speaker_uncertain_reason is None
    assert segment.speaker_mixed_windows == []
    assert outcome.summary["mixed_segment_count"] == 0


def test_mixed_segment_fields_are_only_serialized_for_mixed_segments() -> None:
    mixed = build_speakers_artifact(run_mixed_segment(KnownSpeakerConfig()))
    plain = build_speakers_artifact(run_mixed_segment(KnownSpeakerConfig(mixed_window_duration=0)))

    assert mixed["version"] == 1
    assert mixed["segments"][0]["speaker_uncertain_reason"] == "mixed_speakers"
    window = mixed["segments"][0]["speaker_mixed_windows"][-1]
    assert window == {
        "start": 5.0,
        "end": 6.5,
        "speaker": "Dominik",
        "similarity": 1.0,
        "margin": 1.0,
    }
    assert "speaker_uncertain_reason" not in plain["segments"][0]
    assert "speaker_mixed_windows" not in plain["segments"][0]
    assert mixed["summary"]["mixed_window_duration"] == 1.5
    assert mixed["summary"]["mixed_window_hop"] == 0.75
    assert mixed["summary"]["mixed_window_margin"] == 0.15
    assert mixed["summary"]["mixed_window_min_similarity"] == 0.3


class ScriptedEmbeddingBackend:
    """Returns the queued vectors in order, one per ``embed`` call."""

    embedding_version = "scripted-v1"

    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = list(vectors)
        self.calls = 0

    def embed(self, samples, sample_rate):
        del samples, sample_rate
        self.calls += 1
        return self.vectors.pop(0)


@pytest.mark.parametrize(
    ("vector", "expected_speakers"),
    [
        # Clearly the other speaker: similarity 1.0, margin 1.0.
        ([0.0, 1.0, 0.0], ["Dominik"]),
        # Same speaker as the segment: never a veto.
        ([1.0, 0.0, 0.0], []),
        # Other speaker on top, but below the margin threshold (0.6 vs 0.5).
        ([0.5, 0.6, 0.62], []),
        # Other speaker with a wide margin but a similarity below the floor,
        # like music or silence that resembles nobody.
        ([0.0, 0.25, 0.97], []),
    ],
)
def test_find_mixed_windows_requires_margin_and_similarity(vector, expected_speakers) -> None:
    centroids = {"Johannes": [1.0, 0.0, 0.0], "Dominik": [0.0, 1.0, 0.0]}
    # A 2s segment has two windows: 0.0-1.5 and the end-aligned 0.5-2.0.
    backend = ScriptedEmbeddingBackend([[1.0, 0.0, 0.0], vector])

    windows = find_mixed_windows(
        segment_start=0.0,
        segment_end=2.0,
        segment_speaker="Johannes",
        job_audio_samples=numpy.zeros(2 * 16000, dtype="float32"),
        centroids=centroids,
        config=KnownSpeakerConfig(),
        backend=backend,
    )

    assert backend.calls == 2
    assert [window.speaker for window in windows] == expected_speakers


def test_uncertain_segments_skip_the_sub_window_embeddings() -> None:
    centroids_reference = [make_reference("Johannes", 0.1), make_reference("Dominik", 0.2)]

    class CountingBackend(StubEmbeddingBackend):
        calls = 0

        def embed(self, samples, sample_rate):
            type(self).calls += 1
            return super().embed(samples, sample_rate)

    result = TranscriptionResult(
        text="a",
        language="de",
        segments=[TranscriptionSegment(id=0, start=0.0, end=6.5, text="a")],
    )
    # A top-similarity threshold nothing can reach keeps the segment uncertain.
    run_known_speaker_postprocess(
        result,
        references=centroids_reference,
        job_audio_samples=mixed_segment_audio(),
        raw_turns=[],
        config=KnownSpeakerConfig(min_top_similarity=2.0),
        backend=CountingBackend(),
    )

    # Two reference windows plus one whole-segment embedding, no sub-windows.
    assert CountingBackend.calls == 3


JOHANNES = [1.0, 0.0, 0.0]
DOMINIK = [0.0, 1.0, 0.0]
# Similarity 0.5 to Johannes and 0.0 to Dominik: a clear margin, but below
# the default ``min_top_similarity`` of 0.55.
WEAK_JOHANNES = [0.5, 0.0, 0.866]


def classify_scripted(
    vectors: list[list[float]],
    *,
    raw_turns: list[SpeakerTurn],
    config: KnownSpeakerConfig | None = None,
    segment_seconds: float = 2.0,
):
    """Classify back-to-back segments, one scripted whole-segment vector each."""
    segments = [
        TranscriptionSegment(
            id=index,
            start=index * segment_seconds,
            end=(index + 1) * segment_seconds,
            text="a",
        )
        for index in range(len(vectors))
    ]
    return classify_segments(
        TranscriptionResult(text="a", language="de", segments=segments),
        job_audio_samples=numpy.zeros(int(len(vectors) * segment_seconds * 16000), dtype="float32"),
        centroids={"Johannes": JOHANNES, "Dominik": DOMINIK},
        raw_turns=raw_turns,
        # Without the veto every segment costs exactly one embedding.
        config=config or KnownSpeakerConfig(mixed_window_duration=0),
        backend=ScriptedEmbeddingBackend(vectors),
    )


def test_weak_segment_is_accepted_when_the_diarization_turn_corroborates_it() -> None:
    # Five confident Johannes segments tie SPEAKER_00 to Johannes; the sixth
    # misses only the similarity threshold and lies in the same turn.
    segments = classify_scripted(
        [JOHANNES] * 5 + [WEAK_JOHANNES],
        raw_turns=[SpeakerTurn(start=0.0, end=12.0, speaker="SPEAKER_00")],
    )

    weak = segments[-1]
    assert weak.speaker == "Johannes"
    assert weak.speaker_uncertain is False
    assert weak.speaker_accept_reason == TURN_CORROBORATED_REASON
    assert math.isclose(weak.speaker_confidence, 0.5, abs_tol=1e-3)
    assert all(segment.speaker_accept_reason is None for segment in segments[:5])


@pytest.mark.parametrize(
    ("vectors", "raw_turns", "config"),
    [
        # No diarization turns at all.
        ([JOHANNES] * 5 + [WEAK_JOHANNES], [], None),
        # The weak segment lies in another speaker's turn.
        (
            [JOHANNES] * 5 + [WEAK_JOHANNES],
            [
                SpeakerTurn(start=0.0, end=10.0, speaker="SPEAKER_00"),
                SpeakerTurn(start=10.0, end=12.0, speaker="SPEAKER_01"),
            ],
            None,
        ),
        # A second diarization speaker overlaps the weak segment for 0.5s.
        (
            [JOHANNES] * 5 + [WEAK_JOHANNES],
            [
                SpeakerTurn(start=0.0, end=11.5, speaker="SPEAKER_00"),
                SpeakerTurn(start=11.5, end=12.0, speaker="SPEAKER_01"),
            ],
            None,
        ),
        # Too few confident segments to tie the label to a known speaker.
        (
            [JOHANNES] * 4 + [WEAK_JOHANNES],
            [SpeakerTurn(start=0.0, end=10.0, speaker="SPEAKER_00")],
            None,
        ),
        # The label is split between two known speakers.
        (
            [JOHANNES] * 4 + [DOMINIK] * 2 + [WEAK_JOHANNES],
            [SpeakerTurn(start=0.0, end=14.0, speaker="SPEAKER_00")],
            None,
        ),
        # Below the corroboration floor (similarity 0.3 to Johannes).
        (
            [JOHANNES] * 5 + [[0.3, 0.0, 0.954]],
            [SpeakerTurn(start=0.0, end=12.0, speaker="SPEAKER_00")],
            None,
        ),
        # Margin too small (0.5 vs 0.45): the turn cannot settle a close call.
        (
            [JOHANNES] * 5 + [[0.5, 0.45, 0.74]],
            [SpeakerTurn(start=0.0, end=12.0, speaker="SPEAKER_00")],
            None,
        ),
        # Switched off.
        (
            [JOHANNES] * 5 + [WEAK_JOHANNES],
            [SpeakerTurn(start=0.0, end=12.0, speaker="SPEAKER_00")],
            KnownSpeakerConfig(mixed_window_duration=0, turn_corroborated_min_similarity=0),
        ),
    ],
)
def test_weak_segment_stays_uncertain_without_corroboration(vectors, raw_turns, config) -> None:
    weak = classify_scripted(vectors, raw_turns=raw_turns, config=config)[-1]

    assert weak.speaker is None
    assert weak.speaker_uncertain is True
    assert weak.speaker_accept_reason is None


def test_short_weak_segment_is_not_corroborated() -> None:
    weak = classify_scripted(
        [JOHANNES] * 5 + [WEAK_JOHANNES],
        raw_turns=[SpeakerTurn(start=0.0, end=12.0, speaker="SPEAKER_00")],
        config=KnownSpeakerConfig(mixed_window_duration=0, min_segment_duration=3.0),
    )[-1]

    assert weak.speaker_uncertain is True


def test_corroborated_segment_is_still_subject_to_the_mixed_veto() -> None:
    # With the veto on, a 2s segment has two sub-windows. The five confident
    # segments pass theirs; the corroborated one has a clear Dominik window.
    confident = [JOHANNES, JOHANNES, JOHANNES]
    vectors = confident * 5 + [WEAK_JOHANNES] + [JOHANNES, DOMINIK]
    segments = [
        TranscriptionSegment(id=index, start=index * 2.0, end=(index + 1) * 2.0, text="a")
        for index in range(6)
    ]
    centroids = {"Johannes": JOHANNES, "Dominik": DOMINIK}
    raw_turns = [SpeakerTurn(start=0.0, end=12.0, speaker="SPEAKER_00")]
    config = KnownSpeakerConfig()
    classified = classify_segments(
        TranscriptionResult(text="a", language="de", segments=segments),
        job_audio_samples=numpy.zeros(12 * 16000, dtype="float32"),
        centroids=centroids,
        raw_turns=raw_turns,
        config=config,
        backend=ScriptedEmbeddingBackend(vectors),
    )
    weak = classified[-1]

    assert weak.speaker is None
    assert weak.speaker_uncertain is True
    assert weak.speaker_accept_reason is None
    # The dissenting window is kept for review like on any vetoed segment.
    assert weak.speaker_uncertain_reason == MIXED_SPEAKERS_REASON
    assert [window.speaker for window in weak.speaker_mixed_windows] == ["Dominik"]
    serialized = serialize_known_speaker_segments(classified)[-1]
    assert serialized["speaker_uncertain_reason"] == "mixed_speakers"
    assert serialized["speaker_mixed_windows"][0]["speaker"] == "Dominik"
    assert "speaker_accept_reason" not in serialized
    summary = summarize_known_speaker(
        classified,
        centroids=centroids,
        config=config,
        embedding_version="scripted-v1",
        raw_turns=raw_turns,
    )
    assert summary["mixed_segment_count"] == 1
    assert summary["turn_corroborated_segment_count"] == 0


def test_turn_corroboration_is_serialized_and_summarized() -> None:
    references = [make_reference("Johannes", 0.1), make_reference("Dominik", 0.2)]
    result = TranscriptionResult(
        text="a",
        language="de",
        segments=[
            TranscriptionSegment(id=index, start=index * 2.0, end=(index + 1) * 2.0, text="a")
            for index in range(6)
        ],
    )

    class WeakLastSegmentBackend(StubEmbeddingBackend):
        """One-hot like the stub, but the sixth segment is only a weak Johannes."""

        def embed(self, samples, sample_rate):
            if 0.14 < float(numpy.mean(samples)) < 0.16:
                return [0.0, 0.5, 0.0, 0.866, 0.0, 0.0]
            return super().embed(samples, sample_rate)

    audio = numpy.concatenate(
        [numpy.full(10 * 16000, 0.1, dtype="float32"), numpy.full(2 * 16000, 0.15, "float32")]
    )
    outcome = run_known_speaker_postprocess(
        result,
        references=references,
        job_audio_samples=audio,
        raw_turns=[SpeakerTurn(start=0.0, end=12.0, speaker="SPEAKER_00")],
        config=KnownSpeakerConfig(),
        backend=WeakLastSegmentBackend(),
    )
    artifact = build_speakers_artifact(outcome)

    assert artifact["segments"][-1]["speaker"] == "Johannes"
    assert artifact["segments"][-1]["speaker_accept_reason"] == "turn_corroborated"
    assert "speaker_accept_reason" not in artifact["segments"][0]
    assert artifact["summary"]["turn_corroborated_segment_count"] == 1
    assert artifact["summary"]["turn_corroborated_min_similarity"] == 0.35
    assert artifact["summary"]["confident_segment_count"] == 6


def test_parse_known_speaker_config_accepts_the_corroboration_floor() -> None:
    parsed = parse_diarization_option(
        known_speaker_payload(known_speaker={"turn_corroborated_min_similarity": 0})
    )
    assert parsed["known_speaker"]["turn_corroborated_min_similarity"] == 0.0


def test_known_speaker_config_from_mapping_uses_defaults_and_overrides() -> None:
    assert known_speaker_config_from_mapping({}) == KnownSpeakerConfig()
    config = known_speaker_config_from_mapping(
        {
            "embedding_model": "custom/model",
            "min_segment_duration": 2,
            "auto_accept_margin": 0.2,
            "min_top_similarity": 0.6,
            "mixed_window_duration": 2.0,
            "mixed_window_hop": 0.5,
            "mixed_window_margin": 0.25,
            "mixed_window_min_similarity": 0.4,
            "turn_corroborated_min_similarity": 0.45,
            "resegment_hop": 0.5,
        }
    )
    assert config == KnownSpeakerConfig(
        embedding_model="custom/model",
        min_segment_duration=2.0,
        auto_accept_margin=0.2,
        min_top_similarity=0.6,
        mixed_window_duration=2.0,
        mixed_window_hop=0.5,
        mixed_window_margin=0.25,
        mixed_window_min_similarity=0.4,
        turn_corroborated_min_similarity=0.45,
        resegment_hop=0.5,
    )


def test_remote_claim_config_carries_mixed_window_options() -> None:
    config = known_speaker_config_from_claim(
        {
            "known_speaker": {
                "mixed_window_duration": 0,
                "mixed_window_margin": 0.3,
                "turn_corroborated_min_similarity": 0,
            }
        }
    )
    assert config.turn_corroborated_min_similarity == 0.0
    assert config.mixed_window_duration == 0.0
    assert config.mixed_window_margin == 0.3
    assert config.mixed_window_hop == KnownSpeakerConfig().mixed_window_hop
    assert known_speaker_config_from_claim({}) == KnownSpeakerConfig()


def test_remote_summary_allowlist_covers_every_local_summary_key() -> None:
    # The control plane drops summary keys it does not know, so a key added to
    # the local summary must also be allowlisted or remote jobs lose it.
    summary = run_mixed_segment(KnownSpeakerConfig()).summary
    structured_keys = {
        "known_speakers",
        "confident_speaker_distribution",
        "raw_diarization_available",
        "raw_diarization_labels",
    }
    allowlisted = (
        KNOWN_SPEAKER_SUMMARY_STRING_KEYS
        | KNOWN_SPEAKER_SUMMARY_INTEGER_KEYS
        | KNOWN_SPEAKER_SUMMARY_NUMBER_KEYS
        | structured_keys
    )
    assert set(summary) == allowlisted
    assert all(isinstance(summary[key], int) for key in KNOWN_SPEAKER_SUMMARY_INTEGER_KEYS)


# --------------------------------------------------------------------------- #
# Re-cut at speaker changes
# --------------------------------------------------------------------------- #


def frame(center: float, speaker: str, *, clear: bool = True) -> TrackFrame:
    return TrackFrame(center=center, speaker=speaker, similarity=0.6, margin=0.4, clear=clear)


def words_at(*items: tuple[float, float, str]) -> tuple[TranscriptionWord, ...]:
    return tuple(TranscriptionWord(start=start, end=end, text=text) for start, end, text in items)


def two_speaker_audio(sample_rate: int = 16000):
    # 5s of Johannes (0.1) followed by 5s of Dominik (0.2).
    return numpy.concatenate(
        [
            numpy.full(5 * sample_rate, 0.1, dtype="float32"),
            numpy.full(5 * sample_rate, 0.2, dtype="float32"),
        ]
    )


def test_resegmentation_needs_a_hop_and_the_veto_window() -> None:
    assert resegmentation_enabled(KnownSpeakerConfig()) is True
    assert resegmentation_enabled(KnownSpeakerConfig(resegment_hop=0)) is False
    assert resegmentation_enabled(KnownSpeakerConfig(mixed_window_duration=0)) is False


def test_build_speaker_track_slides_over_the_whole_file() -> None:
    frames = build_speaker_track(
        job_audio_samples=two_speaker_audio(),
        centroids={
            "Johannes": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "Dominik": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        },
        config=KnownSpeakerConfig(resegment_hop=0.5),
        backend=StubEmbeddingBackend(),
    )

    # 1.5s windows every 0.5s over 10s: starts 0.0 .. 8.5.
    assert len(frames) == 18
    assert frames[0].center == 0.75
    assert frames[-1].center == 9.25
    assert frames[0].speaker == "Johannes" and frames[0].clear
    assert frames[-1].speaker == "Dominik" and frames[-1].clear


def test_build_speaker_track_clamps_a_tiny_hop() -> None:
    frames = build_speaker_track(
        job_audio_samples=numpy.full(2 * 16000, 0.1, dtype="float32"),
        centroids={"Johannes": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]},
        config=KnownSpeakerConfig(resegment_hop=0.0001),
        backend=StubEmbeddingBackend(),
    )

    # Hop raised to 0.1s: starts 0.0 .. 0.5.
    assert len(frames) == 6


def test_track_turns_cut_where_the_top_speaker_flips() -> None:
    frames = [
        frame(1.0, "Johannes"),
        frame(1.25, "Johannes"),
        frame(1.5, "Johannes", clear=False),
        frame(1.75, "Dominik", clear=False),
        frame(2.0, "Dominik"),
        frame(2.25, "Dominik"),
    ]

    turns = track_turns(frames)

    assert [(turn.start, turn.speaker) for turn in turns] == [(0.0, "Johannes"), (1.625, "Dominik")]
    assert turns[0].end == 1.625
    assert turns[-1].end == math.inf


def test_track_turns_use_the_gap_middle_without_a_flip() -> None:
    frames = [
        frame(1.0, "Johannes"),
        frame(1.25, "Johannes"),
        frame(1.5, "Other", clear=False),
        frame(1.75, "Dominik"),
        frame(2.0, "Dominik"),
    ]

    assert track_turns(frames)[0].end == 1.5


def test_track_turns_ignore_a_single_clear_frame_of_another_speaker() -> None:
    frames = [
        frame(1.0, "Johannes"),
        frame(1.25, "Johannes"),
        frame(1.5, "Dominik"),
        frame(1.75, "Johannes"),
        frame(2.0, "Johannes"),
    ]

    turns = track_turns(frames)

    assert [turn.speaker for turn in turns] == ["Johannes"]


def test_track_turns_empty_without_clear_frames() -> None:
    assert track_turns([frame(1.0, "Johannes", clear=False)]) == []
    assert track_turns([]) == []


TWO_TURNS = [
    SpeakerTurn(start=0.0, end=5.0, speaker="Johannes"),
    SpeakerTurn(start=5.0, end=math.inf, speaker="Dominik"),
]


def split(segment: TranscriptionSegment) -> list[TranscriptionSegment]:
    return split_segment_at_turns(segment, TWO_TURNS, [turn.start for turn in TWO_TURNS])


def test_split_segment_cuts_at_the_first_word_of_the_next_speaker() -> None:
    segment = TranscriptionSegment(
        id=7,
        start=2.0,
        end=8.0,
        text="hey carlton hello will",
        words=words_at(
            (2.1, 3.0, "hey"), (3.0, 4.6, "carlton"), (5.2, 6.0, "hello"), (6.0, 7.5, "will")
        ),
    )

    pieces = split(segment)

    assert [(piece.start, piece.end, piece.text) for piece in pieces] == [
        (2.0, 5.2, "hey carlton"),
        (5.2, 8.0, "hello will"),
    ]
    assert [len(piece.words) for piece in pieces] == [2, 2]


def test_split_segment_folds_a_single_word_into_its_neighbour() -> None:
    segment = TranscriptionSegment(
        id=0,
        start=2.0,
        end=8.0,
        text="one two three yeah",
        words=words_at(
            (2.1, 3.0, "one"), (3.0, 4.0, "two"), (4.0, 4.8, "three"), (5.2, 6.0, "yeah")
        ),
    )

    assert split(segment) == [segment]


def test_split_segment_folds_a_sliver_of_time() -> None:
    segment = TranscriptionSegment(
        id=0,
        start=2.0,
        end=8.0,
        text="one two oh no",
        words=words_at((2.1, 3.0, "one"), (3.0, 4.0, "two"), (5.2, 5.3, "oh"), (5.3, 5.4, "no")),
    )

    assert split(segment) == [segment]


def test_split_segment_never_emits_empty_or_duplicate_start_pieces() -> None:
    # Word timings the parser accepts: equal starts and long overlaps.
    segment = TranscriptionSegment(
        id=0,
        start=1.0,
        end=10.0,
        text="a b c d",
        words=words_at((1.0, 10.0, "a"), (1.0, 10.0, "b"), (1.0, 2.0, "c"), (1.0, 2.0, "d")),
    )

    pieces = split(segment)

    assert pieces == [segment]


def test_split_pieces_have_increasing_starts_and_positive_durations() -> None:
    segment = TranscriptionSegment(
        id=0,
        start=2.0,
        end=9.0,
        text="a b c d e f",
        words=words_at(
            (2.1, 3.0, "a"),
            (3.0, 4.5, "b"),
            (5.1, 6.0, "c"),
            (6.0, 7.0, "d"),
            (7.0, 8.0, "e"),
            (8.0, 8.9, "f"),
        ),
    )

    pieces = split(segment)

    assert [(piece.start, piece.end) for piece in pieces] == [(2.0, 5.1), (5.1, 9.0)]
    assert all(piece.end - piece.start >= 0.3 for piece in pieces)


def test_split_segment_keeps_segments_without_words_or_within_one_turn() -> None:
    no_words = TranscriptionSegment(id=0, start=2.0, end=8.0, text="a b")
    one_turn = TranscriptionSegment(
        id=1, start=0.0, end=4.0, text="a b", words=words_at((0.5, 1.0, "a"), (1.0, 2.0, "b"))
    )

    assert split(no_words) == [no_words]
    assert split(one_turn) == [one_turn]
    assert split_segment_at_turns(no_words, TWO_TURNS[:1], [0.0]) == [no_words]


def test_resegment_result_renumbers_and_counts_split_segments() -> None:
    result = TranscriptionResult(
        text="a b c d e f",
        language="en",
        segments=[
            TranscriptionSegment(
                id=0,
                start=0.0,
                end=4.0,
                text="a b",
                words=words_at((0.5, 1.0, "a"), (1.0, 2.0, "b")),
            ),
            TranscriptionSegment(
                id=1,
                start=4.0,
                end=8.0,
                text="c d e f",
                words=words_at((4.0, 4.4, "c"), (4.4, 4.8, "d"), (5.2, 6.0, "e"), (6.0, 7.0, "f")),
            ),
        ],
        backend_name="whisper.cpp",
        model_name="model",
    )

    resegmented, from_split = resegment_result(result, TWO_TURNS)

    assert from_split == 2
    assert [(s.id, s.start, s.end, s.text) for s in resegmented.segments] == [
        (0, 0.0, 4.0, "a b"),
        (1, 4.0, 5.2, "c d"),
        (2, 5.2, 8.0, "e f"),
    ]
    assert resegmented.text == result.text
    assert (resegmented.language, resegmented.backend_name, resegmented.model_name) == (
        "en",
        "whisper.cpp",
        "model",
    )


def track_speaker(start: float, end: float, frames: list[TrackFrame]) -> str | None:
    return track_speaker_for_segment(start, end, frames, [item.center for item in frames])


def test_track_speaker_ignores_frames_near_the_segment_edges() -> None:
    frames = [frame(10.2, "Dominik"), frame(11.0, "Johannes"), frame(11.9, "Dominik")]

    assert track_speaker(10.0, 12.0, frames) == "Johannes"


def test_track_speaker_needs_one_clear_speaker() -> None:
    assert track_speaker(10.0, 12.0, [frame(11.0, "Johannes", clear=False)]) is None
    assert track_speaker(10.0, 12.0, [frame(10.8, "Johannes"), frame(11.2, "Dominik")]) is None
    assert track_speaker(10.0, 12.0, []) is None
    assert track_speaker(12.0, 12.0, [frame(12.0, "Johannes")]) is None


def test_track_speaker_needs_a_share_of_clear_frames_on_long_segments() -> None:
    unclear = [frame(10.5 + 0.25 * index, "Johannes", clear=False) for index in range(9)]
    one_clear = [frame(13.0, "Johannes")]

    # 1 of 10 frames is clear: enough for nobody on a 4s segment ...
    assert track_speaker(10.0, 14.0, unclear + one_clear) is None
    # ... but 4 of 10 are.
    mostly = unclear[:6] + [frame(12.0 + 0.25 * index, "Johannes") for index in range(4)]
    assert track_speaker(10.0, 14.0, mostly) == "Johannes"
    # A short segment needs a single clear frame.
    assert track_speaker(12.5, 13.5, one_clear) == "Johannes"


def classify_tracked(
    vectors: list[list[float]],
    *,
    seconds: float = 2.0,
    track_speaker: str = "Dominik",
    config: KnownSpeakerConfig | None = None,
    raw_turns: list[SpeakerTurn] | None = None,
    confident_vectors: int = 1,
) -> KnownSpeakerSegment:
    """Classify one segment after five confident Johannes segments.

    The five segments tie SPEAKER_00 to Johannes; the last segment lies in the
    same diarization turn unless ``raw_turns`` says otherwise, and the track
    has one clear frame for ``track_speaker`` in its middle. ``vectors`` are
    the embeddings requested for the last segment; ``confident_vectors`` is
    how many each confident segment consumes (3 with the veto on).
    """
    segments = [
        TranscriptionSegment(id=index, start=index * 2.0, end=(index + 1) * 2.0, text="a")
        for index in range(5)
    ]
    segments.append(TranscriptionSegment(id=5, start=10.0, end=10.0 + seconds, text="a"))
    if raw_turns is None:
        raw_turns = [SpeakerTurn(start=0.0, end=10.0 + seconds, speaker="SPEAKER_00")]
    return classify_segments(
        TranscriptionResult(text="a", language="de", segments=segments),
        job_audio_samples=numpy.zeros(int((10.0 + seconds) * 16000), dtype="float32"),
        centroids={"Johannes": JOHANNES, "Dominik": DOMINIK},
        raw_turns=raw_turns,
        config=config or KnownSpeakerConfig(mixed_window_duration=0),
        backend=ScriptedEmbeddingBackend([JOHANNES] * (5 * confident_vectors) + vectors),
        track_frames=[frame(10.0 + seconds / 2, track_speaker)],
    )[-1]


def test_track_speaker_is_rejected_when_the_embedding_clearly_names_another() -> None:
    # Similarity 0.3 to Johannes with a clear margin, and the track says Dominik.
    tracked = classify_tracked([[0.3, 0.0, 0.954]])

    assert tracked.speaker is None
    assert tracked.speaker_accept_reason is None


def test_track_speaker_is_accepted_without_a_contradicting_embedding() -> None:
    # Same speaker as the weak embedding.
    agreeing = classify_tracked([[0.0, 0.3, 0.954]])
    assert agreeing.speaker == "Dominik"
    assert agreeing.speaker_uncertain is False
    assert agreeing.speaker_accept_reason == SPEAKER_TRACK_REASON
    # Embedding undecided (margin 0.05): the track settles it.
    undecided = classify_tracked([[0.35, 0.3, 0.9]])
    assert undecided.speaker == "Dominik"
    assert undecided.speaker_candidates[0].speaker == "Johannes"
    # Too short for the embedding to count as a contradiction.
    short = classify_tracked([[0.6, 0.0, 0.8]], seconds=1.0)
    assert short.speaker == "Dominik"
    assert short.speaker_accept_reason == SPEAKER_TRACK_REASON


@pytest.mark.parametrize(
    "raw_turns",
    [
        # No diarization at all.
        [],
        # The segment lies in the turn of a speaker nobody has a reference for.
        [
            SpeakerTurn(start=0.0, end=10.0, speaker="SPEAKER_00"),
            SpeakerTurn(start=10.0, end=12.0, speaker="SPEAKER_GUEST"),
        ],
        # A guest overlaps the segment for half a second.
        [
            SpeakerTurn(start=0.0, end=11.5, speaker="SPEAKER_00"),
            SpeakerTurn(start=11.5, end=12.0, speaker="SPEAKER_GUEST"),
        ],
        # The diarization has no turn on the segment.
        [SpeakerTurn(start=0.0, end=10.0, speaker="SPEAKER_00")],
    ],
)
def test_track_speaker_needs_known_diarization_speakers(raw_turns) -> None:
    tracked = classify_tracked([[0.0, 0.3, 0.954]], raw_turns=raw_turns)

    assert tracked.speaker is None
    assert tracked.speaker_uncertain is True
    assert tracked.speaker_accept_reason is None


def test_track_speaker_ignores_a_brief_unknown_diarization_label() -> None:
    tracked = classify_tracked(
        [[0.0, 0.3, 0.954]],
        raw_turns=[
            SpeakerTurn(start=0.0, end=11.9, speaker="SPEAKER_00"),
            SpeakerTurn(start=11.9, end=12.0, speaker="SPEAKER_GUEST"),
        ],
    )

    assert tracked.speaker == "Dominik"


def test_summary_ties_host_labels_and_leaves_a_guest_label_open() -> None:
    # Five confident segments per host; the guest matches neither reference.
    guest = [0.0, 0.0, 1.0]
    raw_turns = [
        SpeakerTurn(start=0.0, end=10.0, speaker="SPEAKER_00"),
        SpeakerTurn(start=10.0, end=20.0, speaker="SPEAKER_01"),
        SpeakerTurn(start=20.0, end=24.0, speaker="SPEAKER_02"),
    ]
    segments = classify_scripted([JOHANNES] * 5 + [DOMINIK] * 5 + [guest] * 2, raw_turns=raw_turns)

    summary = summarize_known_speaker(
        segments,
        centroids={"Johannes": JOHANNES, "Dominik": DOMINIK},
        config=KnownSpeakerConfig(mixed_window_duration=0),
        embedding_version="scripted-v1",
        raw_turns=raw_turns,
    )

    assert [segment.speaker for segment in segments[-2:]] == [None, None]
    assert summary["raw_diarization_labels"] == {
        "Speaker 1": {"speaker": "Johannes", "seconds": 10.0},
        "Speaker 2": {"speaker": "Dominik", "seconds": 10.0},
        "Speaker 3": {"speaker": None, "seconds": 4.0},
    }


def test_summary_labels_need_enough_whole_segment_votes() -> None:
    # Four votes are below TURN_CORROBORATION_MIN_VOTES, and without
    # diarization there are no labels at all.
    raw_turns = [SpeakerTurn(start=0.0, end=8.0, speaker="SPEAKER_00")]
    segments = classify_scripted([JOHANNES] * 4, raw_turns=raw_turns)

    def labels(turns):
        return summarize_known_speaker(
            segments,
            centroids={"Johannes": JOHANNES, "Dominik": DOMINIK},
            config=KnownSpeakerConfig(mixed_window_duration=0),
            embedding_version="scripted-v1",
            raw_turns=turns,
        )["raw_diarization_labels"]

    assert labels(raw_turns) == {"Speaker 1": {"speaker": None, "seconds": 8.0}}
    assert labels([]) == {}


def test_raw_labels_are_known_uses_the_longest_label_when_all_are_brief() -> None:
    known = {"SPEAKER_00": "Johannes"}

    assert raw_labels_are_known({"SPEAKER_00": 0.2, "SPEAKER_GUEST": 0.1}, known) is True
    assert raw_labels_are_known({"SPEAKER_00": 0.1, "SPEAKER_GUEST": 0.2}, known) is False
    assert raw_labels_are_known({}, known) is False


def test_track_does_not_override_the_mixed_veto() -> None:
    # Whole segment Johannes, first sub-window Johannes, second Dominik.
    vetoed = classify_tracked(
        [JOHANNES, JOHANNES, DOMINIK],
        track_speaker="Johannes",
        config=KnownSpeakerConfig(),
        confident_vectors=3,
    )

    assert vetoed.speaker is None
    assert vetoed.speaker_uncertain_reason == MIXED_SPEAKERS_REASON
    assert vetoed.speaker_accept_reason is None


def test_track_acceptance_runs_the_mixed_veto_first() -> None:
    # The whole segment is too weak for the whole-segment rules, the track
    # supports Johannes, but the end-aligned veto window is clearly Dominik.
    vetoed = classify_tracked(
        [[0.3, 0.0, 0.954], JOHANNES, DOMINIK],
        track_speaker="Johannes",
        config=KnownSpeakerConfig(turn_corroborated_min_similarity=0),
        confident_vectors=3,
    )

    assert vetoed.speaker is None
    assert vetoed.speaker_uncertain is True
    assert vetoed.speaker_accept_reason is None
    assert vetoed.speaker_uncertain_reason == MIXED_SPEAKERS_REASON
    assert [window.speaker for window in vetoed.speaker_mixed_windows] == ["Dominik"]


def run_two_speaker_segment(config: KnownSpeakerConfig, *, with_words: bool = True):
    references = [make_reference("Johannes", 0.1), make_reference("Dominik", 0.2)]
    words = words_at(
        (0.5, 1.5, "eins"),
        (1.5, 3.0, "zwei"),
        (3.0, 4.2, "drei"),
        (5.6, 6.5, "vier"),
        (6.5, 8.0, "fuenf"),
        (8.0, 9.5, "sechs"),
    )
    result = TranscriptionResult(
        text="eins zwei drei vier fuenf sechs",
        language="de",
        segments=[
            TranscriptionSegment(
                id=0,
                start=0.0,
                end=10.0,
                text="eins zwei drei vier fuenf sechs",
                words=words if with_words else (),
            )
        ],
    )
    return run_known_speaker_postprocess(
        result,
        references=references,
        job_audio_samples=two_speaker_audio(),
        raw_turns=[],
        config=config,
        backend=StubEmbeddingBackend(),
    )


def test_postprocess_recuts_a_two_speaker_segment_and_names_both_parts() -> None:
    outcome = run_two_speaker_segment(KnownSpeakerConfig())

    assert [(s.start, s.end, s.text) for s in outcome.result.segments] == [
        (0.0, 5.6, "eins zwei drei"),
        (5.6, 10.0, "vier fuenf sechs"),
    ]
    assert [(s.index, s.start, s.end, s.speaker) for s in outcome.segments] == [
        (0, 0.0, 5.6, "Johannes"),
        (1, 5.6, 10.0, "Dominik"),
    ]
    assert outcome.result.text == "eins zwei drei vier fuenf sechs"
    assert outcome.summary["segment_count"] == 2
    assert outcome.summary["resegmented_segment_count"] == 2
    assert outcome.summary["resegment_hop"] == 0.25
    assert outcome.summary["confident_speaker_distribution"] == {"Johannes": 1, "Dominik": 1}
    assert outcome.summary["mixed_segment_count"] == 0


@pytest.mark.parametrize(
    ("config", "with_words"),
    [
        (KnownSpeakerConfig(resegment_hop=0), True),
        (KnownSpeakerConfig(), False),
    ],
)
def test_postprocess_leaves_segments_alone_without_recut_or_words(config, with_words) -> None:
    outcome = run_two_speaker_segment(config, with_words=with_words)

    assert [(s.start, s.end) for s in outcome.result.segments] == [(0.0, 10.0)]
    assert outcome.summary["segment_count"] == 1
    assert outcome.summary["resegmented_segment_count"] == 0
    assert outcome.summary["speaker_track_segment_count"] == 0
    # The mixed segment is still caught by the veto.
    assert outcome.segments[0].speaker is None
    assert outcome.segments[0].speaker_uncertain_reason == MIXED_SPEAKERS_REASON


class CountingStubBackend(StubEmbeddingBackend):
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, samples, sample_rate):
        self.calls += 1
        return super().embed(samples, sample_rate)


def run_guest_episode(*, guest_label: str, backend=None):
    """Six Johannes segments, then one segment whose last 4 s are a guest.

    The guest (amplitude 0.04, stub tag 0) matches no reference, and no window
    across the change averages to another speaker's tag, so the track sees
    only Johannes and cannot cut there; the diarization gives the guest a label.
    """
    sample_rate = 16000
    references = [make_reference("Johannes", 0.1), make_reference("Dominik", 0.2)]
    audio = numpy.concatenate(
        [
            numpy.full(16 * sample_rate, 0.1, dtype="float32"),
            numpy.full(4 * sample_rate, 0.04, dtype="float32"),
        ]
    )
    segments = [
        TranscriptionSegment(
            id=index,
            start=index * 2.0,
            end=index * 2.0 + 2.0,
            text=f"a{index} b{index}",
            words=words_at(
                (index * 2.0 + 0.2, index * 2.0 + 0.9, f"a{index}"),
                (index * 2.0 + 0.9, index * 2.0 + 1.8, f"b{index}"),
            ),
        )
        for index in range(6)
    ]
    segments.append(
        TranscriptionSegment(
            id=6,
            start=12.0,
            end=20.0,
            text="so welcome thank you for having me",
            words=words_at(
                (12.5, 13.5, "so"),
                (13.5, 15.5, "welcome"),
                (16.3, 17.0, "thank"),
                (17.0, 17.5, "you"),
                (17.5, 18.5, "for"),
                (18.5, 19.0, "having"),
                (19.0, 19.5, "me"),
            ),
        )
    )
    result = TranscriptionResult(
        text=" ".join(segment.text for segment in segments), language="en", segments=segments
    )
    return run_known_speaker_postprocess(
        result,
        references=references,
        job_audio_samples=audio,
        raw_turns=[
            SpeakerTurn(start=0.0, end=16.0, speaker="SPEAKER_00"),
            SpeakerTurn(start=16.0, end=20.0, speaker=guest_label),
        ],
        config=KnownSpeakerConfig(),
        backend=backend or StubEmbeddingBackend(),
    )


def test_postprocess_also_cuts_where_an_unmapped_diarization_label_starts() -> None:
    outcome = run_guest_episode(guest_label="SPEAKER_01")

    last_two = [(s.start, s.end, s.text) for s in outcome.result.segments[-2:]]
    assert last_two == [
        (12.0, 16.3, "so welcome"),
        (16.3, 20.0, "thank you for having me"),
    ]
    host, guest = outcome.segments[-2:]
    assert host.speaker == "Johannes"
    assert guest.speaker is None
    assert guest.raw_diarization_speaker == "Speaker 2"
    assert outcome.summary["raw_diarization_labels"] == {
        "Speaker 1": {"speaker": "Johannes", "seconds": 16.0},
        "Speaker 2": {"speaker": None, "seconds": 4.0},
    }
    assert outcome.summary["resegmented_segment_count"] == 2
    assert outcome.result.text.endswith("so welcome thank you for having me")


def test_postprocess_classifies_once_when_every_label_is_mapped() -> None:
    # The same audio with the guest stretch on the host's label: nothing is
    # unmapped, so there is no second pass and no extra cut.
    single = CountingStubBackend()
    outcome = run_guest_episode(guest_label="SPEAKER_00", backend=single)
    double = CountingStubBackend()
    run_guest_episode(guest_label="SPEAKER_01", backend=double)

    assert [(s.start, s.end) for s in outcome.result.segments][-1] == (12.0, 20.0)
    assert outcome.summary["resegmented_segment_count"] == 0
    assert double.calls > single.calls


def test_overlay_unmapped_turns_cuts_the_track_around_each_turn() -> None:
    track = [
        SpeakerTurn(start=0.0, end=10.0, speaker="Johannes"),
        SpeakerTurn(start=10.0, end=math.inf, speaker="Dominik"),
    ]
    unmapped = [
        SpeakerTurn(start=9.0, end=11.0, speaker="Speaker 3"),
        SpeakerTurn(start=2.0, end=3.0, speaker="Speaker 3"),
    ]

    assert [(t.start, t.end, t.speaker) for t in overlay_unmapped_turns(track, unmapped)] == [
        (0.0, 2.0, "Johannes"),
        (2.0, 3.0, "unmapped:Speaker 3"),
        (3.0, 9.0, "Johannes"),
        (9.0, 11.0, "unmapped:Speaker 3"),
        (11.0, math.inf, "Dominik"),
    ]


def test_overlay_resolves_nested_unmapped_turns() -> None:
    track = [SpeakerTurn(start=0.0, end=math.inf, speaker="Johannes")]
    unmapped = [
        SpeakerTurn(start=2.0, end=8.0, speaker="Speaker 3"),
        SpeakerTurn(start=3.0, end=4.0, speaker="Speaker 4"),
    ]

    assert [(t.start, t.end, t.speaker) for t in overlay_unmapped_turns(track, unmapped)] == [
        (0.0, 2.0, "Johannes"),
        (2.0, 3.0, "unmapped:Speaker 3"),
        (3.0, 4.0, "unmapped:Speaker 4"),
        (4.0, 8.0, "unmapped:Speaker 3"),
        (8.0, math.inf, "Johannes"),
    ]


def test_overlay_without_track_turns_keeps_the_rest_neutral() -> None:
    unmapped = [SpeakerTurn(start=2.0, end=3.0, speaker="Speaker 3")]

    assert [(t.start, t.end, t.speaker) for t in overlay_unmapped_turns([], unmapped)] == [
        (0.0, 2.0, ""),
        (2.0, 3.0, "unmapped:Speaker 3"),
        (3.0, math.inf, ""),
    ]


def test_postprocess_cuts_at_an_unmapped_label_without_track_runs(monkeypatch) -> None:
    # No clear track run anywhere: the whole-segment vote still ties the host
    # label, so the guest turn alone splits the segment.
    monkeypatch.setattr("transcriptions.known_speaker.track_turns", lambda frames: [])
    outcome = run_guest_episode(guest_label="SPEAKER_01")

    assert [(s.start, s.end, s.text) for s in outcome.result.segments[-2:]] == [
        (12.0, 16.3, "so welcome"),
        (16.3, 20.0, "thank you for having me"),
    ]


def test_track_accepted_segments_are_serialized_and_summarized() -> None:
    centroids = {"Johannes": JOHANNES, "Dominik": DOMINIK}
    tracked = classify_tracked([[0.0, 0.3, 0.954]])

    serialized = serialize_known_speaker_segments([tracked])[0]
    summary = summarize_known_speaker(
        [tracked],
        centroids=centroids,
        config=KnownSpeakerConfig(),
        embedding_version="scripted-v1",
        raw_turns=[],
        resegmented_segment_count=3,
    )

    assert serialized["speaker"] == "Dominik"
    assert serialized["speaker_accept_reason"] == "speaker_track"
    assert summary["speaker_track_segment_count"] == 1
    assert summary["turn_corroborated_segment_count"] == 0
    assert summary["resegmented_segment_count"] == 3


def known_speaker_claim(**known_speaker) -> dict:
    return {
        "output": {
            "diarization": {
                "enabled": True,
                "strategy": KNOWN_SPEAKER_STRATEGY,
                "known_speaker": known_speaker,
            }
        }
    }


def test_known_speaker_claims_ask_for_word_timestamps() -> None:
    assert claim_wants_word_timestamps(known_speaker_claim()) is True
    assert claim_wants_word_timestamps(known_speaker_claim(resegment_hop=0)) is False
    assert claim_wants_word_timestamps(known_speaker_claim(mixed_window_duration=0)) is False
    assert claim_wants_word_timestamps({"output": {"diarization": {"enabled": True}}}) is False
    assert claim_wants_word_timestamps({}) is False
    disabled = known_speaker_claim()
    disabled["output"]["diarization"]["enabled"] = False
    assert claim_wants_word_timestamps(disabled) is False


def test_known_speaker_jobs_ask_for_word_timestamps() -> None:
    def job(diarization: object) -> Job:
        return Job(output_data={"diarization": diarization})

    known = {"enabled": True, "strategy": KNOWN_SPEAKER_STRATEGY}
    assert job_wants_word_timestamps(job(known)) is True
    assert job_wants_word_timestamps(job({**known, "known_speaker": {"resegment_hop": 0}})) is False
    assert job_wants_word_timestamps(job({**known, "enabled": False})) is False
    assert job_wants_word_timestamps(job({"enabled": True})) is False
    assert job_wants_word_timestamps(job(None)) is False


def test_parse_known_speaker_config_validates_the_resegment_hop() -> None:
    parsed = parse_diarization_option(known_speaker_payload(known_speaker={"resegment_hop": 0}))
    assert parsed["known_speaker"]["resegment_hop"] == 0.0
    with pytest.raises(ApiError, match="resegment_hop"):
        parse_diarization_option(known_speaker_payload(known_speaker={"resegment_hop": 0.05}))


# --------------------------------------------------------------------------- #
# Embedding backend factory
# --------------------------------------------------------------------------- #


def test_build_known_speaker_backend_none_is_unavailable() -> None:
    backend = build_known_speaker_backend(backend_name="none", embedding_model="m")
    assert isinstance(backend, UnavailableKnownSpeakerEmbeddingBackend)


def test_unavailable_backend_raises_on_embed() -> None:
    backend = UnavailableKnownSpeakerEmbeddingBackend()
    with pytest.raises(KnownSpeakerBackendUnavailableError):
        backend.embed(numpy.zeros(10, dtype="float32"), 16000)


def test_build_known_speaker_backend_rejects_unknown_backend() -> None:
    with pytest.raises(KnownSpeakerConfigurationError):
        build_known_speaker_backend(backend_name="bogus", embedding_model="m")


def test_build_known_speaker_backend_pyannote_is_cached(settings) -> None:
    settings.VOXHELM_HUGGINGFACE_TOKEN = "token"
    settings.VOXHELM_PYANNOTE_DEVICE = "cpu"
    get_pyannote_embedding_backend.cache_clear()
    first = build_known_speaker_backend(backend_name="pyannote", embedding_model="model-x")
    second = build_known_speaker_backend(backend_name="pyannote", embedding_model="model-x")
    assert isinstance(first, PyannoteEmbeddingBackend)
    assert first is second
    assert first.embedding_version == "model-x"


def test_pyannote_embedding_backend_requires_token() -> None:
    backend = PyannoteEmbeddingBackend(model_name="m", auth_token="", device_name="cpu")
    with pytest.raises(KnownSpeakerConfigurationError):
        backend.embed(numpy.zeros(10, dtype="float32"), 16000)


# --------------------------------------------------------------------------- #
# Request contract parsing
# --------------------------------------------------------------------------- #


def known_speaker_payload(**overrides):
    diarization = {
        "enabled": True,
        "num_speakers": 4,
        "strategy": KNOWN_SPEAKER_STRATEGY,
        "known_speakers": [
            {
                "id": "12",
                "name": "Johannes",
                "references": [
                    {
                        "kind": "source_range",
                        "audio": {"kind": "url", "url": "https://cdn.example.com/pp_60.m4a"},
                        "start": 120.0,
                        "end": 150.0,
                    }
                ],
            }
        ],
    }
    diarization.update(overrides)
    return {"diarization": diarization}


def test_parse_anonymous_diarization_payload_is_unchanged() -> None:
    parsed = parse_diarization_option({"diarization": {"enabled": True, "num_speakers": 4}})
    assert parsed == {"enabled": True, "num_speakers": 4}
    assert "strategy" not in parsed


def test_parse_explicit_anonymous_strategy_omits_strategy_key() -> None:
    parsed = parse_diarization_option(
        {"diarization": {"enabled": True, "strategy": ANONYMOUS_STRATEGY}}
    )
    assert parsed == {"enabled": True}


def test_parse_known_speaker_payload_normalizes_references() -> None:
    parsed = parse_diarization_option(known_speaker_payload())
    assert parsed["strategy"] == KNOWN_SPEAKER_STRATEGY
    assert parsed["num_speakers"] == 4
    speaker = parsed["known_speakers"][0]
    assert speaker == {
        "id": "12",
        "name": "Johannes",
        "references": [
            {
                "kind": "source_range",
                "audio": {"kind": "url", "url": "https://cdn.example.com/pp_60.m4a"},
                "start": 120.0,
                "end": 150.0,
            }
        ],
    }


def test_parse_known_speaker_config_is_normalized() -> None:
    parsed = parse_diarization_option(
        known_speaker_payload(
            known_speaker={
                "embedding_model": "pyannote/wespeaker-voxceleb-resnet34-LM",
                "min_segment_duration": 1.5,
                "auto_accept_margin": 0.15,
                "min_top_similarity": 0.55,
                "mixed_window_duration": 2,
                "mixed_window_hop": 0.5,
                "mixed_window_margin": 0.2,
                "mixed_window_min_similarity": 0.35,
            }
        )
    )
    assert parsed["known_speaker"] == {
        "embedding_model": "pyannote/wespeaker-voxceleb-resnet34-LM",
        "min_segment_duration": 1.5,
        "auto_accept_margin": 0.15,
        "min_top_similarity": 0.55,
        "mixed_window_duration": 2.0,
        "mixed_window_hop": 0.5,
        "mixed_window_margin": 0.2,
        "mixed_window_min_similarity": 0.35,
    }


def test_parse_known_speaker_config_allows_disabling_the_mixed_veto() -> None:
    parsed = parse_diarization_option(
        known_speaker_payload(known_speaker={"mixed_window_duration": 0})
    )
    assert parsed["known_speaker"] == {"mixed_window_duration": 0.0}


@pytest.mark.parametrize("hop", [0, 0.05])
def test_parse_known_speaker_config_rejects_too_small_hop(hop) -> None:
    with pytest.raises(ApiError, match="mixed_window_hop must be at least 0.1"):
        parse_diarization_option(known_speaker_payload(known_speaker={"mixed_window_hop": hop}))


def test_parse_known_speaker_config_rejects_too_small_window() -> None:
    with pytest.raises(ApiError, match="mixed_window_duration must be 0 \\(off\\) or at least"):
        parse_diarization_option(
            known_speaker_payload(known_speaker={"mixed_window_duration": 0.01})
        )


@pytest.mark.parametrize("key", ["mixed_window_duration", "min_top_similarity"])
@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_parse_known_speaker_config_rejects_non_finite_numbers(key, value) -> None:
    with pytest.raises(ApiError, match=f"{key} must be a finite number"):
        parse_diarization_option(known_speaker_payload(known_speaker={key: value}))


def test_parse_known_speaker_config_rejects_negative_mixed_options() -> None:
    with pytest.raises(ApiError, match="mixed_window_margin must not be negative"):
        parse_diarization_option(known_speaker_payload(known_speaker={"mixed_window_margin": -0.1}))


def test_clip_artifact_reference_does_not_require_range() -> None:
    parsed = parse_diarization_option(
        known_speaker_payload(
            known_speakers=[
                {
                    "id": "12",
                    "name": "Johannes",
                    "references": [
                        {
                            "kind": "clip_artifact",
                            "audio": {"kind": "url", "url": "https://cdn.example.com/clip.wav"},
                        }
                    ],
                }
            ]
        )
    )
    reference = parsed["known_speakers"][0]["references"][0]
    assert reference == {
        "kind": "clip_artifact",
        "audio": {"kind": "url", "url": "https://cdn.example.com/clip.wav"},
    }


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"strategy": "bogus"}, "strategy"),
        ({"known_speakers": []}, "non-empty list"),
        ({"known_speakers": [{"name": "x", "references": [1]}]}, "id"),
        ({"known_speakers": [{"id": "1", "references": [1]}]}, "name"),
        ({"known_speakers": [{"id": "1", "name": "x", "references": []}]}, "references list"),
    ],
)
def test_parse_known_speaker_rejects_invalid_speaker_lists(overrides, message) -> None:
    with pytest.raises(ApiError, match=message):
        parse_diarization_option(known_speaker_payload(**overrides))


def test_parse_known_speaker_rejects_duplicate_ids() -> None:
    speakers = [
        {"id": "1", "name": "A", "references": [_url_reference()]},
        {"id": "1", "name": "B", "references": [_url_reference()]},
    ]
    with pytest.raises(ApiError, match="Duplicate"):
        parse_diarization_option(known_speaker_payload(known_speakers=speakers))


def _url_reference():
    return {
        "kind": "clip_artifact",
        "audio": {"kind": "url", "url": "https://cdn.example.com/clip.wav"},
    }


def test_parse_reference_rejects_unknown_kind() -> None:
    speakers = [
        {
            "id": "1",
            "name": "A",
            "references": [{"kind": "bogus", "audio": {"kind": "url", "url": "https://x.test/a"}}],
        }
    ]
    with pytest.raises(ApiError, match="reference kind"):
        parse_diarization_option(known_speaker_payload(known_speakers=speakers))


def test_parse_source_range_rejects_inverted_range() -> None:
    speakers = [
        {
            "id": "1",
            "name": "A",
            "references": [
                {
                    "kind": "source_range",
                    "audio": {"kind": "url", "url": "https://x.test/a"},
                    "start": 10.0,
                    "end": 10.0,
                }
            ],
        }
    ]
    with pytest.raises(ApiError, match="start before end"):
        parse_diarization_option(known_speaker_payload(known_speakers=speakers))


def test_parse_reference_audio_requires_absolute_url() -> None:
    speakers = [
        {
            "id": "1",
            "name": "A",
            "references": [{"kind": "clip_artifact", "audio": {"kind": "url", "url": "/relative"}}],
        }
    ]
    with pytest.raises(ApiError, match="absolute URL"):
        parse_diarization_option(known_speaker_payload(known_speakers=speakers))


def test_parse_reference_audio_rejects_upload_kind() -> None:
    speakers = [
        {
            "id": "1",
            "name": "A",
            "references": [
                {
                    "kind": "clip_artifact",
                    "audio": {
                        "kind": "upload",
                        "upload_id": "123e4567-e89b-12d3-a456-426614174000",
                    },
                }
            ],
        }
    ]
    with pytest.raises(ApiError, match="Uploaded known-speaker reference audio is not supported"):
        parse_diarization_option(known_speaker_payload(known_speakers=speakers))


def test_parse_reference_audio_rejects_unknown_audio_kind() -> None:
    speakers = [
        {
            "id": "1",
            "name": "A",
            "references": [{"kind": "clip_artifact", "audio": {"kind": "ftp"}}],
        }
    ]
    with pytest.raises(ApiError, match="audio.kind"):
        parse_diarization_option(known_speaker_payload(known_speakers=speakers))


def test_known_speaker_options_require_known_speaker_strategy() -> None:
    with pytest.raises(ApiError, match="require strategy"):
        parse_diarization_option(
            {
                "diarization": {
                    "enabled": True,
                    "known_speakers": [_url_reference()],
                }
            }
        )


def test_known_speaker_config_rejects_unknown_option() -> None:
    with pytest.raises(ApiError, match="known_speaker option"):
        parse_diarization_option(known_speaker_payload(known_speaker={"bogus": 1}))


def test_diarization_options_require_enabled_true() -> None:
    with pytest.raises(ApiError, match="require diarization.enabled=true"):
        parse_diarization_option(
            {"diarization": {"enabled": False, "strategy": KNOWN_SPEAKER_STRATEGY}}
        )
