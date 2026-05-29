from __future__ import annotations

import math

import numpy
import pytest

from jobs.services import parse_diarization_option
from transcriptions.diarization import SpeakerTurn
from transcriptions.errors import ApiError
from transcriptions.known_speaker import (
    ANONYMOUS_STRATEGY,
    KNOWN_SPEAKER_STRATEGY,
    KnownSpeakerBackendUnavailableError,
    KnownSpeakerConfig,
    KnownSpeakerConfigurationError,
    PyannoteEmbeddingBackend,
    ReferenceAudio,
    SegmentClassification,
    SpeakerCandidate,
    UnavailableKnownSpeakerEmbeddingBackend,
    average_vectors,
    build_centroid,
    build_known_speaker_backend,
    build_speakers_artifact,
    classify_embedding,
    cosine_similarity,
    get_pyannote_embedding_backend,
    is_confident,
    l2_normalize,
    pad_to_min_duration,
    run_known_speaker_postprocess,
    select_reference_window_bounds,
    slice_samples,
)
from transcriptions.service import TranscriptionResult, TranscriptionSegment

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
    bounds = select_reference_window_bounds(
        16000 * 60, 16000, window_seconds=5.0, max_windows=4
    )
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

    assert [segment.speaker for segment in outcome.result.segments] == ["Johannes", "Dominik"]
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
            }
        )
    )
    assert parsed["known_speaker"] == {
        "embedding_model": "pyannote/wespeaker-voxceleb-resnet34-LM",
        "min_segment_duration": 1.5,
        "auto_accept_margin": 0.15,
        "min_top_similarity": 0.55,
    }


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


def test_parse_reference_audio_accepts_upload_kind() -> None:
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
    parsed = parse_diarization_option(known_speaker_payload(known_speakers=speakers))
    assert parsed["known_speakers"][0]["references"][0]["audio"]["kind"] == "upload"


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
