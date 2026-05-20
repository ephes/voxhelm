from __future__ import annotations

import pytest

from transcriptions.diarization import (
    DiarizationError,
    SpeakerTurn,
    apply_speaker_labels,
    choose_speaker_for_segment,
    extract_pyannote_annotation,
    get_pyannote_diarization_backend,
    normalize_speaker_turns,
)
from transcriptions.formats import render_dote, render_podlove, render_verbose_json, render_vtt
from transcriptions.service import TranscriptionResult, TranscriptionSegment


class FakeAnnotation:
    def itertracks(self, *, yield_label: bool):
        del yield_label
        return iter(())


class FakeDiarizeOutput:
    def __init__(self, annotation: FakeAnnotation) -> None:
        self.speaker_diarization = annotation


class FakeExclusiveDiarizeOutput:
    def __init__(self, annotation: FakeAnnotation) -> None:
        self.exclusive_speaker_diarization = annotation


def test_extract_pyannote_annotation_accepts_pyannote_v4_output_wrapper() -> None:
    annotation = FakeAnnotation()

    assert extract_pyannote_annotation(FakeDiarizeOutput(annotation)) is annotation


def test_extract_pyannote_annotation_accepts_exclusive_pyannote_v4_output_wrapper() -> None:
    annotation = FakeAnnotation()

    assert extract_pyannote_annotation(FakeExclusiveDiarizeOutput(annotation)) is annotation


def test_pyannote_backend_is_cached_per_model_and_token() -> None:
    get_pyannote_diarization_backend.cache_clear()

    first = get_pyannote_diarization_backend(model_name="model-a", auth_token="token-a")
    second = get_pyannote_diarization_backend(model_name="model-a", auth_token="token-a")
    different_model = get_pyannote_diarization_backend(model_name="model-b", auth_token="token-a")
    different_token = get_pyannote_diarization_backend(model_name="model-a", auth_token="token-b")

    assert first is second
    assert first is not different_model
    assert first is not different_token


def test_normalize_speaker_turns_uses_stable_generic_labels() -> None:
    turns = normalize_speaker_turns(
        [
            SpeakerTurn(start=4.0, end=5.0, speaker="SPEAKER_00"),
            SpeakerTurn(start=1.0, end=2.0, speaker="SPEAKER_01"),
            SpeakerTurn(start=0.0, end=1.0, speaker="SPEAKER_00"),
        ]
    )

    assert turns == [
        SpeakerTurn(start=0.0, end=1.0, speaker="Speaker 1"),
        SpeakerTurn(start=1.0, end=2.0, speaker="Speaker 2"),
        SpeakerTurn(start=4.0, end=5.0, speaker="Speaker 1"),
    ]


def test_apply_speaker_labels_aligns_turns_to_segments_by_largest_overlap() -> None:
    result = TranscriptionResult(
        text="first second third",
        language="en",
        segments=[
            TranscriptionSegment(id=0, start=0.0, end=1.0, text="first"),
            TranscriptionSegment(id=1, start=1.0, end=3.0, text="second"),
            TranscriptionSegment(id=2, start=3.0, end=4.0, text="third"),
        ],
    )
    turns = [
        SpeakerTurn(start=0.0, end=1.5, speaker="SPEAKER_00"),
        SpeakerTurn(start=1.5, end=4.0, speaker="SPEAKER_01"),
    ]

    labeled = apply_speaker_labels(result, turns)

    assert [segment.speaker for segment in labeled.segments] == [
        "Speaker 1",
        "Speaker 2",
        "Speaker 2",
    ]


def test_apply_speaker_labels_raises_when_no_usable_turns_exist() -> None:
    result = TranscriptionResult(
        text="hello",
        language="en",
        segments=[TranscriptionSegment(id=0, start=0.0, end=1.0, text="hello")],
    )

    with pytest.raises(DiarizationError, match="no usable speaker turns"):
        apply_speaker_labels(
            result,
            [
                SpeakerTurn(start=0.0, end=0.0, speaker="SPEAKER_00"),
                SpeakerTurn(start=0.0, end=1.0, speaker=""),
            ],
        )


def test_apply_speaker_labels_raises_when_turns_do_not_overlap_segments() -> None:
    result = TranscriptionResult(
        text="hello",
        language="en",
        segments=[TranscriptionSegment(id=0, start=10.0, end=11.0, text="hello")],
    )

    with pytest.raises(DiarizationError, match="did not overlap"):
        apply_speaker_labels(
            result,
            [SpeakerTurn(start=0.0, end=1.0, speaker="SPEAKER_00")],
        )


def test_apply_speaker_labels_raises_when_result_has_no_segments_and_no_usable_turns() -> None:
    result = TranscriptionResult(text="", language="en", segments=[])

    with pytest.raises(DiarizationError, match="no usable speaker turns"):
        apply_speaker_labels(result, [])


def test_apply_speaker_labels_raises_when_result_has_no_segments_to_align() -> None:
    result = TranscriptionResult(text="hello", language="en", segments=[])

    with pytest.raises(DiarizationError, match="did not overlap"):
        apply_speaker_labels(
            result,
            [SpeakerTurn(start=0.0, end=1.0, speaker="SPEAKER_00")],
        )


def test_choose_speaker_for_segment_uses_chronological_tiebreak() -> None:
    turns = [
        SpeakerTurn(start=0.0, end=1.0, speaker="Speaker 1"),
        SpeakerTurn(start=1.0, end=2.0, speaker="Speaker 2"),
    ]

    assert (
        choose_speaker_for_segment(segment_start=0.0, segment_end=2.0, turns=turns)
        == "Speaker 1"
    )


def test_speaker_labels_render_only_in_speaker_capable_formats() -> None:
    result = TranscriptionResult(
        text="Hello world",
        language="en",
        segments=[
            TranscriptionSegment(
                id=0,
                start=0.0,
                end=1.0,
                text="Hello",
                speaker="Speaker 1",
            ),
            TranscriptionSegment(
                id=1,
                start=1.0,
                end=2.0,
                text="world",
                speaker="Speaker 2",
            ),
        ],
    )

    assert render_verbose_json(result)["segments"][0]["speaker"] == "Speaker 1"
    assert render_dote(result)["lines"][1]["speakerDesignation"] == "Speaker 2"
    assert render_podlove(result)["transcripts"][0]["speaker"] == "Speaker 1"
    assert render_podlove(result)["transcripts"][0]["voice"] == "Speaker 1"
    assert "Speaker" not in render_vtt(result)


def test_verbose_json_omits_speaker_key_when_segment_is_unlabeled() -> None:
    result = TranscriptionResult(
        text="Hello",
        language="en",
        segments=[TranscriptionSegment(id=0, start=0.0, end=1.0, text="Hello")],
    )

    assert "speaker" not in render_verbose_json(result)["segments"][0]
