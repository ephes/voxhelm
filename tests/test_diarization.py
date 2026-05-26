from __future__ import annotations

import importlib
from typing import Any

import pytest

from transcriptions.diarization import (
    DiarizationBackendUnavailableError,
    DiarizationConfigurationError,
    DiarizationError,
    DiarizationParams,
    PyannoteDiarizationBackend,
    SpeakerTurn,
    apply_speaker_labels,
    choose_speaker_for_segment,
    extract_pyannote_annotation,
    get_pyannote_diarization_backend,
    normalize_speaker_turns,
    resolve_pyannote_device,
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


class BrokenItertracksAnnotation:
    def itertracks(self, *, yield_label: bool):
        del yield_label
        raise AttributeError("broken itertracks")


class BrokenPyannoteDiarizationBackend(PyannoteDiarizationBackend):
    def _load_pipeline(self) -> Any:
        return lambda audio, **kwargs: BrokenItertracksAnnotation()


def test_extract_pyannote_annotation_accepts_pyannote_v4_output_wrapper() -> None:
    annotation = FakeAnnotation()

    assert extract_pyannote_annotation(FakeDiarizeOutput(annotation)) is annotation


def test_extract_pyannote_annotation_accepts_exclusive_pyannote_v4_output_wrapper() -> None:
    annotation = FakeAnnotation()

    assert extract_pyannote_annotation(FakeExclusiveDiarizeOutput(annotation)) is annotation


def test_extract_pyannote_annotation_rejects_unexpected_output_with_diarization_error() -> None:
    with pytest.raises(DiarizationError, match="unexpected diarization result"):
        extract_pyannote_annotation(object())


def test_pyannote_backend_rejects_annotation_without_itertracks_with_diarization_error(
    monkeypatch,
    tmp_path,
) -> None:
    backend = BrokenPyannoteDiarizationBackend(model_name="model", auth_token="token")
    monkeypatch.setattr(
        "transcriptions.diarization.load_audio_for_pyannote",
        lambda audio_path: {"waveform": object(), "sample_rate": 16000},
    )

    with pytest.raises(DiarizationError, match="unexpected diarization result"):
        backend.diarize(tmp_path / "audio.mp3")


def test_pyannote_backend_passes_speaker_hints_to_pipeline(monkeypatch, tmp_path) -> None:
    calls = []

    class HintPyannoteDiarizationBackend(PyannoteDiarizationBackend):
        def _load_pipeline(self) -> Any:
            def pipeline(audio, **kwargs):
                calls.append((audio, kwargs))
                return FakeAnnotation()

            return pipeline

    backend = HintPyannoteDiarizationBackend(model_name="model", auth_token="token")
    audio_payload = {"waveform": object(), "sample_rate": 16000}
    monkeypatch.setattr(
        "transcriptions.diarization.load_audio_for_pyannote",
        lambda audio_path: audio_payload,
    )

    assert backend.diarize(tmp_path / "audio.mp3", DiarizationParams(num_speakers=4)) == []
    assert calls == [(audio_payload, {"num_speakers": 4})]


def test_pyannote_backend_is_cached_per_model_token_and_device() -> None:
    get_pyannote_diarization_backend.cache_clear()

    first = get_pyannote_diarization_backend(
        model_name="model-a", auth_token="token-a", device_name="cpu"
    )
    second = get_pyannote_diarization_backend(
        model_name="model-a", auth_token="token-a", device_name="cpu"
    )
    different_model = get_pyannote_diarization_backend(
        model_name="model-b", auth_token="token-a", device_name="cpu"
    )
    different_token = get_pyannote_diarization_backend(
        model_name="model-a", auth_token="token-b", device_name="cpu"
    )
    different_device = get_pyannote_diarization_backend(
        model_name="model-a", auth_token="token-a", device_name="auto"
    )

    assert first is second
    assert first is not different_model
    assert first is not different_token
    assert first is not different_device


def test_resolve_pyannote_device_cpu_resolves_to_cpu() -> None:
    assert resolve_pyannote_device("cpu").type == "cpu"


def test_resolve_pyannote_device_auto_resolves_to_supported_device() -> None:
    assert resolve_pyannote_device("auto").type in {"cpu", "mps", "cuda"}


def test_resolve_pyannote_device_rejects_unknown_device_name() -> None:
    with pytest.raises(DiarizationConfigurationError, match="Unsupported pyannote device"):
        resolve_pyannote_device("tpu")


@pytest.mark.parametrize("device_name", ["mps", "cuda"])
def test_resolve_pyannote_device_rejects_unavailable_explicit_device(
    device_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = importlib.import_module("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None:
        monkeypatch.setattr(mps_backend, "is_available", lambda: False)
    with pytest.raises(DiarizationBackendUnavailableError, match="is not available"):
        resolve_pyannote_device(device_name)


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
        choose_speaker_for_segment(segment_start=0.0, segment_end=2.0, turns=turns) == "Speaker 1"
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
