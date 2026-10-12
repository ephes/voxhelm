import json
from unittest.mock import Mock

import numpy
import pytest

from jobs.remote_worker_cli import upload_transcript_artifacts
from jobs.services import persist_transcription_output_artifacts
from transcriptions import formats
from transcriptions.known_speaker import (
    KnownSpeakerConfig,
    KnownSpeakerError,
    ReferenceAudio,
    build_speakers_artifact,
    run_known_speaker_postprocess,
)
from transcriptions.sanitizer import sanitize_result
from transcriptions.service import TranscriptionResult, TranscriptionSegment


@pytest.mark.parametrize(
    "vector", [[], [float("nan"), 1.0], [float("inf"), 1.0], [float("-inf"), 1.0]]
)
@pytest.mark.parametrize("bad_reference", [True, False])
def test_invalid_backend_embedding_fails_closed(vector, bad_reference):
    backend = Mock(embedding_version="fake")
    backend.embed.side_effect = [vector if bad_reference else [1.0, 0.0], vector]
    result = TranscriptionResult("speech", "de", [TranscriptionSegment(0, 0.0, 2.0, "speech")])
    with pytest.raises(KnownSpeakerError):
        run_known_speaker_postprocess(
            result,
            references=[ReferenceAudio("a", "Alice", [[1.0]])],
            job_audio_samples=numpy.ones(32000),
            raw_turns=[],
            config=KnownSpeakerConfig(),
            backend=backend,
        )
    assert backend.embed.call_count == (1 if bad_reference else 2)


@pytest.mark.parametrize(
    "vector,confident", [([1e308, 1e308], True), ([1e-308, 1e-308], True), ([0.0, 0.0], False)]
)
def test_extreme_and_zero_embeddings_have_finite_sidecars(vector, confident):
    backend = Mock(embedding_version="fake")
    backend.embed.return_value = vector
    outcome = run_known_speaker_postprocess(
        TranscriptionResult("speech", "de", [TranscriptionSegment(0, 0.0, 2.0, "speech")]),
        references=[ReferenceAudio("a", "Alice", [[1.0], [1.0]])],
        job_audio_samples=numpy.ones(32000),
        raw_turns=[],
        config=KnownSpeakerConfig(),
        backend=backend,
    )
    assert outcome.segments[0].speaker_uncertain is not confident
    json.dumps(build_speakers_artifact(outcome), allow_nan=False)


@pytest.mark.parametrize(
    "starts,expected",
    [
        ([0, 60, 120, 180], 4),
        ([0, 1, 2, 3, 60], 2),
        ([0, 1, 2, 3], 1),
        ([0, 2, 4, 6], 1),
        ([0, 2.01, 4.02, 6.03], 4),
    ],
)
def test_repetition_requires_temporal_continuity(starts, expected):
    segments = [
        TranscriptionSegment(i, float(s), float(s + 1), "Ja.") for i, s in enumerate(starts)
    ]
    result = sanitize_result(
        TranscriptionResult(" ".join(s.text for s in segments), "de", segments)
    )
    assert len(result.segments) == expected
    if expected == len(segments):
        assert result.segments == segments
    else:
        assert result.segments[0].end == starts[3] + 1
        assert result.segments[-1].end == starts[-1] + 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 1e306, -1e306])
@pytest.mark.parametrize("boundary", ["start", "end"])
@pytest.mark.parametrize("text", ["speech", ""])
def test_nonfinite_cue_rejected_before_any_transcript_publication(
    value, boundary, text, monkeypatch
):
    segment = TranscriptionSegment(
        1, value if boundary == "start" else 1.0, value if boundary == "end" else 2.0, text
    )
    result = TranscriptionResult("speech", "de", [segment])
    for render in [
        formats.render_text,
        formats.render_verbose_json,
        formats.render_vtt,
        formats.render_dote,
        formats.render_podlove,
    ]:
        with pytest.raises(RuntimeError, match="timestamp"):
            render(result)
    with pytest.raises(RuntimeError, match="timestamp"):
        sanitize_result(result, enabled=False)
    publish = Mock()
    monkeypatch.setattr("jobs.services.create_or_replace_artifact", publish)
    with pytest.raises(RuntimeError, match="timestamp"):
        persist_transcription_output_artifacts(
            job=Mock(output_data={"formats": ["text", "json"]}), result=result
        )
    publish.assert_not_called()
    store = Mock()
    with pytest.raises(RuntimeError, match="timestamp"):
        upload_transcript_artifacts(
            store=store, artifact_prefix="test", formats=["text", "json"], result=result
        )
    store.put_bytes.assert_not_called()


def test_finite_negative_reversed_timestamps_keep_existing_policy():
    result = TranscriptionResult("speech", "de", [TranscriptionSegment(0, -1.0, -2.0, "speech")])
    assert formats.render_verbose_json(result)["segments"][0]["end"] == -2.0
    assert "00:00:00.000 --> 00:00:00.000" in formats.render_vtt(result)
    assert formats.render_podlove(result)["transcripts"][0]["end_ms"] == 0


@pytest.mark.parametrize(
    "bounds,expected",
    [
        ([(0, 2), (1, 3), (0.5, 4), (2, 5)], 4),  # backward start
        ([(0, 2), (1, 3), (2, 2.5), (3, 4)], 4),  # backward end
        ([(0, 2), (1, 3), (2, 4), (3, 5)], 1),  # advancing overlap
        ([(0, 2)] * 4, 1),  # equal boundaries
    ],
)
def test_loop_boundary_progression(bounds, expected):
    segments = [TranscriptionSegment(i, start, end, "Ja.") for i, (start, end) in enumerate(bounds)]
    result = sanitize_result(TranscriptionResult("Ja. " * 4, "de", segments))
    assert len(result.segments) == expected
    if expected == 4:
        assert result.segments == segments
    else:
        assert result.segments[0].start == bounds[0][0]
        assert result.segments[0].end == bounds[-1][1]
