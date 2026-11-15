"""Episode voiceprints use synthetic audio identities, with no downloaded model."""

from dataclasses import replace
from threading import Event

import numpy
import pytest

from lane_scheduler import InferenceCancelled
from transcriptions.diarization import SpeakerTurn
from transcriptions.known_speaker import (
    EPISODE_VOICEPRINT_REASON,
    KnownSpeakerConfig,
    KnownSpeakerSegment,
    MixedWindow,
    ReferenceAudio,
    SpeakerCandidate,
    apply_episode_voiceprints,
    build_speakers_artifact,
    run_known_speaker_postprocess,
    tile_to_min_duration,
)
from transcriptions.service import TranscriptionResult, TranscriptionSegment


class EpisodeBackend:
    embedding_version = "synthetic-episode"

    def __init__(self):
        self.windows = []

    def embed(self, samples, sample_rate):
        self.windows.append(samples.copy())
        return {1: [1.0, 0.0, 0.0], 2: [0.0, 1.0, 0.0], 3: [0.0, 0.0, 1.0], 4: [1.0, 1.0, 0.0]}[
            int(samples[0])
        ]


def cue(index, start, duration, speaker=None, label=None):
    return KnownSpeakerSegment(
        index=index,
        start=start,
        end=start + duration,
        speaker=speaker,
        speaker_source="known_speaker_voiceprint",
        speaker_confidence=0.8,
        speaker_margin=0.3,
        speaker_candidates=[SpeakerCandidate("A", 0.8)],
        speaker_uncertain=speaker is None,
        raw_diarization_speaker=label,
    )


def episode(*, guest=False, duration=0.7, short_identity=2, short_label="Speaker 1"):
    segments: list[KnownSpeakerSegment] = []
    turns: list[SpeakerTurn] = []
    samples: list[int] = []
    for identity, speaker in ((1, "A"), (2, "B"), *(([(3, None)]) if guest else [])):
        for _ in range(5):
            start = len(samples) / 16000
            label = f"Speaker {identity}"
            length = 6 if identity == 3 else 3
            segments.append(cue(len(segments), start, length, speaker, label))
            turns.append(SpeakerTurn(start, start + length, label))
            samples.extend([identity] * int(length * 16000))
    start = len(samples) / 16000
    segments.append(cue(len(segments), start, duration, label=short_label))
    samples.extend([short_identity] * int(duration * 16000))
    return segments, turns, numpy.array(samples, dtype="float32")


def apply(segments, turns, audio, backend=None, **options):
    return apply_episode_voiceprints(
        segments,
        job_audio_samples=audio,
        reference_centroids={"A": [1.0, 0.0, 0.0], "B": [0.0, 1.0, 0.0]},
        raw_turns=turns,
        config=KnownSpeakerConfig(**options),
        backend=backend or EpisodeBackend(),
    )


def test_tiling_repeats_only_own_audio_and_handles_empty():
    audio = numpy.array([1, 2, 3], dtype="float32")
    assert tile_to_min_duration(audio, 4).tolist() == [1, 2, 3, 1, 2, 3]
    assert len(tile_to_min_duration(audio[:0])) == 0
    assert tile_to_min_duration(audio, 1) is audio


def test_known_pass_uses_own_voice_not_label_and_preserves_reference_evidence():
    original, turns, audio = episode()
    backend = EpisodeBackend()
    updated, skipped = apply(original, turns, audio, backend)
    assert skipped is None
    assert updated[:-1] == original[:-1]
    assert updated[-1].speaker == "B"
    assert updated[-1].speaker_accept_reason == EPISODE_VOICEPRINT_REASON
    assert not updated[-1].speaker_uncertain
    assert updated[-1].speaker_candidates == original[-1].speaker_candidates
    assert updated[-1].speaker_margin == original[-1].speaker_margin
    assert len(backend.windows[-1]) == 24000
    assert numpy.all(backend.windows[-1] == 2)


@pytest.mark.parametrize(
    "duration,decided", [(0.29, False), (0.3, True), (2.5, True), (2.51, False)]
)
def test_candidate_boundaries(duration, decided):
    original, turns, audio = episode(duration=duration)
    updated, _ = apply(original, turns, audio, mixed_window_duration=0)
    assert (updated[-1].speaker == "B") is decided


def test_pass_never_overrides_named_or_vetoed_segments_and_requires_margin():
    original, turns, audio = episode(short_identity=4)
    updated, _ = apply(original, turns, audio)
    assert updated == original  # tied episode voices
    for protected in (
        replace(original[-1], speaker="A", speaker_uncertain=False),
        replace(original[-1], speaker_uncertain_reason="mixed_speakers"),
    ):
        updated, _ = apply([*original[:-1], protected], turns, audio)
        assert updated[-1] == protected


def test_requires_five_whole_segment_acceptances_per_known_speaker():
    original, turns, audio = episode()
    # Track/corroboration acceptances cannot train an episode profile.
    original[0] = replace(original[0], speaker_accept_reason="speaker_track")
    updated, skipped = apply(original, turns, audio)
    assert skipped == "too_few_centroids"
    assert updated == original


def test_substantial_untied_label_skips_known_only_pass():
    original, turns, audio = episode(guest=True)
    updated, skipped = apply(original, turns, audio)
    assert skipped == "unresolved_label"
    assert updated == original


def test_disabled_pass_does_not_embed_or_change_segments():
    original, turns, audio = episode()
    backend = EpisodeBackend()
    updated, skipped = apply(original, turns, audio, backend, short_segment_margin=0)
    assert updated is original
    assert skipped == "disabled"
    assert not backend.windows


def test_mixed_veto_rechecks_episode_winner(monkeypatch):
    original, turns, audio = episode(duration=2)

    def veto(**kwargs):
        assert kwargs["segment_speaker"] == "B"
        assert set(kwargs["centroids"]) == {"A", "B"}
        return [MixedWindow(30, 31.5, "A", 0.9, 0.8)]

    monkeypatch.setattr("transcriptions.known_speaker.find_mixed_windows", veto)
    updated, _ = apply(original, turns, audio)
    assert updated[-1].speaker is None
    assert updated[-1].speaker_uncertain_reason == "mixed_speakers"
    assert updated[-1].speaker_mixed_windows


def test_cancels_between_episode_embeddings():
    original, turns, audio = episode()
    event = Event()

    class CancellingBackend(EpisodeBackend):
        def embed(self, samples, sample_rate):
            event.set()
            return super().embed(samples, sample_rate)

    backend = CancellingBackend()
    with pytest.raises(InferenceCancelled):
        apply_episode_voiceprints(
            original,
            job_audio_samples=audio,
            reference_centroids={"A": [1.0, 0.0, 0.0], "B": [0.0, 1.0, 0.0]},
            raw_turns=turns,
            config=KnownSpeakerConfig(),
            backend=backend,
            cancel_event=event,
        )
    assert len(backend.windows) == 1


def test_integrated_pass_counts_and_zero_preserves_existing_summary():
    original, turns, audio = episode()
    transcript = TranscriptionResult(
        text="",
        language="en",
        segments=[
            TranscriptionSegment(id=s.index, start=s.start, end=s.end, text="Hi") for s in original
        ],
    )

    def run(margin):
        return run_known_speaker_postprocess(
            transcript,
            references=[
                ReferenceAudio("a", "A", [numpy.array([1])]),
                ReferenceAudio("b", "B", [numpy.array([2])]),
            ],
            job_audio_samples=audio,
            raw_turns=turns,
            backend=EpisodeBackend(),
            config=KnownSpeakerConfig(short_segment_margin=margin, resegment_hop=0),
        )

    baseline, enabled = run(0), run(0.15)
    assert baseline.segments[-1].speaker is None
    assert enabled.segments[-1].speaker == "B"
    assert enabled.result == baseline.result
    assert enabled.summary["episode_voiceprint_segment_count"] == 1
    assert enabled.summary["episode_voiceprint_skipped"] is None
    assert baseline.summary["episode_voiceprint_segment_count"] == 0
    assert baseline.summary["episode_voiceprint_relabelled_count"] == 0
    old_keys = set(baseline.summary) - {
        "short_segment_margin",
        "episode_voiceprint_segment_count",
        "episode_voiceprint_relabelled_count",
        "episode_voiceprint_skipped",
        "confident_segment_count",
        "uncertain_segment_count",
        "confident_speaker_distribution",
    }
    assert {k: enabled.summary[k] for k in old_keys} == {k: baseline.summary[k] for k in old_keys}
    assert build_speakers_artifact(baseline)["version"] == 1
