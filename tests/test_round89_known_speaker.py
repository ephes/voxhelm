"""Synthetic regressions for known-speaker admission and bounded work."""

import math

import numpy
import pytest

from jobs.services import parse_known_speaker_config, parse_known_speakers
from transcriptions.errors import ApiError
from transcriptions.known_speaker import (
    KNOWN_SPEAKER_NUMBER_OPTIONS,
    KnownSpeakerConfig,
    KnownSpeakerConfigurationError,
    ReferenceAudio,
    build_speaker_centroids,
    known_speaker_config_from_mapping,
    run_known_speaker_postprocess,
)
from transcriptions.service import TranscriptionResult


class Backend:
    embedding_version = "synthetic"

    def __init__(self):
        self.calls = 0

    def embed(self, samples, sample_rate):
        self.calls += 1
        return [1.0, 0.0]


@pytest.mark.parametrize("key", KNOWN_SPEAKER_NUMBER_OPTIONS)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True, "1", 10**400, -1])
def test_stored_config_rejects_invalid_numbers(key, value):
    with pytest.raises(KnownSpeakerConfigurationError):
        known_speaker_config_from_mapping({key: value})


@pytest.mark.parametrize(
    "config",
    [
        KnownSpeakerConfig(min_segment_duration=1e300),
        KnownSpeakerConfig(mixed_window_hop=True),
        KnownSpeakerConfig(min_top_similarity=math.nan),
    ],
)
def test_direct_config_rejected_before_embeddings(config):
    backend = Backend()
    with pytest.raises(KnownSpeakerConfigurationError):
        run_known_speaker_postprocess(
            TranscriptionResult(text="", language="en", segments=[]),
            references=[ReferenceAudio("one", "Alex", [numpy.zeros(1)])],
            job_audio_samples=numpy.zeros(1),
            raw_turns=[],
            config=config,
            backend=backend,
        )
    assert backend.calls == 0


def test_api_and_stored_config_share_domain():
    for data in (
        {"mixed_window_duration": 0, "resegment_hop": 0},
        {"mixed_window_hop": 0.1, "min_segment_duration": 2},
    ):
        assert known_speaker_config_from_mapping(parse_known_speaker_config(data)) == (
            known_speaker_config_from_mapping(data)
        )
    for data in (
        {"min_segment_duration": 1e300},
        {"mixed_window_hop": 0.01},
        {"mixed_window_duration": 601},
    ):
        with pytest.raises(ApiError):
            parse_known_speaker_config(data)
        with pytest.raises(KnownSpeakerConfigurationError):
            known_speaker_config_from_mapping(data)


def test_duplicate_normalized_names_rejected_at_submission():
    speakers = [
        {
            "id": identity,
            "name": name,
            "references": [
                {
                    "kind": "clip_artifact",
                    "audio": {"kind": "url", "url": "https://example.com/synthetic.wav"},
                }
            ],
        }
        for identity, name in [("one", "Alex"), ("two", "  ALEX  ")]
    ]
    with pytest.raises(ApiError, match="name"):
        parse_known_speakers(speakers)


def test_reference_identity_collisions_rejected_before_embedding():
    backend = Backend()
    with pytest.raises(KnownSpeakerConfigurationError):
        build_speaker_centroids(
            [
                ReferenceAudio("one", "Alex", [numpy.zeros(1)]),
                ReferenceAudio("two", "alex", [numpy.zeros(1)]),
            ],
            backend,
        )
    assert backend.calls == 0


def test_repeated_clips_for_one_identity_merge():
    backend = Backend()
    assert build_speaker_centroids(
        [
            ReferenceAudio("one", "Alex", [numpy.zeros(1)]),
            ReferenceAudio("one", "Alex", [numpy.zeros(1)]),
        ],
        backend,
    ) == {"Alex": [1.0, 0.0]}
    assert backend.calls == 2


@pytest.mark.parametrize(
    "phase,stop_at", [("centroid", 1), ("track", 2), ("segment", 2), ("mixed", 3)]
)
def test_cancellation_stops_between_embedding_units(phase, stop_at):
    from threading import Event

    from lane_scheduler import InferenceCancelled
    from transcriptions.service import TranscriptionSegment, TranscriptionWord

    event = Event()

    class CancellingBackend(Backend):
        def embed(self, samples, sample_rate):
            value = super().embed(samples, sample_rate)
            if self.calls == stop_at:
                event.set()
            return value

    backend = CancellingBackend()
    windows = [numpy.zeros(1)] * (2 if phase == "centroid" else 1)
    words = (TranscriptionWord(start=0, end=1, text="synthetic"),) if phase == "track" else ()
    segment = TranscriptionSegment(id=0, start=0, end=4, text="synthetic", words=words)
    with pytest.raises(InferenceCancelled):
        run_known_speaker_postprocess(
            TranscriptionResult(text="synthetic", language="en", segments=[segment, segment]),
            references=[ReferenceAudio("one", "Alex", windows)],
            job_audio_samples=numpy.zeros(4 * 16000),
            raw_turns=[],
            config=KnownSpeakerConfig(
                resegment_hop=0.25 if phase == "track" else 0,
                mixed_window_duration=1.5 if phase == "mixed" else 0.5,
            ),
            backend=backend,
            cancel_event=event,
        )
    assert backend.calls == stop_at


def test_track_budget_preflight_performs_no_embedding():
    from transcriptions.known_speaker import build_speaker_track

    class LongAudio:
        def __len__(self):
            return 7200 * 16000

        def __getitem__(self, key):
            pytest.fail("over-budget audio must not be sliced")

    backend = Backend()
    with pytest.raises(KnownSpeakerConfigurationError, match="budget"):
        build_speaker_track(
            job_audio_samples=LongAudio(),
            centroids={"Alex": [1, 0]},
            config=KnownSpeakerConfig(resegment_hop=0.1),
            backend=backend,
        )
    assert backend.calls == 0


def test_track_budget_exact_boundary(monkeypatch):
    import transcriptions.known_speaker as module

    # Small operation budget makes the exact admitted/refused boundary observable.
    monkeypatch.setattr(module, "MAX_TRACK_FRAMES", 3, raising=False)
    backend = Backend()
    config = KnownSpeakerConfig(mixed_window_duration=0.5, resegment_hop=0.25)
    frames = module.build_speaker_track(
        job_audio_samples=numpy.zeros(16000),
        centroids={"Alex": [1, 0]},
        config=config,
        backend=backend,
    )
    assert len(frames) == backend.calls == 3
    with pytest.raises(KnownSpeakerConfigurationError, match="budget"):
        module.build_speaker_track(
            job_audio_samples=numpy.zeros(20000),
            centroids={"Alex": [1, 0]},
            config=config,
            backend=backend,
        )
    assert backend.calls == 3


def _count_algorithm_lines(function):
    """Deterministic work counter; includes local helpers in the same module."""
    import sys

    count = 0
    filename = function.__code__.co_filename

    def trace(frame, event, arg):
        nonlocal count
        if event == "line" and frame.f_code.co_filename == filename:
            count += 1
        return trace

    def run(*args):
        nonlocal count
        count = 0
        previous = sys.gettrace()
        sys.settrace(trace)
        try:
            result = function(*args)
        finally:
            sys.settrace(previous)
        return result, count

    return run


def test_piece_fold_linear_work_and_word_identity():
    from transcriptions.diarization import SpeakerTurn
    from transcriptions.known_speaker import split_segment_at_turns
    from transcriptions.service import TranscriptionSegment, TranscriptionWord

    count = 2000
    words = tuple(TranscriptionWord(start=i, end=i + 0.5, text=str(i)) for i in range(count))
    turns = [SpeakerTurn(start=i, end=i + 1, speaker=str(i % 2)) for i in range(count)]
    segment = TranscriptionSegment(id=0, start=0, end=count, text="synthetic", words=words)
    pieces, work = _count_algorithm_lines(split_segment_at_turns)(
        segment, turns, [turn.start for turn in turns]
    )
    assert pieces == [segment]
    assert pieces[0].words is words
    assert work < 180 * count


def _interval_oracle(turns):
    from transcriptions.diarization import SpeakerTurn

    bounds = sorted({value for turn in turns for value in (turn.start, turn.end)})
    result: list[SpeakerTurn] = []
    for start, end in zip(bounds, bounds[1:], strict=False):
        active = [turn for turn in turns if turn.start <= start < turn.end]
        if not active:
            continue
        speaker = max(active, key=lambda turn: (turn.start, turn.speaker)).speaker
        if result and result[-1].end == start and result[-1].speaker == speaker:
            result[-1] = SpeakerTurn(start=result[-1].start, end=end, speaker=speaker)
        else:
            result.append(SpeakerTurn(start=start, end=end, speaker=speaker))
    return result


def test_overlap_resolver_small_oracle():
    import random

    from transcriptions.diarization import SpeakerTurn
    from transcriptions.known_speaker import resolve_overlapping_turns

    rng = random.Random(89)
    for _ in range(100):
        turns = [
            SpeakerTurn(
                start=(start := rng.randrange(10)),
                end=start + rng.randrange(6),
                speaker=rng.choice(["a", "z"]),
            )
            for _ in range(20)
        ]
        assert resolve_overlapping_turns(turns) == _interval_oracle(turns)


def test_overlap_resolver_nested_work_bound():
    from transcriptions.diarization import SpeakerTurn
    from transcriptions.known_speaker import resolve_overlapping_turns

    count = 2000
    turns = [SpeakerTurn(start=i, end=2 * count - i, speaker=str(i)) for i in range(count)]
    result, work = _count_algorithm_lines(resolve_overlapping_turns)(turns)
    assert len(result) == 2 * count - 1
    assert work < 100 * count


def test_unmapped_overlay_work_bound():
    from transcriptions.diarization import SpeakerTurn
    from transcriptions.known_speaker import overlay_unmapped_turns

    count = 2000
    turns = [SpeakerTurn(start=i * 3, end=i * 3 + 3, speaker="known") for i in range(count)]
    unmapped = [SpeakerTurn(start=i * 3 + 1, end=i * 3 + 2, speaker="guest") for i in range(count)]
    result, work = _count_algorithm_lines(overlay_unmapped_turns)(turns, unmapped)
    assert len(result) == 3 * count
    assert result[1].speaker == "unmapped:guest"
    assert work < 160 * count


def test_piece_fold_matches_small_range_oracle():
    import random

    from transcriptions.diarization import SpeakerTurn
    from transcriptions.known_speaker import split_segment_at_turns
    from transcriptions.service import TranscriptionSegment, TranscriptionWord

    rng = random.Random(89)
    for _ in range(200):
        words = tuple(
            TranscriptionWord(
                start=(start := rng.randrange(40) / 10),
                end=start + rng.randrange(10) / 10,
                text=str(i),
            )
            for i in range(20)
        )
        turns = [SpeakerTurn(start=i, end=i + 1, speaker=str(i % 2)) for i in range(5)]
        segment = TranscriptionSegment(id=0, start=0, end=5, text="synthetic", words=words)
        groups: list[tuple[str, list[int]]] = []
        for index, word in enumerate(words):
            speaker = str(min(int((word.start + word.end) / 2), 4) % 2)
            if groups and groups[-1][0] == speaker:
                groups[-1][1].append(index)
            else:
                groups.append((speaker, [index]))
        while len(groups) > 1:
            for i, (_, indices) in enumerate(groups):
                start = 0 if i == 0 else words[indices[0]].start
                end = 5 if i == len(groups) - 1 else words[groups[i + 1][1][0]].start
                if (
                    len(indices) < 2
                    or words[indices[-1]].end - words[indices[0]].start < 0.3
                    or end - start < 0.3
                ):
                    target = max(i - 1, 1 if i == 0 else 0)
                    left, right = sorted((i, target))
                    groups[left : right + 1] = [
                        (groups[target][0], groups[left][1] + groups[right][1])
                    ]
                    compact: list[tuple[str, list[int]]] = []
                    for speaker, indices in groups:
                        if compact and compact[-1][0] == speaker:
                            compact[-1][1].extend(indices)
                        else:
                            compact.append((speaker, indices))
                    groups = compact
                    break
            else:
                break
        actual = split_segment_at_turns(segment, turns, [turn.start for turn in turns])
        assert [tuple(word.text for word in piece.words) for piece in actual] == [
            tuple(words[index].text for index in indices) for _, indices in groups
        ]
        assert all(a is b for a, b in zip((w for p in actual for w in p.words), words, strict=True))
