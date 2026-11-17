"""Synthetic CLI probe regressions: no models, providers or real media."""

import argparse

import numpy as np
import pytest

from evals import mixed_speaker_segments_probe as probe
from transcriptions import known_speaker as ks


def subwindow_args():
    return probe.build_parser().parse_args(
        [
            "subwindow",
            "audio.wav",
            "reference.wav",
            "--reference",
            "Alice=0-2",
            "--segment",
            "0-0.1",
            "--target",
            "0-2",
            "--window",
            "1-0.5",
        ]
    )


@pytest.mark.parametrize("value", ["nan-inf", "2-1", "0-nan", "0-inf", "1-1", "-1-2", "oops"])
def test_range_admission(value):
    with pytest.raises(argparse.ArgumentTypeError):
        probe.parse_range(value)


@pytest.mark.parametrize("field", ["reference", "segment", "target"])
def test_source_extents_refuse_before_backend(monkeypatch, field):
    args = subwindow_args()
    setattr(args, field, [("Alice", 0, 3)] if field == "reference" else [(0, 3)])
    monkeypatch.setattr(ks, "decode_mono_16k", lambda path: np.ones(32000))
    monkeypatch.setattr(ks, "get_known_speaker_backend", lambda *a: pytest.fail("backend acquired"))
    with pytest.raises(ValueError, match="source interval"):
        probe.run_subwindow(args)


def test_exact_end_and_short_segment_keep_padding(monkeypatch):
    args = subwindow_args()
    calls = []

    class Backend:
        def embed(self, samples, sample_rate):
            calls.append(len(samples))
            return [1.0, 0.0]

    monkeypatch.setattr(ks, "decode_mono_16k", lambda path: np.ones(32000))
    monkeypatch.setattr(ks, "get_known_speaker_backend", lambda *a: Backend())
    reports = []
    monkeypatch.setattr(probe, "write_report", lambda report, out: reports.append(report))
    probe.run_subwindow(args)
    assert reports[0]["whole_segments"][0]["end"] == 0.1
    assert calls[1] >= int(ks.KnownSpeakerConfig().min_segment_duration * 16000)
    assert [(r["start"], r["end"]) for r in next(iter(reports[0]["windows"].values()))] == [
        (0, 1),
        (0.5, 1.5),
        (1, 2),
    ]


@pytest.mark.parametrize("case", ["stalled", "overflow", "aggregate"])
def test_window_expansion_refuses_before_decode_or_backend(monkeypatch, case):
    args = subwindow_args()
    if case == "stalled":
        args.target = [(1000, 1002)]
        args.window = [(1, 1e-300)]
    elif case == "overflow":
        args.window = [(1, 1e-308)]
    else:
        args.target = [(0, 2)] * 4000  # Each fits three windows; whole-run cap applies.
    monkeypatch.setattr(ks, "decode_mono_16k", lambda *a: pytest.fail("decoded before preflight"))
    monkeypatch.setattr(ks, "get_known_speaker_backend", lambda *a: pytest.fail("backend acquired"))
    with pytest.raises(ValueError, match="window expansion"):
        probe.run_subwindow(args)


def test_integer_window_order_and_endpoint(monkeypatch):
    args = subwindow_args()
    args.window = [(0.2, 0.1), (3, 1)]
    args.target = [(0, 0.4), (1, 1.4)]
    plans = probe.prepare_subwindows(args)
    assert [len(bounds) for _key, bounds in plans] == [3, 0, 3, 0]
    assert plans[0][1] == [(0, 0.2), (0.1, 0.30000000000000004), (0.2, 0.4)]
    assert plans[2][1][-1] == (1.2, 1.4)
