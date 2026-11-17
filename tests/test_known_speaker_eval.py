"""Known-speaker gold admission and scoring with wholly synthetic evidence."""

import json
import os
import sys
from types import SimpleNamespace

import pytest

from evals import known_speaker_eval as ev


def segment(start=0, end=1, speaker="Alice", text="hello there"):
    return {"start": start, "end": end, "speaker": speaker, "text": text}


def run_args(monkeypatch, tmp_path, gold, limit="0"):
    path = tmp_path / "gold.json"
    path.write_text(json.dumps(gold))
    out = tmp_path / "report.json"
    monkeypatch.setattr(
        sys, "argv", ["eval", "--gold", str(path), "--out", str(out), "--limit", limit]
    )
    return path, out


@pytest.mark.parametrize(
    "gold",
    [
        [],
        {},
        [None],
        [segment(start=float("nan"))],
        [segment(end=float("inf"))],
        [segment(start=True)],
        [segment(start=2, end=1)],
        [segment(end=0)],
        [segment(speaker="")],
        [segment(text=5)],
        [dict(segment(), extra=1)],
    ],
)
def test_bad_gold_refuses_before_model(monkeypatch, tmp_path, gold):
    run_args(monkeypatch, tmp_path, gold)
    monkeypatch.setattr(
        ev, "PyannoteEmbeddingBackend", lambda **kw: pytest.fail("backend acquired")
    )
    monkeypatch.setattr(ev, "build_references", lambda *a: pytest.fail("references embedded"))
    with pytest.raises(SystemExit) as error:
        ev.main()
    assert error.value.code == 2


@pytest.mark.parametrize("kind", ["duplicate", "malformed", "oversize", "fifo", "symlink"])
def test_gold_file_boundary_before_model(monkeypatch, tmp_path, kind):
    path, _ = run_args(monkeypatch, tmp_path, [segment()])
    if kind == "duplicate":
        path.write_text('[{"start":0,"start":0,"end":1,"speaker":"Alice","text":"x"}]')
    elif kind == "malformed":
        path.write_text("[")
    elif kind == "oversize":
        with path.open("wb") as stream:
            stream.truncate(8 * 1024 * 1024 + 1)
    elif kind == "fifo":
        path.unlink()
        os.mkfifo(path)
    else:
        real = tmp_path / "real.json"
        path.rename(real)
        path.symlink_to(real)
    monkeypatch.setattr(
        ev, "PyannoteEmbeddingBackend", lambda **kw: pytest.fail("backend acquired")
    )
    with pytest.raises(SystemExit) as error:
        ev.main()
    assert error.value.code == 2


def test_negative_limit_before_model(monkeypatch, tmp_path):
    run_args(monkeypatch, tmp_path, [segment()], limit="-1")
    monkeypatch.setattr(
        ev, "PyannoteEmbeddingBackend", lambda **kw: pytest.fail("backend acquired")
    )
    with pytest.raises(SystemExit) as error:
        ev.main()
    assert error.value.code == 2


def inert_run(monkeypatch, tmp_path, gold, predicted, limit="0"):
    _path, out = run_args(monkeypatch, tmp_path, gold, limit)
    monkeypatch.setattr(ev, "PyannoteEmbeddingBackend", lambda **kw: object())
    monkeypatch.setattr(ev, "build_references", lambda *a: [])
    monkeypatch.setattr(ev, "decode_mono_16k", lambda *a: [])

    def classify(result, **kwargs):
        return SimpleNamespace(
            segments=[
                SimpleNamespace(
                    start=g.start,
                    end=g.end,
                    speaker=p,
                    speaker_uncertain=False,
                    speaker_candidates=[SimpleNamespace(speaker=p)],
                    speaker_confidence=1.0,
                    speaker_margin=0.8,
                )
                for g, p in zip(result.segments, predicted, strict=True)
            ]
        )

    monkeypatch.setattr(ev, "run_known_speaker_postprocess", classify)
    ev.main()
    return json.loads(out.read_text())


def test_synthetic_accuracy_and_weighted_errors(monkeypatch, tmp_path):
    report = inert_run(
        monkeypatch, tmp_path, [segment(), segment(1, 4, "Bob", "hi")], ["Alice", "Alice"]
    )
    assert report["all_segments_top1_accuracy"] == 0.5
    assert report["der_time_weighted"] == 0.75
    assert report["wder_word_weighted"] == 0.3333
    assert report["auto_accept_policy"]["coverage"] == 1
    assert report["auto_accept_policy"]["accepted_accuracy"] == 0.5
