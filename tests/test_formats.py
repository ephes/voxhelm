from __future__ import annotations

from collections.abc import Iterator
from weakref import WeakSet

import pytest

from transcriptions import formats
from transcriptions.errors import TranscriptionError
from transcriptions.formats import escape_vtt_cue_text, render_vtt
from transcriptions.service import TranscriptionResult, TranscriptionSegment


def _result(*segments: TranscriptionSegment, text: str = "") -> TranscriptionResult:
    return TranscriptionResult(text=text, language="de", segments=list(segments))


def _cue_payloads(vtt: str) -> list[str]:
    blocks = vtt.strip().split("\n\n")
    assert blocks[0] == "WEBVTT"
    payloads = []
    for block in blocks[1:]:
        timing, *payload = block.split("\n")
        assert " --> " in timing
        assert len(payload) == 1, block
        payloads.append(payload[0])
    return payloads


def test_escape_vtt_cue_text_escapes_markup_characters() -> None:
    assert escape_vtt_cue_text("Q&A <b>bold</b> a > b") == (
        "Q&amp;A &lt;b&gt;bold&lt;/b&gt; a &gt; b"
    )


def test_escape_vtt_cue_text_removes_cue_arrow() -> None:
    escaped = escape_vtt_cue_text("a --> b ---> c")
    assert "-->" not in escaped
    assert escaped == "a -&gt; b --&gt; c"


def test_escape_vtt_cue_text_collapses_line_breaks() -> None:
    assert escape_vtt_cue_text(" Zeile eins\n\nZeile zwei\r\nZeile drei ") == (
        "Zeile eins Zeile zwei Zeile drei"
    )


def test_render_vtt_plain_text_is_unchanged() -> None:
    vtt = render_vtt(
        _result(TranscriptionSegment(id=0, start=0.0, end=1.5, text="Hallo und willkommen."))
    )
    assert vtt == "WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nHallo und willkommen.\n"


def test_render_vtt_escapes_payload_and_keeps_one_cue_per_segment() -> None:
    vtt = render_vtt(
        _result(
            TranscriptionSegment(id=0, start=1.0, end=0.5, text=" Q&A: a <b> c --> d"),
            TranscriptionSegment(id=1, start=2.0, end=3.0, text="Zeile eins\n\nZeile zwei"),
        )
    )
    assert vtt == (
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:01.001\nQ&amp;A: a &lt;b&gt; c -&gt; d\n\n"
        "00:00:02.000 --> 00:00:03.000\nZeile eins Zeile zwei\n"
    )
    assert _cue_payloads(vtt) == ["Q&amp;A: a &lt;b&gt; c -&gt; d", "Zeile eins Zeile zwei"]
    assert vtt.count("-->") == 2


def test_render_vtt_skips_empty_segments() -> None:
    vtt = render_vtt(
        _result(
            TranscriptionSegment(id=0, start=0.0, end=1.0, text="   "),
            TranscriptionSegment(id=1, start=1.0, end=2.0, text="\n\n"),
            TranscriptionSegment(id=2, start=2.0, end=3.0, text="Text"),
        )
    )
    assert vtt == "WEBVTT\n\n00:00:02.000 --> 00:00:03.000\nText\n"


def test_render_vtt_orders_cues_by_normalized_start_and_keeps_ties_stable() -> None:
    result = _result(
        TranscriptionSegment(id=0, start=2.0004, end=3, text="First at two"),
        TranscriptionSegment(id=1, start=0, end=1, text="First at zero"),
        TranscriptionSegment(id=2, start=2, end=2.5, text="Second at two"),
        TranscriptionSegment(id=3, start=-1, end=0.5, text="Second at zero"),
    )

    header, *cues = render_vtt(result).strip().split("\n\n")
    assert header == "WEBVTT"
    assert [cue.splitlines() for cue in cues] == [
        ["00:00:00.000 --> 00:00:01.000", "First at zero"],
        ["00:00:00.000 --> 00:00:00.500", "Second at zero"],
        ["00:00:02.000 --> 00:00:03.000", "First at two"],
        ["00:00:02.000 --> 00:00:02.500", "Second at two"],
    ]
    source_order = ["First at two", "First at zero", "Second at two", "Second at zero"]
    assert [cue["text"] for cue in formats.render_verbose_json(result)["segments"]] == source_order
    assert [cue["text"] for cue in formats.render_dote(result)["lines"]] == source_order
    assert [cue["text"] for cue in formats.render_podlove(result)["transcripts"]] == source_order


def test_render_vtt_falls_back_to_single_cue_without_segments() -> None:
    vtt = render_vtt(_result(text="Nur Text & mehr\nzweite Zeile"))
    assert vtt == "WEBVTT\n\n00:00:00.000 --> 00:00:00.001\nNur Text &amp; mehr zweite Zeile\n"


def test_render_vtt_empty_result_has_only_header() -> None:
    assert render_vtt(_result()) == "WEBVTT\n"


@pytest.mark.parametrize("size", [1000, 2000, 4000, 8000])
@pytest.mark.parametrize("line_break", ["", "\r\n \n\t"])
def test_render_vtt_whitespace_is_scanned_once(size: int, line_break: str) -> None:
    class CountedText(str):
        visits = 0

        def __iter__(self) -> Iterator[str]:
            for char in super().__iter__():
                self.visits += 1
                yield char

    # Require the explicit linear scan rather than timing a backtracking regex.
    text = CountedText(" \tA &amp;<" + " " * size + line_break + "\u2003B --> C\t ")
    vtt = render_vtt(_result(TranscriptionSegment(id=0, start=0, end=1, text=text)))
    gap = " " if line_break else " " * size + "\u2003"
    assert _cue_payloads(vtt) == ["A &amp;amp;&lt;" + gap + "B -&gt; C"]
    assert text.visits == len(text)


def test_all_timed_exports_have_positive_quantized_duration():
    from transcriptions.formats import render_dote, render_podlove, render_verbose_json

    pairs = [(0, 0), (1, 0.5), (0.0001, 0.0002), (-2, -1), (1.00049, 1.00051)]
    result = _result(
        *[
            TranscriptionSegment(id=i, start=start, end=end, text=f"Cue {i}")
            for i, (start, end) in enumerate(pairs)
        ]
    )
    expected = [(0, 1), (1000, 1001), (0, 1), (0, 1), (1000, 1001)]
    podlove = render_podlove(result)["transcripts"]
    assert [(cue["start_ms"], cue["end_ms"]) for cue in podlove] == expected
    for cue in podlove:
        assert cue["end_ms"] > cue["start_ms"]
        assert cue["start"] != cue["end"]
    vtt_times = [line for line in render_vtt(result).splitlines() if " --> " in line]
    assert len(vtt_times) == len(pairs)
    assert all(line.split(" --> ")[0] != line.split(" --> ")[1] for line in vtt_times)
    assert all(cue["startTime"] != cue["endTime"] for cue in render_dote(result)["lines"])
    assert [(cue["start"], cue["end"]) for cue in render_verbose_json(result)["segments"]] == pairs


def test_text_only_timed_exports_have_one_millisecond_duration():
    from transcriptions.formats import render_dote, render_podlove

    result = _result(text="Synthetic")
    assert render_podlove(result)["transcripts"][0]["end_ms"] == 1
    assert render_dote(result)["lines"][0]["endTime"] == "00:00:00,001"
    assert "00:00:00.000 --> 00:00:00.001" in render_vtt(result)


@pytest.mark.parametrize("render", [render_vtt, formats.render_dote, formats.render_podlove])
def test_timed_exports_keep_at_most_two_normalized_cues_alive(monkeypatch, render):
    live: WeakSet[formats.SimpleSegment] = WeakSet()
    peak = 0

    class TrackedSegment(formats.SimpleSegment):
        def __init__(self, **kwargs):
            nonlocal peak
            super().__init__(**kwargs)
            live.add(self)
            peak = max(peak, len(live))

    monkeypatch.setattr(formats, "SimpleSegment", TrackedSegment)
    text = "A" + " " * 4096 + "B"
    result = _result(
        *(
            TranscriptionSegment(id=i, start=i, end=i + 1, text=text if i == 0 else "Cue")
            for i in range(1024)
        )
    )

    rendered = render(result)

    if render is render_vtt:
        payloads = _cue_payloads(rendered)
        assert len(payloads) == 1024
        assert payloads[0] == text
    else:
        cues = rendered["lines" if render is formats.render_dote else "transcripts"]
        assert len(cues) == 1024
        assert cues[0]["text"] == text
    assert 0 < peak <= 2
    assert not live


@pytest.mark.parametrize("render", [render_vtt, formats.render_dote, formats.render_podlove])
def test_timed_exports_validate_late_blank_cue_before_normalizing(monkeypatch, render):
    def unexpected_cue(**kwargs):
        pytest.fail("Built a cue before checking the whole transcript")

    monkeypatch.setattr(formats, "SimpleSegment", unexpected_cue)
    result = _result(
        TranscriptionSegment(id=0, start=0, end=1, text="Cue"),
        TranscriptionSegment(id=1, start=1, end=float("inf"), text=" "),
    )

    with pytest.raises(TranscriptionError, match="timestamps must be finite"):
        render(result)


def test_blank_segments_do_not_trigger_text_only_fallback():
    result = _result(
        TranscriptionSegment(id=0, start=0, end=1, text=" \n "),
        text="Synthetic fallback",
    )

    assert render_vtt(result) == "WEBVTT\n"
    assert formats.render_dote(result) == {"lines": []}
    assert formats.render_podlove(result) == {"version": 1, "transcripts": []}
