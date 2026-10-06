from __future__ import annotations

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
        "00:00:01.000 --> 00:00:01.000\nQ&amp;A: a &lt;b&gt; c -&gt; d\n\n"
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


def test_render_vtt_falls_back_to_single_cue_without_segments() -> None:
    vtt = render_vtt(_result(text="Nur Text & mehr\nzweite Zeile"))
    assert vtt == "WEBVTT\n\n00:00:00.000 --> 00:00:00.000\nNur Text &amp; mehr zweite Zeile\n"


def test_render_vtt_empty_result_has_only_header() -> None:
    assert render_vtt(_result()) == "WEBVTT\n"
