from __future__ import annotations

from transcriptions import formats
from transcriptions.sanitizer import sanitize_result
from transcriptions.service import TranscriptionResult, TranscriptionSegment


def _result(segments: list[TranscriptionSegment]) -> TranscriptionResult:
    return TranscriptionResult(
        text=" ".join(segment.text for segment in segments),
        language="de",
        segments=segments,
        backend_name="whisper.cpp",
        model_name="ggml-large-v3.bin",
    )


def test_collapses_repeated_sentence_loop_to_single_segment() -> None:
    # "Das ist auch sehr subjektiv." repeated 18x consecutively (real production case).
    segments = [
        TranscriptionSegment(id=index, start=float(index), end=float(index + 1),
                             text="Das ist auch sehr subjektiv.")
        for index in range(18)
    ]
    result = sanitize_result(_result(segments))

    assert len(result.segments) == 1
    collapsed = result.segments[0]
    assert collapsed.text == "Das ist auch sehr subjektiv."
    assert collapsed.start == 0.0
    assert collapsed.end == 18.0
    assert result.text == "Das ist auch sehr subjektiv."


def test_collapses_loop_with_light_normalization_differences() -> None:
    # 9x repeat of the same clause; surrounding whitespace/case/punctuation differs slightly.
    base = "es ist ja auch ein bisschen von der Sprache abhängig, weil"
    variants = [
        base,
        "  es ist ja auch ein bisschen von der Sprache abhängig, weil ",
        "Es ist ja auch ein bisschen von der Sprache abhängig, weil.",
    ]
    segments = [
        TranscriptionSegment(id=index, start=float(index), end=float(index + 1),
                             text=variants[index % len(variants)])
        for index in range(9)
    ]
    result = sanitize_result(_result(segments))

    assert len(result.segments) == 1
    assert result.segments[0].start == 0.0
    assert result.segments[0].end == 9.0


def test_preserves_speech_around_collapsed_loop_with_monotonic_timestamps() -> None:
    segments = [
        TranscriptionSegment(id=0, start=0.0, end=2.0, text="Vorher echte Rede."),
        *[
            TranscriptionSegment(id=i, start=2.0 + i, end=3.0 + i,
                                 text="und dann muss ich halt den Fall unterscheiden")
            for i in range(1, 6)
        ],
        TranscriptionSegment(id=6, start=8.0, end=10.0, text="Nachher echte Rede."),
    ]
    result = sanitize_result(_result(segments))

    texts = [segment.text for segment in result.segments]
    assert texts == [
        "Vorher echte Rede.",
        "und dann muss ich halt den Fall unterscheiden",
        "Nachher echte Rede.",
    ]
    starts = [segment.start for segment in result.segments]
    ends = [segment.end for segment in result.segments]
    assert starts == sorted(starts)
    for start, end in zip(starts, ends, strict=True):
        assert end >= start


def test_drops_zdf_subtitle_credit_hallucinations() -> None:
    segments = [
        TranscriptionSegment(id=0, start=0.0, end=3.0, text="Echte Aussage zum Thema."),
        TranscriptionSegment(id=1, start=3.0, end=33.0, text="Untertitelung des ZDF, 2020"),
        TranscriptionSegment(id=2, start=33.0, end=63.0,
                             text="Untertitelung des ZDF für funk, 2017"),
        TranscriptionSegment(id=3, start=63.0, end=66.0, text="Und weiter im Gespräch."),
    ]
    result = sanitize_result(_result(segments))

    assert [segment.text for segment in result.segments] == [
        "Echte Aussage zum Thema.",
        "Und weiter im Gespräch.",
    ]


def test_drops_amara_and_untertitel_von_credits() -> None:
    segments = [
        TranscriptionSegment(id=0, start=0.0, end=2.0, text="Inhalt."),
        TranscriptionSegment(id=1, start=2.0, end=5.0, text="Untertitel von Amara.org"),
        TranscriptionSegment(id=2, start=5.0, end=8.0,
                             text="Untertitel im Auftrag des ZDF, 2018"),
    ]
    result = sanitize_result(_result(segments))

    assert [segment.text for segment in result.segments] == ["Inhalt."]


def test_drops_dot_run_punctuation_only_cue_and_keeps_real_continuation() -> None:
    dot_run = " ".join("." for _ in range(220))
    segments = [
        TranscriptionSegment(id=0, start=12.0, end=12.0, text=dot_run),
        TranscriptionSegment(id=1, start=12.0, end=15.0,
                             text="typischerweise halt ... um halt aus dem Pfad heraus"),
    ]
    result = sanitize_result(_result(segments))

    assert [segment.text for segment in result.segments] == [
        "typischerweise halt ... um halt aus dem Pfad heraus",
    ]


def test_strips_long_dot_run_but_keeps_real_text_in_same_segment() -> None:
    dot_run = " ".join("." for _ in range(200))
    segments = [
        TranscriptionSegment(id=0, start=10.0, end=10.0,
                             text=f"typischerweise halt {dot_run}"),
    ]
    result = sanitize_result(_result(segments))

    assert len(result.segments) == 1
    assert result.segments[0].text == "typischerweise halt"


# --- Negative cases: genuine speech must be preserved untouched ---


def test_preserves_stammer_inside_single_cue() -> None:
    segments = [
        TranscriptionSegment(id=0, start=0.0, end=2.0, text="so ein, so ein, so ein Problem"),
    ]
    result = sanitize_result(_result(segments))

    assert result.segments == segments


def test_preserves_rhetorical_repetition_inside_single_cue() -> None:
    segments = [
        TranscriptionSegment(id=0, start=0.0, end=2.0, text="Ticket, Ticket, Ticket!"),
    ]
    result = sanitize_result(_result(segments))

    assert result.segments == segments


def test_preserves_scattered_backchannels() -> None:
    segments = [
        TranscriptionSegment(id=0, start=0.0, end=1.0, text="Ja."),
        TranscriptionSegment(id=1, start=1.0, end=4.0, text="Das sehe ich auch so."),
        TranscriptionSegment(id=2, start=4.0, end=5.0, text="Genau."),
        TranscriptionSegment(id=3, start=5.0, end=8.0, text="Aber es kommt darauf an."),
        TranscriptionSegment(id=4, start=8.0, end=9.0, text="Ja."),
    ]
    result = sanitize_result(_result(segments))

    assert result.segments == segments


def test_preserves_short_consecutive_backchannel_run_below_threshold() -> None:
    # Three consecutive "Ja." is below the conservative collapse threshold.
    segments = [
        TranscriptionSegment(id=0, start=0.0, end=1.0, text="Ja."),
        TranscriptionSegment(id=1, start=1.0, end=2.0, text="Ja."),
        TranscriptionSegment(id=2, start=2.0, end=3.0, text="Ja."),
    ]
    result = sanitize_result(_result(segments))

    assert result.segments == segments


def test_preserves_signoffs_and_topic_talk_about_subtitles() -> None:
    segments = [
        TranscriptionSegment(id=0, start=0.0, end=2.0, text="Bis zum nächsten Mal."),
        TranscriptionSegment(id=1, start=2.0, end=3.0, text="Tschüss."),
        TranscriptionSegment(id=2, start=3.0, end=8.0,
                             text="Das ZDF hat darüber ja auch berichtet."),
        TranscriptionSegment(id=3, start=8.0, end=13.0,
                             text="Untertitel sind wichtig für die Barrierefreiheit."),
    ]
    result = sanitize_result(_result(segments))

    assert result.segments == segments


def test_preserves_genuine_speech_about_subtitling() -> None:
    # Genuine topic talk that merely starts with "Untertitel"/"Untertitelung" but
    # carries no broadcaster/site credit token must survive.
    segments = [
        TranscriptionSegment(id=0, start=0.0, end=4.0,
                             text="Untertitel von Filmen sind oft schlecht synchronisiert."),
        TranscriptionSegment(id=1, start=4.0, end=8.0,
                             text="Untertitelung ist ein unterschätztes Handwerk."),
        TranscriptionSegment(id=2, start=8.0, end=12.0,
                             text="Untertitel der ARD waren früher Teletext."),
    ]
    result = sanitize_result(_result(segments))

    assert result.segments == segments


def test_returns_input_unchanged_when_nothing_to_sanitize() -> None:
    segments = [
        TranscriptionSegment(id=0, start=0.0, end=2.0, text="Ein ganz normaler Satz."),
        TranscriptionSegment(id=1, start=2.0, end=4.0, text="Und noch einer."),
    ]
    original = _result(segments)
    result = sanitize_result(original)

    assert result is original


def test_disabled_sanitizer_is_a_no_op() -> None:
    segments = [
        TranscriptionSegment(id=index, start=float(index), end=float(index + 1),
                             text="Das ist auch sehr subjektiv.")
        for index in range(18)
    ]
    original = _result(segments)
    result = sanitize_result(original, enabled=False)

    assert result is original


def test_all_output_formats_reflect_sanitized_segments() -> None:
    dot_run = " ".join("." for _ in range(220))
    segments = [
        TranscriptionSegment(id=0, start=0.0, end=2.0, text="Echte Aussage."),
        *[
            TranscriptionSegment(id=i, start=2.0 + i, end=3.0 + i,
                                 text="Das ist auch sehr subjektiv.")
            for i in range(1, 19)
        ],
        TranscriptionSegment(id=19, start=21.0, end=51.0, text="Untertitelung des ZDF, 2020"),
        TranscriptionSegment(id=20, start=51.0, end=51.0, text=dot_run),
        TranscriptionSegment(id=21, start=51.0, end=53.0, text="Schlusswort."),
    ]
    result = sanitize_result(_result(segments))

    expected_texts = ["Echte Aussage.", "Das ist auch sehr subjektiv.", "Schlusswort."]
    assert [segment.text for segment in result.segments] == expected_texts

    verbose = formats.render_verbose_json(result)
    assert [segment["text"] for segment in verbose["segments"]] == expected_texts
    assert "ZDF" not in verbose["text"]

    text = formats.render_text(result)
    assert "ZDF" not in text
    assert text.count("Das ist auch sehr subjektiv.") == 1

    vtt = formats.render_vtt(result)
    assert "ZDF" not in vtt
    assert vtt.count("Das ist auch sehr subjektiv.") == 1

    dote = formats.render_dote(result)
    assert [line["text"] for line in dote["lines"]] == expected_texts

    podlove = formats.render_podlove(result)
    assert [entry["text"] for entry in podlove["transcripts"]] == expected_texts


def test_repeat_threshold_is_configurable() -> None:
    segments = [
        TranscriptionSegment(id=index, start=float(index), end=float(index + 1),
                             text="Ja.")
        for index in range(3)
    ]
    # With threshold 3 the run collapses; the default (4) would leave it intact.
    result = sanitize_result(_result(segments), repeat_threshold=3)

    assert len(result.segments) == 1
