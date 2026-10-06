"""Deterministic post-decode transcript sanitizer.

This is a backstop for two classes of Whisper artifact that the decode-level
anti-hallucination guards (``condition_on_previous_text=False`` /
``--max-context 0`` / ``--suppress-nst``) reduce but cannot fully eliminate:

1. Repeated-sentence loops — a run of consecutive segments whose text is
   identical after light normalization, repeated many times.
2. Non-speech / credit hallucinations — subtitle-credit cues
   ("Untertitelung des ZDF, 2020", "Amara.org", ...) and punctuation-only
   noise such as long dot-runs.

It runs once at the segment level (before any format rendering) so every output
format stays consistent, and is deliberately conservative: when in doubt it
leaves the segment untouched, biasing toward false negatives over removing
genuine speech.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transcriptions.service import TranscriptionResult, TranscriptionSegment

# Minimum number of consecutive segments with identical normalized text before a
# run is treated as a degenerate loop and collapsed to one instance. Real
# production artifacts repeat 9-84x; natural backchannels ("Ja.", "Genau.") and
# rhetorical repetition occur within a single cue or scattered across the
# transcript, so a threshold of 4 catches the artifacts with comfortable margin.
DEFAULT_REPEAT_THRESHOLD = 4

# A run of 4+ dots (optionally space-separated) is a degenerate artifact. A
# normal ellipsis is exactly three dots ("..." / "…") and is left intact.
_DOT_RUN_PATTERN = re.compile(r"\.(?:\s*\.){3,}")

# Characters stripped from the edges of a segment when building its comparison
# key, so trailing punctuation/case/whitespace differences do not defeat
# loop detection.
_EDGE_CHARS = " \t\r\n.,;:!?-–—…\"'`„“”»«()[]{}"

# Subtitle-credit hallucination family. Each pattern matches the *whole* cue as a
# terminal attribution — a credit ends at the broadcaster name (optionally
# "für funk") with an optional trailing year, or trails off on "amara.org". This
# is what distinguishes a credit from genuine topic talk that merely mentions
# subtitles or a broadcaster and then continues into a real predicate. All of
# these are therefore left untouched: "Das ZDF hat darüber berichtet.",
# "Untertitel sind wichtig für die Barrierefreiheit.", "Untertitel von Filmen
# sind oft schlecht.", "Untertitelung ist ein Handwerk.", "Untertitelung des
# Films war schlecht synchronisiert.", "Untertitel beim ZDF funktionieren
# automatisch." Documented production artifacts are ZDF/funk credits and
# Amara.org; broadcasters outside that observed set are deliberately left alone.
_CREDIT_PATTERNS = (
    re.compile(
        r"^untertitel(?:ung)?(?:\s+im auftrag)?\s+des\s+zdf"
        r"(?:\s+für\s+funk)?(?:[,\s]+\d{4})?$"
    ),
    re.compile(r"^untertitel(?:ung)?\b.*\bamara\.org$"),
    re.compile(r"^amara\.org$"),
)


def sanitize_result(
    result: TranscriptionResult,
    *,
    enabled: bool = True,
    repeat_threshold: int = DEFAULT_REPEAT_THRESHOLD,
) -> TranscriptionResult:
    """Return a sanitized copy of ``result``, or ``result`` itself if unchanged.

    Sanitization only ever removes or collapses artifact segments; genuine
    speech segments are preserved verbatim. When nothing is removed, collapsed,
    or edited the original object is returned so clean transcripts are
    byte-identical to today's output.
    """

    if not enabled or not result.segments:
        return result

    changed = False

    # 1. Strip degenerate dot-runs and drop punctuation-only cues.
    stripped: list[TranscriptionSegment] = []
    for segment in result.segments:
        cleaned_text = _strip_dot_runs(segment.text)
        if not _has_speech(cleaned_text):
            changed = True
            continue
        if cleaned_text != segment.text:
            changed = True
            segment = _with_text(segment, cleaned_text)
        stripped.append(segment)

    # 2. Drop subtitle-credit hallucinations.
    without_credits: list[TranscriptionSegment] = []
    for segment in stripped:
        if _is_credit_hallucination(segment.text):
            changed = True
            continue
        without_credits.append(segment)

    # 3. Collapse consecutive identical-normalized runs.
    collapsed = _collapse_loops(without_credits, repeat_threshold)
    if len(collapsed) != len(without_credits):
        changed = True

    if not changed:
        return result

    rebuilt_text = " ".join(segment.text for segment in collapsed).strip()
    from transcriptions.service import TranscriptionResult as _Result

    return _Result(
        text=rebuilt_text,
        language=result.language,
        segments=collapsed,
        backend_name=result.backend_name,
        model_name=result.model_name,
    )


def _collapse_loops(
    segments: list[TranscriptionSegment],
    repeat_threshold: int,
) -> list[TranscriptionSegment]:
    if repeat_threshold < 2 or len(segments) < repeat_threshold:
        return segments

    collapsed: list[TranscriptionSegment] = []
    index = 0
    count = len(segments)
    while index < count:
        key = _comparison_key(segments[index].text)
        run_end = index + 1
        if key:
            while run_end < count and _comparison_key(segments[run_end].text) == key:
                run_end += 1
        run_length = run_end - index
        if run_length >= repeat_threshold:
            first = segments[index]
            last = segments[run_end - 1]
            collapsed.append(_with_timing(first, start=first.start, end=max(first.end, last.end)))
        else:
            collapsed.extend(segments[index:run_end])
        index = run_end
    return collapsed


def _comparison_key(text: str) -> str:
    return " ".join(text.split()).casefold().strip(_EDGE_CHARS)


def _strip_dot_runs(text: str) -> str:
    if not _DOT_RUN_PATTERN.search(text):
        return text
    stripped = _DOT_RUN_PATTERN.sub(" ", text)
    return " ".join(stripped.split()).strip()


def _has_speech(text: str) -> bool:
    return any(character.isalnum() for character in text)


def _is_credit_hallucination(text: str) -> bool:
    normalized = " ".join(text.split()).casefold().strip(_EDGE_CHARS)
    if not normalized:
        return False
    return any(pattern.match(normalized) for pattern in _CREDIT_PATTERNS)


def _with_text(segment: TranscriptionSegment, text: str) -> TranscriptionSegment:
    from transcriptions.service import TranscriptionSegment as _Segment

    return _Segment(
        id=segment.id,
        start=segment.start,
        end=segment.end,
        text=text,
        speaker=segment.speaker,
    )


def _with_timing(
    segment: TranscriptionSegment, *, start: float, end: float
) -> TranscriptionSegment:
    from transcriptions.service import TranscriptionSegment as _Segment

    return _Segment(
        id=segment.id,
        start=start,
        end=end,
        text=segment.text,
        speaker=segment.speaker,
    )
