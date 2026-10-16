"""German drift check of the ``ad-analysis-v1`` profile: find and repair English drift.

Parakeet sometimes drifts into English inside German speech (about 8-10 % of German 30 s
windows in the ad research). The profile repairs it with the frozen research rule
(``work/night-2026-10-14/round2-german-drift`` of the ad research workspace, FREEZE.txt
of 2026-10-15, the post-hoc acceptance rule of its REPORT.md):

1. Grid: the 30 s windows ``[30k, 30k + 30)`` of file time over the Parakeet word
   starts (``lib.py`` ``grid``); only windows holding at least one word exist.
2. Check: count English and German function words in the window's text with the
   research lists and tokenizer (``lib.py`` ``EN``, ``DE``, ``toks``, ``counts``); flag
   the window when English >= 4 and English / (English + German) >= 0.10.
3. Retry: re-transcribe each flagged window with whisper large-v3-turbo on
   ``[t - 1, t + 31]``, language detected on the clip (``transcribe_window``).
4. Accept: only when the detected language is the episode language with p >= 0.9;
   otherwise Parakeet's words stay.
5. Splice: every Parakeet word starting in ``[t, t + 30)`` of an accepted window is
   replaced by the retry's words, which start in the same range (``splice.py``).

It runs only for the profile's languages (German); other jobs get a record saying the
check did not run. Every window gets a record (counts, flag, retry outcome, word counts,
retry time), returned with the words. A failing or unavailable retry fails the run: the
check never silently skips a window.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

from transcriptions.windows import WindowTranscription, raise_if_cancelled, transcribe_window

# The research lists (``lib.py``, FREEZE.txt sha256 5948f803...), verbatim. English words
# that are also common German words (in, so, was, an, also, man, will, hat, war, die,
# bin, um) are left out by subtracting the German list, as ``lib.py`` does. REPORT.md
# suggests dropping "these" after the holdout; the frozen rule keeps it.
_ENGLISH_TEXT = """the and you that this with for have what are but they just like it is of to we
our your be been there their which would could should
about from not do does did it's i'm don't can if or at on by my me he she them these those
because when where how who then than really very
going know think yeah get got gonna you're we're they're that's there's let's isn't doesn't
didn't can't won't i've i'll you've into out
some any more most other our us its his her has had were said say one"""
_GERMAN_TEXT = """der die das und ist nicht ich wir sie es ein eine einen einem einer dem den
des mit auf für von zu auch aber dass wie wenn noch nur schon
dann oder sich mal ja sind haben wird werden kann bei aus nach jetzt hier mehr sehr immer gibt
genau halt eben doch du ihr uns mir mich dir dich
wo weil über vor durch gerade eigentlich irgendwie hat habe bin bist war waren wurde würde
können müssen muss soll diese dieser dieses
kein keine nichts etwas viel viele ganz gut also man um"""
GERMAN_FUNCTION_WORDS = frozenset(_GERMAN_TEXT.split())
ENGLISH_FUNCTION_WORDS = frozenset(_ENGLISH_TEXT.split()) - GERMAN_FUNCTION_WORDS
FUNCTION_WORDS_SHA256 = hashlib.sha256(
    json.dumps(
        {"en": sorted(ENGLISH_FUNCTION_WORDS), "de": sorted(GERMAN_FUNCTION_WORDS)},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
_NOT_A_WORD_CHARACTER = re.compile(r"[^a-z0-9äöüß' ]")

WindowRetry = Callable[..., WindowTranscription]


@dataclass(frozen=True)
class DriftRule:
    """Every setting of the drift check; a change is a new profile name."""

    rule: str = "fw-en4-share10-v1"
    languages: tuple[str, ...] = ("de",)
    window_seconds: float = 30.0
    min_english: int = 4
    min_english_share: float = 0.10
    accept_probability: float = 0.9

    def flags(self, english: int, german: int) -> bool:
        """The research rule (``splice.py``): ``e >= E and e / max(1, e + g) >= S``."""
        return (
            english >= self.min_english
            and english / max(1, english + german) >= self.min_english_share
        )

    def record(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "languages": list(self.languages),
            "window_seconds": self.window_seconds,
            "min_english": self.min_english,
            "min_english_share": self.min_english_share,
            "function_words_sha256": FUNCTION_WORDS_SHA256,
            "accept_language": "episode",
            "accept_probability": self.accept_probability,
        }


DRIFT_RULE = DriftRule()


@dataclass(frozen=True)
class DriftCheck:
    words: list[dict[str, Any]]
    record: dict[str, Any]


def function_word_tokens(text: str) -> list[str]:
    """``lib.py`` ``toks``: lower-case, keep letters, digits, umlauts, ß and apostrophes."""
    normalized = _NOT_A_WORD_CHARACTER.sub(" ", text.lower().replace("’", "'"))
    return [
        token.strip("'") if token not in ENGLISH_FUNCTION_WORDS else token
        for token in normalized.split()
    ]


def count_function_words(words: list[str]) -> tuple[int, int]:
    """English and German function words in the words' text (``counts(toks(' '.join(...)))``)."""
    tokens = function_word_tokens(" ".join(words))
    english = sum(token in ENGLISH_FUNCTION_WORDS for token in tokens)
    german = sum(token in GERMAN_FUNCTION_WORDS for token in tokens)
    return english, german


def window_index(start: float, window_seconds: float) -> int:
    return int(start // window_seconds)


def grid_windows(
    words: list[dict[str, Any]], window_seconds: float
) -> dict[int, list[dict[str, Any]]]:
    """Words by grid window of their start, in input order (``lib.py`` ``grid``)."""
    grid: dict[int, list[dict[str, Any]]] = {}
    for word in words:
        grid.setdefault(window_index(word["s"], window_seconds), []).append(word)
    return grid


def splice_words(
    words: list[dict[str, Any]],
    replacements: dict[int, list[dict[str, Any]]],
    window_seconds: float,
) -> list[dict[str, Any]]:
    """``splice.py``: drop the words starting in each replaced window, add its retry words.

    The result is sorted by start; the sort is stable, so words with equal starts keep
    their order.
    """
    kept = [word for word in words if window_index(word["s"], window_seconds) not in replacements]
    added = [word for index in sorted(replacements) for word in replacements[index]]
    spliced = kept + added
    spliced.sort(key=lambda word: word["s"])
    return spliced


def empty_summary() -> dict[str, Any]:
    return {"windows": 0, "flagged": 0, "accepted": 0, "rejected": 0, "retry_seconds": 0.0}


def run_drift_check(
    audio_path: Path,
    words: list[dict[str, Any]],
    *,
    language: str | None,
    rule: DriftRule = DRIFT_RULE,
    cancel_event: threading.Event | None = None,
    retry: WindowRetry | None = None,
) -> DriftCheck:
    """Flag, retry and splice the drifted windows of ``words`` (Parakeet ``{w, s, e, c}``).

    Retry words carry ``c: null`` (whisper's token probabilities are not Parakeet
    confidences). Raises what the retry raises (``BackendUnavailableError`` when the
    turbo runtime went away, ``RuntimeError`` on a failed window, ``InferenceCancelled``).
    ``retry`` defaults to ``transcribe_window``.
    """
    retry_window = retry or transcribe_window
    if language not in rule.languages:
        checked = ", ".join(rule.languages)
        reason = (
            f"language_hint {language!r} is not checked (checked: {checked})."
            if language
            else f"no language_hint (checked: {checked})."
        )
        return DriftCheck(
            words=words,
            record={
                "rule": rule.rule,
                "checked": False,
                "language": language,
                "reason": reason,
                "summary": empty_summary(),
                "windows": [],
            },
        )

    summary = empty_summary()
    records: list[dict[str, Any]] = []
    replacements: dict[int, list[dict[str, Any]]] = {}
    retry_seconds = 0.0
    for index, window_words in sorted(grid_windows(words, rule.window_seconds).items()):
        english, german = count_function_words([word["w"] for word in window_words])
        flagged = rule.flags(english, german)
        record: dict[str, Any] = {
            "t": index * rule.window_seconds,
            "en": english,
            "de": german,
            "flagged": flagged,
            "words_before": len(window_words),
            "words_after": len(window_words),
        }
        if flagged:
            raise_if_cancelled(cancel_event)
            started = monotonic()
            result = retry_window(
                audio_path, index * rule.window_seconds, cancel_event=cancel_event
            )
            elapsed = monotonic() - started
            retry_seconds += elapsed
            accepted = (
                result.language == language
                and result.language_probability >= rule.accept_probability
            )
            record.update(
                {
                    "language": result.language,
                    "p": result.language_probability,
                    "accepted": accepted,
                    "retry_words": len(result.words),
                    "retry_seconds": round(elapsed, 3),
                }
            )
            if accepted:
                replacements[index] = [
                    {"w": word["w"], "s": word["s"], "e": word["e"], "c": None}
                    for word in result.words
                ]
                record["words_after"] = len(result.words)
            summary["flagged"] += 1
            summary["accepted" if accepted else "rejected"] += 1
        records.append(record)
    raise_if_cancelled(cancel_event)
    summary["windows"] = len(records)
    summary["retry_seconds"] = round(retry_seconds, 3)
    return DriftCheck(
        words=splice_words(words, replacements, rule.window_seconds),
        record={
            "rule": rule.rule,
            "checked": True,
            "language": language,
            "reason": None,
            "summary": summary,
            "windows": records,
        },
    )
