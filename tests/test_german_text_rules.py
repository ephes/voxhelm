from __future__ import annotations

import pytest

from synthesis.german_text_rules import normalize_german_text, year_to_words


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Sentence-final numbers are cardinals (or years) and keep their full stop.
        (
            "Das war im Jahr 2024. Danach kam mehr.",
            "Das war im Jahr zweitausendvierundzwanzig. Danach kam mehr.",
        ),
        ("Er wurde 80. Seine Frau auch.", "Er wurde achtzig. Seine Frau auch."),
        ("Die Antwort ist 42.", "Die Antwort ist zweiundvierzig."),
        ("Er kam um 5 vor 12.", "Er kam um 5 vor zwölf."),
        (
            "Im Jahr 1990. Dann kam die Wende.",
            "Im Jahr neunzehnhundertneunzig. Dann kam die Wende.",
        ),
        ("Er zählte bis 1.", "Er zählte bis eins."),
        ("Er wurde 80. 2024 feierte er wieder.", "Er wurde achtzig. 2024 feierte er wieder."),
        ("Sie wurde 3.\nDanach nichts.", "Sie wurde drei.\nDanach nichts."),
        # Real ordinals stay ordinals.
        ("Der 1. Platz geht an Anna.", "Der erste Platz geht an Anna."),
        ("Wir treffen uns am 3. Oktober.", "Wir treffen uns am dritten Oktober."),
        ("Das ist der 2. Versuch.", "Das ist der zweite Versuch."),
        ("Sie kam 3. ins Ziel.", "Sie kam dritter ins Ziel."),
        ("Er kam im 5. Anlauf.", "Er kam im fünften Anlauf."),
        # Prefixed ordinals keep the full stop at a sentence boundary.
        ("Wir treffen uns am 3.", "Wir treffen uns am dritten."),
        (
            "Das Treffen ist am 3. Danach reisen wir ab.",
            "Das Treffen ist am dritten. Danach reisen wir ab.",
        ),
        # Dates, times and label numbers were already right.
        (
            "Am 3.10.2024 war Feiertag.",
            "Am dritten zehnten zweitausendvierundzwanzig war Feiertag.",
        ),
        ("Um 12:30 Uhr.", "Um zwölf Uhr dreißig."),
        ("Gleis Nr. 4.", "Gleis Nummer vier."),
    ],
)
def test_normalize_german_text_numbers(text: str, expected: str) -> None:
    assert normalize_german_text(text) == expected


@pytest.mark.parametrize(
    ("year", "expected"),
    [
        (1990, "neunzehnhundertneunzig"),
        (1800, "achtzehnhundert"),
        (2000, "zweitausend"),
        (2024, "zweitausendvierundzwanzig"),
        (1066, "eintausendsechsundsechzig"),
    ],
)
def test_year_to_words(year: int, expected: str) -> None:
    assert year_to_words(year) == expected
