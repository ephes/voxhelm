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


@pytest.mark.parametrize("raw", ["9" * 5000, "9" * 5000 + ",25", "1," + "9" * 5000])
@pytest.mark.parametrize("suffix", ["kg", "Min.", "EUR", "°C", "°"])
def test_oversized_numeric_tokens_remain_literal(raw: str, suffix: str) -> None:
    # The final EUR abbreviation expansion is independent of number pronunciation.
    expected_suffix = "Euro" if suffix == "EUR" else suffix
    assert normalize_german_text(f"Wert -{raw} {suffix}") == f"Wert -{raw} {expected_suffix}"


def test_numeric_token_digit_budget_boundary() -> None:
    assert normalize_german_text(f"{'0' * 63}1 kg") == "ein Kilogramm"
    assert normalize_german_text(f"{'0' * 64}1 kg") == f"{'0' * 64}1 kg"
    # Count both sides of the separator, including the specialized EUR rule.
    assert normalize_german_text(f"{'0' * 61}1,05 EUR") == "ein Euro fünf"
    assert normalize_german_text(f"{'0' * 62}1,05 EUR") == f"{'0' * 62}1,05 Euro"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Text __GODE_CARDINAL_DOT__ Ende", "Text __GODE_CARDINAL_DOT__ Ende"),
        (
            "__GODE_CARDINAL_DOT__\t__GODE_CARDINAL_DOT__\nEnde",
            "__GODE_CARDINAL_DOT__\t__GODE_CARDINAL_DOT__\nEnde",
        ),
        (
            "Kapitel 2. __GODE_CARDINAL_DOT__ Kapitel 3. lesen",
            "Kapitel zwei. __GODE_CARDINAL_DOT__ Kapitel drei. lesen",
        ),
        ("Kapitel 2. lesen, Seite 4. Danach.", "Kapitel zwei. lesen, Seite vier. Danach."),
        ("Lfd. Nr. 2. und Gleis Nr. 4.", "laufende Nummer zwei. und Gleis Nummer vier."),
    ],
)
def test_cardinal_labels_preserve_literal_marker_text(text: str, expected: str) -> None:
    assert normalize_german_text(text) == expected
