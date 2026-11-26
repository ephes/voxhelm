from __future__ import annotations

import pytest

from synthesis.german_text_rules import normalize_german_text, year_to_words


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Das Beispiel heißt z. Bausch.", "Das Beispiel heißt z. Bausch."),
        ("Das Beispiel heißt z.Bausch.", "Das Beispiel heißt z.Bausch."),
        ("Das gilt z. B. heute.", "Das gilt zum Beispiel heute."),
        ("Das gilt z.B., auch morgen.", "Das gilt zum Beispiel, auch morgen."),
        ("Das gilt zB heute.", "Das gilt zum Beispiel heute."),
        ("Das gilt z. B..", "Das gilt zum Beispiel."),
        ("Das gilt z.\nB. heute.", "Das gilt zum Beispiel heute."),
        ("Das gilt z.", "Das gilt zum Beispiel"),
        ("B. heute.", "heute."),
    ],
)
def test_zum_beispiel_abbreviation_requires_a_complete_b(text: str, expected: str) -> None:
    assert normalize_german_text(text) == expected


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
    "text",
    [
        "Version 1.13.2024.",
        "Version 1.2.3.4.",
        "Am 0.12.2024.",
        "Am 1.0.",
        "Am 31.4.2024.",
        "Am 29.2.2023.",
        "Am 31. April 2024.",
        "Am 29. FEBRUAR 2023.",
        "Am 30. Februar.",
        "Im 31. April 2024.",
        "Nach der 31. April 2024.",
        "Am 29. Februar 1900 war es.",
        "Am 29.2.2100.",
        "Am 29.2. 2100 war es.",
        "Am 31. April 1990 war es.",
        # A year followed by other punctuation still decides the check.
        "Am 29. Februar 1900; danach.",
        "Am 29. Februar 2023; danach.",
        "Am 29.2. 2023; danach.",
        "am 31. April; danach.",
        "am 31. April: danach.",
        "Am 31. April 2024) danach.",
    ],
)
def test_invalid_german_dates_remain_literal(text: str) -> None:
    assert normalize_german_text(text) == text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # A valid named date after an ordinal is read as its spoken ordinal word.
        ("Platz 1. 3. Mai war gut.", "Platz erster dritter Mai war gut."),
        ("Platz 1. 31. April war gut.", "Platz eins. 31. April war gut."),
    ],
)
def test_ordinal_before_named_date(text: str, expected: str) -> None:
    assert normalize_german_text(text) == expected


@pytest.mark.parametrize("text", ["Am 3.4.1990.", "Am 3.4.3000."])
def test_glued_non_20xx_years_are_not_expanded(text: str) -> None:
    # The numeric date rule only accepts a directly attached 20xx year.
    assert normalize_german_text(text) == text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Am 29.2.2024 geht es los.",
            "Am neunundzwanzigsten zweiten zweitausendvierundzwanzig geht es los.",
        ),
        (
            "Am 29. FeBrUaR 2024 geht es los.",
            "Am neunundzwanzigsten FeBrUaR zweitausendvierundzwanzig geht es los.",
        ),
        ("Am 29.2. geht es los.", "Am neunundzwanzigsten zweiten geht es los."),
        ("Am 29. Februar geht es los.", "Am neunundzwanzigsten Februar geht es los."),
        # Years outside 2000-2099 are not spoken, but still validate the day.
        ("Am 1. Mai 1990 war es.", "Am ersten Mai 1990 war es."),
        ("Am 1. Mai 0012 war es.", "Am ersten Mai 0012 war es."),
        ("Am 1.5. 0000 war es.", "Am ersten fünften 0000 war es."),
        ("Am 29. Februar 1996 war es.", "Am neunundzwanzigsten Februar 1996 war es."),
        ("Am 29.2. 1996 war es.", "Am neunundzwanzigsten zweiten 1996 war es."),
        (
            "Am 29. Februar 2024; danach.",
            "Am neunundzwanzigsten Februar zweitausendvierundzwanzig; danach.",
        ),
        # A named date ends at any non-word character after the month.
        ("Am 3. Mai: danach.", "Am dritten Mai: danach."),
        ("(3. Mai) war gut.", "(dritter Mai) war gut."),
        # A number glued to letters is not a year.
        (
            "Am 29. Februar 2023er Riesling verkosten.",
            "Am neunundzwanzigsten Februar 2023er Riesling verkosten.",
        ),
        ("Am 29.2. 2023er Riesling.", "Am neunundzwanzigsten zweiten 2023er Riesling."),
    ],
)
def test_valid_german_dates_are_spoken(text: str, expected: str) -> None:
    assert normalize_german_text(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Stand 12:60 im Spiel.",
        "Wert 25:30 Uhr.",
        "Wert 12:99. Danach geht es weiter.",
        "Wert 24:00.",
        "Protokoll 1:02:03.",
        "Protokoll 12:30.5.",
    ],
)
def test_non_clock_colon_notation_remains_literal(text: str) -> None:
    assert normalize_german_text(text) == text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Um 00:00.", "Um null Uhr."),
        ("Um 23:59 Uhr.", "Um dreiundzwanzig Uhr neunundfünfzig."),
        ("Um 1:00 Uhr.", "Um eins Uhr."),
        (
            "Um 01:30 Uhr kommt 1 kg Mehl dazu.",
            "Um eins Uhr dreißig kommt ein Kilogramm Mehl dazu.",
        ),
    ],
)
def test_valid_german_clock_times_are_spoken(text: str, expected: str) -> None:
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


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1.234,56 EUR", "eintausendzweihundertvierunddreißig Euro sechsundfünfzig"),
        ("1.234 EUR", "eintausendzweihundertvierunddreißig Euro"),
        ("234,56 EUR", "zweihundertvierunddreißig Euro sechsundfünfzig"),
        ("-1.234,05 EUR", "minus eintausendzweihundertvierunddreißig Euro fünf"),
        ("-1 EUR", "minus ein Euro"),
        ("1.05 EUR", "ein Euro fünf"),
        ("1.23,45 EUR", "1.23,45 Euro"),
        ("1.2.2024 EUR", "1.2.2024 Euro"),
        ("1.2.2024,56 EUR", "1.2.2024,56 Euro"),
        ("1..234,56 EUR", "1..234,56 Euro"),
    ],
)
def test_euros_preserve_the_complete_amount(text: str, expected: str) -> None:
    assert normalize_german_text(text) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1", "eine Stunde"),
        ("-1", "minus eine Stunde"),
        ("1,00", "eins komma null null Stunde"),
        ("-1.00", "minus eins komma null null Stunde"),
        (f"1,{'0' * 18}1", f"eins komma {'null ' * 18}eins Stunden"),
    ],
)
def test_unit_inflection_uses_exact_quantity(raw: str, expected: str) -> None:
    assert normalize_german_text(f"{raw} Std.") == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1.234,5 kg", "eintausendzweihundertvierunddreißig komma fünf Kilogramm"),
        ("-1.234,5 kg", "minus eintausendzweihundertvierunddreißig komma fünf Kilogramm"),
        ("1.000 °C", "eintausend Grad Celsius"),
        ("1.000°", "eintausend Grad"),
        ("1.000.000 g", "1000000 Gramm"),
        ("1,5 kg", "eins komma fünf Kilogramm"),
        ("12.45 kg", "zwölf komma vier fünf Kilogramm"),
        ("0.125 kg", "null komma eins zwei fünf Kilogramm"),
        ("-1.5 °C", "minus eins komma fünf Grad Celsius"),
        ("1.23,5 kg", "1.23,5 kg"),
        ("1,23.45°", "1,23.45°"),
        ("1.2.2024,5 °C", "1.2.2024,5 °C"),
    ],
)
def test_physical_quantities_preserve_grouping(text: str, expected: str) -> None:
    assert normalize_german_text(text) == expected


def test_grouped_quantity_digit_budget() -> None:
    # Dots do not count against the budget, but every digit does.
    at_limit = "1" + ".234" * 21
    over_limit = "12" + ".345" * 21
    assert normalize_german_text(f"{at_limit} kg") == f"{at_limit.replace('.', '')} Kilogramm"
    assert normalize_german_text(f"{over_limit} kg") == f"{over_limit} kg"


@pytest.mark.parametrize(
    ("quantity", "expected"),
    [
        ("5 °C", "minus fünf Grad Celsius"),
        ("5°", "minus fünf Grad"),
        ("1 Std.", "minus eine Stunde"),
        ("1,05 EUR", "minus ein Euro fünf"),
        ("1.234,5 kg", "minus eintausendzweihundertvierunddreißig komma fünf Kilogramm"),
        ("1.234,05 EUR", "minus eintausendzweihundertvierunddreißig Euro fünf"),
        (f"{'0' * 63}1 kg", "minus ein Kilogramm"),
    ],
)
def test_unicode_minus_matches_ascii_for_admitted_quantities(quantity: str, expected: str) -> None:
    assert normalize_german_text(f"-{quantity}") == expected
    assert normalize_german_text(f"−{quantity}") == expected


@pytest.mark.parametrize(
    ("suffix", "expected_suffix"), [("kg", "kg"), ("°C", "°C"), ("EUR", "Euro")]
)
def test_unicode_minus_on_oversized_quantities_remains_literal(
    suffix: str, expected_suffix: str
) -> None:
    raw = "12" + ".345" * 21
    assert normalize_german_text(f"−{raw} {suffix}") == f"−{raw} {expected_suffix}"


def test_unicode_minus_outside_admitted_quantities_remains_literal() -> None:
    text = "Nord-Süd − Südost bei −5 und −1.23,5 kg"
    assert normalize_german_text(text) == text
