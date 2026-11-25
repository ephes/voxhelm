# Vendored German TTS text-normalization rules for the Kokoro "Martin" model.
#
# Source: Godelaune/Kokoro-82M-ONNX-German-Martin (Apache-2.0)
#   https://huggingface.co/Godelaune/Kokoro-82M-ONNX-German-Martin
#   revision a1cba7fbf0e72fbae38f0a3a48ce0dc8e6077804
#   upstream files: german_text_rules.py (data tables) and
#                   onnx-docker/tts_normalizer.py (normalization logic)
#
# The two upstream files are combined here into a single authoritative runtime
# source so deployed preprocessing can never drift from the tested code. Applied
# BEFORE espeak phonemization, for German synthesis only. Only the tables and
# functions the TTS normalization path uses are vendored; the upstream
# wyoming-separator helper patterns are intentionally omitted.
#
# Local deviations from upstream:
# - Sentence-final numbers: upstream reads every "N." followed by a word or the
#   end of text as an ordinal and drops the full stop, merging sentences
#   ("im Jahr 2024. Danach" -> "...vierundzwanzigster Danach"). Here, a bare
#   "N." (no article/preposition prefix) is an ordinal only when a lowercase
#   word follows; otherwise (end of text, a capitalised word, a digit) it is
#   read as a cardinal (or as a year after "Jahr") and the full stop is kept.
#   A prefixed ordinal ("am 3.", "der 1.") stays an ordinal but keeps its full
#   stop at the end of the text or before a capitalised sentence starter
#   (pronoun, article, conjunction or adverb, see SENTENCE_STARTERS).
# - year_to_words reads 1100-1999 as spoken years ("neunzehnhundertneunzig").
# - Numeric substitutions preserve tokens above 64 digits without conversion.
# - Cardinal labels emit their punctuation directly; no text markers are reserved.
# - Date expansions require valid Gregorian calendar components.
# - Clock expansions require a time from 00:00 to 23:59.
from __future__ import annotations

import re
from datetime import date
from decimal import Decimal

ABBREVIATIONS = (
    (r"\bLfd\.\s*Nr\.", "laufende Nummer"),
    (r"\bz\.\s*$", "zum Beispiel"),
    (r"^\s*b\.\s+", ""),
    (r"\bz\s*\.?\s*b\.?", "zum Beispiel"),
    (r"\bzB\b", "zum Beispiel"),
    (r"\betc\.", "ezetera"),
    (r"\busw\.", "und so weiter"),
    (r"\bca\.", "zirka"),
    (r"\bd\.\s*h\.", "das heißt"),
    (r"\bu\.\s*a\.", "unter anderem"),
    (r"\bggf\.", "gegebenenfalls"),
    (r"\bzzgl\.", "zuzüglich"),
    (r"\bDr\.", "Doktor"),
    (r"\bProf\.", "Professor"),
    (r"\bAbk\.", "Abkürzung"),
    (r"\bAbb\.", "Abbildung"),
    (r"\bgeb\.", "geboren"),
    (r"\bbspw\.", "beispielsweise"),
    (r"\bNr\.", "Nummer"),
    (r"\bggü\.", "gegenüber"),
    (r"\bKap\.", "Kapitel"),
    (r"\bS\.", "Seite"),
    (r"\bAbs\.", "Absatz"),
    (r"\bTsd\.", "Tausend"),
    (r"\bMio\.", "Millionen"),
    (r"\bMrd\.", "Milliarden"),
    (r"\bGmbH\b", "G-M-B-H"),
    (r"\bAG\b", "A-G"),
)

# Tuple: (pattern, singular, plural, gender)
# Longer/more specific patterns first (mAh before mA, kWh before W, ...).
NUMBERED_UNITS = (
    (r"kWh", "Kilowattstunde", "Kilowattstunden", "f"),
    (r"Wh", "Wattstunde", "Wattstunden", "f"),
    (r"GHz", "Gigahertz", "Gigahertz", None),
    (r"MHz", "Megahertz", "Megahertz", None),
    (r"kHz", "Kilohertz", "Kilohertz", None),
    (r"Hz", "Hertz", "Hertz", None),
    (r"Std\.", "Stunde", "Stunden", "f"),
    (r"Min\.", "Minute", "Minuten", "f"),
    (r"Sek\.", "Sekunde", "Sekunden", "f"),
    (r"Stck\.", "Stück", "Stück", "n"),
    (r"mAh", "Milliamperestunde", "Milliamperestunden", "f"),
    (r"mA", "Milliampere", "Milliampere", None),
    (r"kg", "Kilogramm", "Kilogramm", None),
    (r"g", "Gramm", "Gramm", None),
    (r"km", "Kilometer", "Kilometer", None),
    (r"cm", "Zentimeter", "Zentimeter", None),
    (r"mm", "Millimeter", "Millimeter", None),
    (r"m3", "Kubikmeter", "Kubikmeter", None),
    (r"m", "Meter", "Meter", None),
    (r"ltr\.", "Liter", "Liter", None),
    (r"W", "Watt", "Watt", None),
    (r"V", "Volt", "Volt", None),
    (r"Tsd\.", "Tausend", "Tausend", None),
    (r"Mio\.", "Millionen", "Millionen", None),
    (r"Mrd\.", "Milliarden", "Milliarden", None),
)

NUM_TO_WORD = {
    0: "null",
    1: "eins",
    2: "zwei",
    3: "drei",
    4: "vier",
    5: "fünf",
    6: "sechs",
    7: "sieben",
    8: "acht",
    9: "neun",
    10: "zehn",
    11: "elf",
    12: "zwölf",
    13: "dreizehn",
    14: "vierzehn",
    15: "fünfzehn",
    16: "sechzehn",
    17: "siebzehn",
    18: "achtzehn",
    19: "neunzehn",
    20: "zwanzig",
    30: "dreißig",
    40: "vierzig",
    50: "fünfzig",
    60: "sechzig",
    70: "siebzig",
    80: "achtzig",
    90: "neunzig",
}

ORDINAL_TO_WORD = {
    1: "erster",
    2: "zweiter",
    3: "dritter",
    4: "vierter",
    5: "fünfter",
    6: "sechster",
    7: "siebter",
    8: "achter",
    9: "neunter",
    10: "zehnter",
    11: "elfter",
    12: "zwölfter",
    13: "dreizehnter",
    14: "vierzehnter",
    15: "fünfzehnter",
    16: "sechzehnter",
    17: "siebzehnter",
    18: "achtzehnter",
    19: "neunzehnter",
    20: "zwanzigster",
    30: "dreißigster",
}

# Capitalised words that start a new sentence rather than follow an ordinal as
# a noun ("am 3. Danach ..." vs. "der 1. Platz"). Local addition, see header.
SENTENCE_STARTERS = frozenset(
    (
        "aber als auch auf aus bei da dabei dadurch dafür damals damit danach dann "
        "darauf das dass davor dazu dem den denn der deshalb die dies diese dieser "
        "dieses doch dort du ein eine einen einer er es heute hier ich ihr im in jetzt "
        "man mit nach nun oder seit sie so später trotzdem um und vor während wann was "
        "weil wenn wer wie wir wo zum zur"
    ).split()
)

MONTHS = "januar|februar|märz|april|mai|juni|juli|august|september|oktober|november|dezember"
MONTH_NUMBERS = {month: number for number, month in enumerate(MONTHS.split("|"), start=1)}
MAX_NUMERIC_TOKEN_DIGITS = 64
# Day, month and optional year of a named-month date such as "3. Mai 2024".
# The date ends at any non-word character after the month or year.
# Only 20xx years are spoken; another four-digit year stays literal but still
# decides whether the day exists.
NAMED_DATE = (
    rf"(?P<date_day>\d{{1,2}})\.\s*(?P<month>{MONTHS})"
    r"(?:\s+(?P<year>20\d{2})|(?=\s+(?P<literal_year>\d{4})(?!\w)))?(?!\w)"
)
NAMED_DATE_RE = re.compile(rf"(?i){NAMED_DATE}")


def _is_calendar_date(day: int, month: int, year: str | None) -> bool:
    calendar_year = int(year) if year is not None else 0
    try:
        # A yearless 29 February can refer to a leap year; year 0000 (not
        # representable by datetime) is checked like a missing year.
        date(calendar_year or 2000, month, day)
    except ValueError:
        return False
    return True


def _is_named_calendar_date(match: re.Match[str]) -> bool:
    return _is_calendar_date(
        int(match.group("date_day")),
        MONTH_NUMBERS.get(match.group("month").casefold(), 0),
        match.group("year") or match.group("literal_year"),
    )


# Capture the entire numeric spelling so invalid grouping cannot expose a suffix.
QUANTITY_TOKEN = r"(?<!\w)(?<![\d.,][.,])(-?\d+(?:[.,]+\d+)*)"
GROUPED_QUANTITY_RE = re.compile(r"-?[1-9]\d{0,2}(?:\.\d{3})+(?:,\d+)?")
DECIMAL_QUANTITY_RE = re.compile(r"-?\d+(?:[,.]\d+)?")
EURO_AMOUNT_RE = re.compile(QUANTITY_TOKEN + r"\s*EUR\b", re.IGNORECASE)


def _normalized_quantity(raw: str) -> str | None:
    if not _supports_numeric_token(raw):
        return None
    if GROUPED_QUANTITY_RE.fullmatch(raw):
        return raw.replace(".", "")
    return raw if DECIMAL_QUANTITY_RE.fullmatch(raw) else None


def _supports_numeric_token(raw: str) -> bool:
    # Count both sides of a decimal together, without converting any digits.
    return len(raw.lstrip("-").replace(",", "").replace(".", "")) <= MAX_NUMERIC_TOKEN_DIGITS


def number_to_words(number: int) -> str:
    """Return a German spoken form for common cardinal numbers."""
    if number < 0:
        return f"minus {number_to_words(abs(number))}"

    if number in NUM_TO_WORD:
        return NUM_TO_WORD[number]

    if number < 100:
        ones = number % 10
        tens = number - ones
        one_word = "ein" if ones == 1 else NUM_TO_WORD[ones]
        return f"{one_word}und{NUM_TO_WORD[tens]}"

    if number < 1000:
        hundreds = number // 100
        rest = number % 100
        prefix = "einhundert" if hundreds == 1 else f"{number_to_words(hundreds)}hundert"
        return prefix if rest == 0 else f"{prefix}{number_to_words(rest)}"

    if number < 1_000_000:
        thousands = number // 1000
        rest = number % 1000
        prefix = "eintausend" if thousands == 1 else f"{number_to_words(thousands)}tausend"
        return prefix if rest == 0 else f"{prefix}{number_to_words(rest)}"

    return str(number)


def ordinal_to_words(number: int) -> str:
    if number in ORDINAL_TO_WORD:
        return ORDINAL_TO_WORD[number]
    if number < 1_000_000:
        if number < 100:
            return f"{number_to_words(number)}ster"

        cardinal = number_to_words(number)
        suffix_number = number % 100
        if suffix_number == 0:
            return f"{cardinal}ster"

        suffix_cardinal = number_to_words(suffix_number)
        if not cardinal.endswith(suffix_cardinal):
            return f"{cardinal}ster"

        return f"{cardinal[: -len(suffix_cardinal)]}{ordinal_to_words(suffix_number)}"
    return f"{number}."


def ordinal_dative(number: int) -> str:
    word = ordinal_to_words(number)
    if word.endswith("er"):
        return f"{word[:-2]}en"
    if word.endswith("ster"):
        return f"{word[:-2]}en"
    return word


def ordinal_weak(number: int) -> str:
    word = ordinal_to_words(number)
    return word[:-1] if word.endswith("er") else word


def year_to_words(year: str | int) -> str:
    year_int = int(year)
    if 1100 <= year_int < 2000:
        century, rest = divmod(year_int, 100)
        prefix = f"{number_to_words(century)}hundert"
        return prefix if rest == 0 else f"{prefix}{number_to_words(rest)}"
    if 2000 <= year_int < 2100:
        rest = year_int - 2000
        return "zweitausend" if rest == 0 else f"zweitausend{number_to_words(rest)}"
    return number_to_words(year_int) if year_int < 1_000_000 else str(year)


def decimal_to_words(raw_value: str) -> str:
    normalized = raw_value.replace(".", ",")
    if "," not in normalized:
        return number_to_words(int(normalized))

    integer_part, decimal_part = normalized.split(",", 1)
    integer_word = number_to_words(int(integer_part or "0"))
    decimal_words = " ".join(
        number_to_words(int(digit)) for digit in decimal_part if digit.isdigit()
    )
    return f"{integer_word} komma {decimal_words}".strip()


def _number_word_with_gender(raw: str, gender: str | None) -> str:
    prefix = "minus " if raw.startswith("-") else ""
    unsigned = raw.lstrip("-")

    if re.fullmatch(r"0*1", unsigned):
        if gender == "f":
            return f"{prefix}eine"
        return f"{prefix}ein"

    return f"{prefix}{decimal_to_words(unsigned)}"


def _unit_form(raw: str, singular: str, plural: str) -> str:
    if Decimal(raw.replace(",", ".")).copy_abs() == 1:
        return singular
    return plural


def _numbered_unit_suffix(unit_pattern: str) -> str:
    if unit_pattern.endswith(r"\."):
        return unit_pattern + r"(?=\s|[,!?]|$)"
    return unit_pattern + r"(?=\b|[.,!?])"


PHYSICAL_QUANTITY_RE = re.compile(
    QUANTITY_TOKEN
    + r"\s*(?:°|"
    + "|".join(_numbered_unit_suffix(unit[0]) for unit in NUMBERED_UNITS)
    + ")",
    re.IGNORECASE,
)
DEGREE_QUANTITY_RE = re.compile(QUANTITY_TOKEN + r"\s*°(?:\s*([Cc])\b)?")


def _normalize_numbered_units(text: str) -> str:
    for unit_pattern, singular, plural, gender in NUMBERED_UNITS:

        def replace_unit(
            match: re.Match[str],
            _singular: str = singular,
            _plural: str = plural,
            _gender: str | None = gender,
        ) -> str:
            raw = _normalized_quantity(match.group(1))
            if raw is None:
                return match.group(0)
            number_word = _number_word_with_gender(raw, _gender)
            unit_word = _unit_form(raw, _singular, _plural)
            return f"{number_word} {unit_word}"

        text = re.sub(
            QUANTITY_TOKEN + r"\s*" + _numbered_unit_suffix(unit_pattern),
            replace_unit,
            text,
            flags=re.IGNORECASE,
        )

    return text


def _normalize_euro_amounts(text: str) -> str:
    def replace_euro_amount(match: re.Match[str]) -> str:
        raw = _normalized_quantity(match.group(1))
        if raw is None:
            return match.group(0)
        euros_raw, _, cents_raw = raw.replace(".", ",").partition(",")
        if not 1 <= len(cents_raw) <= 2:
            return f"{_number_word_with_gender(raw, None)} Euro"

        euros_word = _number_word_with_gender(euros_raw, None)
        cents = int(cents_raw.ljust(2, "0"))
        if cents == 0:
            return f"{euros_word} Euro"
        return f"{euros_word} Euro {number_to_words(cents)}"

    return EURO_AMOUNT_RE.sub(replace_euro_amount, text)


def _normalize_cardinal_label_numbers(text: str) -> str:
    """Read label numbers as cardinals, not ordinals, before sentence boundaries."""
    label_pattern = r"(Nummer|laufende\s+Nummer|Gleis|Kapitel|Absatz|Seite)"

    def replace_label_number(match: re.Match[str]) -> str:
        label = match.group(1)
        number = int(match.group(2))
        # The bounded label number is now a word, so later numeric patterns
        # cannot consume its full stop. No sentinel or restoration is needed.
        return f"{label} {number_to_words(number)}."

    return re.sub(
        rf"(?i)\b{label_pattern}\s+(\d{{1,6}})\.(?=\s|$)",
        replace_label_number,
        text,
    )


def normalize_german_text(text: str) -> str:
    """Make German assistant text easier for Kokoro/espeak to pronounce."""
    if not text:
        return text

    text = _normalize_euro_amounts(text)
    text = _normalize_numbered_units(text)

    for pattern, replacement in ABBREVIATIONS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    text = _normalize_cardinal_label_numbers(text)

    def replace_degree(match: re.Match[str]) -> str:
        raw = _normalized_quantity(match.group(1))
        if raw is None:
            return match.group(0)
        prefix = "minus " if raw.startswith("-") else ""
        value = decimal_to_words(raw.lstrip("-"))
        unit = "Grad Celsius" if match.group(2) else "Grad"
        return f"{prefix}{value} {unit}"

    def replace_numeric_date(match: re.Match[str]) -> str:
        prefix = match.group(1) or ""
        day = int(match.group(2))
        month = int(match.group(3))
        year = match.group(4)
        # A malformed quantity can have the same spelling as a numeric date.
        numeric_start = match.start(2)
        if EURO_AMOUNT_RE.match(match.string, numeric_start) or PHYSICAL_QUANTITY_RE.match(
            match.string, numeric_start
        ):
            return match.group(0)
        if not _is_calendar_date(day, month, year or match.group(5)):
            return match.group(0)

        result = f"{prefix}{ordinal_dative(day)} {ordinal_dative(month)}"
        if year:
            result += f" {year_to_words(year)}"
        return result

    def replace_text_date(match: re.Match[str]) -> str:
        prefix_full = match.group("date_prefix") or ""
        prefix = prefix_full.strip().lower()
        day = int(match.group("date_day"))
        month = match.group("month")
        year = match.group("year")
        if not _is_named_calendar_date(match):
            return match.group(0)

        day_word = ordinal_to_words(day)
        if prefix in {"am", "den", "zum", "dem", "vom", "bis zum", "jeden"}:
            day_word = ordinal_dative(day)
        elif prefix in {"der", "die", "das"} and day_word.endswith("er"):
            day_word = day_word[:-1]

        result = f"{prefix_full}{day_word} {month}"
        if year:
            result += f" {year_to_words(year)}"
        return result

    def replace_time(match: re.Match[str]) -> str:
        hour = int(match.group(1))
        minute = int(match.group(2))
        if hour >= 24 or minute >= 60:
            return match.group(0)
        hour_word = "ein" if hour == 1 else number_to_words(hour)
        if minute == 0:
            return f"{hour_word} Uhr"
        return f"{hour_word} Uhr {number_to_words(minute)}"

    def replace_text_date_or_ordinal(match: re.Match[str]) -> str:
        if match.group("month") is not None:
            # Consume the whole candidate even when invalid, so its day cannot
            # be rewritten as a standalone ordinal.
            return replace_text_date(match)

        prefix_full = match.group("ordinal_prefix") or ""
        prefix = prefix_full.strip().lower()
        raw_number = match.group("ordinal_number")
        number = int(raw_number)

        following = match.string[match.end() :].lstrip()
        next_word = re.match(r"\w+", following)
        before_capital = next_word is not None and next_word.group(0)[0].isupper()
        before_lowercase = next_word is not None and next_word.group(0)[0].islower()
        following_date = NAMED_DATE_RE.match(following)
        if following_date is not None and _is_named_calendar_date(following_date):
            # The same pass speaks that date as a lowercase ordinal word.
            before_capital, before_lowercase = False, True

        if not prefix_full:
            if before_lowercase:
                return ordinal_to_words(number)
            preceding = match.string[: match.start()]
            if (
                len(raw_number) == 4
                and 1000 <= number < 2100
                and re.search(r"(?i)\bJahre?s?\s+$", preceding)
            ):
                return f"{year_to_words(number)}."
            return f"{number_to_words(number)}."

        keep_stop = not following or (
            next_word is not None
            and before_capital
            and next_word.group(0).lower() in SENTENCE_STARTERS
        )

        if prefix.startswith("nach ") or prefix in {
            "am",
            "im",
            "zum",
            "vom",
            "dem",
            "den",
            "einen",
            "einem",
            "jeden",
            "jedem",
        }:
            word = ordinal_dative(number)
        elif prefix in {"der", "die", "das", "eine", "einer", "jede", "jeder"}:
            word = ordinal_weak(number)
        else:
            word = ordinal_to_words(number)

        return f"{prefix_full}{word}{'.' if keep_stop else ''}"

    text = DEGREE_QUANTITY_RE.sub(replace_degree, text)
    text = re.sub(
        r"(?i)(?<!\w)(?<!\d[.:])(am\s+|den\s+|zum\s+|vom\s+|bis\s+zum\s+|jeden\s+)?"
        # Only 20xx years are spoken; another four-digit year stays literal
        # but still decides whether the day exists.
        r"(\d{1,2})\.(\d{1,2})\.(?:\s*(20\d{2})|(?=\s*(\d{4})(?!\w)))?"
        r"(?!\.\d)(?=\s|[.,!?]|$)",
        replace_numeric_date,
        text,
    )
    text = re.sub(
        r"(?<!\w)(?<!\d:)(\d{1,2}):(\d{2})(?:\s*[Uu]hr)?(?!\.\d)(?=\s|[.,!?]|$)",
        replace_time,
        text,
    )
    text = re.sub(
        r"(?i)(?<!\w)(?<!\d[.:])(?:"
        r"(?P<date_prefix>am\s+|der\s+|die\s+|das\s+|den\s+|zum\s+|vom\s+|dem\s+|bis\s+zum\s+|jeden\s+)?"
        + NAMED_DATE
        + r"|(?P<ordinal_prefix>am\s+|im\s+|zum\s+|vom\s+|dem\s+|den\s+|der\s+|die\s+|das\s+|"
        r"eine\s+|einer\s+|einen\s+|einem\s+|jeden\s+|jedem\s+|jede\s+|jeder\s+|"
        r"nach\s+der\s+(?:gefühlt\s+)?)?"
        # An ordinal prefix such as "im" can begin before a named date.
        rf"(?!\d{{1,2}}\.\s*(?:{MONTHS})(?!\w))"
        r"(?P<ordinal_number>\d{1,6})\.(?=\s+\w|$))",
        replace_text_date_or_ordinal,
        text,
    )
    text = re.sub(r"(?i)\bHbf\b\.?", "Hauptbahnhof", text)
    text = re.sub(r"(?i)\bEUR\b", "Euro", text)

    return text
