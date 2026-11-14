"""Strict object JSON admission shared by requests and backend responses."""

import json
import math
from typing import Any


def check_json_depth(raw: bytes) -> None:
    # Count only structural delimiters outside strings, before decoder allocation.
    depth = 0
    quoted = escaped = False
    for byte in raw:
        if quoted:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
        elif byte == 34:
            quoted = True
        elif byte in (91, 123):
            depth += 1
            if depth > 128:
                raise ValueError("JSON nesting exceeds 128 levels")
        elif byte in (93, 125):
            depth -= 1


def unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON member")
        result[key] = value
    return result


def reject_json_constant(value: str) -> object:
    raise ValueError("Nonfinite JSON number")


def finite_json_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Nonfinite JSON number")
    return number


def decode_json_object(raw: bytes) -> dict[str, Any]:
    check_json_depth(raw)
    payload = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=unique_json_object,
        parse_constant=reject_json_constant,
        parse_float=finite_json_float,
    )
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    return payload
