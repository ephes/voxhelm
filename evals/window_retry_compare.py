"""Compare window transcription with the research turbo retries (slice 2 acceptance).

Offline validation tooling for ``specs/2026-10-16_window_transcription.md``, not part of
the service. It picks 40 development windows of the German drift research
(``work/night-2026-10-14/round2-german-drift`` of the ad research workspace): 25 flagged
German windows (``dev-cand``), 5 unflagged German windows (``dev-clean``) and 10 windows
of English shows (``dev-enctrl``), evenly spaced through each set's auto-mode retry file
(``retry-dev-<set>-auto.jsonl``, written by ``retry.py`` through ``whisper-server``). Each
window goes through the shipped ``transcribe_window`` on the same source file, and the
detected language, its probability and the words are compared with the research record.
It also checks that no temporary clip directory is left and that no database query ran.

Usage (from the voxhelm checkout):

    VOXHELM_WINDOW_MODEL=/path/to/ggml-large-v3-turbo.bin \
        uv run python evals/window_retry_compare.py --out /tmp/window-compare.json
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
os.environ.setdefault("DJANGO_DEBUG", "true")
django.setup()

from django.db import connection  # noqa: E402
from django.test.utils import CaptureQueriesContext  # noqa: E402

from transcriptions.windows import resolve_window_runtime, transcribe_window  # noqa: E402

WORKSPACE = Path("/Users/jochen/workspaces/ws-podcast-ad-reassess-20261012")
RESEARCH = WORKSPACE / "work" / "night-2026-10-14" / "round2-german-drift"
SETS = (("cand", 25), ("clean", 5), ("enctrl", 10))
LANGUAGE_NAMES = {"german": "de", "english": "en"}


def audio_paths(workspace: Path) -> dict[str, Path]:
    """Source file per corpus key, as the research ``corpus.meta('dev')``."""
    paths: dict[str, Path] = {}
    manifest = json.loads((workspace / "audio" / "manifest.json").read_text(encoding="utf-8"))
    for key, entry in manifest.items():
        paths[key] = workspace / "audio" / entry["slug"] / f"{key}.mp3"
    extern = json.loads((workspace / "audio" / "extern.json").read_text(encoding="utf-8"))
    for key, entry in extern.items():
        paths[key] = Path(entry["path"])
    return paths


def pick_windows(research: Path) -> list[tuple[str, dict[str, Any]]]:
    picked: list[tuple[str, dict[str, Any]]] = []
    for name, count in SETS:
        lines = (research / f"retry-dev-{name}-auto.jsonl").read_text(encoding="utf-8")
        records = [json.loads(line) for line in lines.splitlines() if line.strip()]
        step = len(records) / count
        picked.extend((name, records[int(index * step)]) for index in range(count))
    return picked


def window_dirs() -> set[str]:
    return {entry.name for entry in Path(tempfile.gettempdir()).glob("voxhelm-window-*")}


def compare(reference: dict[str, Any], words: list[dict[str, Any]]) -> dict[str, Any]:
    expected = reference["words"]
    same_text = [word["w"] for word in words] == [word["w"] for word in expected]
    timing_diffs = [
        max(abs(a["s"] - b["s"]), abs(a["e"] - b["e"]))
        for a, b in zip(words, expected, strict=False)
        if a["w"] == b["w"]
    ]
    first_difference = next(
        (index for index, (a, b) in enumerate(zip(words, expected, strict=False)) if a != b),
        None if len(words) == len(expected) else min(len(words), len(expected)),
    )
    return {
        "identical": words == expected,
        "same_text": same_text,
        "words": len(words),
        "reference_words": len(expected),
        "max_time_difference": max(timing_diffs, default=0.0),
        "first_difference": first_difference,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--research", type=Path, default=RESEARCH)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    _, model_path = resolve_window_runtime()  # refuses any file but the pinned one
    paths = audio_paths(args.workspace)
    rows: list[dict[str, Any]] = []
    before = window_dirs()
    for set_name, reference in pick_windows(args.research):
        started = time.monotonic()
        with CaptureQueriesContext(connection) as queries:
            result = transcribe_window(paths[reference["key"]], reference["t"])
        wall = time.monotonic() - started
        residue = sorted(window_dirs() - before)
        row = {
            "set": set_name,
            "key": reference["key"],
            "t": reference["t"],
            "language": result.language,
            "reference_language": LANGUAGE_NAMES.get(reference["lang"], reference["lang"]),
            "p": result.language_probability,
            "reference_p": reference["p"],
            "p_matches_6dp": round(reference["p"], 6) == result.language_probability,
            "queries": len(queries.captured_queries),
            "residue": residue,
            "wall_s": round(wall, 2),
            **compare(reference, result.words),
        }
        row["language_matches"] = row["language"] == row["reference_language"]
        rows.append(row)
        print(
            f"{set_name:6} {reference['key']:18} {reference['t']:>6} "
            f"{row['language']}/{row['reference_language']} p={row['p']:.6f} "
            f"words={row['words']}/{row['reference_words']} "
            f"identical={row['identical']} {wall:.2f}s",
            flush=True,
        )
    summary = {
        "model": str(model_path),
        "identity": result.identity,
        "windows": len(rows),
        "language_matches": sum(row["language_matches"] for row in rows),
        "p_matches_6dp": sum(row["p_matches_6dp"] for row in rows),
        "words_identical": sum(row["identical"] for row in rows),
        "words_same_text": sum(row["same_text"] for row in rows),
        "words_total": sum(row["words"] for row in rows),
        "reference_words_total": sum(row["reference_words"] for row in rows),
        "queries": sum(row["queries"] for row in rows),
        "residue": sorted({entry for row in rows for entry in row["residue"]}),
    }
    args.out.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
