"""Compare the chapter windows with the research turbo windows (D-29 acceptance).

Offline validation tooling for ``specs/2026-10-17_chapter_windows.md``, not part of the
service. For every episode of Delve's gate G6 run (the scratch manifest written by
``apps/delve-backend/scripts/g6-chapters.sh``: the episodes with a sponsor-block edge in
the holdout and never-seen sets and of listening kit 2) it takes the edges the research
transcribed (``round2-chapter-textcheck/out/edges.json``), sends their marks in whole
milliseconds (sorted, each once: what Delve sends) through the shipped
``run_chapter_windows`` on the same source file, and compares every window with the
research's ``cache/w120p/<id>.json`` (written by ``asrwin120.py``): ``ok``, clip start,
shift and the words (text, start, end), bit for bit. It also checks that the profile's
pinned model is the research file (same SHA-256), that no temporary directory is left,
that no database query ran, and that the research directory is not written.

Usage (from the voxhelm checkout):

    VOXHELM_WINDOW_MODEL=/path/to/ggml-large-v3-turbo.bin \\
        uv run python evals/chapter_windows_compare.py \\
        --manifest /path/to/g6run/manifest.json --out /tmp/chapter-windows-compare.json
"""

from __future__ import annotations

import argparse
import hashlib
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

from transcriptions.chapter_windows import (  # noqa: E402
    media_duration_ms,
    resolve_ffprobe_binary,
    run_chapter_windows,
)
from transcriptions.windows import WINDOW_MODEL_SHA256, resolve_window_runtime  # noqa: E402

WORKSPACE = Path("/Users/jochen/workspaces/ws-podcast-ad-reassess-20261012")
RESEARCH = WORKSPACE / "work" / "night-2026-10-14" / "round2-chapter-textcheck"


def temp_dirs() -> set[str]:
    return {p.name for p in Path(tempfile.gettempdir()).glob("voxhelm-chapter-windows-*")}


def newest_mtime(root: Path) -> float:
    return max((p.stat().st_mtime for p in root.rglob("*")), default=0.0)


def compare_window(reference: dict[str, Any], window: dict[str, Any]) -> dict[str, Any]:
    expected = [{"w": w[2], "s": float(w[0]), "e": float(w[1])} for w in reference["words"]]
    got = window["words"]
    first = next(
        (i for i, (a, b) in enumerate(zip(got, expected, strict=False)) if a != b),
        None if len(got) == len(expected) else min(len(got), len(expected)),
    )
    return {
        "id": reference["id"],
        "mark_ms": window["mark_ms"],
        "ok_same": window["ok"] == reference["ok"],
        "clip_start_same": window["clip_start"] == round(reference["t0"], 3),
        "shift_same": window["shift"] == reference["shift"],
        "words_identical": got == expected,
        "words": len(got),
        "reference_words": len(expected),
        "first_difference": first,
        "seconds": window["seconds"],
        "reference_seconds": reference["wall"],
        "exit_codes": window["exit_codes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--research", type=Path, default=RESEARCH)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--keys", nargs="*", default=None)
    args = parser.parse_args()

    executable, model_path = resolve_window_runtime()
    ffprobe = resolve_ffprobe_binary()
    digest = hashlib.sha256()
    with model_path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    model_sha256 = digest.hexdigest()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    edges = json.loads((args.research / "out" / "edges.json").read_text(encoding="utf-8"))
    keys = sorted(manifest) if not args.keys else sorted(args.keys)
    before_dirs = temp_dirs()
    research_mtime = newest_mtime(args.research)

    episodes: list[dict[str, Any]] = []
    started_all = time.monotonic()
    with CaptureQueriesContext(connection) as queries:
        for key in keys:
            audio = Path(manifest[key]["path"])
            references: dict[int, list[dict[str, Any]]] = {}
            for edge in edges:
                if edge["key"] != key:
                    continue
                cache = args.research / "cache" / "w120p" / (edge["id"].replace("#", "_") + ".json")
                reference = json.loads(cache.read_text(encoding="utf-8"))
                mark_ms = round(edge["t"] * 1000)
                assert abs(mark_ms - edge["t"] * 1000) < 1e-6, edge["id"]
                references.setdefault(mark_ms, []).append(reference)
            marks = sorted(references)
            duration_ms = media_duration_ms(audio, ffprobe=ffprobe)
            started = time.monotonic()
            result = run_chapter_windows(
                audio,
                marks,
                executable=executable,
                model_path=model_path,
                duration_ms=duration_ms,
            )
            elapsed = time.monotonic() - started
            rows = [
                compare_window(reference, window)
                for window in result.windows
                for reference in references[window["mark_ms"]]
            ]
            episode = {
                "key": key,
                "marks": len(marks),
                "edges": len(rows),
                "duration_ms": duration_ms,
                "seconds": round(elapsed, 3),
                "identical": all(
                    r["ok_same"]
                    and r["clip_start_same"]
                    and r["shift_same"]
                    and r["words_identical"]
                    for r in rows
                ),
                "words": sum(r["words"] for r in rows),
                "windows": rows,
            }
            episodes.append(episode)
            print(
                f"{key}: {episode['edges']} edges, {episode['words']} words, "
                f"identical={episode['identical']}, {elapsed:.1f} s",
                flush=True,
            )
    summary = {
        "model_path": str(model_path),
        "model_sha256": model_sha256,
        "model_is_pinned": model_sha256 == WINDOW_MODEL_SHA256,
        "identity": result.identity if episodes else None,
        "episodes": len(episodes),
        "episodes_identical": sum(e["identical"] for e in episodes),
        "edges": sum(e["edges"] for e in episodes),
        "edges_identical": sum(
            r["ok_same"] and r["clip_start_same"] and r["shift_same"] and r["words_identical"]
            for e in episodes
            for r in e["windows"]
        ),
        "words": sum(e["words"] for e in episodes),
        "reference_words": sum(r["reference_words"] for e in episodes for r in e["windows"]),
        "windows_not_ok": sum(not r["ok_same"] for e in episodes for r in e["windows"]),
        "retried_clips": sum(len(r["exit_codes"]) > 1 for e in episodes for r in e["windows"]),
        "seconds_total": round(time.monotonic() - started_all, 1),
        "seconds_per_window_median": sorted(r["seconds"] for e in episodes for r in e["windows"])[
            sum(e["edges"] for e in episodes) // 2
        ]
        if episodes
        else None,
        "database_queries": len(queries.captured_queries),
        "temp_dirs_left": sorted(temp_dirs() - before_dirs),
        "research_written": newest_mtime(args.research) > research_mtime,
    }
    args.out.write_text(
        json.dumps({"summary": summary, "episodes": episodes}, indent=1), encoding="utf-8"
    )
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
