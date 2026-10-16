"""Gate G2: the drift check of ``ad-analysis-v1`` against the German drift research.

Offline validation tooling for ``specs/2026-10-16_drift_check.md``, not part of the
service. ``run`` sends each of the 22 German development episodes of the ad research
through the real profile code path, as the G1 run did: ``POST /v1/uploads`` with the
episode bytes, ``POST /v1/jobs`` with ``profile=ad-analysis-v1`` and
``language_hint=de``, executed by the ``django_tasks`` immediate backend in this process
(Parakeet CLI, then the drift check with ``whisper-cli`` and the pinned turbo model),
``GET`` of ``words.json``, then the producer ``DELETE`` (and a check that the artifact
store holds nothing of the job afterwards). The database is a throw-away test database.
``analyse`` compares the saved words documents with the research
(``work/night-2026-10-14/round2-german-drift``, frozen 2026-10-15):

- per-window English / German function-word counts and flags against ``win-dev.json``
  (``dist.py``) with the frozen rule ``e >= 4 and e / max(1, e + g) >= 0.1``;
- drift removal against the whole-file turbo reference (``te``) as ``sweep.py`` (excess
  English captured by the flagged windows) and ``evalretry.py`` (English function words
  in the flagged windows, Parakeet -> retry, turbo reference), plus the excess left after
  the splice over all windows;
- retry time per German audio hour;
- the spliced words against ``splice.py``'s word list on ``retry-dev-cand-auto.jsonl``
  with the adopted acceptance rule (language German, p >= 0.9) applied, and the detector
  lines (``sentence_segs``) against ``tx-dev/pk30s-fix``.

Usage (from the voxhelm checkout):

    uv run python evals/drift_check_g2.py run --out /tmp/g2
    uv run python evals/drift_check_g2.py analyse --out /tmp/g2
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

WORKSPACE = Path("/Users/jochen/workspaces/ws-podcast-ad-reassess-20261012")
RESEARCH = WORKSPACE / "work" / "night-2026-10-14" / "round2-german-drift"
PK_DEV = WORKSPACE / "work" / "night-2026-10-14" / "parakeet-chain" / "pk-dev"
TURBO_MODEL = WORKSPACE / "models" / "ggml-large-v3-turbo.bin"
PARAKEET_MODEL_DIR = (
    Path.home()
    / ".cache/huggingface/hub/models--mlx-community--parakeet-tdt-0.6b-v3"
    / "snapshots/ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15"
)
TOKEN = "g2-producer-token"
RULE_E, RULE_S = 4, 0.1  # the frozen rule, as splice.py / evalretry.py take it ("4,0.1")


def german_episodes(workspace: Path) -> dict[str, dict[str, Any]]:
    """German dev episodes as the research ``corpus.meta('dev')``: path, duration, sha256."""
    episodes: dict[str, dict[str, Any]] = {}
    manifest = json.loads((workspace / "audio" / "manifest.json").read_text(encoding="utf-8"))
    for key, entry in manifest.items():
        path = workspace / "audio" / entry["slug"] / f"{key}.mp3"
        episodes[key] = {"path": path, "dur": entry["dur"], "lang": entry["lang"]}
    extern = json.loads((workspace / "audio" / "extern.json").read_text(encoding="utf-8"))
    for key, entry in extern.items():
        episodes[key] = {
            "path": Path(entry["path"]),
            "dur": entry["dur"],
            "lang": entry["lang"],
            "sha256": entry.get("sha256"),
        }
    return {key: value for key, value in sorted(episodes.items()) if value["lang"] == "de"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


# --- run ------------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    artifact_root = Path(tempfile.mkdtemp(prefix="voxhelm-g2-artifacts-"))
    os.environ.update(
        {
            "DJANGO_SETTINGS_MODULE": "config.settings",
            "DJANGO_DEBUG": "true",
            "VOXHELM_TASKS_BACKEND": "django_tasks.backends.immediate.ImmediateBackend",
            "VOXHELM_BEARER_TOKENS": f"delve={TOKEN}",
            "VOXHELM_ARTIFACT_BACKEND": "filesystem",
            "VOXHELM_ARTIFACT_ROOT": str(artifact_root),
            "VOXHELM_PARAKEET_BIN": args.parakeet_bin,
            "VOXHELM_PARAKEET_MODEL_DIR": str(args.parakeet_model_dir),
            "VOXHELM_WINDOW_MODEL": str(args.turbo_model),
            "VOXHELM_LANE_SCHEDULER_ENABLED": "false",
        }
    )
    import django

    django.setup()
    from django.db import connection
    from django.test import Client
    from django.test.utils import setup_test_environment, teardown_test_environment

    from transcriptions.windows import resolve_window_runtime

    resolve_window_runtime()  # refuses any turbo file but the pinned one
    setup_test_environment()
    old_name = connection.creation.create_test_db(verbosity=0)
    client = Client()
    headers: dict[str, Any] = {"HTTP_AUTHORIZATION": f"Bearer {TOKEN}"}
    episodes = german_episodes(args.workspace)
    if args.keys:
        episodes = {key: episodes[key] for key in args.keys}
    log: list[dict[str, Any]] = []
    try:
        for key, episode in episodes.items():
            target = args.out / f"{key}.json"
            if target.exists():
                print(f"{key}: already done", flush=True)
                continue
            if episode.get("sha256") and sha256_file(episode["path"]) != episode["sha256"]:
                raise SystemExit(f"{key}: source differs from the research file")
            load_before = os.getloadavg()
            started = time.monotonic()
            with episode["path"].open("rb") as handle:
                upload = client.post("/v1/uploads", data={"file": handle}, **headers)
            assert upload.status_code == 201, upload.content
            response = client.post(
                "/v1/jobs",
                data=json.dumps(
                    {
                        "job_type": "transcribe",
                        "profile": "ad-analysis-v1",
                        "language_hint": "de",
                        "input": {"kind": "upload", "upload_id": upload.json()["id"]},
                        "output": {"formats": ["json"]},
                        "task_ref": f"g2-{key}",
                    }
                ),
                content_type="application/json",
                **headers,
            )
            assert response.status_code == 201, response.content
            job = response.json()
            if job["state"] != "succeeded":
                raise SystemExit(f"{key}: job {job['state']}: {job.get('error')}")
            job_seconds = time.monotonic() - started
            words = client.get(job["result"]["artifacts"]["words"], **headers)
            assert words.status_code == 200
            document = json.loads(words.content)
            deleted = client.delete(f"/v1/jobs/{job['id']}", **headers)
            left = sorted(
                str(path.relative_to(artifact_root))
                for path in artifact_root.rglob("*")
                if path.is_file()
            )
            entry = {
                "key": key,
                "job_seconds": round(job_seconds, 1),
                "processing_seconds": job["result"]["metadata"].get("processing_seconds"),
                "load_before": [round(value, 2) for value in load_before],
                "load_after": [round(value, 2) for value in os.getloadavg()],
                "delete_status": deleted.status_code,
                "artifact_files_left": left,
                "drift_summary": document["drift"]["summary"],
            }
            target.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
            log.append(entry)
            print(json.dumps(entry), flush=True)
    finally:
        with (args.out / "run-log.jsonl").open("a", encoding="utf-8") as handle:
            for entry in log:
                handle.write(json.dumps(entry) + "\n")
        connection.creation.destroy_test_db(old_name, verbosity=0)
        teardown_test_environment()


# --- analyse ---------------------------------------------------------------------------


def research_lib(research: Path) -> Any:
    sys.path.insert(0, str(research))
    import lib  # type: ignore[import-not-found]

    return lib


def sentence_segs(words: list[dict[str, Any]]) -> list[tuple[float, float, str]]:
    """``splice.py`` / ``convert.py`` ``sentence_segs``, verbatim."""
    out: list[tuple[float, float, str]] = []
    cur: list[dict[str, Any]] = []

    def flush() -> None:
        if cur:
            out.append((cur[0]["s"], cur[-1]["e"], " ".join(w["w"] for w in cur)))

    for w in words:
        if cur and (
            w["s"] - cur[-1]["e"] > 1.5 or w["e"] - cur[0]["s"] > 6 or cur[-1]["w"][-1:] in ".?!"
        ):
            flush()
            cur.clear()
        cur.append(w)
    flush()
    return out


def research_splice(
    pk_words: list[dict[str, Any]], retries: dict[float, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """``splice.py``'s word list for one key, verbatim apart from names."""
    ts = sorted(retries)
    new = [w for w in pk_words if int(w["s"] // 30.0) * 30 not in ts] + [
        w for t in ts for w in retries[t]
    ]
    new.sort(key=lambda w: w["s"])
    return new


def wse(words: list[dict[str, Any]]) -> list[tuple[str, float, float]]:
    return [(w["w"], w["s"], w["e"]) for w in words]


def analyse(args: argparse.Namespace) -> None:
    lib = research_lib(args.research)
    episodes = german_episodes(args.workspace)
    rows = json.loads((args.research / "win-dev.json").read_text(encoding="utf-8"))
    row_by = {(r["key"], float(r["t"])): r for r in rows}
    retries = {}
    for line in (args.research / "retry-dev-cand-auto.jsonl").read_text().splitlines():
        record = json.loads(line)
        retries[(record["key"], float(record["t"]))] = record

    def research_flag(r: dict[str, Any]) -> bool:
        return r["e"] >= RULE_E and r["e"] / max(1, r["e"] + r["g"]) >= RULE_S

    research_flagged = {k for k, r in row_by.items() if research_flag(r)}
    ours: dict[tuple[str, float], dict[str, Any]] = {}
    documents: dict[str, dict[str, Any]] = {}
    for key in episodes:
        document = json.loads((args.out / f"{key}.json").read_text(encoding="utf-8"))
        documents[key] = document
        assert document["drift"]["checked"] is True
        for record in document["drift"]["windows"]:
            ours[(key, float(record["t"]))] = record
    our_flagged = {k for k, r in ours.items() if r["flagged"]}

    count_mismatch = [
        k
        for k, r in ours.items()
        if (r["en"], r["de"], r["words_before"]) != (row_by[k]["e"], row_by[k]["g"], row_by[k]["n"])
    ]
    missing_rows = [k for k, r in row_by.items() if r["n"] > 0 and k not in ours]

    # Removal, with the research's turbo reference te per window.
    total_excess = sum(max(0, r["e"] - r["te"]) for r in rows)
    captured = sum(max(0, row_by[k]["e"] - row_by[k]["te"]) for k in our_flagged)
    pk_en = sum(row_by[k]["e"] for k in our_flagged)
    ref_en = sum(row_by[k]["te"] for k in our_flagged)
    after_en_by_window: dict[tuple[str, float], int] = {}
    for key, document in documents.items():
        grid: dict[float, list[str]] = collections.defaultdict(list)
        for w in document["words"]:
            grid[int(w["s"] // 30.0) * 30.0].append(w["w"])
        for t, ws in grid.items():
            after_en_by_window[(key, t)] = lib.counts(lib.toks(" ".join(ws)))[0]
    after_en = sum(after_en_by_window.get(k, 0) for k in our_flagged)
    excess_after = sum(
        max(0, after_en_by_window.get((r["key"], float(r["t"])), 0) - r["te"]) for r in rows
    )
    accepted = [k for k in our_flagged if ours[k].get("accepted")]
    rejected = [k for k in our_flagged if not ours[k].get("accepted")]
    rule_fixed = sum(
        after_en_by_window.get(k, 0) <= row_by[k]["te"] + 1 for k in our_flagged
    )  # evalretry.py's rule_fixed

    retry_seconds = [ours[k]["retry_seconds"] for k in our_flagged]
    hours = sum(episode["dur"] for episode in episodes.values()) / 3600

    # Splice agreement with splice.py (+ the adopted language and p >= 0.9 rule).
    word_rows: list[dict[str, Any]] = []
    lines_identical = 0
    window_words_identical = 0
    research_accept = 0
    for key, document in documents.items():
        pk = json.loads((PK_DEV / f"{key}.json").read_text(encoding="utf-8"))["words"]
        accepted_retries = {}
        for (k, t), record in retries.items():
            if k != key or (k, t) not in research_flagged:
                continue
            if record["lang"] == "german" and (record["p"] or 0) >= 0.9:
                accepted_retries[t] = record["words"]
        research_accept += len(accepted_retries)
        expected = research_splice(pk, accepted_retries)
        got = document["words"]
        same = wse(got) == wse(expected)
        same_text = [w["w"] for w in got] == [w["w"] for w in expected]
        for t, words in accepted_retries.items():
            mine = [w for w in got if t <= w["s"] < t + 30]
            window_words_identical += wse(mine) == wse(words)
        research_lines = [
            (seg["offsets"]["from"] / 1000, seg["offsets"]["to"] / 1000, seg["text"].strip())
            for seg in json.loads(
                (args.research / "tx-dev" / "pk30s-fix" / f"{key}.json").read_text("utf-8")
            )["transcription"]
        ]
        my_lines = [
            (round(a * 1000) / 1000, round(b * 1000) / 1000, t) for a, b, t in sentence_segs(got)
        ]
        lines_same = my_lines == research_lines
        lines_identical += lines_same
        word_rows.append(
            {
                "key": key,
                "words": len(got),
                "research_words": len(expected),
                "identical": same,
                "same_text": same_text,
                "lines_identical": lines_same,
                "lines": len(my_lines),
            }
        )

    flagged_retry_records = [r for k, r in retries.items() if k in research_flagged]
    research_wall = [r["wall_s"] + r["ffmpeg_s"] for r in flagged_retry_records]
    log = []
    log_path = args.out / "run-log.jsonl"
    if log_path.exists():
        log = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    summary = {
        "episodes": len(documents),
        "german_hours": round(hours, 2),
        "grid_windows_research": len(rows),
        "grid_windows_ours": len(ours),
        "pk_windows_research": sum(1 for r in rows if r["n"] > 0),
        "missing_windows": len(missing_rows),
        "count_mismatches": len(count_mismatch),
        "flagged_ours": len(our_flagged),
        "flagged_research": len(research_flagged),
        "flag_sets_equal": our_flagged == research_flagged,
        "flagged_only_ours": sorted(our_flagged - research_flagged),
        "flagged_only_research": sorted(research_flagged - our_flagged),
        "flagged_share_pct": round(100 * len(our_flagged) / len(rows), 2),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "rejected_windows": [
            [k[0], k[1], ours[k]["language"], ours[k]["p"]] for k in sorted(rejected)
        ],
        "excess_en_total": total_excess,
        "excess_en_captured": captured,
        "excess_en_captured_pct": round(100 * captured / total_excess, 1),
        "flagged_en_parakeet": pk_en,
        "flagged_en_after": after_en,
        "flagged_en_turbo_reference": ref_en,
        "flagged_rule_fixed": rule_fixed,
        "excess_en_after": excess_after,
        "excess_en_removed_pct": round(100 * (total_excess - excess_after) / total_excess, 1),
        "retry_seconds_total": round(sum(retry_seconds), 1),
        "retry_seconds_median": round(statistics.median(retry_seconds), 3),
        "retry_seconds_per_german_hour": round(sum(retry_seconds) / hours, 1),
        "research_retry_seconds_median": round(statistics.median(research_wall), 3),
        "research_retry_seconds_per_german_hour": round(sum(research_wall) / hours, 1),
        "research_accepted": research_accept,
        "episodes_words_identical": sum(r["identical"] for r in word_rows),
        "episodes_same_text": sum(r["same_text"] for r in word_rows),
        "words_total": sum(r["words"] for r in word_rows),
        "research_words_total": sum(r["research_words"] for r in word_rows),
        "accepted_windows_words_identical": window_words_identical,
        "episodes_lines_identical": lines_identical,
        "job_seconds_total": round(sum(e["job_seconds"] for e in log), 1),
        "load_min": min((e["load_before"][0] for e in log), default=None),
        "load_max": max((max(e["load_before"][0], e["load_after"][0]) for e in log), default=None),
        "delete_statuses": sorted({e["delete_status"] for e in log}),
        "artifact_files_left": sorted({f for e in log for f in e["artifact_files_left"]}),
    }
    (args.out / "analysis.json").write_text(
        json.dumps({"summary": summary, "episodes": word_rows}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    for row in word_rows:
        if not (row["identical"] and row["lines_identical"]):
            print("differs:", row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["run", "analyse"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--research", type=Path, default=RESEARCH)
    parser.add_argument("--parakeet-bin", default="parakeet-mlx")
    parser.add_argument("--parakeet-model-dir", type=Path, default=PARAKEET_MODEL_DIR)
    parser.add_argument("--turbo-model", type=Path, default=TURBO_MODEL)
    parser.add_argument("--keys", nargs="*", help="only these episodes (run)")
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        analyse(args)


if __name__ == "__main__":
    main()
