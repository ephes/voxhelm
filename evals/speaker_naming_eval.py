"""Offline published-speaker evaluation; see docs/design/diarization-quality.md.

Run as ``uv run python -m evals.speaker_naming_eval score ...``.
Truth JSON: {"fps": 10, "frames": ["Alice", null, ...]}; null is not clear speech.
Frame i samples time i/fps. Names must match the sidecar's display names.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from time import monotonic
from typing import Any

BUCKETS = ((0.5, "<0.5"), (1, "0.5-1"), (1.5, "1-1.5"), (2.5, "1.5-2.5"), (5, "2.5-5"))


def duration_bucket(duration: float) -> str:
    return next((name for upper, name in BUCKETS if duration < upper), ">=5")


def published_names(sidecar: dict[str, Any], guests: list[str]) -> list[str | None]:
    """Mirror cast: one unreferenced contributor + one untied label, no mixed cues."""
    labels = sidecar["summary"].get("raw_diarization_labels", {})
    untied = [label for label, entry in labels.items() if entry.get("speaker") is None]
    guest_label = untied[0] if len(untied) == 1 and len(guests) == 1 else None
    return [
        segment.get("speaker")
        if not segment["speaker_uncertain"]
        else guests[0]
        if guest_label is not None
        and segment.get("raw_diarization_speaker") == guest_label
        and segment.get("speaker_uncertain_reason") != "mixed_speakers"
        else None
        for segment in sidecar["segments"]
    ]


def rates(counts: Counter[str]) -> dict[str, Any]:
    total = sum(counts.values())
    return {
        "total": total,
        **{key: counts[key] for key in ("right", "wrong", "unnamed")},
        **{
            f"{key}_pct": 100 * counts[key] / total if total else None
            for key in ("right", "wrong", "unnamed")
        },
    }


def outcome(name: str | None, gold: str) -> str:
    return "unnamed" if name is None else "right" if name == gold else "wrong"


def score(
    sidecar: dict[str, Any],
    truth: dict[str, Any] | None = None,
    *,
    guests: Sequence[str] = (),
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    names = published_names(sidecar, list(guests))
    frames = truth["frames"] if truth else []
    fps = float(truth["fps"]) if truth else 10.0
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Truth fps must be finite and positive")
    frame_counts: Counter[str] = Counter()
    covered: set[int] = set()
    buckets: dict[str, Counter[str]] = {name: Counter() for _, name in BUCKETS}
    buckets[">=5"] = Counter()
    reasons: dict[str, Counter[str]] = {}
    lines: Counter[str] = Counter()
    changes: list[dict[str, Any]] = []
    before_names = published_names(baseline, list(guests)) if baseline else []
    before = (
        {
            (s["start"], s["end"]): name
            for s, name in zip(baseline["segments"], before_names, strict=True)
        }
        if baseline
        else {}
    )
    for segment, name in zip(sidecar["segments"], names, strict=True):
        start, end = float(segment["start"]), float(segment["end"])
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError("Invalid segment boundaries")
        indices = range(max(0, math.ceil(start * fps)), min(len(frames), math.ceil(end * fps)))
        golds = Counter(frames[i] for i in indices if frames[i] is not None)
        for i in indices:
            if i not in covered and frames[i] is not None:
                frame_counts[outcome(name, frames[i])] += 1
                covered.add(i)
        gold = None
        if golds:
            top, count = golds.most_common(1)[0]
            if sum(golds.values()) >= 3 and count / sum(golds.values()) >= 0.8:
                gold = top
        status = outcome(name, gold) if gold else "unclear"
        bucket = buckets[duration_bucket(end - start)]
        reason = (
            "guest_rule"
            if name and segment["speaker_uncertain"]
            else segment.get("speaker_accept_reason")
            or ("uncertain" if segment["speaker_uncertain"] else "whole_segment")
        )
        for counter in (lines, bucket, reasons.setdefault(reason, Counter())):
            counter["lines"] += 1
            counter["unnamed_lines"] += name is None
            counter[status] += 1
        if baseline:
            key = (segment["start"], segment["end"])
            if key not in before:
                raise ValueError("Baseline segment boundaries differ; cannot compare lines")
            if before[key] != name:
                changes.append(
                    {
                        "start": start,
                        "end": end,
                        "text": segment.get("text", ""),
                        "before": before[key],
                        "after": name,
                        "truth": gold,
                        "before_right": before[key] == gold if gold else None,
                        "after_right": name == gold if gold else None,
                    }
                )

    def line_metrics(counter: Counter[str]) -> dict[str, Any]:
        clear = sum(counter[key] for key in ("right", "wrong", "unnamed"))
        return {
            **dict(counter),
            "unnamed_pct": 100 * counter["unnamed_lines"] / counter["lines"]
            if counter["lines"]
            else None,
            "clear_lines": clear,
            "right_pct": 100 * counter["right"] / clear if clear else None,
        }

    clear_changes = [c for c in changes if c["truth"] is not None]
    return {
        "frames": rates(frame_counts),
        "lines": line_metrics(lines),
        "duration_buckets": {key: line_metrics(value) for key, value in buckets.items()},
        "accept_reasons": {key: line_metrics(value) for key, value in reasons.items()},
        "changes": changes,
        "changes_precision": {
            "clear_lines": len(clear_changes),
            **{
                f"{key}_right_pct": 100
                * sum(c[f"{key}_right"] for c in clear_changes)
                / len(clear_changes)
                if clear_changes
                else None
                for key in ("before", "after")
            },
        },
    }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def create_bundle(manifest_path: Path, destination: Path) -> None:
    """Freeze production ASR, both anonymous annotations and decoded reference windows.

    Manifest paths are relative to the manifest, not the current directory.
    No network downloads, jobs or control-plane writes are performed.
    """
    import numpy
    from django.conf import settings

    from transcriptions.diarization import PyannoteDiarizationBackend
    from transcriptions.known_speaker import (
        REFERENCE_MAX_FRAMES,
        SAMPLE_RATE,
        decode_mono_16k,
        extract_reference_windows,
    )
    from transcriptions.service import TranscribeParams, transcribe_audio

    manifest = read_json(manifest_path)
    audio = (manifest_path.parent / manifest["audio"]).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    result = transcribe_audio(
        audio,
        TranscribeParams(
            request_model=manifest.get("model", "auto"),
            prompt=manifest.get("prompt"),
            language=manifest.get("language"),
            word_timestamps=True,
        ),
    )
    annotations = PyannoteDiarizationBackend(
        model_name=settings.VOXHELM_PYANNOTE_MODEL,
        auth_token=settings.VOXHELM_HUGGINGFACE_TOKEN,
        device_name=settings.VOXHELM_PYANNOTE_DEVICE,
    ).diarize_annotations(audio)
    numpy.save(destination / "audio.npy", decode_mono_16k(audio), allow_pickle=False)
    references = []
    for index, reference in enumerate(manifest["references"]):
        path = (manifest_path.parent / reference.get("audio", manifest["audio"])).resolve()
        samples = decode_mono_16k(
            path,
            start=reference.get("start"),
            end=reference.get("end"),
            max_frames=REFERENCE_MAX_FRAMES,
        )
        windows = extract_reference_windows(samples, SAMPLE_RATE)
        files = []
        for window_index, window in enumerate(windows):
            name = f"reference-{index}-{window_index}.npy"
            numpy.save(destination / name, window, allow_pickle=False)
            files.append(name)
        references.append(
            {
                "id": reference["id"],
                "name": reference["name"],
                "windows": files,
                "source": reference,
            }
        )
    write_json(
        destination / "bundle.json",
        {
            "version": 1,
            "audio": "audio.npy",
            "result": asdict(result),
            "turns": {
                name: [asdict(turn) for turn in turns] for name, turns in annotations.items()
            },
            "references": references,
            "manifest": manifest,
        },
    )


def run_bundle(bundle_path: Path, config_data: dict[str, Any]) -> tuple[dict[str, Any], float]:
    """Only embeddings/postprocessing run here; wall time stays outside the sidecar."""
    import numpy

    from transcriptions.diarization import SpeakerTurn
    from transcriptions.known_speaker import (
        ReferenceAudio,
        build_speakers_artifact,
        get_known_speaker_backend,
        known_speaker_config_from_mapping,
        run_known_speaker_postprocess,
    )
    from transcriptions.service import TranscriptionResult, TranscriptionSegment, TranscriptionWord

    bundle = read_json(bundle_path / "bundle.json")
    if bundle["version"] != 1:
        raise ValueError("Unsupported bundle version")
    config = known_speaker_config_from_mapping(config_data)
    result_data = dict(bundle["result"])
    result_data["segments"] = [
        TranscriptionSegment(
            **{
                **segment,
                "words": tuple(TranscriptionWord(**word) for word in segment.get("words", [])),
            }
        )
        for segment in result_data["segments"]
    ]
    references = [
        ReferenceAudio(
            speaker_id=reference["id"],
            name=reference["name"],
            windows=[
                numpy.load(bundle_path / name, allow_pickle=False) for name in reference["windows"]
            ],
        )
        for reference in bundle["references"]
    ]
    started = monotonic()
    outcome = run_known_speaker_postprocess(
        TranscriptionResult(**result_data),
        references=references,
        job_audio_samples=numpy.load(bundle_path / bundle["audio"], allow_pickle=False),
        raw_turns=[SpeakerTurn(**turn) for turn in bundle["turns"]["regular"]],
        config=config,
        backend=get_known_speaker_backend(config.embedding_model),
    )
    elapsed = monotonic() - started
    sidecar = build_speakers_artifact(outcome)
    for segment, transcript in zip(sidecar["segments"], outcome.result.segments, strict=True):
        segment["text"] = transcript.text
    return sidecar, elapsed


def run_suite(root: Path, config_data: dict[str, Any], tag: str) -> None:
    """The six required episodes, with deterministic replay and published-name diffs."""
    output = root / "results" / tag
    output.mkdir(parents=True, exist_ok=True)
    for episode in ("pp59", "pp77", "pp78", "pp80", "dc209", "dc210"):
        manifest_path = root / "manifests" / f"{episode}.json"
        manifest = read_json(manifest_path)
        bundle = root / "bundles" / episode
        if not (bundle / "bundle.json").exists():
            create_bundle(manifest_path, bundle)
        first, elapsed = run_bundle(bundle, config_data)
        second, _ = run_bundle(bundle, config_data)
        if first != second:
            raise ValueError(f"Nondeterministic replay: {episode}")
        write_json(output / f"{episode}.json", first)
        baseline_path = root / "results" / "baseline" / f"{episode}.json"
        baseline = read_json(baseline_path) if tag != "baseline" else None
        truth_path = root / "truth" / f"{episode}.json"
        if episode.startswith("pp") and not truth_path.is_file():
            raise ValueError(f"Missing required truth: {truth_path}")
        report = score(
            first,
            read_json(truth_path) if truth_path.is_file() else None,
            guests=manifest.get("unreferenced_speakers", []),
            baseline=baseline,
        )
        report["postprocess_seconds"] = elapsed
        report["deterministic_replay"] = True
        write_json(output / f"{episode}.score.json", report)
        print(episode, json.dumps({key: report[key] for key in ("frames", "lines")}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    scoring = commands.add_parser("score")
    scoring.add_argument("sidecar", type=Path)
    scoring.add_argument("--truth", type=Path)
    scoring.add_argument("--baseline", type=Path)
    scoring.add_argument("--guest", action="append", default=[])
    scoring.add_argument("--out", type=Path, required=True)
    bundling = commands.add_parser("bundle")
    bundling.add_argument("manifest", type=Path)
    bundling.add_argument("--out", type=Path, required=True)
    bundling.add_argument("--env-file", type=Path, required=True)
    running = commands.add_parser("run")
    running.add_argument("bundle", type=Path)
    running.add_argument("--config", type=Path, help="KnownSpeakerConfig JSON overrides")
    running.add_argument("--env-file", type=Path, required=True)
    running.add_argument("--out", type=Path, required=True)
    suite = commands.add_parser("suite")
    suite.add_argument("root", type=Path)
    suite.add_argument("--env-file", type=Path, required=True)
    suite.add_argument("--config", type=Path)
    suite.add_argument("--tag", required=True)
    args = parser.parse_args()
    if args.command != "score":
        from jobs.remote_worker_cli import load_env_file, setup_django_for_worker

        load_env_file(args.env_file)
        setup_django_for_worker()
        if args.command == "suite":
            run_suite(args.root, read_json(args.config) if args.config else {}, args.tag)
        elif args.command == "bundle":
            create_bundle(args.manifest, args.out)
        else:
            sidecar, elapsed = run_bundle(
                args.bundle, read_json(args.config) if args.config else {}
            )
            write_json(args.out, sidecar)
            write_json(args.out.with_suffix(".timing.json"), {"postprocess_seconds": elapsed})
        return
    write_json(
        args.out,
        score(
            read_json(args.sidecar),
            read_json(args.truth) if args.truth else None,
            guests=args.guest,
            baseline=read_json(args.baseline) if args.baseline else None,
        ),
    )


if __name__ == "__main__":
    main()
