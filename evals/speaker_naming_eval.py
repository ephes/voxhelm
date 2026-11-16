"""Offline published-speaker evaluation; see docs/design/diarization-quality.md.

Run as ``uv run python -m evals.speaker_naming_eval score ...``.
Truth JSON: {"fps": 10, "frames": ["Alice", null, ...]}; null is not clear speech.
Frame i samples time i/fps. Names must match the sidecar's display names.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import tempfile
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from time import monotonic
from typing import Any

from evals.speaker_generations import (
    file_digest,
    json_digest,
    publisher,
    source_identity,
    verify_bundle_identity,
)
from evals.speaker_inputs import (
    load_arrays,
    read_json,
    require,
    validate_bundle,
    validate_manifest,
    validate_sidecar,
    validate_truth,
)

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
    frames, fps = validate_truth(truth) if truth is not None else ([], 10.0)
    validate_sidecar(sidecar, frames, fps)
    if baseline is not None:
        validate_sidecar(baseline, frames, fps)
    names = published_names(sidecar, list(guests))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Truth fps must be finite and positive")
    frame_counts: Counter[str] = Counter()
    events: dict[int, Counter[str | None]] = {}
    covered_count = 0
    buckets: dict[str, Counter[str]] = {name: Counter() for _, name in BUCKETS}
    buckets[">=5"] = Counter()
    reasons: dict[str, Counter[str]] = {}
    lines: Counter[str] = Counter()
    changes: list[dict[str, Any]] = []
    short_guest_known_names = {"before": 0, "after": 0}
    known = sidecar["summary"].get("known_speakers", [])
    before_names = published_names(baseline, list(guests)) if baseline else []
    before = (
        {
            (s["start"], s["end"]): name
            for s, name in zip(baseline["segments"], before_names, strict=True)
        }
        if baseline
        else {}
    )
    if baseline:
        boundaries = [(s["start"], s["end"]) for s in sidecar["segments"]]
        if (
            len(before) != len(baseline["segments"])
            or len(boundaries) != len(before)
            or set(boundaries) != set(before)
        ):
            raise ValueError("Baseline segment boundaries differ; cannot compare lines")
    for segment, name in zip(sidecar["segments"], names, strict=True):
        start, end = float(segment["start"]), float(segment["end"])
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError("Invalid segment boundaries")
        indices = range(max(0, math.ceil(start * fps)), min(len(frames), math.ceil(end * fps)))
        golds = Counter(frames[i] for i in indices if frames[i] is not None)
        if indices:
            events.setdefault(indices.start, Counter())[name] += 1
            events.setdefault(indices.stop, Counter())[name] -= 1
        gold = None
        if golds:
            top, count = golds.most_common(1)[0]
            if sum(golds.values()) >= 3 and count / sum(golds.values()) >= 0.8:
                gold = top
        if gold in guests and 0.3 <= end - start <= 2.5:
            short_guest_known_names["after"] += name in known
            if baseline:
                short_guest_known_names["before"] += before.get((start, end)) in known
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

    active: Counter[str | None] = Counter()
    for i, gold_frame in enumerate(frames):
        active.update(events.get(i, {}))
        labels = {label for label, count in active.items() if count > 0}
        if gold_frame is not None:
            covered_count += bool(labels)
            label = next(iter(labels)) if len(labels) == 1 else None
            frame_counts[outcome(label, gold_frame)] += 1
    clear_changes = [c for c in changes if c["truth"] is not None]
    return {
        "frames": {
            **rates(frame_counts),
            "covered": covered_count,
            "coverage_pct": 100 * covered_count / sum(frame_counts.values())
            if frame_counts
            else None,
        },
        "lines": line_metrics(lines),
        "duration_buckets": {key: line_metrics(value) for key, value in buckets.items()},
        "accept_reasons": {key: line_metrics(value) for key, value in reasons.items()},
        "published_speaker_distribution": dict(Counter(name for name in names if name)),
        "new_names_by_speaker": dict(
            Counter(c["after"] for c in changes if c["before"] is None and c["after"] is not None)
        ),
        "short_guest_known_names": short_guest_known_names,
        "guest_to_host_lines": [
            c for c in changes if c["before"] in guests and c["after"] in known
        ],
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


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def manifest_diarization_params(manifest: dict[str, Any]) -> Any:
    """Speaker counts as the production request passes them to pyannote."""
    from transcriptions.diarization import DiarizationParams

    counts = {key: manifest.get(key) for key in ("num_speakers", "min_speakers", "max_speakers")}
    for key, value in counts.items():
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise ValueError(f"Manifest {key} must be a positive integer")
    if counts["num_speakers"] is not None and (
        counts["min_speakers"] is not None or counts["max_speakers"] is not None
    ):
        raise ValueError("Manifest num_speakers cannot be combined with min/max_speakers")
    if (
        counts["min_speakers"] is not None
        and counts["max_speakers"] is not None
        and counts["min_speakers"] > counts["max_speakers"]
    ):
        raise ValueError("Manifest min_speakers cannot exceed max_speakers")
    return DiarizationParams(**counts)


def create_bundle(manifest_path: Path, destination: Path) -> None:
    """Publish a complete absent bundle; remove only this attempt's failed stage."""
    with publisher(destination.parent, destination.name):
        if os.path.lexists(destination):
            raise ValueError("Bundle destination already exists; choose a new directory")
        stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
        try:
            _capture_bundle(manifest_path, stage)
            frozen = read_json(stage / "bundle.json")
            load_arrays(stage, validate_bundle(frozen))
            if os.path.lexists(destination):
                raise ValueError("Bundle destination already exists; choose a new directory")
            stage.rename(destination)
        finally:
            if stage.exists():
                shutil.rmtree(stage)


def _capture_bundle(manifest_path: Path, destination: Path) -> None:
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
    validate_manifest(manifest)
    audio = (manifest_path.parent / manifest["audio"]).resolve()
    identity = source_identity(manifest_path, manifest)
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
    ).diarize_annotations(audio, manifest_diarization_params(manifest))
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
    if source_identity(manifest_path, read_json(manifest_path)) != identity:
        raise ValueError("Evaluation source identity changed during capture")
    write_json(
        destination / "bundle.json",
        {
            "version": 2,
            "identity": identity,
            "audio": "audio.npy",
            "result": asdict(result),
            "turns": {
                name: [asdict(turn) for turn in turns] for name, turns in annotations.items()
            },
            "references": references,
            "manifest": manifest,
        },
    )


def run_bundle(
    bundle_path: Path,
    config_data: dict[str, Any],
    *,
    diagnostics_path: Path | None = None,
) -> tuple[dict[str, Any], float]:
    """Only embeddings/postprocessing run here; wall time stays outside the sidecar."""
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
    members = validate_bundle(bundle)
    arrays = iter(load_arrays(bundle_path, members))
    audio = next(arrays)
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
            windows=[next(arrays) for name in reference["windows"]],
        )
        for reference in bundle["references"]
    ]
    backend = get_known_speaker_backend(config.embedding_model)
    raw_turns = [SpeakerTurn(**turn) for turn in bundle["turns"]["regular"]]
    started = monotonic()
    outcome = run_known_speaker_postprocess(
        TranscriptionResult(**result_data),
        references=references,
        job_audio_samples=audio,
        raw_turns=raw_turns,
        config=config,
        backend=backend,
    )
    elapsed = monotonic() - started
    sidecar = build_speakers_artifact(outcome)
    for segment, transcript in zip(sidecar["segments"], outcome.result.segments, strict=True):
        segment["text"] = transcript.text
    if diagnostics_path is not None:
        write_json(
            diagnostics_path,
            profile_diagnostics(
                outcome.segments,
                audio,
                raw_turns,
                backend,
                list(dict.fromkeys(reference.name for reference in references)),
            ),
        )
    return sidecar, elapsed


def profile_diagnostics(
    segments: Any, audio: Any, raw_turns: Any, backend: Any, known_speakers: list[str]
) -> dict[str, Any]:
    """Listening-check sanity metric for long guest-rule cues; outside job runtime."""
    from transcriptions.diarization import SpeakerTurnIndex, normalize_speaker_turns
    from transcriptions.known_speaker import (
        SAMPLE_RATE,
        build_episode_centroids,
        classify_embedding,
        raw_turn_overlaps,
        single_raw_label,
        slice_samples,
        summarize_raw_labels,
    )

    labels = summarize_raw_labels(segments, raw_turns)
    profiles = build_episode_centroids(
        segments,
        job_audio_samples=audio,
        reference_centroids={name: [] for name in known_speakers},
        raw_turns=raw_turns,
        backend=backend,
    )
    checks = {
        label: {"lines": 0, "top_label": 0}
        for label, entry in labels.items()
        if entry["speaker"] is None
    }
    turns = SpeakerTurnIndex(normalize_speaker_turns(raw_turns) if raw_turns else [])
    for segment in segments:
        label = segment.raw_diarization_speaker
        if (
            label not in checks
            or not segment.speaker_uncertain
            or segment.speaker_uncertain_reason == "mixed_speakers"
            or segment.end - segment.start < 3
            or single_raw_label(raw_turn_overlaps(segment.start, segment.end, turns)) != label
        ):
            continue
        samples = slice_samples(audio, SAMPLE_RATE, segment.start, segment.end)
        if not profiles or not len(samples):
            continue
        winner = classify_embedding(backend.embed(samples, SAMPLE_RATE), profiles).top_speaker
        checks[label]["lines"] += 1
        checks[label]["top_label"] += winner == f"label:{label}"
    return {
        label: {
            **counts,
            "top_label_pct": 100 * counts["top_label"] / counts["lines"]
            if counts["lines"]
            else None,
        }
        for label, counts in checks.items()
    }


EPISODES = ("pp59", "pp77", "pp78", "pp80", "dc209", "dc210")


def validate_tag(tag: str) -> None:
    reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    require(
        isinstance(tag, str)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", tag) is not None
        and not tag.endswith(".")
        and tag.split(".")[0].upper() not in reserved,
        "suite.tag",
    )


def complete_generation(path: Path) -> dict[str, Any]:
    complete = read_json(path / "complete.json")
    require(
        complete.get("version") == 1 and isinstance(complete.get("members"), dict),
        "suite.completion",
    )
    expected = {f"{ep}{suffix}.json" for ep in EPISODES for suffix in ("", ".score", ".profiles")}
    require(
        set(complete["members"]) == expected
        and {p.name for p in path.iterdir()} == expected | {"complete.json"},
        "suite.members",
    )
    for name, digest in complete["members"].items():
        require(file_digest(path / name) == digest, "suite.member.identity")
    return complete


def run_suite(root: Path, config_data: dict[str, Any], tag: str) -> None:
    """Publish only a complete deterministic six-episode generation."""
    from transcriptions.known_speaker import known_speaker_config_from_mapping

    validate_tag(tag)
    config = asdict(known_speaker_config_from_mapping(config_data))
    parent = root / "results"
    with publisher(parent, tag):
        output = parent / tag
        # A legacy real directory cannot be atomically replaced by a pointer.
        require(
            not os.path.lexists(output) or output.is_symlink(),
            "suite.tag (archive legacy directory before rerunning)",
        )
        if os.path.lexists(output):
            complete_generation(output.resolve(strict=True))
        baseline_dir = None
        if tag != "baseline":
            baseline_dir = (parent / "baseline").resolve(strict=True)
            baseline_complete = complete_generation(baseline_dir)
            require(
                baseline_complete["config"]["embedding_model"] == config["embedding_model"],
                "suite.embedding_model (keep the baseline embedding model)",
            )
        # Freeze all controls and validate existing bundle identities before inference.
        inputs = []
        for episode in EPISODES:
            manifest_path = root / "manifests" / f"{episode}.json"
            manifest = read_json(manifest_path)
            validate_manifest(manifest)
            truth_path = root / "truth" / f"{episode}.json"
            if episode.startswith("pp") and not truth_path.is_file():
                raise ValueError(f"Missing required truth: {episode}")
            truth = read_json(truth_path) if truth_path.exists() else None
            if truth is not None:
                validate_truth(truth)
            baseline = read_json(baseline_dir / f"{episode}.json") if baseline_dir else None
            if baseline is not None:
                validate_sidecar(baseline, [], 10)
            bundle = root / "bundles" / episode
            if os.path.lexists(bundle):
                verify_bundle_identity(manifest_path, bundle)
            inputs.append((episode, manifest_path, manifest, truth, baseline, bundle))
        stage = Path(tempfile.mkdtemp(prefix=f".{tag}.stage-", dir=parent))
        try:
            identities = {}
            for episode, manifest_path, manifest, truth, baseline, bundle in inputs:
                if not os.path.lexists(bundle):
                    create_bundle(manifest_path, bundle)
                identity = verify_bundle_identity(manifest_path, bundle)
                bundle_digest = file_digest(bundle / "bundle.json")
                first, elapsed = run_bundle(
                    bundle, config_data, diagnostics_path=stage / f"{episode}.profiles.json"
                )
                second, _ = run_bundle(bundle, config_data)
                if first != second:
                    raise ValueError(f"Nondeterministic replay: {episode}")
                require(
                    file_digest(bundle / "bundle.json") == bundle_digest
                    and verify_bundle_identity(manifest_path, bundle) == identity,
                    "suite.bundle.identity",
                )
                write_json(stage / f"{episode}.json", first)
                report = score(
                    first,
                    truth,
                    guests=manifest.get("unreferenced_speakers", []),
                    baseline=baseline,
                )
                report["postprocess_seconds"] = elapsed
                report["deterministic_replay"] = True
                write_json(stage / f"{episode}.score.json", report)
                identities[episode] = {
                    "bundle_identity": identity,
                    "bundle_json_sha256": bundle_digest,
                    "truth_sha256": json_digest(truth),
                    "baseline_sha256": json_digest(baseline),
                }
            members = {path.name: file_digest(path) for path in stage.iterdir()}
            write_json(
                stage / "complete.json",
                {"version": 1, "config": config, "episodes": identities, "members": members},
            )
            complete_generation(stage)
            generation = parent / f".{tag}.generation-{uuid.uuid4().hex}"
            stage.rename(generation)
            pointer = parent / f".{tag}.pointer-{uuid.uuid4().hex}"
            try:
                pointer.symlink_to(generation.name, target_is_directory=True)
                pointer.replace(output)
            finally:
                pointer.unlink(missing_ok=True)
        except BaseException as exc:
            exc.add_note(f"Owned suite recovery stage retained: {stage}")
            raise


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
    running.add_argument("--profile-checks", action="store_true")
    suite = commands.add_parser("suite")
    suite.add_argument("root", type=Path)
    suite.add_argument("--env-file", type=Path, required=True)
    suite.add_argument("--config", type=Path)
    suite.add_argument("--tag", required=True)
    args = parser.parse_args()
    if args.command == "suite":
        validate_tag(args.tag)
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
                args.bundle,
                read_json(args.config) if args.config else {},
                diagnostics_path=args.out.with_suffix(".profiles.json")
                if args.profile_checks
                else None,
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
    try:
        main()
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from None
