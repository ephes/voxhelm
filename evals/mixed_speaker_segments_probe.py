"""Probes for the mixed-speaker segment investigation.

Offline validation tooling for ``specs/2026-10-12_mixed_speaker_segments.md``,
not part of the service. Every probe goes through the shipped code path and
loads the same env file the remote worker uses, so models, devices and STT
settings match what the worker would run.

Three subcommands:

``turns``
    Diarize a whole file and keep every raw pyannote turn (the service only
    keeps the dominant label per transcript segment).
``subwindow``
    Build known-speaker centroids from reference source ranges, classify whole
    segments, then classify sliding windows inside one segment.
``stt``
    Transcribe a file with each local STT backend the way the worker does, to
    compare segmentation and casing against a job's transcript.

To reproduce what a deployed worker did, run it with that worker's interpreter
so the installed package versions are the ones under test:

    ~/.local/share/uv/tools/voxhelm/bin/python -I evals/mixed_speaker_segments_probe.py \
        --env-file /etc/voxhelm-remote-worker/worker.env \
        turns episode.mp3 --hint none --hint num=2 --out turns.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

HINT_KEYS = {"num": "num_speakers", "min": "min_speakers", "max": "max_speakers"}
PACKAGES = ("voxhelm", "pyannote.audio", "torch", "mlx-whisper")
STT_SETTINGS = (
    "VOXHELM_STT_BACKEND",
    "VOXHELM_STT_FALLBACK_BACKEND",
    "VOXHELM_MLX_MODEL",
    "VOXHELM_MLX_CONDITION_ON_PREVIOUS_TEXT",
    "VOXHELM_WHISPERCPP_MODEL",
    "VOXHELM_WHISPERCPP_PROCESSORS",
    "VOXHELM_WHISPERCPP_MAX_CONTEXT",
    "VOXHELM_WHISPERCPP_SUPPRESS_NST",
    "VOXHELM_SANITIZE_TRANSCRIPT",
    "VOXHELM_SANITIZE_REPEAT_THRESHOLD",
)


def setup(env_file: Path | None) -> None:
    from jobs.remote_worker_cli import load_env_file, setup_django_for_worker

    if env_file is not None:
        load_env_file(env_file)
    setup_django_for_worker()

    import django

    django.setup()


def installed_versions() -> dict[str, str]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def parse_hint(hint: str) -> dict[str, int]:
    if hint == "none":
        return {}
    kwargs = {}
    for part in hint.split(","):
        key, _, value = part.partition("=")
        if key not in HINT_KEYS:
            raise SystemExit(f"Unknown speaker-count hint '{part}'; use none or num=/min=/max=.")
        kwargs[HINT_KEYS[key]] = int(value)
    return kwargs


def parse_range(value: str) -> tuple[float, float]:
    start, _, end = value.partition("-")
    return float(start), float(end)


def parse_window(value: str) -> tuple[float, float]:
    size, hop = parse_range(value)
    # A zero or negative hop would never advance past the target range.
    if not (math.isfinite(size) and math.isfinite(hop) and size > 0 and hop > 0):
        raise argparse.ArgumentTypeError(
            f"Window '{value}' must be size-hop with both finite and above zero."
        )
    return size, hop


def parse_reference(value: str) -> tuple[str, float, float]:
    name, _, bounds = value.partition("=")
    if not name or not bounds:
        raise SystemExit(f"Reference '{value}' must look like Name=start-end.")
    start, end = parse_range(bounds)
    return name, start, end


def write_report(report: dict[str, Any], out: Path | None) -> None:
    if out is not None:
        out.write_text(json.dumps(report, indent=1) + "\n")
        print(f"wrote {out}")


def serialize_turns(annotation: Any) -> list[dict[str, Any]]:
    return [
        {
            "start": round(float(segment.start), 3),
            "end": round(float(segment.end), 3),
            "speaker": str(label),
        }
        for segment, _track, label in annotation.itertracks(yield_label=True)
    ]


def run_turns(args: argparse.Namespace) -> None:
    from django.conf import settings

    from transcriptions.diarization import (
        DiarizationParams,
        PyannoteDiarizationBackend,
        build_diarization_backend_service,
        extract_pyannote_annotation,
        load_audio_for_pyannote,
    )

    backend = build_diarization_backend_service(backend_name=settings.VOXHELM_DIARIZATION_BACKEND)
    if not isinstance(backend, PyannoteDiarizationBackend):
        raise SystemExit("Set VOXHELM_DIARIZATION_BACKEND=pyannote to keep raw turns.")
    pipeline = backend._load_pipeline()
    audio_input = load_audio_for_pyannote(args.audio)

    report: dict[str, Any] = {
        "audio": args.audio.name,
        "pyannote_model": settings.VOXHELM_PYANNOTE_MODEL,
        "pyannote_device": str(pipeline.device),
        "versions": installed_versions(),
        "runs": {},
    }
    for hint in args.hint or ["none"]:
        params = DiarizationParams(**parse_hint(hint))
        started = time.monotonic()
        result = pipeline(audio_input, **params.pyannote_kwargs())
        elapsed = time.monotonic() - started
        turns = serialize_turns(extract_pyannote_annotation(result))
        exclusive = getattr(result, "exclusive_speaker_diarization", None)
        exclusive_turns = serialize_turns(exclusive) if exclusive is not None else None
        report["runs"][hint] = {
            "pyannote_kwargs": params.pyannote_kwargs(),
            "elapsed_seconds": round(elapsed, 1),
            "labels": sorted({turn["speaker"] for turn in turns}),
            "turns": turns,
            "exclusive_turns": exclusive_turns,
        }
        print(
            f"hint={hint} labels={report['runs'][hint]['labels']} turns={len(turns)} "
            f"elapsed={elapsed:.1f}s"
        )
        for title, rows in (("turns", turns), ("exclusive turns", exclusive_turns)):
            if rows is None:
                continue
            print(f"  {title} starting before {args.list_until:.0f}s:")
            for turn in rows:
                if turn["start"] < args.list_until:
                    duration = turn["end"] - turn["start"]
                    print(
                        f"    {turn['start']:8.3f}-{turn['end']:8.3f} ({duration:5.2f}s) "
                        f"{turn['speaker']}"
                    )
    write_report(report, args.out)


def run_subwindow(args: argparse.Namespace) -> None:
    from transcriptions.known_speaker import (
        SAMPLE_RATE,
        KnownSpeakerConfig,
        ReferenceAudio,
        build_speaker_centroids,
        classify_embedding,
        decode_mono_16k,
        extract_reference_windows,
        get_known_speaker_backend,
        is_confident,
        pad_to_min_duration,
        slice_samples,
    )

    config = KnownSpeakerConfig()
    backend = get_known_speaker_backend(config.embedding_model)

    reference_samples = decode_mono_16k(args.reference_audio)
    references = []
    for name, start, end in args.reference:
        clip = slice_samples(reference_samples, SAMPLE_RATE, start, end)
        windows = extract_reference_windows(clip, SAMPLE_RATE)
        references.append(ReferenceAudio(speaker_id=name.lower(), name=name, windows=windows))
        print(f"reference {name}: {start}-{end}s -> {len(windows)} windows")
    centroids = build_speaker_centroids(references, backend)
    names = sorted(centroids)

    samples = decode_mono_16k(args.audio)
    embed_seconds: list[float] = []

    def classify(start: float, end: float) -> dict[str, Any]:
        window = slice_samples(samples, SAMPLE_RATE, start, end)
        window = pad_to_min_duration(window, SAMPLE_RATE, config.min_segment_duration)
        began = time.monotonic()
        embedding = backend.embed(window, SAMPLE_RATE)
        embed_seconds.append(time.monotonic() - began)
        result = classify_embedding(embedding, centroids)
        return {
            "start": round(start, 2),
            "end": round(end, 2),
            "top": result.top_speaker,
            "margin": round(result.margin, 3),
            "confident": is_confident(result, duration=end - start, config=config),
            "similarity": {c.speaker: round(c.similarity, 3) for c in result.candidates},
        }

    def show(title: str, rows: list[dict[str, Any]]) -> None:
        print(title)
        for row in rows:
            scores = "  ".join(f"{name} {row['similarity'][name]:6.3f}" for name in names)
            verdict = "confident" if row["confident"] else "uncertain"
            print(
                f"  {row['start']:7.2f}-{row['end']:7.2f}  {scores}  "
                f"top {row['top']} margin {row['margin']:6.3f}  {verdict}"
            )

    whole = [classify(start, end) for start, end in args.segment]
    show("whole segments:", whole)
    report: dict[str, Any] = {
        "embedding_model": config.embedding_model,
        "versions": installed_versions(),
        "whole_segments": whole,
        "windows": {},
    }
    for target_start, target_end in args.target:
        for size, hop in args.window:
            rows = []
            cursor = target_start
            while cursor + size <= target_end + 1e-9:
                rows.append(classify(cursor, cursor + size))
                cursor += hop
            key = f"{target_start}-{target_end} {size}s/{hop}s"
            report["windows"][key] = rows
            show(f"windows {key}:", rows)
    if embed_seconds:
        # The first call includes the lazy model load, so keep it out of the median.
        steady = sorted(embed_seconds[1:]) or embed_seconds
        report["embed_cost"] = {
            "count": len(embed_seconds),
            "first_call_seconds": round(embed_seconds[0], 3),
            "median_seconds": round(steady[len(steady) // 2], 4),
        }
        print("embed cost:", report["embed_cost"])
    else:
        print("no window fits the given targets; nothing was embedded")
    write_report(report, args.out)


def run_stt(args: argparse.Namespace) -> None:
    from django.conf import settings

    from transcriptions.sanitizer import sanitize_result
    from transcriptions.service import (
        TranscribeParams,
        build_backend_service,
        resolve_model_name_for_backend,
    )

    report: dict[str, Any] = {
        "audio": args.audio.name,
        "language": args.language,
        "versions": installed_versions(),
        "settings": {name: getattr(settings, name) for name in STT_SETTINGS},
        "runs": {},
    }
    print("settings:", report["settings"])
    for backend_name in args.backend or ["mlx", "whispercpp"]:
        model_name = resolve_model_name_for_backend(request_model="auto", backend_name=backend_name)
        service = build_backend_service(backend_name=backend_name, model_name=model_name)
        started = time.monotonic()
        # The remote worker never forwards a prompt, so the probe does not either.
        result = service.transcribe(
            args.audio,
            TranscribeParams(request_model=model_name, prompt=None, language=args.language),
        )
        elapsed = time.monotonic() - started
        result = sanitize_result(
            result,
            enabled=settings.VOXHELM_SANITIZE_TRANSCRIPT,
            repeat_threshold=settings.VOXHELM_SANITIZE_REPEAT_THRESHOLD,
        )
        segments = [
            {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text}
            for s in result.segments
        ]
        report["runs"][backend_name] = {
            "backend_name": result.backend_name,
            "model_name": result.model_name,
            "elapsed_seconds": round(elapsed, 1),
            "segments": segments,
        }
        print(f"{backend_name}: backend={result.backend_name} model={result.model_name}")
        for segment in segments:
            print(f"  {segment['start']:7.2f}-{segment['end']:7.2f}  {segment['text']}")
    write_report(report, args.out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env-file", type=Path, help="worker env file to load before Django")
    commands = parser.add_subparsers(dest="command", required=True)

    turns = commands.add_parser("turns", help="diarize and keep raw pyannote turns")
    turns.add_argument("audio", type=Path)
    turns.add_argument("--hint", action="append", help="none, num=2, or min=2,max=3; repeatable")
    turns.add_argument("--list-until", type=float, default=60.0, help="print turns before this")
    turns.add_argument("--out", type=Path)
    turns.set_defaults(run=run_turns)

    subwindow = commands.add_parser("subwindow", help="classify sliding windows in a segment")
    subwindow.add_argument("audio", type=Path)
    subwindow.add_argument("reference_audio", type=Path)
    subwindow.add_argument(
        "--reference",
        action="append",
        type=parse_reference,
        required=True,
        help="Name=start-end source range in the reference audio; repeatable",
    )
    subwindow.add_argument(
        "--segment",
        action="append",
        type=parse_range,
        default=[],
        help="start-end transcript segment to classify whole; repeatable",
    )
    subwindow.add_argument(
        "--target",
        action="append",
        type=parse_range,
        required=True,
        help="start-end range to cover with sliding windows; repeatable",
    )
    subwindow.add_argument(
        "--window",
        action="append",
        type=parse_window,
        default=None,
        help="size-hop in seconds, for example 1.5-0.75; repeatable",
    )
    subwindow.add_argument("--out", type=Path)
    subwindow.set_defaults(run=run_subwindow)

    stt = commands.add_parser("stt", help="transcribe with each local STT backend")
    stt.add_argument("audio", type=Path)
    stt.add_argument("--backend", action="append", choices=["mlx", "whispercpp"])
    stt.add_argument("--language", default=None)
    stt.add_argument("--out", type=Path)
    stt.set_defaults(run=run_stt)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "subwindow" and args.window is None:
        args.window = [(1.5, 0.75)]
    setup(args.env_file)
    args.run(args)


if __name__ == "__main__":
    main()
