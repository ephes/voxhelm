"""Named transcription profiles: fixed engine settings a producer selects by name.

A profile pins everything that changes its output, so a producer can cache results per
profile name. Any change to a profile's settings is a new profile name. A profile never
falls back to another engine: when its engine is unavailable the job waits (remote
workers that cannot run it do not claim it) or fails, it never runs something else.

``ad-analysis-v1`` (podcast server-side ad analysis) runs Parakeet TDT through the
``parakeet-mlx`` CLI with 30 s chunks, 5 s overlap, full attention and no
``--max-duration``: the settings of the research runs whose words the ad detector was
tuned on. Besides the usual transcript artifacts it writes a ``words`` artifact with
per-word start, end and mean token confidence, and records the engine identity (model
commit, CLI and MLX versions, profile settings) in the job result and in that artifact.
Its artifacts and job row are removed by the producer (``DELETE /v1/jobs/{id}``) or, at
the latest, by ``prune_job_artifacts`` seven days after the job finished.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from django.conf import settings

from transcriptions.service import (
    PARAKEET_BACKEND_NAME,
    BackendUnavailableError,
    ParakeetBackend,
    TranscribeParams,
    TranscriptionResult,
    normalize_language_code,
    normalize_parakeet_payload,
    parakeet_child_env,
    parakeet_supports_language,
    resolve_parakeet_runtime,
)

LOGGER = logging.getLogger(__name__)

AD_ANALYSIS_V1 = "ad-analysis-v1"
WORDS_ARTIFACT_NAME = "words.json"
WORDS_ARTIFACT_FORMAT = "words"
WORDS_CONTENT_TYPE = "application/json"
WORDS_SCHEMA_VERSION = 1
MODEL_REVISION_FILE = "VOXHELM_MODEL_REVISION"
SEVEN_DAYS_SECONDS = 7 * 24 * 60 * 60
_HEX_COMMIT = re.compile(r"[0-9a-f]{40}")
_VERSION_SCRIPT = (
    "import importlib.metadata as m, json\n"
    "out = {}\n"
    "for name in ('parakeet-mlx', 'mlx'):\n"
    "    try:\n"
    "        out[name] = m.version(name)\n"
    "    except m.PackageNotFoundError:\n"
    "        pass\n"
    "print(json.dumps(out))\n"
)


@dataclass(frozen=True)
class TranscriptionProfile:
    name: str
    backend: str
    chunk_seconds: int
    overlap_seconds: int
    local_attention: bool
    max_sentence_seconds: float | None
    # Every artifact and the job row of a terminal job are deleted this long after the
    # job finished, whether or not the producer deleted the job itself.
    artifact_retention_seconds: int

    def settings_record(self) -> dict[str, Any]:
        return {
            "chunk_seconds": self.chunk_seconds,
            "overlap_seconds": self.overlap_seconds,
            "local_attention": self.local_attention,
            "max_sentence_seconds": self.max_sentence_seconds,
            "decoding": "greedy",
            "precision": "bf16",
        }


PROFILES: dict[str, TranscriptionProfile] = {
    AD_ANALYSIS_V1: TranscriptionProfile(
        name=AD_ANALYSIS_V1,
        backend=PARAKEET_BACKEND_NAME,
        chunk_seconds=30,
        overlap_seconds=5,
        local_attention=False,
        max_sentence_seconds=None,
        artifact_retention_seconds=SEVEN_DAYS_SECONDS,
    ),
}


class ProfileError(ValueError):
    """A request names an unknown profile or combines it with an unsupported option."""


def get_transcription_profile(name: str) -> TranscriptionProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        accepted = ", ".join(sorted(PROFILES))
        raise ProfileError(f"Unsupported profile '{name}'. Accepted values: {accepted}.") from exc


def normalize_language_hint(language_hint: str | None) -> str | None:
    """The hint as a bare language code; raises ``ProfileError`` outside Parakeet's languages."""
    normalized = normalize_language_code(language_hint)
    if not normalized:
        return None
    if not parakeet_supports_language(normalized):
        raise ProfileError(f"language_hint '{language_hint}' is not supported by the profile.")
    return normalized


@dataclass(frozen=True)
class ProfileTranscription:
    result: TranscriptionResult
    words: list[dict[str, Any]]
    identity: dict[str, Any]

    def words_document(self) -> dict[str, Any]:
        return {
            "schema_version": WORDS_SCHEMA_VERSION,
            "profile": self.identity["profile"],
            "identity": self.identity,
            "words": self.words,
        }


def profile_runtime_available(profile: TranscriptionProfile) -> bool:
    """True when this host can run the profile (its CLI and model directory resolve)."""
    try:
        resolve_parakeet_runtime(
            binary_path=settings.VOXHELM_PARAKEET_BIN,
            model_dir=settings.VOXHELM_PARAKEET_MODEL_DIR,
        )
    except BackendUnavailableError:
        return False
    return profile.backend == PARAKEET_BACKEND_NAME


def available_profile_names() -> list[str]:
    return sorted(name for name, profile in PROFILES.items() if profile_runtime_available(profile))


def run_profile_transcription(
    profile: TranscriptionProfile,
    audio_path: Path,
    *,
    language_hint: str | None,
    cancel_event: threading.Event | None = None,
) -> ProfileTranscription:
    """Transcribe with exactly the profile's engine and settings; never another engine.

    Raises ``BackendUnavailableError`` when the CLI or model is missing on this host. The
    caller admits the run through the lane scheduler. The transcript is not sanitized: the
    words artifact and the transcript artifacts are the engine's output as it is.
    """
    model_name = settings.VOXHELM_PARAKEET_MODEL_NAME
    backend = ParakeetBackend(
        binary_path=settings.VOXHELM_PARAKEET_BIN,
        model_dir=settings.VOXHELM_PARAKEET_MODEL_DIR,
        model_name=model_name,
        chunk_seconds=profile.chunk_seconds,
        overlap_seconds=profile.overlap_seconds,
        max_sentence_seconds=profile.max_sentence_seconds,
        timeout_seconds=settings.VOXHELM_PARAKEET_TIMEOUT_SECONDS,
        local_attention=profile.local_attention,
    )
    executable, model_dir = resolve_parakeet_runtime(
        binary_path=settings.VOXHELM_PARAKEET_BIN,
        model_dir=settings.VOXHELM_PARAKEET_MODEL_DIR,
    )
    payload = backend.run(
        audio_path,
        TranscribeParams(
            request_model=model_name,
            prompt=None,
            language=language_hint,
            word_timestamps=True,
            cancel_event=cancel_event,
        ),
    )
    result = normalize_parakeet_payload(payload, model_name=model_name)
    words = parakeet_payload_words(payload)
    identity = profile_identity(
        profile,
        executable=executable,
        model_dir=model_dir,
        model_name=model_name,
        language_hint=language_hint,
    )
    return ProfileTranscription(result=result, words=words, identity=identity)


def parakeet_payload_words(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Words ``{w, s, e, c}`` from all tokens of the CLI output, in order.

    Same rule as the research runner: a token whose text starts with a space starts a
    new word, any other token is appended to the current word and moves its end; ``c``
    is the mean confidence of the word's tokens, times and confidence rounded to 3
    decimals. Tokens are taken across sentence boundaries, as from the result's token
    list, so a sentence split never splits a word. Malformed output fails the run.
    """
    raw_sentences = payload.get("sentences")
    if not isinstance(raw_sentences, list):
        raise RuntimeError("Parakeet output has no sentence list.")
    words: list[dict[str, Any]] = []
    confidences: list[list[float]] = []
    for sentence in raw_sentences:
        tokens = sentence.get("tokens") if isinstance(sentence, dict) else None
        if not isinstance(tokens, list):
            raise RuntimeError("Parakeet output has a sentence without tokens.")
        for token in tokens:
            text, start, end, confidence = _token_fields(token)
            if not words or text.startswith(" "):
                words.append({"w": text.strip(), "s": start, "e": end})
                confidences.append([confidence])
            else:
                words[-1]["w"] += text
                words[-1]["e"] = end
                confidences[-1].append(confidence)
    for word, values in zip(words, confidences, strict=True):
        word["s"] = round(word["s"], 3)
        word["e"] = round(word["e"], 3)
        word["c"] = round(sum(values) / len(values), 3)
    return words


def _token_fields(token: object) -> tuple[str, float, float, float]:
    if not isinstance(token, dict):
        raise RuntimeError("Parakeet output has a malformed token.")
    text = token.get("text")
    if not isinstance(text, str):
        raise RuntimeError("Parakeet output has a token without text.")
    numbers: list[float] = []
    for key in ("start", "end", "confidence"):
        value = token.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise RuntimeError(f"Parakeet output has a token without {key}.")
        numbers.append(float(value))
    return text, numbers[0], numbers[1], numbers[2]


def profile_identity(
    profile: TranscriptionProfile,
    *,
    executable: str,
    model_dir: Path,
    model_name: str,
    language_hint: str | None,
) -> dict[str, Any]:
    versions = cli_versions(executable)
    return {
        "profile": profile.name,
        "backend": profile.backend,
        "model": model_name,
        **model_revision(model_dir),
        "cli": "parakeet-mlx",
        "cli_version": versions.get("parakeet-mlx", ""),
        "mlx_version": versions.get("mlx", ""),
        "settings": profile.settings_record(),
        "language_hint": language_hint,
    }


def model_revision(model_dir: Path) -> dict[str, str]:
    """The pinned model's repo, commit and weights checksum.

    Read from the ``VOXHELM_MODEL_REVISION`` marker the deployment writes next to the
    weights (``repo=``, ``revision=``, ``model.safetensors sha256=`` lines); for a
    Hugging Face cache snapshot directory the commit is its directory name. Unknown
    fields are empty.
    """
    record = {"model_repo": "", "model_revision": "", "model_sha256": ""}
    try:
        marker = (model_dir / MODEL_REVISION_FILE).read_text(encoding="utf-8")
    except OSError:
        marker = ""
    for line in marker.splitlines():
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key == "repo":
            record["model_repo"] = value
        elif key == "revision":
            record["model_revision"] = value
        elif key == "model.safetensors sha256":
            record["model_sha256"] = value
    if not record["model_revision"] and model_dir.parent.name == "snapshots":
        if _HEX_COMMIT.fullmatch(model_dir.name):
            record["model_revision"] = model_dir.name
            repo_dir = model_dir.parent.parent.name
            if repo_dir.startswith("models--"):
                record["model_repo"] = repo_dir.removeprefix("models--").replace("--", "/")
    return record


def cli_versions(executable: str) -> dict[str, str]:
    path = Path(executable)
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        mtime_ns = 0
    return dict(_cli_versions(str(path), mtime_ns))


@lru_cache(maxsize=8)
def _cli_versions(executable: str, mtime_ns: int) -> tuple[tuple[str, str], ...]:
    """``parakeet-mlx`` and ``mlx`` versions of the CLI's own environment, cached per file.

    The CLI is a script whose interpreter lives in its tool environment, so that
    interpreter reports both package versions; ``--version`` is the fallback for the CLI.
    """
    del mtime_ns
    versions: dict[str, str] = {}
    interpreter = _script_interpreter(Path(executable))
    if interpreter is not None:
        output = _run_quietly([interpreter, "-I", "-c", _VERSION_SCRIPT])
        try:
            parsed = json.loads(output) if output else {}
        except json.JSONDecodeError:
            parsed = {}
        if isinstance(parsed, dict):
            versions.update(
                {str(key): str(value) for key, value in parsed.items() if isinstance(value, str)}
            )
    if "parakeet-mlx" not in versions:
        match = re.search(r"parakeet-mlx\s+(\S+)", _run_quietly([executable, "--version"]))
        if match:
            versions["parakeet-mlx"] = match.group(1)
    LOGGER.info("parakeet cli versions=%s executable=%s", versions, executable)
    return tuple(sorted(versions.items()))


def _script_interpreter(executable: Path) -> str | None:
    try:
        with executable.open("rb") as handle:
            first_line = handle.readline(4096)
    except OSError:
        return None
    if not first_line.startswith(b"#!"):
        return None
    parts = first_line[2:].decode("utf-8", errors="replace").split()
    if len(parts) != 1 or "python" not in Path(parts[0]).name:
        return None
    return parts[0] if Path(parts[0]).is_file() else None


def _run_quietly(args: list[str]) -> str:
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=60,
            env=parakeet_child_env(),
        )
    except OSError, subprocess.TimeoutExpired:
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""
