"""Request fingerprints backing the batch-job ``task_ref`` idempotency constraint.

The fingerprint covers exactly the fields that ``existing_job_matches_request`` compares, so two
submissions that would deduplicate to the same job always share a fingerprint, and the partial
unique constraint on ``(producer, task_ref, request_fingerprint)`` for non-failed jobs closes the
race between concurrent identical submissions.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

TRANSCRIBE_JOB_TYPE = "transcribe"


def job_request_fingerprint_from_fields(
    *,
    job_type: str,
    backend: str,
    model: str,
    language: str | None,
    input_data: dict[str, Any],
    output_formats: list[str],
    diarization: dict[str, Any] | None,
) -> str:
    if job_type != TRANSCRIBE_JOB_TYPE:
        # Synthesis keeps the original looser task_ref behaviour: any non-failed job of the same
        # type with the same task_ref is reused, regardless of the rest of the payload.
        material: dict[str, Any] = {"job_type": job_type}
    else:
        material = {
            "job_type": job_type,
            "backend": backend,
            "model": model,
            "language": language or None,
            "input": normalized_transcription_input(input_data),
            "output_formats": sorted(output_formats),
            "diarization": diarization if diarization is not None else {"enabled": False},
        }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def normalized_transcription_input(input_data: dict[str, Any]) -> dict[str, Any]:
    if input_data.get("kind") == "upload":
        # Persisted upload inputs carry staging metadata; only the upload id is request identity.
        return {"kind": "upload", "upload_id": str(input_data.get("upload_id") or "")}
    return input_data


def stored_job_request_fingerprint(
    *,
    job_type: str,
    backend: str,
    model: str,
    language: str,
    input_data: dict[str, Any],
    output_data: dict[str, Any],
) -> str:
    return job_request_fingerprint_from_fields(
        job_type=job_type,
        backend=backend,
        model=model,
        language=language or None,
        input_data=input_data or {},
        output_formats=list((output_data or {}).get("formats", [])),
        diarization=(output_data or {}).get("diarization", {"enabled": False}),
    )
