"""Owned decoder execution with bounded diagnostics and finite retirement."""

import subprocess

from transcriptions.owned_process import (
    PROCESS_OWNERSHIP_UNCERTAIN,
    OwnedProcessMessages,
    process_ownership_uncertain,
    run_owned_process,
)


class AudioDecodeError(RuntimeError):
    """A decoder failed or exceeded its diagnostic allowance."""


def run_audio_decode(argv: list[str], *, timeout: float) -> None:
    """Retire the decoder group before returning, preserving safe failure categories."""
    try:
        run_owned_process(
            argv,
            timeout=timeout,
            messages=OwnedProcessMessages(
                start="Audio decode could not be started.",
                deadline="Audio decode exceeded the configured decode deadline.",
                diagnostics="Audio decode exceeded the diagnostic byte limit.",
                failure="Audio decode failed.",
            ),
        )
    except RuntimeError as exc:
        if isinstance(exc.__cause__, subprocess.TimeoutExpired):
            error = subprocess.TimeoutExpired(argv, timeout)
            if process_ownership_uncertain(exc):
                error.add_note(PROCESS_OWNERSHIP_UNCERTAIN)
            raise error from exc
        raise AudioDecodeError(str(exc)) from exc
