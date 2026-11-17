"""Source-preserving report output for the standalone speaker evaluation CLIs."""

import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def admit_report_output(out: Path | None, sources: Sequence[Path]) -> None:
    if out is None:
        return
    try:
        resolved = out.resolve()
        for source in sources:
            if resolved == source.resolve() or (
                out.exists() and source.exists() and out.samefile(source)
            ):
                raise ValueError("Speaker report output aliases an input.")
    except (OSError, RuntimeError) as exc:
        raise ValueError("Invalid speaker report output.") from exc


def publish_report(
    report: dict[str, Any], out: Path, sources: Sequence[Path], *, indent: int = 2
) -> None:
    admit_report_output(out, sources)
    serialized = json.dumps(report, indent=indent, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=out.parent,
            prefix=f".{out.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(serialized)
        # Inference and staging may take time: repeat source-identity admission
        # directly before replacing the final name, after the owned file closes.
        admit_report_output(out, sources)
        os.replace(temporary, out)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
