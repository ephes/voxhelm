"""Source-preserving report output for the standalone speaker evaluation CLIs."""

import json
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
    out.write_text(json.dumps(report, indent=indent, ensure_ascii=False) + "\n", encoding="utf-8")
