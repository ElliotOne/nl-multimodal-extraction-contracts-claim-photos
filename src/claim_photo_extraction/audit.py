from __future__ import annotations

import json
from pathlib import Path

from .models import RunReport


def write_report(report: RunReport, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = report.run_at_utc.strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"claim-photo-gate-{report.claim_id}-{stamp}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2)
        handle.write("\n")
    return path
