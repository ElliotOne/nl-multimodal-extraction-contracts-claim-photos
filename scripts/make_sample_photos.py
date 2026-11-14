"""Regenerate the synthetic sample photos under data/claims/CLM-2026-004471/photos."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from claim_photo_extraction.samples import write_sample_set  # noqa: E402


def main() -> int:
    target = ROOT / "data" / "claims" / "CLM-2026-004471" / "photos"
    paths = write_sample_set(target)
    for photo_id, path in paths.items():
        print(f"{photo_id}: {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
