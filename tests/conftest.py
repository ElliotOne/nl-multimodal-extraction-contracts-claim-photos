from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from claim_photo_extraction.claims import load_claim
from claim_photo_extraction.config import AppConfig, load_config
from claim_photo_extraction.models import Claim
from claim_photo_extraction.samples import write_sample_set

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def sample_photos(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return write_sample_set(tmp_path_factory.mktemp("photos"))


@pytest.fixture(scope="session")
def dry_config() -> AppConfig:
    return load_config(ROOT / "config" / "dry-run.json")


@pytest.fixture()
def sample_claim(sample_photos: dict[str, Path], tmp_path: Path) -> Claim:
    source = json.loads((ROOT / "data" / "claims" / "CLM-2026-004471" / "claim.json").read_text(encoding="utf-8"))
    for photo in source["photos"]:
        photo["file"] = sample_photos[photo["photo_id"]].as_posix()
    claim_path = tmp_path / "claim.json"
    claim_path.write_text(json.dumps(source), encoding="utf-8")
    return load_claim(claim_path)


@pytest.fixture()
def now_utc() -> datetime:
    return datetime(2026, 11, 12, 9, 0, tzinfo=UTC)
