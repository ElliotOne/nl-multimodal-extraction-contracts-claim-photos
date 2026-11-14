from __future__ import annotations

from datetime import datetime
from pathlib import Path

from claim_photo_extraction.config import AppConfig
from claim_photo_extraction.engine import ClaimPhotoEngine
from claim_photo_extraction.foundry import ReplayVisionClient
from claim_photo_extraction.html_report import render_html, write_html_report
from claim_photo_extraction.models import Claim

ROOT = Path(__file__).resolve().parents[1]
REPLAY = ROOT / "data" / "replay" / "CLM-2026-004471.response.json"


def test_html_report_contains_every_stage(dry_config: AppConfig, sample_claim: Claim, now_utc: datetime, tmp_path: Path) -> None:
    report = ClaimPhotoEngine(dry_config, ReplayVisionClient(REPLAY)).run(sample_claim, now_utc=now_utc)
    page = render_html(report, sample_claim)
    assert "Fast track" in page
    assert "confirmed_damage_in_reported_zones_only" in page
    for photo_id in ("P1", "P2", "P3", "P4", "P5", "P6"):
        assert f"<h3>{photo_id} " in page
    assert "same image as P1" in page
    assert "data:image/jpeg;base64," in page
    assert "jpeg_base64" not in page


def test_html_report_without_model(dry_config: AppConfig, sample_claim: Claim, now_utc: datetime, tmp_path: Path) -> None:
    report = ClaimPhotoEngine(dry_config, None).run(sample_claim, now_utc=now_utc)
    target = write_html_report(report, sample_claim, tmp_path / "r.json")
    assert target.name == "r.html"
    text = target.read_text(encoding="utf-8")
    assert "Not called" in text and "Not run because the model was not called" in text
