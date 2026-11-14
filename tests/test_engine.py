from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from claim_photo_extraction.audit import write_report
from claim_photo_extraction.config import AppConfig
from claim_photo_extraction.engine import ClaimPhotoEngine
from claim_photo_extraction.foundry import ReplayVisionClient
from claim_photo_extraction.models import AdmissionDecision, Claim, RoutingDecision

ROOT = Path(__file__).resolve().parents[1]
REPLAY = ROOT / "data" / "replay" / "CLM-2026-004471.response.json"


def test_dry_run_stops_before_the_model(dry_config: AppConfig, sample_claim: Claim, now_utc: datetime) -> None:
    report = ClaimPhotoEngine(dry_config, None).run(sample_claim, now_utc=now_utc)
    assert report.model_call.called is False
    assert report.budget.included_photo_ids == ("P1", "P4", "P5")
    assert report.contract is None
    assert report.routing.decision is RoutingDecision.ADJUSTER_REVIEW


def test_replay_run_reaches_contract_and_routing(dry_config: AppConfig, sample_claim: Claim, now_utc: datetime, tmp_path: Path) -> None:
    engine = ClaimPhotoEngine(dry_config, ReplayVisionClient(REPLAY))
    report = engine.run(sample_claim, now_utc=now_utc, sent_copies_dir=tmp_path / "sent")

    decisions = {record.photo_id: record.decision for record in report.admission}
    assert decisions == {
        "P1": AdmissionDecision.ADMIT,
        "P2": AdmissionDecision.REJECT,
        "P3": AdmissionDecision.REJECT,
        "P4": AdmissionDecision.ADMIT_FLAGGED,
        "P5": AdmissionDecision.ADMIT_FLAGGED,
        "P6": AdmissionDecision.REJECT,
    }
    assert report.model_call.called is True
    assert report.model_call.source.startswith("replay:")
    assert report.contract is not None and report.contract.valid_schema
    assert report.routing.decision is RoutingDecision.FAST_TRACK
    assert report.routing.reasons == ("confirmed_damage_in_reported_zones_only",)
    assert sorted(path.name for path in (tmp_path / "sent").iterdir()) == ["P1.sent.jpg", "P4.sent.jpg", "P5.sent.jpg"]


def test_report_is_json_serialisable_without_image_bytes(dry_config: AppConfig, sample_claim: Claim, now_utc: datetime, tmp_path: Path) -> None:
    report = ClaimPhotoEngine(dry_config, ReplayVisionClient(REPLAY)).run(sample_claim, now_utc=now_utc)
    path = write_report(report, tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["claim_id"] == "CLM-2026-004471"
    assert data["routing"]["decision"] == "fast_track"
    assert "jpeg_base64" not in path.read_text(encoding="utf-8")
    assert data["admission"][0]["facts"]["dhash_hex"]
