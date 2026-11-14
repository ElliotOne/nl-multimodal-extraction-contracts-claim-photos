from __future__ import annotations

import copy
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from claim_photo_extraction.config import AppConfig
from claim_photo_extraction.consensus import aggregate_samples
from claim_photo_extraction.contract import ObservationContractGate, validate_against_schema
from claim_photo_extraction.engine import ClaimPhotoEngine
from claim_photo_extraction.models import Claim, ModelCall, RoutingDecision
from claim_photo_extraction.schema import response_json_schema

ROOT = Path(__file__).resolve().parents[1]
REPLAY = ROOT / "data" / "replay" / "CLM-2026-004471.response.json"


def _base() -> dict:
    body = json.loads(REPLAY.read_text(encoding="utf-8"))
    return json.loads(body["choices"][0]["message"]["content"])


def _photo(output: dict, photo_id: str) -> dict:
    return next(p for p in output["photos"] if p["photo_id"] == photo_id)


def test_majority_keeps_agreed_parts_and_drops_one_off_findings() -> None:
    a, b, c = _base(), _base(), _base()
    # One sample adds a right-wing scratch nobody else sees; one sample drops the door.
    _photo(b, "P4")["damages"].append({"part": "front_fender_right", "damage_type": "scratch", "severity": "minor", "confidence": 0.9})
    _photo(c, "P1")["damages"] = [d for d in _photo(c, "P1")["damages"] if d["part"] != "door_front_left"]

    output, agreement = aggregate_samples([a, b, c], min_agreement=2)
    p4_parts = {d["part"] for d in _photo(output, "P4")["damages"]}
    p1_parts = {d["part"] for d in _photo(output, "P1")["damages"]}
    assert "front_fender_right" not in p4_parts
    assert "door_front_left" in p1_parts
    assert agreement["P4"]["front_fender_right"] == 1
    assert agreement["P1"]["door_front_left"] == 2
    assert validate_against_schema(output, response_json_schema()) == []


def test_severity_votes_resolve_ties_upwards_and_confidence_is_averaged() -> None:
    a, b = _base(), _base()
    wing_a = next(d for d in _photo(a, "P1")["damages"] if d["part"] == "front_fender_left")
    wing_b = next(d for d in _photo(b, "P1")["damages"] if d["part"] == "front_fender_left")
    wing_a["severity"], wing_a["confidence"] = "moderate", 0.9
    wing_b["severity"], wing_b["confidence"] = "severe", 1.0
    a["overall_severity"], b["overall_severity"] = "moderate", "severe"

    output, _ = aggregate_samples([a, b], min_agreement=2)
    wing = next(d for d in _photo(output, "P1")["damages"] if d["part"] == "front_fender_left")
    assert wing["severity"] == "severe"
    assert wing["confidence"] == 0.95
    assert output["overall_severity"] == "severe"


def test_three_way_severity_split_takes_the_median_not_the_maximum() -> None:
    a, b, c = _base(), _base(), _base()
    for sample, severity in ((a, "minor"), (b, "moderate"), (c, "severe")):
        bumper = next(d for d in _photo(sample, "P4")["damages"] if d["part"] == "front_bumper")
        bumper["severity"] = severity
    output, _ = aggregate_samples([a, b, c], min_agreement=2)
    bumper = next(d for d in _photo(output, "P4")["damages"] if d["part"] == "front_bumper")
    assert bumper["severity"] == "moderate"

    d, e, f = _base(), _base(), _base()
    for sample, severity in ((d, "moderate"), (e, "severe"), (f, "severe")):
        sample["overall_severity"] = severity
    output, _ = aggregate_samples([d, e, f], min_agreement=2)
    assert output["overall_severity"] == "severe"


def test_plate_needs_agreement_and_vehicle_consistency_needs_all() -> None:
    a, b, c = _base(), _base(), _base()
    _photo(b, "P1")["plate_text"] = "AB12 CDE"
    c["same_vehicle_across_photos"] = False
    output, _ = aggregate_samples([a, b, c], min_agreement=2)
    assert _photo(output, "P1")["plate_text"] == "LM64 LVG"
    assert output["same_vehicle_across_photos"] is False


def test_plate_vote_counts_agreement_not_just_legibility() -> None:
    a, b, c = _base(), _base(), _base()
    # Two samples read different registrations, the third cannot read the plate.
    _photo(a, "P1")["plate_text"] = "LM64 LVG"
    _photo(b, "P1")["plate_text"] = "AB12 CDE"
    _photo(c, "P1")["plate_text"], _photo(c, "P1")["plate_legible"] = None, False
    output, _ = aggregate_samples([a, b, c], min_agreement=2)
    assert _photo(output, "P1")["plate_text"] is None
    assert _photo(output, "P1")["plate_legible"] is False

    # Spacing differences are the same registration.
    d, e, f = _base(), _base(), _base()
    _photo(d, "P1")["plate_text"] = "LM64 LVG"
    _photo(e, "P1")["plate_text"] = "LM64LVG"
    _photo(f, "P1")["plate_text"] = "AB12CDE"
    output, _ = aggregate_samples([d, e, f], min_agreement=2)
    assert _photo(output, "P1")["plate_text"] == "LM64 LVG"


def test_damage_type_follows_the_voted_severity() -> None:
    a, b, c = _base(), _base(), _base()
    # Mirrors a live run: "scratch, minor" and "deformation, moderate", third sample silent.
    bumper_a = next(d for d in _photo(a, "P4")["damages"] if d["part"] == "front_bumper")
    bumper_b = next(d for d in _photo(b, "P4")["damages"] if d["part"] == "front_bumper")
    bumper_a.update(damage_type="scratch", severity="minor")
    bumper_b.update(damage_type="deformation", severity="moderate")
    _photo(c, "P4")["damages"] = [d for d in _photo(c, "P4")["damages"] if d["part"] != "front_bumper"]
    output, _ = aggregate_samples([a, b, c], min_agreement=2)
    bumper = next(d for d in _photo(output, "P4")["damages"] if d["part"] == "front_bumper")
    assert (bumper["damage_type"], bumper["severity"]) == ("deformation", "moderate")

    # Same severity, tied types: the more serious type wins, as severity ties round up.
    d, e, f = _base(), _base(), _base()
    for sample, damage_type in ((d, "scratch"), (e, "deformation")):
        next(x for x in _photo(sample, "P4")["damages"] if x["part"] == "front_bumper").update(damage_type=damage_type, severity="moderate")
    _photo(f, "P4")["damages"] = [x for x in _photo(f, "P4")["damages"] if x["part"] != "front_bumper"]
    output, _ = aggregate_samples([d, e, f], min_agreement=2)
    bumper = next(x for x in _photo(output, "P4")["damages"] if x["part"] == "front_bumper")
    assert bumper["damage_type"] == "deformation"


def test_quality_issues_are_voted_not_unioned() -> None:
    a, b, c = _base(), _base(), _base()
    for sample in (a, b, c):
        _photo(sample, "P4")["quality_issues"] = ["none"]
        _photo(sample, "P1")["quality_issues"] = ["none"]
    _photo(a, "P4")["quality_issues"] = ["partial_view"]
    _photo(a, "P1")["quality_issues"] = ["glare"]
    _photo(b, "P1")["quality_issues"] = ["glare"]
    output, _ = aggregate_samples([a, b, c], min_agreement=2)
    assert _photo(output, "P4")["quality_issues"] == ["none"]
    assert _photo(output, "P1")["quality_issues"] == ["glare"]


def test_single_sample_passes_through_unchanged() -> None:
    a = _base()
    output, agreement = aggregate_samples([a], min_agreement=1)
    assert output == a
    assert agreement["P1"]["front_fender_left"] == 1


def test_uncorroborated_minor_unreported_finding_warns_instead_of_escalating(dry_config: AppConfig, sample_claim: Claim) -> None:
    output = _base()
    _photo(output, "P4")["damages"].append({"part": "front_fender_right", "damage_type": "scratch", "severity": "minor", "confidence": 0.9})
    result = ObservationContractGate(dry_config.contract).evaluate(output, sample_claim, ["P1", "P4", "P5"])
    codes = [f.code for f in result.findings]
    assert "possible_unreported_damage" in codes
    assert "unreported_zone_damaged" not in codes
    assert result.unreported_observed_zones == ()


def test_corroborated_unreported_finding_escalates(dry_config: AppConfig, sample_claim: Claim) -> None:
    output = _base()
    _photo(output, "P4")["damages"].append({"part": "front_fender_right", "damage_type": "dent", "severity": "moderate", "confidence": 0.85})
    result = ObservationContractGate(dry_config.contract).evaluate(output, sample_claim, ["P1", "P4", "P5"])
    assert "unreported_zone_damaged" in [f.code for f in result.findings]

    repeated = _base()
    for photo_id in ("P1", "P4"):
        _photo(repeated, photo_id)["damages"].append({"part": "front_fender_right", "damage_type": "scratch", "severity": "minor", "confidence": 0.7})
    result = ObservationContractGate(dry_config.contract).evaluate(repeated, sample_claim, ["P1", "P4", "P5"])
    assert result.unreported_observed_zones == ("right",)


class _VaryingClient:
    def __init__(self, outputs: list[dict], errors_at: set[int] = frozenset()) -> None:
        self._outputs = outputs
        self._errors_at = errors_at
        self.calls = 0

    def observe(self, claim, images) -> ModelCall:
        index = self.calls
        self.calls += 1
        if index in self._errors_at:
            return ModelCall(True, True, "fake", "fake-model", None, None, None, {"prompt_tokens": 100, "completion_tokens": 0, "total_tokens": 100}, "HTTP 429")
        return ModelCall(
            True, True, "fake", "fake-model", "{}", copy.deepcopy(self._outputs[index % len(self._outputs)]), None,
            {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "completion_tokens_details": {"reasoning_tokens": 10}}, None,
        )


def test_engine_requests_configured_samples_and_sums_usage(dry_config: AppConfig, sample_claim: Claim, now_utc: datetime) -> None:
    a, b = _base(), _base()
    _photo(b, "P4")["damages"].append({"part": "front_fender_right", "damage_type": "scratch", "severity": "minor", "confidence": 0.9})
    config = replace(dry_config, foundry=replace(dry_config.foundry, samples=3), contract=replace(dry_config.contract, min_sample_agreement=2))
    client = _VaryingClient([a, b, a])
    report = ClaimPhotoEngine(config, client).run(sample_claim, now_utc=now_utc)

    assert client.calls == 3
    assert report.model_call.sample_count == 3
    assert report.model_call.usage == {"samples": 3, "prompt_tokens": 300, "completion_tokens": 150, "total_tokens": 450, "completion_tokens_details": {"reasoning_tokens": 30}}
    assert "front_fender_right" not in {d["part"] for d in _photo(report.model_call.parsed_output, "P4")["damages"]}
    assert report.routing.decision is RoutingDecision.FAST_TRACK


def test_engine_tolerates_a_failed_sample_but_not_all_failed(dry_config: AppConfig, sample_claim: Claim, now_utc: datetime) -> None:
    config = replace(dry_config, foundry=replace(dry_config.foundry, samples=3), contract=replace(dry_config.contract, min_sample_agreement=2))
    partial = ClaimPhotoEngine(config, _VaryingClient([_base()], errors_at={1})).run(sample_claim, now_utc=now_utc)
    assert partial.model_call.parsed_output is not None
    assert partial.model_call.error is not None and "1 of 3 samples failed" in partial.model_call.error
    assert partial.routing.decision is RoutingDecision.FAST_TRACK

    total = ClaimPhotoEngine(config, _VaryingClient([_base()], errors_at={0, 1, 2})).run(sample_claim, now_utc=now_utc)
    assert total.model_call.parsed_output is None
    assert total.routing.decision is RoutingDecision.ADJUSTER_REVIEW
    assert total.routing.reasons[0].startswith("model_unavailable")
