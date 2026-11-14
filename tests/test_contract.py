from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from claim_photo_extraction.config import AppConfig
from claim_photo_extraction.contract import ObservationContractGate, colors_compatible, normalise_plate, validate_against_schema
from claim_photo_extraction.models import Claim
from claim_photo_extraction.schema import response_json_schema

ROOT = Path(__file__).resolve().parents[1]
SENT = ["P1", "P4", "P5"]


def _replay_output() -> dict[str, Any]:
    body = json.loads((ROOT / "data" / "replay" / "CLM-2026-004471.response.json").read_text(encoding="utf-8"))
    return json.loads(body["choices"][0]["message"]["content"])


def _walk(schema: dict[str, Any]) -> list[dict[str, Any]]:
    found = [schema] if schema.get("type") == "object" else []
    for child in list(schema.get("properties", {}).values()) + ([schema["items"]] if "items" in schema else []):
        found.extend(_walk(child))
    return found


def test_schema_obeys_strict_structured_output_rules() -> None:
    for obj in _walk(response_json_schema()):
        assert obj["additionalProperties"] is False
        assert sorted(obj["required"]) == sorted(obj["properties"].keys())
    text = json.dumps(response_json_schema())
    for banned in ("minimum", "maximum", "pattern", "minItems", "maxItems", "format", "minLength"):
        assert f'"{banned}"' not in text


def test_replay_output_validates_against_schema() -> None:
    assert validate_against_schema(_replay_output(), response_json_schema()) == []


def test_schema_validator_reports_additional_and_missing_properties() -> None:
    output = _replay_output()
    output["photos"][0]["repair_cost"] = 1200
    del output["photos"][1]["plate_text"]
    errors = validate_against_schema(output, response_json_schema())
    assert any("repair_cost" in error for error in errors)
    assert any("plate_text" in error for error in errors)


def test_contract_result_for_replay_fixture(dry_config: AppConfig, sample_claim: Claim) -> None:
    result = ObservationContractGate(dry_config.contract).evaluate(_replay_output(), sample_claim, SENT)
    assert result.valid_schema
    assert [item["part"] for item in result.confirmed_damages] == ["front_fender_left", "door_front_left", "front_bumper", "front_bumper", "front_fender_left", "front_fender_left", "door_front_left"]
    assert [item["part"] for item in result.unconfirmed_damages] == ["hood"]
    assert result.observed_zones == ("front", "left")
    assert result.uncovered_reported_zones == ()
    assert result.unreported_observed_zones == ()
    assert result.identity_flags == ()
    codes = {finding.code for finding in result.findings}
    assert {"plate_verified", "model_reported_quality_issue"} <= codes
    assert "unreported_zone_damaged" not in codes


def test_photo_id_mismatch_blocks(dry_config: AppConfig, sample_claim: Claim) -> None:
    output = _replay_output()
    output["photos"][2]["photo_id"] = "P9"
    result = ObservationContractGate(dry_config.contract).evaluate(output, sample_claim, SENT)
    assert any(f.code == "photo_id_set_mismatch" and f.severity == "block" for f in result.findings)


def test_plate_mismatch_is_identity_flag(dry_config: AppConfig, sample_claim: Claim) -> None:
    output = _replay_output()
    output["photos"][0]["plate_text"] = "AB12 CDE"
    result = ObservationContractGate(dry_config.contract).evaluate(output, sample_claim, SENT)
    assert "plate_mismatch" in result.identity_flags


def test_color_mismatch_is_identity_flag(dry_config: AppConfig, sample_claim: Claim) -> None:
    output = _replay_output()
    output["photos"][1]["body_color"] = "red"
    result = ObservationContractGate(dry_config.contract).evaluate(output, sample_claim, SENT)
    assert "color_mismatch" in result.identity_flags


def test_confidence_out_of_range_is_blocked_by_code_not_schema(dry_config: AppConfig, sample_claim: Claim) -> None:
    output = _replay_output()
    output["photos"][0]["damages"][0]["confidence"] = 1.7
    assert validate_against_schema(output, response_json_schema()) == []
    result = ObservationContractGate(dry_config.contract).evaluate(output, sample_claim, SENT)
    assert any(f.code == "confidence_out_of_range" for f in result.findings)


def test_uncovered_reported_zone_is_reported(dry_config: AppConfig, sample_claim: Claim) -> None:
    output = copy.deepcopy(_replay_output())
    left_parts = {"front_fender_left", "door_front_left", "mirror_left"}
    for photo in output["photos"]:
        photo["damages"] = [damage for damage in photo["damages"] if damage["part"] not in left_parts]
    result = ObservationContractGate(dry_config.contract).evaluate(output, sample_claim, SENT)
    assert result.uncovered_reported_zones == ("left",)


def test_plate_and_color_helpers() -> None:
    assert normalise_plate("ln68 kxr") == "LN68KXR"
    assert normalise_plate("LN68-KXR") == "LN68KXR"
    assert colors_compatible("silver", "light grey")
    assert colors_compatible("Silver", "metallic silver")
    assert not colors_compatible("silver", "red")
