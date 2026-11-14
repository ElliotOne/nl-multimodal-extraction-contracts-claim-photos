from __future__ import annotations

import re
from typing import Any

from .config import ContractConfig
from .models import Claim, ContractFinding, ContractResult
from .schema import PART_ZONES, response_json_schema

SEVERITY_RANK = {"minor": 0, "moderate": 1, "severe": 2}

COLOR_SYNONYMS: dict[str, set[str]] = {
    "silver": {"silver", "grey", "gray", "light grey", "light gray", "metallic grey", "metallic silver"},
    "grey": {"grey", "gray", "silver", "dark grey", "dark gray", "charcoal", "graphite"},
    "white": {"white", "pearl", "pearl white", "cream", "ivory"},
    "black": {"black", "dark grey", "charcoal", "graphite"},
    "blue": {"blue", "navy", "dark blue", "light blue", "teal"},
    "red": {"red", "maroon", "burgundy", "dark red"},
    "green": {"green", "dark green", "teal", "olive"},
    "yellow": {"yellow", "gold", "amber"},
    "orange": {"orange", "copper", "amber"},
    "brown": {"brown", "bronze", "beige", "tan", "copper"},
    "beige": {"beige", "tan", "cream", "champagne", "gold"},
}


def validate_against_schema(instance: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Minimal validator for the structured-output subset used here."""
    errors: list[str] = []
    expected = schema.get("type")
    types = expected if isinstance(expected, list) else [expected]

    if instance is None:
        if "null" not in types:
            errors.append(f"{path}: null not allowed")
        return errors

    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: value {instance!r} not in enum")

    if "object" in types and isinstance(instance, dict):
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{path}.{key}: required property missing")
        for key, value in instance.items():
            if key in properties:
                errors.extend(validate_against_schema(value, properties[key], f"{path}.{key}"))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}.{key}: additional property not allowed")
        return errors
    if "array" in types and isinstance(instance, list):
        item_schema = schema.get("items", {})
        for index, item in enumerate(instance):
            errors.extend(validate_against_schema(item, item_schema, f"{path}[{index}]"))
        return errors
    if "string" in types and isinstance(instance, str):
        return errors
    if "boolean" in types and isinstance(instance, bool):
        return errors
    if "integer" in types and isinstance(instance, int) and not isinstance(instance, bool):
        return errors
    if "number" in types and isinstance(instance, (int, float)) and not isinstance(instance, bool):
        return errors

    errors.append(f"{path}: expected {types}, got {type(instance).__name__}")
    return errors


class ObservationContractGate:
    """Decides what the model's structured output is allowed to mean for the claim."""

    def __init__(self, config: ContractConfig) -> None:
        self._config = config

    def _corroborated(self, damages: list[dict[str, Any]]) -> bool:
        return _corroborated_rule(
            damages, self._config.unreported_zone_min_confidence, self._config.unreported_zone_min_severity
        )

    def evaluate(
        self,
        output: dict[str, Any],
        claim: Claim,
        sent_photo_ids: list[str],
        agreement: dict[str, dict[str, int]] | None = None,
    ) -> ContractResult:
        schema_errors = validate_against_schema(output, response_json_schema())
        if schema_errors:
            findings = tuple(ContractFinding("schema_violation", "block", error) for error in schema_errors[:20])
            return ContractResult(
                valid_schema=False, findings=findings, confirmed_damages=(), unconfirmed_damages=(),
                observed_zones=(), uncovered_reported_zones=(), unreported_observed_zones=(),
                overall_severity=None, identity_flags=(),
            )

        findings: list[ContractFinding] = []
        identity_flags: list[str] = []
        photos: list[dict[str, Any]] = output["photos"]

        returned_ids = [photo["photo_id"] for photo in photos]
        if returned_ids != sent_photo_ids:
            findings.append(
                ContractFinding(
                    "photo_id_set_mismatch", "block",
                    f"sent {sent_photo_ids} but model returned {returned_ids}",
                )
            )

        policy_plate = normalise_plate(claim.vehicle.plate)
        confirmed: list[dict[str, Any]] = []
        unconfirmed: list[dict[str, Any]] = []
        observed_zones: set[str] = set()
        usable_photos = 0

        for photo in photos:
            photo_id = photo["photo_id"]
            if not photo["vehicle_present"]:
                findings.append(ContractFinding("no_vehicle_in_photo", "warn", "model saw no vehicle", photo_id))
                continue
            usable_photos += 1

            if photo["plate_legible"] and photo["plate_text"]:
                observed_plate = normalise_plate(photo["plate_text"])
                if self._config.require_plate_match_when_legible and observed_plate != policy_plate:
                    identity_flags.append("plate_mismatch")
                    findings.append(
                        ContractFinding("plate_mismatch", "block", f"observed {observed_plate} vs policy {policy_plate}", photo_id)
                    )
                else:
                    findings.append(ContractFinding("plate_verified", "info", f"observed {observed_plate}", photo_id))

            if self._config.require_color_match and photo["body_color"]:
                if not colors_compatible(claim.vehicle.color, photo["body_color"]):
                    identity_flags.append("color_mismatch")
                    findings.append(
                        ContractFinding("color_mismatch", "block", f"observed {photo['body_color']!r} vs policy {claim.vehicle.color!r}", photo_id)
                    )

            for damage in photo["damages"]:
                confidence = damage["confidence"]
                record = {**damage, "photo_id": photo_id, "zone": PART_ZONES.get(damage["part"], "unknown")}
                if not 0.0 <= confidence <= 1.0:
                    findings.append(ContractFinding("confidence_out_of_range", "block", f"confidence {confidence}", photo_id))
                    continue
                if damage["part"] == "unknown" or confidence < self._config.min_damage_confidence:
                    unconfirmed.append(record)
                    continue
                confirmed.append(record)
                observed_zones.add(record["zone"])

            if photo["airbag_deployed"] is True:
                findings.append(ContractFinding("airbag_deployed", "escalate", "airbag deployment observed", photo_id))
            if photo["fluid_leak_visible"]:
                findings.append(ContractFinding("fluid_leak_visible", "escalate", "fluid leak observed", photo_id))
            issues = [issue for issue in photo["quality_issues"] if issue != "none"]
            if issues:
                findings.append(ContractFinding("model_reported_quality_issue", "warn", ", ".join(issues), photo_id))

        if usable_photos < self._config.min_admitted_photos:
            findings.append(
                ContractFinding("insufficient_usable_photos", "block", f"{usable_photos} usable photo(s), need {self._config.min_admitted_photos}")
            )

        if output["same_vehicle_across_photos"] is False:
            identity_flags.append("vehicle_inconsistent_across_photos")
            findings.append(ContractFinding("vehicle_inconsistent_across_photos", "block", "model reports different vehicles"))

        reported = {zone.value for zone in claim.reported_impact_zones}
        uncovered = sorted(reported - observed_zones)
        candidate_unreported = sorted(observed_zones - reported - {"glass", "interior", "undercarriage"})
        for zone in uncovered:
            findings.append(ContractFinding("reported_zone_not_observed", "warn", f"no confirmed damage in reported zone {zone}"))
        unreported: list[str] = []
        for zone in candidate_unreported:
            zone_damages = [record for record in confirmed if record["zone"] == zone]
            if self._corroborated(zone_damages):
                unreported.append(zone)
                findings.append(ContractFinding("unreported_zone_damaged", "escalate", f"confirmed damage in unreported zone {zone}"))
            else:
                parts = ", ".join(sorted({record["part"] for record in zone_damages}))
                findings.append(
                    ContractFinding("possible_unreported_damage", "warn", f"uncorroborated minor finding in unreported zone {zone}: {parts}")
                )

        severity = output["overall_severity"]
        if severity in {"severe", "possible_total_loss"}:
            findings.append(ContractFinding("high_severity", "escalate", f"overall severity {severity}"))
        if output["drivable_assessment"] == "likely_not_drivable":
            findings.append(ContractFinding("not_drivable", "escalate", "model assesses vehicle as not drivable"))
        if confirmed and max_severity(confirmed) == "minor" and severity in {"severe", "possible_total_loss"}:
            findings.append(
                ContractFinding("severity_inconsistent_with_parts", "warn", "overall severity exceeds every confirmed part severity")
            )

        return ContractResult(
            valid_schema=True,
            findings=tuple(findings),
            confirmed_damages=tuple(confirmed),
            unconfirmed_damages=tuple(unconfirmed),
            observed_zones=tuple(sorted(observed_zones)),
            uncovered_reported_zones=tuple(uncovered),
            unreported_observed_zones=tuple(unreported),
            overall_severity=severity,
            identity_flags=tuple(dict.fromkeys(identity_flags)),
        )


def _corroborated_rule(damages: list[dict[str, Any]], min_confidence: float, min_severity: str) -> bool:
    """Escalate an unreported zone only when the evidence is strong or repeated.

    Strong: a damage at or above the severity floor with confidence at or above the floor.
    Repeated: the same part reported in two or more photos.
    """
    floor = SEVERITY_RANK.get(min_severity, 1)
    for record in damages:
        if SEVERITY_RANK.get(record["severity"], 0) >= floor and float(record["confidence"]) >= min_confidence:
            return True
    photos_per_part: dict[str, set[str]] = {}
    for record in damages:
        photos_per_part.setdefault(record["part"], set()).add(record["photo_id"])
    return any(len(photos) >= 2 for photos in photos_per_part.values())


def normalise_plate(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def colors_compatible(policy_color: str, observed_color: str) -> bool:
    policy = policy_color.strip().lower()
    observed = observed_color.strip().lower()
    if policy == observed or policy in observed or observed in policy:
        return True
    accepted = COLOR_SYNONYMS.get(policy, {policy})
    return observed in accepted or any(token in accepted for token in observed.split())


def max_severity(damages: list[dict[str, Any]]) -> str:
    order = {"minor": 0, "moderate": 1, "severe": 2}
    return max((item["severity"] for item in damages), key=lambda value: order.get(value, -1))
