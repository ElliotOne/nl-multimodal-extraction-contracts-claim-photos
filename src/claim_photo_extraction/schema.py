from __future__ import annotations

from typing import Any

from .models import Claim, PreparedImage
from .prepare import data_uri

SCHEMA_NAME = "ClaimPhotoObservations"

VIEWS = [
    "front", "rear", "left_side", "right_side", "front_left", "front_right",
    "rear_left", "rear_right", "interior", "closeup", "unknown",
]

PARTS = [
    "front_bumper", "hood", "grille", "headlight_left", "headlight_right", "windshield",
    "front_fender_left", "front_fender_right", "door_front_left", "door_front_right",
    "door_rear_left", "door_rear_right", "mirror_left", "mirror_right",
    "rocker_panel_left", "rocker_panel_right", "rear_bumper", "trunk_lid",
    "taillight_left", "taillight_right", "rear_window", "quarter_panel_left",
    "quarter_panel_right", "roof", "wheel_front_left", "wheel_front_right",
    "wheel_rear_left", "wheel_rear_right", "side_window", "interior_airbag",
    "interior_dashboard", "undercarriage", "unknown",
]

DAMAGE_TYPES = [
    "dent", "scratch", "crack", "broken", "missing", "deformation",
    "paint_transfer", "glass_shatter", "unknown",
]

SEVERITIES = ["minor", "moderate", "severe"]
OVERALL_SEVERITIES = ["minor", "moderate", "severe", "possible_total_loss"]
DRIVABLE = ["likely_drivable", "likely_not_drivable", "unknown"]
QUALITY_ISSUES = ["blur", "glare", "too_dark", "too_far", "obstructed", "partial_view", "none"]

PART_ZONES: dict[str, str] = {
    "front_bumper": "front", "hood": "front", "grille": "front", "headlight_left": "front",
    "headlight_right": "front", "windshield": "glass", "front_fender_left": "left",
    "front_fender_right": "right", "door_front_left": "left", "door_front_right": "right",
    "door_rear_left": "left", "door_rear_right": "right", "mirror_left": "left",
    "mirror_right": "right", "rocker_panel_left": "left", "rocker_panel_right": "right",
    "rear_bumper": "rear", "trunk_lid": "rear", "taillight_left": "rear",
    "taillight_right": "rear", "rear_window": "glass", "quarter_panel_left": "left",
    "quarter_panel_right": "right", "roof": "roof", "wheel_front_left": "left",
    "wheel_front_right": "right", "wheel_rear_left": "left", "wheel_rear_right": "right",
    "side_window": "glass", "interior_airbag": "interior", "interior_dashboard": "interior",
    "undercarriage": "undercarriage",
}


def response_json_schema() -> dict[str, Any]:
    """Strict structured-output schema.

    Every object sets additionalProperties=false and lists every property as required.
    Optional values are expressed as nullable unions. Range constraints such as a
    confidence between 0 and 1 are not expressible in the supported subset, so they
    are enforced afterwards by the contract gate.
    """
    damage = {
        "type": "object",
        "properties": {
            "part": {"type": "string", "enum": PARTS},
            "damage_type": {"type": "string", "enum": DAMAGE_TYPES},
            "severity": {"type": "string", "enum": SEVERITIES},
            "confidence": {"type": "number"},
        },
        "required": ["part", "damage_type", "severity", "confidence"],
        "additionalProperties": False,
    }
    photo = {
        "type": "object",
        "properties": {
            "photo_id": {"type": "string"},
            "vehicle_present": {"type": "boolean"},
            "observed_view": {"type": "string", "enum": VIEWS},
            "plate_text": {"type": ["string", "null"]},
            "plate_legible": {"type": "boolean"},
            "body_color": {"type": ["string", "null"]},
            "damages": {"type": "array", "items": damage},
            "airbag_deployed": {"type": ["boolean", "null"]},
            "fluid_leak_visible": {"type": "boolean"},
            "quality_issues": {"type": "array", "items": {"type": "string", "enum": QUALITY_ISSUES}},
        },
        "required": [
            "photo_id", "vehicle_present", "observed_view", "plate_text", "plate_legible",
            "body_color", "damages", "airbag_deployed", "fluid_leak_visible", "quality_issues",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "photos": {"type": "array", "items": photo},
            "same_vehicle_across_photos": {"type": ["boolean", "null"]},
            "overall_severity": {"type": "string", "enum": OVERALL_SEVERITIES},
            "drivable_assessment": {"type": "string", "enum": DRIVABLE},
            "notes": {"type": "string"},
        },
        "required": [
            "photos", "same_vehicle_across_photos", "overall_severity",
            "drivable_assessment", "notes",
        ],
        "additionalProperties": False,
    }


def response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {"name": SCHEMA_NAME, "strict": True, "schema": response_json_schema()},
    }


SYSTEM_PROMPT = (
    "You are a vehicle damage observation service for an insurance claims intake system. "
    "Describe only what is clearly visible in each photo. Do not estimate repair costs and do not "
    "infer hidden or structural damage. "
    "Left and right always mean the vehicle's own left and right, as seen by a driver sitting in "
    "the vehicle facing forward, never the viewer's left and right. "
    "Severity rubric: minor = cosmetic marks with no panel deformation; moderate = a deformed panel "
    "or a broken part that is localised; severe = crumpled or torn panels, multiple damaged panels, "
    "or a destroyed component. "
    "Report a part for a photo only when its damage is visible in that photo; do not carry findings "
    "from one photo into another. Do not report normal wear, dirt, reflections, or shadows as damage. "
    "If a licence plate is not clearly readable, set plate_legible to false and plate_text to null "
    "rather than guessing. Report one entry per photo id, in the order given, and never invent photo ids. "
    "Use the confidence field honestly: 0.9 or higher only when the damage is unambiguous, below 0.6 "
    "when you are unsure."
)


def build_messages(claim: Claim, images: list[PreparedImage], detail: str) -> list[dict[str, Any]]:
    """Compose the multimodal request. Claimant identity and the policy plate are withheld.

    The model is asked to read the plate; code compares it with the policy record later.
    Supplying the expected plate would turn a check into a suggestion.
    """
    zones = ", ".join(zone.value for zone in claim.reported_impact_zones) or "none reported"
    photo_ids = ", ".join(image.photo_id for image in images)
    context = (
        f"Claim context (do not repeat it back): insured vehicle is a {claim.vehicle.year} "
        f"{claim.vehicle.make} {claim.vehicle.model}, colour recorded as {claim.vehicle.color}. "
        f"Claimant reported impact zones: {zones}. "
        f"You will receive {len(images)} photo(s) with ids: {photo_ids}. "
        "Return exactly one observation per listed photo id."
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": context}]
    for image in images:
        content.append(
            {
                "type": "text",
                "text": f"Photo {image.photo_id} follows. Claimant declared view: {image.declared_view.value}.",
            }
        )
        content.append({"type": "image_url", "image_url": {"url": data_uri(image), "detail": detail}})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
