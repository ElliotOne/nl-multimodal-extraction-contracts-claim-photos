from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from .models import Claim, ClaimPhoto, DamageZone, DeclaredView, Vehicle


def load_claim(path: Path) -> Claim:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    base_dir = path.parent
    vehicle_data = data["vehicle"]
    photos = tuple(
        ClaimPhoto(
            photo_id=str(item["photo_id"]),
            path=(base_dir / str(item["file"])).resolve(),
            declared_view=DeclaredView(str(item.get("declared_view", "unknown"))),
            priority=int(item.get("priority", 50)),
        )
        for item in data.get("photos", [])
    )
    photo_ids = [photo.photo_id for photo in photos]
    if len(photo_ids) != len(set(photo_ids)):
        raise ValueError("claim photo_id values must be unique")

    return Claim(
        claim_id=str(data["claim_id"]),
        incident_at_utc=parse_utc(str(data["incident_at_utc"])),
        reported_at_utc=parse_utc(str(data["reported_at_utc"])),
        vehicle=Vehicle(
            year=int(vehicle_data["year"]),
            make=str(vehicle_data["make"]),
            model=str(vehicle_data["model"]),
            color=str(vehicle_data["color"]),
            plate=str(vehicle_data["plate"]),
        ),
        reported_impact_zones=tuple(DamageZone(str(zone)) for zone in data.get("reported_impact_zones", [])),
        claimant_statement=str(data.get("claimant_statement", "")),
        photos=photos,
    )


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
