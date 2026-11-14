"""Aggregate several independent model samples into one schema-shaped observation.

A vision model asked the same question twice will not always list the same parts.
Instead of trusting one sample, the engine asks for N samples and this module keeps
what a majority of them agree on. The result has exactly the response schema shape,
so the contract gate treats it like any single answer. Agreement counts are returned
separately so the report and the corroboration rules can see them.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .contract import normalise_plate

SEVERITY_ORDER = {"minor": 0, "moderate": 1, "severe": 2}
DAMAGE_TYPE_ORDER = {
    "unknown": -1, "scratch": 0, "paint_transfer": 0, "dent": 1, "crack": 2,
    "deformation": 2, "broken": 3, "missing": 3, "glass_shatter": 3,
}
OVERALL_ORDER = {"minor": 0, "moderate": 1, "severe": 2, "possible_total_loss": 3}


def aggregate_samples(samples: list[dict[str, Any]], min_agreement: int) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    """Majority-vote aggregation across samples. Returns (output, agreement).

    agreement maps photo_id -> part -> number of samples that reported that part.
    """
    if not samples:
        raise ValueError("no samples to aggregate")
    if len(samples) == 1:
        single = samples[0]
        agreement = {
            photo["photo_id"]: {damage["part"]: 1 for damage in photo.get("damages", [])}
            for photo in single.get("photos", [])
        }
        return single, agreement

    photo_ids = [photo["photo_id"] for photo in samples[0]["photos"]]
    photos_out: list[dict[str, Any]] = []
    agreement: dict[str, dict[str, int]] = {}

    for photo_id in photo_ids:
        versions = [
            next((photo for photo in sample["photos"] if photo["photo_id"] == photo_id), None)
            for sample in samples
        ]
        versions = [version for version in versions if version is not None]
        if not versions:
            continue

        part_votes: Counter[str] = Counter()
        part_records: dict[str, list[dict[str, Any]]] = {}
        for version in versions:
            seen_parts: set[str] = set()
            for damage in version.get("damages", []):
                part = damage["part"]
                if part in seen_parts:
                    continue
                seen_parts.add(part)
                part_votes[part] += 1
                part_records.setdefault(part, []).append(damage)
        agreement[photo_id] = dict(part_votes)

        damages_out = []
        for part, votes in part_votes.most_common():
            if votes < min_agreement:
                continue
            records = part_records[part]
            severity = _ordinal_vote([record["severity"] for record in records], SEVERITY_ORDER)
            matching = [record for record in records if record["severity"] == severity] or records
            damages_out.append(
                {
                    "part": part,
                    "damage_type": _type_vote([record["damage_type"] for record in matching]),
                    "severity": severity,
                    "confidence": round(sum(float(record["confidence"]) for record in records) / len(records), 3),
                }
            )

        plate_values = [version["plate_text"] for version in versions if version.get("plate_legible") and version.get("plate_text")]
        plate_text = _agreed_plate(plate_values, min_agreement)
        colors = [version["body_color"] for version in versions if version.get("body_color")]
        airbags = [version["airbag_deployed"] for version in versions if version.get("airbag_deployed") is not None]
        issue_votes: Counter[str] = Counter()
        for version in versions:
            issue_votes.update({issue for issue in version.get("quality_issues", []) if issue != "none"})
        issues = sorted(issue for issue, votes in issue_votes.items() if votes >= min_agreement)

        photos_out.append(
            {
                "photo_id": photo_id,
                "vehicle_present": sum(1 for version in versions if version["vehicle_present"]) >= min_agreement,
                "observed_view": _majority([version["observed_view"] for version in versions]),
                "plate_text": plate_text,
                "plate_legible": plate_text is not None,
                "body_color": _majority(colors) if colors else None,
                "damages": damages_out,
                "airbag_deployed": (any(airbags) if airbags else None),
                "fluid_leak_visible": sum(1 for version in versions if version["fluid_leak_visible"]) >= min_agreement,
                "quality_issues": issues or ["none"],
            }
        )

    same_vehicle_values = [sample["same_vehicle_across_photos"] for sample in samples if sample.get("same_vehicle_across_photos") is not None]
    output = {
        "photos": photos_out,
        "same_vehicle_across_photos": (all(same_vehicle_values) if same_vehicle_values else None),
        "overall_severity": _ordinal_vote([sample["overall_severity"] for sample in samples], OVERALL_ORDER),
        "drivable_assessment": _majority([sample["drivable_assessment"] for sample in samples]),
        "notes": samples[0].get("notes", ""),
    }
    return output, agreement


def _agreed_plate(values: list[str], min_agreement: int) -> str | None:
    """A plate counts only if enough samples read the same registration after normalisation.

    Counting samples that read *a* plate is not enough: two samples reading different
    registrations must not pass one of them off as agreed.
    """
    if not values:
        return None
    winner, votes = Counter(normalise_plate(value) for value in values).most_common(1)[0]
    if votes < min_agreement:
        return None
    return next(value for value in values if normalise_plate(value) == winner)


def _type_vote(values: list[str]) -> str:
    """Most common damage type among the samples that agree on severity; ties go to the more serious type.

    Taking the type only from samples that match the voted severity keeps the label coherent:
    one sample's "scratch, minor" and another's "deformation, moderate" must not become
    "scratch, moderate", a combination neither sample reported.
    """
    counts = Counter(values)
    top = max(counts.values())
    tied = [value for value, count in counts.items() if count == top]
    return max(tied, key=lambda value: DAMAGE_TYPE_ORDER.get(value, -1))


def _majority(values: list[str]) -> str:
    """Most common categorical value; ties keep the first value seen."""
    return Counter(values).most_common(1)[0][0]


def _ordinal_vote(values: list[str], order: dict[str, int]) -> str:
    """Strict majority if one exists, otherwise the median, rounding up on an even split.

    minor/moderate/severe with one vote each gives moderate, not severe: a single
    sample should not decide a severity the others disagree with. Two samples that
    split moderate/severe give severe, because under-reporting is the costlier error.
    """
    value, votes = Counter(values).most_common(1)[0]
    if votes * 2 > len(values):
        return value
    ranked = sorted(values, key=lambda item: order.get(item, -1))
    return ranked[len(ranked) // 2]
