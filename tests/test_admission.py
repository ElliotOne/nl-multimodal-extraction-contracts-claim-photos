from __future__ import annotations

from pathlib import Path

from PIL import Image

from claim_photo_extraction.admission import ImageAdmissionGate, hamming_distance, read_image_facts
from claim_photo_extraction.config import AppConfig
from claim_photo_extraction.models import AdmissionDecision, Claim, ClaimPhoto, DeclaredView
from claim_photo_extraction.samples import build_exif, draw_scene


def _records(dry_config: AppConfig, claim: Claim) -> dict[str, object]:
    gate = ImageAdmissionGate(dry_config.admission)
    return {record.photo_id: record for record in gate.evaluate_claim(claim)}


def test_sharp_photo_with_exif_is_admitted_clean(dry_config: AppConfig, sample_claim: Claim) -> None:
    record = _records(dry_config, sample_claim)["P1"]
    assert record.decision is AdmissionDecision.ADMIT
    assert record.flags == ()
    assert record.facts is not None and record.facts.captured_at_utc is not None


def test_blurred_photo_is_rejected(dry_config: AppConfig, sample_claim: Claim) -> None:
    record = _records(dry_config, sample_claim)["P2"]
    assert record.decision is AdmissionDecision.REJECT
    assert "blur_detected" in record.reasons


def test_resent_copy_is_rejected_as_duplicate_of_higher_priority_photo(dry_config: AppConfig, sample_claim: Claim) -> None:
    record = _records(dry_config, sample_claim)["P3"]
    assert record.decision is AdmissionDecision.REJECT
    assert "duplicate_of:P1" in record.reasons


def test_missing_exif_is_flagged_not_rejected(dry_config: AppConfig, sample_claim: Claim) -> None:
    record = _records(dry_config, sample_claim)["P4"]
    assert record.decision is AdmissionDecision.ADMIT_FLAGGED
    assert "capture_time_missing" in record.flags


def test_rotated_gps_photo_is_admitted_with_informational_flags(dry_config: AppConfig, sample_claim: Claim) -> None:
    record = _records(dry_config, sample_claim)["P5"]
    assert record.decision is AdmissionDecision.ADMIT_FLAGGED
    assert {"gps_metadata_present", "orientation_normalised"} <= set(record.flags)
    assert record.facts is not None
    assert record.facts.orientation == 6


def test_low_resolution_photo_is_rejected(dry_config: AppConfig, sample_claim: Claim) -> None:
    record = _records(dry_config, sample_claim)["P6"]
    assert record.decision is AdmissionDecision.REJECT
    assert "resolution_too_low" in record.reasons


def test_editing_software_and_old_capture_time_are_integrity_flags(dry_config: AppConfig, sample_claim: Claim, tmp_path: Path) -> None:
    path = tmp_path / "edited.jpg"
    draw_scene(1600, 1200, seed=9, body=(188, 190, 196), view="front").save(
        path, "JPEG", quality=90, exif=build_exif(captured="2026:09:01 10:00:00", software="Adobe Photoshop 26.0")
    )
    photo = ClaimPhoto(photo_id="PX", path=path, declared_view=DeclaredView.FRONT, priority=5)
    record = ImageAdmissionGate(dry_config.admission).evaluate(photo, sample_claim)
    assert record.decision is AdmissionDecision.ADMIT_FLAGGED
    assert {"editing_software_detected", "capture_time_outside_window"} <= set(record.flags)


def test_underexposed_photo_is_rejected(dry_config: AppConfig, sample_claim: Claim, tmp_path: Path) -> None:
    path = tmp_path / "dark.jpg"
    Image.new("RGB", (1600, 1200), (8, 8, 10)).save(path, "JPEG")
    photo = ClaimPhoto(photo_id="PD", path=path, declared_view=DeclaredView.FRONT, priority=5)
    record = ImageAdmissionGate(dry_config.admission).evaluate(photo, sample_claim)
    assert record.decision is AdmissionDecision.REJECT
    assert "underexposed" in record.reasons


def test_missing_file_is_rejected(dry_config: AppConfig, sample_claim: Claim, tmp_path: Path) -> None:
    photo = ClaimPhoto(photo_id="PM", path=tmp_path / "nope.jpg", declared_view=DeclaredView.FRONT, priority=5)
    record = ImageAdmissionGate(dry_config.admission).evaluate(photo, sample_claim)
    assert record.reasons == ("file_missing",)


def test_facts_read_exif_without_sending_it_anywhere(sample_photos: dict[str, Path]) -> None:
    facts = read_image_facts(sample_photos["P5"])
    assert facts.has_gps is True
    assert facts.captured_at_utc is not None
    assert facts.captured_at_utc.isoformat() == "2026-11-09T17:45:30+00:00"


def test_hamming_distance() -> None:
    assert hamming_distance("0000000000000000", "0000000000000003") == 2


def test_rotated_sample_is_upright_after_exif_normalisation(sample_photos: dict[str, Path]) -> None:
    """P5 is stored sideways with orientation 6; after exif_transpose it must match P2's framing.

    P2 is the same close-up, blurred and stored upright. If the sample generator rotates the wrong
    way, the normalised P5 ends up upside down and this distance comparison flips.
    """
    from PIL import ImageOps

    from claim_photo_extraction.admission import difference_hash

    with Image.open(sample_photos["P5"]) as image:
        upright = (ImageOps.exif_transpose(image) or image).convert("L")
    p2_hash = read_image_facts(sample_photos["P2"]).dhash_hex
    assert hamming_distance(p2_hash, difference_hash(upright)) < hamming_distance(p2_hash, difference_hash(upright.rotate(180)))
