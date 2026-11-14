from __future__ import annotations

import base64
from dataclasses import replace
from io import BytesIO
from pathlib import Path

from PIL import Image

from claim_photo_extraction.admission import ImageAdmissionGate
from claim_photo_extraction.config import AppConfig
from claim_photo_extraction.models import Claim
from claim_photo_extraction.prepare import (
    budget_images,
    estimate_patch_tokens,
    estimate_tile_tokens,
    prepare_image,
)


def test_patch_tokens_match_documented_example() -> None:
    # OpenAI vision guide: a 1024x1024 image is 1024 patches; at multiplier 1.2 that bills 1229 tokens.
    assert estimate_patch_tokens(1024, 1024, 1.2, 6144) == 1229
    # gpt-4.1-mini multiplier 1.62 on the same image.
    assert estimate_patch_tokens(1024, 1024, 1.62, 6144) == 1659


def test_patch_budget_shrinks_oversized_images() -> None:
    # 2048x2048 is 4096 patches; a 2500-patch budget shrinks it to 1600x1600 = 2500 patches -> 3000 tokens.
    assert estimate_patch_tokens(2048, 2048, 1.2, 2500) == 3000


def test_tile_tokens_match_documented_rules() -> None:
    # 1024x1024 at high detail: shortest side scaled to 768 -> 768x768 -> 4 tiles.
    assert estimate_tile_tokens(1024, 1024, 85, 170) == 85 + 4 * 170
    assert estimate_tile_tokens(1024, 1024, 70, 140) == 70 + 4 * 140


def test_prepared_image_is_bounded_oriented_and_stripped(dry_config: AppConfig, sample_claim: Claim) -> None:
    closeup = next(photo for photo in sample_claim.photos if photo.photo_id == "P5")
    prepared = prepare_image(closeup, dry_config.preparation)
    assert max(prepared.width, prepared.height) <= dry_config.preparation.max_longest_side_px
    # Stored sideways with orientation 6, so the normalised image is landscape again.
    assert prepared.width > prepared.height
    with Image.open(BytesIO(base64.b64decode(prepared.jpeg_base64))) as sent:
        assert sent.format == "JPEG"
        assert not sent.getexif()
    assert prepared.estimated_image_tokens > 0


def test_budget_packs_by_priority_and_records_exclusions(dry_config: AppConfig, sample_claim: Claim) -> None:
    gate = ImageAdmissionGate(dry_config.admission)
    admission = {record.photo_id: record for record in gate.evaluate_claim(sample_claim) if record.is_admitted}
    photos = {photo.photo_id: photo for photo in sample_claim.photos}
    prepared = [prepare_image(photos[photo_id], dry_config.preparation) for photo_id in admission]

    included, budget = budget_images(prepared, admission, dry_config.preparation, "high")
    assert [image.photo_id for image in included] == ["P1", "P4", "P5"]
    assert budget.used_image_tokens <= budget.max_image_tokens

    tight = replace(dry_config.preparation, max_image_tokens_per_claim=prepared[0].estimated_image_tokens + 10)
    included, budget = budget_images(prepared, admission, tight, "high")
    assert [image.photo_id for image in included] == ["P1"]
    assert {item["photo_id"] for item in budget.excluded} == {"P4", "P5"}
    assert all(item["reason"] == "image_token_budget_exceeded" for item in budget.excluded)


def test_budget_respects_request_image_limit(dry_config: AppConfig, sample_claim: Claim, tmp_path: Path) -> None:
    gate = ImageAdmissionGate(dry_config.admission)
    admission = {record.photo_id: record for record in gate.evaluate_claim(sample_claim) if record.is_admitted}
    photos = {photo.photo_id: photo for photo in sample_claim.photos}
    prepared = [prepare_image(photos[photo_id], dry_config.preparation) for photo_id in admission]
    one_only = replace(dry_config.preparation, max_images_per_request=1)
    included, budget = budget_images(prepared, admission, one_only, "high")
    assert len(included) == 1
    assert all(item["reason"] == "max_images_per_request" for item in budget.excluded)
