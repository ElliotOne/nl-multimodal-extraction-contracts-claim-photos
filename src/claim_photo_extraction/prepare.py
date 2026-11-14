from __future__ import annotations

import base64
import math
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps

from .config import PreparationConfig
from .models import AdmissionRecord, ClaimPhoto, ImageBudget, PreparedImage

PATCH_SIZE = 32
TILE_SIZE = 512
TILE_FIT_LONGEST = 2048
TILE_SHORTEST_TARGET = 768


def prepare_image(photo: ClaimPhoto, config: PreparationConfig) -> PreparedImage:
    """Normalise orientation, bound resolution, strip metadata, and estimate image tokens.

    The encoded JPEG carries no EXIF block: capture time, GPS, and software tags were
    already read by the admission gate and must not travel to the model endpoint.
    """
    with Image.open(photo.path) as image:
        oriented = ImageOps.exif_transpose(image) or image
        rgb = oriented.convert("RGB")
        rgb.thumbnail((config.max_longest_side_px, config.max_longest_side_px), Image.Resampling.LANCZOS)
        width, height = rgb.size
        buffer = BytesIO()
        rgb.save(buffer, format="JPEG", quality=config.jpeg_quality, optimize=True)

    encoded = buffer.getvalue()
    return PreparedImage(
        photo_id=photo.photo_id,
        declared_view=photo.declared_view,
        width=width,
        height=height,
        encoded_bytes=len(encoded),
        estimated_image_tokens=estimate_image_tokens(width, height, config),
        jpeg_base64=base64.b64encode(encoded).decode("ascii"),
    )


def estimate_image_tokens(width: int, height: int, config: PreparationConfig) -> int:
    if config.token_method == "patch":
        return estimate_patch_tokens(width, height, config.patch_multiplier, config.patch_budget)
    return estimate_tile_tokens(width, height, config.tile_base_tokens, config.tile_tokens)


def estimate_patch_tokens(width: int, height: int, multiplier: float, patch_budget: int) -> int:
    """Patch-based accounting used by gpt-5.4-mini, gpt-5-mini, gpt-4.1-mini and similar models.

    Count 32x32 patches; if the count exceeds the budget, shrink the image so the whole
    number of patches fits, recount, then multiply by the model multiplier and round up.
    """
    patches = math.ceil(width / PATCH_SIZE) * math.ceil(height / PATCH_SIZE)
    if patches > patch_budget:
        shrink = math.sqrt((PATCH_SIZE**2 * patch_budget) / (width * height))
        scaled_w = width * shrink / PATCH_SIZE
        scaled_h = height * shrink / PATCH_SIZE
        adjusted = shrink * min(math.floor(scaled_w) / scaled_w, math.floor(scaled_h) / scaled_h)
        width = int(width * adjusted)
        height = int(height * adjusted)
        patches = math.ceil(width / PATCH_SIZE) * math.ceil(height / PATCH_SIZE)
    return math.ceil(patches * multiplier)


def estimate_tile_tokens(width: int, height: int, base_tokens: int, tile_tokens: int) -> int:
    """Tile-based accounting used by gpt-4o, gpt-4.1, gpt-5, and gpt-5.1 at high detail.

    Fit within 2048x2048, scale the shortest side to 768, count 512px tiles,
    then charge base tokens plus tile tokens per tile.
    """
    longest = max(width, height)
    if longest > TILE_FIT_LONGEST:
        scale = TILE_FIT_LONGEST / longest
        width, height = int(width * scale), int(height * scale)
    shortest = min(width, height)
    if shortest > TILE_SHORTEST_TARGET:
        scale = TILE_SHORTEST_TARGET / shortest
        width, height = int(width * scale), int(height * scale)
    tiles = math.ceil(width / TILE_SIZE) * math.ceil(height / TILE_SIZE)
    return base_tokens + tiles * tile_tokens


def budget_images(
    prepared: list[PreparedImage],
    admission: dict[str, AdmissionRecord],
    config: PreparationConfig,
    detail: str,
) -> tuple[list[PreparedImage], ImageBudget]:
    """Pack admitted images by priority until the per-claim image token budget is spent."""
    ordered = sorted(prepared, key=lambda item: (admission[item.photo_id].priority, item.photo_id))
    included: list[PreparedImage] = []
    excluded: list[dict[str, str]] = []
    used = 0
    for image in ordered:
        if len(included) >= config.max_images_per_request:
            excluded.append({"photo_id": image.photo_id, "reason": "max_images_per_request"})
            continue
        if used + image.estimated_image_tokens > config.max_image_tokens_per_claim:
            excluded.append({"photo_id": image.photo_id, "reason": "image_token_budget_exceeded"})
            continue
        included.append(image)
        used += image.estimated_image_tokens

    budget = ImageBudget(
        method=config.token_method,
        detail=detail,
        max_image_tokens=config.max_image_tokens_per_claim,
        used_image_tokens=used,
        included_photo_ids=tuple(item.photo_id for item in included),
        excluded=tuple(excluded),
    )
    return included, budget


def data_uri(image: PreparedImage) -> str:
    return f"data:image/jpeg;base64,{image.jpeg_base64}"


def write_prepared_copy(image: PreparedImage, directory: Path) -> Path:
    """Persist the exact bytes sent to the model so a reviewer can see what the model saw."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{image.photo_id}.sent.jpg"
    target.write_bytes(base64.b64decode(image.jpeg_base64))
    return target
