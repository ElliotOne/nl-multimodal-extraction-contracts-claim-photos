from __future__ import annotations

from pathlib import Path

import pytest

from claim_photo_extraction.config import (
    AppConfig,
    apply_environment_overrides,
    load_config,
    resolve_foundry_openai_base_url,
)

ROOT = Path(__file__).resolve().parents[1]


def test_dry_run_config_validates(dry_config: AppConfig) -> None:
    assert dry_config.foundry.enabled is False
    assert dry_config.preparation.max_images_per_request == 10
    assert dry_config.admission.max_file_megabytes == 20


def test_live_config_rejects_placeholders() -> None:
    with pytest.raises(ValueError, match="base_url"):
        load_config(ROOT / "config" / "foundry.json")


def test_environment_overrides_apply(dry_config: AppConfig) -> None:
    config = apply_environment_overrides(
        dry_config,
        {
            "CPX_FOUNDRY_ENABLED": "true",
            "CPX_FOUNDRY_BASE_URL": "https://example.services.ai.azure.com/api/projects/p1",
            "CPX_FOUNDRY_API_KEY": "k",
            "CPX_FOUNDRY_MODEL": "gpt-4.1-mini",
            "CPX_IMAGE_DETAIL": "LOW",
            "CPX_TOKEN_METHOD": "patch",
            "CPX_PATCH_MULTIPLIER": "1.62",
            "CPX_MAX_IMAGE_TOKENS": "5000",
        },
    )
    config.validate()
    assert config.foundry.model == "gpt-4.1-mini"
    assert config.foundry.image_detail == "low"
    assert config.preparation.patch_multiplier == 1.62
    assert config.preparation.max_image_tokens_per_claim == 5000


def test_endpoint_resolution_forms() -> None:
    assert resolve_foundry_openai_base_url("https://r.services.ai.azure.com/api/projects/x") == "https://r.services.ai.azure.com/openai/v1/"
    assert resolve_foundry_openai_base_url("https://r.openai.azure.com/") == "https://r.openai.azure.com/openai/v1/"
    assert resolve_foundry_openai_base_url("https://r.services.ai.azure.com/openai/v1") == "https://r.services.ai.azure.com/openai/v1/"
    with pytest.raises(ValueError):
        resolve_foundry_openai_base_url("http://insecure.example.com/openai/v1/")


def test_image_limits_are_bounded_by_documented_foundry_ceilings(dry_config: AppConfig) -> None:
    from dataclasses import replace

    too_many = replace(dry_config.preparation, max_images_per_request=11)
    with pytest.raises(ValueError, match="max_images_per_request"):
        too_many.validate()
    too_big = replace(dry_config.admission, max_file_megabytes=25)
    with pytest.raises(ValueError, match="max_file_megabytes"):
        too_big.validate()
