from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import urlparse

ENV_PREFIX = "CPX_"

VALID_IMAGE_DETAIL = ("low", "high", "auto")
VALID_TOKEN_METHODS = ("patch", "tile")
VALID_REASONING_EFFORT = ("none", "minimal", "low", "medium", "high")


@dataclass(frozen=True)
class FoundryConfig:
    enabled: bool
    base_url: str
    api_key: str
    model: str
    image_detail: str
    timeout_seconds: int
    max_output_tokens: int
    reasoning_effort: str
    samples: int = 1

    def validate(self) -> None:
        if not 1 <= self.samples <= 5:
            raise ValueError("foundry.samples must be between 1 and 5")
        if self.timeout_seconds < 5 or self.timeout_seconds > 300:
            raise ValueError("foundry.timeout_seconds must be between 5 and 300")
        if self.max_output_tokens < 256 or self.max_output_tokens > 16384:
            raise ValueError("foundry.max_output_tokens must be between 256 and 16384")
        if self.image_detail not in VALID_IMAGE_DETAIL:
            raise ValueError("foundry.image_detail must be low, high, or auto")
        if self.reasoning_effort not in VALID_REASONING_EFFORT:
            raise ValueError("foundry.reasoning_effort is not a recognised value")
        if not self.enabled:
            return
        if not self.base_url or "YOUR-RESOURCE" in self.base_url:
            raise ValueError("foundry.base_url must be set when Foundry is enabled")
        if not self.api_key or self.api_key == "replace-me":
            raise ValueError("foundry.api_key must be set when Foundry is enabled")
        if not self.model:
            raise ValueError("foundry.model must be set when Foundry is enabled")
        resolve_foundry_openai_base_url(self.base_url)


@dataclass(frozen=True)
class AdmissionConfig:
    allowed_formats: tuple[str, ...]
    max_file_megabytes: float
    min_shortest_side_px: int
    min_sharpness_score: float
    min_mean_luminance: float
    max_mean_luminance: float
    duplicate_hamming_threshold: int
    max_days_before_incident: int
    max_days_after_incident: int
    flag_editing_software: bool

    def validate(self) -> None:
        if not self.allowed_formats:
            raise ValueError("admission.allowed_formats must not be empty")
        if self.max_file_megabytes <= 0 or self.max_file_megabytes > 20:
            # Foundry documents a 20 MB input image ceiling for vision chat models.
            raise ValueError("admission.max_file_megabytes must be between 0 and 20")
        if self.min_shortest_side_px < 64:
            raise ValueError("admission.min_shortest_side_px must be at least 64")
        if self.min_sharpness_score < 0:
            raise ValueError("admission.min_sharpness_score must be non-negative")
        if not 0 <= self.min_mean_luminance < self.max_mean_luminance <= 255:
            raise ValueError("admission luminance bounds must satisfy 0 <= min < max <= 255")
        if not 0 <= self.duplicate_hamming_threshold <= 64:
            raise ValueError("admission.duplicate_hamming_threshold must be between 0 and 64")
        if self.max_days_before_incident < 0 or self.max_days_after_incident < 0:
            raise ValueError("admission capture-window days must be non-negative")


@dataclass(frozen=True)
class PreparationConfig:
    max_longest_side_px: int
    jpeg_quality: int
    token_method: str
    patch_multiplier: float
    patch_budget: int
    tile_base_tokens: int
    tile_tokens: int
    max_image_tokens_per_claim: int
    max_images_per_request: int

    def validate(self) -> None:
        if self.max_longest_side_px < 256 or self.max_longest_side_px > 2048:
            raise ValueError("preparation.max_longest_side_px must be between 256 and 2048")
        if self.jpeg_quality < 50 or self.jpeg_quality > 95:
            raise ValueError("preparation.jpeg_quality must be between 50 and 95")
        if self.token_method not in VALID_TOKEN_METHODS:
            raise ValueError("preparation.token_method must be patch or tile")
        if self.patch_multiplier < 1.0:
            raise ValueError("preparation.patch_multiplier must be at least 1.0")
        if self.patch_budget < 64:
            raise ValueError("preparation.patch_budget must be at least 64")
        if self.tile_base_tokens < 0 or self.tile_tokens <= 0:
            raise ValueError("preparation tile token values must be positive")
        if self.max_image_tokens_per_claim < 256:
            raise ValueError("preparation.max_image_tokens_per_claim must be at least 256")
        if not 1 <= self.max_images_per_request <= 10:
            # Foundry documents a 10 image limit per chat request.
            raise ValueError("preparation.max_images_per_request must be between 1 and 10")


@dataclass(frozen=True)
class ContractConfig:
    min_damage_confidence: float
    min_admitted_photos: int
    require_plate_match_when_legible: bool
    require_color_match: bool
    min_sample_agreement: int = 1
    unreported_zone_min_confidence: float = 0.8
    unreported_zone_min_severity: str = "moderate"

    def validate(self) -> None:
        if not 0 <= self.min_damage_confidence <= 1:
            raise ValueError("contract.min_damage_confidence must be between 0 and 1")
        if self.min_admitted_photos < 1:
            raise ValueError("contract.min_admitted_photos must be at least 1")
        if self.min_sample_agreement < 1:
            raise ValueError("contract.min_sample_agreement must be at least 1")
        if not 0 <= self.unreported_zone_min_confidence <= 1:
            raise ValueError("contract.unreported_zone_min_confidence must be between 0 and 1")
        if self.unreported_zone_min_severity not in ("minor", "moderate", "severe"):
            raise ValueError("contract.unreported_zone_min_severity must be minor, moderate, or severe")


@dataclass(frozen=True)
class AppConfig:
    foundry: FoundryConfig
    admission: AdmissionConfig
    preparation: PreparationConfig
    contract: ContractConfig
    report_directory: Path

    def validate(self) -> None:
        self.foundry.validate()
        self.admission.validate()
        self.preparation.validate()
        self.contract.validate()
        if self.contract.min_sample_agreement > self.foundry.samples:
            raise ValueError("contract.min_sample_agreement cannot exceed foundry.samples")

    def without_foundry(self) -> AppConfig:
        return replace(self, foundry=replace(self.foundry, enabled=False))


def load_config(path: Path, *, validate: bool = True) -> AppConfig:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    foundry_data = data.get("foundry", {})
    admission_data = data.get("admission", {})
    preparation_data = data.get("preparation", {})
    contract_data = data.get("contract", {})

    foundry = FoundryConfig(
        enabled=_bool(foundry_data.get("enabled", False)),
        base_url=str(foundry_data.get("base_url", "")),
        api_key=str(foundry_data.get("api_key", "")),
        model=str(foundry_data.get("model", "")),
        image_detail=str(foundry_data.get("image_detail", "high")),
        timeout_seconds=int(foundry_data.get("timeout_seconds", 90)),
        max_output_tokens=int(foundry_data.get("max_output_tokens", 4000)),
        reasoning_effort=str(foundry_data.get("reasoning_effort", "low")),
        samples=int(foundry_data.get("samples", 1)),
    )
    admission = AdmissionConfig(
        allowed_formats=tuple(str(item).upper() for item in admission_data.get("allowed_formats", ["JPEG", "PNG", "WEBP"])),
        max_file_megabytes=float(admission_data.get("max_file_megabytes", 20)),
        min_shortest_side_px=int(admission_data.get("min_shortest_side_px", 640)),
        min_sharpness_score=float(admission_data.get("min_sharpness_score", 60.0)),
        min_mean_luminance=float(admission_data.get("min_mean_luminance", 40.0)),
        max_mean_luminance=float(admission_data.get("max_mean_luminance", 215.0)),
        duplicate_hamming_threshold=int(admission_data.get("duplicate_hamming_threshold", 6)),
        max_days_before_incident=int(admission_data.get("max_days_before_incident", 1)),
        max_days_after_incident=int(admission_data.get("max_days_after_incident", 14)),
        flag_editing_software=_bool(admission_data.get("flag_editing_software", True)),
    )
    preparation = PreparationConfig(
        max_longest_side_px=int(preparation_data.get("max_longest_side_px", 1536)),
        jpeg_quality=int(preparation_data.get("jpeg_quality", 85)),
        token_method=str(preparation_data.get("token_method", "patch")),
        patch_multiplier=float(preparation_data.get("patch_multiplier", 1.2)),
        patch_budget=int(preparation_data.get("patch_budget", 2500)),
        tile_base_tokens=int(preparation_data.get("tile_base_tokens", 70)),
        tile_tokens=int(preparation_data.get("tile_tokens", 140)),
        max_image_tokens_per_claim=int(preparation_data.get("max_image_tokens_per_claim", 9000)),
        max_images_per_request=int(preparation_data.get("max_images_per_request", 10)),
    )
    contract = ContractConfig(
        min_damage_confidence=float(contract_data.get("min_damage_confidence", 0.6)),
        min_admitted_photos=int(contract_data.get("min_admitted_photos", 2)),
        require_plate_match_when_legible=_bool(contract_data.get("require_plate_match_when_legible", True)),
        require_color_match=_bool(contract_data.get("require_color_match", True)),
        min_sample_agreement=int(contract_data.get("min_sample_agreement", 1)),
        unreported_zone_min_confidence=float(contract_data.get("unreported_zone_min_confidence", 0.8)),
        unreported_zone_min_severity=str(contract_data.get("unreported_zone_min_severity", "moderate")),
    )

    config = AppConfig(
        foundry=foundry,
        admission=admission,
        preparation=preparation,
        contract=contract,
        report_directory=Path(str(data.get("report_directory", "reports"))),
    )
    config = apply_environment_overrides(config)
    if validate:
        config.validate()
    return config


def apply_environment_overrides(config: AppConfig, environ: dict[str, str] | None = None) -> AppConfig:
    env = os.environ if environ is None else environ
    foundry = config.foundry
    preparation = config.preparation

    if value := env.get(f"{ENV_PREFIX}FOUNDRY_ENABLED"):
        foundry = replace(foundry, enabled=_bool(value))
    if value := env.get(f"{ENV_PREFIX}FOUNDRY_BASE_URL"):
        foundry = replace(foundry, base_url=value)
    if value := env.get(f"{ENV_PREFIX}FOUNDRY_API_KEY"):
        foundry = replace(foundry, api_key=value)
    if value := env.get(f"{ENV_PREFIX}FOUNDRY_MODEL"):
        foundry = replace(foundry, model=value)
    if value := env.get(f"{ENV_PREFIX}IMAGE_DETAIL"):
        foundry = replace(foundry, image_detail=value.lower())
    if value := env.get(f"{ENV_PREFIX}REASONING_EFFORT"):
        foundry = replace(foundry, reasoning_effort=value.lower())
    if value := env.get(f"{ENV_PREFIX}SAMPLES"):
        foundry = replace(foundry, samples=int(value))
    if value := env.get(f"{ENV_PREFIX}MAX_IMAGE_TOKENS"):
        preparation = replace(preparation, max_image_tokens_per_claim=int(value))
    if value := env.get(f"{ENV_PREFIX}TOKEN_METHOD"):
        preparation = replace(preparation, token_method=value.lower())
    if value := env.get(f"{ENV_PREFIX}PATCH_MULTIPLIER"):
        preparation = replace(preparation, patch_multiplier=float(value))

    return replace(config, foundry=foundry, preparation=preparation)


def resolve_foundry_openai_base_url(value: str) -> str:
    """Normalise a Foundry project or Azure OpenAI endpoint to its /openai/v1/ base URL."""
    parsed = urlparse(value)
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        raise ValueError("Foundry endpoint must be an absolute HTTPS URL")

    host = parsed.netloc.lower()
    path = parsed.path.rstrip("/")

    if path.endswith("/openai/v1"):
        return f"https://{parsed.netloc}{path}/"
    if host.endswith(".services.ai.azure.com"):
        return f"https://{parsed.netloc}/openai/v1/"
    if host.endswith(".openai.azure.com"):
        return f"https://{parsed.netloc}/openai/v1/"
    return f"https://{parsed.netloc}/openai/v1/"


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
