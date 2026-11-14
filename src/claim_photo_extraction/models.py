from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class DeclaredView(str, Enum):
    FRONT = "front"
    REAR = "rear"
    LEFT_SIDE = "left_side"
    RIGHT_SIDE = "right_side"
    FRONT_LEFT = "front_left"
    FRONT_RIGHT = "front_right"
    REAR_LEFT = "rear_left"
    REAR_RIGHT = "rear_right"
    INTERIOR = "interior"
    CLOSEUP = "closeup"
    UNKNOWN = "unknown"


class DamageZone(str, Enum):
    FRONT = "front"
    REAR = "rear"
    LEFT = "left"
    RIGHT = "right"
    ROOF = "roof"
    GLASS = "glass"
    INTERIOR = "interior"
    UNDERCARRIAGE = "undercarriage"


class AdmissionDecision(str, Enum):
    ADMIT = "admit"
    ADMIT_FLAGGED = "admit_flagged"
    REJECT = "reject"


class RoutingDecision(str, Enum):
    FAST_TRACK = "fast_track"
    ADJUSTER_REVIEW = "adjuster_review"
    RESUBMIT_PHOTOS = "resubmit_photos"
    INVESTIGATE = "investigate"


@dataclass(frozen=True)
class Vehicle:
    year: int
    make: str
    model: str
    color: str
    plate: str


@dataclass(frozen=True)
class ClaimPhoto:
    photo_id: str
    path: Path
    declared_view: DeclaredView
    priority: int


@dataclass(frozen=True)
class Claim:
    claim_id: str
    incident_at_utc: datetime
    reported_at_utc: datetime
    vehicle: Vehicle
    reported_impact_zones: tuple[DamageZone, ...]
    claimant_statement: str
    photos: tuple[ClaimPhoto, ...]


@dataclass(frozen=True)
class ImageFacts:
    """Deterministic facts read from the file before any model is involved."""

    format: str
    width: int
    height: int
    file_bytes: int
    sharpness_score: float
    mean_luminance: float
    dhash_hex: str
    captured_at_utc: datetime | None
    software: str | None
    has_gps: bool
    orientation: int


@dataclass(frozen=True)
class AdmissionRecord:
    photo_id: str
    declared_view: DeclaredView
    priority: int
    decision: AdmissionDecision
    reasons: tuple[str, ...]
    flags: tuple[str, ...]
    facts: ImageFacts | None

    @property
    def is_admitted(self) -> bool:
        return self.decision is not AdmissionDecision.REJECT


@dataclass(frozen=True)
class PreparedImage:
    photo_id: str
    declared_view: DeclaredView
    width: int
    height: int
    encoded_bytes: int
    estimated_image_tokens: int
    jpeg_base64: str = field(repr=False)


@dataclass(frozen=True)
class ImageBudget:
    method: str
    detail: str
    max_image_tokens: int
    used_image_tokens: int
    included_photo_ids: tuple[str, ...]
    excluded: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class ModelCall:
    enabled: bool
    called: bool
    source: str
    model: str | None
    raw_output: str | None
    parsed_output: dict[str, Any] | None
    refusal: str | None
    usage: dict[str, Any] | None
    error: str | None
    sample_count: int = 1
    sample_outputs: tuple[dict[str, Any], ...] = ()
    agreement: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass(frozen=True)
class ContractFinding:
    code: str
    severity: str
    detail: str
    photo_id: str | None = None


@dataclass(frozen=True)
class ContractResult:
    valid_schema: bool
    findings: tuple[ContractFinding, ...]
    confirmed_damages: tuple[dict[str, Any], ...]
    unconfirmed_damages: tuple[dict[str, Any], ...]
    observed_zones: tuple[str, ...]
    uncovered_reported_zones: tuple[str, ...]
    unreported_observed_zones: tuple[str, ...]
    overall_severity: str | None
    identity_flags: tuple[str, ...]


@dataclass(frozen=True)
class Routing:
    decision: RoutingDecision
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RunReport:
    claim_id: str
    run_at_utc: datetime
    admission: tuple[AdmissionRecord, ...]
    budget: ImageBudget
    model_call: ModelCall
    contract: ContractResult | None
    routing: Routing

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items() if key != "jpeg_base64"}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, Path):
        return value.as_posix()
    return value
