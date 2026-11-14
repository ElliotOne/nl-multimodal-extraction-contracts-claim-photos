from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps, ImageStat, UnidentifiedImageError

from .config import AdmissionConfig
from .models import AdmissionDecision, AdmissionRecord, Claim, ClaimPhoto, ImageFacts

# EXIF tag ids used by the gate. Read deterministically by code; the model never sees them.
TAG_ORIENTATION = 274
TAG_SOFTWARE = 305
IFD_EXIF = 0x8769
IFD_GPS = 0x8825
TAG_DATETIME_ORIGINAL = 36867
TAG_OFFSET_TIME_ORIGINAL = 36881

SHARPNESS_ANALYSIS_WIDTH = 800
LAPLACIAN_KERNEL = (0, 1, 0, 1, -4, 1, 0, 1, 0)

EDITING_SOFTWARE_MARKERS = (
    "photoshop",
    "lightroom",
    "gimp",
    "affinity",
    "snapseed",
    "picsart",
    "pixlr",
    "canva",
    "facetune",
)


def read_image_facts(path: Path) -> ImageFacts:
    with Image.open(path) as image:
        image_format = (image.format or "UNKNOWN").upper()
        exif = image.getexif()
        orientation = int(exif.get(TAG_ORIENTATION, 1) or 1)
        software_raw = exif.get(TAG_SOFTWARE)
        software = str(software_raw).strip() if software_raw else None
        exif_ifd = exif.get_ifd(IFD_EXIF)
        gps_ifd = exif.get_ifd(IFD_GPS)
        captured_at = _parse_exif_datetime(
            exif_ifd.get(TAG_DATETIME_ORIGINAL), exif_ifd.get(TAG_OFFSET_TIME_ORIGINAL)
        )

        oriented = ImageOps.exif_transpose(image) or image
        width, height = oriented.size
        gray = oriented.convert("L")
        mean_luminance = float(ImageStat.Stat(gray).mean[0])
        sharpness = _sharpness_score(gray)
        dhash_hex = difference_hash(gray)

    return ImageFacts(
        format=image_format,
        width=width,
        height=height,
        file_bytes=path.stat().st_size,
        sharpness_score=round(sharpness, 2),
        mean_luminance=round(mean_luminance, 2),
        dhash_hex=dhash_hex,
        captured_at_utc=captured_at,
        software=software,
        has_gps=bool(gps_ifd),
        orientation=orientation,
    )


class ImageAdmissionGate:
    """Deterministic per-photo admission. Runs before any bytes are sent to the model."""

    def __init__(self, config: AdmissionConfig) -> None:
        self._config = config

    def evaluate_claim(self, claim: Claim) -> tuple[AdmissionRecord, ...]:
        records: list[AdmissionRecord] = []
        admitted_hashes: dict[str, str] = {}
        ordered = sorted(claim.photos, key=lambda photo: (photo.priority, photo.photo_id))
        for photo in ordered:
            record = self.evaluate(photo, claim, admitted_hashes)
            if record.is_admitted and record.facts is not None:
                admitted_hashes[photo.photo_id] = record.facts.dhash_hex
            records.append(record)
        return tuple(records)

    def evaluate(
        self,
        photo: ClaimPhoto,
        claim: Claim,
        admitted_hashes: dict[str, str] | None = None,
    ) -> AdmissionRecord:
        admitted_hashes = admitted_hashes or {}
        if not photo.path.is_file():
            return self._reject(photo, ("file_missing",))

        try:
            facts = read_image_facts(photo.path)
        except (UnidentifiedImageError, OSError):
            return self._reject(photo, ("unreadable_image",))

        reasons: list[str] = []
        flags: list[str] = []

        if facts.format not in self._config.allowed_formats:
            reasons.append("format_not_allowed")
        if facts.file_bytes > self._config.max_file_megabytes * 1024 * 1024:
            reasons.append("file_too_large")
        if min(facts.width, facts.height) < self._config.min_shortest_side_px:
            reasons.append("resolution_too_low")
        if facts.mean_luminance < self._config.min_mean_luminance:
            reasons.append("underexposed")
        elif facts.mean_luminance > self._config.max_mean_luminance:
            reasons.append("overexposed")
        if facts.sharpness_score < self._config.min_sharpness_score:
            reasons.append("blur_detected")

        for other_id, other_hash in admitted_hashes.items():
            if hamming_distance(facts.dhash_hex, other_hash) <= self._config.duplicate_hamming_threshold:
                reasons.append(f"duplicate_of:{other_id}")
                break

        if reasons:
            return self._reject(photo, tuple(reasons), facts)

        if facts.captured_at_utc is None:
            flags.append("capture_time_missing")
        else:
            window_start = claim.incident_at_utc - timedelta(days=self._config.max_days_before_incident)
            window_end = claim.incident_at_utc + timedelta(days=self._config.max_days_after_incident)
            if not window_start <= facts.captured_at_utc <= window_end:
                flags.append("capture_time_outside_window")

        if self._config.flag_editing_software and facts.software and _looks_like_editor(facts.software):
            flags.append("editing_software_detected")
        if facts.has_gps:
            flags.append("gps_metadata_present")
        if facts.orientation != 1:
            flags.append("orientation_normalised")

        decision = AdmissionDecision.ADMIT_FLAGGED if flags else AdmissionDecision.ADMIT
        return AdmissionRecord(
            photo_id=photo.photo_id,
            declared_view=photo.declared_view,
            priority=photo.priority,
            decision=decision,
            reasons=(),
            flags=tuple(flags),
            facts=facts,
        )

    @staticmethod
    def _reject(photo: ClaimPhoto, reasons: tuple[str, ...], facts: ImageFacts | None = None) -> AdmissionRecord:
        return AdmissionRecord(
            photo_id=photo.photo_id,
            declared_view=photo.declared_view,
            priority=photo.priority,
            decision=AdmissionDecision.REJECT,
            reasons=reasons,
            flags=(),
            facts=facts,
        )


def _sharpness_score(gray: Image.Image) -> float:
    """Variance of a Laplacian response on a width-normalised grayscale copy."""
    width, height = gray.size
    if width > SHARPNESS_ANALYSIS_WIDTH:
        scale = SHARPNESS_ANALYSIS_WIDTH / width
        gray = gray.resize((SHARPNESS_ANALYSIS_WIDTH, max(1, int(height * scale))), Image.Resampling.BILINEAR)
    laplacian = gray.filter(ImageFilter.Kernel((3, 3), LAPLACIAN_KERNEL, scale=1, offset=128))
    return float(ImageStat.Stat(laplacian).var[0])


def difference_hash(gray: Image.Image) -> str:
    """64-bit dHash: compares horizontally adjacent pixels on a 9x8 downsample."""
    small = gray.resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(small.getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            left = pixels[row * 9 + col]
            right = pixels[row * 9 + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return f"{bits:016x}"


def hamming_distance(left_hex: str, right_hex: str) -> int:
    return bin(int(left_hex, 16) ^ int(right_hex, 16)).count("1")


def _parse_exif_datetime(value: object, offset: object) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    try:
        naive = datetime.strptime(text, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None

    tz = UTC
    if offset:
        offset_text = str(offset).strip()
        try:
            sign = -1 if offset_text.startswith("-") else 1
            hours, minutes = offset_text.lstrip("+-").split(":")
            tz = timezone(sign * timedelta(hours=int(hours), minutes=int(minutes)))
        except ValueError:
            tz = UTC
    return naive.replace(tzinfo=tz).astimezone(UTC)


def _looks_like_editor(software: str) -> bool:
    lowered = software.lower()
    return any(marker in lowered for marker in EDITING_SOFTWARE_MARKERS)
