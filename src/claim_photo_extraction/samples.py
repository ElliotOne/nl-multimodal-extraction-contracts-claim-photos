"""Sample photo set derivation.

The sample claim uses two real photographs of a crash-damaged Suzuki Swift published on
Wikimedia Commons by Philafrenzy under CC BY-SA 4.0 (see data/source-photos/ATTRIBUTION.md).
From those two originals this module derives the six-photo upload set the sample claim uses:
a primary wide shot, a blurred close-up, a resent duplicate, a front view with no metadata,
a sharp close-up carrying GPS and a rotation tag, and a tiny thumbnail. EXIF timestamps are
rewritten to the fictional claim's incident window; the originals were taken in July 2017.

`draw_scene` remains available for tests that need a synthetic image with known properties.
"""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

IFD_EXIF = 0x8769
IFD_GPS = 0x8825
TAG_ORIENTATION = 274
TAG_SOFTWARE = 305
TAG_MAKE = 271
TAG_MODEL = 272
TAG_DATETIME_ORIGINAL = 36867

SOURCE_DIR = Path(__file__).resolve().parents[2] / "data" / "source-photos"
SOURCE_WIDE = SOURCE_DIR / "abandoned-suzuki-osidge-london-2.jpg"
SOURCE_FRONT = SOURCE_DIR / "abandoned-suzuki-osidge-london-1.jpg"

# Crop boxes are fractions of the wide source: (left, top, right, bottom).
DAMAGE_CROP = (0.52, 0.30, 0.92, 0.80)


def build_exif(
    *,
    captured: str | None,
    software: str | None = None,
    orientation: int = 1,
    gps: bool = False,
) -> Image.Exif:
    exif = Image.Exif()
    exif[TAG_MAKE] = "SampleCam"
    exif[TAG_MODEL] = "Phone 1.0"
    exif[TAG_ORIENTATION] = orientation
    if software:
        exif[TAG_SOFTWARE] = software
    if captured:
        exif.get_ifd(IFD_EXIF)[TAG_DATETIME_ORIGINAL] = captured
    if gps:
        gps_ifd = exif.get_ifd(IFD_GPS)
        gps_ifd[1] = "N"
        gps_ifd[2] = (51.0, 38.0, 12.0)
        gps_ifd[3] = "W"
        gps_ifd[4] = (0.0, 8.0, 41.0)
    return exif


def _load(path: Path, longest: int) -> Image.Image:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        rgb.thumbnail((longest, longest), Image.Resampling.LANCZOS)
        return rgb.copy()


def _crop_fraction(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    w, h = image.size
    return image.crop((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)))


def write_sample_set(directory: Path) -> dict[str, Path]:
    if not SOURCE_WIDE.is_file() or not SOURCE_FRONT.is_file():
        raise FileNotFoundError(f"source photos missing under {SOURCE_DIR}")
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    wide = _load(SOURCE_WIDE, 2048)
    front = _load(SOURCE_FRONT, 2048)

    # P1: primary wide shot of the damaged front-left corner, clean EXIF.
    paths["P1"] = directory / "p1-front-left.jpg"
    wide.save(paths["P1"], "JPEG", quality=90, exif=build_exif(captured="2026:11:09 17:42:10"))

    # P2: close-up of the damage that came out blurred.
    closeup = _crop_fraction(_load(SOURCE_WIDE, 4096), DAMAGE_CROP)
    closeup.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    paths["P2"] = directory / "p2-closeup-blurred.jpg"
    closeup.filter(ImageFilter.GaussianBlur(9)).save(
        paths["P2"], "JPEG", quality=88, exif=build_exif(captured="2026:11:09 17:43:02")
    )

    # P3: the claimant re-sent P1 at lower quality.
    paths["P3"] = directory / "p3-front-left-resend.jpg"
    wide.save(paths["P3"], "JPEG", quality=70, exif=build_exif(captured="2026:11:09 17:42:10"))

    # P4: straight-on front view forwarded through a messaging app: no EXIF at all.
    paths["P4"] = directory / "p4-front.jpg"
    front.save(paths["P4"], "JPEG", quality=88)

    # P5: sharp close-up stored rotated with an orientation tag and GPS data.
    paths["P5"] = directory / "p5-closeup.jpg"
    # EXIF orientation 6 means "rotate 90 degrees clockwise to display", so store it
    # rotated 90 degrees counter-clockwise (PIL rotate is counter-clockwise for positive angles).
    rotated = closeup.rotate(90, expand=True)
    rotated.save(paths["P5"], "JPEG", quality=90, exif=build_exif(captured="2026:11:09 17:45:30", orientation=6, gps=True))

    # P6: a thumbnail-sized copy of the front view.
    tiny = front.copy()
    tiny.thumbnail((480, 480), Image.Resampling.LANCZOS)
    paths["P6"] = directory / "p6-thumbnail.jpg"
    tiny.save(paths["P6"], "JPEG", quality=85, exif=build_exif(captured="2026:11:09 17:46:00"))

    return paths


def draw_scene(width: int, height: int, *, seed: int, body: tuple[int, int, int], view: str) -> Image.Image:
    """Synthetic scene with sharp edges and noise, used by tests that need known image properties."""
    rng = random.Random(seed)
    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)
    for y in range(height):
        t = y / max(1, height - 1)
        colour = (int(150 * (1 - t) + 70 * t), int(170 * (1 - t) + 72 * t), int(200 * (1 - t) + 76 * t))
        draw.line([(0, y), (width, y)], fill=colour)
    left, right = int(width * 0.12), int(width * 0.88)
    top, bottom = int(height * 0.38), int(height * 0.78)
    draw.rounded_rectangle([left, top + (bottom - top) // 3, right, bottom], radius=40, fill=body, outline=(30, 30, 30), width=6)
    draw.rounded_rectangle(
        [left + (right - left) // 5, top, right - (right - left) // 5, top + (bottom - top) // 2],
        radius=50, fill=body, outline=(30, 30, 30), width=6,
    )
    wheel_r = (bottom - top) // 5
    for cx in (left + (right - left) // 5, right - (right - left) // 5):
        draw.ellipse([cx - wheel_r, bottom - wheel_r, cx + wheel_r, bottom + wheel_r], fill=(20, 20, 20), outline=(120, 120, 120), width=8)
    pixels = image.load()
    for _ in range(width * height // 25):
        x, y = rng.randrange(width), rng.randrange(height)
        r, g, b = pixels[x, y]
        delta = rng.randint(-18, 18)
        pixels[x, y] = (max(0, min(255, r + delta)), max(0, min(255, g + delta)), max(0, min(255, b + delta)))
    return image
