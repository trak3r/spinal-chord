"""Spine appearance from the photo crop (size in px + dominant color)."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageStat


@dataclass
class Appearance:
    """Visual traits of the detected book in the photo."""

    spine_width_px: int
    spine_height_px: int
    # 0–1 relative to full image (useful for sorting within one shelf photo)
    spine_width_frac: float
    spine_height_frac: float
    color_hex: str
    color_rgb: tuple[int, int, int]
    color_name: str  # coarse bucket for sorting
    color_source: str  # always "photo" for now


def _bbox_spine_px(
    bbox: tuple[int, int, int, int],
    orientation: str,
) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    w, h = abs(x2 - x1), abs(y2 - y1)
    if orientation == "face":
        return w, h
    # Spines are tall and thin in the photo before OCR rotation.
    return min(w, h), max(w, h)


def _coarse_color_name(r: int, g: int, b: int) -> str:
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx < 40:
        return "black"
    if mn > 220:
        return "white"
    if mx - mn < 25:
        return "gray" if mx < 180 else "white"
    # Simple HSV-ish hue from RGB
    rn, gn, bn = r / 255.0, g / 255.0, b / 255.0
    mx_f, mn_f = max(rn, gn, bn), min(rn, gn, bn)
    d = mx_f - mn_f or 1.0
    if mx_f == rn:
        h = ((gn - bn) / d) % 6
    elif mx_f == gn:
        h = (bn - rn) / d + 2
    else:
        h = (rn - gn) / d + 4
    hue = h * 60
    sat = d / mx_f if mx_f else 0
    val = mx_f
    if val < 0.25:
        return "black"
    if sat < 0.15:
        return "gray" if val < 0.85 else "white"
    # Dark warm neutrals → brown; cool darks keep their hue family.
    if val < 0.45 and sat < 0.45 and 15 <= hue < 70:
        return "brown"
    if hue < 20 or hue >= 340:
        return "red"
    if hue < 45:
        return "orange" if val > 0.45 else "brown"
    if hue < 70:
        return "yellow" if val > 0.55 else "brown"
    if hue < 160:
        return "green"
    if hue < 260:
        return "blue"
    if hue < 320:
        return "purple"
    return "red"


def _dominant_color(crop: Image.Image) -> tuple[int, int, int]:
    """Median RGB of a center band (avoids shelf background at edges)."""
    img = crop.convert("RGB")
    w, h = img.size
    # Sample the middle 50% — for oriented spines this is the face of the spine.
    left, right = int(w * 0.25), int(w * 0.75)
    top, bottom = int(h * 0.2), int(h * 0.8)
    if right <= left or bottom <= top:
        band = img
    else:
        band = img.crop((left, top, right, bottom))
    # Downscale for speed
    band = band.resize((max(1, band.width // 4), max(1, band.height // 4)))
    stat = ImageStat.Stat(band)
    # median is more robust than mean for text on colored spines
    med = stat.median
    return int(med[0]), int(med[1]), int(med[2])


def analyze_appearance(
    crop: Image.Image,
    bbox: tuple[int, int, int, int],
    orientation: str,
    image_size: tuple[int, int],
) -> Appearance:
    """Measure spine size (px) and approximate color from the photo crop."""
    img_w, img_h = image_size
    width_px, height_px = _bbox_spine_px(bbox, orientation)
    r, g, b = _dominant_color(crop)
    return Appearance(
        spine_width_px=width_px,
        spine_height_px=height_px,
        spine_width_frac=round(width_px / img_w, 4) if img_w else 0.0,
        spine_height_frac=round(height_px / img_h, 4) if img_h else 0.0,
        color_hex=f"#{r:02x}{g:02x}{b:02x}",
        color_rgb=(r, g, b),
        color_name=_coarse_color_name(r, g, b),
        color_source="photo",
    )
