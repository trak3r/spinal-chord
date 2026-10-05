"""Spine appearance from the photo crop (size in px + dominant color)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from PIL import Image


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
    # Simple HSV-ish hue from RGB
    rn, gn, bn = r / 255.0, g / 255.0, b / 255.0
    mx_f, mn_f = max(rn, gn, bn), min(rn, gn, bn)
    delta = mx_f - mn_f
    val = mx_f
    if delta == 0:
        hue, sat = 0.0, 0.0
    else:
        if mx_f == rn:
            h = ((gn - bn) / delta) % 6
        elif mx_f == gn:
            h = (bn - rn) / delta + 2
        else:
            h = (rn - gn) / delta + 4
        hue = h * 60
        sat = delta / mx_f
    if val < 0.22:
        return "black"
    # Low-saturation: ivory / chrome / gray — not warm "orange" from lighting.
    if sat < 0.22:
        if val >= 0.75:
            return "white"
        if val >= 0.23:
            return "gray"  # includes silver / chrome
        return "black"
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


def _quantize(r: int, g: int, b: int, step: int = 24) -> tuple[int, int, int]:
    """Bucket RGB so stripes/noise don't fragment the mode."""
    return (
        min(255, (r // step) * step + step // 2),
        min(255, (g // step) * step + step // 2),
        min(255, (b // step) * step + step // 2),
    )


def _dominant_color(crop: Image.Image) -> tuple[int, int, int]:
    """
    Spine *field* color by area, not lettering or a center stripe.

    Vote by coarse color name (orange/red/gray/…) across quantized buckets,
    then take the mode bucket inside the winning name. That keeps chrome
    (Showstopper) gray and cream (Refactoring) white, while thick gray title
    text on a brick spine (Monster Overhaul) cannot outvote the red field —
    and dark ink on mustard AD&D spines cannot tip the vote to brown.
    """
    img = crop.convert("RGB")
    # Bound work; scale by long side so tall unoriented spines aren't crushed
    # to a 3×64 smear (old short-side cap of 64).
    w, h = img.size
    long, short = max(w, h), min(w, h)
    scale = min(1.0, 160 / max(long, 1))
    if short > 0 and short * scale < 8:
        scale = min(1.0, 8 / short)
    if scale < 1.0:
        img = img.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.Resampling.BILINEAR,
        )

    pixels = list(img.getdata())
    if not pixels:
        return 128, 128, 128

    dark = [p for p in pixels if max(p) < 50]
    dark_frac = len(dark) / len(pixels)

    if dark_frac >= 0.55:
        # Mostly black/navy spine — keep dark pixels.
        pool = dark if dark else pixels
    else:
        # Drop near-black ink/shadows so they don't fragment the field vote.
        pool = [p for p in pixels if max(p) >= 50]
        if len(pool) < max(20, len(pixels) // 20):
            pool = pixels

    counts: Counter[tuple[int, int, int]] = Counter(
        _quantize(r, g, b) for r, g, b in pool
    )
    if not counts:
        return 128, 128, 128

    by_name: Counter[str] = Counter()
    buckets_for: dict[str, Counter[tuple[int, int, int]]] = {}
    for color, n in counts.items():
        name = _coarse_color_name(*color)
        by_name[name] += n
        buckets_for.setdefault(name, Counter())[color] += n

    best_name = by_name.most_common(1)[0][0]
    return buckets_for[best_name].most_common(1)[0][0]


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
