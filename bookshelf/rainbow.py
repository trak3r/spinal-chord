"""Print catalog titles in rainbow spine-color order with colored backgrounds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _rgb(book: dict) -> tuple[int, int, int]:
    rgb = book.get("spine_color_rgb")
    if isinstance(rgb, (list, tuple)) and len(rgb) >= 3:
        return int(rgb[0]), int(rgb[1]), int(rgb[2])
    hex_color = book.get("spine_color_hex") or "#808080"
    h = hex_color.lstrip("#")
    if len(h) == 6:
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return 128, 128, 128


def _hsv(r: int, g: int, b: int) -> tuple[float, float, float]:
    rn, gn, bn = r / 255.0, g / 255.0, b / 255.0
    mx, mn = max(rn, gn, bn), min(rn, gn, bn)
    d = mx - mn
    if d == 0:
        hue = 0.0
    elif mx == rn:
        hue = (60 * ((gn - bn) / d) + 360) % 360
    elif mx == gn:
        hue = 60 * ((bn - rn) / d) + 120
    else:
        hue = 60 * ((rn - gn) / d) + 240
    sat = 0.0 if mx == 0 else d / mx
    return hue, sat, mx


def _luminance(r: int, g: int, b: int) -> float:
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0


def _title(book: dict) -> str:
    return (book.get("title") or book.get("ocr_title") or "(untitled)").strip()


def _sort_key(book: dict) -> tuple:
    """Rainbow: chromatic by hue, then neutrals by dark→light."""
    r, g, b = _rgb(book)
    hue, sat, val = _hsv(r, g, b)
    # Near-black / gray spines sit with neutrals, not mid-hue.
    neutral = sat < 0.2 or val < 0.2
    if neutral:
        return (1, _luminance(r, g, b), _title(book).lower())
    return (0, hue, _title(book).lower())


def _paint_line(text: str, r: int, g: int, b: int) -> str:
    fg = (0, 0, 0) if _luminance(r, g, b) > 0.55 else (255, 255, 255)
    # Pad so the background reads as a full-width color chip
    padded = f"  {text}  "
    return (
        f"\033[48;2;{r};{g};{b}m\033[38;2;{fg[0]};{fg[1]};{fg[2]}m"
        f"{padded}\033[0m"
    )


def rainbow_lines(books: list[dict]) -> list[str]:
    ordered = sorted(books, key=_sort_key)
    lines: list[str] = []
    for book in ordered:
        r, g, b = _rgb(book)
        lines.append(_paint_line(_title(book), r, g, b))
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Print book titles from a catalog JSON in rainbow spine-color order, "
            "with each line's background set to that spine color."
        )
    )
    parser.add_argument(
        "catalog",
        type=Path,
        help="Path to bookshelf catalog JSON (from python -m bookshelf)",
    )
    args = parser.parse_args(argv)

    path = args.catalog.expanduser().resolve()
    if not path.is_file():
        print(f"error: catalog not found: {path}", file=sys.stderr)
        return 1

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        print(f"error: invalid JSON: {e}", file=sys.stderr)
        return 1

    if not isinstance(data, list):
        print("error: catalog must be a JSON array of books", file=sys.stderr)
        return 1

    for line in rainbow_lines(data):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
