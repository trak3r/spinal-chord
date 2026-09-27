"""CLI and end-to-end pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from bookshelf.appearance import analyze_appearance
from bookshelf.decide import verify_match
from bookshelf.detect import detect_books, save_crops
from bookshelf.lookup import fetch_publisher_size, search_books
from bookshelf.read import read_book


@dataclass
class BookResult:
    index: int
    bbox: list[int]
    detect_confidence: float
    orientation: str
    crop_path: str | None
    ocr_title: str
    ocr_author: str
    matched: bool
    title: str | None
    author: str | None
    year: int | None
    openlibrary_key: str | None
    choice_id: str
    choice_confidence: float
    confident_noul: float
    # Photo-measured (for sorting this shelf)
    spine_width_px: int | None = None
    spine_height_px: int | None = None
    spine_width_frac: float | None = None
    spine_height_frac: float | None = None
    spine_color_hex: str | None = None
    spine_color_rgb: list[int] | None = None
    spine_color_name: str | None = None
    spine_color_source: str | None = None
    # Catalog-measured when Open Library has it (publisher-ish; often missing)
    publisher_height_mm: float | None = None
    publisher_width_mm: float | None = None
    publisher_thickness_mm: float | None = None
    publisher_pages: int | None = None
    publisher_dimensions_raw: str | None = None
    publisher_format: str | None = None
    publisher_edition_key: str | None = None
    publisher_size_source: str | None = None


def identify_image(
    image_path: Path,
    *,
    crops_dir: Path,
    conf: float = 0.02,
    noul_threshold: float = 0.55,
    conf_threshold: float = 0.35,
    max_books: int | None = None,
) -> list[BookResult]:
    image = Image.open(image_path).convert("RGB")
    detections = detect_books(image, conf=conf)
    detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
    if max_books is not None:
        detections = detections[: max(0, max_books)]
    for i, det in enumerate(detections):
        det.index = i
    crop_paths = save_crops(detections, crops_dir)
    print(f"Detected {len(detections)} book(s)", file=sys.stderr)

    results: list[BookResult] = []
    for det, crop_path in zip(detections, crop_paths):
        appearance = analyze_appearance(
            det.crop, det.bbox, det.orientation, image.size
        )

        print(f"[{det.index}] reading crop…", file=sys.stderr)
        reading = read_book(det.crop)
        print(
            f"[{det.index}] OCR: {reading.title!r} / {reading.author!r}",
            file=sys.stderr,
        )

        candidates = search_books(reading.title, reading.author)
        decision = verify_match(
            reading.title,
            reading.author,
            candidates,
            noul_threshold=noul_threshold,
            conf_threshold=conf_threshold,
        )

        cand = decision.candidate
        pub = None
        if cand is not None:
            pub = fetch_publisher_size(cand)

        results.append(
            BookResult(
                index=det.index,
                bbox=list(det.bbox),
                detect_confidence=det.confidence,
                orientation=det.orientation,
                crop_path=str(crop_path),
                ocr_title=reading.title,
                ocr_author=reading.author,
                matched=decision.matched,
                title=cand.title if cand else None,
                author=cand.author if cand else None,
                year=cand.year if cand else None,
                openlibrary_key=cand.openlibrary_key if cand else None,
                choice_id=decision.choice_id,
                choice_confidence=decision.choice_confidence,
                confident_noul=decision.confident_noul,
                spine_width_px=appearance.spine_width_px,
                spine_height_px=appearance.spine_height_px,
                spine_width_frac=appearance.spine_width_frac,
                spine_height_frac=appearance.spine_height_frac,
                spine_color_hex=appearance.color_hex,
                spine_color_rgb=list(appearance.color_rgb),
                spine_color_name=appearance.color_name,
                spine_color_source=appearance.color_source,
                publisher_height_mm=pub.height_mm if pub else None,
                publisher_width_mm=pub.width_mm if pub else None,
                publisher_thickness_mm=pub.thickness_mm if pub else None,
                publisher_pages=pub.pages if pub else None,
                publisher_dimensions_raw=pub.raw_dimensions if pub else None,
                publisher_format=pub.physical_format if pub else None,
                publisher_edition_key=pub.edition_key if pub else None,
                publisher_size_source=pub.source if pub else None,
            )
        )
        status = "MATCH" if decision.matched else "miss"
        label = f"{cand.title} — {cand.author}" if cand else "(unmatched)"
        color = appearance.color_name
        size_note = (
            f"{pub.thickness_mm}×{pub.height_mm}mm"
            if pub and pub.thickness_mm and pub.height_mm
            else (f"{pub.pages}p" if pub and pub.pages else "no pub size")
        )
        print(
            f"[{det.index}] {status}: {label} | spine {color} {appearance.color_hex} | {size_note}",
            file=sys.stderr,
        )

    return results


def write_json(results: list[BookResult], path: Path) -> None:
    path.write_text(json.dumps([asdict(r) for r in results], indent=2) + "\n")


def write_csv(results: list[BookResult], path: Path) -> None:
    rows = [asdict(r) for r in results]
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Identify books in a bookshelf photograph (local, free tools)."
    )
    parser.add_argument("image", type=Path, help="Path to bookshelf photo")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON or CSV path (default: <image_stem>.json)",
    )
    parser.add_argument(
        "--crops",
        type=Path,
        default=None,
        help="Directory for per-book crops (default: <image_stem>_crops/)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.02,
        help="YOLO-World detection confidence threshold (default: 0.02)",
    )
    parser.add_argument(
        "--noul-threshold",
        type=float,
        default=0.55,
        help="Minimum Laya noul to accept a match (default: 0.55)",
    )
    parser.add_argument(
        "--match-confidence",
        type=float,
        default=0.35,
        help="Minimum Laya choice confidence to accept a match (default: 0.35)",
    )
    parser.add_argument(
        "--max-books",
        type=int,
        default=None,
        help="Process only the top-N detections by confidence (default: all)",
    )
    args = parser.parse_args(argv)

    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        print(f"error: image not found: {image_path}", file=sys.stderr)
        return 1

    stem = image_path.stem
    out_path = (args.output or image_path.with_suffix(".json")).expanduser().resolve()
    crops_dir = (
        args.crops or image_path.parent / f"{stem}_crops"
    ).expanduser().resolve()

    results = identify_image(
        image_path,
        crops_dir=crops_dir,
        conf=args.conf,
        noul_threshold=args.noul_threshold,
        conf_threshold=args.match_confidence,
        max_books=args.max_books,
    )

    if out_path.suffix.lower() == ".csv":
        write_csv(results, out_path)
    else:
        write_json(results, out_path)

    matched = sum(1 for r in results if r.matched)
    print(
        f"Done: {matched}/{len(results)} matched → {out_path} (crops: {crops_dir})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
