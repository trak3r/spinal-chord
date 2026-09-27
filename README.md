# spinal-chord

Identify books in a photograph of a bookshelf. Fully local and free — no API keys
or subscriptions.

**Pipeline:** YOLO-World (detect) → Qwen3-VL-4B (read title/author) → Open Library
(search) → Laya (calibrated match decision) → spine size/color metadata.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

First run downloads YOLO-World, Qwen3-VL-4B, and Laya weights (~several GB).
YOLO-World may also auto-install the Ultralytics CLIP dependency on first detection.

Apple Silicon uses MPS for the VLM when available.

## Usage

```bash
python -m bookshelf samples/bookshelf.jpeg
```

Output:

- `<image_stem>.json` — catalog (match + OCR + spine size/color)
- `<image_stem>_crops/` — one crop per detected book (for manual ID of misses)

```bash
python -m bookshelf photo.jpg -o catalog.csv --crops ./crops --conf 0.03
```

### Catalog fields (size & color)

| Field | Source | Notes |
|-------|--------|-------|
| `spine_width_px` / `spine_height_px` | photo bbox | Always present; good for sorting *this* shelf |
| `spine_*_frac` | photo | Fraction of full image (cross-book compare in one shot) |
| `spine_color_hex` / `spine_color_name` | photo crop | Publishers almost never publish spine color |
| `publisher_height_mm` / `width_mm` / `thickness_mm` | Open Library edition | Sparse — often missing |
| `publisher_pages` | Open Library | Common; rough thickness proxy |

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `-o` / `--output` | `<stem>.json` | JSON or CSV (by extension) |
| `--crops` | `<stem>_crops/` | Crop output directory |
| `--conf` | `0.02` | YOLO detection threshold (raise to reduce false positives) |
| `--noul-threshold` | `0.55` | Min Laya yes/no probability to accept a match |
| `--match-confidence` | `0.35` | Min Laya choice confidence to accept a match |
| `--max-books` | all | Only process top-N detections (useful for smoke tests) |

## Notes

- YOLO-World scores for `"book"` prompts are often low (0.01–0.10); `0.02` is intentional.
- Spine OCR is hard; expect many unmatched crops for manual review.
- Laya verifies Open Library candidates — it does not invent titles.
- Open Library needs network access; everything else runs offline after model download.
- Free catalogs do not include publisher spine colors; use photo `spine_color_*` for rainbow sorts.
