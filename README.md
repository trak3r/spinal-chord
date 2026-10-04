# spinal-chord

Identify books in a photograph of a bookshelf. Fully local and free — no API keys
or subscriptions.

**Pipeline:** YOLO-World (detect) → VLM read (OpenRouter if `OPENROUTER_API_KEY` is
set, else local Qwen3-VL-4B) → Open Library (search) → Laya (match) → spine
size/color metadata.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

First run downloads YOLO-World and Laya weights. Local VLM weights are only needed
if you use `--reader local` (or omit `OPENROUTER_API_KEY`).

For faster spine reading, set a free OpenRouter key (uses `qwen/qwen3.8-27b:free`):

```bash
export OPENROUTER_API_KEY=sk-or-...
```

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

### Rainbow shelf order

Print titles sorted by spine hue (red→violet, then neutrals), each line
background-colored to match:

```bash
python -m bookshelf.rainbow samples/bookshelf.json
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
| `--conf` | `0.5` | YOLO detection threshold (lower finds more, more false positives) |
| `--noul-threshold` | `0.55` | Min Laya yes/no probability to accept a match |
| `--match-confidence` | `0.35` | Min Laya choice confidence to accept a match |
| `--max-books` | all | Only process top-N detections (useful for smoke tests) |
| `--reader` | `auto` | `openrouter` / `local` / `auto` (key → OpenRouter) |
| `--vlm-model` | free Qwen on OR | Override OpenRouter or local model id |

## Notes

- Default `--conf 0.5` targets roughly real book counts; lower if books are missed.
- Spine OCR is hard; expect some unmatched crops for manual review.
- Laya verifies Open Library candidates — it does not invent titles.
- OpenRouter free models are rate-limited; use `--reader local` to stay offline.
- Free catalogs do not include publisher spine colors; use photo `spine_color_*` for rainbow sorts.
