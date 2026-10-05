# spinal-chord

Identify books in a photograph of a bookshelf. Fully local and free — no API keys
or subscriptions.

**Pipeline:** YOLO-World (detect) → VLM read (Gemini if `GEMINI_API_KEY` is set,
else OpenRouter if `OPENROUTER_API_KEY`, else local Qwen3-VL-4B) → Open Library
(search) → Laya (match) → spine size/color metadata.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

First run downloads YOLO-World and Laya weights. Local VLM weights are only needed
if you use `--reader local` (or omit cloud API keys).

For faster spine reading, prefer a free Google AI Studio key (default model:
`gemini-3.5-flash-lite`):

```bash
export GEMINI_API_KEY=...
```

Or OpenRouter (default: `google/gemma-4-26b-a4b-it:free`):

```bash
export OPENROUTER_API_KEY=sk-or-...
```

Other free OpenRouter vision options if that endpoint disappears (pass `--vlm-model`):
`google/gemma-4-31b-it:free`, `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`,
or `openrouter/free` (auto-picks a free model that supports images).

Gemini alternatives: `gemini-2.5-flash-lite`, `gemini-2.5-flash`.
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
| `--conf` | `0.2` | YOLO detection threshold (lower finds more thin spines) |
| `--noul-threshold` | `0.55` | Min Laya yes/no probability to accept a match |
| `--match-confidence` | `0.35` | Min Laya choice confidence to accept a match |
| `--max-books` | all | Only process top-N detections (useful for smoke tests) |
| `--reader` | `auto` | `gemini` / `openrouter` / `local` / `auto` (Gemini → OpenRouter → local) |
| `--vlm-model` | cloud default | Override Gemini, OpenRouter, or local model id |

## Notes

- Default `--conf 0.2` plus spine-shape filtering finds thin RPG spines without flooding denser false positives; raise `--conf` if noise appears.
- Spine OCR is hard; expect some unmatched crops for manual review.
- Laya verifies Open Library candidates — it does not invent titles.
- OpenRouter / Gemini free tiers are rate-limited; use `--reader local` to stay offline.
- Free catalogs do not include publisher spine colors; use photo `spine_color_*` for rainbow sorts.
