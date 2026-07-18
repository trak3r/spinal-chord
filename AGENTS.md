# spinal_chord

Detects books in a bookshelf photo via YOLOv11 (local, no API key needed), OCRs each
spine with Tesseract, and identifies books via the OpenLibrary API.

## Quick start

```bash
source .venv/bin/activate
pip install -r requirements.txt
python bookshelf.py photo.jpg -o catalog.json
```

Output: `catalog.json` + a `photo_crops/` directory with one cropped spine per book.

## System dependency

- **Tesseract OCR** — `brew install tesseract` on macOS; `pytesseract` talks to it

## CLI

```
python bookshelf.py <image> [-o output.json|csv] [--conf 0.3] [--crops DIR]
```

- `--conf`: YOLO detection threshold (default 0.3). Raise to ~0.35 to reduce false
  positives; lower to find more books.
- `--crops`: where to save per-book spine images (default: `<image_stem>_crops/`).
  Saved automatically; useful for manual identification of unmatched books.
- `--output`: saves catalog as JSON or CSV (detected by extension).

## Architecture

One file (`bookshelf.py`): **YOLOv11 (COCO class 73) → crop → OCR (Tesseract, multiple
preprocessing strategies) → OpenLibrary search → catalog JSON/CSV**.

- Replaced Roboflow API (deprecated: credits exhausted). YOLO runs locally in ~50ms.
- Replaced Google Books API (rate-limited) with OpenLibrary (free, no auth).
- False-positive filter in `_verify()` requires 2+ word overlap between OCR and result
  title (Unicode-aware, strips diacritics for matching).

## Known limitations

- Book spine OCR is inherently unreliable (curved surfaces, gold/shiny text, shadows).
  Typically identifies ~30–50% of spines automatically.
- The rest produce crops for manual identification.
- No tests, no CI, no linter config.

## Legacy

The original `spines.ipynb` used Roboflow + Google Books API and is kept for reference
but is non-functional (both services lack valid credentials/credits).
