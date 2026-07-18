# spinal_chord

Detects books in a bookshelf photo via YOLOv11 (local, no API key needed), OCRs each
spine with Tesseract, and identifies books via the OpenLibrary API.

## Quick start

```bash
source .venv/bin/activate
pip install -r requirements.txt
python bookshelf.py photo.jpg
```

Output: `<photo>.json` + a `<photo>_crops/` directory with one cropped spine per book.

## System dependency

- **Tesseract OCR** — `brew install tesseract` on macOS; `pytesseract` talks to it

## CLI

```
python bookshelf.py <image> [-o output.json|csv] [--conf 0.3] [--crops DIR] [--mode spine|face]
```

- `--conf`: YOLO detection threshold (default 0.3). Raise to ~0.35 to reduce false
  positives; lower to find more books.
- `--crops`: where to save per-book spine images (default: `<image_stem>_crops/`).
  Saved automatically; useful for manual identification of unmatched books.
- `--output`: saves catalog as JSON or CSV (detected by extension). Defaults to
  `<input_stem>.json` if omitted.
- `--mode`: `spine` (default) for bookshelf photos; `face` for top-down photos of
  books laid flat. In face mode, the aspect-ratio filter is relaxed, OCR skips the
  90° rotation, and books are sorted top-to-bottom, left-to-right.
