# spinal_chord

Detects books in a photo using YOLO-World (text-prompted, open-vocabulary), OCRs
each book with Tesseract, and identifies them via the OpenLibrary API. Works with
both bookshelf (spine) and top-down (face) photos automatically.

## Quick start

```bash
source .venv/bin/activate
pip install -r requirements.txt
python bookshelf.py photo.jpg
```

Output: `<photo>.json` + a `<photo>_crops/` directory with one cropped image per book.

## System dependency

- **Tesseract OCR** — `brew install tesseract` on macOS; `pytesseract` talks to it

## CLI

```
python bookshelf.py <image> [-o output.json|csv] [--conf 0.02] [--crops DIR]
```

- `--conf`: detection confidence threshold (default 0.02). Lower values find more
  books but may include false positives; raise to reduce noise.
- `--crops`: where to save per-book crop images (default: `<image_stem>_crops/`).
  Saved automatically; useful for manual identification of unmatched books.
- `--output`: saves catalog as JSON or CSV (detected by extension). Defaults to
  `<input_stem>.json` if omitted.

No `--mode` flag needed — orientation (spine vs face) is auto-detected per crop.
