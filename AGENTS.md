# spinal_chord

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
