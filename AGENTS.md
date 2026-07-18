# spinal_chord

One file (`bookshelf.py`): **YOLO-World (yolov8m-worldv2, text prompt "book cover") →
crop → OCR (Tesseract, multiple preprocessing strategies) → OpenLibrary search →
catalog JSON/CSV**.

- Detection uses YOLO-World (CLIP-guided, open-vocabulary) prompted with "book cover",
  which works for both book spines and face-up books. Model is cached globally after
  first inference (~400ms/image on M-series Mac).
- Replaced Roboflow API (deprecated: credits exhausted) and Google Books API (rate-limited).
- False-positive filter in `_verify()` requires 2+ word overlap between OCR and result
  title (Unicode-aware, strips diacritics for matching).
- Orientation (spine vs face) is auto-detected per crop by aspect ratio; no `--mode` flag.
- Default confidence threshold is 0.02 (low to handle YOLO-World's confidence distribution).

## Known limitations

- YOLO-World confidence is typically 0.01–0.10 for "book cover" prompts, so the default
  threshold is very low. At 0.02 you may get ~2 false positives per bookshelf image.
  Raise with `--conf 0.03` or higher if noise is an issue.
- Book spine/cover OCR is inherently unreliable (curved surfaces, gold/shiny text, shadows,
  glare on flat covers). Typically identifies ~30–50% of books automatically.
- The rest produce crops for manual identification.
- No tests, no CI, no linter config.

## Legacy

The original `spines.ipynb` used Roboflow + Google Books API and is kept for reference
but is non-functional (both services lack valid credentials/credits).
