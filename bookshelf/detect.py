"""YOLO-World book detection, cropping, and orientation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from ultralytics import YOLOWorld

_model: YOLOWorld | None = None
DEFAULT_PROMPTS = ["book", "book spine", "book cover"]


@dataclass
class Detection:
    """One detected book region."""

    index: int
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    crop: Image.Image
    orientation: str  # "spine" | "face"


def _get_model(weights: str = "yolov8m-worldv2.pt") -> YOLOWorld:
    global _model
    if _model is None:
        _model = YOLOWorld(weights)
        _model.set_classes(DEFAULT_PROMPTS)
    return _model


def _orient(crop: Image.Image) -> tuple[Image.Image, str]:
    """Rotate tall/thin spines so text reads horizontally; leave face-up covers alone."""
    w, h = crop.size
    if h > w * 1.3:
        # Vertical spine — try both 90° directions; prefer wider result for OCR later.
        # Spines are usually taller than wide; rotate to landscape.
        return crop.rotate(90, expand=True), "spine"
    if w > h * 1.3:
        return crop, "spine"
    return crop, "face"


def detect_books(
    image: Image.Image | Path | str,
    *,
    conf: float = 0.02,
    weights: str = "yolov8m-worldv2.pt",
) -> list[Detection]:
    """Detect books in an image and return oriented crops."""
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    model = _get_model(weights)
    results = model.predict(image, conf=conf, verbose=False)
    if not results:
        return []

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    detections: list[Detection] = []
    for i, box in enumerate(boxes):
        xyxy = box.xyxy[0].tolist()
        x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.width, x2), min(image.height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image.crop((x1, y1, x2, y2))
        oriented, orientation = _orient(crop)
        score = float(box.conf[0]) if box.conf is not None else 0.0
        detections.append(
            Detection(
                index=i,
                bbox=(x1, y1, x2, y2),
                confidence=score,
                crop=oriented,
                orientation=orientation,
            )
        )
    return detections


def save_crops(detections: list[Detection], out_dir: Path | str) -> list[Path]:
    """Save each detection crop as crop_NNN.jpg. Returns paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for d in detections:
        path = out / f"crop_{d.index:03d}.jpg"
        d.crop.save(path, quality=95)
        paths.append(path)
    return paths
