"""YOLO-World book detection, cropping, and orientation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from ultralytics import YOLOWorld

_model: YOLOWorld | None = None
DEFAULT_PROMPTS = ["book", "book spine", "book cover"]
# Dense shelves (thin RPG spines) score lower than fat programming tomes.
DEFAULT_CONF = 0.2


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
        return crop.rotate(90, expand=True), "spine"
    if w > h * 1.3:
        return crop, "spine"
    return crop, "face"


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def filter_spine_detections(
    detections: list[Detection],
    image_size: tuple[int, int],
    *,
    min_height_frac: float = 0.25,
    max_aspect: float = 0.4,
    min_width_px: int = 25,
    max_width_frac: float = 0.15,
    iou_thresh: float = 0.45,
) -> list[Detection]:
    """
    Keep upright spine-like boxes; drop shelf junk and near-duplicate overlaps.

    Lower YOLO conf finds thin RPG spines; geometry filtering keeps the
    programming shelf from exploding with false positives.
    """
    img_w, img_h = image_size
    candidates: list[Detection] = []
    for d in detections:
        x1, y1, x2, y2 = d.bbox
        w, h = x2 - x1, y2 - y1
        spine_w, spine_h = min(w, h), max(w, h)
        if spine_h < img_h * min_height_frac:
            continue
        if spine_w < min_width_px:
            continue
        if spine_w > img_w * max_width_frac:
            continue
        if spine_w / spine_h > max_aspect:
            continue
        candidates.append(d)

    candidates.sort(key=lambda d: d.confidence, reverse=True)
    kept: list[Detection] = []
    for d in candidates:
        x1, y1, x2, y2 = d.bbox
        cx = (x1 + x2) / 2
        width = x2 - x1
        ok = True
        for k in kept:
            if _iou(d.bbox, k.bbox) > iou_thresh:
                ok = False
                break
            kx1, _, kx2, _ = k.bbox
            kcx = (kx1 + kx2) / 2
            # Near-duplicate vertical strips (same spine, two prompts).
            if abs(cx - kcx) < 0.4 * min(width, kx2 - kx1) and _iou(d.bbox, k.bbox) > 0.12:
                ok = False
                break
        if ok:
            kept.append(d)
    return kept


def detect_books(
    image: Image.Image | Path | str,
    *,
    conf: float = DEFAULT_CONF,
    weights: str = "yolov8m-worldv2.pt",
    filter_spines: bool = True,
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

    if filter_spines:
        detections = filter_spine_detections(detections, image.size)

    # Dense, stable indices left→right (shelf order).
    detections.sort(key=lambda d: (d.bbox[0] + d.bbox[2]) / 2)
    for i, d in enumerate(detections):
        d.index = i
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
