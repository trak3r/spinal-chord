#!/usr/bin/env python3
"""Detect books in a bookshelf photo, OCR their spines, and identify via OpenLibrary API.

Usage:
  python bookshelf.py photo.jpg
  python bookshelf.py photo.jpg -o catalog.json
  python bookshelf.py photo.jpg -o catalog.csv --crops ./crops
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import certifi
import numpy as np
import pytesseract
import requests
import ssl
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from ultralytics import YOLO

os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()
ssl._create_default_https_context = ssl.create_default_context

# ── data ────────────────────────────────────────────────────────────────────

@dataclass
class Book:
    title: str = ""
    author: str = ""
    confidence: str = "low"
    isbn: Optional[str] = None
    published: Optional[str] = None
    ocr_text: str = ""
    detection_score: float = 0.0
    position: int = 0


# ── detection ───────────────────────────────────────────────────────────────

_MODEL = None

def detect_books(image_path: str, conf_threshold: float = 0.02) -> list:
    """Detect books using YOLO-World (text-prompted, open-vocabulary).
    Returns list of (bbox, confidence), auto-sorted by orientation.
    """
    global _MODEL
    if _MODEL is None:
        _MODEL = YOLO("yolov8m-worldv2.pt")
        _MODEL.set_classes(["book cover"])
    results = _MODEL.predict(image_path, conf=conf_threshold)
    books = []
    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf >= conf_threshold:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            w, h = x2 - x1, y2 - y1
            if w * h > 10000:
                books.append(((x1, y1, x2, y2), conf))
    if books:
        aspects = [(x2 - x1) / max(y2 - y1, 1) for (x1, y1, x2, y2), _ in books]
        median = sorted(aspects)[len(aspects) // 2]
        if median < 0.6:
            books.sort(key=lambda b: b[0][0])  # left-to-right (spines)
        else:
            books.sort(key=lambda b: (b[0][1], b[0][0]))  # top-to-bottom, left-to-right
    return books


# ── OCR ─────────────────────────────────────────────────────────────────────

def _ocr_strategies(img: Image.Image, rotate: bool = True) -> list[tuple[str, float]]:
    """Try multiple OCR approaches and return (text, quality_score) pairs."""
    results = []
    r = img.rotate(90, expand=True) if rotate else img
    gray = r.convert("L")
    arr = np.array(gray, dtype=np.uint8)

    recipes = [
        ("contrast+sharpen", lambda: (
            ImageEnhance.Contrast(gray).enhance(2.0).filter(ImageFilter.SHARPEN))),
        ("histogram stretch", lambda: (
            Image.fromarray(np.clip((arr.astype(float) - np.percentile(arr, 5)) /
                                     (np.percentile(arr, 95) - np.percentile(arr, 5) + 1) * 255,
                                     0, 255).astype(np.uint8)))),
        ("equalize", lambda: ImageOps.equalize(gray)),
    ]
    if np.mean(arr) < 127:
        recipes.append(("inverted", lambda: Image.fromarray(255 - arr)))

    for label, fn in recipes:
        try:
            text = pytesseract.image_to_string(fn(), config="--psm 3 --oem 3")
            results.append((text.strip(), _score_text(text)))
        except Exception:
            pass

    results.sort(key=lambda x: -x[1])
    return results


def _score_text(text: str) -> float:
    """Score OCR quality — prefers mostly-alpha words and long capitalized phrases."""
    words = text.strip().split()
    if not words:
        return 0
    clean_upper = sum(1 for w in words if len(w) > 2 and w[0].isupper() and w.isalpha())
    all_alpha = sum(1 for w in words if w.isalpha())
    upper_ratio = clean_upper / max(len(words), 1)
    alpha_ratio = all_alpha / max(len(words), 1)
    length_bonus = min(len(text) / 300, 1.0)
    return upper_ratio * 0.5 + alpha_ratio * 0.3 + length_bonus * 0.2


def extract_texts(spine_img: Image.Image, rotate: bool = True) -> list[str]:
    """Return deduplicated OCR texts from all strategies."""
    seen = set()
    texts = []
    for text, score in _ocr_strategies(spine_img, rotate=rotate):
        t = re.sub(r"\s+", " ", text).strip()
        if t and t not in seen:
            seen.add(t)
            texts.append(t)
    return texts if texts else [""]


# ── search ──────────────────────────────────────────────────────────────────

STOP = {"the", "and", "for", "with", "from", "that", "this", "edition",
        "design", "implementation", "guide", "introduction", "second",
        "systems", "computer", "programming", "principles", "using",
        "software", "data", "book", "new", "volume", "vol"}

PUBLISHERS = {"addison", "wesley", "prentice", "hall", "oreilly", "mcgraw",
              "hill", "morgan", "kaufmann", "mit", "press", "springer"}

_last_req = 0.0


def _search(query: str) -> Optional[dict]:
    """Search OpenLibrary. Returns first result doc or None."""
    global _last_req
    elapsed = time.time() - _last_req
    if elapsed < 0.5:
        time.sleep(0.5 - elapsed)
    _last_req = time.time()
    try:
        resp = requests.get(
            "https://openlibrary.org/search.json",
            params={"q": query, "limit": 3},
            timeout=10,
        )
        if resp.status_code == 200:
            docs = resp.json().get("docs", [])
            return docs[0] if docs else None
    except Exception:
        return None
    return None


def _norm(w: str) -> str:
    """Strip diacritics and lowercase."""
    nfkd = unicodedata.normalize("NFKD", w)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


def _words(text: str) -> list[str]:
    """Extract words (including Unicode letters), 4+ chars."""
    pattern = re.compile(r"[^\W\d_]{4,}", re.UNICODE)
    return pattern.findall(text)


def _norm_words(text: str) -> set:
    """Words 4+ chars from text, normalized for comparison."""
    return {_norm(w) for w in _words(text)}


def _verify(doc: dict, query: str) -> bool:
    """Returns True if the search result is a plausible match for the query."""
    q_words = _norm_words(query)
    if not q_words:
        return False
    title = doc.get("title", "") or ""
    t_words = _norm_words(title)
    overlap = q_words & t_words
    significant = overlap - STOP - PUBLISHERS
    return len(significant) >= 2 or (len(significant) == 1 and len(next(iter(significant))) >= 7)


def _queries_from_text(text: str) -> list[str]:
    """Generate search queries from noisy OCR text. Best candidates first."""
    texts = []

    # ISBN
    m = re.search(r"(?:^|\s)(?:ISBN[:\s]*)?(\d{13}|\d{9,10}[\dXx])(?:\s|$)", text, re.IGNORECASE)
    if m:
        texts.append(m.group(1))
        return texts

    clean = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    if len(clean) < 4:
        return texts

    words = clean.split()
    # Keep only words that look like real words (mostly alpha, no embedded digits)
    good = [w for w in words if len(w) >= 2 and
            not any(c.isdigit() for c in w) and
            sum(c.isalpha() for c in w) >= len(w) * 0.6]

    if not good:
        return texts

    connectors = {"and", "of", "the", "for", "with", "from", "that", "this"}

    # Extract capitalized phrases (these are most likely title words)
    phrases = []
    cur = []
    for w in good:
        if w[0].isupper() or w.lower() in connectors:
            if len(w) >= 3:
                cur.append(w)
        else:
            if len(cur) >= 2:
                phrases.append(" ".join(cur))
            cur = []
    if len(cur) >= 2:
        phrases.append(" ".join(cur))

    texts.extend(phrases)

    # Also add sub-phrases for long ones
    for p in phrases:
        parts = p.split()
        if len(parts) > 3:
            texts.append(" ".join(parts[-3:]))

    # Also try all capitalized words strung together
    caps = [w for w in good if w[0].isupper() and len(w) >= 3]
    if caps:
        texts.append(" ".join(caps))
        if len(caps) > 3:
            texts.append(" ".join(caps[-4:]))

    # Deduplicate preserving order; remove phrases with short/garbage words
    def _clean(q: str) -> Optional[str]:
        parts = q.split()
        filtered = [p for p in parts if len(p) >= 3]
        return " ".join(filtered) if len(filtered) >= 2 else None

    seen = set()
    result = []
    for t in texts:
        t = _clean(t)
        if t and t not in seen:
            seen.add(t)
            result.append(t)
    return result


def _book_from_doc(doc: dict, ocr_text: str) -> Book:
    b = Book(ocr_text=ocr_text)
    b.title = doc.get("title", "Unknown")
    authors = doc.get("author_name", [])
    b.author = ", ".join(authors[:3]) if authors else ""
    b.confidence = "high"
    isbns = doc.get("isbn", [])
    b.isbn = isbns[0] if isbns else None
    pub = doc.get("first_publish_year")
    b.published = str(pub) if pub else None
    return b


def identify(texts: list[str], position: int) -> Book:
    """Try OCR texts as search queries. Returns first verified match or no-match."""
    seen = set()
    for text in texts:
        for query in _queries_from_text(text):
            if query in seen:
                continue
            seen.add(query)
            doc = _search(query)
            if doc and _verify(doc, query):
                return _book_from_doc(doc, text)

    best = next((t for t in texts if t.strip()), "")
    if not best:
        return Book(title="(OCR returned no text)", position=position)
    return Book(title="(no match)", ocr_text=best, position=position, confidence="low")


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Identify books from a bookshelf photo")
    ap.add_argument("image", help="Path to the photo")
    ap.add_argument("-o", "--output", help="Output file (.json or .csv)")
    ap.add_argument("--conf", type=float, default=0.02,
                    help="Detection confidence threshold (default 0.02)")
    ap.add_argument("--crops", help="Crop output directory (default: <image stem>_crops)")
    args = ap.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: {args.image} not found", file=sys.stderr)
        sys.exit(1)

    crops_dir = args.crops or Path(args.image).stem + "_crops"
    os.makedirs(crops_dir, exist_ok=True)

    print(f"Scanning {args.image} ...")
    detections = detect_books(args.image, args.conf)
    print(f"Detected {len(detections)} books\n")
    if not detections:
        print("No books found. Try lowering --conf (e.g. --conf 0.2).")
        sys.exit(0)

    results = []
    for i, (bbox, score) in enumerate(detections, 1):
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        is_spine = (w / max(h, 1)) < 0.6
        label = "spine" if is_spine else "face"
        print(f"── Book {i} ({label}, detection: {score:.2f}) ──")
        spine = Image.open(args.image).crop(bbox)

        spine.save(os.path.join(crops_dir, f"book_{i:03d}.jpg"))

        texts = extract_texts(spine, rotate=is_spine)
        if texts[0]:
            print(f"  OCR: {texts[0][:100]}")
        else:
            print(f"  OCR: (empty)")

        book = identify(texts, i)
        book.detection_score = score
        tag = {"high": "✓", "low": "·"}.get(book.confidence, "·")
        line = f"  {tag} [{book.confidence}] {book.title}"
        if book.author and book.author != "Unknown":
            line += f" — {book.author}"
        print(line)
        results.append(book)

    # Output
    data = [asdict(b) for b in results]
    path = args.output or Path(args.image).with_suffix(".json")
    ext = Path(path).suffix.lower() or ".json"
    path = str(path)
    if not Path(path).suffix:
        path += ".json"
        ext = ".json"
    if ext == ".json":
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    elif ext == ".csv":
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
    print(f"\nSaved catalog to {path}")

    # Summary
    found = sum(1 for b in results if b.confidence == "high")
    print(f"\n{'=' * 50}")
    print(f"  {found}/{len(results)} books identified  (crops in {crops_dir})")
    print(f"{'=' * 50}")
    for b in results:
        t = "✓" if b.confidence == "high" else " "
        print(f"  {t}  {b.title}")
        if b.author:
            print(f"      {b.author}")


if __name__ == "__main__":
    main()
