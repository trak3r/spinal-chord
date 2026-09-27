"""Open Library catalog search + edition physical metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass

import requests

SEARCH_URL = "https://openlibrary.org/search.json"
USER_AGENT = "spinal-chord/0.1 (bookshelf identifier; local personal use)"


@dataclass
class Candidate:
    id: str
    title: str
    author: str
    year: int | None
    openlibrary_key: str
    edition_keys: list[str]
    pages_median: int | None
    cover_id: int | None


@dataclass
class PublisherSize:
    """Physical size from catalog (Open Library edition), when available."""

    height_mm: float | None
    width_mm: float | None
    thickness_mm: float | None  # spine depth
    pages: int | None
    raw_dimensions: str | None
    physical_format: str | None
    edition_key: str | None
    source: str  # "openlibrary" | "none"


def _normalize_query(text: str) -> str:
    text = text.replace(",", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _docs_to_candidates(docs: list[dict], limit: int) -> list[Candidate]:
    out: list[Candidate] = []
    for i, doc in enumerate(docs[:limit]):
        key = str(doc.get("key") or f"hit_{i}")
        authors = doc.get("author_name") or []
        author_str = ", ".join(authors) if isinstance(authors, list) else str(authors)
        year = doc.get("first_publish_year")
        pages = doc.get("number_of_pages_median")
        cover = doc.get("cover_i")
        eds = doc.get("edition_key") or []
        if not isinstance(eds, list):
            eds = []
        out.append(
            Candidate(
                id=f"OL{i}",
                title=str(doc.get("title") or "").strip(),
                author=author_str.strip(),
                year=int(year) if year is not None else None,
                openlibrary_key=key,
                edition_keys=[str(e) for e in eds[:40]],
                pages_median=int(pages) if pages is not None else None,
                cover_id=int(cover) if cover is not None else None,
            )
        )
    return out


def _get(params: dict, timeout: float) -> list[dict]:
    resp = requests.get(
        SEARCH_URL,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("docs") or []


def search_books(
    title: str,
    author: str = "",
    *,
    limit: int = 5,
    timeout: float = 15.0,
) -> list[Candidate]:
    """Search Open Library; returns up to `limit` candidates."""
    title = _normalize_query(title or "")
    author = _normalize_query(author or "")
    if not title and not author:
        return []

    fields = (
        "key,title,author_name,first_publish_year,"
        "edition_key,number_of_pages_median,cover_i"
    )
    docs: list[dict] = []

    q = " ".join(p for p in (title, author) if p)
    docs = _get({"q": q, "limit": limit, "fields": fields}, timeout)

    if not docs and title:
        docs = _get({"title": title, "limit": limit, "fields": fields}, timeout)

    if not docs and title and author:
        docs = _get(
            {"title": title, "author": author, "limit": limit, "fields": fields},
            timeout,
        )

    return _docs_to_candidates(docs, limit)


_DIM_RE = re.compile(
    r"([\d.]+)\s*(?:x|×)\s*([\d.]+)\s*(?:x|×)\s*([\d.]+)\s*(centimeters?|cm|inches?|in)?",
    re.IGNORECASE,
)


def _to_mm(value: float, unit: str | None) -> float:
    u = (unit or "cm").lower()
    if u.startswith("in"):
        return round(value * 25.4, 1)
    return round(value * 10.0, 1) if u.startswith("c") or u == "cm" else round(value, 1)


def parse_physical_dimensions(raw: str) -> tuple[float | None, float | None, float | None]:
    """
    Parse OL `physical_dimensions` into (height_mm, width_mm, thickness_mm).

    OL strings are inconsistent in axis order. Heuristic for typical books:
    smallest → thickness (spine), largest → height, middle → width.
    """
    m = _DIM_RE.search(raw.replace(",", ""))
    if not m:
        return None, None, None
    a, b, c = float(m.group(1)), float(m.group(2)), float(m.group(3))
    unit = m.group(4)
    vals = sorted(_to_mm(v, unit) for v in (a, b, c))
    thickness, width, height = vals[0], vals[1], vals[2]
    return height, width, thickness


def fetch_publisher_size(
    candidate: Candidate,
    *,
    max_editions: int = 20,
    timeout: float = 12.0,
) -> PublisherSize:
    """
    Best-effort physical size from Open Library editions.

    Publisher spine *color* is not available in free catalogs — use photo color.
    Dimensions are sparse on OL; pages_median is a useful thickness proxy.
    """
    empty = PublisherSize(
        height_mm=None,
        width_mm=None,
        thickness_mm=None,
        pages=candidate.pages_median,
        raw_dimensions=None,
        physical_format=None,
        edition_key=None,
        source="none",
    )
    if not candidate.edition_keys and candidate.pages_median is None:
        return empty

    for ek in candidate.edition_keys[:max_editions]:
        try:
            resp = requests.get(
                f"https://openlibrary.org/books/{ek}.json",
                headers={"User-Agent": USER_AGENT},
                timeout=timeout,
            )
            if not resp.ok:
                continue
            ed = resp.json()
        except requests.RequestException:
            continue

        raw = ed.get("physical_dimensions")
        pages = ed.get("number_of_pages") or candidate.pages_median
        fmt = ed.get("physical_format")
        if raw:
            height, width, thickness = parse_physical_dimensions(str(raw))
            return PublisherSize(
                height_mm=height,
                width_mm=width,
                thickness_mm=thickness,
                pages=int(pages) if pages is not None else None,
                raw_dimensions=str(raw),
                physical_format=str(fmt) if fmt else None,
                edition_key=ek,
                source="openlibrary",
            )

    if candidate.pages_median is not None:
        return PublisherSize(
            height_mm=None,
            width_mm=None,
            thickness_mm=None,
            pages=candidate.pages_median,
            raw_dimensions=None,
            physical_format=None,
            edition_key=None,
            source="openlibrary",
        )
    return empty
