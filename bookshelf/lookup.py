"""Open Library catalog search."""

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
        out.append(
            Candidate(
                id=f"OL{i}",
                title=str(doc.get("title") or "").strip(),
                author=author_str.strip(),
                year=int(year) if year is not None else None,
                openlibrary_key=key,
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

    fields = "key,title,author_name,first_publish_year"
    docs: list[dict] = []

    # Prefer unstructured q= — more tolerant of OCR noise than title=/author=.
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
