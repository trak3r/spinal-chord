"""Laya (open Jev-style) match verification."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from bookshelf.lookup import Candidate

_agent = None


@dataclass
class Decision:
    matched: bool
    candidate: Candidate | None
    choice_id: str
    choice_confidence: float
    confident_noul: float


def _get_agent():
    global _agent
    if _agent is None:
        import laya

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*invalid temperatures.*",
                category=RuntimeWarning,
            )
            _agent = laya.load("convaiinnovations/laya")
    return _agent


def verify_match(
    ocr_title: str,
    ocr_author: str,
    candidates: list[Candidate],
    *,
    noul_threshold: float = 0.55,
    conf_threshold: float = 0.35,
) -> Decision:
    """Pick the best Open Library hit for OCR text, or none."""
    if not candidates or (not ocr_title and not ocr_author):
        return Decision(
            matched=False,
            candidate=None,
            choice_id="none",
            choice_confidence=0.0,
            confident_noul=0.0,
        )

    by_id = {c.id: c for c in candidates}
    criteria = {
        c.id: f"{c.title} by {c.author}"
        + (f" ({c.year})" if c.year else "")
        for c in candidates
    }
    criteria["none"] = "none of these / cannot tell"

    state = (
        "OCR from a book spine or cover reads:\n"
        f"title: {ocr_title or '(unknown)'}\n"
        f"author: {ocr_author or '(unknown)'}\n\n"
        "Pick which Open Library catalog entry is the same book."
    )
    questions = {
        "match": {
            "type": "choice",
            "instructions": (
                "Which catalog entry matches the OCR? "
                "Choose none if unclear."
            ),
            "criteria": criteria,
        },
        "confident": {
            "type": "noul",
            "instructions": "Is the OCR clearly the same book as your chosen entry?",
        },
    }

    agent = _get_agent()
    result = agent.predict(state, questions)
    answers = result["answers"]

    match = answers["match"]
    choice_id = str(match.get("choice") or "none")
    # Prefer answer_confidence (chosen option probability). Calibrated
    # `confidence` is often depressed when several near-duplicate editions compete.
    choice_conf = float(
        match.get("answer_confidence")
        or match.get("confidence")
        or 0.0
    )
    confident = answers["confident"]
    noul = float(confident.get("noul") or 0.0)

    cand = by_id.get(choice_id)
    matched = (
        choice_id != "none"
        and cand is not None
        and noul >= noul_threshold
        and choice_conf >= conf_threshold
    )
    return Decision(
        matched=matched,
        candidate=cand if matched else None,
        choice_id=choice_id,
        choice_confidence=choice_conf,
        confident_noul=noul,
    )
