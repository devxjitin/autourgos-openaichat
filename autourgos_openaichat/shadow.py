"""
Shadow-mode dual dispatch for autourgos-openaichat.

Compares a primary response against one or more "shadow" providers run
concurrently, purely for observation — the primary's answer is always what
the caller gets back. No new dependency: similarity uses stdlib difflib.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Optional


def compute_similarity(text_a: Optional[str], text_b: Optional[str]) -> Optional[float]:
    """Return a 0.0-1.0 similarity ratio between two texts, or None if either is missing."""
    if text_a is None or text_b is None:
        return None
    return SequenceMatcher(None, text_a, text_b).ratio()


__all__ = ["compute_similarity"]
