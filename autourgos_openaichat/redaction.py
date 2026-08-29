"""
Best-effort PII / secret redaction for autourgos-openaichat.

Regex-based, opt-in, heuristic scrubbing of prompt text before it leaves the
process. This is NOT a compliance-grade DLP solution — it will miss PII that
doesn't match a known pattern (false negatives) and will occasionally mask
legitimate content that happens to match a pattern (false positives, e.g. a
user asking about the shape of an SSN). Treat it as defense-in-depth, not a
guarantee.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_PATTERNS: Dict[str, str] = {
    "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "credit_card": r"\b(?:\d[ -]?){13,16}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "phone": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "api_key": (
        r"\b(?:"
        r"sk-[A-Za-z0-9]{10,}"
        r"|ghp_[A-Za-z0-9]{20,}"
        r"|github_pat_[A-Za-z0-9_]{20,}"
        r"|AKIA[0-9A-Z]{16}"
        r"|AIza[0-9A-Za-z_-]{35}"
        r"|xox[baprs]-[0-9A-Za-z-]{10,}"
        r"|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"
        r")\b"
    ),
}


def compile_patterns(
    categories: Optional[List[str]] = None,
    custom_patterns: Optional[Dict[str, str]] = None,
) -> Dict[str, "re.Pattern"]:
    """Build the {category: compiled_regex} map to use for redaction."""
    selected = categories if categories is not None else list(DEFAULT_PATTERNS.keys())
    compiled: Dict[str, "re.Pattern"] = {}
    for name in selected:
        if name not in DEFAULT_PATTERNS:
            raise ValueError(f"Unknown redact_categories entry: {name!r}")
        compiled[name] = re.compile(DEFAULT_PATTERNS[name])
    for name, pattern in (custom_patterns or {}).items():
        compiled[name] = re.compile(pattern)
    return compiled


def redact_text(text: str, patterns: Dict[str, "re.Pattern"]) -> Tuple[str, List[str]]:
    """Mask every match in ``text``. Returns (redacted_text, categories_found)."""
    found: List[str] = []
    for category, pattern in patterns.items():
        def _sub(match: "re.Match", _category: str = category) -> str:
            found.append(_category)
            return f"[REDACTED:{_category}]"

        text = pattern.sub(_sub, text)
    return text, found


def _redact_content(content: Any, patterns: Dict[str, "re.Pattern"], found: List[str]) -> Any:
    if isinstance(content, str):
        redacted, hits = redact_text(content, patterns)
        found.extend(hits)
        return redacted
    if isinstance(content, list):
        new_parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                redacted, hits = redact_text(part["text"], patterns)
                found.extend(hits)
                part = {**part, "text": redacted}
            new_parts.append(part)
        return new_parts
    return content


def redact_value(value: Any, patterns: Dict[str, "re.Pattern"]) -> Tuple[Any, List[str]]:
    """
    Redact a resolved prompt value, which is either a plain string or a
    pre-built messages list (``[{"role": ..., "content": ...}, ...]``).
    Returns (redacted_value, categories_found — deduplicated, order-preserving).
    """
    found: List[str] = []
    if isinstance(value, str):
        redacted, hits = redact_text(value, patterns)
        found.extend(hits)
    elif isinstance(value, list):
        redacted = []
        for message in value:
            if isinstance(message, dict) and "content" in message:
                new_content = _redact_content(message["content"], patterns, found)
                message = {**message, "content": new_content}
            redacted.append(message)
    else:
        redacted = value

    seen = set()
    unique_found = [c for c in found if not (c in seen or seen.add(c))]
    return redacted, unique_found


__all__ = [
    "DEFAULT_PATTERNS",
    "compile_patterns",
    "redact_text",
    "redact_value",
]
