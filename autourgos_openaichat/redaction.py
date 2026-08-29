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

import json
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


def terms_to_patterns(terms: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Convert {category: [literal term, ...]} into {category: regex}, so callers
    can list exact known-sensitive strings (codenames, asset IDs, unit names)
    without writing any regex themselves. Each term is escaped and the whole
    category becomes one word-boundary-wrapped alternation.
    """
    patterns: Dict[str, str] = {}
    for category, values in terms.items():
        if not values:
            continue
        alternation = "|".join(re.escape(v) for v in values)
        patterns[category] = rf"\b(?:{alternation})\b"
    return patterns


def load_patterns_file(path: str) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Load a JSON dictionary file of the form::

        {
          "patterns": {"category": "regex", ...},
          "terms": {"category": ["literal value", ...], ...}
        }

    Both top-level keys are optional. Returns (patterns, terms).
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except OSError as exc:
        raise ValueError(f"Could not read redact_patterns_file {path!r}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"redact_patterns_file {path!r} is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"redact_patterns_file {path!r} must contain a JSON object at the top level.")

    patterns = data.get("patterns", {})
    terms = data.get("terms", {})
    if not isinstance(patterns, dict):
        raise ValueError(f"redact_patterns_file {path!r}: 'patterns' must be an object of {{name: regex}}.")
    if not isinstance(terms, dict):
        raise ValueError(f"redact_patterns_file {path!r}: 'terms' must be an object of {{name: [values]}}.")
    return patterns, terms


def compile_patterns(
    categories: Optional[List[str]] = None,
    custom_patterns: Optional[Dict[str, str]] = None,
    custom_terms: Optional[Dict[str, List[str]]] = None,
    patterns_file: Optional[str] = None,
) -> Dict[str, "re.Pattern"]:
    """
    Build the {category: compiled_regex} map to use for redaction, merging
    (in this precedence order, later wins on a name collision):

        built-in categories  ->  patterns_file (patterns + terms)  ->
        custom_terms  ->  custom_patterns
    """
    selected = categories if categories is not None else list(DEFAULT_PATTERNS.keys())
    merged: Dict[str, str] = {}
    for name in selected:
        if name not in DEFAULT_PATTERNS:
            raise ValueError(f"Unknown redact_categories entry: {name!r}")
        merged[name] = DEFAULT_PATTERNS[name]

    if patterns_file:
        file_patterns, file_terms = load_patterns_file(patterns_file)
        merged.update(file_patterns)
        merged.update(terms_to_patterns(file_terms))

    if custom_terms:
        merged.update(terms_to_patterns(custom_terms))

    if custom_patterns:
        merged.update(custom_patterns)

    return {name: re.compile(pattern) for name, pattern in merged.items()}


class _RedactionState:
    """
    Accumulates categories found (and, if tracking, an original-value mapping
    with globally unique placeholders) across every text field of one prompt.
    """

    def __init__(self, track_mapping: bool) -> None:
        self.track_mapping = track_mapping
        self.mapping: Dict[str, str] = {}
        self.found: List[str] = []
        self._counter = 0

    def redact(self, text: str, patterns: Dict[str, "re.Pattern"]) -> str:
        for category, pattern in patterns.items():
            def _sub(match: "re.Match", _category: str = category) -> str:
                self.found.append(_category)
                if self.track_mapping:
                    self._counter += 1
                    placeholder = f"[REDACTED:{_category}:{self._counter}]"
                    self.mapping[placeholder] = match.group(0)
                    return placeholder
                return f"[REDACTED:{_category}]"

            text = pattern.sub(_sub, text)
        return text


def redact_text(text: str, patterns: Dict[str, "re.Pattern"]) -> Tuple[str, List[str]]:
    """Mask every match in ``text`` with non-unique placeholders. Returns (redacted_text, categories_found)."""
    state = _RedactionState(track_mapping=False)
    redacted = state.redact(text, patterns)
    return redacted, state.found


def _redact_content(content: Any, patterns: Dict[str, "re.Pattern"], state: _RedactionState) -> Any:
    if isinstance(content, str):
        return state.redact(content, patterns)
    if isinstance(content, list):
        new_parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                part = {**part, "text": state.redact(part["text"], patterns)}
            new_parts.append(part)
        return new_parts
    return content


def redact_value(
    value: Any,
    patterns: Dict[str, "re.Pattern"],
    *,
    track_mapping: bool = False,
) -> Tuple[Any, List[str], Dict[str, str]]:
    """
    Redact a resolved prompt value, which is either a plain string or a
    pre-built messages list (``[{"role": ..., "content": ...}, ...]``).

    If ``track_mapping`` is True, placeholders are made unique
    (``[REDACTED:category:N]``) and ``mapping`` records each placeholder's
    original matched text, so the caller can restore it later with
    ``restore_text()``. If False (the default), placeholders stay
    ``[REDACTED:category]`` and ``mapping`` is empty — unchanged from the
    original mask-only behavior.

    Returns (redacted_value, categories_found — deduplicated, order-preserving, mapping).
    """
    state = _RedactionState(track_mapping=track_mapping)
    if isinstance(value, str):
        redacted = state.redact(value, patterns)
    elif isinstance(value, list):
        redacted = []
        for message in value:
            if isinstance(message, dict) and "content" in message:
                message = {**message, "content": _redact_content(message["content"], patterns, state)}
            redacted.append(message)
    else:
        redacted = value

    seen = set()
    unique_found = [c for c in state.found if not (c in seen or seen.add(c))]
    return redacted, unique_found, state.mapping


def restore_text(text: Optional[str], mapping: Dict[str, str]) -> Optional[str]:
    """Replace each tracked placeholder in ``text`` with its original matched value."""
    if text is None or not mapping:
        return text
    for placeholder, original in mapping.items():
        text = text.replace(placeholder, original)
    return text


__all__ = [
    "DEFAULT_PATTERNS",
    "compile_patterns",
    "terms_to_patterns",
    "load_patterns_file",
    "redact_text",
    "redact_value",
    "restore_text",
]
