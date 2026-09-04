"""
Shared model runtime helpers for autourgos-openaichat.

Depends only on the `openai` SDK and `autourgos-core` (a separate,
zero-dependency stdlib utility library shared across the framework) --
no other third-party or autourgos-* dependency.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from string import Formatter
from typing import Any, Dict, Iterator, Optional
import time

logger = logging.getLogger(__name__)


# ── Latency tracking ──────────────────────────────────────────────────────────

@contextmanager
def track_latency() -> Iterator[Dict[str, float]]:
    """Record elapsed wall-clock milliseconds."""
    timing: Dict[str, float] = {"latency_ms": 0.0}
    start = time.perf_counter()
    try:
        yield timing
    finally:
        timing["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)


# ── Token/usage extraction ────────────────────────────────────────────────────

def extract_usage_metadata(resp: Any) -> Dict[str, Optional[int]]:
    """Extract token counts from an OpenAI response object."""
    if resp is None:
        return {"input_tokens": None, "output_tokens": None, "total_tokens": None}

    usage = None
    if isinstance(resp, dict):
        usage = resp.get("usage") or resp.get("usage_metadata")
    else:
        usage = getattr(resp, "usage", None) or getattr(resp, "usage_metadata", None)

    if usage is not None:
        def _get(obj: Any, *keys: str) -> Optional[int]:
            for k in keys:
                v = obj.get(k) if isinstance(obj, dict) else getattr(obj, k, None)
                if v is not None:
                    return int(v)
            return None

        return {
            "input_tokens": _get(usage, "prompt_tokens", "input_tokens", "prompt_token_count"),
            "output_tokens": _get(usage, "completion_tokens", "output_tokens", "candidates_token_count"),
            "total_tokens": _get(usage, "total_tokens", "total_token_count"),
        }

    return {"input_tokens": None, "output_tokens": None, "total_tokens": None}


# ── Structured response payload ───────────────────────────────────────────────

def build_structured_output(
    *,
    model_name: str,
    response_text: str,
    raw_response: Any,
    latency_ms: Optional[float] = None,
    input_pricing: Optional[float] = None,
    output_pricing: Optional[float] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a normalized dict with usage metadata and optional cost fields."""
    usage = extract_usage_metadata(raw_response)

    payload: Dict[str, Any] = {
        "model": model_name,
        "response": response_text,
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "total_tokens": usage["total_tokens"],
    }

    if input_pricing is not None and usage["input_tokens"] is not None:
        payload["input_cost"] = (usage["input_tokens"] / 1_000_000) * input_pricing
    if output_pricing is not None and usage["output_tokens"] is not None:
        payload["output_cost"] = (usage["output_tokens"] / 1_000_000) * output_pricing
    if "input_cost" in payload and "output_cost" in payload:
        payload["total_cost"] = payload["input_cost"] + payload["output_cost"]
    if latency_ms is not None:
        payload["latency_ms"] = latency_ms
    if extra_fields:
        payload.update(extra_fields)

    return payload


# ── Text extraction ───────────────────────────────────────────────────────────

def extract_refusal_from_response(resp: Any) -> Optional[str]:
    """
    Extract the model's refusal message from a non-streaming Chat Completions
    response, if present. ``message.refusal`` is a separate field from
    ``message.content`` -- a pure refusal leaves ``content`` empty, so
    ``extract_text_from_response()`` alone would return ``None`` with no way
    to tell a refusal apart from a genuinely malformed/empty response.
    """
    if resp is None:
        return None
    choices = resp.get("choices") if isinstance(resp, dict) else getattr(resp, "choices", None)
    if choices:
        first = choices[0]
        msg = first.get("message") if isinstance(first, dict) else getattr(first, "message", None)
        if msg:
            refusal = msg.get("refusal") if isinstance(msg, dict) else getattr(msg, "refusal", None)
            if isinstance(refusal, str) and refusal.strip():
                return refusal
    return None


def extract_text_from_response(resp: Any) -> Optional[str]:
    """Extract generated text from an OpenAI completion response."""
    if resp is None:
        return None
    if isinstance(resp, str) and resp.strip():
        return resp

    # OpenAI Chat Completions / Responses style
    choices = resp.get("choices") if isinstance(resp, dict) else getattr(resp, "choices", None)
    if choices:
        first = choices[0]
        msg = first.get("message") if isinstance(first, dict) else getattr(first, "message", None)
        if msg:
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                parts = [
                    (p.get("text") if isinstance(p, dict) else getattr(p, "text", None))
                    for p in content
                ]
                joined = "".join(p for p in parts if isinstance(p, str))
                if joined.strip():
                    return joined

    # Responses API: output[] array
    output = resp.get("output") if isinstance(resp, dict) else getattr(resp, "output", None)
    if output:
        for item in output:
            item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
            if item_type == "message":
                content_list = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
                if content_list:
                    for part in content_list:
                        text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
                        if isinstance(text, str) and text.strip():
                            return text

    # output_text shortcut (Responses API)
    output_text = resp.get("output_text") if isinstance(resp, dict) else getattr(resp, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    # Fallback key scan -- last resort for a response shape none of the
    # known formats above matched. Logged (not silent) because a top-level
    # dict that happens to carry an unrelated string field named exactly
    # "text"/"delta"/"content" would otherwise have that field returned as
    # if it were the actual completion text, with no signal anything was
    # unusual about the response shape.
    if isinstance(resp, dict):
        for key in ("text", "delta", "content"):
            val = resp.get(key)
            if isinstance(val, str) and val.strip():
                logger.warning(
                    "extract_text_from_response: response matched none of the known "
                    "shapes (choices/output/output_text); falling back to top-level "
                    "%r key. This may not be the actual completion text.",
                    key,
                )
                return val

    return None


# ── Template helpers ──────────────────────────────────────────────────────────

def extract_template_fields(template: str) -> set:
    """Return the set of {placeholder} names in a format string."""
    fields: set = set()
    for _, field_name, _, _ in Formatter().parse(template):
        if not field_name:
            continue
        base = field_name.split("!", 1)[0].split(":", 1)[0].strip()
        if base:
            fields.add(base)
    return fields


def coerce_prompt_variable(value: Any) -> str:
    """Coerce a prompt variable to string."""
    return "" if value is None else str(value)


# ── Process-level runtime setup ───────────────────────────────────────────────

def configure_runtime_environment() -> None:
    """
    No-op, kept only for backward compatibility.

    Used to set GRPC_VERBOSITY/GLOG_minloglevel/TF_CPP_MIN_LOG_LEVEL and
    filter gRPC UserWarnings globally -- this library only depends on
    `openai` (transport is httpx, not gRPC) and has no TensorFlow/glog
    dependency anywhere, so those were irrelevant, unconditional, process-
    wide side effects triggered merely by importing this package. Left as an
    empty no-op rather than removed since it's a public exported symbol.
    """


__all__ = [
    "track_latency",
    "extract_usage_metadata",
    "build_structured_output",
    "extract_text_from_response",
    "extract_refusal_from_response",
    "extract_template_fields",
    "coerce_prompt_variable",
    "configure_runtime_environment",
]
