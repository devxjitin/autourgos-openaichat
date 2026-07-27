"""
OpenAI Chat Completions API helpers for autourgos-openaichat.

All utilities needed to configure clients, build request params,
and parse streaming deltas.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Module loading ────────────────────────────────────────────────────────────

def load_openai_module() -> Tuple[bool, Any, Any, Optional[str]]:
    """Try to import the openai package. Returns (available, OpenAI, AsyncOpenAI, error)."""
    try:
        import openai as _openai
        return True, _openai.OpenAI, _openai.AsyncOpenAI, None
    except ImportError as exc:
        return False, None, None, str(exc)


# ── Key / URL resolution ──────────────────────────────────────────────────────

def resolve_api_key(api_key: Optional[str]) -> Optional[str]:
    """Return the provided key or fall back to OPENAI_API_KEY env var."""
    return api_key or os.environ.get("OPENAI_API_KEY")


def resolve_base_url(base_url: Optional[str]) -> Optional[str]:
    """Return the provided URL or fall back to OPENAI_BASE_URL env var."""
    return base_url or os.environ.get("OPENAI_BASE_URL")


# ── Client helpers ────────────────────────────────────────────────────────────

def configure_openai_client(
    openai_cls: Any,
    *,
    api_key: Optional[str],
    base_url: Optional[str],
    organization: Optional[str],
    project: Optional[str],
    timeout: Optional[float],
) -> Any:
    """Create and return a synchronous OpenAI client."""
    kwargs: Dict[str, Any] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    if organization:
        kwargs["organization"] = organization
    if project:
        kwargs["project"] = project
    if timeout is not None:
        kwargs["timeout"] = timeout
    return openai_cls(**kwargs)


def configure_async_openai_client(
    async_openai_cls: Any,
    *,
    api_key: Optional[str],
    base_url: Optional[str],
    organization: Optional[str],
    project: Optional[str],
    timeout: Optional[float],
) -> Any:
    """Create and return an asynchronous OpenAI client."""
    kwargs: Dict[str, Any] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    if organization:
        kwargs["organization"] = organization
    if project:
        kwargs["project"] = project
    if timeout is not None:
        kwargs["timeout"] = timeout
    return async_openai_cls(**kwargs)


def release_openai_client(client: Any) -> None:
    """Close the synchronous client if it exposes a close() method."""
    closer = getattr(client, "close", None)
    if callable(closer):
        try:
            closer()
        except Exception:
            logger.warning("Failed to close OpenAI client cleanly", exc_info=True)


async def release_async_openai_client(client: Any) -> None:
    """Close the async client if it exposes an aclose() or close() method."""
    acloser = getattr(client, "aclose", None)
    if callable(acloser):
        try:
            await acloser()
        except Exception:
            logger.warning("Failed to aclose() async OpenAI client cleanly", exc_info=True)
        return
    closer = getattr(client, "close", None)
    if callable(closer):
        try:
            closer()
        except Exception:
            logger.warning("Failed to close() async OpenAI client cleanly", exc_info=True)


# ── Model name normalization ──────────────────────────────────────────────────

def normalize_model_name(model: str) -> str:
    """Strip whitespace and lowercase the model identifier."""
    return model.strip().lower() if model else model


# ── Multi-modal message building ──────────────────────────────────────────────

def _encode_file(file: Any) -> Optional[Dict[str, Any]]:
    """
    Encode a file into an OpenAI image_url content part.

    Accepts:
        - str / bytes  — raw path or bytes
        - dict         — {"path": ..., "mime_type": ...} or {"data": ..., "mime_type": ...}
    """
    if isinstance(file, bytes):
        data = base64.b64encode(file).decode()
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{data}"},
        }
    if isinstance(file, str):
        # Treat as a file path
        try:
            with open(file, "rb") as fh:
                raw = fh.read()
            ext = file.rsplit(".", 1)[-1].lower() if "." in file else "png"
            mime = f"image/{ext}"
            data = base64.b64encode(raw).decode()
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{data}"},
            }
        except (OSError, IOError):
            # Treat as a direct URL
            return {"type": "image_url", "image_url": {"url": file}}
    if isinstance(file, dict):
        if "data" in file:
            raw = file["data"]
            if isinstance(raw, bytes):
                raw = base64.b64encode(raw).decode()
            mime = file.get("mime_type", "image/png")
            return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{raw}"}}
        if "path" in file:
            return _encode_file(file["path"])
        if "url" in file:
            return {"type": "image_url", "image_url": {"url": file["url"]}}
    return None


def build_multimodal_messages(
    text: str,
    files: Optional[List[Any]] = None,
    image_detail: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Build the user message list for a Chat Completions request.

    If ``files`` is empty/None, returns a plain-text message.
    Otherwise builds a content-array message with text + image parts.
    """
    if not files:
        return [{"role": "user", "content": text}]

    content: List[Any] = [{"type": "text", "text": text}]
    for f in files:
        part = _encode_file(f)
        if part is not None:
            if image_detail and "image_url" in part:
                part["image_url"]["detail"] = image_detail
            content.append(part)

    return [{"role": "user", "content": content}]


# ── Response format builder ───────────────────────────────────────────────────

def build_response_format(
    response_schema: Any = None,
    response_mime_type: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Build the response_format parameter for Chat Completions.

    - ``response_mime_type="application/json"``  →  {"type": "json_object"}
    - ``response_schema`` (Pydantic model)       →  {"type": "json_schema", ...}
    - Otherwise                                  →  None
    """
    if response_schema is not None:
        # Pydantic v2 model class
        schema_fn = getattr(response_schema, "model_json_schema", None)
        if callable(schema_fn):
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": getattr(response_schema, "__name__", "response"),
                    "schema": schema_fn(),
                    "strict": True,
                },
            }
        # Plain dict schema
        if isinstance(response_schema, dict):
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": response_schema,
                    "strict": True,
                },
            }

    if response_mime_type and "json" in response_mime_type.lower():
        return {"type": "json_object"}

    return None


# ── Chat Completions params builder ───────────────────────────────────────────

def build_chat_completion_create_params(
    model: str,
    messages: List[Dict[str, Any]],
    *,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    response_format: Optional[Dict[str, Any]] = None,
    stream: bool = False,
) -> Dict[str, Any]:
    """Build the kwargs dict for client.chat.completions.create()."""
    params: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    if temperature is not None:
        params["temperature"] = temperature
    if top_p is not None:
        params["top_p"] = top_p
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if response_format is not None:
        params["response_format"] = response_format
    return params


# ── Streaming delta extraction ────────────────────────────────────────────────

def extract_text_delta_from_event(event: Any) -> Optional[str]:
    """Extract incremental text from a Chat Completions streaming chunk."""
    # event.choices[0].delta.content
    choices = getattr(event, "choices", None)
    if choices:
        delta = getattr(choices[0], "delta", None)
        if delta is not None:
            content = getattr(delta, "content", None)
            if isinstance(content, str):
                return content

    # Dict fallback
    if isinstance(event, dict):
        choices = event.get("choices")
        if choices:
            delta = choices[0].get("delta", {})
            content = delta.get("content")
            if isinstance(content, str):
                return content

    return None


__all__ = [
    "logger",
    "load_openai_module",
    "resolve_api_key",
    "resolve_base_url",
    "configure_openai_client",
    "configure_async_openai_client",
    "release_openai_client",
    "release_async_openai_client",
    "normalize_model_name",
    "build_multimodal_messages",
    "build_response_format",
    "build_chat_completion_create_params",
    "extract_text_delta_from_event",
]
