"""
OpenAI Chat Completions API helpers for autourgos-openaichat.

All utilities needed to configure clients, build request params,
and parse streaming deltas.
"""

from __future__ import annotations

import base64
import logging
import os
import re
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
    # The wrapper owns retry/backoff (see chat.py's retry loop). Without this,
    # the openai SDK's own default max_retries=2 retries underneath the
    # wrapper's retries, silently multiplying attempts and latency.
    kwargs["max_retries"] = 0
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
    # See configure_openai_client: the wrapper owns retry/backoff, so the SDK
    # must not retry underneath it.
    kwargs["max_retries"] = 0
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
    """
    Strip surrounding whitespace from the model identifier. Case is
    preserved: this value is sent as-is in the ``model`` request field, and
    forcing it to lowercase used to silently break case-sensitive
    identifiers -- Azure OpenAI deployment names (user-chosen, case-sensitive
    strings, not the base model name) and self-hosted/vLLM model tags -- by
    sending a wrong (lowercased) value the API would then 404 on.
    """
    return model.strip() if model else model


# ── Multi-modal message building ──────────────────────────────────────────────

# The only image formats OpenAI vision (Chat Completions and Responses API
# alike) actually supports -- everything else the provider will reject
# regardless of what MIME type we send.
_IMAGE_EXTENSION_MIME_TYPES = {
    "jpg": "image/jpeg", "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
}


def _guess_image_mime_type(file_path: str, ext: str) -> str:
    """
    Map a file extension to its correct image MIME type for vision input.

    A plain ``f"image/{ext}"`` guess is wrong for real, supported formats
    whose extension doesn't match their IANA subtype name -- ``.jpg``/
    ``.jpeg`` are ``image/jpeg``, not ``image/jpg``, which some providers
    reject outright. For an extension outside the four formats OpenAI vision
    actually supports (png/jpeg/gif/webp), a best-effort ``image/<ext>``
    guess is still returned (so the file isn't silently dropped from the
    message -- matches the existing send-anyway-with-a-warning philosophy
    used for the URL-fallback case below), but it's logged clearly since the
    provider will very likely reject it -- better a local warning naming the
    real problem than a confusing remote 400.
    """
    mime = _IMAGE_EXTENSION_MIME_TYPES.get(ext)
    if mime is not None:
        return mime
    logger.warning(
        "_encode_file: %r has extension %r, which isn't a recognized image "
        "type (supported: png, jpg/jpeg, gif, webp). Sending as image/%s "
        "anyway, but the provider will likely reject it.",
        file_path, ext, ext,
    )
    return f"image/{ext}"


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
            mime = _guess_image_mime_type(file, ext)
            data = base64.b64encode(raw).decode()
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{data}"},
            }
        except (OSError, IOError) as exc:
            # Treat as a direct URL. Only silent when the string actually
            # looks like one -- a typo'd local path (e.g. "photo.jpg" that
            # doesn't exist) doesn't start with a URL scheme, and used to
            # silently become a bogus "URL" sent straight to the API
            # instead of a clear file-not-found error surfacing here.
            if not file.startswith(("http://", "https://", "data:")):
                logger.warning(
                    "_encode_file: %r could not be opened as a file (%s) and doesn't "
                    "look like a URL; sending it to the API as-is anyway.",
                    file, exc,
                )
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

def enforce_additional_properties_false(schema: Any) -> Any:
    """
    Recursively set ``additionalProperties: False`` on every object node, and
    make ``required`` list every key in that node's ``properties``.

    OpenAI/Azure's ``strict: true`` json_schema mode rejects any object node
    that doesn't explicitly forbid extra properties. Pydantic's
    ``model_json_schema()`` doesn't set this by default, so a plain Pydantic
    model passed as ``output_schema`` fails with a 400 ("additionalProperties'
    is required to be supplied and to be false") unless this is applied —
    including inside ``$defs``, ``properties``, ``items``, and the
    ``anyOf``/``allOf``/``oneOf`` branches Pydantic emits for nested/optional
    models.

    Strict mode also requires *every* property key to appear in ``required``,
    but Pydantic only lists fields without a default (``Optional``/defaulted
    fields are omitted) -- so any schema with such a field would otherwise
    still 400 even with ``additionalProperties`` fixed. ``required`` is
    overwritten (not merged) with the full property-key list to cover that.
    Note this means a defaulted-but-non-``Optional`` field becomes something
    the model must always emit a value for -- strict mode has no notion of
    "optional with a default", only "required" vs. a nullable type; callers
    wanting true optionality should use ``Optional[X] = None``, which Pydantic
    already serializes as a nullable type unaffected by this.
    """
    if isinstance(schema, dict):
        # Returns a new dict rather than mutating `schema` in place -- this is
        # a public function (also called directly by autourgos-responses on
        # its own schemas) and the caller's input dict (including nested
        # dicts reachable from it) must be left untouched, not silently
        # rewritten as a side effect.
        result = dict(schema)
        if result.get("type") == "object" or "properties" in result:
            result.setdefault("additionalProperties", False)
            properties = result.get("properties")
            if isinstance(properties, dict):
                result["required"] = list(properties.keys())
        for key in ("properties", "$defs", "definitions"):
            sub = result.get(key)
            if isinstance(sub, dict):
                result[key] = {k: enforce_additional_properties_false(v) for k, v in sub.items()}
        for key in ("items", "additionalProperties"):
            sub = result.get(key)
            if isinstance(sub, dict):
                result[key] = enforce_additional_properties_false(sub)
        for key in ("anyOf", "allOf", "oneOf"):
            sub = result.get(key)
            if isinstance(sub, list):
                result[key] = [enforce_additional_properties_false(v) for v in sub]
        return result
    return schema


# Backward-compat alias for the old private name. autourgos-responses (and
# potentially other sibling packages) used to import this directly from
# `autourgos_openaichat.core` rather than through the public `__init__.py`
# surface -- keep the old name importable so that isn't a breaking change,
# but new/sibling-package code should use the public
# `from autourgos_openaichat import enforce_additional_properties_false`.
_enforce_additional_properties_false = enforce_additional_properties_false


def build_response_format(
    output_schema: Any = None,
    response_mime_type: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Build the response_format parameter for Chat Completions.

    - ``response_mime_type="application/json"``  →  {"type": "json_object"}
    - ``output_schema`` (Pydantic model)         →  {"type": "json_schema", ...}
    - Otherwise                                  →  None
    """
    if output_schema is not None:
        # Pydantic v2 model class
        schema_fn = getattr(output_schema, "model_json_schema", None)
        if callable(schema_fn):
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": getattr(output_schema, "__name__", "response"),
                    "schema": enforce_additional_properties_false(schema_fn()),
                    "strict": True,
                },
            }
        # Plain dict schema
        if isinstance(output_schema, dict):
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": enforce_additional_properties_false(dict(output_schema)),
                    "strict": True,
                },
            }

    if response_mime_type and "json" in response_mime_type.lower():
        return {"type": "json_object"}

    return None


# ── Per-target max_tokens / max_completion_tokens handling ───────────────────
# OpenAI's Chat Completions `max_tokens` is deprecated in favor of
# `max_completion_tokens` and is documented as "not compatible with o-series
# models" (o1, o3, o4-mini, ...) -- sending it to one of those models 400s.
# `build_chat_completion_create_params()` is called once per logical call,
# before it's known which target (primary/fallback[i]/shadow[i]) will
# actually receive the request -- a fallback or shadow provider can be a
# different model family than the primary. So the rename can't happen at
# build time; it happens per-target, right where each dispatch site already
# swaps in that target's `model` right before sending.

_MAX_COMPLETION_TOKENS_MODEL_PATTERN = re.compile(r"^o\d", re.IGNORECASE)


def model_requires_max_completion_tokens(model_name: str) -> bool:
    """
    True for OpenAI's o-series reasoning models (o1, o1-mini, o1-preview, o3,
    o3-mini, o3-pro, o4-mini, ...), detected by the documented ``o<digit>``
    naming convention. Any other model -- including third-party/self-hosted
    models -- keeps using ``max_tokens`` as before.
    """
    return bool(_MAX_COMPLETION_TOKENS_MODEL_PATTERN.match(model_name.strip()))


def apply_max_tokens_param(params: Dict[str, Any], model_name: str) -> None:
    """
    Rename an already-set ``params["max_tokens"]`` to
    ``"max_completion_tokens"`` if ``model_name`` is an o-series model.

    If the caller already explicitly set ``max_completion_tokens`` themselves
    via a per-call override, that value is never touched/renamed -- but a
    stale ``max_tokens`` can still be sitting in ``params`` alongside it (the
    constructor's ``max_tokens=`` default, baked in by ``_build_base_params``
    before the override was merged in under the *other* key name -- merging
    overrides only adds/replaces same-named keys, it doesn't know these two
    key names mean the same thing). Sending both in one request is at best
    redundant and at worst rejected by the provider, so the stale one is
    dropped in that case too.
    """
    if "max_completion_tokens" in params:
        params.pop("max_tokens", None)
        return
    if "max_tokens" not in params:
        return
    if model_requires_max_completion_tokens(model_name):
        params["max_completion_tokens"] = params.pop("max_tokens")


def strip_unsupported_sampling_params(params: Dict[str, Any], model_name: str) -> None:
    """
    Drop ``temperature``/``top_p`` from ``params`` in place if ``model_name``
    is an o-series reasoning model -- those models reject both params
    outright (400), same model family and same reasoning as
    ``apply_max_tokens_param()`` above. Called at the same per-target sites,
    for the same reason: a fallback/shadow target can be a different model
    family than the primary, so this can't be decided at build time.

    Dropped rather than raised, so a caller with temperature/top_p set for a
    non-reasoning primary (with an o-series fallback configured, say) doesn't
    get a hard failure -- just a warning and the call proceeds without them.
    """
    if not model_requires_max_completion_tokens(model_name):
        return
    for key in ("temperature", "top_p"):
        if key in params:
            del params[key]
            logger.warning(
                "%s doesn't support %r -- dropped from the request instead of "
                "sending it and getting a 400 (o-series reasoning models reject "
                "temperature/top_p entirely).",
                model_name, key,
            )


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
    if stream:
        # Without this, a Chat Completions stream never includes token usage
        # anywhere in its SSE chunks -- callers that need cost/ledger data
        # from a streaming call (invoke(streaming=True)) would have no way
        # to get it. The extra terminal chunk this adds has no `delta`, so
        # extract_text_delta_from_event() already ignores it for callers
        # that only want text (stream()/astream()).
        params["stream_options"] = {"include_usage": True}
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


def extract_refusal_delta_from_event(event: Any) -> Optional[str]:
    """
    Extract incremental refusal text from a Chat Completions streaming chunk.

    ``delta.refusal`` is a separate field from ``delta.content`` -- a refusal
    chunk never populates ``content``, so ``extract_text_delta_from_event()``
    alone would silently drop it and the caller would see no text at all.
    """
    choices = getattr(event, "choices", None)
    if choices:
        delta = getattr(choices[0], "delta", None)
        if delta is not None:
            refusal = getattr(delta, "refusal", None)
            if isinstance(refusal, str):
                return refusal

    if isinstance(event, dict):
        choices = event.get("choices")
        if choices:
            delta = choices[0].get("delta", {})
            refusal = delta.get("refusal")
            if isinstance(refusal, str):
                return refusal

    return None


def extract_usage_bearing_event(event: Any) -> Optional[Any]:
    """
    Return ``event`` itself if it carries token usage, else ``None``.

    With ``stream_options={"include_usage": True}`` set (see
    ``build_chat_completion_create_params``), a Chat Completions stream ends
    with one extra chunk that has ``usage`` populated and an empty/absent
    ``choices`` -- this is how a streaming caller (``invoke(streaming=True)``)
    gets the token counts needed for cost tracking and the ledger, since no
    other chunk carries them.
    """
    usage = event.get("usage") if isinstance(event, dict) else getattr(event, "usage", None)
    return event if usage is not None else None


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
    "model_requires_max_completion_tokens",
    "apply_max_tokens_param",
    "strip_unsupported_sampling_params",
    "extract_text_delta_from_event",
    "extract_refusal_delta_from_event",
    "extract_usage_bearing_event",
]
