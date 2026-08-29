# Changelog

## [2.1.0] - 2026-08-29

- Added: automatic provider fallback chain via a new `fallback_providers=` constructor param — `invoke()`, `ainvoke()`, `stream()`, `astream()`, `invoke_with_tools()`, and `ainvoke_with_tools()` now transparently retry against ordered backup providers if the primary exhausts its retries, with no proxy/gateway service required. Each fallback entry resolves its own `api_key`/`base_url` independently (no credential inheritance from the primary). Streaming fallback only triggers before any chunk has been emitted, to avoid duplicating/corrupting partial output already sent to the caller.
- Added: `OpenAIChatModelAllProvidersFailedError` (subclass of `OpenAIChatModelAPIError`) raised when the primary and every fallback provider fail, carrying an `.attempts` list of `(label, exception)` pairs.
- Added: `"provider_used"` key in the structured-output metadata dict / `llm.last_metadata`, identifying which provider (`"primary"` or `"fallback[N]:<model>"`) actually served the request.
- Non-breaking: with `fallback_providers` unset (the default), behavior and exception types are unchanged from 2.0.1. Added 8 new tests covering fallback success, exhaustion, tool calling, and the streaming pre/post-emit boundary (65 total).

## [2.0.1] - 2026-08-29

- Maintenance release: no functional or documentation changes. Version bump to keep in step with the `autourgos-responses` companion package release.

## [2.0.0] - 2026-08-29

- Changed (breaking): relicensed from MIT to Apache License 2.0 — adds an explicit patent grant and patent-retaliation clause. `LICENSE`, `pyproject.toml` classifiers, and README updated accordingly.
- Docs: rewrote README.md — added PyPI/license/downloads badges, a Features summary, a Supported Providers table, full (non-collapsed) per-provider examples with sample output (added Google Gemini, xAI Grok, and OpenRouter, previously missing), and reorganized the feature walkthrough under one "Core Usage" section with a matching Table of Contents. No code changes.
- Changed (breaking): `OpenAIChatModel` constructor params renamed to standard agentic-AI terminology — `system_instruction` → `system_prompt`, `response_schema` → `output_schema`. No backward-compat aliases; callers on the old names must update. No other behavior changes — same class, same features.
- Fixed: `build_response_format()` (core.py) now sets `additionalProperties: false` on every object node of a `response_schema` — including nested models under Pydantic's `$defs` — required by OpenAI/Azure `strict: true` json_schema mode. Previously any Pydantic model passed as `response_schema` failed with a 400 `BadRequestError` ("additionalProperties is required to be supplied and to be false"), found via live testing against a real Azure OpenAI deployment. The dict-schema path now copies the caller's dict instead of mutating it in place.
- Added: full test suite (`tests/test_openai_chat.py`, 56 tests, 86% coverage) covering every documented feature — previously the package shipped with zero tests. Added `tests/live_azure_smoke.py` for manual real-network verification against a live deployment.
- Fixed: `_create_raw`/`_acreate_raw`/`_invoke_stream_mode`/`_ainvoke_stream_mode` (chat.py) no longer retry on non-retryable client errors (HTTP 400/401/403/404/422) — these fail immediately instead of burning the full retry budget with exponential backoff on errors that can never succeed. Found via live testing against a real Azure OpenAI deployment.

## [1.0.2] - 2026-07-27

- Fixed: standardized logger to logging.getLogger(__name__); release_openai_client/release_async_openai_client close failures are now logged; guarded the async circuit-breaker lock lazy-init against a check-then-set race with double-checked locking. Docs: Quick Start now notes OPENAI_API_KEY; added a path-validation warning for the file-based vision-input helper.

## [1.0.1] - 2026-06-16

- Update Documentation
