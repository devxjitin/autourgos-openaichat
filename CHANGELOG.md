# Changelog

## [2.2.0] - 2026-08-30

- Added: `extra_body=` on `OpenAIChatModel` — a raw passthrough dict merged into every request (primary, fallback, and shadow providers alike) via the single shared `_build_base_params()`. Lets provider-specific, non-standard fields reach self-hosted OpenAI-compatible servers — e.g. vLLM's `guided_json`/`guided_regex`/`guided_choice` for constrained/guided decoding, or llama.cpp's `grammar` (GBNF). Not validated or interpreted by the library, and not portable across providers that don't recognize the same keys. Constructor-level only (no per-call override in this version), consistent with how `temperature`/`system_prompt`/`output_schema` etc. already work.
- Note: the low-level `create()`/`acreate()` methods already supported this via their existing `**overrides` passthrough — this release adds the same capability to the high-level convenience methods (`invoke`/`ainvoke`/`stream`/`astream`/`invoke_with_tools`/`ainvoke_with_tools`/`invoke_structured`/`ainvoke_structured`), which previously had no way to attach it.
- Non-breaking: `extra_body=None` (the default) means the `extra_body` key is never added to outgoing request params — identical to 2.1.0. 5 new tests (119 total).

## [2.1.0] - 2026-08-30

Consolidated feature release — eight additive capabilities layered on top of 2.0.1, none of them changing existing behavior, return types, or exception types when left at their default (off) settings.

- **Provider fallback chain** (`fallback_providers=`): `invoke`, `ainvoke`, `stream`, `astream`, `invoke_with_tools`, and `ainvoke_with_tools` transparently retry against ordered backup providers if the primary exhausts its retries — no proxy/gateway service required. Each entry resolves its own `api_key`/`base_url` independently. New `OpenAIChatModelAllProvidersFailedError` (`.attempts`); `"provider_used"` added to `llm.last_metadata`.
- **Validated structured output** (`invoke_structured()` / `ainvoke_structured()`): returns a validated Pydantic instance directly instead of a raw JSON string; on validation failure, the error is fed back to the model and retried (`max_validation_retries=`, default `2`). New `OpenAIChatModelValidationError` (`.raw_text`/`.validation_error`); `"validation_retries"` added to `llm.last_metadata`.
- **Local call ledger** (`ledger_path=`): records every `invoke`/`ainvoke`/`invoke_structured`/`ainvoke_structured` call to a local SQLite file (model, provider, prompt/response, tokens, cost, latency) — no external service, `sqlite3` is stdlib. `ledger_store_content=False` omits prompt/response text.
- **Budget governor** (`max_session_cost=`): once accumulated session cost reaches the USD cap, further calls raise `BudgetExceededException` *before* hitting the API. Backed by generic `max_session_cost`/`session_cost_used`/`reset_session_budget()` on `BaseLLM` itself. Requires `input_pricing`/`output_pricing`. A budget stop doesn't count toward the circuit breaker's failure threshold.
- **PII/secret redaction** (`redact_pii=True`): scans the resolved prompt for emails, credit cards, SSNs, phone numbers, and API keys before sending, via a new `redaction.py` module. `redact_mode="mask"` (default) replaces matches with `[REDACTED:<category>]`; `redact_mode="block"` raises `OpenAIChatModelRedactionBlockedError` instead. Explicitly a heuristic, best-effort scrubber — not a compliance-grade DLP solution.
  - **Restore-in-response** (`redact_restore_in_response=True`): swaps an echoed placeholder back for its original value in the text returned to the caller — the model itself still never sees the real secret, so this only helps pass-through/reference cases, not computations on the secret's value. `invoke_structured()` restores *before* validation. The ledger always records the still-masked text regardless.
  - **Bring your own dictionary**: `redact_custom_patterns=` (regex), `redact_custom_terms=` (exact literal values, auto-escaped, no regex needed), and `redact_patterns_file=` (a JSON file of patterns/terms, for a team-maintained dictionary outside code) all merge together; `redact_categories=[]` turns off the 5 built-ins entirely.
- **Shadow-mode dual dispatch** (`shadow_providers=`): dispatches the same prompt concurrently to backup providers for observation only — `invoke`/`ainvoke` always return the primary's result. New `shadow.py` provides `compute_similarity()` (stdlib `difflib`) for a rough text-overlap score. `on_shadow_result=` callback; `llm.last_shadow_results` exposes tokens/cost/latency/error per shadow provider. Shadow cost is tracked but not counted toward `max_session_cost`.

Ledger and redaction compose throughout: the ledger's `prompt`/`response` columns always reflect the already-redacted text, and gained `redacted_categories` and `shadow_calls` columns along the way.

Documented gaps, consistent across features: `invoke_with_tools()`/`ainvoke_with_tools()`/`stream()`/`astream()` are not covered by the ledger, budget governor, or shadow dispatch in this release — those paths don't compute usage/cost metadata today.

57 new tests (114 total, up from 56).

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
