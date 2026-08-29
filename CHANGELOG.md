# Changelog

## [2.8.0] - 2026-08-29

- Added: "bring your own redaction dictionary" — two new ways to extend redaction beyond regex, alongside the existing `redact_custom_patterns=`. `redact_custom_terms=` takes `{category: [literal values]}` and redacts them as exact (regex-escaped) matches — no regex needed for fixed known-sensitive strings like codenames or asset IDs. `redact_patterns_file=` loads a JSON file (`{"patterns": {...}, "terms": {...}}`) at construction time, so a team can maintain its redaction dictionary as a separate, version-controlled file instead of in code; malformed/missing files raise `OpenAIChatModelConfigError`. All sources merge together (built-ins, file, `redact_custom_terms`, `redact_custom_patterns`, in that precedence order — later wins on a name collision). `redact_categories=[]` disables all 5 built-ins so only your own dictionary is active.
- `redaction.py`: new `terms_to_patterns()` and `load_patterns_file()` helpers; `compile_patterns()` signature extended (additive, existing positional/keyword usage unaffected).
- No new built-in categories were added — the 5 general-purpose ones (`email`, `credit_card`, `ssn`, `phone`, `api_key`) are unchanged; domain-specific vocabulary (classification markings, codenames, geolocation formats, etc.) is meant to be supplied via this extension mechanism rather than hardcoded into the library.
- Non-breaking: no new params set means identical behavior to 2.7.0. 7 new tests (114 total).

## [2.7.0] - 2026-08-29

- Added: `redact_restore_in_response=True` on `OpenAIChatModel` — when a redacted prompt's placeholder is echoed back by the model, the final text returned to the caller has it swapped back for the original value. The model itself still never sees the real secret; this only helps pass-through/reference cases (a computation that needs the secret's actual value still can't work, since the model never saw it). Requires `redact_pii=True` and `redact_mode="mask"` (config error otherwise). Works with `invoke()`/`ainvoke()`/`invoke_structured()`/`ainvoke_structured()` — `invoke_structured()` restores *before* validating against `output_schema`, which also fixes validation failures caused by strict-format fields rejecting a raw placeholder.
- `redaction.py`: placeholders become unique per occurrence (`[REDACTED:category:N]`) only when `redact_restore_in_response=True`; with it off, placeholders are unchanged (`[REDACTED:category]`) — zero behavior change to the existing 2.5.0 mask-mode format. New `restore_text()` helper.
- Privacy: the [Call Ledger](#call-ledger-audit-trail) always records the still-masked text regardless of this setting — restoration only affects the value returned to the caller, never what's persisted. On a failed `invoke_structured()` validation retry, the correction message fed back to the model always uses the masked text too, so the real secret is never leaked into the model's own conversation history even indirectly.
- Non-breaking: `redact_restore_in_response=False` (the default) is zero behavior change from 2.6.0. 8 new tests (107 total).

## [2.6.0] - 2026-08-29

- Added: shadow-mode dual dispatch — `shadow_providers=` on `OpenAIChatModel` dispatches the same prompt concurrently to one or more backup providers for observation only; `invoke()`/`ainvoke()` always return the primary's result. Runs primary + shadows in parallel (`ThreadPoolExecutor` for `invoke()`, `asyncio.gather` for `ainvoke()`) — total latency is roughly `max(primary, slowest shadow)`, not their sum, but not zero-overhead either (the call waits for all shadows to finish or fail). New `shadow.py` module provides `compute_similarity()` (stdlib `difflib`, no new dependency) for a rough 0.0-1.0 text-overlap score against the primary's response. `on_shadow_result=` callback fires per shadow result as it completes. `llm.last_shadow_results` exposes the full list after each call.
- Each shadow provider gets a single attempt (no retries); a shadow failure never raises and never affects the primary result — it surfaces as `error` in the result dict. Ledger gets a new `shadow_calls` table when enabled.
- Documented limitations: shadow-provider cost is tracked per-result but NOT counted toward `max_session_cost`; only `invoke()`/`ainvoke()` dispatch shadows (not `stream()`/`astream()`/`invoke_with_tools()`/`invoke_structured()`).
- Non-breaking: `shadow_providers` unset (the default) means zero new behavior on any existing code path. 8 new tests (99 total).

## [2.5.0] - 2026-08-29

- Added: PII/secret redaction — `redact_pii=True` scans the resolved prompt (string or messages list) for likely secrets/PII before it's sent, via a new `redaction.py` module. Built-in categories: `email`, `credit_card`, `ssn`, `phone`, `api_key` (OpenAI/GitHub/AWS/Google/Slack/JWT prefixes). `redact_categories=` restricts to a subset, `redact_custom_patterns=` adds extra `{name: regex}` entries. `redact_mode="mask"` (default) replaces matches with `[REDACTED:<category>]` and proceeds; `redact_mode="block"` raises the new `OpenAIChatModelRedactionBlockedError` (`.categories_found`) instead. Implemented as a single hook inside `_resolve_prompt()`, so it covers all 8 call paths uniformly (invoke/ainvoke/stream/astream/invoke_with_tools/ainvoke_with_tools/invoke_structured/ainvoke_structured). `llm.last_redacted_categories` exposes what matched the most recent call.
- Explicitly documented as a heuristic, best-effort scrubber (regex-based) — not a compliance-grade DLP solution; false positives and false negatives are expected. Disabled by default. Only the resolved prompt is scanned — `system_prompt` and vision `files=` content are not touched.
- Ledger integration: the [Call Ledger](#call-ledger-audit-trail) now records the already-redacted prompt (never the raw text) and a new `redacted_categories` column.
- Non-breaking: `redact_pii=False` (the default) means `_resolve_prompt()`'s behavior is unchanged on every existing code path. 9 new tests (91 total).

## [2.4.0] - 2026-08-29

- Added: budget governor — `max_session_cost=` (USD) on `OpenAIChatModel`, backed by a new generic `max_session_cost`/`session_cost_used`/`reset_session_budget()` in `BaseLLM` itself (so any future wrapper subclassing `BaseLLM` gets this for free). Once accumulated session cost reaches the cap, `invoke()`/`ainvoke()`/`invoke_structured()`/`ainvoke_structured()` raise `BudgetExceededException` **before** making the API call. Requires `input_pricing`/`output_pricing` to be set (raises `OpenAIChatModelConfigError` at construction otherwise). `reset_session_budget()` unblocks a tripped cap. A budget stop does not count toward the circuit breaker's failure threshold.
- Known limitation (documented in README): the cap is checked against cost already accumulated from prior calls, not an exact prediction of the upcoming call's cost. `invoke_structured()`'s failed-validation retry attempts aren't individually tracked into `session_cost_used`. `invoke_with_tools()`/`ainvoke_with_tools()`/`stream()`/`astream()` are not budget-protected (no usage/cost metadata computed on those paths today, same gap as the call ledger).
- Non-breaking: `max_session_cost` unset (the default) means zero new behavior on any existing code path. 5 new tests (82 total).

## [2.3.0] - 2026-08-29

- Added: optional local call ledger — `ledger_path=` records every `invoke()`/`ainvoke()`/`invoke_structured()`/`ainvoke_structured()` call to a local SQLite file (model, provider used, prompt/response, tokens, cost, latency, validation retries). No external service, no extra dependency (`sqlite3` is stdlib). Disabled by default (`ledger_path=None`) — zero overhead unless enabled. `ledger_store_content=False` logs only metadata, omitting prompt/response text. A ledger write failure is logged as a warning and never breaks the actual LLM call. `invoke_with_tools()`/`ainvoke_with_tools()`/`stream()`/`astream()` are not logged in this version (no usage/cost metadata computed on those paths today).
- Non-breaking: `ledger_path` unset (the default) means no new code runs on any existing code path. 5 new tests (77 total).

## [2.2.0] - 2026-08-29

- Added: `invoke_structured()` / `ainvoke_structured()` — a validated structured-output loop on top of `output_schema=`. Instead of a raw JSON string, returns a validated Pydantic instance directly; on validation failure, the error is fed back to the model as a correction message and the request is retried (`max_validation_retries=`, default `2`). Composes with the provider fallback chain (each attempt goes through the same primary → fallback sequence) and with the circuit breaker (registered in `BaseLLM.__init_subclass__` like `invoke`/`invoke_with_tools`).
- Added: `OpenAIChatModelValidationError` (subclass of `OpenAIChatModelResponseError`), raised when validation retries are exhausted, carrying `.raw_text` and `.validation_error`.
- Added: `"validation_retries"` key in `llm.last_metadata` after a successful `invoke_structured()`/`ainvoke_structured()` call.
- Non-breaking: `invoke()`/`ainvoke()`/`structured_output=True` behavior and return types are completely unchanged — this is a new, separate pair of methods. Added 7 new tests (72 total).

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
