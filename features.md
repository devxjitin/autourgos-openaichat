# autourgos-openaichat — Features

A self-contained Python wrapper around the **OpenAI Chat Completions API** (`client.chat.completions.create`), and by extension any provider that speaks the same protocol (Azure, Groq, Gemini, Mistral, DeepSeek, Ollama, LM Studio, vLLM, OpenRouter, Together AI, Perplexity, xAI). Zero required dependency beyond `openai` itself.

## Full Feature List

### Core generation
- Sync (`invoke`) and async (`ainvoke`) text generation
- Sync (`stream`) and async (`astream`) streaming, including streaming with full cost/usage recovery via `invoke(streaming=True)`
- Multi-turn conversations (pass a `messages` list directly)
- System prompts and `{placeholder}` prompt templates
- Multi-modal vision input — image file paths, raw bytes, or URLs, with correct MIME-type detection
- Batch invocation — `batch_invoke()` (sequential) / `abatch_invoke()` (concurrent via `asyncio.gather`)
- Native function/tool calling (`invoke_with_tools`/`ainvoke_with_tools`)
- Structured output: plain JSON mode, or a Pydantic-validated result with an automatic validation-retry loop that feeds the schema error back to the model (`invoke_structured`/`ainvoke_structured`)

### Reliability
- Automatic retries with exponential back-off, skipping non-retryable 4xx status codes
- Circuit breaker — opens after N consecutive failures, cools down for a configurable window, protecting the rest of an app from a cascading failure
- Automatic provider fallback chain — an ordered list of backup providers (each with its own API key/base URL/pricing), tried in order with no external proxy/gateway required
- Optional aggregate call deadline (`max_call_duration`) capping total wall-clock time across every retry and fallback attempt for one logical call

### Cost & budget
- Built-in per-call cost and latency tracking (`last_metadata`), computed from configurable per-1M-token input/output pricing
- Session budget governor (`max_session_cost`) that hard-stops further calls once a USD cap is reached, with concurrency-safe admission (no overshoot from concurrent calls on a shared instance)

### Observability
- Optional local call ledger — SQLite file, no external service, records every call's prompt/response/tokens/cost/latency/provider
- Shadow-mode dual dispatch — send the same prompt concurrently to one or more "shadow" providers purely for comparison (similarity score, cost, latency), without ever changing what the caller receives

### Security
- Optional PII/secret redaction — heuristic pre-flight scrubber for emails, API keys, credit cards, SSNs, phone numbers, with custom regex/literal-term dictionaries and an external patterns file
- `mask` (replace and proceed) or `block` (raise instead of sending) modes, with optional reversible restore-in-response

### Advanced / escape hatches
- `extra_body=` passthrough for provider-specific fields (e.g. vLLM's `guided_json`/`guided_regex`, llama.cpp's `grammar`) for constrained decoding
- `store=` passthrough for OpenAI's server-side response persistence
- Low-level `create()`/`acreate()` for direct, retry-managed access to the raw completion object
- Sync/async context manager support for deterministic client cleanup
- Fully typed (`py.typed`)

## Internal architecture note

As of this refactor, `autourgos-openaichat` and its sibling package `autourgos-responses` share a common `BaseProviderLLM` base class for all of the above except the parts that are genuinely API-shape-specific (message format, prompt/param building, tool-schema shape) — so both packages get identical reliability/cost/observability/security behavior with a single, tested implementation.

---

## Competitor Comparison

Landscape research (LLM gateways, orchestration frameworks, and structured-output libraries), current as of the search date.

| Capability | **autourgos-openaichat** | Raw `openai` Python SDK | [LiteLLM](https://docs.litellm.ai/) | [LangChain](https://www.langchain.com/) (`ChatOpenAI`) | [Instructor](https://python.useinstructor.com/) | [Portkey AI Gateway](https://github.com/portkey-ai/gateway) |
|---|---|---|---|---|---|---|
| Scope | In-process Python library, no separate service | In-process Python library | Library **or** a self-hosted proxy/gateway | In-process orchestration framework | In-process library, wraps an LLM client | Hosted or self-hosted **gateway/proxy** (separate service) |
| Multi-provider via one interface | Yes, any OpenAI-compatible `base_url` | No (OpenAI/Azure only) | Yes, 100+ providers, including non-OpenAI-compatible ones | Yes, via separate integration packages per provider | Depends on the wrapped client | Yes, 1,600+ models |
| Automatic retries | Yes, exponential back-off, skips non-retryable 4xx | Yes, basic exponential back-off (`max_retries`, default 2) | Yes, configurable per model/group | Not native — needs external tooling | Yes, via Tenacity, largely for validation-failure retries | Yes |
| Provider fallback chain | Yes, built into the library, no proxy needed | No | Yes — a core feature, this is LiteLLM's specialty | Not native | No | Yes — a core feature |
| Circuit breaker | Yes, built-in | No | Not a standard built-in primitive (proxy-level health checks exist) | No | No | Partial (via gateway-level health/routing) |
| Aggregate call deadline across retries+fallback | Yes (`max_call_duration`) | No (per-request `timeout` only) | Not as a single explicit cross-attempt budget | No | No | Partial (gateway request timeouts) |
| Cost tracking | Yes, per-call, configurable pricing | No | Yes, with real-time provider pricing lookup | Only via LangSmith (external service) | No | Yes, with per-team/app dashboards |
| Session budget hard-cap | Yes (`max_session_cost`), concurrency-safe | No | Yes, via virtual-key budget routing (server-side) | No native | Yes, but scoped to validation-retry token budget only | Yes, via budget/limits on virtual keys |
| Local audit ledger (no external service) | Yes, SQLite, opt-in | No | No — logging goes to an external/observability backend | No — needs LangSmith/Langfuse | No | No — logs live in the gateway's own store |
| Shadow-mode dual dispatch (compare providers on live traffic, in-process) | Yes, built-in, with similarity scoring | No | Not as an in-library primitive | No | No | Not as a code-level primitive (would need external traffic mirroring) |
| PII/secret redaction | Yes, built-in, mask/block, custom dictionaries | No | No | No | No | Yes, via 50+ built-in guardrails (hosted feature) |
| Native tool/function calling | Yes | Yes (raw API surface only) | Yes (passthrough) | Yes | Not its focus | Passthrough |
| Structured output + validation-retry loop | Yes, Pydantic, automatic retry with schema-error feedback | No (raw `response_format` only) | Passthrough only | Via separate output-parser abstractions | Yes — this is Instructor's core specialty | No |
| Requires infrastructure/ops | No — pure library | No | Optional (proxy mode) or none (SDK mode) | No | No | Yes for self-hosted; hosted plan is a paid SaaS |
| Pricing | Free, open source | Free | Free, open source (self-hosted) | Free, open source (LangSmith is paid) | Free, open source | Free tier + paid plans (~$49/mo+ for production features) |

### How to read this

- **vs. the raw OpenAI SDK**: autourgos-openaichat adds everything the raw SDK doesn't have — fallback, circuit breaking, budget caps, ledger, shadow mode, redaction, validated structured output — while keeping a comparably simple, single-`pip install` footprint.
- **vs. LiteLLM**: LiteLLM's specialty is breadth (100+ providers) and gateway-mode deployment; it has strong fallback/cost tracking but no in-process shadow-mode dispatch, no built-in circuit breaker, and no PII redaction. autourgos-openaichat trades provider breadth for a narrower (OpenAI-compatible-only), zero-infrastructure, single-library footprint with more reliability/security primitives built directly into the call path.
- **vs. LangChain**: LangChain's strength is orchestration and ecosystem breadth (agents, retrievers, chains); reliability/cost/observability features are largely delegated to external tools (LangSmith) rather than built into the client itself.
- **vs. Instructor**: Instructor is the deepest tool specifically for Pydantic-validated structured output with retry-on-validation-failure — a narrower, more specialized version of one feature this library also has, but Instructor has no fallback chain, circuit breaker, budget governor, ledger, or redaction.
- **vs. Portkey**: Portkey is a hosted/self-hosted gateway (a separate network service in front of your calls) with a broad guardrail/PII feature set and strong dashboards, but it's infrastructure to run and often a paid product at production scale; autourgos-openaichat is a plain importable library with no service to operate.

Sources:
- [Portkey vs LiteLLM: Routing, Fallbacks, Cost Tracking, and Control](https://medium.com/@adnanmasood/portkey-vs-litellm-routing-fallbacks-cost-tracking-and-control-the-llm-gateway-playbook-part-195855dc25c3)
- [Fallbacks (Provider Failover) | LiteLLM docs](https://docs.litellm.ai/docs/proxy/reliability)
- [Reliability - Retries, Fallbacks | LiteLLM docs](https://docs.litellm.ai/docs/completion/reliable_completions)
- [LangChain Observability: Monitoring Guide for Production Apps](https://uptrace.dev/blog/langchain-observability)
- [Retry Logic with Tenacity - Instructor docs](https://python.useinstructor.com/concepts/retrying/)
- [Retry Mechanisms - Instructor docs](https://python.useinstructor.com/learning/validation/retry_mechanisms/)
- [Portkey AI Gateway GitHub](https://github.com/portkey-ai/gateway)
- [Best LiteLLM Alternatives in 2026](https://www.getmaxim.ai/articles/best-litellm-alternatives-in-2026/)
- [Best LLM Routing Platforms Compared (2026)](https://www.requesty.ai/blog/best-llm-routing-platforms-compared-2026-requesty-portkey-litellm-openrouter)
- [Retries - OpenAI Python SDK docs](https://openai-openai-python-73.mintlify.app/concepts/retries)
