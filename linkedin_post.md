Most "LLM wrapper" libraries stop at .invoke(). We didn't.

Introducing autourgos-openaichat — a single, self-contained Python wrapper for the OpenAI Chat Completions API, built for anyone shipping LLM calls to production, not just a notebook demo.

One interface, every OpenAI-compatible provider
Switch between OpenAI, Azure, Groq, Gemini, Mistral, DeepSeek, Ollama, vLLM, and more — just change base_url + model. No rewrites, no provider-specific SDKs.

What makes it different from "just call openai.chat.completions.create()":

→ Reliability built in, not bolted on
Automatic retries with exponential backoff, a circuit breaker for cascading failures, and an automatic provider fallback chain — if your primary provider goes down, it fails over on its own. No proxy or gateway required.

→ Cost control that actually stops spend
Real-time cost/latency tracking plus a budget governor that hard-stops calls once you hit a USD cap. Not a dashboard you check after the bill arrives — a circuit that trips before it happens.

→ Structured output you can trust
Pydantic-validated structured output with an automatic validation-retry loop, plain JSON mode, and native tool/function calling — the model's mistakes get caught and corrected before they reach your code.

→ Security-conscious by default
Optional PII/secret redaction — a heuristic scrubber that masks or blocks emails, API keys, credit cards, SSNs, and phone numbers before they ever leave your process, with a bring-your-own-dictionary option for domain-specific data.

→ Observability without an external service
A local SQLite call ledger for a full audit trail, and shadow-mode dual dispatch to compare providers side-by-side on real traffic — zero new infrastructure.

→ Zero required dependencies
`pip install openai` and you're done. It works standalone or as part of the Autourgos agent framework — your choice.

This is the difference between "calling an LLM" and running one in production: retries, fallback, budget caps, redaction, and an audit trail are things every team eventually builds by hand. We built them once, tested them properly, and open-sourced them.

pip install autourgos-openaichat

#OpenSource #Python #LLM #AI #Agents #OpenAI #MachineLearning
