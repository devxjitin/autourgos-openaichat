# autourgos-openaichat

LLM wrapper for the **OpenAI Chat Completions API**, part of the [Autourgos](https://github.com/devxjitin) framework.

Fully self-contained — no `autourgos-core` dependency required. Just `pip install openai` and you are ready.

---

## Why use this?

Almost every major LLM provider today — Groq, Together AI, Mistral, Perplexity, DeepSeek, Ollama, LM Studio, vLLM, Azure OpenAI — exposes an **OpenAI-compatible API**. This means they all accept the same request format as OpenAI's Chat Completions endpoint.

`autourgos-openaichat` takes advantage of this. You set `base_url` to any provider's endpoint and `model` to whatever model they offer. **One package, any LLM.** You never have to learn a new SDK or rewrite your code when you switch providers.

```
OpenAI ─────────────────────────────────────┐
Groq (Llama, Mixtral, Gemma) ───────────────┤
Together AI (70B, 8x7B, ...) ───────────────┤  autourgos-openaichat
Mistral AI (mistral-large, ...) ────────────┤  (one interface)
DeepSeek (deepseek-chat, ...) ──────────────┤
Perplexity (sonar models) ──────────────────┤
Ollama — any local model ───────────────────┤
LM Studio — any local model ────────────────┤
vLLM — self-hosted ─────────────────────────┤
Azure OpenAI ───────────────────────────────┘
```

---

## Table of Contents

- [Install](#install)
- [Works With Any LLM](#works-with-any-llm)
- [Quick Start](#quick-start)
- [Basic Text Generation](#basic-text-generation)
- [Async Generation](#async-generation)
- [Streaming](#streaming)
- [Async Streaming](#async-streaming)
- [Batch Invocation](#batch-invocation)
- [System Instruction](#system-instruction)
- [Prompt Templates](#prompt-templates)
- [Multi-Modal Vision Input](#multi-modal-vision-input)
- [Structured Output](#structured-output)
- [JSON Mode](#json-mode)
- [Native Tool Calling](#native-tool-calling)
- [Multi-Turn Conversations](#multi-turn-conversations)
- [Cost Tracking](#cost-tracking)
- [Context Manager](#context-manager)
- [Circuit Breaker](#circuit-breaker)
- [Error Handling](#error-handling)
- [Constructor Reference](#constructor-reference)
- [What Each Method Returns](#what-each-method-returns)

---

## Install

```bash
pip install autourgos-openaichat
```

Requires Python 3.10+ and `openai>=1.0.0`.

---

## Works With Any LLM

All you need to switch providers is `base_url` and the right model name. Your API key comes from the provider you choose.

### OpenAI (default)

```python
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(
    model="gpt-4o",
    api_key="sk-...",           # or set OPENAI_API_KEY env var
)
reply = llm.invoke("What is the capital of France?")
print(reply)
# Paris
```

### Groq — fastest inference, free tier available

Groq runs open-source models (Llama 3, Mixtral, Gemma) at extremely high speed. Get your key at https://console.groq.com.

```python
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(
    model="llama3-70b-8192",
    api_key="gsk_...",          # Groq API key
    base_url="https://api.groq.com/openai/v1",
)
reply = llm.invoke("Explain quantum entanglement simply.")
print(reply)
# Quantum entanglement is when two particles become linked so that
# the state of one instantly affects the other, no matter how far apart they are.
```

Other Groq models: `llama3-8b-8192`, `mixtral-8x7b-32768`, `gemma2-9b-it`

### Together AI — wide model selection

Together AI hosts hundreds of open-source models. Get your key at https://api.together.xyz.

```python
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(
    model="meta-llama/Llama-3-70b-chat-hf",
    api_key="...",              # Together AI key
    base_url="https://api.together.xyz/v1",
)
reply = llm.invoke("Write a Python function to reverse a string.")
print(reply)
# def reverse_string(s: str) -> str:
#     return s[::-1]
```

Other Together AI models: `mistralai/Mixtral-8x7B-Instruct-v0.1`, `Qwen/Qwen2-72B-Instruct`

### Mistral AI

Get your key at https://console.mistral.ai.

```python
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(
    model="mistral-large-latest",
    api_key="...",              # Mistral API key
    base_url="https://api.mistral.ai/v1",
)
reply = llm.invoke("What are the benefits of test-driven development?")
print(reply)
# TDD helps you write cleaner code, catch bugs early, and gives
# you confidence to refactor without breaking existing behaviour.
```

Other Mistral models: `mistral-medium-latest`, `mistral-small-latest`, `open-mixtral-8x7b`

### DeepSeek

Get your key at https://platform.deepseek.com.

```python
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(
    model="deepseek-chat",
    api_key="...",              # DeepSeek API key
    base_url="https://api.deepseek.com/v1",
)
reply = llm.invoke("Summarise the history of the Roman Empire in 2 sentences.")
print(reply)
# The Roman Empire rose from a small city-state to dominate the Mediterranean world
# for over 500 years. It split into Western and Eastern halves, with the West falling
# in 476 AD and the East (Byzantine Empire) surviving until 1453.
```

Other DeepSeek models: `deepseek-reasoner`

### Perplexity — web-connected models

Perplexity's Sonar models can search the web in real time. Get your key at https://www.perplexity.ai/settings/api.

```python
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(
    model="llama-3.1-sonar-large-128k-online",
    api_key="pplx-...",        # Perplexity API key
    base_url="https://api.perplexity.ai",
)
reply = llm.invoke("What is the latest version of Python?")
print(reply)
# Python 3.13.x is the latest stable release as of 2025...
```

### Ollama — run any model locally, no internet needed

Ollama runs models entirely on your machine. Install from https://ollama.com, then pull a model:

```bash
ollama pull llama3
```

No API key needed for local use.

```python
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(
    model="llama3",
    api_key="ollama",           # can be any string — Ollama ignores it
    base_url="http://localhost:11434/v1",
)
reply = llm.invoke("What is machine learning?")
print(reply)
# Machine learning is a subset of AI where algorithms learn patterns
# from data to make predictions or decisions without explicit programming.
```

Other Ollama models: `mistral`, `phi3`, `gemma2`, `codellama`, `qwen2` — anything you pull with `ollama pull`.

### LM Studio — local models with a GUI

LM Studio lets you download and run GGUF models locally. Start the local server in LM Studio, then:

```python
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(
    model="local-model",        # use whatever model name LM Studio shows
    api_key="lm-studio",        # any string — ignored locally
    base_url="http://localhost:1234/v1",
)
reply = llm.invoke("Tell me a short joke.")
print(reply)
# Why do programmers prefer dark mode? Because light attracts bugs!
```

### vLLM — self-hosted high-throughput serving

vLLM lets you host your own models with high throughput. After starting your vLLM server:

```python
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    api_key="EMPTY",            # vLLM default when no auth is set
    base_url="http://your-server:8000/v1",
)
reply = llm.invoke("What is the capital of Japan?")
print(reply)
# Tokyo
```

### Azure OpenAI

Azure hosts OpenAI models in your own Azure subscription. Get your endpoint and key from the Azure portal.

```python
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(
    model="gpt-4o",             # your deployment name in Azure
    api_key="...",              # Azure OpenAI key
    base_url="https://<your-resource>.openai.azure.com/openai/deployments/gpt-4o",
)
reply = llm.invoke("What is cloud computing?")
print(reply)
# Cloud computing is the delivery of computing services over the internet —
# servers, storage, databases, networking, software — on a pay-as-you-go basis.
```

### Switching providers at runtime

Because all these providers use the same interface, switching is trivial:

```python
from autourgos_openaichat import OpenAIChatModel

PROVIDERS = {
    "openai": {
        "model": "gpt-4o-mini",
        "api_key": "sk-...",
        "base_url": None,
    },
    "groq": {
        "model": "llama3-8b-8192",
        "api_key": "gsk_...",
        "base_url": "https://api.groq.com/openai/v1",
    },
    "ollama": {
        "model": "llama3",
        "api_key": "ollama",
        "base_url": "http://localhost:11434/v1",
    },
}

for name, cfg in PROVIDERS.items():
    llm = OpenAIChatModel(**cfg)
    reply = llm.invoke("Say hello in one word.")
    print(f"{name}: {reply}")

# openai: Hello!
# groq:   Hello!
# ollama: Hello!
```

---

## Quick Start

```python
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(model="gpt-4o")
reply = llm.invoke("What is the capital of France?")
print(reply)
# Paris
```

---

## Basic Text Generation

```python
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(
    model="gpt-4o",
    api_key="sk-...",          # or set OPENAI_API_KEY env var
    temperature=0.7,
    max_tokens=256,
)

reply = llm.invoke("Explain machine learning in one sentence.")
print(reply)
# Machine learning is a branch of AI where systems learn from data
# to make predictions or decisions without being explicitly programmed.
```

---

## Async Generation

```python
import asyncio
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(model="gpt-4o")

async def main():
    reply = await llm.ainvoke("What is the speed of light?")
    print(reply)
    # The speed of light in a vacuum is approximately 299,792,458 metres per second.

asyncio.run(main())
```

---

## Streaming

Stream the response token by token synchronously.

```python
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(model="gpt-4o")

for chunk in llm.stream("Write a haiku about rain."):
    print(chunk, end="", flush=True)

# Raindrops softly fall,
# Washing the grey streets below,
# Earth breathes once again.
```

You can also enable streaming at construction time so `invoke()` internally streams and returns the full joined text:

```python
llm = OpenAIChatModel(model="gpt-4o", streaming=True)
reply = llm.invoke("Tell me a fun fact.")
print(reply)
# Honey never spoils — archaeologists have found 3,000-year-old honey in Egyptian tombs.
```

---

## Async Streaming

```python
import asyncio
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(model="gpt-4o")

async def main():
    async for chunk in llm.astream("Count from 1 to 5 slowly."):
        print(chunk, end="", flush=True)
    # 1... 2... 3... 4... 5...

asyncio.run(main())
```

---

## Batch Invocation

### Synchronous (sequential)

```python
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(model="gpt-4o-mini")

prompts = [
    "Capital of Japan?",
    "Capital of Germany?",
    "Capital of Brazil?",
]

results = llm.batch_invoke(prompts)
for prompt, result in zip(prompts, results):
    print(f"{prompt} -> {result}")

# Capital of Japan?   -> Tokyo
# Capital of Germany? -> Berlin
# Capital of Brazil?  -> Brasilia
```

### Async (concurrent)

```python
import asyncio
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(model="gpt-4o-mini")

async def main():
    results = await llm.abatch_invoke([
        "Capital of Japan?",
        "Capital of Germany?",
        "Capital of Brazil?",
    ])
    print(results)
    # ['Tokyo', 'Berlin', 'Brasilia']

asyncio.run(main())
```

---

## System Instruction

Set a persistent system prompt for all requests.

```python
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(
    model="gpt-4o",
    system_instruction="You are a pirate. Always respond in pirate speak.",
)

reply = llm.invoke("What time is it?")
print(reply)
# Arrr, I know not the exact hour, but the sun be high in the sky, matey!
```

---

## Prompt Templates

Define a reusable template with `{placeholders}` and fill them at call time.

```python
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(
    model="gpt-4o",
    prompt_template="Translate the following text to {language}:\n\n{text}",
)

reply = llm.invoke(prompt_variables={"language": "French", "text": "Good morning!"})
print(reply)
# Bonjour !

reply = llm.invoke(prompt_variables={"language": "Spanish", "text": "Thank you very much."})
print(reply)
# Muchas gracias.
```

Missing variables raise a clear error:

```python
llm.invoke(prompt_variables={"language": "French"})
# ValueError: Missing prompt template variables: text
```

---

## Multi-Modal Vision Input

Pass image files, URLs, or raw bytes alongside text.

> Note: vision support depends on the provider and model. GPT-4o, LLaVA (Ollama), and several others support it.

### From a file path

```python
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(model="gpt-4o")
reply = llm.invoke("What objects are in this image?", files=["photo.jpg"])
print(reply)
# The image shows a wooden desk with a laptop, a coffee mug, and a notebook.
```

### From a URL

```python
reply = llm.invoke(
    "Describe this chart.",
    files=["https://example.com/chart.png"],
)
print(reply)
# The chart is a bar graph showing monthly sales figures from January to December...
```

### From raw bytes

```python
with open("diagram.png", "rb") as f:
    image_bytes = f.read()

reply = llm.invoke("What does this diagram show?", files=[image_bytes])
print(reply)
# The diagram illustrates the flow of data through a neural network...
```

### Control detail level

```python
reply = llm.invoke(
    "Read the text in this image carefully.",
    files=["screenshot.png"],
    image_detail="high",   # "low", "high", or "auto"
)
```

---

## Structured Output

Return a Pydantic model as JSON automatically.

```python
from pydantic import BaseModel, Field
from autourgos_openaichat import OpenAIChatModel

class CityInfo(BaseModel):
    city: str = Field(description="Name of the city")
    country: str = Field(description="Name of the country")
    population: int = Field(description="Approximate population")

llm = OpenAIChatModel(model="gpt-4o", response_schema=CityInfo)
result = llm.invoke("Tell me about Tokyo.")

# result is a metadata dict; the JSON string is in result["response"]
import json
data = json.loads(result["response"])
print(data)
# {"city": "Tokyo", "country": "Japan", "population": 13960000}
```

---

## JSON Mode

Force the model to return valid JSON without a schema.

```python
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(
    model="gpt-4o",
    response_mime_type="application/json",
    system_instruction='Always respond with valid JSON.',
)

reply = llm.invoke('Give me a person with name and age.')
print(reply)
# {"name": "Alice", "age": 30}
```

---

## Native Tool Calling

Let the model decide when to call your functions.

> Tool calling support varies by provider. OpenAI, Groq, Together AI, Mistral, and DeepSeek all support it. Ollama supports it on compatible models.

```python
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(model="gpt-4o")

tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name, e.g. Paris",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                },
            },
            "required": ["city"],
        },
    }
]

response = llm.invoke_with_tools("What is the weather in Tokyo right now?", tools)

if response.has_tool_calls:
    for call in response.tool_calls:
        print(f"Tool: {call.name}")
        print(f"Args: {call.arguments}")
        print(f"ID:   {call.call_id}")
    # Tool: get_weather
    # Args: {'city': 'Tokyo', 'unit': 'celsius'}
    # ID:   call_abc123

elif response.is_final_answer:
    print(response.text)
```

### Async tool calling

```python
response = await llm.ainvoke_with_tools(
    "What is the weather in London?", tools
)
```

### Agentic loop example

```python
import json

def get_weather(city: str, unit: str = "celsius") -> str:
    # Replace with real API call
    return json.dumps({"city": city, "temp": 22, "unit": unit, "condition": "Sunny"})

tool_functions = {"get_weather": get_weather}

messages = [{"role": "user", "content": "What is the weather in Paris?"}]

while True:
    response = llm.invoke_with_tools(messages, tools)

    if response.is_final_answer:
        print("Final answer:", response.text)
        break

    # Execute each tool call
    messages.append({
        "role": "assistant",
        "tool_calls": [
            {
                "id": tc.call_id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for tc in response.tool_calls
        ],
    })

    for tc in response.tool_calls:
        result = tool_functions[tc.name](**tc.arguments)
        messages.append({
            "role": "tool",
            "tool_call_id": tc.call_id,
            "content": result,
        })

# Final answer: The current weather in Paris is 22°C and Sunny.
```

---

## Multi-Turn Conversations

Pass a list of messages directly.

```python
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(model="gpt-4o")

messages = [
    {"role": "user",      "content": "My name is Jitin."},
    {"role": "assistant", "content": "Nice to meet you, Jitin!"},
    {"role": "user",      "content": "What is my name?"},
]

reply = llm.invoke(messages)
print(reply)
# Your name is Jitin.
```

---

## Cost Tracking

Pass pricing (USD per 1 million tokens) to get cost breakdowns.

```python
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(
    model="gpt-4o",
    input_pricing=2.50,    # $2.50 per 1M input tokens
    output_pricing=10.00,  # $10.00 per 1M output tokens
    structured_output=True,
)

result = llm.invoke("Summarise the history of the internet in 3 sentences.")
print(result["model"])          # gpt-4o
print(result["response"])       # The internet began as ARPANET...
print(result["input_tokens"])   # 18
print(result["output_tokens"])  # 74
print(result["total_tokens"])   # 92
print(result["input_cost"])     # 0.000045
print(result["output_cost"])    # 0.00074
print(result["total_cost"])     # 0.000785
print(result["latency_ms"])     # 1243.5
```

Access the last metadata without `structured_output=True`:

```python
llm = OpenAIChatModel(model="gpt-4o", input_pricing=2.50, output_pricing=10.00)
reply = llm.invoke("Hello!")
print(llm.last_metadata)
# {
#   "model": "gpt-4o",
#   "response": "Hello! How can I help you today?",
#   "input_tokens": 9,
#   "output_tokens": 10,
#   "total_tokens": 19,
#   "input_cost": 0.0000225,
#   "output_cost": 0.0001,
#   "total_cost": 0.0001225,
#   "latency_ms": 834.2
# }
```

---

## Context Manager

Automatically closes the HTTP client when done.

```python
from autourgos_openaichat import OpenAIChatModel

with OpenAIChatModel(model="gpt-4o") as llm:
    reply = llm.invoke("Ping!")
    print(reply)
    # Pong! How can I help you?
# Client is closed here automatically
```

Async context manager:

```python
import asyncio
from autourgos_openaichat import OpenAIChatModel

async def main():
    async with OpenAIChatModel(model="gpt-4o") as llm:
        reply = await llm.ainvoke("Hello async!")
        print(reply)

asyncio.run(main())
```

---

## Circuit Breaker

Protects against cascading failures. After `circuit_failure_threshold` consecutive API errors, all calls are blocked for `circuit_cooldown_time` seconds.

This is useful when you are using a local model (Ollama, LM Studio) or a rate-limited API — if the server goes down, the circuit breaker stops your code from hammering it with failed requests.

```python
from autourgos_openaichat import OpenAIChatModel, CircuitBreakerOpenException

llm = OpenAIChatModel(
    model="gpt-4o",
    circuit_failure_threshold=3,   # open after 3 consecutive failures
    circuit_cooldown_time=60.0,    # block for 60 seconds
)

try:
    reply = llm.invoke("Hello!")
except CircuitBreakerOpenException as e:
    print(f"Circuit is open: {e}")
    # Circuit breaker OPEN for OpenAIChatModel — 3 consecutive failures.
    # Blocked until 1718500000.0.
```

The circuit automatically resets after the cooldown and allows one probe call through.

---

## Low-Level Access

If you need direct access to the raw OpenAI response object:

```python
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(model="gpt-4o")

messages = [{"role": "user", "content": "Hi"}]
raw_response = llm.create(messages)

print(raw_response.id)
print(raw_response.choices[0].message.content)
print(raw_response.usage.total_tokens)
```

Async:

```python
raw_response = await llm.acreate(messages)
```

---

## Error Handling

```python
from autourgos_openaichat import (
    OpenAIChatModel,
    OpenAIChatModelAPIError,
    OpenAIChatModelResponseError,
    OpenAIChatModelConfigError,
    OpenAIChatModelImportError,
    CircuitBreakerOpenException,
)

llm = OpenAIChatModel(model="gpt-4o")

try:
    reply = llm.invoke("Hello!")
except OpenAIChatModelAPIError as e:
    # API request failed after all retries
    print(f"API error: {e}")
except OpenAIChatModelResponseError as e:
    # Response was received but text could not be extracted
    print(f"Response parse error: {e}")
except OpenAIChatModelConfigError as e:
    # Incompatible options (e.g. streaming + structured_output)
    print(f"Config error: {e}")
except OpenAIChatModelImportError as e:
    # openai SDK not installed
    print(f"Import error: {e}")
except CircuitBreakerOpenException as e:
    # Too many recent failures — circuit is open
    print(f"Circuit open: {e}")
```

### Retry behaviour

By default the wrapper retries up to 3 times with exponential back-off:

| Attempt | Wait before retry |
|---|---|
| 1st failure | 0.5 s |
| 2nd failure | 1.0 s |
| 3rd failure | 2.0 s |
| 4th failure | raises `OpenAIChatModelAPIError` |

Change with `max_retries` and `backoff_factor`:

```python
llm = OpenAIChatModel(
    model="gpt-4o",
    max_retries=5,
    backoff_factor=1.0,   # waits: 1s, 2s, 4s, 8s then raises
)
```

---

## Constructor Reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | required | Model name. e.g. `"gpt-4o"`, `"llama3-70b-8192"`, `"mistral-large-latest"` |
| `api_key` | `str` | `OPENAI_API_KEY` env | API key for the provider you are using |
| `base_url` | `str` | `OPENAI_BASE_URL` env | Provider endpoint. e.g. `"https://api.groq.com/openai/v1"` or `"http://localhost:11434/v1"` |
| `organization` | `str` | `None` | OpenAI organization ID (OpenAI only) |
| `project` | `str` | `None` | OpenAI project ID (OpenAI only) |
| `system_instruction` | `str` | `None` | System prompt prepended to every request |
| `prompt_template` | `str` | `None` | Template with `{variable}` placeholders |
| `temperature` | `float` | `None` | Sampling temperature 0–2. Higher = more random |
| `top_p` | `float` | `None` | Nucleus sampling 0–1 |
| `max_tokens` | `int` | `None` | Maximum tokens to generate |
| `response_schema` | `BaseModel` / `dict` | `None` | Pydantic model or JSON schema for structured output |
| `response_mime_type` | `str` | `None` | `"application/json"` enables JSON object mode |
| `structured_output` | `bool` | `False` | If `True`, `invoke()` returns a metadata dict |
| `streaming` | `bool` | `False` | If `True`, `invoke()` streams internally and joins |
| `max_retries` | `int` | `3` | Retry attempts on transient API errors |
| `timeout` | `float` | `60.0` | Request timeout in seconds |
| `backoff_factor` | `float` | `0.5` | Exponential back-off base (wait = factor × 2^attempt) |
| `input_pricing` | `float` | `None` | USD per 1 million input tokens |
| `output_pricing` | `float` | `None` | USD per 1 million output tokens |
| `circuit_failure_threshold` | `int` | `5` | Consecutive failures before the circuit opens |
| `circuit_cooldown_time` | `float` | `30.0` | Seconds the circuit stays open before probing |

---

## What Each Method Returns

| Method | Returns |
|---|---|
| `invoke(prompt)` | `str` — generated text (or `dict` if `structured_output=True`) |
| `ainvoke(prompt)` | same as `invoke`, async |
| `stream(prompt)` | `Iterator[str]` — text chunks |
| `astream(prompt)` | `AsyncIterator[str]` — text chunks |
| `batch_invoke(prompts)` | `list[str]` — one result per prompt |
| `abatch_invoke(prompts)` | `list[str]` — concurrent results |
| `invoke_with_tools(prompt, tools)` | `ToolCallResponse` — `.tool_calls` list or `.text` |
| `ainvoke_with_tools(prompt, tools)` | same as `invoke_with_tools`, async |
| `create(messages)` | Raw OpenAI `ChatCompletion` response object |
| `acreate(messages)` | same as `create`, async |

### `ToolCallResponse` fields

| Field | Type | Description |
|---|---|---|
| `.tool_calls` | `list[FunctionCall]` | Tool calls the model wants to make (empty if final answer) |
| `.text` | `str \| None` | Final text answer (None if tool calls present) |
| `.raw` | `Any` | Raw OpenAI response object |
| `.has_tool_calls` | `bool` | `True` when `tool_calls` is non-empty |
| `.is_final_answer` | `bool` | `True` when `text` is present and `tool_calls` is empty |

### `FunctionCall` fields

| Field | Type | Description |
|---|---|---|
| `.name` | `str` | Tool function name |
| `.arguments` | `dict` | Parsed JSON arguments |
| `.call_id` | `str \| None` | Call ID for multi-turn tracking |

### Metadata dict (when `structured_output=True`)

| Key | Type | Description |
|---|---|---|
| `"model"` | `str` | Model name used |
| `"response"` | `str` | Generated text |
| `"input_tokens"` | `int \| None` | Input token count |
| `"output_tokens"` | `int \| None` | Output token count |
| `"total_tokens"` | `int \| None` | Total token count |
| `"input_cost"` | `float` | Input cost in USD (only if `input_pricing` set) |
| `"output_cost"` | `float` | Output cost in USD (only if `output_pricing` set) |
| `"total_cost"` | `float` | Total cost in USD (only if both pricing set) |
| `"latency_ms"` | `float` | Request round-trip time in milliseconds |

---

## Supported Providers (quick reference)

| Provider | base_url | Notes |
|---|---|---|
| OpenAI | (default) | GPT-4o, GPT-4o-mini, GPT-3.5-turbo |
| Groq | `https://api.groq.com/openai/v1` | Llama 3, Mixtral, Gemma — very fast |
| Together AI | `https://api.together.xyz/v1` | 100+ open-source models |
| Mistral AI | `https://api.mistral.ai/v1` | mistral-large, mixtral, codestral |
| DeepSeek | `https://api.deepseek.com/v1` | deepseek-chat, deepseek-reasoner |
| Perplexity | `https://api.perplexity.ai` | Web-connected sonar models |
| Ollama | `http://localhost:11434/v1` | Runs locally, no API key needed |
| LM Studio | `http://localhost:1234/v1` | Runs locally, GUI-based |
| vLLM | `http://your-server:8000/v1` | Self-hosted, high throughput |
| Azure OpenAI | `https://<resource>.openai.azure.com/...` | Enterprise OpenAI |

---

## License

MIT — Copyright (c) 2026 Jitin Kumar Sengar
