# autourgos-openaichat

[![Framework: Autourgos](https://img.shields.io/badge/Framework-Autourgos-orange.svg)](https://github.com/devxjitin)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://pypi.org/project/autourgos-openaichat/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://github.com/devxjitin/autourgos-openaichat/blob/main/LICENSE)
[![Author](https://img.shields.io/badge/Author-Jitin%20Kumar%20Sengar-blue.svg)](https://github.com/devxjitin)
[![Contributor](https://img.shields.io/badge/Contributor-Sonia-blueviolet.svg)]()
[![Contributor](https://img.shields.io/badge/Contributor-Vishwanil%20Suman-blueviolet.svg)]()

A single, self-contained LLM wrapper for the **OpenAI Chat Completions API**, and by extension every provider that speaks the same protocol (Groq, Gemini, Azure, Ollama, and more). Part of the [Autourgos](https://github.com/devxjitin) agentic-AI framework, but has zero dependency on it: `pip install openai` and you're ready.

```python
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(model="gpt-4o")           # reads OPENAI_API_KEY
reply = llm.invoke("What is the capital of France?")
print(reply)
# Paris
```

---

## Features

- **One interface, any OpenAI-compatible provider**: OpenAI, Azure, Groq, Gemini, Mistral, DeepSeek, Ollama, and more, switched with just `base_url` + `model`
- Sync and async generation, plus streaming for both
- Native tool / function calling, sync and async
- Structured output validated against a Pydantic model, or plain JSON mode
- Multi-modal vision input: file paths, URLs, or raw bytes
- Prompt templates with `{placeholder}` variables
- Multi-turn conversations via a plain message list
- Automatic retries with exponential back-off, plus a circuit breaker for cascading-failure protection
- Automatic provider fallback chain: define backup providers and `invoke()` transparently switches to them if the primary fails, no proxy/gateway needed
- Built-in cost and latency tracking
- Fully typed (`py.typed`), sync/async context managers, low-level raw-response access

---

## Table of Contents

- [Install](#install)
- [Supported Providers](#supported-providers)
- [Provider Examples](#provider-examples)
  - [OpenAI](#openai)
  - [Azure OpenAI](#azure-openai)
  - [Google Gemini](#google-gemini)
  - [Groq](#groq-fastest-inference-free-tier-available)
  - [xAI (Grok)](#xai-grok)
  - [OpenRouter](#openrouter-one-key-hundreds-of-models)
  - [Together AI](#together-ai-wide-model-selection)
  - [Mistral AI](#mistral-ai)
  - [DeepSeek](#deepseek)
  - [Perplexity](#perplexity-web-connected-models)
  - [Ollama](#ollama-run-any-model-locally-no-internet-needed)
  - [LM Studio](#lm-studio-local-models-with-a-gui)
  - [vLLM](#vllm-self-hosted-high-throughput-serving)
  - [Switching providers at runtime](#switching-providers-at-runtime)
- [Core Usage](#core-usage)
  - [Text Generation](#text-generation)
  - [Async Generation](#async-generation)
  - [Streaming](#streaming)
  - [Async Streaming](#async-streaming)
  - [Batch Invocation](#batch-invocation)
  - [System Prompt](#system-prompt)
  - [Prompt Templates](#prompt-templates)
  - [Vision Input](#vision-input)
  - [Structured Output](#structured-output)
  - [JSON Mode](#json-mode)
  - [Native Tool Calling](#native-tool-calling)
  - [Multi-Turn Conversations](#multi-turn-conversations)
  - [Cost Tracking](#cost-tracking)
  - [Context Manager](#context-manager)
  - [Circuit Breaker](#circuit-breaker)
  - [Provider Fallback Chain](#provider-fallback-chain)
  - [Low-Level Access](#low-level-access)
  - [Error Handling](#error-handling)
- [Constructor Reference](#constructor-reference)
- [API Reference](#api-reference)
- [License](#license)

---

## Install

```bash
pip install autourgos-openaichat
```

Requires Python 3.10+ and `openai>=1.0.0`. Structured output (`output_schema=`) additionally needs `pydantic>=2.0` if you use it.

---

## Supported Providers

Almost every major LLM provider exposes an **OpenAI-compatible API**: same request format as OpenAI's Chat Completions endpoint. Point `base_url` at the provider and `model` at whatever they offer; nothing else changes.

| Provider | `base_url` | Get a key |
|---|---|---|
| OpenAI | *(default, omit)* | https://platform.openai.com/api-keys |
| Azure OpenAI | `https://<resource>.openai.azure.com/openai/deployments/<deployment>` | Azure Portal |
| Google Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` | https://aistudio.google.com/apikey |
| Groq | `https://api.groq.com/openai/v1` | https://console.groq.com |
| xAI (Grok) | `https://api.x.ai/v1` | https://console.x.ai |
| OpenRouter | `https://openrouter.ai/api/v1` | https://openrouter.ai/keys |
| Together AI | `https://api.together.xyz/v1` | https://api.together.xyz |
| Mistral AI | `https://api.mistral.ai/v1` | https://console.mistral.ai |
| DeepSeek | `https://api.deepseek.com/v1` | https://platform.deepseek.com |
| Perplexity | `https://api.perplexity.ai` | https://www.perplexity.ai/settings/api |
| Ollama (local) | `http://localhost:11434/v1` | none, runs on your machine |
| LM Studio (local) | `http://localhost:1234/v1` | none, runs on your machine |
| vLLM (self-hosted) | `http://your-server:8000/v1` | none, you host it |

---

## Provider Examples

Every example below is the full, runnable snippet. Swap in your own key and go.

### OpenAI

The default provider. No `base_url` needed.

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

### Azure OpenAI

Azure hosts OpenAI models in your own subscription. `model` is your **deployment name** in Azure, not the base model name. Get your endpoint and key from the Azure Portal.

```python
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(
    model="gpt-4o",              # your deployment name in Azure
    api_key="...",               # Azure OpenAI key
    base_url="https://<your-resource>.openai.azure.com/openai/deployments/gpt-4o",
)
reply = llm.invoke("What is cloud computing?")
print(reply)
# Cloud computing is the delivery of computing services over the internet
# (servers, storage, databases, networking, software) on a pay-as-you-go basis.
```

### Google Gemini

Gemini exposes an OpenAI-compatible endpoint, so no separate Google SDK is needed. Get your key at https://aistudio.google.com/apikey.

```python
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(
    model="gemini-2.0-flash",
    api_key="...",               # Gemini API key
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
reply = llm.invoke("Explain photosynthesis in one sentence.")
print(reply)
# Photosynthesis is the process by which plants convert sunlight, water, and
# carbon dioxide into glucose and oxygen.
```

Other Gemini models: `gemini-2.0-flash-lite`, `gemini-1.5-pro`, `gemini-1.5-flash`.

### Groq (fastest inference, free tier available)

Groq runs open-source models (Llama 3, Mixtral, Gemma) at extremely high speed. Get your key at https://console.groq.com.

```python
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(
    model="llama3-70b-8192",
    api_key="gsk_...",           # Groq API key
    base_url="https://api.groq.com/openai/v1",
)
reply = llm.invoke("Explain quantum entanglement simply.")
print(reply)
# Quantum entanglement is when two particles become linked so that
# the state of one instantly affects the other, no matter how far apart they are.
```

Other Groq models: `llama3-8b-8192`, `mixtral-8x7b-32768`, `gemma2-9b-it`.

### xAI (Grok)

Get your key at https://console.x.ai.

```python
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(
    model="grok-2-latest",
    api_key="xai-...",           # xAI API key
    base_url="https://api.x.ai/v1",
)
reply = llm.invoke("What makes Mars red?")
print(reply)
# Mars appears red because its surface is covered in iron oxide (rust),
# formed when iron in the soil reacted with trace oxygen long ago.
```

### OpenRouter (one key, hundreds of models)

OpenRouter proxies dozens of providers (including Anthropic Claude and Google Gemini) behind a single OpenAI-compatible API and one API key. Get your key at https://openrouter.ai/keys.

```python
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(
    model="anthropic/claude-3.5-sonnet",   # or "google/gemini-2.0-flash-001", "openai/gpt-4o", ...
    api_key="sk-or-...",         # OpenRouter API key
    base_url="https://openrouter.ai/api/v1",
)
reply = llm.invoke("Write a Python one-liner to reverse a string.")
print(reply)
# s[::-1]
```

### Together AI (wide model selection)

Together AI hosts hundreds of open-source models. Get your key at https://api.together.xyz.

```python
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(
    model="meta-llama/Llama-3-70b-chat-hf",
    api_key="...",                # Together AI key
    base_url="https://api.together.xyz/v1",
)
reply = llm.invoke("Write a Python function to reverse a string.")
print(reply)
# def reverse_string(s: str) -> str:
#     return s[::-1]
```

Other Together AI models: `mistralai/Mixtral-8x7B-Instruct-v0.1`, `Qwen/Qwen2-72B-Instruct`.

### Mistral AI

Get your key at https://console.mistral.ai.

```python
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(
    model="mistral-large-latest",
    api_key="...",                # Mistral API key
    base_url="https://api.mistral.ai/v1",
)
reply = llm.invoke("What are the benefits of test-driven development?")
print(reply)
# TDD helps you write cleaner code, catch bugs early, and gives
# you confidence to refactor without breaking existing behaviour.
```

Other Mistral models: `mistral-medium-latest`, `mistral-small-latest`, `open-mixtral-8x7b`.

### DeepSeek

Get your key at https://platform.deepseek.com.

```python
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(
    model="deepseek-chat",
    api_key="...",                # DeepSeek API key
    base_url="https://api.deepseek.com/v1",
)
reply = llm.invoke("Summarise the history of the Roman Empire in 2 sentences.")
print(reply)
# The Roman Empire rose from a small city-state to dominate the Mediterranean world
# for over 500 years. It split into Western and Eastern halves, with the West falling
# in 476 AD and the East (Byzantine Empire) surviving until 1453.
```

Other DeepSeek models: `deepseek-reasoner`.

### Perplexity (web-connected models)

Perplexity's Sonar models can search the web in real time. Get your key at https://www.perplexity.ai/settings/api.

```python
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(
    model="llama-3.1-sonar-large-128k-online",
    api_key="pplx-...",           # Perplexity API key
    base_url="https://api.perplexity.ai",
)
reply = llm.invoke("What is the latest version of Python?")
print(reply)
# Python 3.13.x is the latest stable release as of 2025...
```

### Ollama (run any model locally, no internet needed)

Ollama runs models entirely on your machine. Install from https://ollama.com, then pull a model:

```bash
ollama pull llama3
```

No API key needed for local use.

```python
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(
    model="llama3",
    api_key="ollama",             # can be any string, Ollama ignores it
    base_url="http://localhost:11434/v1",
)
reply = llm.invoke("What is machine learning?")
print(reply)
# Machine learning is a subset of AI where algorithms learn patterns
# from data to make predictions or decisions without explicit programming.
```

Other Ollama models: `mistral`, `phi3`, `gemma2`, `codellama`, `qwen2`, and anything you pull with `ollama pull`.

### LM Studio (local models with a GUI)

LM Studio lets you download and run GGUF models locally. Start the local server in LM Studio, then:

```python
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(
    model="local-model",          # use whatever model name LM Studio shows
    api_key="lm-studio",          # any string, ignored locally
    base_url="http://localhost:1234/v1",
)
reply = llm.invoke("Tell me a short joke.")
print(reply)
# Why do programmers prefer dark mode? Because light attracts bugs!
```

### vLLM (self-hosted high-throughput serving)

vLLM lets you host your own models with high throughput. After starting your vLLM server:

```python
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    api_key="EMPTY",              # vLLM's default when no auth is configured
    base_url="http://your-server:8000/v1",
)
reply = llm.invoke("What is the capital of Japan?")
print(reply)
# Tokyo
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
    "gemini": {
        "model": "gemini-2.0-flash",
        "api_key": "...",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
}

for name, cfg in PROVIDERS.items():
    llm = OpenAIChatModel(**cfg)
    reply = llm.invoke("Say hello in one word.")
    print(f"{name}: {reply}")

# openai: Hello!
# groq:   Hello!
# gemini: Hello!
```

---

## Core Usage

### Text Generation

```python
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(
    model="gpt-4o",
    api_key="sk-...",             # or set OPENAI_API_KEY env var
    temperature=0.7,
    max_tokens=256,
)

reply = llm.invoke("Explain machine learning in one sentence.")
print(reply)
# Machine learning is a branch of AI where systems learn from data
# to make predictions or decisions without being explicitly programmed.
```

### Async Generation

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

### Streaming

Stream the response token by token, synchronously.

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
# Honey never spoils. Archaeologists have found 3,000-year-old honey in Egyptian tombs.
```

### Async Streaming

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

### Batch Invocation

Synchronous (sequential):

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

Async (concurrent):

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

### System Prompt

Set a persistent system prompt for all requests.

```python
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(
    model="gpt-4o",
    system_prompt="You are a pirate. Always respond in pirate speak.",
)

reply = llm.invoke("What time is it?")
print(reply)
# Arrr, I know not the exact hour, but the sun be high in the sky, matey!
```

### Prompt Templates

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

### Vision Input

Pass image files, URLs, or raw bytes alongside text.

> Note: vision support depends on the provider and model. GPT-4o, Gemini, LLaVA (on Ollama), and several others support it.

> **Warning:** the file-path branch reads whatever local path it's given and base64-embeds its contents into the outgoing API request, with no path validation. Do not pass LLM- or tool-controlled paths through unchecked. An unchecked path could be used to exfiltrate arbitrary local files.

From a file path:

```python
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(model="gpt-4o")
reply = llm.invoke("What objects are in this image?", files=["photo.jpg"])
print(reply)
# The image shows a wooden desk with a laptop, a coffee mug, and a notebook.
```

From a URL:

```python
reply = llm.invoke(
    "Describe this chart.",
    files=["https://example.com/chart.png"],
)
print(reply)
# The chart is a bar graph showing monthly sales figures from January to December...
```

From raw bytes:

```python
with open("diagram.png", "rb") as f:
    image_bytes = f.read()

reply = llm.invoke("What does this diagram show?", files=[image_bytes])
print(reply)
# The diagram illustrates the flow of data through a neural network...
```

Control the detail level:

```python
reply = llm.invoke(
    "Read the text in this image carefully.",
    files=["screenshot.png"],
    image_detail="high",   # "low", "high", or "auto"
)
print(reply)
# The screenshot shows a terminal window with the command "pip install autourgos-openaichat" ...
```

### Structured Output

Return a Pydantic model as JSON automatically.

```python
from pydantic import BaseModel, Field
from autourgos_openaichat import OpenAIChatModel
import json

class CityInfo(BaseModel):
    city: str = Field(description="Name of the city")
    country: str = Field(description="Name of the country")
    population: int = Field(description="Approximate population")

llm = OpenAIChatModel(model="gpt-4o", output_schema=CityInfo)
result = llm.invoke("Tell me about Tokyo.")

# result is a metadata dict; the JSON string is in result["response"]
data = json.loads(result["response"])
print(data)
# {"city": "Tokyo", "country": "Japan", "population": 13960000}
```

### JSON Mode

Force the model to return valid JSON without a schema.

```python
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(
    model="gpt-4o",
    response_mime_type="application/json",
    system_prompt="Always respond with valid JSON.",
)

reply = llm.invoke("Give me a person with name and age.")
print(reply)
# {"name": "Alice", "age": 30}
```

### Native Tool Calling

Let the model decide when to call your functions.

> Tool calling support varies by provider. OpenAI, Groq, Gemini, Together AI, Mistral, and DeepSeek all support it. Ollama supports it on compatible models.

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

Async tool calling:

```python
response = await llm.ainvoke_with_tools(
    "What is the weather in London?", tools
)
```

Full agentic loop example:

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

### Multi-Turn Conversations

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

### Cost Tracking

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

### Context Manager

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

### Circuit Breaker

Protects against cascading failures. After `circuit_failure_threshold` consecutive API errors, all calls are blocked for `circuit_cooldown_time` seconds.

This is useful when you are using a local model (Ollama, LM Studio) or a rate-limited API. If the server goes down, the circuit breaker stops your code from hammering it with failed requests.

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
    # Circuit breaker OPEN for OpenAIChatModel: 3 consecutive failures.
    # Blocked until 1718500000.0.
```

The circuit automatically resets after the cooldown and allows one probe call through.

### Provider Fallback Chain

Configure backup providers that `invoke()`, `ainvoke()`, `stream()`, `astream()`, `invoke_with_tools()`, and `ainvoke_with_tools()` transparently switch to if the primary provider fails (after its own retries are exhausted) — no proxy or gateway service needed.

```python
from autourgos_openaichat import OpenAIChatModel

llm = OpenAIChatModel(
    model="gpt-4o",
    api_key="sk-...",                       # primary: OpenAI
    fallback_providers=[
        {
            "model": "llama3-70b-8192",     # 1st backup: Groq
            "api_key": "gsk_...",
            "base_url": "https://api.groq.com/openai/v1",
        },
        {
            "model": "llama3",              # 2nd backup: local Ollama
            "api_key": "ollama",
            "base_url": "http://localhost:11434/v1",
        },
    ],
)

reply = llm.invoke("What is the capital of France?")
print(reply)
# Paris (served by whichever provider succeeded first)

print(llm.last_metadata["provider_used"])
# "primary"  or  "fallback[0]:llama3-70b-8192"  or  "fallback[1]:llama3"
```

Each fallback entry resolves its own `api_key`/`base_url` (falling back to `OPENAI_API_KEY`/`OPENAI_BASE_URL` env vars, exactly like the primary) — nothing is inherited from the primary provider's credentials, so a backup on a different host never sees the primary's key.

If every provider fails, `OpenAIChatModelAllProvidersFailedError` (a subclass of `OpenAIChatModelAPIError`) is raised with an `.attempts` list of `(label, exception)` pairs, one per provider tried:

```python
from autourgos_openaichat import OpenAIChatModelAllProvidersFailedError

try:
    llm.invoke("Hello!")
except OpenAIChatModelAllProvidersFailedError as e:
    for label, exc in e.attempts:
        print(f"{label}: {exc}")
    # primary: [primary] Chat Completions request failed after 3 attempts. ...
    # fallback[0]:llama3-70b-8192: [fallback[0]:llama3-70b-8192] Chat Completions request failed ...
```

**Streaming limitation:** fallback only kicks in if a provider fails *before* it has streamed any text. Once partial output has already reached the caller, switching providers mid-stream would duplicate or corrupt the output, so the error is raised as-is instead of silently trying the next provider.

`create()`/`acreate()` (low-level raw access) are unaffected by `fallback_providers` — they always call the primary client only, since their contract is "the raw response of the client you configured."

### Low-Level Access

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

### Error Handling

```python
from autourgos_openaichat import (
    OpenAIChatModel,
    OpenAIChatModelAPIError,
    OpenAIChatModelAllProvidersFailedError,
    OpenAIChatModelResponseError,
    OpenAIChatModelConfigError,
    OpenAIChatModelImportError,
    CircuitBreakerOpenException,
)

llm = OpenAIChatModel(model="gpt-4o")

try:
    reply = llm.invoke("Hello!")
except OpenAIChatModelAllProvidersFailedError as e:
    # Primary AND every configured fallback provider failed
    print(f"All providers failed: {e.attempts}")
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
    # Too many recent failures, circuit is open
    print(f"Circuit open: {e}")
```

Retry behaviour: by default the wrapper retries up to 3 times with exponential back-off.

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
| `model` | `str` | required | Model name. e.g. `"gpt-4o"`, `"llama3-70b-8192"`, `"gemini-2.0-flash"`, `"mistral-large-latest"` |
| `api_key` | `str` | `OPENAI_API_KEY` env | API key for the provider you are using |
| `base_url` | `str` | `OPENAI_BASE_URL` env | Provider endpoint. e.g. `"https://api.groq.com/openai/v1"` or `"http://localhost:11434/v1"` |
| `organization` | `str` | `None` | OpenAI organization ID (OpenAI only) |
| `project` | `str` | `None` | OpenAI project ID (OpenAI only) |
| `system_prompt` | `str` | `None` | System prompt prepended to every request |
| `prompt_template` | `str` | `None` | Template with `{variable}` placeholders |
| `temperature` | `float` | `None` | Sampling temperature 0 to 2. Higher = more random |
| `top_p` | `float` | `None` | Nucleus sampling 0 to 1 |
| `max_tokens` | `int` | `None` | Maximum tokens to generate |
| `output_schema` | `BaseModel` / `dict` | `None` | Pydantic model or JSON schema for structured output |
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
| `fallback_providers` | `list[dict]` | `None` | Ordered backup providers, each `{"model", "api_key"?, "base_url"?, "organization"?, "project"?}`, tried after the primary exhausts its retries |

---

## API Reference

### What Each Method Returns

| Method | Returns |
|---|---|
| `invoke(prompt)` | `str`, generated text (or `dict` if `structured_output=True`) |
| `ainvoke(prompt)` | same as `invoke`, async |
| `stream(prompt)` | `Iterator[str]`, text chunks |
| `astream(prompt)` | `AsyncIterator[str]`, text chunks |
| `batch_invoke(prompts)` | `list[str]`, one result per prompt |
| `abatch_invoke(prompts)` | `list[str]`, concurrent results |
| `invoke_with_tools(prompt, tools)` | `ToolCallResponse`, `.tool_calls` list or `.text` |
| `ainvoke_with_tools(prompt, tools)` | same as `invoke_with_tools`, async |
| `create(messages)` | Raw OpenAI `ChatCompletion` response object |
| `acreate(messages)` | same as `create`, async |

### `ToolCallResponse` fields

| Field | Type | Description |
|---|---|---|
| `.tool_calls` | `list[FunctionCall]` | Tool calls the model wants to make (empty if final answer) |
| `.text` | `str \| None` | Final text answer (None if tool calls present) |
| `.raw` | `Any` | Raw provider response object |
| `.has_tool_calls` | `bool` | `True` when `tool_calls` is non-empty |
| `.is_final_answer` | `bool` | `True` when `text` is present and `tool_calls` is empty |

### `FunctionCall` fields

| Field | Type | Description |
|---|---|---|
| `.name` | `str` | Tool function name |
| `.arguments` | `dict` | Parsed JSON arguments |
| `.call_id` | `str \| None` | Call ID for multi-turn tracking |

### Metadata dict (when `structured_output=True`, or via `llm.last_metadata`)

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
| `"provider_used"` | `str` | `"primary"` or `"fallback[N]:<model>"` — which provider actually served the request |

---

## License

Apache License 2.0, Copyright (c) 2026 Jitin Kumar Sengar
