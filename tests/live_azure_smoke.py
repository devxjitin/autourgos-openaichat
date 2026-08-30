"""
Live, real-network feature verification against Azure OpenAI.

Not part of the pytest suite (no network in CI). Run manually:

    .venv/Scripts/python tests/live_azure_smoke.py

Reads AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY / AZURE_OPENAI_DEPLOYMENT
from the environment and exercises every documented feature against the real
deployment. Each feature is isolated in its own try/except so one failure
doesn't stop the rest from running; results are summarized at the end.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import traceback

from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autourgos_openaichat import (
    OpenAIChatModel,
    OpenAIChatModelAPIError,
    OpenAIChatModelConfigError,
)

ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))


def check(name: str, fn):
    try:
        detail = fn()
        record(name, True, detail if isinstance(detail, str) else "")
    except Exception as exc:
        record(name, False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc(limit=3)


def make_llm(**kwargs) -> OpenAIChatModel:
    return OpenAIChatModel(
        model=DEPLOYMENT,
        api_key=API_KEY,
        base_url=ENDPOINT,
        timeout=60.0,
        **kwargs,
    )


# ── 1. Basic invoke ──────────────────────────────────────────────────────────

def t_basic_invoke():
    llm = make_llm()
    reply = llm.invoke("Reply with exactly one word: the capital of France.")
    assert isinstance(reply, str) and reply.strip(), "empty reply"
    return reply[:80]


# ── 2. Async invoke ───────────────────────────────────────────────────────────

def t_async_invoke():
    llm = make_llm()

    async def run():
        return await llm.ainvoke("Reply with exactly one word: the speed-of-light unit (m/s).")

    reply = asyncio.run(run())
    assert isinstance(reply, str) and reply.strip()
    return reply[:80]


# ── 3. Streaming ──────────────────────────────────────────────────────────────

def t_streaming():
    llm = make_llm()
    chunks = list(llm.stream("Count from 1 to 5, comma separated, nothing else."))
    joined = "".join(chunks)
    assert len(chunks) >= 1 and joined.strip()
    return f"{len(chunks)} chunks: {joined[:60]!r}"


# ── 4. Async streaming ─────────────────────────────────────────────────────────

def t_async_streaming():
    llm = make_llm()

    async def run():
        out = []
        async for c in llm.astream("Say the word 'streaming' three times, space separated."):
            out.append(c)
        return out

    chunks = asyncio.run(run())
    joined = "".join(chunks)
    assert joined.strip()
    return f"{len(chunks)} chunks: {joined[:60]!r}"


# ── 5. System prompt ──────────────────────────────────────────────────────────

def t_system_prompt():
    llm = make_llm(system_prompt="You always answer in exactly one word, no punctuation.")
    reply = llm.invoke("What color is the sky on a clear day?")
    assert reply.strip()
    return reply.strip()


# ── 6. Prompt template ─────────────────────────────────────────────────────────

def t_prompt_template():
    llm = make_llm(prompt_template="Translate to {language}: {text}")
    reply = llm.invoke(prompt_variables={"language": "French", "text": "Good morning"})
    assert reply.strip()
    return reply.strip()[:60]


# ── 7. Multi-turn conversation ─────────────────────────────────────────────────

def t_multiturn():
    llm = make_llm()
    messages = [
        {"role": "user", "content": "My favorite number is 42."},
        {"role": "assistant", "content": "Got it, 42 is your favorite number."},
        {"role": "user", "content": "What is my favorite number? Reply with digits only."},
    ]
    reply = llm.invoke(messages)
    assert "42" in reply
    return reply.strip()


# ── 8. JSON mode ────────────────────────────────────────────────────────────────

def t_json_mode():
    llm = make_llm(
        response_mime_type="application/json",
        system_prompt="Always respond with valid JSON only, no markdown fences.",
    )
    reply = llm.invoke("Give me a JSON object with keys name and age for a fictional person.")
    parsed = json.loads(reply)
    assert "name" in parsed and "age" in parsed
    return json.dumps(parsed)


# ── 9. Structured output (Pydantic schema) ──────────────────────────────────────

class CityInfo(BaseModel):
    city: str = Field(description="Name of the city")
    country: str = Field(description="Name of the country")


def t_structured_output():
    llm = make_llm(output_schema=CityInfo, structured_output=True)
    result = llm.invoke("Tell me about Tokyo.")
    assert isinstance(result, dict) and "response" in result
    data = json.loads(result["response"])
    assert data["city"].lower() == "tokyo"
    return json.dumps(data)


# ── 10. Native tool calling ─────────────────────────────────────────────────────

TOOLS = [{
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
    },
}]


def t_tool_calling():
    llm = make_llm()
    resp = llm.invoke_with_tools("What's the weather in Tokyo right now? Use the tool.", TOOLS)
    assert resp.has_tool_calls, f"model did not call the tool; text={resp.text!r}"
    call = resp.tool_calls[0]
    assert call.name == "get_weather"
    assert "city" in call.arguments
    return f"{call.name}({call.arguments})"


def t_async_tool_calling():
    llm = make_llm()

    async def run():
        return await llm.ainvoke_with_tools("What's the weather in Paris? Use the tool.", TOOLS)

    resp = asyncio.run(run())
    assert resp.has_tool_calls
    return f"{resp.tool_calls[0].name}({resp.tool_calls[0].arguments})"


def t_agentic_tool_loop():
    """Full round trip: tool call -> execute -> feed result back -> final answer."""
    llm = make_llm()

    def get_weather(city: str) -> str:
        return json.dumps({"city": city, "temp_c": 22, "condition": "Sunny"})

    messages = [{"role": "user", "content": "What is the weather in Paris? Use the tool, then answer in one sentence."}]
    resp = llm.invoke_with_tools(messages, TOOLS)
    assert resp.has_tool_calls, f"expected tool call, got text={resp.text!r}"

    messages.append({
        "role": "assistant",
        "tool_calls": [
            {"id": tc.call_id, "type": "function",
             "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
            for tc in resp.tool_calls
        ],
    })
    for tc in resp.tool_calls:
        result = get_weather(**tc.arguments)
        messages.append({"role": "tool", "tool_call_id": tc.call_id, "content": result})

    final = llm.invoke_with_tools(messages, TOOLS)
    assert final.is_final_answer, f"expected final answer, got another tool call: {final.tool_calls}"
    assert "22" in final.text or "sunny" in final.text.lower()
    return final.text.strip()


# ── 11. Vision — local file path ────────────────────────────────────────────────

def _make_local_test_image(path: str) -> None:
    # 1x1 red pixel PNG
    png_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    with open(path, "wb") as f:
        f.write(base64.b64decode(png_b64))


def t_vision_local_file():
    path = os.path.join(os.path.dirname(__file__), "_live_test_pixel.png")
    _make_local_test_image(path)
    try:
        llm = make_llm()
        reply = llm.invoke("What color is this image? Answer in one word.", files=[path])
        assert reply.strip()
        return reply.strip()
    finally:
        if os.path.exists(path):
            os.remove(path)


# ── 12. Vision — online URL ──────────────────────────────────────────────────────

def t_vision_online_url():
    llm = make_llm()
    url = "https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_92x30dp.png"
    reply = llm.invoke("What company logo is shown in this image? One word.", files=[url])
    assert reply.strip()
    return reply.strip()


# ── 13. Vision — raw bytes ────────────────────────────────────────────────────────

def t_vision_raw_bytes():
    png_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    raw = base64.b64decode(png_b64)
    llm = make_llm()
    reply = llm.invoke("What color is this image? One word.", files=[raw])
    assert reply.strip()
    return reply.strip()


# ── 14. Cost tracking ────────────────────────────────────────────────────────────

def t_cost_tracking():
    llm = make_llm(input_pricing=2.50, output_pricing=10.00, structured_output=True)
    result = llm.invoke("Say hello in one short sentence.")
    assert result["input_tokens"] is not None
    assert result["total_cost"] >= 0
    return f"in={result['input_tokens']} out={result['output_tokens']} cost=${result['total_cost']:.6f} lat={result['latency_ms']}ms"


# ── 15. Batch invoke ──────────────────────────────────────────────────────────────

def t_batch_invoke():
    llm = make_llm()
    results = llm.batch_invoke(["Capital of Japan? One word.", "Capital of Germany? One word."])
    assert len(results) == 2 and all(r.strip() for r in results)
    return " | ".join(results)


def t_abatch_invoke():
    llm = make_llm()

    async def run():
        return await llm.abatch_invoke(["2+2? Digit only.", "3+3? Digit only."])

    results = asyncio.run(run())
    assert len(results) == 2 and all(r.strip() for r in results)
    return " | ".join(results)


# ── 16. Context manager ────────────────────────────────────────────────────────────

def t_context_manager():
    with make_llm() as llm:
        reply = llm.invoke("Reply with 'pong'.")
        assert reply.strip()
    assert llm._client is None
    return reply.strip()


# ── 17. Error handling — real invalid deployment name ───────────────────────────

def t_error_handling_bad_deployment():
    llm = OpenAIChatModel(
        model="this-deployment-does-not-exist-xyz",
        api_key=API_KEY,
        base_url=ENDPOINT,
        timeout=60.0,
        max_retries=1,
    )
    try:
        llm.invoke("hi")
        return "UNEXPECTED: no error raised for nonexistent deployment"
    except OpenAIChatModelAPIError as e:
        return f"correctly raised OpenAIChatModelAPIError: {str(e)[:100]}"


# ── 18. Config error (local, no network) ─────────────────────────────────────────

def t_config_error():
    try:
        make_llm(structured_output=True, streaming=True)
        return "UNEXPECTED: no error raised"
    except OpenAIChatModelConfigError:
        return "correctly raised OpenAIChatModelConfigError"


# ── 19. Low-level create() ───────────────────────────────────────────────────────

def t_low_level_create():
    llm = make_llm()
    raw = llm.create([{"role": "user", "content": "Say hi."}])
    assert hasattr(raw, "choices")
    return f"id={getattr(raw, 'id', '?')}"


def main() -> int:
    print(f"Azure endpoint: {ENDPOINT}")
    print(f"Deployment:     {DEPLOYMENT}")
    print("=" * 70)

    checks = [
        ("basic_invoke", t_basic_invoke),
        ("async_invoke", t_async_invoke),
        ("streaming", t_streaming),
        ("async_streaming", t_async_streaming),
        ("system_prompt", t_system_prompt),
        ("prompt_template", t_prompt_template),
        ("multiturn_conversation", t_multiturn),
        ("json_mode", t_json_mode),
        ("structured_output_pydantic", t_structured_output),
        ("native_tool_calling", t_tool_calling),
        ("async_tool_calling", t_async_tool_calling),
        ("agentic_tool_loop_roundtrip", t_agentic_tool_loop),
        ("vision_local_file_path", t_vision_local_file),
        ("vision_online_url", t_vision_online_url),
        ("vision_raw_bytes", t_vision_raw_bytes),
        ("cost_tracking", t_cost_tracking),
        ("batch_invoke", t_batch_invoke),
        ("abatch_invoke", t_abatch_invoke),
        ("context_manager", t_context_manager),
        ("error_handling_bad_deployment", t_error_handling_bad_deployment),
        ("config_error_local", t_config_error),
        ("low_level_create", t_low_level_create),
    ]

    for name, fn in checks:
        check(name, fn)

    print("=" * 70)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"{passed}/{len(RESULTS)} passed")
    if failed:
        print("FAILED:", ", ".join(failed))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
