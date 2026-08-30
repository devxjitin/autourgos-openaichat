"""
Feature-by-feature test suite for OpenAIChatModel.

Every test mocks the underlying openai.OpenAI / openai.AsyncOpenAI client so
no network calls are made, but exercises the real autourgos_openaichat code
paths (message building, retries, parsing, circuit breaker, etc.) end to end.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from pydantic import BaseModel, Field, field_validator

from autourgos_openaichat import (
    OpenAIChatModel,
    OpenAIChatModelAPIError,
    OpenAIChatModelConfigError,
    OpenAIChatModelImportError,
    OpenAIChatModelResponseError,
    CircuitBreakerOpenException,
)


# ── Helpers to build fake OpenAI SDK response objects ──────────────────────

def make_completion(text="Paris", tool_calls=None, usage=(9, 10, 19)):
    msg = SimpleNamespace(content=text, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg)
    usage_obj = SimpleNamespace(
        prompt_tokens=usage[0], completion_tokens=usage[1], total_tokens=usage[2]
    )
    return SimpleNamespace(choices=[choice], usage=usage_obj)


def make_tool_call(name, arguments, call_id="call_1"):
    fn = SimpleNamespace(name=name, arguments=json.dumps(arguments))
    return SimpleNamespace(id=call_id, function=fn)


def make_stream_chunks(words):
    chunks = []
    for w in words:
        delta = SimpleNamespace(content=w)
        choice = SimpleNamespace(delta=delta)
        chunks.append(SimpleNamespace(choices=[choice]))
    return chunks


class FakeSyncStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __iter__(self):
        return iter(self._chunks)


class FakeAsyncStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        self._iter = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


def make_llm(**kwargs):
    """Construct an OpenAIChatModel with mocked sync/async clients."""
    llm = OpenAIChatModel(model="gpt-4o", api_key="sk-test", **kwargs)
    llm._client = MagicMock()
    llm._async_client = MagicMock()
    llm._async_client.chat = MagicMock()
    llm._async_client.chat.completions = MagicMock()
    llm._async_client.chat.completions.create = AsyncMock()
    return llm


# ── 1. Basic text generation (invoke) ───────────────────────────────────────

def test_invoke_basic_text():
    llm = make_llm()
    llm._client.chat.completions.create.return_value = make_completion("Paris")
    assert llm.invoke("What is the capital of France?") == "Paris"
    kwargs = llm._client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-4o"
    assert kwargs["messages"] == [{"role": "user", "content": "What is the capital of France?"}]
    assert kwargs["stream"] is False


def test_invoke_passes_sampling_params():
    llm = make_llm(temperature=0.7, top_p=0.9, max_tokens=256)
    llm._client.chat.completions.create.return_value = make_completion("ok")
    llm.invoke("hi")
    kwargs = llm._client.chat.completions.create.call_args.kwargs
    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 0.9
    assert kwargs["max_tokens"] == 256


def test_invoke_accepts_per_call_overrides():
    # Mirrors autourgos-agent's AgentLoopMixin, which calls
    # self.llm.invoke(messages, **call_kwargs) with per-iteration overrides
    # (e.g. from an on_before_iteration middleware hook).
    llm = make_llm(temperature=0.7)
    llm._client.chat.completions.create.return_value = make_completion("ok")
    llm.invoke("hi", temperature=0.1, stop=["Observation:"])
    kwargs = llm._client.chat.completions.create.call_args.kwargs
    assert kwargs["temperature"] == 0.1  # per-call override wins over constructor default
    assert kwargs["stop"] == ["Observation:"]


def test_invoke_overrides_cannot_hijack_model_or_messages():
    llm = make_llm()
    llm._client.chat.completions.create.return_value = make_completion("ok")
    llm.invoke("hi", model="not-a-real-model", messages=["hijacked"])
    kwargs = llm._client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-4o"
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]


def test_ainvoke_accepts_per_call_overrides():
    llm = make_llm()
    llm._async_client.chat.completions.create.return_value = make_completion("ok")

    async def run():
        return await llm.ainvoke("hi", temperature=0.2, max_tokens=64)

    asyncio.run(run())
    kwargs = llm._async_client.chat.completions.create.call_args.kwargs
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_tokens"] == 64


def test_invoke_streaming_mode_accepts_per_call_overrides():
    llm = make_llm(streaming=True)
    llm._client.chat.completions.create.return_value = FakeSyncStream(
        make_stream_chunks(["ok"])
    )
    llm.invoke("hi", temperature=0.3)
    kwargs = llm._client.chat.completions.create.call_args.kwargs
    assert kwargs["temperature"] == 0.3


# ── 2. Async generation (ainvoke) ───────────────────────────────────────────

def test_ainvoke_basic_text():
    llm = make_llm()
    llm._async_client.chat.completions.create.return_value = make_completion("Light travels fast")

    async def run():
        return await llm.ainvoke("What is the speed of light?")

    assert asyncio.run(run()) == "Light travels fast"


# ── 3. Streaming (sync) ─────────────────────────────────────────────────────

def test_stream_sync_joins_chunks():
    llm = make_llm()
    llm._client.chat.completions.create.return_value = FakeSyncStream(
        make_stream_chunks(["Rain", "drops ", "fall."])
    )
    chunks = list(llm.stream("Write a haiku about rain."))
    assert chunks == ["Rain", "drops ", "fall."]


def test_invoke_with_streaming_flag_joins_internally():
    llm = make_llm(streaming=True)
    llm._client.chat.completions.create.return_value = FakeSyncStream(
        make_stream_chunks(["Hello", " world"])
    )
    assert llm.invoke("hi") == "Hello world"


def test_stream_raises_on_empty_stream():
    llm = make_llm()
    llm._client.chat.completions.create.return_value = FakeSyncStream([])
    with pytest.raises(OpenAIChatModelResponseError):
        list(llm.stream("hi"))


def test_stream_accepts_per_call_overrides():
    llm = make_llm(temperature=0.7)
    llm._client.chat.completions.create.return_value = FakeSyncStream(
        make_stream_chunks(["ok"])
    )
    list(llm.stream("hi", temperature=0.1, stop=["Observation:"]))
    kwargs = llm._client.chat.completions.create.call_args.kwargs
    assert kwargs["temperature"] == 0.1
    assert kwargs["stop"] == ["Observation:"]


# ── 4. Async streaming ───────────────────────────────────────────────────────

def test_astream_yields_chunks():
    llm = make_llm()
    llm._async_client.chat.completions.create.return_value = FakeAsyncStream(
        make_stream_chunks(["1... ", "2... ", "3..."])
    )

    async def run():
        return [c async for c in llm.astream("count")]

    assert asyncio.run(run()) == ["1... ", "2... ", "3..."]


def test_astream_accepts_per_call_overrides():
    llm = make_llm()
    llm._async_client.chat.completions.create.return_value = FakeAsyncStream(
        make_stream_chunks(["ok"])
    )

    async def run():
        return [c async for c in llm.astream("hi", temperature=0.2, max_tokens=64)]

    asyncio.run(run())
    kwargs = llm._async_client.chat.completions.create.call_args.kwargs
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_tokens"] == 64


# ── 5. Batch invocation ──────────────────────────────────────────────────────

def test_batch_invoke_sequential():
    llm = make_llm()
    llm._client.chat.completions.create.side_effect = [
        make_completion("Tokyo"), make_completion("Berlin"), make_completion("Brasilia"),
    ]
    results = llm.batch_invoke(["Capital of Japan?", "Capital of Germany?", "Capital of Brazil?"])
    assert results == ["Tokyo", "Berlin", "Brasilia"]


def test_abatch_invoke_concurrent():
    llm = make_llm()
    llm._async_client.chat.completions.create.side_effect = [
        make_completion("Tokyo"), make_completion("Berlin"), make_completion("Brasilia"),
    ]

    async def run():
        return await llm.abatch_invoke(["a", "b", "c"])

    assert asyncio.run(run()) == ["Tokyo", "Berlin", "Brasilia"]


# ── 6. System prompt ────────────────────────────────────────────────────────

def test_system_prompt_prepended():
    llm = make_llm(system_prompt="You are a pirate.")
    llm._client.chat.completions.create.return_value = make_completion("Arrr")
    llm.invoke("What time is it?")
    messages = llm._client.chat.completions.create.call_args.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "You are a pirate."}
    assert messages[1]["role"] == "user"


def test_unknown_kwarg_raises_type_error():
    with pytest.raises(TypeError):
        make_llm(totally_made_up_kwarg="x")


# ── 7. Prompt templates ──────────────────────────────────────────────────────

def test_prompt_template_renders_variables():
    llm = make_llm(prompt_template="Translate the following text to {language}:\n\n{text}")
    llm._client.chat.completions.create.return_value = make_completion("Bonjour !")
    llm.invoke(prompt_variables={"language": "French", "text": "Good morning!"})
    messages = llm._client.chat.completions.create.call_args.kwargs["messages"]
    assert messages[0]["content"] == "Translate the following text to French:\n\nGood morning!"


def test_prompt_template_missing_variable_raises():
    llm = make_llm(prompt_template="Translate to {language}: {text}")
    with pytest.raises(ValueError, match="Missing prompt template variables: text"):
        llm.invoke(prompt_variables={"language": "French"})


def test_prompt_and_no_template_configured_raises():
    llm = make_llm()
    with pytest.raises(ValueError, match="prompt is required"):
        llm.invoke()


# ── 8. Multi-modal vision input ──────────────────────────────────────────────

def test_multimodal_from_bytes():
    llm = make_llm()
    llm._client.chat.completions.create.return_value = make_completion("A cat")
    llm.invoke("What is this?", files=[b"\x89PNG..."])
    messages = llm._client.chat.completions.create.call_args.kwargs["messages"]
    content = messages[0]["content"]
    assert content[0] == {"type": "text", "text": "What is this?"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_multimodal_from_url_string_treated_as_url_when_not_a_file():
    llm = make_llm()
    llm._client.chat.completions.create.return_value = make_completion("A chart")
    llm.invoke("Describe this chart.", files=["https://example.com/chart.png"])
    messages = llm._client.chat.completions.create.call_args.kwargs["messages"]
    content = messages[0]["content"]
    assert content[1]["image_url"]["url"] == "https://example.com/chart.png"


def test_multimodal_image_detail_applied():
    llm = make_llm()
    llm._client.chat.completions.create.return_value = make_completion("text")
    llm.invoke("Read this", files=[b"data"], image_detail="high")
    messages = llm._client.chat.completions.create.call_args.kwargs["messages"]
    assert messages[0]["content"][1]["image_url"]["detail"] == "high"


def test_multimodal_from_dict_url():
    llm = make_llm()
    llm._client.chat.completions.create.return_value = make_completion("ok")
    llm.invoke("desc", files=[{"url": "https://x.test/a.png"}])
    messages = llm._client.chat.completions.create.call_args.kwargs["messages"]
    assert messages[0]["content"][1]["image_url"]["url"] == "https://x.test/a.png"


# ── 9. Structured output ─────────────────────────────────────────────────────

class CityCountryInfo(BaseModel):
    city: str = Field(description="Name of the city")
    country: str = Field(description="Name of the country")


def test_structured_output_with_pydantic_schema():
    llm = make_llm(output_schema=CityCountryInfo, structured_output=True)
    payload = json.dumps({"city": "Tokyo", "country": "Japan"})
    llm._client.chat.completions.create.return_value = make_completion(payload)
    result = llm.invoke("Tell me about Tokyo.")
    assert isinstance(result, dict)
    assert json.loads(result["response"]) == {"city": "Tokyo", "country": "Japan"}
    kwargs = llm._client.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"]["type"] == "json_schema"
    assert kwargs["response_format"]["json_schema"]["name"] == "CityCountryInfo"
    schema_properties = kwargs["response_format"]["json_schema"]["schema"]["properties"]
    assert set(schema_properties.keys()) == {"city", "country"}


def test_structured_streaming_incompatible_raises_at_construction():
    with pytest.raises(OpenAIChatModelConfigError):
        make_llm(structured_output=True, streaming=True)


# ── 10. JSON mode ─────────────────────────────────────────────────────────────

def test_json_mode_sets_response_format():
    llm = make_llm(response_mime_type="application/json")
    llm._client.chat.completions.create.return_value = make_completion('{"name": "Alice"}')
    llm.invoke("Give me a person.")
    kwargs = llm._client.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}


# ── 11. Native tool calling ──────────────────────────────────────────────────

TOOLS = [{
    "name": "get_weather",
    "description": "Get the weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}]


def test_invoke_with_tools_returns_tool_calls():
    llm = make_llm()
    raw = make_completion(text=None, tool_calls=[make_tool_call("get_weather", {"city": "Tokyo"})])
    llm._client.chat.completions.create.return_value = raw
    resp = llm.invoke_with_tools("Weather in Tokyo?", TOOLS)
    assert resp.has_tool_calls
    assert not resp.is_final_answer
    assert resp.tool_calls[0].name == "get_weather"
    assert resp.tool_calls[0].arguments == {"city": "Tokyo"}
    assert resp.tool_calls[0].call_id == "call_1"
    kwargs = llm._client.chat.completions.create.call_args.kwargs
    assert kwargs["tools"][0]["type"] == "function"
    assert kwargs["tool_choice"] == "auto"


def test_invoke_with_tools_accepts_per_call_overrides():
    llm = make_llm(temperature=0.7)
    llm._client.chat.completions.create.return_value = make_completion("ok")
    llm.invoke_with_tools("hi", TOOLS, temperature=0.1, stop=["Observation:"])
    kwargs = llm._client.chat.completions.create.call_args.kwargs
    assert kwargs["temperature"] == 0.1  # per-call override wins over constructor default
    assert kwargs["stop"] == ["Observation:"]


def test_invoke_with_tools_returns_final_answer_when_no_tool_calls():
    llm = make_llm()
    llm._client.chat.completions.create.return_value = make_completion("It's sunny.")
    resp = llm.invoke_with_tools("Weather?", TOOLS)
    assert not resp.has_tool_calls
    assert resp.is_final_answer
    assert resp.text == "It's sunny."


def test_invoke_with_tools_malformed_arguments_json_falls_back_to_empty_dict():
    llm = make_llm()
    fn = SimpleNamespace(name="get_weather", arguments="{not valid json")
    tc = SimpleNamespace(id="call_2", function=fn)
    llm._client.chat.completions.create.return_value = make_completion(text=None, tool_calls=[tc])
    resp = llm.invoke_with_tools("Weather?", TOOLS)
    assert resp.tool_calls[0].arguments == {}


def test_ainvoke_with_tools():
    llm = make_llm()
    raw = make_completion(text=None, tool_calls=[make_tool_call("get_weather", {"city": "London"})])
    llm._async_client.chat.completions.create.return_value = raw

    async def run():
        return await llm.ainvoke_with_tools("Weather in London?", TOOLS)

    resp = asyncio.run(run())
    assert resp.tool_calls[0].arguments == {"city": "London"}


def test_ainvoke_with_tools_accepts_per_call_overrides():
    llm = make_llm()
    llm._async_client.chat.completions.create.return_value = make_completion("ok")

    async def run():
        return await llm.ainvoke_with_tools("hi", TOOLS, temperature=0.2, max_tokens=64)

    asyncio.run(run())
    kwargs = llm._async_client.chat.completions.create.call_args.kwargs
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_tokens"] == 64


# ── 12. Multi-turn conversations ─────────────────────────────────────────────

def test_multiturn_list_prompt_passed_through():
    llm = make_llm()
    llm._client.chat.completions.create.return_value = make_completion("Your name is Jitin.")
    messages = [
        {"role": "user", "content": "My name is Jitin."},
        {"role": "assistant", "content": "Nice to meet you, Jitin!"},
        {"role": "user", "content": "What is my name?"},
    ]
    reply = llm.invoke(messages)
    assert reply == "Your name is Jitin."
    sent = llm._client.chat.completions.create.call_args.kwargs["messages"]
    assert sent == messages


# ── 13. Cost tracking ────────────────────────────────────────────────────────

def test_cost_tracking_computes_costs():
    llm = make_llm(input_pricing=2.50, output_pricing=10.00, structured_output=True)
    llm._client.chat.completions.create.return_value = make_completion("hi", usage=(1000, 1000, 2000))
    result = llm.invoke("Summarise.")
    assert result["input_tokens"] == 1000
    assert result["output_tokens"] == 1000
    assert result["input_cost"] == pytest.approx(0.0025)
    assert result["output_cost"] == pytest.approx(0.01)
    assert result["total_cost"] == pytest.approx(0.0125)
    assert "latency_ms" in result


def test_last_metadata_populated_without_structured_output():
    llm = make_llm(input_pricing=2.50, output_pricing=10.00)
    llm._client.chat.completions.create.return_value = make_completion("Hello!", usage=(9, 10, 19))
    reply = llm.invoke("Hello!")
    assert reply == "Hello!"
    assert llm.last_metadata["input_tokens"] == 9
    assert llm.last_metadata["total_cost"] > 0


# ── 14. Context manager ──────────────────────────────────────────────────────

def test_sync_context_manager_closes_client():
    llm = make_llm()
    llm._client.chat.completions.create.return_value = make_completion("Pong!")
    close_mock = llm._client.close
    with llm as ctx:
        assert ctx.invoke("Ping!") == "Pong!"
    close_mock.assert_called_once()
    assert llm._client is None


def test_async_context_manager_closes_both_clients():
    llm = make_llm()
    llm._async_client.aclose = AsyncMock()
    aclose_mock = llm._async_client.aclose
    llm._async_client.chat.completions.create.return_value = make_completion("hi")

    async def run():
        async with llm as ctx:
            return await ctx.ainvoke("hi")

    result = asyncio.run(run())
    assert result == "hi"
    aclose_mock.assert_called_once()
    assert llm._client is None
    assert llm._async_client is None


# ── 15. Circuit breaker ──────────────────────────────────────────────────────

def test_circuit_breaker_opens_after_threshold_and_blocks():
    llm = make_llm(circuit_failure_threshold=2, circuit_cooldown_time=60.0, max_retries=1)
    llm._client.chat.completions.create.side_effect = ConnectionError("boom")

    for _ in range(2):
        with pytest.raises(OpenAIChatModelAPIError):
            llm.invoke("hi")

    with pytest.raises(CircuitBreakerOpenException):
        llm.invoke("hi")


def test_circuit_breaker_resets_on_success():
    llm = make_llm(circuit_failure_threshold=3, max_retries=1)
    llm._client.chat.completions.create.side_effect = ConnectionError("boom")
    with pytest.raises(OpenAIChatModelAPIError):
        llm.invoke("hi")

    llm._client.chat.completions.create.side_effect = None
    llm._client.chat.completions.create.return_value = make_completion("ok")
    assert llm.invoke("hi") == "ok"
    assert llm._consecutive_failures == 0


def test_circuit_breaker_ignores_value_errors_as_non_transient():
    """A ValueError (e.g. bad local input) should not count toward the circuit breaker."""
    llm = make_llm(circuit_failure_threshold=1)
    with pytest.raises(ValueError):
        llm.invoke()  # no prompt, no template configured -> ValueError, raised before any API call
    assert llm._consecutive_failures == 0


# ── 16. Error handling ───────────────────────────────────────────────────────

def test_config_error_streaming_and_structured_output():
    with pytest.raises(OpenAIChatModelConfigError, match="incompatible with streaming"):
        make_llm(structured_output=True, streaming=True)


def test_import_error_when_openai_sdk_missing():
    # _OPENAI_AVAILABLE is resolved once at module import time (chat.py:40), not
    # per-instantiation, so the SDK-missing path is exercised by patching that
    # cached module-level state directly rather than the loader function.
    import autourgos_openaichat.chat as chat_mod
    with patch.object(chat_mod, "_OPENAI_AVAILABLE", False), \
         patch.object(chat_mod, "_OPENAI_IMPORT_ERROR", "no module named openai"):
        with pytest.raises(OpenAIChatModelImportError):
            OpenAIChatModel(model="gpt-4o", api_key="sk-test")


def test_response_error_when_no_text_extracted():
    llm = make_llm()
    llm._client.chat.completions.create.return_value = make_completion(text=None)
    with pytest.raises(OpenAIChatModelResponseError):
        llm.invoke("hi")


def test_api_error_after_retries_exhausted():
    llm = make_llm(max_retries=2, backoff_factor=0.001)
    llm._client.chat.completions.create.side_effect = RuntimeError("network down")
    with pytest.raises(OpenAIChatModelAPIError, match="failed after 2 attempts"):
        llm.invoke("hi")
    assert llm._client.chat.completions.create.call_count == 2


def test_retries_then_succeeds():
    llm = make_llm(max_retries=3, backoff_factor=0.001)
    llm._client.chat.completions.create.side_effect = [
        RuntimeError("flaky"), RuntimeError("flaky"), make_completion("recovered"),
    ]
    assert llm.invoke("hi") == "recovered"
    assert llm._client.chat.completions.create.call_count == 3


# ── 17. Low-level create()/acreate() ─────────────────────────────────────────

def test_create_low_level_returns_raw_response():
    llm = make_llm()
    raw = make_completion("Hi there")
    llm._client.chat.completions.create.return_value = raw
    result = llm.create([{"role": "user", "content": "Hi"}])
    assert result is raw


def test_create_requires_messages():
    llm = make_llm()
    with pytest.raises(ValueError, match="input_data"):
        llm.create()


def test_acreate_low_level():
    llm = make_llm()
    raw = make_completion("Hi there")
    llm._async_client.chat.completions.create.return_value = raw

    async def run():
        return await llm.acreate([{"role": "user", "content": "Hi"}])

    assert asyncio.run(run()) is raw


# ── 18. Repr / misc ──────────────────────────────────────────────────────────

def test_repr_contains_model_and_flags():
    llm = make_llm(streaming=False)
    r = repr(llm)
    assert "gpt-4o" in r
    assert "streaming=False" in r


def test_normalize_model_name_lowercases():
    llm = OpenAIChatModel(model="  GPT-4O  ", api_key="sk-test")
    assert llm._model_name == "gpt-4o"


# ── 19. Additional async / edge-case coverage ────────────────────────────────

def test_ainvoke_with_streaming_flag_joins_internally():
    llm = make_llm(streaming=True)
    llm._async_client.chat.completions.create.return_value = FakeAsyncStream(
        make_stream_chunks(["Hello", " async"])
    )

    async def run():
        return await llm.ainvoke("hi")

    assert asyncio.run(run()) == "Hello async"


def test_astream_raises_on_empty_stream():
    llm = make_llm()
    llm._async_client.chat.completions.create.return_value = FakeAsyncStream([])

    async def run():
        return [c async for c in llm.astream("hi")]

    with pytest.raises(OpenAIChatModelResponseError):
        asyncio.run(run())


def test_astream_retries_then_succeeds():
    llm = make_llm(max_retries=2, backoff_factor=0.001)
    llm._async_client.chat.completions.create.side_effect = [
        RuntimeError("flaky"), FakeAsyncStream(make_stream_chunks(["ok"])),
    ]

    async def run():
        return [c async for c in llm.astream("hi")]

    assert asyncio.run(run()) == ["ok"]


def test_async_api_error_after_retries_exhausted():
    llm = make_llm(max_retries=2, backoff_factor=0.001)
    llm._async_client.chat.completions.create.side_effect = RuntimeError("network down")

    async def run():
        return await llm.ainvoke("hi")

    with pytest.raises(OpenAIChatModelAPIError, match="Async Chat Completions request failed"):
        asyncio.run(run())


def test_acreate_defaults_messages_from_kwarg():
    llm = make_llm()
    raw = make_completion("hi")
    llm._async_client.chat.completions.create.return_value = raw

    async def run():
        return await llm.acreate(messages=[{"role": "user", "content": "hi"}])

    assert asyncio.run(run()) is raw


def test_tools_already_in_openai_format_passed_through_unchanged():
    llm = make_llm()
    llm._client.chat.completions.create.return_value = make_completion("sunny")
    preformatted = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    llm.invoke_with_tools("weather?", preformatted)
    sent_tools = llm._client.chat.completions.create.call_args.kwargs["tools"]
    assert sent_tools == preformatted


def test_parse_tool_calls_returns_empty_when_no_choices():
    calls = OpenAIChatModel._parse_tool_calls(SimpleNamespace(choices=None))
    assert calls == []


def test_multimodal_from_real_file_path(tmp_path):
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0fakejpegdata")
    llm = make_llm()
    llm._client.chat.completions.create.return_value = make_completion("A desk")
    llm.invoke("What is this?", files=[str(img)])
    messages = llm._client.chat.completions.create.call_args.kwargs["messages"]
    url = messages[0]["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/jpg;base64,")


def test_multimodal_from_dict_data_field():
    llm = make_llm()
    llm._client.chat.completions.create.return_value = make_completion("ok")
    llm.invoke("desc", files=[{"data": b"rawbytes", "mime_type": "image/webp"}])
    messages = llm._client.chat.completions.create.call_args.kwargs["messages"]
    assert messages[0]["content"][1]["image_url"]["url"].startswith("data:image/webp;base64,")


def test_multimodal_from_dict_path_field(tmp_path):
    img = tmp_path / "diagram.png"
    img.write_bytes(b"pngdata")
    llm = make_llm()
    llm._client.chat.completions.create.return_value = make_completion("ok")
    llm.invoke("desc", files=[{"path": str(img)}])
    messages = llm._client.chat.completions.create.call_args.kwargs["messages"]
    assert messages[0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_structured_output_with_dict_schema():
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    llm = make_llm(output_schema=schema, structured_output=True)
    llm._client.chat.completions.create.return_value = make_completion('{"x": "y"}')
    llm.invoke("go")
    kwargs = llm._client.chat.completions.create.call_args.kwargs
    sent_schema = kwargs["response_format"]["json_schema"]["schema"]
    assert sent_schema["properties"] == schema["properties"]
    assert sent_schema["additionalProperties"] is False
    assert kwargs["response_format"]["json_schema"]["name"] == "response"
    # the caller's dict must not be mutated in place
    assert "additionalProperties" not in schema


def test_structured_output_strict_schema_sets_additional_properties_false():
    """
    Regression test: OpenAI/Azure strict json_schema mode (build_response_format,
    core.py) rejects any object node missing `additionalProperties: false`.
    Pydantic's model_json_schema() doesn't set this by default, including for
    nested models under `$defs` — verified against a real Azure deployment,
    which returned a 400 before this was fixed.
    """
    class Address(BaseModel):
        city: str
        country: str

    class Person(BaseModel):
        name: str
        address: Address

    llm = make_llm(output_schema=Person, structured_output=True)
    llm._client.chat.completions.create.return_value = make_completion('{"name": "x"}')
    llm.invoke("go")
    schema = llm._client.chat.completions.create.call_args.kwargs["response_format"]["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    nested = schema["$defs"]["Address"]
    assert nested["additionalProperties"] is False


def test_stream_delta_extraction_dict_fallback():
    from autourgos_openaichat.core import extract_text_delta_from_event
    event = {"choices": [{"delta": {"content": "hi"}}]}
    assert extract_text_delta_from_event(event) == "hi"


def test_multimodal_path_read_failure_falls_back_to_url_treatment():
    """A string that isn't a real file path is treated as a direct image URL."""
    llm = make_llm()
    llm._client.chat.completions.create.return_value = make_completion("ok")
    llm.invoke("desc", files=["not-a-real-path-on-disk.jpg"])
    messages = llm._client.chat.completions.create.call_args.kwargs["messages"]
    assert messages[0]["content"][1]["image_url"]["url"] == "not-a-real-path-on-disk.jpg"


# ── 18. Provider fallback chain ─────────────────────────────────────────────

from autourgos_openaichat import OpenAIChatModelAllProvidersFailedError


def make_llm_with_fallback(fallback_models=("backup-model",), **kwargs):
    """Construct an OpenAIChatModel with a mocked primary and mocked fallback clients."""
    llm = OpenAIChatModel(
        model="gpt-4o",
        api_key="sk-test",
        fallback_providers=[{"model": m, "api_key": "sk-backup"} for m in fallback_models],
        **kwargs,
    )
    llm._client = MagicMock()
    llm._async_client = MagicMock()
    llm._async_client.chat.completions.create = AsyncMock()
    for i in range(len(fallback_models)):
        fb_sync = MagicMock()
        fb_async = MagicMock()
        fb_async.chat.completions.create = AsyncMock()
        llm._fallback_sync_clients[i] = fb_sync
        llm._fallback_async_clients[i] = fb_async
    return llm


def test_fallback_config_missing_model_raises():
    with pytest.raises(OpenAIChatModelConfigError):
        OpenAIChatModel(model="gpt-4o", api_key="sk-test", fallback_providers=[{"api_key": "x"}])


def test_fallback_not_used_when_primary_succeeds():
    llm = make_llm_with_fallback(max_retries=1)
    llm._client.chat.completions.create.return_value = make_completion("Paris")
    result = llm.invoke("hi")
    assert result == "Paris"
    assert llm.last_metadata["provider_used"] == "primary"
    llm._fallback_sync_clients[0].chat.completions.create.assert_not_called()


def test_fallback_used_when_primary_fails():
    llm = make_llm_with_fallback(max_retries=1)
    llm._client.chat.completions.create.side_effect = RuntimeError("primary down")
    llm._fallback_sync_clients[0].chat.completions.create.return_value = make_completion("Berlin")
    result = llm.invoke("hi")
    assert result == "Berlin"
    assert llm.last_metadata["provider_used"] == "fallback[0]:backup-model"
    sent_model = llm._fallback_sync_clients[0].chat.completions.create.call_args.kwargs["model"]
    assert sent_model == "backup-model"


def test_ainvoke_uses_fallback_on_primary_failure():
    llm = make_llm_with_fallback(max_retries=1)
    llm._async_client.chat.completions.create.side_effect = RuntimeError("primary down")
    llm._fallback_async_clients[0].chat.completions.create.return_value = make_completion("Tokyo")

    async def run():
        return await llm.ainvoke("hi")

    assert asyncio.run(run()) == "Tokyo"
    assert llm.last_metadata["provider_used"] == "fallback[0]:backup-model"


def test_all_providers_fail_raises_aggregate_error():
    llm = make_llm_with_fallback(fallback_models=["backup-1", "backup-2"], max_retries=1)
    llm._client.chat.completions.create.side_effect = RuntimeError("primary down")
    llm._fallback_sync_clients[0].chat.completions.create.side_effect = RuntimeError("backup-1 down")
    llm._fallback_sync_clients[1].chat.completions.create.side_effect = RuntimeError("backup-2 down")

    with pytest.raises(OpenAIChatModelAllProvidersFailedError) as exc_info:
        llm.invoke("hi")
    assert len(exc_info.value.attempts) == 3
    labels = [label for label, _ in exc_info.value.attempts]
    assert labels == ["primary", "fallback[0]:backup-1", "fallback[1]:backup-2"]


def test_invoke_with_tools_uses_fallback_on_primary_failure():
    llm = make_llm_with_fallback(max_retries=1)
    llm._client.chat.completions.create.side_effect = RuntimeError("primary down")
    llm._fallback_sync_clients[0].chat.completions.create.return_value = make_completion("sunny")
    response = llm.invoke_with_tools("weather?", [{"name": "get_weather", "parameters": {}}])
    assert response.text == "sunny"


def test_stream_fallback_before_any_chunk_emitted():
    llm = make_llm_with_fallback(max_retries=1)
    llm._client.chat.completions.create.side_effect = RuntimeError("primary down")
    llm._fallback_sync_clients[0].chat.completions.create.return_value = FakeSyncStream(
        make_stream_chunks(["Hel", "lo"])
    )
    assert list(llm.stream("hi")) == ["Hel", "lo"]


def test_stream_no_fallback_after_partial_emit():
    """Once a provider has streamed partial text, a mid-stream failure must not
    silently switch to the fallback provider (would duplicate/corrupt output)."""

    def failing_stream(**kwargs):
        yield from make_stream_chunks(["Hel"])
        raise RuntimeError("dropped mid-stream")

    llm = make_llm_with_fallback(max_retries=1)
    llm._client.chat.completions.create.side_effect = lambda **kwargs: failing_stream(**kwargs)

    with pytest.raises(OpenAIChatModelAPIError, match="mid-response"):
        list(llm.stream("hi"))
    llm._fallback_sync_clients[0].chat.completions.create.assert_not_called()


# ── 19. Validated structured output (invoke_structured) ─────────────────────

from autourgos_openaichat import OpenAIChatModelValidationError


class CityInfo(BaseModel):
    city: str
    population: int


def test_invoke_structured_requires_pydantic_schema():
    llm = make_llm(output_schema={"type": "object"})
    with pytest.raises(OpenAIChatModelConfigError):
        llm.invoke_structured("Tokyo")


def test_invoke_structured_requires_output_schema_at_all():
    llm = make_llm()
    with pytest.raises(OpenAIChatModelConfigError):
        llm.invoke_structured("Tokyo")


def test_invoke_structured_incompatible_with_streaming():
    llm = make_llm(output_schema=CityInfo, streaming=True)
    with pytest.raises(OpenAIChatModelConfigError):
        llm.invoke_structured("Tokyo")


def test_invoke_structured_success_first_try():
    llm = make_llm(output_schema=CityInfo)
    llm._client.chat.completions.create.return_value = make_completion(
        json.dumps({"city": "Tokyo", "population": 13960000})
    )
    result = llm.invoke_structured("Tell me about Tokyo.")
    assert isinstance(result, CityInfo)
    assert result.city == "Tokyo"
    assert result.population == 13960000
    assert llm.last_metadata["validation_retries"] == 0
    assert llm.last_metadata["provider_used"] == "primary"


def test_invoke_structured_retries_on_validation_failure_then_succeeds():
    llm = make_llm(output_schema=CityInfo)
    llm._client.chat.completions.create.side_effect = [
        make_completion(json.dumps({"city": "Tokyo"})),          # missing population -> invalid
        make_completion(json.dumps({"city": "Tokyo", "population": 13960000})),
    ]
    result = llm.invoke_structured("Tell me about Tokyo.")
    assert result.population == 13960000
    assert llm.last_metadata["validation_retries"] == 1
    assert llm._client.chat.completions.create.call_count == 2
    # second call includes the correction messages fed back to the model
    second_messages = llm._client.chat.completions.create.call_args_list[1].kwargs["messages"]
    assert second_messages[-2]["role"] == "assistant"
    assert second_messages[-1]["role"] == "user"
    assert "failed schema validation" in second_messages[-1]["content"]


def test_invoke_structured_raises_after_retries_exhausted():
    llm = make_llm(output_schema=CityInfo)
    llm._client.chat.completions.create.return_value = make_completion(
        json.dumps({"city": "Tokyo"})  # always missing population
    )
    with pytest.raises(OpenAIChatModelValidationError) as exc_info:
        llm.invoke_structured("Tell me about Tokyo.", max_validation_retries=1)
    assert llm._client.chat.completions.create.call_count == 2  # 1 initial + 1 retry
    assert exc_info.value.raw_text == json.dumps({"city": "Tokyo"})
    assert exc_info.value.validation_error is not None


def test_ainvoke_structured_success():
    llm = make_llm(output_schema=CityInfo)
    llm._async_client.chat.completions.create.return_value = make_completion(
        json.dumps({"city": "Paris", "population": 2148000})
    )

    async def run():
        return await llm.ainvoke_structured("Tell me about Paris.")

    result = asyncio.run(run())
    assert isinstance(result, CityInfo)
    assert result.city == "Paris"


# ── 20. Call ledger ──────────────────────────────────────────────────────────

import sqlite3


def _read_ledger_rows(db_path):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute("SELECT * FROM calls ORDER BY id").fetchall()]
    conn.close()
    return rows


def test_ledger_disabled_by_default():
    llm = make_llm()
    assert llm._ledger_conn is None
    llm._client.chat.completions.create.return_value = make_completion("Paris")
    assert llm.invoke("hi") == "Paris"  # no ledger, no crash


def test_ledger_records_successful_invoke(tmp_path):
    db_path = tmp_path / "calls.db"
    llm = make_llm(ledger_path=str(db_path))
    llm._client.chat.completions.create.return_value = make_completion("Paris")
    llm.invoke("What is the capital of France?")

    rows = _read_ledger_rows(db_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["call_type"] == "invoke"
    assert row["provider_used"] == "primary"
    assert row["prompt"] == "What is the capital of France?"
    assert row["response"] == "Paris"
    assert row["input_tokens"] == 9
    assert row["output_tokens"] == 10


def test_ledger_store_content_false_omits_text(tmp_path):
    db_path = tmp_path / "calls.db"
    llm = make_llm(ledger_path=str(db_path), ledger_store_content=False)
    llm._client.chat.completions.create.return_value = make_completion("Paris")
    llm.invoke("What is the capital of France?")

    row = _read_ledger_rows(db_path)[0]
    assert row["prompt"] is None
    assert row["response"] is None
    assert row["input_tokens"] == 9


def test_ledger_records_ainvoke_and_invoke_structured(tmp_path):
    db_path = tmp_path / "calls.db"
    llm = make_llm(ledger_path=str(db_path))
    llm._async_client.chat.completions.create.return_value = make_completion("Berlin")

    async def run():
        return await llm.ainvoke("Capital of Germany?")

    asyncio.run(run())

    llm2 = make_llm(ledger_path=str(db_path), output_schema=CityInfo)
    llm2._client.chat.completions.create.return_value = make_completion(
        json.dumps({"city": "Tokyo", "population": 13960000})
    )
    llm2.invoke_structured("Tell me about Tokyo.")

    rows = _read_ledger_rows(db_path)
    call_types = [r["call_type"] for r in rows]
    assert call_types == ["ainvoke", "invoke_structured"]
    assert rows[1]["validation_retries"] == 0


def test_ledger_write_failure_does_not_break_invoke(tmp_path):
    """A broken ledger connection must never break the actual LLM call."""
    db_path = tmp_path / "calls.db"
    llm = make_llm(ledger_path=str(db_path))
    llm._ledger_conn.close()  # simulate a broken/unwritable ledger
    llm._client.chat.completions.create.return_value = make_completion("Paris")
    assert llm.invoke("hi") == "Paris"


# ── 21. Budget governor ──────────────────────────────────────────────────────

from autourgos_openaichat import BudgetExceededException


def test_max_session_cost_requires_both_pricings():
    with pytest.raises(OpenAIChatModelConfigError):
        OpenAIChatModel(model="gpt-4o", api_key="sk-test", max_session_cost=1.0, input_pricing=1.0)


def test_budget_governor_allows_calls_under_cap():
    llm = make_llm(input_pricing=1000, output_pricing=1000, max_session_cost=10.0)
    llm._client.chat.completions.create.return_value = make_completion("Paris")  # usage (9, 10, 19)
    assert llm.invoke("hi") == "Paris"
    assert llm.invoke("hi again") == "Paris"
    assert llm._client.chat.completions.create.call_count == 2


def test_budget_governor_blocks_once_cap_reached():
    # each call costs (9/1e6)*1000 + (10/1e6)*1000 = 0.019; cap is below that.
    llm = make_llm(input_pricing=1000, output_pricing=1000, max_session_cost=0.01)
    llm._client.chat.completions.create.return_value = make_completion("Paris")
    assert llm.invoke("hi") == "Paris"  # 1st call goes through, pushes usage over the cap
    assert llm.session_cost_used >= 0.01

    with pytest.raises(BudgetExceededException):
        llm.invoke("hi again")
    assert llm._client.chat.completions.create.call_count == 1  # 2nd call never reached the API


def test_reset_session_budget_unblocks():
    llm = make_llm(input_pricing=1000, output_pricing=1000, max_session_cost=0.01)
    llm._client.chat.completions.create.return_value = make_completion("Paris")
    llm.invoke("hi")
    with pytest.raises(BudgetExceededException):
        llm.invoke("hi again")

    llm.reset_session_budget()
    assert llm.session_cost_used == 0.0
    assert llm.invoke("hi again") == "Paris"


def test_budget_exceeded_does_not_trip_circuit_breaker():
    llm = make_llm(
        input_pricing=1000, output_pricing=1000, max_session_cost=0.01,
        circuit_failure_threshold=1,
    )
    llm._client.chat.completions.create.return_value = make_completion("Paris")
    llm.invoke("hi")
    for _ in range(3):
        with pytest.raises(BudgetExceededException):
            llm.invoke("hi again")

    llm.reset_session_budget()
    assert llm.invoke("hi again") == "Paris"  # circuit breaker never tripped


# ── 22. PII / secret redaction ───────────────────────────────────────────────

from autourgos_openaichat import OpenAIChatModelRedactionBlockedError


def test_redaction_disabled_by_default():
    llm = make_llm()
    llm._client.chat.completions.create.return_value = make_completion("ok")
    llm.invoke("my email is bob@example.com")
    sent = llm._client.chat.completions.create.call_args.kwargs["messages"]
    assert sent[0]["content"] == "my email is bob@example.com"
    assert llm.last_redacted_categories == []


def test_redaction_mask_mode_replaces_email_and_api_key():
    llm = make_llm(redact_pii=True)
    llm._client.chat.completions.create.return_value = make_completion("ok")
    llm.invoke("contact me at bob@example.com, my key is sk-abcdefghijklmnopqrst")
    sent = llm._client.chat.completions.create.call_args.kwargs["messages"]
    text = sent[0]["content"]
    assert "bob@example.com" not in text
    assert "sk-abcdefghijklmnopqrst" not in text
    assert "[REDACTED:email]" in text
    assert "[REDACTED:api_key]" in text
    assert set(llm.last_redacted_categories) == {"email", "api_key"}


def test_redaction_block_mode_raises_before_api_call():
    llm = make_llm(redact_pii=True, redact_mode="block")
    with pytest.raises(OpenAIChatModelRedactionBlockedError) as exc_info:
        llm.invoke("my email is bob@example.com")
    assert "email" in exc_info.value.categories_found
    llm._client.chat.completions.create.assert_not_called()


def test_redaction_categories_filter_limits_scanning():
    llm = make_llm(redact_pii=True, redact_categories=["api_key"])
    llm._client.chat.completions.create.return_value = make_completion("ok")
    llm.invoke("my email is bob@example.com")  # email not in the enabled category list
    sent = llm._client.chat.completions.create.call_args.kwargs["messages"]
    assert sent[0]["content"] == "my email is bob@example.com"


def test_redaction_custom_pattern():
    llm = make_llm(redact_pii=True, redact_custom_patterns={"internal_id": r"EMP-\d{5}"})
    llm._client.chat.completions.create.return_value = make_completion("ok")
    llm.invoke("employee EMP-12345 requested access")
    sent = llm._client.chat.completions.create.call_args.kwargs["messages"]
    assert "[REDACTED:internal_id]" in sent[0]["content"]


def test_redaction_applies_to_multi_turn_message_list():
    llm = make_llm(redact_pii=True)
    llm._client.chat.completions.create.return_value = make_completion("ok")
    messages = [{"role": "user", "content": "email me at bob@example.com"}]
    llm.invoke(messages)
    sent = llm._client.chat.completions.create.call_args.kwargs["messages"]
    assert "[REDACTED:email]" in sent[0]["content"]


def test_invalid_redact_mode_raises_config_error():
    with pytest.raises(OpenAIChatModelConfigError):
        OpenAIChatModel(model="gpt-4o", api_key="sk-test", redact_pii=True, redact_mode="bogus")


def test_unknown_redact_category_raises_config_error():
    with pytest.raises(OpenAIChatModelConfigError):
        OpenAIChatModel(model="gpt-4o", api_key="sk-test", redact_pii=True, redact_categories=["not_a_real_category"])


def test_ledger_records_redacted_categories(tmp_path):
    db_path = tmp_path / "calls.db"
    llm = make_llm(ledger_path=str(db_path), redact_pii=True)
    llm._client.chat.completions.create.return_value = make_completion("ok")
    llm.invoke("my email is bob@example.com")
    row = _read_ledger_rows(db_path)[0]
    assert row["redacted_categories"] == "email"


# ── 23. Shadow-mode dual dispatch ────────────────────────────────────────────

def make_llm_with_shadow(shadow_models=("backup-model",), **kwargs):
    """Construct an OpenAIChatModel with a mocked primary and mocked shadow clients."""
    llm = OpenAIChatModel(
        model="gpt-4o",
        api_key="sk-test",
        shadow_providers=[{"model": m, "api_key": "sk-shadow"} for m in shadow_models],
        **kwargs,
    )
    llm._client = MagicMock()
    llm._async_client = MagicMock()
    llm._async_client.chat.completions.create = AsyncMock()
    for i in range(len(shadow_models)):
        sh_sync = MagicMock()
        sh_async = MagicMock()
        sh_async.chat.completions.create = AsyncMock()
        llm._shadow_sync_clients[i] = sh_sync
        llm._shadow_async_clients[i] = sh_async
    return llm


def test_shadow_disabled_by_default():
    llm = make_llm()
    llm._client.chat.completions.create.return_value = make_completion("Paris")
    assert llm.invoke("hi") == "Paris"
    assert llm.last_shadow_results == []


def test_shadow_dispatch_returns_primary_and_records_shadow():
    llm = make_llm_with_shadow(input_pricing=1000, output_pricing=1000)
    llm._client.chat.completions.create.return_value = make_completion("Paris")
    llm._shadow_sync_clients[0].chat.completions.create.return_value = make_completion("Paris")

    result = llm.invoke("What is the capital of France?")
    assert result == "Paris"  # always the primary's answer
    assert len(llm.last_shadow_results) == 1
    shadow = llm.last_shadow_results[0]
    assert shadow["provider_used"] == "shadow[0]:backup-model"
    assert shadow["response"] == "Paris"
    assert shadow["similarity"] == 1.0
    assert shadow["error"] is None
    assert shadow["total_cost"] is not None
    sent_model = llm._shadow_sync_clients[0].chat.completions.create.call_args.kwargs["model"]
    assert sent_model == "backup-model"


def test_shadow_low_similarity_for_different_text():
    llm = make_llm_with_shadow()
    llm._client.chat.completions.create.return_value = make_completion("Paris")
    llm._shadow_sync_clients[0].chat.completions.create.return_value = make_completion(
        "The capital of France is a large European city called Paris, founded long ago."
    )
    llm.invoke("hi")
    assert llm.last_shadow_results[0]["similarity"] < 0.5


def test_shadow_provider_failure_does_not_break_primary():
    llm = make_llm_with_shadow()
    llm._client.chat.completions.create.return_value = make_completion("Paris")
    llm._shadow_sync_clients[0].chat.completions.create.side_effect = RuntimeError("shadow down")

    result = llm.invoke("hi")
    assert result == "Paris"
    shadow = llm.last_shadow_results[0]
    assert shadow["error"] is not None
    assert shadow["response"] is None


def test_shadow_on_result_callback_invoked():
    seen = []
    llm = make_llm_with_shadow(on_shadow_result=lambda r: seen.append(r))
    llm._client.chat.completions.create.return_value = make_completion("Paris")
    llm._shadow_sync_clients[0].chat.completions.create.return_value = make_completion("Paris")
    llm.invoke("hi")
    assert len(seen) == 1
    assert seen[0]["provider_used"] == "shadow[0]:backup-model"


def test_shadow_recorded_in_ledger(tmp_path):
    db_path = tmp_path / "calls.db"
    llm = make_llm_with_shadow(ledger_path=str(db_path))
    llm._client.chat.completions.create.return_value = make_completion("Paris")
    llm._shadow_sync_clients[0].chat.completions.create.return_value = make_completion("Paris")
    llm.invoke("hi")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute("SELECT * FROM shadow_calls").fetchall()]
    conn.close()
    assert len(rows) == 1
    assert rows[0]["provider_used"] == "shadow[0]:backup-model"
    assert rows[0]["similarity"] == 1.0


def test_ainvoke_shadow_dispatch():
    llm = make_llm_with_shadow()
    llm._async_client.chat.completions.create.return_value = make_completion("Berlin")
    llm._shadow_async_clients[0].chat.completions.create.return_value = make_completion("Berlin")

    async def run():
        return await llm.ainvoke("Capital of Germany?")

    result = asyncio.run(run())
    assert result == "Berlin"
    assert len(llm.last_shadow_results) == 1
    assert llm.last_shadow_results[0]["similarity"] == 1.0


def test_shadow_config_missing_model_raises():
    with pytest.raises(OpenAIChatModelConfigError):
        OpenAIChatModel(model="gpt-4o", api_key="sk-test", shadow_providers=[{"api_key": "x"}])


# ── 24. Redaction restore-in-response ────────────────────────────────────────

def test_redact_restore_requires_redact_pii():
    with pytest.raises(OpenAIChatModelConfigError):
        OpenAIChatModel(model="gpt-4o", api_key="sk-test", redact_restore_in_response=True)


def test_redact_restore_requires_mask_mode():
    with pytest.raises(OpenAIChatModelConfigError):
        OpenAIChatModel(
            model="gpt-4o", api_key="sk-test",
            redact_pii=True, redact_mode="block", redact_restore_in_response=True,
        )


def test_redact_restore_disabled_placeholder_stays_in_output():
    """Without redact_restore_in_response, an echoed placeholder is returned as-is."""
    llm = make_llm(redact_pii=True, redact_categories=["email"])
    llm._client.chat.completions.create.return_value = make_completion(
        "Sure, noted contact: [REDACTED:email]"
    )
    result = llm.invoke("contact bob@example.com please")
    assert result == "Sure, noted contact: [REDACTED:email]"


def test_redact_restore_swaps_placeholder_back_in_output():
    llm = make_llm(redact_pii=True, redact_categories=["email"], redact_restore_in_response=True)
    llm._client.chat.completions.create.return_value = make_completion(
        "Sure, noted contact: [REDACTED:email:1]"
    )
    result = llm.invoke("contact bob@example.com please")
    assert result == "Sure, noted contact: bob@example.com"

    sent = llm._client.chat.completions.create.call_args.kwargs["messages"]
    assert "bob@example.com" not in sent[0]["content"]  # the model itself never saw it
    assert "[REDACTED:email:1]" in sent[0]["content"]


def test_redact_restore_multiple_secrets_each_correct():
    llm = make_llm(redact_pii=True, redact_categories=["email"], redact_restore_in_response=True)
    llm._client.chat.completions.create.return_value = make_completion(
        "Contacts: [REDACTED:email:1] and [REDACTED:email:2]"
    )
    result = llm.invoke("emails: alice@example.com and bob@example.com")
    assert result == "Contacts: alice@example.com and bob@example.com"


def test_redact_restore_ledger_stays_masked(tmp_path):
    db_path = tmp_path / "calls.db"
    llm = make_llm(
        ledger_path=str(db_path), redact_pii=True, redact_categories=["email"],
        redact_restore_in_response=True,
    )
    llm._client.chat.completions.create.return_value = make_completion(
        "Contact: [REDACTED:email:1]"
    )
    result = llm.invoke("email bob@example.com")
    assert result == "Contact: bob@example.com"

    row = _read_ledger_rows(db_path)[0]
    assert row["response"] == "Contact: [REDACTED:email:1]"
    assert "bob@example.com" not in row["response"]


def test_invoke_structured_restore_fixes_validation():
    class ContactInfo(BaseModel):
        email: str

        @field_validator("email")
        @classmethod
        def must_look_like_email(cls, v: str) -> str:
            if "@" not in v:
                raise ValueError("not a valid email")
            return v

    llm = make_llm(
        output_schema=ContactInfo, redact_pii=True, redact_categories=["email"],
        redact_restore_in_response=True,
    )
    llm._client.chat.completions.create.return_value = make_completion(
        json.dumps({"email": "[REDACTED:email:1]"})
    )
    result = llm.invoke_structured("extract contact from: reach me at bob@example.com")
    assert result.email == "bob@example.com"
    llm._client.chat.completions.create.assert_called_once()  # validated on the first attempt


def test_invoke_structured_correction_message_uses_masked_text_not_restored():
    """A failed-validation retry must feed the model back its own masked output,
    never the restored secret — otherwise the secret leaks into the model's context."""
    llm = make_llm(
        output_schema=CityInfo, redact_pii=True, redact_categories=["email"],
        redact_restore_in_response=True,
    )
    llm._client.chat.completions.create.side_effect = [
        make_completion(json.dumps({"city": "[REDACTED:email:1]"})),  # missing population -> invalid
        make_completion(json.dumps({"city": "Tokyo", "population": 13960000})),
    ]
    llm.invoke_structured("contact bob@example.com about Tokyo")
    second_messages = llm._client.chat.completions.create.call_args_list[1].kwargs["messages"]
    correction_text = second_messages[-2]["content"]
    assert "[REDACTED:email:1]" in correction_text
    assert "bob@example.com" not in correction_text


# ── 25. Bring-your-own redaction dictionary ──────────────────────────────────

import tempfile
import os


def test_redact_custom_terms_literal_match():
    llm = make_llm(
        redact_pii=True,
        redact_categories=[],  # built-ins fully off
        redact_custom_terms={"codenames": ["EAGLE STRIKE", "MIDNIGHT RAVEN"]},
    )
    llm._client.chat.completions.create.return_value = make_completion("ok")
    llm.invoke("Operation EAGLE STRIKE begins at dawn")
    sent = llm._client.chat.completions.create.call_args.kwargs["messages"]
    assert "EAGLE STRIKE" not in sent[0]["content"]
    assert "[REDACTED:codenames]" in sent[0]["content"]
    assert "codenames" in llm.last_redacted_categories


def test_redact_custom_terms_matches_literally_not_as_regex():
    """A term containing regex-special characters must still match literally."""
    llm = make_llm(
        redact_pii=True,
        redact_categories=[],
        redact_custom_terms={"asset_id": ["ASSET(7).X"]},
    )
    llm._client.chat.completions.create.return_value = make_completion("ok")
    llm.invoke("Deploy ASSET(7).X immediately")
    sent = llm._client.chat.completions.create.call_args.kwargs["messages"]
    assert "ASSET(7).X" not in sent[0]["content"]
    assert "[REDACTED:asset_id]" in sent[0]["content"]


def test_redact_only_custom_dictionary_builtins_disabled():
    llm = make_llm(
        redact_pii=True,
        redact_categories=[],
        redact_custom_patterns={"secret_project": r"PROJECT-\d+"},
    )
    llm._client.chat.completions.create.return_value = make_completion("ok")
    llm.invoke("my email is bob@example.com, see PROJECT-42")  # email NOT redacted, built-ins off
    sent = llm._client.chat.completions.create.call_args.kwargs["messages"]
    assert "bob@example.com" in sent[0]["content"]
    assert "PROJECT-42" not in sent[0]["content"]


def test_redact_patterns_file_loads_patterns_and_terms():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "dictionary.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({
                "patterns": {"classification_marking": r"TOP SECRET"},
                "terms": {"codenames": ["MIDNIGHT RAVEN"]},
            }, fh)

        llm = make_llm(redact_pii=True, redact_categories=[], redact_patterns_file=path)
        llm._client.chat.completions.create.return_value = make_completion("ok")
        llm.invoke("TOP SECRET: operation MIDNIGHT RAVEN is active")
        sent = llm._client.chat.completions.create.call_args.kwargs["messages"]
        assert "TOP SECRET" not in sent[0]["content"]
        assert "MIDNIGHT RAVEN" not in sent[0]["content"]
        assert set(llm.last_redacted_categories) == {"classification_marking", "codenames"}


def test_redact_patterns_file_missing_raises_config_error():
    with pytest.raises(OpenAIChatModelConfigError):
        OpenAIChatModel(
            model="gpt-4o", api_key="sk-test",
            redact_pii=True, redact_patterns_file="/no/such/file.json",
        )


def test_redact_patterns_file_malformed_json_raises_config_error():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bad.json")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("{not valid json")
        with pytest.raises(OpenAIChatModelConfigError):
            OpenAIChatModel(
                model="gpt-4o", api_key="sk-test",
                redact_pii=True, redact_patterns_file=path,
            )


def test_redact_inline_custom_patterns_override_file_on_name_collision():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "dictionary.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"patterns": {"secret_project": r"PROJECT-\d+"}}, fh)

        llm = make_llm(
            redact_pii=True,
            redact_categories=[],
            redact_patterns_file=path,
            redact_custom_patterns={"secret_project": r"OVERRIDE-\d+"},  # same name, inline wins
        )
        llm._client.chat.completions.create.return_value = make_completion("ok")
        llm.invoke("see PROJECT-42 and OVERRIDE-99")
        sent = llm._client.chat.completions.create.call_args.kwargs["messages"]
        assert "PROJECT-42" in sent[0]["content"]       # file pattern was overridden, no longer matches
        assert "OVERRIDE-99" not in sent[0]["content"]  # inline pattern wins


# ── 26. Constrained decoding passthrough (extra_body) ────────────────────────

def test_extra_body_absent_by_default():
    llm = make_llm()
    llm._client.chat.completions.create.return_value = make_completion("ok")
    llm.invoke("hi")
    kwargs = llm._client.chat.completions.create.call_args.kwargs
    assert "extra_body" not in kwargs


def test_extra_body_sent_on_invoke():
    guided = {"guided_json": {"type": "object", "properties": {"x": {"type": "string"}}}}
    llm = make_llm(extra_body=guided)
    llm._client.chat.completions.create.return_value = make_completion("ok")
    llm.invoke("hi")
    kwargs = llm._client.chat.completions.create.call_args.kwargs
    assert kwargs["extra_body"] == guided


def test_extra_body_sent_on_ainvoke_and_stream():
    guided = {"grammar": 'root ::= "yes" | "no"'}
    llm = make_llm(extra_body=guided)

    llm._async_client.chat.completions.create.return_value = make_completion("ok")

    async def run():
        return await llm.ainvoke("hi")

    asyncio.run(run())
    assert llm._async_client.chat.completions.create.call_args.kwargs["extra_body"] == guided

    llm._client.chat.completions.create.return_value = FakeSyncStream(make_stream_chunks(["hi"]))
    list(llm.stream("hi"))
    assert llm._client.chat.completions.create.call_args.kwargs["extra_body"] == guided


def test_extra_body_sent_on_invoke_with_tools_and_invoke_structured():
    guided = {"guided_choice": ["yes", "no"]}
    llm = make_llm(extra_body=guided)
    llm._client.chat.completions.create.return_value = make_completion("ok")
    llm.invoke_with_tools("hi", [{"name": "noop", "parameters": {}}])
    assert llm._client.chat.completions.create.call_args.kwargs["extra_body"] == guided

    llm2 = make_llm(extra_body=guided, output_schema=CityCountryInfo)
    llm2._client.chat.completions.create.return_value = make_completion(
        json.dumps({"city": "Tokyo", "country": "Japan"})
    )
    llm2.invoke_structured("hi")
    assert llm2._client.chat.completions.create.call_args.kwargs["extra_body"] == guided


def test_extra_body_propagates_to_fallback_provider():
    guided = {"guided_json": {"type": "object"}}
    llm = make_llm_with_fallback(extra_body=guided, max_retries=1)
    llm._client.chat.completions.create.side_effect = RuntimeError("primary down")
    llm._fallback_sync_clients[0].chat.completions.create.return_value = make_completion("ok")
    llm.invoke("hi")
    fb_kwargs = llm._fallback_sync_clients[0].chat.completions.create.call_args.kwargs
    assert fb_kwargs["extra_body"] == guided
