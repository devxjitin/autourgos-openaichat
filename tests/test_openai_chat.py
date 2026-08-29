"""
Feature-by-feature test suite for OpenAIChatModel.

Every test mocks the underlying openai.OpenAI / openai.AsyncOpenAI client so
no network calls are made, but exercises the real autourgos_openaichat code
paths (message building, retries, parsing, circuit breaker, etc.) end to end.
"""
from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from pydantic import BaseModel, Field

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


# ── 4. Async streaming ───────────────────────────────────────────────────────

def test_astream_yields_chunks():
    llm = make_llm()
    llm._async_client.chat.completions.create.return_value = FakeAsyncStream(
        make_stream_chunks(["1... ", "2... ", "3..."])
    )

    async def run():
        return [c async for c in llm.astream("count")]

    assert asyncio.run(run()) == ["1... ", "2... ", "3..."]


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

class CityInfo(BaseModel):
    city: str = Field(description="Name of the city")
    country: str = Field(description="Name of the country")


def test_structured_output_with_pydantic_schema():
    llm = make_llm(output_schema=CityInfo, structured_output=True)
    payload = json.dumps({"city": "Tokyo", "country": "Japan"})
    llm._client.chat.completions.create.return_value = make_completion(payload)
    result = llm.invoke("Tell me about Tokyo.")
    assert isinstance(result, dict)
    assert json.loads(result["response"]) == {"city": "Tokyo", "country": "Japan"}
    kwargs = llm._client.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"]["type"] == "json_schema"
    assert kwargs["response_format"]["json_schema"]["name"] == "CityInfo"


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
