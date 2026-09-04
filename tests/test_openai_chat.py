"""
Feature-by-feature test suite for OpenAIChatModel.

Every test mocks the underlying openai.OpenAI / openai.AsyncOpenAI client so
no network calls are made, but exercises the real autourgos_openaichat code
paths (message building, retries, parsing, circuit breaker, etc.) end to end.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from pydantic import BaseModel, Field, field_validator

from autourgos_openaichat import (
    OpenAIChatModel,
    OpenAIChatModelAPIError,
    OpenAIChatModelConfigError,
    OpenAIChatModelImportError,
    OpenAIChatModelRedactionBlockedError,
    OpenAIChatModelRefusalError,
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


def make_usage_chunk(prompt_tokens=9, completion_tokens=10, total_tokens=19):
    """
    The terminal SSE chunk sent when stream_options={"include_usage": True}
    is set -- no `delta`/`choices`, just `usage`. See extract_usage_bearing_event().
    """
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens
    )
    return SimpleNamespace(choices=[], usage=usage)


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


def test_store_true_is_sent_when_set_at_construction():
    llm = make_llm(store=True)
    llm._client.chat.completions.create.return_value = make_completion("ok")
    llm.invoke("hi")
    kwargs = llm._client.chat.completions.create.call_args.kwargs
    assert kwargs["store"] is True


def test_store_false_is_sent_when_set_at_construction():
    llm = make_llm(store=False)
    llm._client.chat.completions.create.return_value = make_completion("ok")
    llm.invoke("hi")
    kwargs = llm._client.chat.completions.create.call_args.kwargs
    assert kwargs["store"] is False


def test_store_omitted_by_default():
    """Regression: store used to be unreachable as a constructor setting at
    all. None (default) must omit the key entirely, not send a literal null."""
    llm = make_llm()
    llm._client.chat.completions.create.return_value = make_completion("ok")
    llm.invoke("hi")
    kwargs = llm._client.chat.completions.create.call_args.kwargs
    assert "store" not in kwargs


def test_store_per_call_override_wins_over_constructor_default():
    llm = make_llm(store=True)
    llm._client.chat.completions.create.return_value = make_completion("ok")
    llm.invoke("hi", store=False)
    kwargs = llm._client.chat.completions.create.call_args.kwargs
    assert kwargs["store"] is False


def test_o_series_model_gets_max_completion_tokens_not_max_tokens():
    """
    Regression test: OpenAI's Chat Completions `max_tokens` is deprecated
    and documented as "not compatible with o-series models" (o1, o3,
    o4-mini, ...) -- sending it 400s. o-series models must get
    `max_completion_tokens` instead.
    """
    llm = OpenAIChatModel(model="o3-mini", api_key="sk-test", max_tokens=500)
    llm._client = MagicMock()
    llm._client.chat.completions.create.return_value = make_completion("ok")
    llm.invoke("hi")
    kwargs = llm._client.chat.completions.create.call_args.kwargs
    assert kwargs.get("max_completion_tokens") == 500
    assert "max_tokens" not in kwargs


def test_o_series_model_drops_temperature_and_top_p():
    """
    Regression: o-series reasoning models reject temperature/top_p outright
    (400) -- same model family, same "not compatible" constraint as
    max_tokens (see test above). They must be dropped, not sent.
    """
    llm = OpenAIChatModel(model="o3-mini", api_key="sk-test", temperature=0.7, top_p=0.9)
    llm._client = MagicMock()
    llm._client.chat.completions.create.return_value = make_completion("ok")
    llm.invoke("hi")
    kwargs = llm._client.chat.completions.create.call_args.kwargs
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs


def test_regular_model_still_gets_temperature_and_top_p():
    llm = make_llm(temperature=0.7, top_p=0.9)
    llm._client.chat.completions.create.return_value = make_completion("ok")
    llm.invoke("hi")
    kwargs = llm._client.chat.completions.create.call_args.kwargs
    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 0.9


def test_regular_model_still_gets_max_tokens():
    """Non-o-series models (the vast majority, including third-party/self-hosted
    ones) must keep sending max_tokens as before -- no behavior change."""
    llm = make_llm(max_tokens=256)
    llm._client.chat.completions.create.return_value = make_completion("ok")
    llm.invoke("hi")
    kwargs = llm._client.chat.completions.create.call_args.kwargs
    assert kwargs.get("max_tokens") == 256
    assert "max_completion_tokens" not in kwargs


def test_fallback_to_different_model_family_gets_correct_max_tokens_key():
    """
    The max_tokens/max_completion_tokens decision must be made per-target,
    not once for the whole call -- a fallback provider can be a different
    model family than the primary (e.g. primary gpt-4o, fallback o1-mini).
    """
    llm = make_llm_with_fallback(fallback_models=("o1-mini",), max_retries=1, max_tokens=500)
    llm._client.chat.completions.create.side_effect = RuntimeError("primary down")
    llm._fallback_sync_clients[0].chat.completions.create.return_value = make_completion("ok")

    llm.invoke("hi")
    kwargs = llm._fallback_sync_clients[0].chat.completions.create.call_args.kwargs
    assert kwargs.get("max_completion_tokens") == 500
    assert "max_tokens" not in kwargs


def test_fallback_to_different_model_family_drops_temperature_top_p_per_target():
    """
    Same per-target reasoning as the max_tokens test above, applied to
    temperature/top_p: a normal primary with temperature/top_p set must
    still send them, but an o-series fallback must not.
    """
    llm = make_llm_with_fallback(
        fallback_models=("o1-mini",), max_retries=1, temperature=0.7, top_p=0.9
    )
    llm._client.chat.completions.create.side_effect = RuntimeError("primary down")
    llm._fallback_sync_clients[0].chat.completions.create.return_value = make_completion("ok")

    llm.invoke("hi")
    kwargs = llm._fallback_sync_clients[0].chat.completions.create.call_args.kwargs
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs


def test_invoke_streaming_o_series_model_drops_temperature_top_p():
    llm = OpenAIChatModel(
        model="o3-mini", api_key="sk-test", temperature=0.7, top_p=0.9, streaming=True
    )
    llm._client = MagicMock()
    llm._client.chat.completions.create.return_value = FakeSyncStream(make_stream_chunks(["ok"]))
    llm.invoke("hi")
    kwargs = llm._client.chat.completions.create.call_args.kwargs
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs


def test_max_tokens_override_to_max_completion_tokens_key_drops_stale_default():
    """
    A caller overriding with max_completion_tokens= (the "other" key name
    than the constructor's max_tokens= default) must not end up sending
    both keys -- overrides only add/replace same-named keys, so the stale
    max_tokens default has to be explicitly cleaned up.
    """
    llm = make_llm(max_tokens=500)
    llm._client.chat.completions.create.return_value = make_completion("ok")
    llm.invoke("hi", max_completion_tokens=999)
    kwargs = llm._client.chat.completions.create.call_args.kwargs
    assert kwargs.get("max_completion_tokens") == 999
    assert "max_tokens" not in kwargs


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


def test_stream_request_includes_stream_options_include_usage():
    """stream_options={"include_usage": True} must be set for stream=True requests
    (both stream() and invoke(streaming=True)) so the terminal usage chunk arrives."""
    llm = make_llm()
    llm._client.chat.completions.create.return_value = FakeSyncStream(make_stream_chunks(["ok"]))
    list(llm.stream("hi"))
    kwargs = llm._client.chat.completions.create.call_args.kwargs
    assert kwargs["stream_options"] == {"include_usage": True}


def test_invoke_streaming_tracks_budget_cost_and_ledger(tmp_path):
    """
    Regression test: invoke(streaming=True) used to bypass budget admission,
    cost tracking, and the ledger entirely -- max_session_cost was silently
    inert for streaming calls (a real billing/safety gap), and streaming
    calls never appeared in the audit ledger. Now the terminal usage chunk
    (stream_options={"include_usage": True}) lets the streaming path compute
    cost and log to the ledger exactly like the non-streaming path.
    """
    db_path = tmp_path / "calls.db"
    llm = make_llm(
        streaming=True, input_pricing=1000, output_pricing=1000,
        max_session_cost=0.01, ledger_path=str(db_path),
    )
    llm._client.chat.completions.create.return_value = FakeSyncStream(
        make_stream_chunks(["Hello"]) + [make_usage_chunk(9, 10, 19)]
    )

    assert llm.invoke("hi") == "Hello"
    assert llm.session_cost_used >= 0.01  # (9/1e6)*1000 + (10/1e6)*1000 = 0.019
    assert llm.last_metadata["total_tokens"] == 19
    assert llm.last_metadata["total_cost"] > 0

    row = _read_ledger_rows(db_path)[0]
    assert row["call_type"] == "invoke"
    assert row["total_tokens"] == 19

    with pytest.raises(BudgetExceededException):
        llm.invoke("hi again")  # cap now exceeded -- must actually block


def test_invoke_streaming_without_usage_chunk_degrades_gracefully():
    """A provider that doesn't honor stream_options still returns text; cost
    is simply left untracked for that call rather than crashing."""
    llm = make_llm(streaming=True, input_pricing=1000, output_pricing=1000)
    llm._client.chat.completions.create.return_value = FakeSyncStream(
        make_stream_chunks(["ok"])  # no trailing usage chunk
    )
    assert llm.invoke("hi") == "ok"
    assert llm.last_metadata.get("total_cost") is None
    assert llm.session_cost_used == 0.0


def make_refusal_stream_chunks(refusal_text):
    """A refusal-only stream: delta.refusal is set, delta.content stays None
    on every chunk -- matches the real Chat Completions SDK's ChoiceDelta."""
    chunks = []
    for w in refusal_text:
        delta = SimpleNamespace(content=None, refusal=w)
        choice = SimpleNamespace(delta=delta)
        chunks.append(SimpleNamespace(choices=[choice]))
    return chunks


def test_invoke_streaming_refusal_raises_refusal_error_not_generic():
    """
    Regression: a refusal-only stream (delta.refusal set, delta.content never
    set) used to be silently dropped by extract_text_delta_from_event(),
    surfacing only as a misleading generic "No text deltas" error.
    """
    llm = make_llm(streaming=True)
    llm._client.chat.completions.create.return_value = FakeSyncStream(
        make_refusal_stream_chunks(["I can't ", "help with that."])
    )
    with pytest.raises(OpenAIChatModelRefusalError) as exc_info:
        llm.invoke("hi")
    assert exc_info.value.refusal_text == "I can't help with that."


def test_invoke_streaming_refusal_does_not_fall_through_to_fallback():
    """
    Regression: a refusal on the primary used to be indistinguishable from a
    real failure, so the streaming loop fell through to the next fallback
    provider -- wasting a real API call for what was actually a valid,
    final answer from a working primary.
    """
    llm = make_llm_with_fallback(max_retries=1)
    llm.streaming = True
    llm._client.chat.completions.create.return_value = FakeSyncStream(
        make_refusal_stream_chunks(["nope"])
    )
    with pytest.raises(OpenAIChatModelRefusalError):
        llm.invoke("hi")
    llm._fallback_sync_clients[0].chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_ainvoke_streaming_refusal_raises_refusal_error():
    llm = make_llm(streaming=True)
    llm._async_client.chat.completions.create.return_value = FakeAsyncStream(
        make_refusal_stream_chunks(["refused"])
    )
    with pytest.raises(OpenAIChatModelRefusalError) as exc_info:
        await llm.ainvoke("hi")
    assert exc_info.value.refusal_text == "refused"


def test_invoke_non_stream_refusal_raises_refusal_error():
    """Non-streaming path: message.refusal set, message.content None."""
    llm = make_llm()
    msg = SimpleNamespace(content=None, refusal="cannot comply", tool_calls=None)
    choice = SimpleNamespace(message=msg)
    usage = SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8)
    llm._client.chat.completions.create.return_value = SimpleNamespace(choices=[choice], usage=usage)
    with pytest.raises(OpenAIChatModelRefusalError) as exc_info:
        llm.invoke("hi")
    assert exc_info.value.refusal_text == "cannot comply"


def test_invoke_streaming_restores_redacted_text():
    """redact_restore_in_response must apply on the streaming invoke() path too --
    it used to be skipped entirely, since streaming short-circuited before that code ran."""
    llm = make_llm(streaming=True, redact_pii=True, redact_categories=["email"],
                    redact_restore_in_response=True)
    llm._client.chat.completions.create.return_value = FakeSyncStream(
        make_stream_chunks(["Sure, noted: ", "[REDACTED:email:1]"])
    )
    result = llm.invoke("contact bob@example.com please")
    assert result == "Sure, noted: bob@example.com"


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

def test_ainvoke_streaming_tracks_budget_cost_and_ledger(tmp_path):
    """Async counterpart of test_invoke_streaming_tracks_budget_cost_and_ledger."""
    db_path = tmp_path / "calls.db"
    llm = make_llm(
        streaming=True, input_pricing=1000, output_pricing=1000,
        max_session_cost=0.01, ledger_path=str(db_path),
    )
    llm._async_client.chat.completions.create.return_value = FakeAsyncStream(
        make_stream_chunks(["Hello"]) + [make_usage_chunk(9, 10, 19)]
    )

    async def run():
        return await llm.ainvoke("hi")

    assert asyncio.run(run()) == "Hello"
    assert llm.session_cost_used >= 0.01
    assert llm.last_metadata["total_tokens"] == 19

    row = _read_ledger_rows(db_path)[0]
    assert row["call_type"] == "ainvoke"
    assert row["total_tokens"] == 19

    async def run_again():
        return await llm.ainvoke("hi again")

    with pytest.raises(BudgetExceededException):
        asyncio.run(run_again())


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


def test_system_prompt_not_duplicated_when_list_prompt_has_its_own():
    """
    Regression: prompt= can be a pre-built messages list. If the caller's own
    list already includes a system message, the constructor's system_prompt
    must not also be prepended -- it used to produce two system messages.
    """
    llm = make_llm(system_prompt="Default system prompt")
    llm._client.chat.completions.create.return_value = make_completion("ok")
    llm.invoke([
        {"role": "system", "content": "Caller specific system msg"},
        {"role": "user", "content": "hi"},
    ])
    messages = llm._client.chat.completions.create.call_args.kwargs["messages"]
    assert messages == [
        {"role": "system", "content": "Caller specific system msg"},
        {"role": "user", "content": "hi"},
    ]


def test_system_prompt_still_prepended_when_list_prompt_has_none():
    """No regression: a list prompt without its own system message still
    gets the constructor's system_prompt prepended, same as before."""
    llm = make_llm(system_prompt="Default system prompt")
    llm._client.chat.completions.create.return_value = make_completion("ok")
    llm.invoke([{"role": "user", "content": "hi"}])
    messages = llm._client.chat.completions.create.call_args.kwargs["messages"]
    assert messages == [
        {"role": "system", "content": "Default system prompt"},
        {"role": "user", "content": "hi"},
    ]


def test_system_prompt_not_duplicated_when_list_prompt_has_developer_role():
    """The newer "developer" role (system's documented replacement for
    reasoning models) also suppresses the constructor default."""
    llm = make_llm(system_prompt="Default system prompt")
    llm._client.chat.completions.create.return_value = make_completion("ok")
    llm.invoke([
        {"role": "developer", "content": "Caller developer msg"},
        {"role": "user", "content": "hi"},
    ])
    messages = llm._client.chat.completions.create.call_args.kwargs["messages"]
    assert messages == [
        {"role": "developer", "content": "Caller developer msg"},
        {"role": "user", "content": "hi"},
    ]


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


def test_invoke_with_tools_missing_name_raises_config_error_not_key_error():
    """
    Regression: a tool dict missing "name" used to raise a raw, unhelpful
    KeyError instead of this library's own OpenAIChatModelConfigError.
    """
    llm = make_llm()
    with pytest.raises(OpenAIChatModelConfigError, match="index 0"):
        llm.invoke_with_tools("hi", [{"description": "missing its name key"}])


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
    # the fallback-to-{} used to be silent -- callers had no way to tell a
    # tool call with genuinely no arguments apart from one whose JSON failed
    # to parse. arguments_parse_error now surfaces that distinction.
    assert resp.tool_calls[0].arguments_parse_error is not None


def test_invoke_with_tools_valid_arguments_json_has_no_parse_error():
    llm = make_llm()
    raw = make_completion(text=None, tool_calls=[make_tool_call("get_weather", {"city": "Tokyo"})])
    llm._client.chat.completions.create.return_value = raw
    resp = llm.invoke_with_tools("Weather?", TOOLS)
    assert resp.tool_calls[0].arguments == {"city": "Tokyo"}
    assert resp.tool_calls[0].arguments_parse_error is None


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


def test_close_method_releases_sync_client_without_with_statement():
    llm = make_llm()
    close_mock = llm._client.close
    llm.close()
    close_mock.assert_called_once()
    assert llm._client is None


def test_aclose_method_releases_both_clients_without_async_with():
    llm = make_llm()
    llm._async_client.aclose = AsyncMock()
    aclose_mock = llm._async_client.aclose
    asyncio.run(llm.aclose())
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


def test_ainvoke_from_multiple_concurrent_event_loops_does_not_deadlock():
    """
    Regression: the async circuit-breaker/budget-admission lock used to be a
    single asyncio.Lock shared for the instance's whole lifetime. asyncio.Lock
    is not thread-safe -- two threads each running their own event loop and
    concurrently `async with`-ing the *same* Lock object could deadlock
    (reproduced directly before the fix: 4 threads hammering one shared Lock
    hung indefinitely). Each event loop must get its own lock instead.

    Runs several threads, each with its own asyncio.run() loop, all calling
    ainvoke() concurrently on one shared instance, with a bounded join
    timeout so a regression fails the test instead of hanging the suite.
    """
    llm = make_llm(max_session_cost=1000.0, input_pricing=1.0, output_pricing=1.0)
    llm._async_client.chat.completions.create.return_value = make_completion("ok")

    errors: List[Exception] = []

    def worker():
        async def run_calls():
            for _ in range(10):
                await llm.ainvoke("hi")
        try:
            asyncio.run(run_calls())
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    assert not any(t.is_alive() for t in threads), "deadlocked -- a thread never finished"
    assert not errors, f"unexpected errors: {errors}"


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


def test_circuit_breaker_ignores_config_errors_as_non_transient():
    """
    Regression: a caller/config mistake (e.g. invoke_structured() with a
    non-Pydantic output_schema) must not trip the circuit breaker -- it's not
    a sign the provider is unhealthy, and previously counted as a failure,
    letting repeated config mistakes block unrelated, healthy invoke() calls
    on the same instance.
    """
    llm = make_llm(circuit_failure_threshold=1)
    with pytest.raises(OpenAIChatModelConfigError):
        llm.invoke_structured("give me a number")  # output_schema=None -> ConfigError
    assert llm._consecutive_failures == 0
    llm._client.chat.completions.create.return_value = make_completion("Paris")
    assert llm.invoke("hi") == "Paris"  # not blocked by a tripped circuit


def test_circuit_breaker_ignores_redaction_blocked_as_non_transient():
    """
    Regression: redact_mode="block" correctly refusing a PII-matching prompt
    is the redaction policy working as designed, not a provider failure --
    previously counted toward the circuit breaker, so a burst of legitimately
    blocked prompts could trip it and block unrelated, clean calls too.
    """
    llm = make_llm(redact_pii=True, redact_mode="block", circuit_failure_threshold=1)
    with pytest.raises(OpenAIChatModelRedactionBlockedError):
        llm.invoke("my email is bob@example.com")
    assert llm._consecutive_failures == 0
    llm._client.chat.completions.create.return_value = make_completion("ok")
    assert llm.invoke("hi") == "ok"  # not blocked by a tripped circuit


def test_stream_failures_trip_the_circuit_breaker():
    """
    Regression: stream()/astream() used to bypass the circuit breaker
    entirely -- a mid-stream failure never incremented consecutive_failures,
    so no number of failing stream() calls could ever trip it.
    """
    def failing_stream(**kwargs):
        raise ConnectionError("boom")
        yield  # pragma: no cover -- makes this a generator function

    llm = make_llm(circuit_failure_threshold=2, max_retries=1)
    llm._client.chat.completions.create.side_effect = lambda **kwargs: failing_stream(**kwargs)

    for _ in range(2):
        with pytest.raises(OpenAIChatModelAPIError):
            list(llm.stream("hi"))
    assert llm._consecutive_failures == 2

    # Circuit is now open -- must raise immediately, without even attempting
    # to iterate (and thus without ever reaching the client).
    llm._client.chat.completions.create.reset_mock(side_effect=True)
    with pytest.raises(CircuitBreakerOpenException):
        llm.stream("hi")
    llm._client.chat.completions.create.assert_not_called()


def test_stream_success_resets_the_circuit_breaker():
    llm = make_llm(circuit_failure_threshold=2, max_retries=1)
    llm._client.chat.completions.create.side_effect = lambda **kwargs: (_ for _ in ()).throw(ConnectionError("boom"))
    with pytest.raises(OpenAIChatModelAPIError):
        list(llm.stream("hi"))
    assert llm._consecutive_failures == 1

    llm._client.chat.completions.create.side_effect = None
    llm._client.chat.completions.create.return_value = FakeSyncStream(make_stream_chunks(["ok"]))
    assert list(llm.stream("hi")) == ["ok"]
    assert llm._consecutive_failures == 0


def test_circuit_breaker_already_open_blocks_stream_before_iteration():
    """A circuit tripped by invoke() failures must also block stream() calls,
    instead of letting them keep hitting a known-down provider."""
    llm = make_llm(circuit_failure_threshold=1, max_retries=1)
    llm._client.chat.completions.create.side_effect = ConnectionError("boom")
    with pytest.raises(OpenAIChatModelAPIError):
        llm.invoke("hi")

    llm._client.chat.completions.create.reset_mock(side_effect=True)
    with pytest.raises(CircuitBreakerOpenException):
        llm.stream("hi")  # raised by calling stream(), before any iteration
    llm._client.chat.completions.create.assert_not_called()


def test_astream_failures_trip_the_circuit_breaker():
    async def failing_astream(**kwargs):
        raise ConnectionError("boom")
        yield  # pragma: no cover -- makes this an async generator function

    llm = make_llm(circuit_failure_threshold=2, max_retries=1)
    llm._async_client.chat.completions.create.side_effect = lambda **kwargs: failing_astream(**kwargs)

    async def run():
        return [c async for c in llm.astream("hi")]

    for _ in range(2):
        with pytest.raises(OpenAIChatModelAPIError):
            asyncio.run(run())
    assert llm._consecutive_failures == 2

    llm._async_client.chat.completions.create.reset_mock(side_effect=True)

    async def run_after_trip():
        return [c async for c in llm.astream("hi")]

    with pytest.raises(CircuitBreakerOpenException):
        asyncio.run(run_after_trip())
    llm._async_client.chat.completions.create.assert_not_called()


# ── 16. Error handling ───────────────────────────────────────────────────────

def test_config_error_streaming_and_structured_output():
    with pytest.raises(OpenAIChatModelConfigError, match="incompatible with streaming"):
        make_llm(structured_output=True, streaming=True)


def test_config_error_max_retries_zero_or_negative():
    """
    Regression test: max_retries is used as range(1, max_retries + 1) in every
    retry loop, so max_retries=0 made the loop never run at all -- the
    non-stream path silently sent zero API calls while raising a misleading
    "Unexpected retry exhaustion" error, and the stream path crashed with
    TypeError ("exceptions must derive from BaseException") from `raise None`.
    Since max_retries is the *total* attempt count (not retries after the
    first), 0 attempts is nonsensical and is now rejected at construction.
    """
    with pytest.raises(OpenAIChatModelConfigError, match="max_retries must be >= 1"):
        make_llm(max_retries=0)
    with pytest.raises(OpenAIChatModelConfigError, match="max_retries must be >= 1"):
        make_llm(max_retries=-1)


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


def test_create_messages_override_cannot_hijack_validated_input_data():
    """
    Regression: a `messages=` kwarg passed alongside `input_data=` used to
    silently replace the already-validated input_data via the unfiltered
    **overrides merge -- the ValueError check above it became meaningless.
    """
    llm = make_llm()
    llm._client.chat.completions.create.return_value = make_completion("ok")
    llm.create(
        input_data=[{"role": "user", "content": "validated one"}],
        messages=[{"role": "user", "content": "SNUCK IN"}],
    )
    kwargs = llm._client.chat.completions.create.call_args.kwargs
    assert kwargs["messages"] == [{"role": "user", "content": "validated one"}]


def test_create_stream_override_is_ignored():
    """
    Regression: `stream=True` used to silently flip create() into streaming
    mode, returning a raw stream iterator that _create_raw()'s retry logic
    doesn't handle, instead of the documented non-streaming completion.
    """
    llm = make_llm()
    llm._client.chat.completions.create.return_value = make_completion("ok")
    llm.create(input_data=[{"role": "user", "content": "hi"}], stream=True)
    kwargs = llm._client.chat.completions.create.call_args.kwargs
    assert kwargs["stream"] is False


def test_create_model_override_still_works():
    """model= must stay overridable -- only messages/stream are protected."""
    llm = make_llm()
    llm._client.chat.completions.create.return_value = make_completion("ok")
    llm.create(input_data=[{"role": "user", "content": "hi"}], model="gpt-4o-mini")
    kwargs = llm._client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_acreate_messages_and_stream_overrides_are_ignored():
    llm = make_llm()
    llm._async_client.chat.completions.create.return_value = make_completion("ok")
    await llm.acreate(
        input_data=[{"role": "user", "content": "validated one"}],
        messages=[{"role": "user", "content": "SNUCK IN"}],
        stream=True,
    )
    kwargs = llm._async_client.chat.completions.create.call_args.kwargs
    assert kwargs["messages"] == [{"role": "user", "content": "validated one"}]
    assert kwargs["stream"] is False


# ── 18. Repr / misc ──────────────────────────────────────────────────────────

def test_repr_contains_model_and_flags():
    llm = make_llm(streaming=False)
    r = repr(llm)
    assert "gpt-4o" in r
    assert "streaming=False" in r


def test_normalize_model_name_strips_whitespace_and_preserves_case():
    """
    Regression: model names/deployment identifiers can be case-sensitive
    (Azure OpenAI deployment names, self-hosted/vLLM model tags) -- only
    surrounding whitespace should be stripped, never the case.
    """
    llm = OpenAIChatModel(model="  GPT4o-Prod-Deployment  ", api_key="sk-test")
    assert llm._model_name == "GPT4o-Prod-Deployment"


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
    """
    .jpg must map to the correct IANA media type image/jpeg, not the
    invalid image/jpg some providers reject outright -- see
    test_encode_file_mime_types below for full coverage of this mapping.
    """
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0fakejpegdata")
    llm = make_llm()
    llm._client.chat.completions.create.return_value = make_completion("A desk")
    llm.invoke("What is this?", files=[str(img)])
    messages = llm._client.chat.completions.create.call_args.kwargs["messages"]
    url = messages[0]["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")


@pytest.mark.parametrize("ext,expected_mime", [
    ("jpg", "image/jpeg"),
    ("jpeg", "image/jpeg"),
    ("png", "image/png"),
    ("gif", "image/gif"),
    ("webp", "image/webp"),
])
def test_encode_file_mime_types(tmp_path, ext, expected_mime):
    """Every OpenAI-vision-supported extension must map to its correct IANA media type."""
    img = tmp_path / f"photo.{ext}"
    img.write_bytes(b"fakedata")
    llm = make_llm()
    llm._client.chat.completions.create.return_value = make_completion("ok")
    llm.invoke("describe", files=[str(img)])
    url = llm._client.chat.completions.create.call_args.kwargs["messages"][0]["content"][1]["image_url"]["url"]
    assert url.startswith(f"data:{expected_mime};base64,")


def test_encode_file_unsupported_extension_warns_but_still_sends(tmp_path, caplog):
    """
    Regression test: an unrecognized extension (e.g. a PDF passed to
    files=, which this module has no dedicated support for -- vision-only)
    used to silently become a fabricated image/<ext> MIME type with zero
    warning. It must still be sent (not silently dropped), but now with a
    clear local warning naming the actual problem.
    """
    doc = tmp_path / "report.pdf"
    doc.write_bytes(b"%PDF-1.4 fake")
    llm = make_llm()
    llm._client.chat.completions.create.return_value = make_completion("ok")
    with caplog.at_level("WARNING"):
        llm.invoke("describe", files=[str(doc)])
    url = llm._client.chat.completions.create.call_args.kwargs["messages"][0]["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/pdf;base64,")  # still sent -- best-effort, not silently dropped
    assert any("isn't a recognized image" in r.message for r in caplog.records)


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


def test_structured_output_strict_schema_requires_every_property():
    """
    Regression test: OpenAI/Azure strict json_schema mode also requires every
    key in `properties` to appear in `required` -- but Pydantic's
    model_json_schema() only lists fields without a default, omitting
    Optional/defaulted fields from `required`. A model with any such field
    used to still 400 even after additionalProperties was fixed. Covers both
    the top-level schema and a nested $defs model.
    """
    class Address(BaseModel):
        city: str
        zip_code: Optional[str] = None

    class Person(BaseModel):
        name: str
        nickname: str = "N/A"
        address: Optional[Address] = None

    llm = make_llm(output_schema=Person, structured_output=True)
    llm._client.chat.completions.create.return_value = make_completion('{"name": "x"}')
    llm.invoke("go")
    schema = llm._client.chat.completions.create.call_args.kwargs["response_format"]["json_schema"]["schema"]

    assert set(schema["required"]) == set(schema["properties"].keys())
    assert "nickname" in schema["required"]  # has a default, but must still be required
    assert "address" in schema["required"]   # Optional, but must still be required

    nested = schema["$defs"]["Address"]
    assert set(nested["required"]) == set(nested["properties"].keys())
    assert "zip_code" in nested["required"]


def test_enforce_additional_properties_false_does_not_mutate_input():
    """
    Regression: enforce_additional_properties_false() used to mutate its
    input dict (and nested dicts reachable from it) in place. A caller
    passing a raw schema dict who still holds a reference to a nested object
    node (e.g. a property that is itself an object) used to see that node
    silently gain additionalProperties/required as a side effect, even
    though they never passed it to this function directly.
    """
    from autourgos_openaichat.core import enforce_additional_properties_false

    nested_object_node = {"type": "object", "properties": {"city": {"type": "string"}}}
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "address": nested_object_node},
    }
    original_nested_snapshot = {"type": "object", "properties": {"city": {"type": "string"}}}

    result = enforce_additional_properties_false(schema)

    # The caller's own nested object must be completely untouched.
    assert nested_object_node == original_nested_snapshot
    assert "additionalProperties" not in nested_object_node
    assert "required" not in nested_object_node
    assert schema["properties"]["address"] is nested_object_node

    # The returned schema must have the strict-mode fields applied.
    assert result["additionalProperties"] is False
    assert result["properties"]["address"]["additionalProperties"] is False
    assert result["properties"]["address"]["required"] == ["city"]


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


def test_multimodal_path_read_failure_for_non_url_string_logs_a_warning(caplog):
    """A typo'd local path used to silently become a bogus 'URL' sent to the
    API with no signal anything was wrong. Behavior (still sending it) is
    unchanged, but it must now be diagnosable via a warning log."""
    import logging
    llm = make_llm()
    llm._client.chat.completions.create.return_value = make_completion("ok")
    with caplog.at_level(logging.WARNING, logger="autourgos_openaichat"):
        llm.invoke("desc", files=["not-a-real-path-on-disk.jpg"])
    assert any("not-a-real-path-on-disk.jpg" in r.message for r in caplog.records)


def test_multimodal_real_url_string_read_failure_does_not_warn(caplog):
    """A genuine http(s)/data URL failing to open() as a file is expected --
    no warning needed since this is exactly the documented URL path."""
    import logging
    llm = make_llm()
    llm._client.chat.completions.create.return_value = make_completion("ok")
    with caplog.at_level(logging.WARNING, logger="autourgos_openaichat"):
        llm.invoke("desc", files=["https://example.com/chart.png"])
    assert not any("could not be opened" in r.message for r in caplog.records)


# ── 18. Provider fallback chain ─────────────────────────────────────────────

from autourgos_openaichat import OpenAIChatModelAllProvidersFailedError


def make_llm_with_fallback(fallback_models=("backup-model",), fallback_pricing=None, **kwargs):
    """Construct an OpenAIChatModel with a mocked primary and mocked fallback clients."""
    fallback_pricing = fallback_pricing or {}
    llm = OpenAIChatModel(
        model="gpt-4o",
        api_key="sk-test",
        fallback_providers=[
            {"model": m, "api_key": "sk-backup", **fallback_pricing} for m in fallback_models
        ],
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


def test_fallback_metadata_reports_its_own_model_not_the_primarys():
    """Regression: llm.last_metadata['model'] must reflect whichever provider
    actually answered, not always the primary's model name."""
    llm = make_llm_with_fallback(max_retries=1)
    llm._client.chat.completions.create.side_effect = RuntimeError("primary down")
    llm._fallback_sync_clients[0].chat.completions.create.return_value = make_completion("Berlin")
    llm.invoke("hi")
    assert llm.last_metadata["model"] == "backup-model"


def test_fallback_cost_uses_its_own_pricing_not_the_primarys():
    """Regression: cost for a fallback response must come from that fallback
    entry's own pricing, not the primary's (different model, different price)."""
    llm = make_llm_with_fallback(
        max_retries=1, input_pricing=1000, output_pricing=1000,
        fallback_pricing={"input_pricing": 1.0, "output_pricing": 2.0},
    )
    llm._client.chat.completions.create.side_effect = RuntimeError("primary down")
    llm._fallback_sync_clients[0].chat.completions.create.return_value = make_completion(
        "Berlin", usage=(10, 5, 15)
    )
    llm.invoke("hi")
    expected = (10 / 1_000_000) * 1.0 + (5 / 1_000_000) * 2.0
    assert llm.last_metadata["total_cost"] == pytest.approx(expected)


def test_fallback_without_its_own_pricing_omits_cost():
    """A fallback entry with no pricing of its own must not silently borrow
    the primary's price for a different model — cost fields stay unset."""
    llm = make_llm_with_fallback(max_retries=1, input_pricing=1000, output_pricing=1000)
    llm._client.chat.completions.create.side_effect = RuntimeError("primary down")
    llm._fallback_sync_clients[0].chat.completions.create.return_value = make_completion("Berlin")
    llm.invoke("hi")
    assert "total_cost" not in llm.last_metadata
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


# ── 18b. Aggregate call deadline (max_call_duration) ─────────────────────────

from autourgos_openaichat import OpenAIChatModelDeadlineExceededError


def test_max_call_duration_none_by_default_no_cap():
    llm = make_llm()
    assert llm.max_call_duration is None
    llm._client.chat.completions.create.return_value = make_completion("Paris")
    assert llm.invoke("hi") == "Paris"


def test_max_call_duration_exceeded_blocks_before_any_attempt():
    llm = make_llm(max_call_duration=0.0)
    with pytest.raises(OpenAIChatModelDeadlineExceededError):
        llm.invoke("hi")
    llm._client.chat.completions.create.assert_not_called()


def test_max_call_duration_stops_before_trying_fallback():
    """
    Regression: without an aggregate deadline, retries and fallback each get
    their own full retry budget independently. With max_call_duration set,
    once it's exceeded partway through the primary's retries, the fallback
    provider must never even be tried -- the call fails fast with
    OpenAIChatModelDeadlineExceededError instead of quietly moving on to
    burn through the fallback's retry budget too.
    """
    llm = make_llm_with_fallback(max_retries=3, backoff_factor=1.0, max_call_duration=0.05)
    llm._client.chat.completions.create.side_effect = RuntimeError("primary flaky")

    with pytest.raises(OpenAIChatModelDeadlineExceededError):
        llm.invoke("hi")

    # Primary was attempted at least once before the ~1s backoff sleep pushed
    # elapsed time past the 0.05s deadline on the next retry-loop check.
    assert llm._client.chat.completions.create.call_count >= 1
    llm._fallback_sync_clients[0].chat.completions.create.assert_not_called()


def test_max_call_duration_applies_to_stream():
    llm = make_llm(max_call_duration=0.0)
    with pytest.raises(OpenAIChatModelDeadlineExceededError):
        list(llm.stream("hi"))
    llm._client.chat.completions.create.assert_not_called()


def test_max_call_duration_applies_to_ainvoke():
    llm = make_llm(max_retries=3, backoff_factor=1.0, max_call_duration=0.05)
    llm._async_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))

    async def run():
        return await llm.ainvoke("hi")

    with pytest.raises(OpenAIChatModelDeadlineExceededError):
        asyncio.run(run())


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


def test_budget_admission_serializes_concurrent_invocations():
    """
    Regression: concurrent invoke() calls sharing a capped instance must not
    all pass the budget check before any of them records cost. Without
    admission control, N threads racing a cap that a single call's cost
    alone exceeds could all slip through before the first one's cost is
    recorded. With admission control serializing them, only the first one
    admitted succeeds -- every other thread must observe the now-exceeded
    budget and raise.
    """
    llm = make_llm(input_pricing=1_000_000.0, output_pricing=1_000_000.0, max_session_cost=1.0)

    def slow_create(*args, **kwargs):
        time.sleep(0.05)  # simulate real API latency, widening the race window
        return make_completion("ok")  # usage (9, 10, 19) -> cost $19, well over the $1 cap

    llm._client.chat.completions.create.side_effect = slow_create

    results: List[str] = []
    results_lock = threading.Lock()

    def worker() -> None:
        try:
            llm.invoke("hi")
            outcome = "ok"
        except BudgetExceededException:
            outcome = "blocked"
        with results_lock:
            results.append(outcome)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count("ok") == 1
    assert results.count("blocked") == 4


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


def test_redaction_applies_to_invoke_with_tools():
    """
    Regression: invoke_with_tools()/ainvoke_with_tools() used to build
    messages directly from the raw prompt, bypassing _resolve_prompt()'s
    redaction step entirely — redact_pii=True had no effect on tool-calling
    requests. They must go through the same redaction pipeline as invoke().
    """
    llm = make_llm(redact_pii=True)
    llm._client.chat.completions.create.return_value = make_completion("ok")
    llm.invoke_with_tools(
        "contact me at bob@example.com", [{"name": "noop", "parameters": {}}]
    )
    sent = llm._client.chat.completions.create.call_args.kwargs["messages"]
    text = sent[0]["content"]
    assert "bob@example.com" not in text
    assert "[REDACTED:email]" in text
    assert llm.last_redacted_categories == ["email"]


def test_redaction_applies_to_ainvoke_with_tools():
    llm = make_llm(redact_pii=True)
    llm._async_client.chat.completions.create = AsyncMock(return_value=make_completion("ok"))

    async def run():
        return await llm.ainvoke_with_tools(
            "contact me at bob@example.com", [{"name": "noop", "parameters": {}}]
        )

    asyncio.run(run())
    sent = llm._async_client.chat.completions.create.call_args.kwargs["messages"]
    text = sent[0]["content"]
    assert "bob@example.com" not in text
    assert "[REDACTED:email]" in text


def test_redact_mode_block_raises_before_api_call_for_invoke_with_tools():
    llm = make_llm(redact_pii=True, redact_mode="block")
    with pytest.raises(OpenAIChatModelRedactionBlockedError):
        llm.invoke_with_tools("my email is bob@example.com", [{"name": "noop", "parameters": {}}])
    llm._client.chat.completions.create.assert_not_called()


# ── Sprint 0 regression: invoke_with_tools()/ainvoke_with_tools() used to
# bypass budget/ledger/redaction-restore entirely (native tool-calling mode
# called _create_across_providers() directly, skipping the same machinery
# invoke() goes through). ────────────────────────────────────────────────────

def test_budget_governor_blocks_invoke_with_tools_once_cap_reached():
    llm = make_llm(input_pricing=1000, output_pricing=1000, max_session_cost=0.01)
    llm._client.chat.completions.create.return_value = make_completion("ok")  # usage (9,10,19) -> $0.019
    llm.invoke_with_tools("hi", TOOLS)  # 1st call goes through, pushes usage over the cap
    assert llm.session_cost_used >= 0.01

    with pytest.raises(BudgetExceededException):
        llm.invoke_with_tools("hi again", TOOLS)
    assert llm._client.chat.completions.create.call_count == 1  # 2nd call never reached the API


def test_budget_governor_blocks_ainvoke_with_tools_once_cap_reached():
    llm = make_llm(input_pricing=1000, output_pricing=1000, max_session_cost=0.01)
    llm._async_client.chat.completions.create = AsyncMock(return_value=make_completion("ok"))

    async def run():
        await llm.ainvoke_with_tools("hi", TOOLS)
        with pytest.raises(BudgetExceededException):
            await llm.ainvoke_with_tools("hi again", TOOLS)

    asyncio.run(run())
    assert llm._async_client.chat.completions.create.call_count == 1


def test_ledger_records_invoke_with_tools(tmp_path):
    db_path = tmp_path / "calls.db"
    llm = make_llm(ledger_path=str(db_path))
    llm._client.chat.completions.create.return_value = make_completion("Paris")
    llm.invoke_with_tools("Capital of France?", TOOLS)

    rows = _read_ledger_rows(db_path)
    assert len(rows) == 1
    assert rows[0]["call_type"] == "invoke_with_tools"
    assert rows[0]["response"] == "Paris"


def test_ledger_records_ainvoke_with_tools(tmp_path):
    db_path = tmp_path / "calls.db"
    llm = make_llm(ledger_path=str(db_path))
    llm._async_client.chat.completions.create = AsyncMock(return_value=make_completion("Berlin"))

    async def run():
        return await llm.ainvoke_with_tools("Capital of Germany?", TOOLS)

    asyncio.run(run())
    rows = _read_ledger_rows(db_path)
    assert len(rows) == 1
    assert rows[0]["call_type"] == "ainvoke_with_tools"
    assert rows[0]["response"] == "Berlin"


def test_ledger_does_not_record_tool_calls_response_text(tmp_path):
    """When the model calls a tool (no final-answer text), the ledger's
    response column should reflect that (None), not crash on it."""
    db_path = tmp_path / "calls.db"
    llm = make_llm(ledger_path=str(db_path))
    raw = make_completion(text=None, tool_calls=[make_tool_call("get_weather", {"city": "Tokyo"})])
    llm._client.chat.completions.create.return_value = raw
    llm.invoke_with_tools("Weather in Tokyo?", TOOLS)

    rows = _read_ledger_rows(db_path)
    assert len(rows) == 1
    assert rows[0]["call_type"] == "invoke_with_tools"
    assert rows[0]["response"] is None


def test_redact_restore_in_response_applies_to_invoke_with_tools():
    """
    Regression: invoke_with_tools() discarded its redaction_map entirely, so
    redact_restore_in_response=True (working correctly in invoke()) silently
    had no effect in native tool-calling mode -- the caller got back the
    masked placeholder instead of the original text.
    """
    llm = make_llm(redact_pii=True, redact_mode="mask", redact_restore_in_response=True)
    llm._client.chat.completions.create.return_value = make_completion(
        "Sure, I'll email bob@example.com"
    )
    resp = llm.invoke_with_tools("contact bob@example.com please", TOOLS)
    assert resp.text == "Sure, I'll email bob@example.com"


def test_redact_restore_in_response_applies_to_ainvoke_with_tools():
    llm = make_llm(redact_pii=True, redact_mode="mask", redact_restore_in_response=True)
    llm._async_client.chat.completions.create = AsyncMock(
        return_value=make_completion("Sure, I'll email bob@example.com")
    )

    async def run():
        return await llm.ainvoke_with_tools("contact bob@example.com please", TOOLS)

    resp = asyncio.run(run())
    assert resp.text == "Sure, I'll email bob@example.com"


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


def test_invalid_custom_regex_raises_config_error_not_raw_re_error():
    """
    Regression: re.error is not a ValueError subclass, so a malformed
    redact_custom_patterns regex used to bypass the constructor's
    `except ValueError` guard entirely and leak out as a raw re.error
    instead of the library's own OpenAIChatModelConfigError.
    """
    with pytest.raises(OpenAIChatModelConfigError):
        OpenAIChatModel(
            model="gpt-4o", api_key="sk-test", redact_pii=True,
            redact_custom_patterns={"bad": "(unbalanced"},
        )


def test_ledger_records_redacted_categories(tmp_path):
    db_path = tmp_path / "calls.db"
    llm = make_llm(ledger_path=str(db_path), redact_pii=True)
    llm._client.chat.completions.create.return_value = make_completion("ok")
    llm.invoke("my email is bob@example.com")
    row = _read_ledger_rows(db_path)[0]
    assert row["redacted_categories"] == "email"


def test_ledger_records_redacted_categories_correctly_under_concurrency(tmp_path):
    """
    Regression test: _log_to_ledger() used to read self.last_redacted_categories,
    a shared instance attribute -- a concurrent call whose prompt matched a
    *different* category could overwrite it before this call's own ledger
    write happened, mislabeling the audit trail. Now each call's ledger row
    always carries its own call-local categories.
    """
    db_path = tmp_path / "calls.db"
    llm = make_llm(ledger_path=str(db_path), redact_pii=True)

    async def create(**kwargs):
        content = kwargs["messages"][0]["content"]
        # The "ssn" call resolves *last*, well after the "email" call --
        # if categories were shared, the email call's ledger row could end
        # up mislabeled with "ssn" (or vice versa).
        await asyncio.sleep(0.05 if "ssn" in content else 0.01)
        return make_completion("ok")

    llm._async_client.chat.completions.create.side_effect = create

    async def run():
        await asyncio.gather(
            llm.ainvoke("my ssn is 123-45-6789"),
            llm.ainvoke("my email is bob@example.com"),
        )

    asyncio.run(run())
    rows = _read_ledger_rows(db_path)
    ssn_row = next(r for r in rows if "REDACTED:ssn" in r["prompt"])
    email_row = next(r for r in rows if "REDACTED:email" in r["prompt"])
    assert ssn_row["redacted_categories"] == "ssn"
    assert email_row["redacted_categories"] == "email"


# ── 23. Shadow-mode dual dispatch ────────────────────────────────────────────

def make_llm_with_shadow(shadow_models=("backup-model",), shadow_pricing=None, **kwargs):
    """Construct an OpenAIChatModel with a mocked primary and mocked shadow clients."""
    shadow_pricing = shadow_pricing or {}
    llm = OpenAIChatModel(
        model="gpt-4o",
        api_key="sk-test",
        shadow_providers=[
            {"model": m, "api_key": "sk-shadow", **shadow_pricing} for m in shadow_models
        ],
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
    # Primary has pricing configured; the shadow entry deliberately does not,
    # to prove cost isn't silently computed with the primary's (wrong) price
    # for a different model — see test_shadow_cost_uses_its_own_pricing below
    # for the case where the shadow entry does set its own pricing.
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
    assert shadow["total_cost"] is None  # shadow entry has no pricing of its own
    sent_model = llm._shadow_sync_clients[0].chat.completions.create.call_args.kwargs["model"]
    assert sent_model == "backup-model"


def test_shadow_cost_uses_its_own_pricing_not_the_primarys():
    """A shadow provider's cost must come from its own pricing, not the primary's."""
    llm = make_llm_with_shadow(
        input_pricing=1000, output_pricing=1000,
        shadow_pricing={"input_pricing": 1.0, "output_pricing": 2.0},
    )
    llm._client.chat.completions.create.return_value = make_completion("Paris")
    llm._shadow_sync_clients[0].chat.completions.create.return_value = make_completion(
        "Paris", usage=(10, 5, 15)
    )

    llm.invoke("What is the capital of France?")
    shadow = llm.last_shadow_results[0]
    expected = (10 / 1_000_000) * 1.0 + (5 / 1_000_000) * 2.0
    assert shadow["total_cost"] == pytest.approx(expected)


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


def test_shadow_dispatch_runs_concurrently_with_primary():
    """
    Regression test: shadow dispatch used to only start after the primary
    call's entire budget-admission block (including its own network
    round-trip) had already finished -- sequential, not concurrent, despite
    the docstring/README claiming total latency is roughly
    max(primary, slowest shadow) rather than their sum. Both a slow primary
    and a slow shadow call here should now overlap.
    """
    llm = make_llm_with_shadow()
    delay = 0.2

    def slow_primary(**kw):
        time.sleep(delay)
        return make_completion("Paris")

    def slow_shadow(**kw):
        time.sleep(delay)
        return make_completion("Paris")

    llm._client.chat.completions.create.side_effect = slow_primary
    llm._shadow_sync_clients[0].chat.completions.create.side_effect = slow_shadow

    start = time.perf_counter()
    result = llm.invoke("hi")
    elapsed = time.perf_counter() - start

    assert result == "Paris"
    assert len(llm.last_shadow_results) == 1
    # Sequential would take ~2*delay; concurrent should stay well under that.
    assert elapsed < delay * 1.8, f"expected concurrent dispatch (~{delay}s), took {elapsed}s"


def test_shadow_still_fires_and_is_logged_when_primary_fails():
    """
    Regression test: since shadow dispatch now starts before the primary's
    own request (to achieve real concurrency), an in-flight shadow request
    can no longer be silently skipped if the primary ends up failing -- it's
    already been sent and can't be un-billed. Its result must still be
    collected and logged (with similarity=None, since there's no primary
    text to compare against), and the primary's own exception must still
    propagate correctly to the caller.
    """
    llm = make_llm_with_shadow(max_retries=1)
    llm._client.chat.completions.create.side_effect = RuntimeError("primary down")
    llm._shadow_sync_clients[0].chat.completions.create.return_value = make_completion("shadow-answer")

    with pytest.raises(OpenAIChatModelAPIError):
        llm.invoke("hi")

    assert len(llm.last_shadow_results) == 1
    shadow = llm.last_shadow_results[0]
    assert shadow["response"] == "shadow-answer"
    assert shadow["similarity"] is None
    assert shadow["error"] is None


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


def test_ainvoke_shadow_still_fires_and_is_logged_when_primary_fails():
    """Async counterpart of test_shadow_still_fires_and_is_logged_when_primary_fails."""
    llm = make_llm_with_shadow(max_retries=1)
    llm._async_client.chat.completions.create.side_effect = RuntimeError("primary down")
    llm._shadow_async_clients[0].chat.completions.create.return_value = make_completion("shadow-answer")

    async def run():
        return await llm.ainvoke("hi")

    with pytest.raises(OpenAIChatModelAPIError):
        asyncio.run(run())

    assert len(llm.last_shadow_results) == 1
    shadow = llm.last_shadow_results[0]
    assert shadow["response"] == "shadow-answer"
    assert shadow["similarity"] is None


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


def test_redact_restore_concurrent_ainvoke_never_cross_contaminates():
    """
    Regression test: _apply_redaction()/_resolve_prompt() used to write the
    redaction map to shared instance attributes (self._last_redaction_map),
    so two concurrent ainvoke() calls sharing one instance could restore
    using each other's mapping -- e.g. user A's response getting user B's
    secret spliced in if B's call happened to finish first. The fix makes
    the redaction map call-local (threaded through the return value), so
    each call must always restore its own secret regardless of completion
    order.
    """
    llm = make_llm(redact_pii=True, redact_categories=["email"], redact_restore_in_response=True)

    async def create(**kwargs):
        content = kwargs["messages"][0]["content"]
        # Whichever call is for "userA" resolves *last*, well after "userB" --
        # if redaction state were shared, userA's restore would race against
        # userB's already-overwritten mapping.
        await asyncio.sleep(0.05 if "userA" in content else 0.01)
        placeholder = "[REDACTED:email:1]"
        return make_completion(f"Confirming contact: {placeholder}")

    llm._async_client.chat.completions.create.side_effect = create

    async def run():
        return await asyncio.gather(
            llm.ainvoke("userA email is alice-private@corp.internal"),
            llm.ainvoke("userB email is bob-private@corp.internal"),
        )

    results = asyncio.run(run())
    assert results[0] == "Confirming contact: alice-private@corp.internal"
    assert results[1] == "Confirming contact: bob-private@corp.internal"
    assert "bob-private@corp.internal" not in results[0]
    assert "alice-private@corp.internal" not in results[1]


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


# ── extract_text_from_response fallback key-scan ─────────────────────────────

def test_extract_text_from_response_fallback_key_scan_logs_a_warning(caplog):
    """The fallback key-scan (used only when choices/output/output_text all
    miss) used to silently return a top-level 'text'/'delta'/'content' field
    with no signal the response didn't match a known shape. It must now log
    a warning when this path is taken."""
    import logging
    from autourgos_openaichat.model_runtime import extract_text_from_response

    with caplog.at_level(logging.WARNING, logger="autourgos_openaichat"):
        result = extract_text_from_response({"text": "fallback text"})

    assert result == "fallback text"
    assert any("falling back" in r.message.lower() for r in caplog.records)


def test_extract_text_from_response_known_shape_does_not_warn(caplog):
    import logging
    from autourgos_openaichat.model_runtime import extract_text_from_response

    with caplog.at_level(logging.WARNING, logger="autourgos_openaichat"):
        result = extract_text_from_response(
            {"choices": [{"message": {"content": "hello"}}]}
        )

    assert result == "hello"
    assert not any("falling back" in r.message for r in caplog.records)
