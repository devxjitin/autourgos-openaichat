"""
OpenAIChatModel — LLM wrapper for the OpenAI Chat Completions API.

Self-contained: no autourgos-core dependency.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from .llm import BaseLLM, FunctionCall, ToolCallResponse
from .model_runtime import (
    build_structured_output,
    coerce_prompt_variable,
    configure_runtime_environment,
    extract_template_fields,
    extract_text_from_response,
    track_latency,
)
from .core import (
    build_chat_completion_create_params,
    build_multimodal_messages,
    build_response_format,
    configure_async_openai_client,
    configure_openai_client,
    extract_text_delta_from_event,
    load_openai_module,
    logger,
    normalize_model_name,
    release_async_openai_client,
    release_openai_client,
    resolve_api_key,
    resolve_base_url,
)

configure_runtime_environment()
_OPENAI_AVAILABLE, openai_cls, async_openai_cls, _OPENAI_IMPORT_ERROR = load_openai_module()

# Client errors that will never succeed on retry — fail fast instead of
# burning the retry budget and adding latency.
_NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404, 422}


# ── Custom exceptions ─────────────────────────────────────────────────────────

class OpenAIChatModelError(Exception):
    """Base exception for OpenAIChatModel errors."""


class OpenAIChatModelImportError(OpenAIChatModelError):
    """Raised when the openai SDK cannot be imported."""


class OpenAIChatModelAPIError(OpenAIChatModelError):
    """Raised when an API request fails after all retries."""


class OpenAIChatModelResponseError(OpenAIChatModelError):
    """Raised when the API response cannot be interpreted."""


class OpenAIChatModelConfigError(OpenAIChatModelError):
    """Raised for incompatible configuration options."""


# ── Main class ────────────────────────────────────────────────────────────────

class OpenAIChatModel(BaseLLM):
    """
    LLM wrapper for the OpenAI Chat Completions API.

    Supports text generation, multi-modal input (images), streaming,
    structured output, native function-calling, and automatic retries
    with exponential back-off.

    Example::

        from autourgos_openaichat import OpenAIChatModel

        llm = OpenAIChatModel(model="gpt-4o", api_key="sk-...")
        reply = llm.invoke("What is the capital of France?")
        print(reply)  # "Paris"

    Async example::

        reply = await llm.ainvoke("Translate 'hello' to Spanish.")

    Streaming example::

        for chunk in llm.stream("Tell me a joke."):
            print(chunk, end="", flush=True)

    Native tool-calling example::

        from autourgos_openaichat import OpenAIChatModel

        llm = OpenAIChatModel(model="gpt-4o")
        tools = [{"name": "get_weather", "description": "...", "parameters": {...}}]
        response = llm.invoke_with_tools("What's the weather in Paris?", tools)
        if response.has_tool_calls:
            print(response.tool_calls)
    """

    supports_tool_calling: bool = True

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        *,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        system_prompt: Optional[str] = None,
        prompt_template: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        output_schema: Any = None,
        response_mime_type: Optional[str] = None,
        structured_output: bool = False,
        streaming: bool = False,
        max_retries: int = 3,
        timeout: Optional[float] = 60.0,
        backoff_factor: float = 0.5,
        input_pricing: Optional[float] = None,
        output_pricing: Optional[float] = None,
        circuit_failure_threshold: int = 5,
        circuit_cooldown_time: float = 30.0,
    ) -> None:
        """
        Args:
            model: OpenAI model name, e.g. "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo".
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            base_url: Override the API base URL (e.g. for proxies or local servers).
            organization: OpenAI organization ID.
            project: OpenAI project ID.
            system_prompt: System prompt prepended to every request.
            prompt_template: Optional template string with {variable} placeholders.
                When set, invoke() accepts prompt_variables= instead of a direct prompt.
            temperature: Sampling temperature (0–2). Higher = more random.
            top_p: Nucleus sampling probability (0–1).
            max_tokens: Maximum tokens to generate.
            output_schema: Pydantic model or dict for structured/JSON output.
            response_mime_type: e.g. "application/json" to enable json_object mode.
            structured_output: If True, invoke() returns a metadata dict instead of a string.
            streaming: If True, invoke()/ainvoke() internally stream and return the full text.
            max_retries: Number of retry attempts on transient API errors.
            timeout: Request timeout in seconds.
            backoff_factor: Base multiplier for exponential back-off between retries.
            input_pricing: USD per 1 million input tokens (for cost tracking).
            output_pricing: USD per 1 million output tokens (for cost tracking).
            circuit_failure_threshold: Consecutive failures before the circuit opens.
            circuit_cooldown_time: Seconds the circuit stays open before a probe attempt.
        """
        super().__init__(
            input_pricing=input_pricing,
            output_pricing=output_pricing,
            circuit_failure_threshold=circuit_failure_threshold,
            circuit_cooldown_time=circuit_cooldown_time,
        )
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.organization = organization
        self.project = project
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.output_schema = output_schema
        self.response_mime_type = response_mime_type
        self.structured_output = structured_output
        self.streaming = streaming
        self.max_retries = max_retries
        self.timeout = timeout
        self.backoff_factor = backoff_factor

        if self.structured_output and self.streaming:
            raise OpenAIChatModelConfigError(
                "structured_output=True is incompatible with streaming=True."
            )

        self._model_name = normalize_model_name(self.model)
        self._client: Any = None
        self._async_client: Any = None
        self._init_clients()

    # ── Client init ───────────────────────────────────────────────────────────

    def _init_clients(self) -> None:
        if not _OPENAI_AVAILABLE or openai_cls is None or async_openai_cls is None:
            detail = f" Details: {_OPENAI_IMPORT_ERROR}" if _OPENAI_IMPORT_ERROR else ""
            raise OpenAIChatModelImportError(
                "Failed to import openai SDK. Install it with: pip install openai" + detail
            )
        key = resolve_api_key(self.api_key)
        url = resolve_base_url(self.base_url)
        self._client = configure_openai_client(
            openai_cls,
            api_key=key,
            base_url=url,
            organization=self.organization,
            project=self.project,
            timeout=self.timeout,
        )
        self._async_client = configure_async_openai_client(
            async_openai_cls,
            api_key=key,
            base_url=url,
            organization=self.organization,
            project=self.project,
            timeout=self.timeout,
        )

    # ── Context managers ──────────────────────────────────────────────────────

    def __enter__(self) -> "OpenAIChatModel":
        return self

    def __exit__(self, *args: Any) -> None:
        if self._client is not None:
            release_openai_client(self._client)
            self._client = None

    async def __aenter__(self) -> "OpenAIChatModel":
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._async_client is not None:
            await release_async_openai_client(self._async_client)
            self._async_client = None
        if self._client is not None:
            release_openai_client(self._client)
            self._client = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resolve_prompt(
        self,
        prompt: Any,
        prompt_variables: Optional[Dict[str, Any]],
        files: Optional[Any] = None,
    ) -> Any:
        """Resolve prompt or render from template."""
        if prompt is not None:
            if isinstance(prompt, str) and not prompt.strip():
                raise ValueError("prompt must be a non-empty string or list when provided")
            if isinstance(prompt, list) and not prompt:
                raise ValueError("prompt must be a non-empty list when provided")
            if not isinstance(prompt, (str, list)):
                raise ValueError("prompt must be a string or list")
            return prompt

        if self.prompt_template is None:
            raise ValueError("prompt is required when prompt_template is not configured")

        merged = dict(prompt_variables or {})
        required = extract_template_fields(self.prompt_template)
        missing = sorted(f for f in required if f not in merged or not str(merged[f]).strip())
        if missing:
            raise ValueError(f"Missing prompt template variables: {', '.join(missing)}")
        rendered = self.prompt_template.format(**{k: coerce_prompt_variable(v) for k, v in merged.items()})
        if not rendered.strip():
            raise ValueError("Rendered prompt template is empty")

        if files and isinstance(rendered, list):
            raise OpenAIChatModelConfigError(
                "Cannot combine files with a pre-formatted multi-modal list prompt."
            )
        return rendered

    def _build_messages(
        self,
        prompt: Any,
        *,
        files: Optional[Any] = None,
        image_detail: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Build the full messages list including optional system instruction."""
        messages: List[Dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if isinstance(prompt, list):
            messages.extend(prompt)
            if files:
                messages.extend(build_multimodal_messages("", files=files, image_detail=image_detail))
            return messages
        messages.extend(build_multimodal_messages(prompt, files=files, image_detail=image_detail))
        return messages

    def _build_base_params(self, *, messages: List[Dict[str, Any]], stream: bool) -> Dict[str, Any]:
        response_format = build_response_format(
            output_schema=self.output_schema,
            response_mime_type=self.response_mime_type,
        )
        return build_chat_completion_create_params(
            self._model_name,
            messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            response_format=response_format,
            stream=stream,
        )

    # ── Raw API calls ─────────────────────────────────────────────────────────

    def _create_raw(self, params: Dict[str, Any]) -> Any:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._client.chat.completions.create(**params)
            except Exception as exc:
                last_exc = exc
                status_code = getattr(exc, "status_code", None)
                if status_code in _NON_RETRYABLE_STATUS_CODES:
                    raise OpenAIChatModelAPIError(
                        f"Chat Completions request failed with non-retryable status "
                        f"{status_code}. Error: {type(exc).__name__}: {exc}"
                    ) from exc
                if attempt == self.max_retries:
                    raise OpenAIChatModelAPIError(
                        f"Chat Completions request failed after {self.max_retries} attempts. "
                        f"Last error: {type(exc).__name__}: {exc}"
                    ) from exc
                time.sleep(self.backoff_factor * (2 ** (attempt - 1)))
        raise OpenAIChatModelAPIError("Unexpected retry exhaustion") from last_exc

    async def _acreate_raw(self, params: Dict[str, Any]) -> Any:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return await self._async_client.chat.completions.create(**params)
            except Exception as exc:
                last_exc = exc
                status_code = getattr(exc, "status_code", None)
                if status_code in _NON_RETRYABLE_STATUS_CODES:
                    raise OpenAIChatModelAPIError(
                        f"Async Chat Completions request failed with non-retryable status "
                        f"{status_code}. Error: {type(exc).__name__}: {exc}"
                    ) from exc
                if attempt == self.max_retries:
                    raise OpenAIChatModelAPIError(
                        f"Async Chat Completions request failed after {self.max_retries} attempts. "
                        f"Last error: {type(exc).__name__}: {exc}"
                    ) from exc
                await asyncio.sleep(self.backoff_factor * (2 ** (attempt - 1)))
        raise OpenAIChatModelAPIError("Unexpected async retry exhaustion") from last_exc

    # ── Non-stream invocation ─────────────────────────────────────────────────

    def _invoke_non_stream(self, *, messages: List[Dict[str, Any]]) -> Any:
        params = self._build_base_params(messages=messages, stream=False)
        resp = self._create_raw(params)
        text = extract_text_from_response(resp)
        if text:
            return text, resp
        raise OpenAIChatModelResponseError(
            "No text could be extracted from the Chat Completions response."
        )

    async def _ainvoke_non_stream(self, *, messages: List[Dict[str, Any]]) -> Any:
        params = self._build_base_params(messages=messages, stream=False)
        resp = await self._acreate_raw(params)
        text = extract_text_from_response(resp)
        if text:
            return text, resp
        raise OpenAIChatModelResponseError(
            "No text could be extracted from the async Chat Completions response."
        )

    # ── Streaming ─────────────────────────────────────────────────────────────

    def _invoke_stream_mode(self, *, messages: List[Dict[str, Any]]) -> Iterator[str]:
        params = self._build_base_params(messages=messages, stream=True)
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            emitted = False
            try:
                stream = self._client.chat.completions.create(**params)
                for event in stream:
                    delta = extract_text_delta_from_event(event)
                    if delta:
                        emitted = True
                        yield delta
                if emitted:
                    return
                raise OpenAIChatModelResponseError("No text deltas in streaming response")
            except (OpenAIChatModelResponseError, OpenAIChatModelAPIError):
                raise
            except Exception as exc:
                last_exc = exc
                status_code = getattr(exc, "status_code", None)
                if status_code in _NON_RETRYABLE_STATUS_CODES:
                    raise OpenAIChatModelAPIError(
                        f"Streaming failed with non-retryable status {status_code}. "
                        f"Error: {type(exc).__name__}: {exc}"
                    ) from exc
                if emitted or attempt == self.max_retries:
                    raise OpenAIChatModelAPIError(
                        f"Streaming failed after {attempt} attempt(s). "
                        f"Last error: {type(exc).__name__}: {exc}"
                    ) from exc
                time.sleep(self.backoff_factor * (2 ** (attempt - 1)))
        raise OpenAIChatModelAPIError("Streaming failed unexpectedly") from last_exc

    async def _ainvoke_stream_mode(self, *, messages: List[Dict[str, Any]]) -> AsyncIterator[str]:
        params = self._build_base_params(messages=messages, stream=True)
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            emitted = False
            try:
                stream = await self._async_client.chat.completions.create(**params)
                async for event in stream:
                    delta = extract_text_delta_from_event(event)
                    if delta:
                        emitted = True
                        yield delta
                if emitted:
                    return
                raise OpenAIChatModelResponseError("No text deltas in async streaming response")
            except (OpenAIChatModelResponseError, OpenAIChatModelAPIError):
                raise
            except Exception as exc:
                last_exc = exc
                status_code = getattr(exc, "status_code", None)
                if status_code in _NON_RETRYABLE_STATUS_CODES:
                    raise OpenAIChatModelAPIError(
                        f"Async streaming failed with non-retryable status {status_code}. "
                        f"Error: {type(exc).__name__}: {exc}"
                    ) from exc
                if emitted or attempt == self.max_retries:
                    raise OpenAIChatModelAPIError(
                        f"Async streaming failed after {attempt} attempt(s). "
                        f"Last error: {type(exc).__name__}: {exc}"
                    ) from exc
                await asyncio.sleep(self.backoff_factor * (2 ** (attempt - 1)))
        raise OpenAIChatModelAPIError("Async streaming failed unexpectedly") from last_exc

    # ── Public API ────────────────────────────────────────────────────────────

    def invoke(
        self,
        prompt: Any = None,
        prompt_variables: Optional[Dict[str, Any]] = None,
        *,
        files: Optional[Any] = None,
        image_detail: Optional[str] = None,
    ) -> Any:
        """
        Generate a response synchronously.

        Args:
            prompt: Text prompt string, or pre-built messages list, or None to use
                prompt_template.
            prompt_variables: Variables to fill prompt_template placeholders.
            files: Image file paths, bytes, or dicts to include as vision input.
            image_detail: OpenAI image detail level — "low", "high", or "auto".

        Returns:
            Generated text string, or a metadata dict if structured_output=True.
        """
        resolved = self._resolve_prompt(prompt, prompt_variables, files)
        messages = self._build_messages(resolved, files=files, image_detail=image_detail)

        if self.streaming:
            return "".join(self._invoke_stream_mode(messages=messages))

        with track_latency() as timing:
            response_text, raw_response = self._invoke_non_stream(messages=messages)

        metadata = build_structured_output(
            model_name=self._model_name,
            response_text=response_text,
            raw_response=raw_response,
            latency_ms=timing["latency_ms"],
            input_pricing=self.input_pricing,
            output_pricing=self.output_pricing,
        )
        self.last_metadata = metadata
        return metadata if self.structured_output else response_text

    async def ainvoke(
        self,
        prompt: Any = None,
        prompt_variables: Optional[Dict[str, Any]] = None,
        *,
        files: Optional[Any] = None,
        image_detail: Optional[str] = None,
    ) -> Any:
        """Async version of invoke()."""
        resolved = self._resolve_prompt(prompt, prompt_variables, files)
        messages = self._build_messages(resolved, files=files, image_detail=image_detail)

        if self.streaming:
            chunks: List[str] = []
            async for delta in self._ainvoke_stream_mode(messages=messages):
                chunks.append(delta)
            return "".join(chunks)

        with track_latency() as timing:
            response_text, raw_response = await self._ainvoke_non_stream(messages=messages)

        metadata = build_structured_output(
            model_name=self._model_name,
            response_text=response_text,
            raw_response=raw_response,
            latency_ms=timing["latency_ms"],
            input_pricing=self.input_pricing,
            output_pricing=self.output_pricing,
        )
        self.last_metadata = metadata
        return metadata if self.structured_output else response_text

    def stream(
        self,
        prompt: Any = None,
        prompt_variables: Optional[Dict[str, Any]] = None,
        *,
        files: Optional[Any] = None,
        image_detail: Optional[str] = None,
    ) -> Iterator[str]:
        """Stream text chunks synchronously."""
        resolved = self._resolve_prompt(prompt, prompt_variables, files)
        messages = self._build_messages(resolved, files=files, image_detail=image_detail)
        return self._invoke_stream_mode(messages=messages)

    async def astream(
        self,
        prompt: Any = None,
        prompt_variables: Optional[Dict[str, Any]] = None,
        *,
        files: Optional[Any] = None,
        image_detail: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream text chunks asynchronously."""
        resolved = self._resolve_prompt(prompt, prompt_variables, files)
        messages = self._build_messages(resolved, files=files, image_detail=image_detail)
        async for chunk in self._ainvoke_stream_mode(messages=messages):
            yield chunk

    # ── Low-level create() / acreate() ───────────────────────────────────────

    def create(self, input_data: Any = None, **overrides: Any) -> Any:
        """Direct access to client.chat.completions.create() with managed retries."""
        if input_data is None:
            input_data = overrides.pop("messages", None)
        if input_data is None:
            raise ValueError("input_data (messages) is required")
        params = self._build_base_params(messages=input_data, stream=False)
        params.update(overrides)
        return self._create_raw(params)

    async def acreate(self, input_data: Any = None, **overrides: Any) -> Any:
        """Async version of create()."""
        if input_data is None:
            input_data = overrides.pop("messages", None)
        if input_data is None:
            raise ValueError("input_data (messages) is required")
        params = self._build_base_params(messages=input_data, stream=False)
        params.update(overrides)
        return await self._acreate_raw(params)

    # ── Batch ─────────────────────────────────────────────────────────────────

    def batch_invoke(self, prompts: List[Any]) -> List[Any]:
        """Run invoke() for each prompt sequentially and return a list of results."""
        return [self.invoke(prompt=p) for p in prompts]

    async def abatch_invoke(self, prompts: List[Any]) -> List[Any]:
        """Run ainvoke() for each prompt concurrently and return a list of results."""
        return list(await asyncio.gather(*[self.ainvoke(prompt=p) for p in prompts]))

    # ── Native function-calling ───────────────────────────────────────────────

    @staticmethod
    def _tools_to_openai_format(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert Autourgos tool dicts to the OpenAI {type, function} schema."""
        result: List[Dict[str, Any]] = []
        for t in tools:
            if t.get("type") == "function":
                result.append(t)
                continue
            fn_schema: Dict[str, Any] = {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("parameters") or {"type": "object", "properties": {}},
            }
            result.append({"type": "function", "function": fn_schema})
        return result

    @staticmethod
    def _parse_tool_calls(raw_response: Any) -> List[FunctionCall]:
        """Extract FunctionCall objects from an OpenAI completion response."""
        calls: List[FunctionCall] = []
        choices = getattr(raw_response, "choices", None)
        if not choices:
            return calls
        tc_list = getattr(choices[0].message, "tool_calls", None) or []
        for tc in tc_list:
            try:
                arguments = json.loads(tc.function.arguments or "{}")
            except (json.JSONDecodeError, AttributeError):
                arguments = {}
            calls.append(FunctionCall(
                name=tc.function.name,
                arguments=arguments,
                call_id=getattr(tc, "id", None),
            ))
        return calls

    def invoke_with_tools(
        self,
        prompt: Any,
        tools: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> ToolCallResponse:
        """
        Call the Chat Completions API with native function-calling tools.

        Args:
            prompt: User message string or messages list.
            tools: List of tool dicts with keys name/description/parameters.
            **kwargs: Optional overrides such as tool_choice, files, image_detail.

        Returns:
            ToolCallResponse with tool_calls (if the model called tools)
            or text (if the model gave a final answer).
        """
        image_detail = kwargs.pop("image_detail", None)
        files = kwargs.pop("files", None)
        messages = self._build_messages(prompt, files=files, image_detail=image_detail)
        openai_tools = self._tools_to_openai_format(tools)
        params = self._build_base_params(messages=messages, stream=False)
        if openai_tools:
            params["tools"] = openai_tools
            params["tool_choice"] = kwargs.get("tool_choice", "auto")
        raw = self._create_raw(params)
        tool_calls = self._parse_tool_calls(raw) if openai_tools else []
        if tool_calls:
            return ToolCallResponse(tool_calls=tool_calls, raw=raw)
        text = extract_text_from_response(raw)
        return ToolCallResponse(text=text, raw=raw)

    async def ainvoke_with_tools(
        self,
        prompt: Any,
        tools: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> ToolCallResponse:
        """Async version of invoke_with_tools()."""
        image_detail = kwargs.pop("image_detail", None)
        files = kwargs.pop("files", None)
        messages = self._build_messages(prompt, files=files, image_detail=image_detail)
        openai_tools = self._tools_to_openai_format(tools)
        params = self._build_base_params(messages=messages, stream=False)
        if openai_tools:
            params["tools"] = openai_tools
            params["tool_choice"] = kwargs.get("tool_choice", "auto")
        raw = await self._acreate_raw(params)
        tool_calls = self._parse_tool_calls(raw) if openai_tools else []
        if tool_calls:
            return ToolCallResponse(tool_calls=tool_calls, raw=raw)
        text = extract_text_from_response(raw)
        return ToolCallResponse(text=text, raw=raw)

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"OpenAIChatModel(model={self.model!r}, "
            f"streaming={self.streaming}, "
            f"structured_output={self.structured_output})"
        )


__all__ = [
    "OpenAIChatModel",
    "OpenAIChatModelError",
    "OpenAIChatModelAPIError",
    "OpenAIChatModelImportError",
    "OpenAIChatModelResponseError",
    "OpenAIChatModelConfigError",
]
