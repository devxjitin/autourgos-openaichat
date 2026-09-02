"""
autourgos-openaichat
====================
Self-contained LLM wrapper for the OpenAI Chat Completions API.

Quick start::

    from autourgos_openaichat import OpenAIChatModel

    llm = OpenAIChatModel(model="gpt-4o", api_key="sk-...")
    reply = llm.invoke("What is the capital of France?")

    # Async
    reply = await llm.ainvoke("Translate 'hello' to Spanish.")

    # Streaming
    for chunk in llm.stream("Tell me a joke."):
        print(chunk, end="", flush=True)

    # Native tool-calling
    response = llm.invoke_with_tools("What's the weather?", tools=[...])
    if response.has_tool_calls:
        print(response.tool_calls)
"""

from .chat import (
    OpenAIChatModel,
    OpenAIChatModelAllProvidersFailedError,
    OpenAIChatModelAPIError,
    OpenAIChatModelConfigError,
    OpenAIChatModelError,
    OpenAIChatModelImportError,
    OpenAIChatModelRedactionBlockedError,
    OpenAIChatModelResponseError,
    OpenAIChatModelValidationError,
)
from .llm import (
    BaseLLM,
    BudgetExceededException,
    CircuitBreakerOpenException,
    FunctionCall,
    ToolCallResponse,
)
from .core import enforce_additional_properties_false
from .model_runtime import (
    build_structured_output,
    configure_runtime_environment,
    extract_text_from_response,
    extract_usage_metadata,
    track_latency,
)

try:
    from importlib.metadata import version as _v
    __version__ = _v("autourgos-openaichat")
except Exception:
    __version__ = "2.4.2"

__all__ = [
    # Main class
    "OpenAIChatModel",
    # Exceptions
    "OpenAIChatModelError",
    "OpenAIChatModelAPIError",
    "OpenAIChatModelImportError",
    "OpenAIChatModelResponseError",
    "OpenAIChatModelConfigError",
    "OpenAIChatModelAllProvidersFailedError",
    "OpenAIChatModelValidationError",
    "OpenAIChatModelRedactionBlockedError",
    # Base types
    "BaseLLM",
    "FunctionCall",
    "ToolCallResponse",
    "CircuitBreakerOpenException",
    "BudgetExceededException",
    # Runtime helpers
    "track_latency",
    "extract_usage_metadata",
    "build_structured_output",
    "extract_text_from_response",
    "configure_runtime_environment",
    "enforce_additional_properties_false",
]
