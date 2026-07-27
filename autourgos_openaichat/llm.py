"""
BaseLLM — base interface for all autourgos-openaichat model wrappers.

Fully self-contained: no autourgos-core dependency.
"""

from __future__ import annotations

import asyncio
import functools
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional

_lazy_init_lock = threading.Lock()


class CircuitBreakerOpenException(Exception):
    """Raised when the circuit breaker is open, blocking LLM calls."""


@dataclass
class FunctionCall:
    """A single tool call requested by the LLM."""
    name: str
    arguments: Dict[str, Any]
    call_id: Optional[str] = None


@dataclass
class ToolCallResponse:
    """Return type for invoke_with_tools."""
    tool_calls: List[FunctionCall] = field(default_factory=list)
    text: Optional[str] = None
    raw: Any = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def is_final_answer(self) -> bool:
        return not self.tool_calls and self.text is not None


class BaseLLM(ABC):
    """
    Standard interface for all autourgos-openaichat model wrappers.

    Subclasses MUST implement:
        - invoke()     — synchronous generation
        - ainvoke()    — asynchronous generation

    Subclasses MAY override:
        - stream()             — sync streaming
        - astream()            — async streaming
        - invoke_with_tools()  — native function-calling
        - ainvoke_with_tools() — async native function-calling
    """

    supports_tool_calling: bool = False

    def __init__(
        self,
        input_pricing: Optional[float] = None,
        output_pricing: Optional[float] = None,
        circuit_failure_threshold: int = 5,
        circuit_cooldown_time: float = 30.0,
    ) -> None:
        self.input_pricing = input_pricing
        self.output_pricing = output_pricing
        self.last_metadata: Dict[str, Any] = {}

        self._consecutive_failures = 0
        self._circuit_tripped_until: Optional[float] = None
        self._circuit_lock = threading.Lock()
        self._async_circuit_lock: Optional[asyncio.Lock] = None
        self.circuit_failure_threshold = circuit_failure_threshold
        self.circuit_cooldown_time = circuit_cooldown_time

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "invoke" in cls.__dict__:
            cls.invoke = cls._wrap_sync(cls.invoke)
        if "ainvoke" in cls.__dict__:
            cls.ainvoke = cls._wrap_async(cls.ainvoke)
        if "invoke_with_tools" in cls.__dict__:
            cls.invoke_with_tools = cls._wrap_sync(cls.invoke_with_tools)
        if "ainvoke_with_tools" in cls.__dict__:
            cls.ainvoke_with_tools = cls._wrap_async(cls.ainvoke_with_tools)

    # ── Circuit breaker wrappers ──────────────────────────────────────────────

    @staticmethod
    def _wrap_sync(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self: "BaseLLM", *args: Any, **kwargs: Any) -> Any:
            if not hasattr(self, "_circuit_lock"):
                with _lazy_init_lock:
                    if not hasattr(self, "_circuit_lock"):
                        self._circuit_lock = threading.Lock()
                        self._consecutive_failures = 0
                        self._circuit_tripped_until = None
                        self.circuit_failure_threshold = 5
                        self.circuit_cooldown_time = 30.0

            with self._circuit_lock:
                if self._circuit_tripped_until is not None:
                    if time.time() < self._circuit_tripped_until:
                        raise CircuitBreakerOpenException(
                            f"Circuit breaker OPEN for {type(self).__name__} — "
                            f"{self._consecutive_failures} consecutive failures. "
                            f"Blocked until {self._circuit_tripped_until}."
                        )
                    self._circuit_tripped_until = None

            try:
                result = func(self, *args, **kwargs)
                with self._circuit_lock:
                    self._consecutive_failures = 0
                return result
            except Exception as exc:
                if not isinstance(exc, (
                    TypeError, ValueError, KeyError, AttributeError,
                    NotImplementedError, CircuitBreakerOpenException,
                )):
                    with self._circuit_lock:
                        self._consecutive_failures += 1
                        if self._consecutive_failures >= self.circuit_failure_threshold:
                            self._circuit_tripped_until = time.time() + self.circuit_cooldown_time
                raise

        return wrapper

    @staticmethod
    def _wrap_async(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(self: "BaseLLM", *args: Any, **kwargs: Any) -> Any:
            if not hasattr(self, "_circuit_lock"):
                with _lazy_init_lock:
                    if not hasattr(self, "_circuit_lock"):
                        self._circuit_lock = threading.Lock()
                        self._async_circuit_lock = None
                        self._consecutive_failures = 0
                        self._circuit_tripped_until = None
                        self.circuit_failure_threshold = 5
                        self.circuit_cooldown_time = 30.0

            # Guard the check-then-create with a plain threading.Lock so two
            # threads (each driving their own event loop) can't both observe
            # `None` and each create a separate asyncio.Lock — a narrow but
            # real check-then-set race. Simple double-checked locking: cheap
            # fast path when already initialized, safe slow path otherwise.
            if self._async_circuit_lock is None:
                with _lazy_init_lock:
                    if self._async_circuit_lock is None:
                        self._async_circuit_lock = asyncio.Lock()

            async with self._async_circuit_lock:
                if self._circuit_tripped_until is not None:
                    if time.time() < self._circuit_tripped_until:
                        raise CircuitBreakerOpenException(
                            f"Circuit breaker OPEN for {type(self).__name__} — "
                            f"{self._consecutive_failures} consecutive failures. "
                            f"Blocked until {self._circuit_tripped_until}."
                        )
                    self._circuit_tripped_until = None

            try:
                result = await func(self, *args, **kwargs)
                async with self._async_circuit_lock:
                    self._consecutive_failures = 0
                return result
            except Exception as exc:
                if not isinstance(exc, (
                    TypeError, ValueError, KeyError, AttributeError,
                    NotImplementedError, CircuitBreakerOpenException,
                )):
                    async with self._async_circuit_lock:
                        self._consecutive_failures += 1
                        if self._consecutive_failures >= self.circuit_failure_threshold:
                            self._circuit_tripped_until = time.time() + self.circuit_cooldown_time
                raise

        return wrapper

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def invoke(self, prompt: Any = None, **kwargs: Any) -> Any:
        """Synchronous generation."""

    @abstractmethod
    async def ainvoke(self, prompt: Any = None, **kwargs: Any) -> Any:
        """Asynchronous generation."""

    def stream(self, prompt: Any = None, **kwargs: Any) -> Iterator[str]:
        raise NotImplementedError(f"{type(self).__name__} does not support streaming.")

    async def astream(self, prompt: Any = None, **kwargs: Any) -> AsyncIterator[str]:
        raise NotImplementedError(f"{type(self).__name__} does not support async streaming.")

    def invoke_with_tools(
        self,
        prompt: Any,
        tools: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> "ToolCallResponse":
        raise NotImplementedError(
            f"{type(self).__name__} does not support native function-calling."
        )

    async def ainvoke_with_tools(
        self,
        prompt: Any,
        tools: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> "ToolCallResponse":
        raise NotImplementedError(
            f"{type(self).__name__} does not support async native function-calling."
        )


__all__ = [
    "BaseLLM",
    "FunctionCall",
    "ToolCallResponse",
    "CircuitBreakerOpenException",
]
