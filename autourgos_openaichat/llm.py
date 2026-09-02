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
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional

_lazy_init_lock = threading.Lock()


class CircuitBreakerOpenException(Exception):
    """Raised when the circuit breaker is open, blocking LLM calls."""


class BudgetExceededException(Exception):
    """Raised when a call is blocked because max_session_cost has already been reached."""


class NonTransientError(Exception):
    """
    Marker mixin for exceptions that must never count toward the circuit
    breaker's consecutive-failure threshold.

    The circuit breaker exists to detect a genuinely unhealthy provider and
    fail fast instead of hammering it. A caller/config mistake (e.g. an
    invalid ``output_schema``) or a by-design policy block (e.g.
    ``redact_mode="block"`` correctly refusing to send a prompt that matched
    a redaction pattern) is neither -- the provider may be perfectly healthy.
    Without this, five such mistakes/blocks in a row trip the breaker and
    block every *other* call on the same instance (including unrelated,
    healthy ones) for ``circuit_cooldown_time`` seconds.

    Subclasses in concrete wrappers (e.g. ``OpenAIChatModelConfigError``,
    ``OpenAIChatModelRedactionBlockedError``) mix this in alongside their
    normal exception base so existing ``except OpenAIChatModelConfigError``
    callers are unaffected.
    """


@dataclass
class FunctionCall:
    """A single tool call requested by the LLM."""
    name: str
    arguments: Dict[str, Any]
    call_id: Optional[str] = None
    # Set when the model returned malformed JSON for `arguments` -- `arguments`
    # still falls back to {} in that case (so existing callers keep working
    # without a null-check), but callers that care can check this field
    # instead of silently getting a tool call with wrong/missing arguments
    # and no signal anything went wrong.
    arguments_parse_error: Optional[str] = None


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
        max_session_cost: Optional[float] = None,
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

        self.max_session_cost = max_session_cost
        self.session_cost_used: float = 0.0
        self._budget_lock = threading.Lock()
        self._budget_admission_lock = threading.Lock()
        self._async_budget_admission_lock: Optional[asyncio.Lock] = None

    # ── Budget governor ───────────────────────────────────────────────────────
    # A call's cost is only known after its response, so the *check* and the
    # *record* can't be one atomic operation the way a simple counter could
    # be. Without admission control, N concurrent calls sharing one capped
    # instance (e.g. via abatch_invoke()) can all pass _check_budget() before
    # any of them records its cost, overshooting the cap by up to N-1 calls'
    # worth. _budget_admission()/_async_budget_admission() close that window
    # by holding an admission lock from the (re-)check through to whichever
    # comes first for that call: the caller recording its cost, or raising.
    # Only one call can be inside the admission section at a time *within its
    # own concurrency domain* (sync calls serialize against other sync calls
    # via a threading.Lock; async calls serialize against other async calls
    # via an asyncio.Lock) -- consistent with this class's circuit-breaker
    # locking, mixing sync invoke() and async ainvoke() calls concurrently on
    # the same instance is not itself serialized against the other domain.
    # This trades away concurrency for calls sharing a capped instance in
    # exchange for the cap actually holding under concurrent load; an
    # uncapped instance (max_session_cost=None) pays no serialization cost,
    # since _check_budget() is a no-op and the lock is uncontended.

    def _check_budget(self) -> None:
        if self.max_session_cost is not None and self.session_cost_used >= self.max_session_cost:
            raise BudgetExceededException(
                f"Session budget exceeded for {type(self).__name__}: "
                f"${self.session_cost_used:.6f} used of ${self.max_session_cost:.6f} cap."
            )

    def _record_session_cost(self, cost: Optional[float]) -> None:
        if cost is None:
            return
        with self._budget_lock:
            self.session_cost_used += cost

    @contextmanager
    def _budget_admission(self) -> Iterator[None]:
        """
        Sync admission section: acquire the sync admission lock, re-check the
        budget (authoritative -- closes the race an earlier optimistic
        `_check_budget()` call can't), then hold the lock for the caller's
        entire critical section (the real API call plus `_record_session_cost()`)
        so no other sync call on this instance can be admitted until this one
        finishes or raises.
        """
        with self._budget_admission_lock:
            self._check_budget()
            yield

    @asynccontextmanager
    async def _async_budget_admission(self) -> AsyncIterator[None]:
        """Async counterpart of `_budget_admission()`. See its docstring."""
        if self._async_budget_admission_lock is None:
            with _lazy_init_lock:
                if self._async_budget_admission_lock is None:
                    self._async_budget_admission_lock = asyncio.Lock()
        async with self._async_budget_admission_lock:
            self._check_budget()
            yield

    def reset_session_budget(self) -> None:
        """Reset accumulated session cost back to 0, unblocking a tripped budget cap."""
        with self._budget_lock:
            self.session_cost_used = 0.0

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
        if "invoke_structured" in cls.__dict__:
            cls.invoke_structured = cls._wrap_sync(cls.invoke_structured)
        if "ainvoke_structured" in cls.__dict__:
            cls.ainvoke_structured = cls._wrap_async(cls.ainvoke_structured)

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
                    NotImplementedError, CircuitBreakerOpenException, BudgetExceededException,
                    NonTransientError,
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
                    NotImplementedError, CircuitBreakerOpenException, BudgetExceededException,
                    NonTransientError,
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
    "BudgetExceededException",
    "NonTransientError",
]
