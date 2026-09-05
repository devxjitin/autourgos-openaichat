"""
BaseLLM — base interface for all autourgos-openaichat model wrappers.

Depends only on the `openai` SDK and `autourgos-core` (a separate,
zero-dependency stdlib utility library shared across the framework) --
no other third-party or autourgos-* dependency.
"""

from __future__ import annotations

import asyncio
import functools
import threading
import time
import weakref
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional

from concurrent.futures import ThreadPoolExecutor

from autourgos_core import aretry_with_backoff, retry_with_backoff

from .core import (
    configure_async_openai_client,
    configure_openai_client,
    normalize_model_name,
    release_async_openai_client,
    release_openai_client,
    resolve_api_key,
    resolve_base_url,
)
from .ledger import close_ledger, open_ledger, write_ledger_entry, write_shadow_ledger_entry
from .model_runtime import build_structured_output, track_latency
from .redaction import compile_patterns, restore_text
from .shadow import compute_similarity

_lazy_init_lock = threading.Lock()

_NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404, 422}

# Sentinel distinguishing "no override passed" from "override explicitly None"
# in _log_to_ledger()'s response_override= parameter.
_UNSET = object()


def _get_loop_local_lock(
    locks: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]",
) -> asyncio.Lock:
    """
    Get-or-create the ``asyncio.Lock`` for the *currently running* event loop.

    ``asyncio.Lock`` is not thread-safe and is only valid within a single
    event loop at a time -- a single Lock object shared for an instance's
    whole lifetime works fine for sequential ``asyncio.run()`` calls, but two
    threads each driving their own loop and concurrently `async with`-ing the
    *same* Lock object can deadlock (reproduced directly: 4 threads each
    running their own loop, both hammering one shared Lock, hung
    indefinitely). Keying by the running loop in a ``WeakKeyDictionary`` gives
    every loop its own Lock, so no two loops ever touch the same mutable Lock
    state concurrently; entries for short-lived loops (e.g. one per
    ``asyncio.run()`` call in a thread-pool worker) are garbage-collected
    automatically once their loop is, instead of accumulating forever.
    """
    loop = asyncio.get_running_loop()
    lock = locks.get(loop)
    if lock is None:
        with _lazy_init_lock:
            lock = locks.get(loop)
            if lock is None:
                lock = asyncio.Lock()
                locks[loop] = lock
    return lock


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
        self._async_circuit_locks: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = weakref.WeakKeyDictionary()
        self.circuit_failure_threshold = circuit_failure_threshold
        self.circuit_cooldown_time = circuit_cooldown_time

        self.max_session_cost = max_session_cost
        self.session_cost_used: float = 0.0
        self._budget_lock = threading.Lock()
        self._budget_admission_lock = threading.Lock()
        self._async_budget_admission_locks: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = weakref.WeakKeyDictionary()

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
        lock = _get_loop_local_lock(self._async_budget_admission_locks)
        async with lock:
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
        # chat()/achat() (multi-turn, e.g. autourgos-responses' OpenAIResponse)
        # go through the same wrapper as invoke()/ainvoke() -- same call shape
        # (return a value or raise; not a generator), so a provider outage
        # detected via chat() trips the breaker just like every other path.
        if "chat" in cls.__dict__:
            cls.chat = cls._wrap_sync(cls.chat)
        if "achat" in cls.__dict__:
            cls.achat = cls._wrap_async(cls.achat)
        if "invoke_with_tools" in cls.__dict__:
            cls.invoke_with_tools = cls._wrap_sync(cls.invoke_with_tools)
        if "ainvoke_with_tools" in cls.__dict__:
            cls.ainvoke_with_tools = cls._wrap_async(cls.ainvoke_with_tools)
        if "invoke_structured" in cls.__dict__:
            cls.invoke_structured = cls._wrap_sync(cls.invoke_structured)
        if "ainvoke_structured" in cls.__dict__:
            cls.ainvoke_structured = cls._wrap_async(cls.ainvoke_structured)
        if "stream" in cls.__dict__:
            cls.stream = cls._wrap_sync_stream(cls.stream)
        if "astream" in cls.__dict__:
            cls.astream = cls._wrap_async_stream(cls.astream)

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
                        self._async_circuit_locks = weakref.WeakKeyDictionary()
                        self._consecutive_failures = 0
                        self._circuit_tripped_until = None
                        self.circuit_failure_threshold = 5
                        self.circuit_cooldown_time = 30.0

            # See _get_loop_local_lock()'s docstring: one Lock per running
            # event loop, not one shared for the instance's whole lifetime --
            # a single shared asyncio.Lock deadlocks under genuinely
            # concurrent cross-loop use (e.g. two thread-pool workers each
            # running their own asyncio.run()).
            async_circuit_lock = _get_loop_local_lock(self._async_circuit_locks)

            async with async_circuit_lock:
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
                async with async_circuit_lock:
                    self._consecutive_failures = 0
                return result
            except Exception as exc:
                if not isinstance(exc, (
                    TypeError, ValueError, KeyError, AttributeError,
                    NotImplementedError, CircuitBreakerOpenException, BudgetExceededException,
                    NonTransientError,
                )):
                    async with async_circuit_lock:
                        self._consecutive_failures += 1
                        if self._consecutive_failures >= self.circuit_failure_threshold:
                            self._circuit_tripped_until = time.time() + self.circuit_cooldown_time
                raise

        return wrapper

    # ── Circuit breaker wrappers for streaming ───────────────────────────────
    # stream()/astream() return an iterator/async-iterator rather than a
    # value or awaitable, so _wrap_sync/_wrap_async can't be reused as-is:
    # calling a generator function doesn't run its body until it's first
    # iterated, so treating a mere call as "success" would reset the circuit
    # before any actual request happened, and a mid-stream failure (raised
    # partway through iteration, not from the call itself) would never be
    # observed at all. These wrappers check/trip the circuit around the call
    # (so a call while the circuit is open still raises immediately, without
    # requiring the caller to start iterating first) and around the full
    # iteration of whatever stream()/astream() returns.

    @staticmethod
    def _wrap_sync_stream(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self: "BaseLLM", *args: Any, **kwargs: Any) -> Iterator[Any]:
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

            # func(...) itself must run eagerly here (not inside the generator
            # below) so a caller building the iterator while the circuit is
            # open sees CircuitBreakerOpenException immediately, matching
            # invoke()'s behavior, instead of only on first iteration.
            stream = func(self, *args, **kwargs)
            return self._consume_stream_for_circuit(stream)

        return wrapper

    def _consume_stream_for_circuit(self, stream: Iterator[Any]) -> Iterator[Any]:
        """Drive `stream`, updating the circuit breaker based on whether it raises."""
        try:
            for item in stream:
                yield item
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
        else:
            with self._circuit_lock:
                self._consecutive_failures = 0

    @staticmethod
    def _wrap_async_stream(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(self: "BaseLLM", *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            if not hasattr(self, "_circuit_lock"):
                with _lazy_init_lock:
                    if not hasattr(self, "_circuit_lock"):
                        self._circuit_lock = threading.Lock()
                        self._async_circuit_locks = weakref.WeakKeyDictionary()
                        self._consecutive_failures = 0
                        self._circuit_tripped_until = None
                        self.circuit_failure_threshold = 5
                        self.circuit_cooldown_time = 30.0

            # See _wrap_async()'s identical comment / _get_loop_local_lock().
            async_circuit_lock = _get_loop_local_lock(self._async_circuit_locks)

            async with async_circuit_lock:
                if self._circuit_tripped_until is not None:
                    if time.time() < self._circuit_tripped_until:
                        raise CircuitBreakerOpenException(
                            f"Circuit breaker OPEN for {type(self).__name__} — "
                            f"{self._consecutive_failures} consecutive failures. "
                            f"Blocked until {self._circuit_tripped_until}."
                        )
                    self._circuit_tripped_until = None

            # `wrapper` is itself an async generator (it yields below), so
            # nothing above this point actually runs until the caller's first
            # `__anext__()` -- consistent with how an unwrapped astream()
            # already behaves (nothing happens until iteration starts), so
            # this doesn't change when the circuit is observably checked
            # relative to the pre-fix behavior for astream() specifically.
            stream = func(self, *args, **kwargs)
            try:
                async for item in stream:
                    yield item
            except Exception as exc:
                if not isinstance(exc, (
                    TypeError, ValueError, KeyError, AttributeError,
                    NotImplementedError, CircuitBreakerOpenException, BudgetExceededException,
                    NonTransientError,
                )):
                    async with async_circuit_lock:
                        self._consecutive_failures += 1
                        if self._consecutive_failures >= self.circuit_failure_threshold:
                            self._circuit_tripped_until = time.time() + self.circuit_cooldown_time
                raise
            else:
                async with async_circuit_lock:
                    self._consecutive_failures = 0

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


class _OpenAIClientLifecycleMixin:
    """
    Shared OpenAI SDK client lifecycle: import-check + primary/fallback/
    shadow client construction, and context-manager/close/aclose cleanup.

    Identical between OpenAIChatModel (Chat Completions) and OpenAIResponse
    (Responses API) -- both wrap the same ``openai`` SDK client shape and
    fallback/shadow-provider machinery, only the API endpoint they call
    differs (handled elsewhere, in ``_do_sync_create``/``_do_async_create``).

    A subclass must set ``_import_error_cls`` (exception type raised when the
    ``openai`` SDK isn't installed, e.g. ``OpenAIChatModelImportError``) and
    override ``_get_openai_client_classes()`` to return that module's own
    ``load_openai_module()`` four-tuple -- as a classmethod re-reading the
    module's globals on every call, not a value copied once at class-body
    execution time, so a test patching those module globals (e.g. to
    simulate the SDK being missing) is observed on the next call, matching
    the pre-extraction behavior where these methods read module globals
    directly.

    And must already have (from ``BaseProviderLLM._init_provider_common``):
    ``self.api_key``/``base_url``/``organization``/``project``/``timeout``,
    ``self.fallback_providers``/``shadow_providers``,
    ``self._fallback_sync_clients``/``_fallback_async_clients``/
    ``_shadow_sync_clients``/``_shadow_async_clients``, ``self._ledger_conn``.
    """

    _import_error_cls: type

    @classmethod
    def _get_openai_client_classes(cls) -> "tuple[Any, Any, bool, Optional[str]]":
        """Return (openai_cls, async_openai_cls, available, import_error).

        Subclasses override this to return their own module's
        ``load_openai_module()`` result, read fresh on every call.
        """
        raise NotImplementedError

    # ── Client init ───────────────────────────────────────────────────────────

    def _init_clients(self) -> None:
        openai_cls, async_openai_cls, available, import_error = self._get_openai_client_classes()
        if not available or openai_cls is None or async_openai_cls is None:
            detail = f" Details: {import_error}" if import_error else ""
            raise self._import_error_cls(
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

    def _get_fallback_sync_client(self, index: int) -> Any:
        """Lazily create and cache the sync client for fallback_providers[index]."""
        if index not in self._fallback_sync_clients:
            openai_cls, _async_openai_cls, _available, _import_error = self._get_openai_client_classes()
            cfg = self.fallback_providers[index]
            self._fallback_sync_clients[index] = configure_openai_client(
                openai_cls,
                api_key=resolve_api_key(cfg.get("api_key")),
                base_url=resolve_base_url(cfg.get("base_url")),
                organization=cfg.get("organization"),
                project=cfg.get("project"),
                timeout=self.timeout,
            )
        return self._fallback_sync_clients[index]

    def _get_fallback_async_client(self, index: int) -> Any:
        """Lazily create and cache the async client for fallback_providers[index]."""
        if index not in self._fallback_async_clients:
            _openai_cls, async_openai_cls, _available, _import_error = self._get_openai_client_classes()
            cfg = self.fallback_providers[index]
            self._fallback_async_clients[index] = configure_async_openai_client(
                async_openai_cls,
                api_key=resolve_api_key(cfg.get("api_key")),
                base_url=resolve_base_url(cfg.get("base_url")),
                organization=cfg.get("organization"),
                project=cfg.get("project"),
                timeout=self.timeout,
            )
        return self._fallback_async_clients[index]

    def _get_shadow_sync_client(self, index: int) -> Any:
        """Lazily create and cache the sync client for shadow_providers[index]."""
        if index not in self._shadow_sync_clients:
            openai_cls, _async_openai_cls, _available, _import_error = self._get_openai_client_classes()
            cfg = self.shadow_providers[index]
            self._shadow_sync_clients[index] = configure_openai_client(
                openai_cls,
                api_key=resolve_api_key(cfg.get("api_key")),
                base_url=resolve_base_url(cfg.get("base_url")),
                organization=cfg.get("organization"),
                project=cfg.get("project"),
                timeout=self.timeout,
            )
        return self._shadow_sync_clients[index]

    def _get_shadow_async_client(self, index: int) -> Any:
        """Lazily create and cache the async client for shadow_providers[index]."""
        if index not in self._shadow_async_clients:
            _openai_cls, async_openai_cls, _available, _import_error = self._get_openai_client_classes()
            cfg = self.shadow_providers[index]
            self._shadow_async_clients[index] = configure_async_openai_client(
                async_openai_cls,
                api_key=resolve_api_key(cfg.get("api_key")),
                base_url=resolve_base_url(cfg.get("base_url")),
                organization=cfg.get("organization"),
                project=cfg.get("project"),
                timeout=self.timeout,
            )
        return self._shadow_async_clients[index]

    # ── Context managers ──────────────────────────────────────────────────────

    def __enter__(self) -> "_OpenAIClientLifecycleMixin":
        return self

    def __exit__(self, *args: Any) -> None:
        if self._client is not None:
            release_openai_client(self._client)
            self._client = None
        for client in self._fallback_sync_clients.values():
            release_openai_client(client)
        self._fallback_sync_clients = {}
        for client in self._shadow_sync_clients.values():
            release_openai_client(client)
        self._shadow_sync_clients = {}
        close_ledger(self._ledger_conn)
        self._ledger_conn = None

    def close(self) -> None:
        """
        Release the underlying client(s) synchronously.

        Equivalent to ``__exit__()`` — lets callers that hold this LLM via
        composition (e.g. autourgos-agent's ``Agent``, whose context-manager
        cleanup calls ``llm.close()``/``llm.aclose()`` if present) release
        resources without needing to use ``with`` directly on this object.
        """
        self.__exit__()

    async def aclose(self) -> None:
        """Release the underlying client(s) asynchronously. Equivalent to ``__aexit__()``."""
        await self.__aexit__()

    async def __aenter__(self) -> "_OpenAIClientLifecycleMixin":
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._async_client is not None:
            await release_async_openai_client(self._async_client)
            self._async_client = None
        if self._client is not None:
            release_openai_client(self._client)
            self._client = None
        for client in self._fallback_async_clients.values():
            await release_async_openai_client(client)
        self._fallback_async_clients = {}
        for client in self._fallback_sync_clients.values():
            release_openai_client(client)
        self._fallback_sync_clients = {}
        for client in self._shadow_async_clients.values():
            await release_async_openai_client(client)
        self._shadow_async_clients = {}
        for client in self._shadow_sync_clients.values():
            release_openai_client(client)
        self._shadow_sync_clients = {}
        close_ledger(self._ledger_conn)
        self._ledger_conn = None


class BaseProviderLLM(BaseLLM):
    """
    Intermediate base for concrete provider wrappers (OpenAIChatModel,
    OpenAIResponse). Shared provider-level logic (constructor setup,
    retry/fallback dispatch, streaming, shadow mode) lands here
    incrementally.
    """

    # Concrete subclasses must set this to their own ConfigError type
    # (e.g. OpenAIChatModelConfigError) before calling _init_provider_common,
    # so shared validation raises the package's own exception type/message.
    _config_error_cls: type

    # Concrete subclasses must set this to their own DeadlineExceededError
    # type (e.g. OpenAIChatModelDeadlineExceededError).
    _deadline_exceeded_cls: type

    # Concrete subclasses must set these before any create/retry call:
    #   _api_error_cls: e.g. OpenAIChatModelAPIError
    #   _all_providers_failed_cls: e.g. OpenAIChatModelAllProvidersFailedError
    #   _api_name: human-readable API label used in error messages, e.g.
    #       "Chat Completions" (giving "Chat Completions request failed...")
    #       or "Responses API" (giving "Responses API request failed...").
    _api_error_cls: type
    _all_providers_failed_cls: type
    _api_name: str

    def _do_sync_create(self, client: Any, params: Dict[str, Any]) -> Any:
        raise NotImplementedError

    async def _do_async_create(self, client: Any, params: Dict[str, Any]) -> Any:
        raise NotImplementedError

    def _apply_per_target_param_guards(self, params: Dict[str, Any], model_name: str) -> None:
        raise NotImplementedError

    # Concrete subclasses must set this to their own module logger (e.g.
    # ``_logger = logger`` where ``logger`` is that package's already-imported
    # ``logging.getLogger(__name__)``) -- log records must keep coming from
    # each package's own logger name, not a shared one.
    _logger: Any

    def _extract_response_text(self, raw: Any) -> Optional[str]:
        """Subclass one-liner delegating to that package's own extract_text_from_response."""
        raise NotImplementedError

    def _extract_usage(self, raw: Any) -> Dict[str, Optional[int]]:
        """Subclass one-liner delegating to that package's own extract_usage_metadata."""
        raise NotImplementedError

    def _build_base_params_for_call(
        self, prompt_input: Any, *, stream: bool, overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Subclass one-liner forwarding to the local _build_base_params(), whose keyword
        name for the prompt payload (``messages=`` vs ``input_data=``) differs per package."""
        raise NotImplementedError

    # Concrete subclasses must set this to their own ResponseError type
    # (e.g. OpenAIChatModelResponseError).
    _response_error_cls: type

    # Concrete subclasses may set this to their own RefusalError type (e.g.
    # OpenAIChatModelRefusalError) if they support refusal detection during
    # streaming. None (the default) means "this package has no refusal path"
    # -- _extract_refusal_delta() always returning None then keeps the
    # refusal branch permanently dead code, exactly as before.
    _refusal_error_cls: Optional[type] = None

    def _extract_text_delta(self, event: Any) -> Optional[str]:
        """Subclass one-liner delegating to that package's own extract_text_delta_from_event."""
        raise NotImplementedError

    def _extract_usage_event(self, event: Any) -> Optional[Any]:
        """
        Subclass-specific: return the object to record in usage_sink for this
        streaming event, or None if this event carries no usage/final data.
        Chat Completions extracts a per-chunk usage-bearing event; the
        Responses API extracts a terminal final-response object instead --
        different semantics, same usage_sink.append() wrapper shape.
        """
        raise NotImplementedError

    def _extract_refusal_delta(self, event: Any) -> Optional[str]:
        """Subclass override for packages with a refusal path (see _refusal_error_cls).
        Default: no refusal path -- always returns None."""
        return None

    # Concrete subclasses must set this to their own ValidationError type
    # (e.g. OpenAIChatModelValidationError).
    _validation_error_cls: type

    # create()/acreate(): the key popped from **overrides when input_data=None
    # (e.g. "messages" for Chat Completions, "input" for the Responses API),
    # and the ValueError message raised when neither is provided.
    _create_input_key: str
    _create_missing_input_message: str

    def _apply_create_param_guards(self, params: Dict[str, Any], model_name: str) -> None:
        """
        Per-target param guards applied inside create()/acreate() specifically.
        Default: none -- deliberately NOT the same as _apply_per_target_param_guards()
        (used by invoke()'s retry/fallback loop), because OpenAIResponse's
        create() has never applied any guard here (a pre-existing asymmetry,
        not something this refactor should silently "fix"). Override only
        where the original create() already applied guards.
        """
        return

    def _sync_targets(self) -> Iterator[tuple]:
        """
        Yield (label, model_name, client, pricing) for the primary, then each
        fallback in order. ``pricing`` is (input_pricing, output_pricing) —
        the primary's constructor-level pricing, or a fallback entry's own
        "input_pricing"/"output_pricing" keys (None if it doesn't set them,
        never inherited from the primary — a fallback is typically a
        different model with a different price).
        """
        yield "primary", self._model_name, self._client, (self.input_pricing, self.output_pricing)
        for i, cfg in enumerate(self.fallback_providers):
            yield (
                f"fallback[{i}]:{cfg['model']}",
                normalize_model_name(cfg["model"]),
                self._get_fallback_sync_client(i),
                (cfg.get("input_pricing"), cfg.get("output_pricing")),
            )

    def _async_targets(self) -> Iterator[tuple]:
        """Async counterpart of ``_sync_targets()``. See its docstring for ``pricing``."""
        yield "primary", self._model_name, self._async_client, (self.input_pricing, self.output_pricing)
        for i, cfg in enumerate(self.fallback_providers):
            yield (
                f"fallback[{i}]:{cfg['model']}",
                normalize_model_name(cfg["model"]),
                self._get_fallback_async_client(i),
                (cfg.get("input_pricing"), cfg.get("output_pricing")),
            )

    def _new_deadline(self, agent_deadline_seconds: Optional[float] = None) -> Optional[float]:
        """Return an absolute perf_counter() deadline, or None if uncapped.

        ``agent_deadline_seconds``, when given, is a caller-supplied "time
        remaining" (e.g. an owning Agent's max_execution_time budget) that's
        combined with this instance's own ``max_call_duration`` via min() --
        whichever runs out first wins. A non-positive value means the caller
        is already out of time, so the returned deadline is already-past
        (``time.perf_counter()``) rather than in the future, tripping the
        very next ``_raise_if_deadline_exceeded`` check immediately instead
        of allowing one more attempt/provider to be tried.
        """
        candidates: List[float] = []
        if self.max_call_duration is not None:
            candidates.append(time.perf_counter() + self.max_call_duration)
        if agent_deadline_seconds is not None:
            candidates.append(time.perf_counter() + max(agent_deadline_seconds, 0.0))
        return min(candidates) if candidates else None

    def _raise_if_deadline_exceeded(self, deadline: Optional[float], label: str) -> None:
        if deadline is not None and time.perf_counter() >= deadline:
            raise self._deadline_exceeded_cls(
                f"[{label}] max_call_duration exceeded before this attempt/provider could "
                f"be tried."
            )

    def _init_provider_common(
        self,
        *,
        max_retries: int,
        fallback_providers: Optional[List[Dict[str, Any]]],
        ledger_path: Optional[str],
        ledger_store_content: bool,
        redact_pii: bool,
        redact_mode: str,
        redact_categories: Optional[List[str]],
        redact_custom_patterns: Optional[Dict[str, str]],
        redact_custom_terms: Optional[Dict[str, List[str]]],
        redact_patterns_file: Optional[str],
        redact_restore_in_response: bool,
        shadow_providers: Optional[List[Dict[str, Any]]],
        on_shadow_result: Optional[Callable[[Dict[str, Any]], None]],
        extra_body: Optional[Dict[str, Any]],
        store: Optional[bool],
    ) -> None:
        """
        Shared constructor-time setup for concrete provider wrappers:
        max_retries validation, fallback/shadow provider normalization,
        ledger setup, redaction pattern compilation, and extra_body/store
        assignment. Raises ``self._config_error_cls`` on invalid input, so
        each subclass sees its own exception type unchanged.
        """
        self.max_retries = max_retries
        if self.max_retries < 1:
            raise self._config_error_cls(
                f"max_retries must be >= 1 (it's the total number of attempts per "
                f"provider, not retries after the first), got {self.max_retries}."
            )

        for i, entry in enumerate(fallback_providers or []):
            if not entry.get("model"):
                raise self._config_error_cls(
                    f"fallback_providers[{i}] is missing required key 'model'."
                )
        self.fallback_providers: List[Dict[str, Any]] = list(fallback_providers or [])
        self._fallback_sync_clients: Dict[int, Any] = {}
        self._fallback_async_clients: Dict[int, Any] = {}

        self.ledger_path = ledger_path
        self.ledger_store_content = ledger_store_content
        self._ledger_conn = open_ledger(ledger_path) if ledger_path else None
        self._ledger_lock = threading.Lock()

        if redact_mode not in ("mask", "block"):
            raise self._config_error_cls(
                f"redact_mode must be 'mask' or 'block', got {redact_mode!r}."
            )
        self.redact_pii = redact_pii
        self.redact_mode = redact_mode
        # Best-effort convenience attribute reflecting the most recently
        # *started* call's redaction categories -- like last_metadata, safe
        # to read in single-call-at-a-time usage; under concurrent calls on
        # a shared instance it's a racy snapshot (whichever call finishes
        # last wins). The actual redaction map used for restore_text()/the
        # ledger is always call-local -- see _apply_redaction().
        self.last_redacted_categories: List[str] = []
        try:
            self._redact_patterns = (
                compile_patterns(
                    redact_categories, redact_custom_patterns, redact_custom_terms, redact_patterns_file
                )
                if redact_pii else {}
            )
        except ValueError as exc:
            raise self._config_error_cls(str(exc)) from exc

        if redact_restore_in_response:
            if not redact_pii:
                raise self._config_error_cls(
                    "redact_restore_in_response requires redact_pii=True."
                )
            if redact_mode != "mask":
                raise self._config_error_cls(
                    "redact_restore_in_response requires redact_mode='mask' — with "
                    "redact_mode='block' the call never reaches the model, so there is "
                    "nothing to restore."
                )
        self.redact_restore_in_response = redact_restore_in_response

        for i, entry in enumerate(shadow_providers or []):
            if not entry.get("model"):
                raise self._config_error_cls(
                    f"shadow_providers[{i}] is missing required key 'model'."
                )
        self.shadow_providers: List[Dict[str, Any]] = list(shadow_providers or [])
        self.on_shadow_result = on_shadow_result
        self.last_shadow_results: List[Dict[str, Any]] = []
        self._shadow_sync_clients: Dict[int, Any] = {}
        self._shadow_async_clients: Dict[int, Any] = {}

        self.extra_body = extra_body
        self.store = store

    # ── Raw API calls (single client, with retry/back-off) ──────────────────────

    def _attempt_sync_create(
        self, client: Any, params: Dict[str, Any], label: str, deadline: Optional[float] = None
    ) -> Any:
        def _do_attempt() -> Any:
            self._raise_if_deadline_exceeded(deadline, label)
            return self._do_sync_create(client, params)

        def _should_retry(exc: BaseException) -> bool:
            if isinstance(exc, self._deadline_exceeded_cls):
                return False
            return getattr(exc, "status_code", None) not in _NON_RETRYABLE_STATUS_CODES

        try:
            return retry_with_backoff(
                _do_attempt, max_attempts=self.max_retries, backoff_base=self.backoff_factor,
                should_retry=_should_retry,
            )
        except Exception as exc:
            if isinstance(exc, self._deadline_exceeded_cls):
                raise
            status_code = getattr(exc, "status_code", None)
            if status_code in _NON_RETRYABLE_STATUS_CODES:
                raise self._api_error_cls(
                    f"[{label}] {self._api_name} request failed with non-retryable status "
                    f"{status_code}. Error: {type(exc).__name__}: {exc}"
                ) from exc
            raise self._api_error_cls(
                f"[{label}] {self._api_name} request failed after {self.max_retries} "
                f"attempts. Last error: {type(exc).__name__}: {exc}"
            ) from exc

    async def _attempt_async_create(
        self, client: Any, params: Dict[str, Any], label: str, deadline: Optional[float] = None
    ) -> Any:
        async def _do_attempt() -> Any:
            self._raise_if_deadline_exceeded(deadline, label)
            return await self._do_async_create(client, params)

        def _should_retry(exc: BaseException) -> bool:
            if isinstance(exc, self._deadline_exceeded_cls):
                return False
            return getattr(exc, "status_code", None) not in _NON_RETRYABLE_STATUS_CODES

        try:
            return await aretry_with_backoff(
                _do_attempt, max_attempts=self.max_retries, backoff_base=self.backoff_factor,
                should_retry=_should_retry,
            )
        except Exception as exc:
            if isinstance(exc, self._deadline_exceeded_cls):
                raise
            status_code = getattr(exc, "status_code", None)
            if status_code in _NON_RETRYABLE_STATUS_CODES:
                raise self._api_error_cls(
                    f"[{label}] Async {self._api_name} request failed with non-retryable "
                    f"status {status_code}. Error: {type(exc).__name__}: {exc}"
                ) from exc
            raise self._api_error_cls(
                f"[{label}] Async {self._api_name} request failed after "
                f"{self.max_retries} attempts. Last error: {type(exc).__name__}: {exc}"
            ) from exc
        raise self._api_error_cls(f"[{label}] Unexpected async retry exhaustion") from last_exc

    def _create_raw(self, params: Dict[str, Any]) -> Any:
        return self._attempt_sync_create(self._client, params, "primary", self._new_deadline())

    async def _acreate_raw(self, params: Dict[str, Any]) -> Any:
        return await self._attempt_async_create(self._async_client, params, "primary", self._new_deadline())

    # ── Raw API calls across primary + fallback providers ───────────────────────

    def _create_across_providers(
        self, params: Dict[str, Any], agent_deadline_seconds: Optional[float] = None
    ) -> tuple:
        """
        Try the primary, then each fallback provider in order.

        Returns (raw, label, model_name, pricing) — ``model_name`` and
        ``pricing`` describe whichever target actually answered, not always
        the primary, so callers can attribute metadata/cost correctly.

        ``agent_deadline_seconds``: see ``_new_deadline()`` -- an owning
        Agent's remaining run time, combined with ``max_call_duration`` so
        retries/fallback providers stop once either budget runs out.
        """
        deadline = self._new_deadline(agent_deadline_seconds)
        attempts: List[Any] = []
        for label, model_name, client, pricing in self._sync_targets():
            self._raise_if_deadline_exceeded(deadline, label)
            attempt_params = dict(params)
            attempt_params["model"] = model_name
            self._apply_per_target_param_guards(attempt_params, model_name)
            try:
                return (
                    self._attempt_sync_create(client, attempt_params, label, deadline),
                    label, model_name, pricing,
                )
            except self._deadline_exceeded_cls:
                raise
            except self._api_error_cls as exc:
                attempts.append((label, exc))
        if len(attempts) == 1:
            raise attempts[0][1]
        raise self._all_providers_failed_cls(
            f"All {len(attempts)} provider(s) failed: "
            + "; ".join(f"{label}: {exc}" for label, exc in attempts),
            attempts=attempts,
        )

    async def _acreate_across_providers(
        self, params: Dict[str, Any], agent_deadline_seconds: Optional[float] = None
    ) -> tuple:
        """Async counterpart of ``_create_across_providers()``. See its docstring for the return shape."""
        deadline = self._new_deadline(agent_deadline_seconds)
        attempts: List[Any] = []
        for label, model_name, client, pricing in self._async_targets():
            self._raise_if_deadline_exceeded(deadline, label)
            attempt_params = dict(params)
            attempt_params["model"] = model_name
            self._apply_per_target_param_guards(attempt_params, model_name)
            try:
                return (
                    await self._attempt_async_create(client, attempt_params, label, deadline),
                    label, model_name, pricing,
                )
            except self._deadline_exceeded_cls:
                raise
            except self._api_error_cls as exc:
                attempts.append((label, exc))
        if len(attempts) == 1:
            raise attempts[0][1]
        raise self._all_providers_failed_cls(
            f"All {len(attempts)} provider(s) failed: "
            + "; ".join(f"{label}: {exc}" for label, exc in attempts),
            attempts=attempts,
        )

    # ── Shadow-mode dual dispatch ────────────────────────────────────────────
    # Runs concurrently with (not after) the primary call. invoke()/ainvoke()
    # always return the primary's result; shadow results are observation-only.

    def _shadow_targets(self) -> Iterator[tuple]:
        """
        Yield (label, model_name, client, pricing) for each configured shadow
        provider. ``pricing`` is that shadow entry's own "input_pricing"/
        "output_pricing" keys (None if unset) — never the primary's, since a
        shadow provider is typically a different model with a different price.
        """
        for i, cfg in enumerate(self.shadow_providers):
            yield (
                f"shadow[{i}]:{cfg['model']}",
                normalize_model_name(cfg["model"]),
                self._get_shadow_sync_client(i),
                (cfg.get("input_pricing"), cfg.get("output_pricing")),
            )

    def _async_shadow_targets(self) -> Iterator[tuple]:
        """Async counterpart of ``_shadow_targets()``. See its docstring for ``pricing``."""
        for i, cfg in enumerate(self.shadow_providers):
            yield (
                f"shadow[{i}]:{cfg['model']}",
                normalize_model_name(cfg["model"]),
                self._get_shadow_async_client(i),
                (cfg.get("input_pricing"), cfg.get("output_pricing")),
            )

    def _build_shadow_result(
        self,
        label: str,
        response_text: Optional[str],
        raw_response: Any,
        primary_text: Optional[str],
        latency_ms: float,
        error: Optional[str],
        pricing: "tuple[Optional[float], Optional[float]]",
    ) -> Dict[str, Any]:
        if error is not None:
            return {
                "provider_used": label, "response": None, "similarity": None,
                "input_tokens": None, "output_tokens": None, "total_cost": None,
                "latency_ms": latency_ms, "error": error,
            }
        usage = self._extract_usage(raw_response)
        input_pricing, output_pricing = pricing
        total_cost = None
        if (
            input_pricing is not None and output_pricing is not None
            and usage["input_tokens"] is not None and usage["output_tokens"] is not None
        ):
            total_cost = (
                (usage["input_tokens"] / 1_000_000) * input_pricing
                + (usage["output_tokens"] / 1_000_000) * output_pricing
            )
        return {
            "provider_used": label,
            "response": response_text,
            "similarity": compute_similarity(primary_text, response_text),
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_cost": total_cost,
            "latency_ms": latency_ms,
            "error": None,
        }

    def _log_shadow_to_ledger(self, result: Dict[str, Any]) -> None:
        if self._ledger_conn is None:
            return
        write_shadow_ledger_entry(
            self._ledger_conn,
            self._ledger_lock,
            provider_used=result["provider_used"],
            response=result["response"] if self.ledger_store_content else None,
            similarity=result["similarity"],
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            total_cost=result["total_cost"],
            latency_ms=result["latency_ms"],
            error=result["error"],
        )

    def _finalize_shadow_results(self, raw_results: List[tuple], primary_text: Optional[str]) -> None:
        results = []
        for label, text, raw, latency_ms, error, pricing in raw_results:
            result = self._build_shadow_result(label, text, raw, primary_text, latency_ms, error, pricing)
            results.append(result)
            self._log_shadow_to_ledger(result)
            if self.on_shadow_result is not None:
                try:
                    self.on_shadow_result(result)
                except Exception:
                    self._logger.warning("on_shadow_result callback raised", exc_info=True)
        self.last_shadow_results = results

    def _execute_shadow_attempt_sync(
        self, label: str, model_name: str, client: Any, prompt_input: Any, pricing: tuple
    ) -> tuple:
        params = dict(self._build_base_params_for_call(prompt_input, stream=False))
        params["model"] = model_name
        self._apply_per_target_param_guards(params, model_name)
        start = time.perf_counter()
        try:
            raw = self._do_sync_create(client, params)
            text = self._extract_response_text(raw)
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            return label, text, raw, latency_ms, None, pricing
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            return label, None, None, latency_ms, f"{type(exc).__name__}: {exc}", pricing

    async def _execute_shadow_attempt_async(
        self, label: str, model_name: str, client: Any, prompt_input: Any, pricing: tuple
    ) -> tuple:
        params = dict(self._build_base_params_for_call(prompt_input, stream=False))
        params["model"] = model_name
        self._apply_per_target_param_guards(params, model_name)
        start = time.perf_counter()
        try:
            raw = await self._do_async_create(client, params)
            text = self._extract_response_text(raw)
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            return label, text, raw, latency_ms, None, pricing
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            return label, None, None, latency_ms, f"{type(exc).__name__}: {exc}", pricing

    def _start_shadow_sync(self, prompt_input: Any) -> Optional[tuple]:
        """
        Submit every shadow provider's request now (non-blocking) and return
        a handle to collect later via ``_finish_shadow_sync()``. Called right
        after the primary's budget admission passes and right before the
        primary's own request is sent, so both are in flight at (close to)
        the same time -- total latency ends up roughly
        ``max(primary, slowest shadow)`` instead of their sum.
        """
        if not self.shadow_providers:
            return None
        targets = list(self._shadow_targets())
        executor = ThreadPoolExecutor(max_workers=len(targets))
        futures = [
            executor.submit(self._execute_shadow_attempt_sync, label, model_name, client, prompt_input, pricing)
            for label, model_name, client, pricing in targets
        ]
        return executor, futures

    def _finish_shadow_sync(self, handle: Optional[tuple], primary_text: Optional[str]) -> None:
        """
        Collect the shadow dispatch started by ``_start_shadow_sync()``.
        ``primary_text`` may be ``None`` if the primary call itself failed --
        an already-in-flight shadow request can't be un-sent or un-billed,
        so its result is still collected and logged (with similarity=None
        in that case) rather than silently discarded.
        """
        if handle is None:
            self.last_shadow_results = []
            return
        executor, futures = handle
        raw_results = [f.result() for f in futures]
        executor.shutdown(wait=True)
        self._finalize_shadow_results(raw_results, primary_text)

    async def _gather_shadow_attempts_async(
        self, targets: List[tuple], prompt_input: Any
    ) -> List[tuple]:
        return list(await asyncio.gather(*[
            self._execute_shadow_attempt_async(label, model_name, client, prompt_input, pricing)
            for label, model_name, client, pricing in targets
        ]))

    def _start_shadow_async(self, prompt_input: Any) -> Optional["asyncio.Task"]:
        """
        Async counterpart of ``_start_shadow_sync()``. See its docstring.

        ``asyncio.create_task()`` requires an actual coroutine object, not
        the Future ``asyncio.gather()`` returns directly -- so the gather is
        wrapped in a small coroutine method to hand to it.
        """
        if not self.shadow_providers:
            return None
        targets = list(self._async_shadow_targets())
        return asyncio.create_task(self._gather_shadow_attempts_async(targets, prompt_input))

    async def _finish_shadow_async(self, task: Optional["asyncio.Task"], primary_text: Optional[str]) -> None:
        """Async counterpart of ``_finish_shadow_sync()``. See its docstring."""
        if task is None:
            self.last_shadow_results = []
            return
        raw_results = list(await task)
        self._finalize_shadow_results(raw_results, primary_text)

    # ── Streaming ─────────────────────────────────────────────────────────────
    # Fallback only kicks in if a target fails before it has emitted any chunk —
    # once partial text has reached the caller, switching providers mid-stream
    # would duplicate or corrupt output, so the error is raised as-is instead.

    def _invoke_stream_mode(
        self,
        *,
        prompt_input: Any,
        overrides: Optional[Dict[str, Any]] = None,
        usage_sink: Optional[List[Dict[str, Any]]] = None,
        agent_deadline_seconds: Optional[float] = None,
    ) -> Iterator[str]:
        """
        ``usage_sink``, if provided, is a call-local list this appends
        ``{"raw", "label", "model_name", "pricing"}`` to whenever a
        usage-bearing event is seen (see ``_extract_usage_event``).
        Only ``invoke()`` (via ``streaming=True``) passes this, to recover
        token/cost data for a streaming call; ``stream()`` leaves it ``None``
        and behaves exactly as before -- this is call-local state, never
        written to ``self``, so it can't cross-contaminate concurrent calls.
        """
        base_params = self._build_base_params_for_call(prompt_input, stream=True, overrides=overrides)
        deadline = self._new_deadline(agent_deadline_seconds)
        attempts: List[Any] = []
        for label, model_name, client, pricing in self._sync_targets():
            self._raise_if_deadline_exceeded(deadline, label)
            params = dict(base_params)
            params["model"] = model_name
            self._apply_per_target_param_guards(params, model_name)
            last_exc: Optional[Exception] = None
            for attempt in range(1, self.max_retries + 1):
                self._raise_if_deadline_exceeded(deadline, label)
                emitted = False
                refusal_chunks: List[str] = []
                try:
                    stream = self._do_sync_create(client, params)
                    for event in stream:
                        delta = self._extract_text_delta(event)
                        if delta:
                            emitted = True
                            yield delta
                        refusal_delta = self._extract_refusal_delta(event)
                        if refusal_delta:
                            refusal_chunks.append(refusal_delta)
                        if usage_sink is not None:
                            usage_event = self._extract_usage_event(event)
                            if usage_event is not None:
                                usage_sink.append({
                                    "raw": usage_event, "label": label,
                                    "model_name": model_name, "pricing": pricing,
                                })
                    if emitted:
                        return
                    if refusal_chunks:
                        # A refusal is a valid, final answer from a working provider
                        # -- raised directly (not via `last_exc`/`break`) so it
                        # propagates immediately instead of falling through to the
                        # next fallback provider, which would waste an extra call.
                        refusal_text = "".join(refusal_chunks)
                        raise self._refusal_error_cls(
                            f"[{label}] Model declined to answer: {refusal_text}",
                            refusal_text=refusal_text,
                        )
                    raise self._response_error_cls(f"[{label}] No text deltas in streaming response")
                except Exception as exc:
                    if self._refusal_error_cls is not None and isinstance(exc, self._refusal_error_cls):
                        raise
                    if isinstance(exc, self._response_error_cls):
                        if emitted:
                            raise
                        last_exc = exc
                        break
                    status_code = getattr(exc, "status_code", None)
                    if emitted:
                        raise self._api_error_cls(
                            f"[{label}] Streaming failed mid-response after emitting output: "
                            f"{type(exc).__name__}: {exc}"
                        ) from exc
                    if status_code in _NON_RETRYABLE_STATUS_CODES:
                        last_exc = self._api_error_cls(
                            f"[{label}] Streaming failed with non-retryable status {status_code}. "
                            f"Error: {type(exc).__name__}: {exc}"
                        )
                        last_exc.__cause__ = exc
                        break
                    if attempt == self.max_retries:
                        last_exc = self._api_error_cls(
                            f"[{label}] Streaming failed after {attempt} attempt(s). "
                            f"Last error: {type(exc).__name__}: {exc}"
                        )
                        last_exc.__cause__ = exc
                        break
                    time.sleep(self.backoff_factor * (2 ** (attempt - 1)))
                    continue
                break
            attempts.append((label, last_exc))
        if len(attempts) == 1:
            raise attempts[0][1]
        raise self._all_providers_failed_cls(
            f"Streaming failed on all {len(attempts)} provider(s): "
            + "; ".join(f"{lbl}: {exc}" for lbl, exc in attempts),
            attempts=attempts,
        )

    async def _ainvoke_stream_mode(
        self,
        *,
        prompt_input: Any,
        overrides: Optional[Dict[str, Any]] = None,
        usage_sink: Optional[List[Dict[str, Any]]] = None,
        agent_deadline_seconds: Optional[float] = None,
    ) -> AsyncIterator[str]:
        """Async counterpart of ``_invoke_stream_mode()``. See its docstring for ``usage_sink``."""
        base_params = self._build_base_params_for_call(prompt_input, stream=True, overrides=overrides)
        deadline = self._new_deadline(agent_deadline_seconds)
        attempts: List[Any] = []
        for label, model_name, client, pricing in self._async_targets():
            self._raise_if_deadline_exceeded(deadline, label)
            params = dict(base_params)
            params["model"] = model_name
            self._apply_per_target_param_guards(params, model_name)
            last_exc: Optional[Exception] = None
            for attempt in range(1, self.max_retries + 1):
                self._raise_if_deadline_exceeded(deadline, label)
                emitted = False
                refusal_chunks: List[str] = []
                try:
                    stream = await self._do_async_create(client, params)
                    async for event in stream:
                        delta = self._extract_text_delta(event)
                        if delta:
                            emitted = True
                            yield delta
                        refusal_delta = self._extract_refusal_delta(event)
                        if refusal_delta:
                            refusal_chunks.append(refusal_delta)
                        if usage_sink is not None:
                            usage_event = self._extract_usage_event(event)
                            if usage_event is not None:
                                usage_sink.append({
                                    "raw": usage_event, "label": label,
                                    "model_name": model_name, "pricing": pricing,
                                })
                    if emitted:
                        return
                    if refusal_chunks:
                        # See the sync version's identical comment: a refusal is a
                        # valid, final answer, so it must not fall through to the
                        # next fallback provider.
                        refusal_text = "".join(refusal_chunks)
                        raise self._refusal_error_cls(
                            f"[{label}] Model declined to answer: {refusal_text}",
                            refusal_text=refusal_text,
                        )
                    raise self._response_error_cls(f"[{label}] No text deltas in async streaming response")
                except Exception as exc:
                    if self._refusal_error_cls is not None and isinstance(exc, self._refusal_error_cls):
                        raise
                    if isinstance(exc, self._response_error_cls):
                        if emitted:
                            raise
                        last_exc = exc
                        break
                    status_code = getattr(exc, "status_code", None)
                    if emitted:
                        raise self._api_error_cls(
                            f"[{label}] Async streaming failed mid-response after emitting output: "
                            f"{type(exc).__name__}: {exc}"
                        ) from exc
                    if status_code in _NON_RETRYABLE_STATUS_CODES:
                        last_exc = self._api_error_cls(
                            f"[{label}] Async streaming failed with non-retryable status {status_code}. "
                            f"Error: {type(exc).__name__}: {exc}"
                        )
                        last_exc.__cause__ = exc
                        break
                    if attempt == self.max_retries:
                        last_exc = self._api_error_cls(
                            f"[{label}] Async streaming failed after {attempt} attempt(s). "
                            f"Last error: {type(exc).__name__}: {exc}"
                        )
                        last_exc.__cause__ = exc
                        break
                    await asyncio.sleep(self.backoff_factor * (2 ** (attempt - 1)))
                    continue
                break
            attempts.append((label, last_exc))
        if len(attempts) == 1:
            raise attempts[0][1]
        raise self._all_providers_failed_cls(
            f"Async streaming failed on all {len(attempts)} provider(s): "
            + "; ".join(f"{lbl}: {exc}" for lbl, exc in attempts),
            attempts=attempts,
        )

    # ── Call ledger ───────────────────────────────────────────────────────────

    def _log_to_ledger(
        self,
        *,
        call_type: str,
        prompt: Any,
        metadata: Dict[str, Any],
        redacted_categories: List[str],
        response_override: Any = _UNSET,
    ) -> None:
        if self._ledger_conn is None:
            return
        prompt_text = str(prompt) if (self.ledger_store_content and prompt is not None) else None
        raw_response = metadata.get("response") if response_override is _UNSET else response_override
        response_text = raw_response if self.ledger_store_content else None
        write_ledger_entry(
            self._ledger_conn,
            self._ledger_lock,
            model=metadata.get("model"),
            provider_used=metadata.get("provider_used"),
            call_type=call_type,
            prompt=prompt_text,
            response=response_text,
            metadata=metadata,
            redacted_categories=redacted_categories,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resolve_prompt(
        self,
        prompt: Any,
        prompt_variables: Optional[Dict[str, Any]],
        files: Optional[Any] = None,
    ) -> "tuple[Any, List[str], Dict[str, str]]":
        """
        Resolve prompt or render from template, then apply redaction if enabled.

        Returns (resolved_value, redacted_categories, redaction_map) — all three
        are call-local (never read from ``self``), so concurrent calls on a
        shared instance can't cross-contaminate each other's redaction state.
        See ``_apply_redaction()``.
        """
        resolved = self._resolve_prompt_raw(prompt, prompt_variables, files)
        return self._apply_redaction(resolved)

    def _invoke_non_stream_for_call(
        self, prepared_input: Any, *, overrides: Optional[Dict[str, Any]],
        agent_deadline_seconds: Optional[float] = None,
    ) -> Any:
        """Subclass one-liner forwarding to the local _invoke_non_stream(), whose keyword
        name for the prompt payload (``messages=`` vs ``input_data=``) differs per package."""
        raise NotImplementedError

    async def _ainvoke_non_stream_for_call(
        self, prepared_input: Any, *, overrides: Optional[Dict[str, Any]],
        agent_deadline_seconds: Optional[float] = None,
    ) -> Any:
        """Async counterpart of ``_invoke_non_stream_for_call()``."""
        raise NotImplementedError

    # ── Outer orchestration (invoke()/ainvoke()) ────────────────────────────────
    # Each subclass's public invoke()/ainvoke() keeps its own signature (they
    # differ -- e.g. OpenAIChatModel's image_detail= param has no OpenAIResponse
    # equivalent) and does its own prompt resolution + input-shape prep, then
    # delegates everything downstream of that to these shared methods.

    def _run_invoke(
        self,
        *,
        prepared_input: Any,
        resolved: Any,
        redacted_categories: List[str],
        redaction_map: Dict[str, str],
        overrides: Dict[str, Any],
        call_type: str = "invoke",
        force_non_stream: bool = False,
        agent_deadline_seconds: Optional[float] = None,
    ) -> Any:
        """
        ``call_type`` names this call in the ledger (e.g. "invoke", "chat").
        ``force_non_stream``, when True, always uses the non-stream path
        regardless of ``self.streaming`` -- for callers like OpenAIResponse's
        ``chat()`` that don't offer streaming at all, so a caller with
        ``streaming=True`` set on the instance still gets a real (non-stream)
        response instead of being incorrectly routed through the streaming path.
        """
        use_streaming = self.streaming and not force_non_stream
        shadow_handle = None
        response_text = None
        try:
            with self._budget_admission():
                # Started right after the authoritative budget check passes
                # and right before the primary's own request -- both are in
                # flight at (close to) the same time. Not dispatched for
                # streaming calls in this version -- same documented
                # limitation as stream()/astream().
                if not use_streaming:
                    shadow_handle = self._start_shadow_sync(prepared_input)

                with track_latency() as timing:
                    if use_streaming:
                        # usage_sink is call-local (never written to self), so
                        # concurrent streaming invoke() calls on a shared instance
                        # can't cross-contaminate each other's cost accounting.
                        usage_sink: List[Dict[str, Any]] = []
                        response_text = "".join(self._invoke_stream_mode(
                            prompt_input=prepared_input, overrides=overrides, usage_sink=usage_sink,
                            agent_deadline_seconds=agent_deadline_seconds,
                        ))
                        if usage_sink:
                            entry = usage_sink[-1]
                            raw_response = entry["raw"]
                            provider_label = entry["label"]
                            provider_model = entry["model_name"]
                            provider_pricing = entry["pricing"]
                        else:
                            # Provider didn't return a usage-bearing/terminal
                            # event (e.g. some self-hosted/proxy servers, or a
                            # Responses API stream with no final response) --
                            # degrade gracefully to no usage/cost data.
                            raw_response = None
                            provider_label = "primary"
                            provider_model = self._model_name
                            provider_pricing = (self.input_pricing, self.output_pricing)
                    else:
                        # agent_deadline_seconds is only ever passed as a kwarg
                        # when set -- a subclass's _invoke_non_stream_for_call()
                        # override that predates this parameter (e.g.
                        # autourgos-responses' OpenAIResponse) has no such
                        # parameter in its signature at all, so passing it
                        # explicitly even as None would raise TypeError for
                        # every call, not just ones that use this feature.
                        _extra = {"agent_deadline_seconds": agent_deadline_seconds} if agent_deadline_seconds is not None else {}
                        response_text, raw_response, provider_label, provider_model, provider_pricing = (
                            self._invoke_non_stream_for_call(prepared_input, overrides=overrides, **_extra)
                        )

                masked_response_text = response_text
                if self.redact_restore_in_response:
                    response_text = restore_text(response_text, redaction_map)

                metadata = build_structured_output(
                    model_name=provider_model,
                    response_text=response_text,
                    raw_response=raw_response,
                    latency_ms=timing["latency_ms"],
                    input_pricing=provider_pricing[0],
                    output_pricing=provider_pricing[1],
                    extra_fields={"provider_used": provider_label},
                )
                self.last_metadata = metadata
                self._record_session_cost(metadata.get("total_cost"))
                self._log_to_ledger(
                    call_type=call_type, prompt=resolved, metadata=metadata,
                    redacted_categories=redacted_categories, response_override=masked_response_text,
                )
        finally:
            # Shadow dispatch is observation-only and its cost is never
            # counted toward max_session_cost -- collected here, after the
            # admission lock is released (so it doesn't hold up other calls'
            # admission checks), and unconditionally (even if the primary
            # call raised) since an already-in-flight shadow request can't
            # be un-sent or un-billed -- its result is still logged, just
            # with similarity=None since there's no primary text to compare.
            if shadow_handle is not None:
                self._finish_shadow_sync(shadow_handle, response_text)
        return metadata if self.structured_output else response_text

    async def _arun_invoke(
        self,
        *,
        prepared_input: Any,
        resolved: Any,
        redacted_categories: List[str],
        redaction_map: Dict[str, str],
        overrides: Dict[str, Any],
        call_type: str = "ainvoke",
        force_non_stream: bool = False,
        agent_deadline_seconds: Optional[float] = None,
    ) -> Any:
        """Async counterpart of ``_run_invoke()``. See its docstring/comments for rationale."""
        use_streaming = self.streaming and not force_non_stream
        shadow_task = None
        response_text = None
        try:
            async with self._async_budget_admission():
                if not use_streaming:
                    shadow_task = self._start_shadow_async(prepared_input)

                with track_latency() as timing:
                    if use_streaming:
                        usage_sink: List[Dict[str, Any]] = []
                        chunks: List[str] = []
                        async for delta in self._ainvoke_stream_mode(
                            prompt_input=prepared_input, overrides=overrides, usage_sink=usage_sink,
                            agent_deadline_seconds=agent_deadline_seconds,
                        ):
                            chunks.append(delta)
                        response_text = "".join(chunks)
                        if usage_sink:
                            entry = usage_sink[-1]
                            raw_response = entry["raw"]
                            provider_label = entry["label"]
                            provider_model = entry["model_name"]
                            provider_pricing = entry["pricing"]
                        else:
                            raw_response = None
                            provider_label = "primary"
                            provider_model = self._model_name
                            provider_pricing = (self.input_pricing, self.output_pricing)
                    else:
                        # See _run_invoke's identical comment: only passed when set.
                        _extra = {"agent_deadline_seconds": agent_deadline_seconds} if agent_deadline_seconds is not None else {}
                        response_text, raw_response, provider_label, provider_model, provider_pricing = (
                            await self._ainvoke_non_stream_for_call(prepared_input, overrides=overrides, **_extra)
                        )

                masked_response_text = response_text
                if self.redact_restore_in_response:
                    response_text = restore_text(response_text, redaction_map)

                metadata = build_structured_output(
                    model_name=provider_model,
                    response_text=response_text,
                    raw_response=raw_response,
                    latency_ms=timing["latency_ms"],
                    input_pricing=provider_pricing[0],
                    output_pricing=provider_pricing[1],
                    extra_fields={"provider_used": provider_label},
                )
                self.last_metadata = metadata
                self._record_session_cost(metadata.get("total_cost"))
                self._log_to_ledger(
                    call_type=call_type, prompt=resolved, metadata=metadata,
                    redacted_categories=redacted_categories, response_override=masked_response_text,
                )
        finally:
            # See _run_invoke()'s finally block for why this runs unconditionally,
            # after the admission lock is released.
            if shadow_task is not None:
                await self._finish_shadow_async(shadow_task, response_text)
        return metadata if self.structured_output else response_text

    # ── Validated structured output ──────────────────────────────────────────
    # Server-side json_schema strict mode (build_structured_output's caller-side
    # setup) already constrains the shape of the JSON. This adds a feedback
    # loop on top: if the result still fails Pydantic validation (e.g. a
    # custom @field_validator, or a provider that ignores strict mode), the
    # error is sent back to the model and it gets another chance to correct
    # itself.

    @staticmethod
    def _is_pydantic_model_class(obj: Any) -> bool:
        return isinstance(obj, type) and hasattr(obj, "model_validate_json")

    def _require_structured_schema(self) -> None:
        if not self._is_pydantic_model_class(self.output_schema):
            raise self._config_error_cls(
                "invoke_structured()/ainvoke_structured() require output_schema= to be a "
                "Pydantic BaseModel class (not a plain dict or None)."
            )
        if self.streaming:
            raise self._config_error_cls(
                "invoke_structured()/ainvoke_structured() are incompatible with streaming=True."
            )

    @staticmethod
    def _correction_messages(bad_text: str, error: Exception) -> List[Dict[str, Any]]:
        return [
            {"role": "assistant", "content": bad_text},
            {
                "role": "user",
                "content": (
                    "Your last response failed schema validation with this error:\n"
                    f"{error}\n\nReturn corrected JSON that matches the schema exactly."
                ),
            },
        ]

    def _run_invoke_structured(
        self,
        *,
        prepared_input: Any,
        resolved: Any,
        redacted_categories: List[str],
        redaction_map: Dict[str, str],
        max_validation_retries: int,
    ) -> Any:
        with self._budget_admission():
            last_error: Optional[Exception] = None
            last_text: Optional[str] = None
            for attempt in range(max_validation_retries + 1):
                with track_latency() as timing:
                    response_text, raw_response, provider_label, provider_model, provider_pricing = (
                        self._invoke_non_stream_for_call(prepared_input, overrides=None)
                    )
                masked_response_text = response_text
                text_to_validate = response_text
                if self.redact_restore_in_response:
                    text_to_validate = restore_text(response_text, redaction_map)
                last_text = text_to_validate
                try:
                    validated = self.output_schema.model_validate_json(text_to_validate)
                except Exception as exc:
                    last_error = exc
                    # Feed back the model's own (still-masked) output — never the restored
                    # text, which would leak the real secret into the model's own context.
                    prepared_input = prepared_input + self._correction_messages(masked_response_text, exc)
                    continue
                self.last_metadata = build_structured_output(
                    model_name=provider_model,
                    response_text=text_to_validate,
                    raw_response=raw_response,
                    latency_ms=timing["latency_ms"],
                    input_pricing=provider_pricing[0],
                    output_pricing=provider_pricing[1],
                    extra_fields={"provider_used": provider_label, "validation_retries": attempt},
                )
                self._record_session_cost(self.last_metadata.get("total_cost"))
                self._log_to_ledger(
                    call_type="invoke_structured", prompt=resolved, metadata=self.last_metadata,
                    redacted_categories=redacted_categories, response_override=masked_response_text,
                )
                return validated

            raise self._validation_error_cls(
                f"Output failed schema validation after {max_validation_retries + 1} attempt(s). "
                f"Last error: {last_error}",
                raw_text=last_text,
                validation_error=last_error,
            )

    async def _arun_invoke_structured(
        self,
        *,
        prepared_input: Any,
        resolved: Any,
        redacted_categories: List[str],
        redaction_map: Dict[str, str],
        max_validation_retries: int,
    ) -> Any:
        """Async counterpart of ``_run_invoke_structured()``. See its docstring for rationale."""
        async with self._async_budget_admission():
            last_error: Optional[Exception] = None
            last_text: Optional[str] = None
            for attempt in range(max_validation_retries + 1):
                with track_latency() as timing:
                    response_text, raw_response, provider_label, provider_model, provider_pricing = (
                        await self._ainvoke_non_stream_for_call(prepared_input, overrides=None)
                    )
                masked_response_text = response_text
                text_to_validate = response_text
                if self.redact_restore_in_response:
                    text_to_validate = restore_text(response_text, redaction_map)
                last_text = text_to_validate
                try:
                    validated = self.output_schema.model_validate_json(text_to_validate)
                except Exception as exc:
                    last_error = exc
                    prepared_input = prepared_input + self._correction_messages(masked_response_text, exc)
                    continue
                self.last_metadata = build_structured_output(
                    model_name=provider_model,
                    response_text=text_to_validate,
                    raw_response=raw_response,
                    latency_ms=timing["latency_ms"],
                    input_pricing=provider_pricing[0],
                    output_pricing=provider_pricing[1],
                    extra_fields={"provider_used": provider_label, "validation_retries": attempt},
                )
                self._record_session_cost(self.last_metadata.get("total_cost"))
                self._log_to_ledger(
                    call_type="ainvoke_structured", prompt=resolved, metadata=self.last_metadata,
                    redacted_categories=redacted_categories, response_override=masked_response_text,
                )
                return validated

            raise self._validation_error_cls(
                f"Output failed schema validation after {max_validation_retries + 1} attempt(s). "
                f"Last error: {last_error}",
                raw_text=last_text,
                validation_error=last_error,
            )

    # ── Low-level create() / acreate() ───────────────────────────────────────

    def _prepare_create_params(self, input_data: Any, overrides: Dict[str, Any]) -> Dict[str, Any]:
        if input_data is None:
            input_data = overrides.pop(self._create_input_key, None)
        if input_data is None:
            raise ValueError(self._create_missing_input_message)
        params = self._build_base_params_for_call(input_data, stream=False)
        # The input key/"stream" stay structurally managed -- an override can't
        # silently replace the already-validated input_data or flip this into
        # a streaming call (_create_raw()'s retry logic only handles a
        # non-streaming response object). "model" stays overridable.
        params.update({k: v for k, v in overrides.items() if k not in (self._create_input_key, "stream")})
        self._apply_create_param_guards(params, params.get("model", self._model_name))
        return params


__all__ = [
    "BaseLLM",
    "BaseProviderLLM",
    "FunctionCall",
    "ToolCallResponse",
    "CircuitBreakerOpenException",
    "BudgetExceededException",
    "NonTransientError",
]
