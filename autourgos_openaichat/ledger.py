"""
Local call ledger for autourgos-openaichat.

A SQLite-backed audit trail: every invoke()/ainvoke()/invoke_structured()/
ainvoke_structured() call can optionally be recorded to a local .db file —
no external service, no extra dependency (sqlite3 is stdlib).

Opt-in only: nothing here runs unless a caller passes ledger_path=.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    model TEXT,
    provider_used TEXT,
    call_type TEXT,
    prompt TEXT,
    response TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER,
    input_cost REAL,
    output_cost REAL,
    total_cost REAL,
    latency_ms REAL,
    validation_retries INTEGER,
    redacted_categories TEXT
)
"""

_SHADOW_SCHEMA = """
CREATE TABLE IF NOT EXISTS shadow_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    provider_used TEXT,
    response TEXT,
    similarity REAL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_cost REAL,
    latency_ms REAL,
    error TEXT
)
"""

_COLUMNS = (
    "created_at", "model", "provider_used", "call_type", "prompt", "response",
    "input_tokens", "output_tokens", "total_tokens",
    "input_cost", "output_cost", "total_cost",
    "latency_ms", "validation_retries", "redacted_categories",
)

_INSERT_SQL = (
    f"INSERT INTO calls ({', '.join(_COLUMNS)}) "
    f"VALUES ({', '.join('?' for _ in _COLUMNS)})"
)

_SHADOW_COLUMNS = (
    "created_at", "provider_used", "response", "similarity",
    "input_tokens", "output_tokens", "total_cost", "latency_ms", "error",
)

_SHADOW_INSERT_SQL = (
    f"INSERT INTO shadow_calls ({', '.join(_SHADOW_COLUMNS)}) "
    f"VALUES ({', '.join('?' for _ in _SHADOW_COLUMNS)})"
)


def open_ledger(path: str) -> sqlite3.Connection:
    """Open (creating if needed) the SQLite ledger file at ``path``."""
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute(_SCHEMA)
    conn.execute(_SHADOW_SCHEMA)
    conn.commit()
    return conn


def write_ledger_entry(
    conn: sqlite3.Connection,
    lock: threading.Lock,
    *,
    model: Optional[str],
    provider_used: Optional[str],
    call_type: str,
    prompt: Optional[str],
    response: Optional[str],
    metadata: Dict[str, Any],
    redacted_categories: Optional[List[str]] = None,
) -> None:
    """
    Best-effort insert of one call record. Never raises — a ledger write
    failure (disk full, permissions, closed connection, ...) must not break
    the caller's actual LLM call, so failures are logged and swallowed.
    """
    row = (
        datetime.now(timezone.utc).isoformat(),
        model,
        provider_used,
        call_type,
        prompt,
        response,
        metadata.get("input_tokens"),
        metadata.get("output_tokens"),
        metadata.get("total_tokens"),
        metadata.get("input_cost"),
        metadata.get("output_cost"),
        metadata.get("total_cost"),
        metadata.get("latency_ms"),
        metadata.get("validation_retries"),
        ",".join(redacted_categories) if redacted_categories else None,
    )
    try:
        with lock:
            conn.execute(_INSERT_SQL, row)
            conn.commit()
    except Exception:
        logger.warning("Failed to write call ledger entry", exc_info=True)


def write_shadow_ledger_entry(
    conn: sqlite3.Connection,
    lock: threading.Lock,
    *,
    provider_used: str,
    response: Optional[str],
    similarity: Optional[float],
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    total_cost: Optional[float],
    latency_ms: Optional[float],
    error: Optional[str],
) -> None:
    """Best-effort insert of one shadow-dispatch result. Never raises."""
    row = (
        datetime.now(timezone.utc).isoformat(),
        provider_used,
        response,
        similarity,
        input_tokens,
        output_tokens,
        total_cost,
        latency_ms,
        error,
    )
    try:
        with lock:
            conn.execute(_SHADOW_INSERT_SQL, row)
            conn.commit()
    except Exception:
        logger.warning("Failed to write shadow ledger entry", exc_info=True)


def close_ledger(conn: Optional[sqlite3.Connection]) -> None:
    """Close the ledger connection, if any, swallowing close-time errors."""
    if conn is None:
        return
    try:
        conn.close()
    except Exception:
        logger.warning("Failed to close call ledger connection", exc_info=True)


__all__ = [
    "logger",
    "open_ledger",
    "write_ledger_entry",
    "write_shadow_ledger_entry",
    "close_ledger",
]
