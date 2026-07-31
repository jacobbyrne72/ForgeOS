"""Prompt prefix caching — deterministic, cache-hit responses for free."""
from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock

from ._sqlite import connect as _sql_connect

MAX_CACHE = 1000
TTL = 24 * 60 * 60


@dataclass(frozen=True, slots=True)
class CacheEntry:
    prompt_hash: str
    provider: str
    model: str
    response: str
    cost_usd: float = 0.0
    tokens_out: int = 0


class PromptCache:
    def __init__(self, db_path: Path | None = None):
        self._db_path = db_path or Path.home() / ".forgeos" / "prompt_cache.db"
        self._lock = RLock()
        self._conn = _sql_connect(self._db_path)
        self._init_db()

    def _init_db(self) -> None:
        conn = self._conn._conn if hasattr(self._conn, '_conn') else self._conn
        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS prompt_cache ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "prompt_hash TEXT NOT NULL,"
            "provider TEXT NOT NULL,"
            "model TEXT NOT NULL,"
            "response TEXT NOT NULL,"
            "cost_usd REAL DEFAULT 0.0,"
            "tokens_out INTEGER DEFAULT 0,"
            "created_at REAL,"
            "expires_at REAL"
            ")",
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_lookup "
            "ON prompt_cache(prompt_hash, provider, model)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_expires "
            "ON prompt_cache(expires_at)"
        )
        cur.execute(
            "DELETE FROM prompt_cache WHERE expires_at < strftime('%s','now')"
        )
        conn.commit()

    def lookup(self, provider: str, model: str, prompt: str) -> CacheEntry | None:
        h = hashlib.sha256(prompt.encode()).hexdigest()
        with self._lock:
            conn = self._conn._conn if hasattr(self._conn, '_conn') else self._conn
            row = conn.execute(
                "SELECT response, cost_usd, tokens_out FROM prompt_cache "
                "WHERE prompt_hash = ? AND provider = ? AND model = ? AND expires_at > strftime('%s','now')",
                (h, provider, model),
            ).fetchone()
            if row is None:
                return None
            return CacheEntry(
                prompt_hash=h, provider=provider, model=model,
                response=row[0], cost_usd=row[1], tokens_out=row[2],
            )

    def store(
        self, provider: str, model: str, prompt: str, response: str,
        cost_usd: float = 0.03, tokens_out: int = 0,
    ) -> None:
        h = hashlib.sha256(prompt.encode()).hexdigest()
        with self._lock:
            conn = self._conn._conn if hasattr(self._conn, '_conn') else self._conn
            conn.execute(
                "INSERT OR REPLACE INTO prompt_cache "
                "(prompt_hash, provider, model, response, cost_usd, tokens_out, created_at, expires_at) "
                "VALUES (?,?,?,?,?,?,strftime('%s','now'),strftime('%s','now')+?)",
                (h, provider, model, response, cost_usd, tokens_out, TTL),
            )
            conn.commit()
            self._purge()

    def _purge(self) -> None:
        conn = self._conn._conn if hasattr(self._conn, '_conn') else self._conn
        conn.execute("DELETE FROM prompt_cache WHERE expires_at < strftime('%s','now')")
        count = conn.execute("SELECT COUNT(*) FROM prompt_cache").fetchone()[0]
        if count > MAX_CACHE:
            conn.execute(
                "DELETE FROM prompt_cache WHERE id NOT IN "
                "(SELECT id FROM prompt_cache ORDER BY created_at DESC LIMIT ?)",
                (MAX_CACHE,),
            )
        conn.commit()

    def size(self) -> int:
        conn = self._conn._conn if hasattr(self._conn, '_conn') else self._conn
        return conn.execute("SELECT COUNT(*) FROM prompt_cache").fetchone()[0]

    def total_saved(self) -> dict:
        conn = self._conn._conn if hasattr(self._conn, '_conn') else self._conn
        rows = conn.execute(
            "SELECT cost_usd, tokens_out FROM prompt_cache"
        ).fetchall()
        return {
            "total_saved": round(sum(r[0] for r in rows), 6),
            "total_tokens_saved": sum(r[1] for r in rows),
            "entries": len(rows),
        }

    def clear(self) -> int:
        with self._lock:
            conn = self._conn._conn if hasattr(self._conn, '_conn') else self._conn
            cur = conn.execute("DELETE FROM prompt_cache")
            conn.commit()
            return cur.rowcount
