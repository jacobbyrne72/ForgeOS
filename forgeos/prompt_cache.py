"""Prompt prefix caching — SQLite-backed LRU with TTL."""

from __future__ import annotations
import hashlib
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

DEFAULT_CACHE_SIZE = 1000
DEFAULT_TTL_SECONDS = 3600.0
CACHE_DB_NAME = "prompt_cache.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS prompt_cache (
    key TEXT PRIMARY KEY,
    prompt_sha256 TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    response TEXT NOT NULL,
    tokens_in INTEGER NOT NULL DEFAULT 0,
    tokens_out INTEGER NOT NULL DEFAULT 0,
    usd_micros INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    hit_count INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_cache_provider ON prompt_cache(provider, model);
CREATE INDEX IF NOT EXISTS idx_cache_expires ON prompt_cache(expires_at);
"""

CLEANUP_SQL = "DELETE FROM prompt_cache WHERE expires_at < ?"


@dataclass
class CacheEntry:
    key: str
    provider: str
    model: str
    response: str
    tokens_in: int = 0
    tokens_out: int = 0
    usd_micros: int = 0
    created_at: float = 0.0
    expires_at: float = 0.0
    hit_count: int = 1

    def is_expired(self, at: float | None = None) -> bool:
        return (at or time.time()) >= self.expires_at


class PromptCache:
    def __init__(
        self,
        *,
        home: Path | None = None,
        max_size: int = DEFAULT_CACHE_SIZE,
        default_ttl: float = DEFAULT_TTL_SECONDS,
        cleanup_threshold: int = 100,
    ):
        self.home = home or Path.home() / ".forgeos"
        self.home.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cleanup_threshold = cleanup_threshold
        self._write_count = 0
        self._path = self.home / CACHE_DB_NAME
        self._conn = sqlite3.connect(str(self._path))
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self._lock = RLock()
        self._purge_expired()

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _make_key(provider: str, model: str, prompt: str) -> str:
        raw = f"\x00{provider}\x00{model}\x00{prompt}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, provider: str, model: str, prompt: str) -> CacheEntry | None:
        key = self._make_key(provider, model, prompt)
        with self._lock:
            row = self._conn.execute(
                "SELECT response, tokens_in, tokens_out, usd_micros, expires_at, hit_count "
                "FROM prompt_cache WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        response, tokens_in, tokens_out, usd_micros, expires_at, hit_count = row
        if expires_at < time.time():
            self._delete(key)
            return None
        with self._lock:
            self._conn.execute(
                "UPDATE prompt_cache SET hit_count = ? WHERE key = ?",
                (hit_count + 1, key),
            )
            self._conn.commit()
        entry = CacheEntry(
            key=key,
            provider=provider,
            model=model,
            response=response,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            usd_micros=usd_micros,
            expires_at=expires_at,
        )
        entry.hit_count = hit_count + 1
        return entry

    def put(
        self,
        provider: str,
        model: str,
        prompt: str,
        response: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        usd_micros: int = 0,
        ttl_seconds: float | None = None,
    ) -> None:
        key = self._make_key(provider, model, prompt)
        ttl = ttl_seconds or self.default_ttl
        now = time.time()
        entry = CacheEntry(
            key=key,
            provider=provider,
            model=model,
            response=response,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            usd_micros=usd_micros,
            created_at=now,
            expires_at=now + ttl,
        )
        with self._lock:
            count = self._conn.execute("SELECT COUNT(*) FROM prompt_cache").fetchone()[0]
            if count >= self.max_size:
                self._conn.execute(
                    "DELETE FROM prompt_cache WHERE key IN "
                    "(SELECT key FROM prompt_cache ORDER BY hit_count ASC, created_at ASC LIMIT 1)"
                )
            self._conn.execute(
                "INSERT OR REPLACE INTO prompt_cache "
                "(key, prompt_sha256, provider, model, response, "
                "tokens_in, tokens_out, usd_micros, created_at, expires_at, hit_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)",
                (
                    key,
                    key,
                    provider,
                    model,
                    response,
                    tokens_in,
                    tokens_out,
                    usd_micros,
                    entry.created_at,
                    entry.expires_at,
                ),
            )
            self._conn.commit()
            self._write_count += 1
            if self._write_count % self.cleanup_threshold == 0:
                self._purge_expired()

    def _delete(self, key: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM prompt_cache WHERE key = ?", (key,))
            self._conn.commit()

    def _purge_expired(self) -> None:
        with self._lock:
            now = time.time()
            self._conn.execute(CLEANUP_SQL, (now,))
            # Also purge expired in bulk — more thorough cleanup
            self._conn.execute("DELETE FROM prompt_cache WHERE expires_at < ?", (now,))
            self._conn.commit()

    def stats(self) -> dict:
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM prompt_cache").fetchone()[0]
            expired = self._conn.execute(
                "SELECT COUNT(*) FROM prompt_cache WHERE expires_at < ?", (time.time(),)
            ).fetchone()[0]
            hits = self._conn.execute("SELECT SUM(hit_count) FROM prompt_cache").fetchone()[0] or 0
            return {
                "entries": total,
                "expired": expired,
                "total_hits": hits,
                "max_size": self.max_size,
                "default_ttl_seconds": self.default_ttl,
                "utilization_pct": round(total / self.max_size * 100, 1) if self.max_size else 0,
            }

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM prompt_cache")
            self._conn.commit()
