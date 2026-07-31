"""Dedup and cache API calls — three distinct, complementary techniques.

Consolidates result_cache + request_coalescer + smart_dedup (2026-07-31).
Unlike the other five consolidated groups, these are NOT duplicates of one
another — each answers a different question, at a different layer:

- ResultCache: "have we ever seen this exact (prompt, model) before?" —
  exact-hash match, persisted to disk, survives across process restarts.
- RequestCoalescer: "is this exact call already in flight right now?" —
  exact key match, in-memory only, concurrent-request de-duplication via a
  threading.Event; has no notion of history beyond the current moment.
- SmartDedup: "is this similar enough to something recent?" — fuzzy Jaccard
  similarity over tokenized text, in-memory, windowed to the last 50 calls
  per model; catches near-duplicates ResultCache's exact hash would miss.

They were merged into one file anyway because none of the three was ever
wired into forgeos/__init__.py or forge.py/cli.py/gateway, none is used by
another module, and none has automated test coverage — three unmaintained
single-class files is needless surface area even though the classes
themselves are legitimately different. Each class's public API is preserved
exactly as it was in its original module.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from threading import Lock


class ResultCache:
    """Cache completed API responses by content hash. (from result_cache.py)"""

    def __init__(self, db_path: Path | None = None):
        self._path = db_path or Path.home() / ".forgeos" / "result_cache.json"
        self._lock = Lock()
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text())
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2))

    def _hash(self, prompt: str, model: str) -> str:
        key = f"{model}:{prompt}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def get(self, prompt: str, model: str) -> dict | None:
        h = self._hash(prompt, model)
        with self._lock:
            entry = self._data.get(h)
        if entry:
            return dict(**entry, cached=True)
        return None

    def store(self, prompt: str, model: str, response: str, cost_usd: float):
        h = self._hash(prompt, model)
        with self._lock:
            self._data[h] = dict(
                prompt_hash=h, model=model, response=response,
                cost_usd=cost_usd, tokens_out=max(1, len(response) // 4),
            )
            if len(self._data) > 10000:
                # Evict oldest 20%
                keys = list(self._data.keys())[:2000]
                for k in keys:
                    del self._data[k]
            self._save()

    def size(self) -> int:
        with self._lock:
            return len(self._data)

    def total_saved(self) -> float:
        with self._lock:
            return sum(e.get("cost_usd", 0) for e in self._data.values())


class RequestCoalescer:
    """Deduplicate in-flight API requests. (from request_coalescer.py)

    If two identical requests are made concurrently, only one hits the API."""

    def __init__(self):
        self._lock = threading.Lock()
        self._inflight: dict[str, threading.Event] = {}
        self._results: dict[str, str] = {}
        self._deduped = 0
        self._total = 0

    def acquire(self, key: str) -> bool:
        """Return True if this caller should make the API call, False if waiting."""
        self._total += 1
        with self._lock:
            if key in self._inflight:
                # Another request is in-flight for this key — wait for it
                self._deduped += 1
                return False
            # First request for this key
            event = threading.Event()
            self._inflight[key] = event
            return True

    def complete(self, key: str, result: str) -> None:
        """Store result and signal waiters."""
        with self._lock:
            self._results[key] = result
            event = self._inflight.pop(key, None)
            if event:
                event.set()

    def wait_and_get(self, key: str, timeout: float = 30.0) -> str | None:
        """Wait for a concurrent request to complete and return its result."""
        with self._lock:
            event = self._inflight.get(key)
        if event and event.wait(timeout=timeout):
            with self._lock:
                return self._results.get(key)
        return None

    def stats(self) -> dict:
        return dict(
            total_requests=self._total,
            deduped=self._deduped,
            savings_pct=round(self._deduped / max(1, self._total) * 100, 1),
        )


class SmartDedup:
    """Fuzzy dedup of API requests via Jaccard token similarity. (from smart_dedup.py)"""

    def __init__(self, similarity_threshold=0.85):
        self.threshold = similarity_threshold
        self._calls = []
        self._deduped_calls = 0

    def _tokenize(self, text):
        return frozenset(text.lower().split())

    def _jaccard(self, a, b):
        if not a and not b:
            return 1.0
        return len(a & b) / max(1, len(a | b))

    def is_duplicate(self, prompt, model):
        tokens = self._tokenize(prompt)
        for prev in self._calls[-50:]:
            if prev["model"] != model:
                continue
            sim = self._jaccard(tokens, prev["tokens"])
            if sim >= self.threshold:
                self._deduped_calls += 1
                return True, sim
        self._calls.append(dict(tokens=tokens, model=model))
        return False, 0.0

    def stats(self):
        total = len(self._calls) + self._deduped_calls
        return dict(
            total_calls=total,
            deduped=self._deduped_calls,
            savings_pct=round(self._deduped_calls / max(1, total) * 100, 1),
        )
