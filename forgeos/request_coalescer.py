"""Coalesce duplicate requests in-flight to avoid redundant API calls."""
from __future__ import annotations
import threading


class RequestCoalescer:
    """Deduplicate in-flight API requests.

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
                event = self._inflight[key]
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
