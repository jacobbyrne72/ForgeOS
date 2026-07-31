"""Cache purge - clean up prompt cache to free resources."""
from __future__ import annotations

from .prompt_cache import PromptCache


class CachePurge:
    """Purge stale cache entries."""

    def __init__(self, cache: PromptCache):
        self.cache = cache

    def purge_expired(self) -> dict:
        """Remove expired entries."""
        stats = self.cache.stats()
        expired = stats.get("expired", 0)
        self.cache.clear()
        return dict(
            action="purge_expired",
            evicted=expired,
            remaining=0,
        )

    def purge_all(self) -> dict:
        """Purge entire cache."""
        self.cache.clear()
        return dict(
            action="purge_all",
            evicted=1,
            remaining=0,
        )

    def stats(self) -> dict:
        return self.cache.stats()

