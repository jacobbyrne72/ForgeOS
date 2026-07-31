"""Result cache — persistent dedup for completed API responses."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from threading import Lock


class ResultCache:
    """Cache completed API responses by content hash."""

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
