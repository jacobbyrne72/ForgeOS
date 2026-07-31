"""Compact multi-turn history."""
from __future__ import annotations

class CompactMultiTurn:
    def __init__(self, max_turns=10, max_tokens=2000):
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self._compactions = 0
        self._tokens_saved = 0

    def compact(self, history):
        recent = history[-self.max_turns:]
        older = history[:-self.max_turns] if len(history) > self.max_turns else []
        if older:
            summary = self._summarize(older)
            self._compactions += 1
            old_tokens = sum(len(t.get("content", "")) // 4 for t in older)
            new_tokens = max(1, len(summary) // 4)
            self._tokens_saved += max(0, old_tokens - new_tokens)
            recent.insert(0, dict(role="system", content=summary, compacted=True))
        return recent

    def _summarize(self, turns):
        return "Earlier " + str(len(turns)) + " turns summarized."

    def stats(self):
        return dict(compactions=self._compactions, tokens_saved=self._tokens_saved)
