"""Smart dedup - fuzzy dedup of API requests."""
from __future__ import annotations


class SmartDedup:
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
