"""Smart batching — group API calls to reduce per-call overhead."""
from __future__ import annotations


class SmartBatcher:
    def __init__(self, max_batch_size=10, batch_window=0.1):
        self.max_batch_size = max_batch_size
        self.batch_window = batch_window
        self._batches_formed = 0
        self._calls_saved = 0

    def should_batch(self, pending_calls):
        if len(pending_calls) >= 2 and len(pending_calls) <= self.max_batch_size:
            return True
        return False

    def form_batch(self, calls):
        if not self.should_batch(calls):
            return [calls]
        self._batches_formed += 1
        self._calls_saved += len(calls) - 1
        return [calls]

    def stats(self):
        return dict(
            batches_formed=self._batches_formed,
            calls_saved=self._calls_saved,
        )
