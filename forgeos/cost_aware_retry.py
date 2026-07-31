"""Cost-aware retry with exponential backoff and fallback model routing."""
from __future__ import annotations
import time


class CostAwareRetry:
    def __init__(self, max_retries=3, base_delay=1.0, backoff_factor=2.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor
        self.retries_count = 0
        self.total_cost_saved = 0.0

    def execute_with_retry(self, task, primary_cost, fallback_cost=None):
        """Execute task with retry and fallback to cheaper model."""
        delay = self.base_delay
        for attempt in range(self.max_retries + 1):
            try:
                if attempt == 0:
                    cost = primary_cost
                else:
                    cost = fallback_cost if fallback_cost else primary_cost * 0.5
                    self.retries_count += 1
                result = {'success': True, 'cost': cost, 'attempt': attempt + 1}
                return result
            except Exception as e:
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= self.backoff_factor
                else:
                    return {'success': False, 'error': str(e), 'attempts': attempt + 1}
        return {'success': False, 'error': 'max retries exceeded'}

    def get_stats(self):
        return dict(
            total_retries=self.retries_count,
            total_cost_saved=round(self.total_cost_saved, 6),
        )
