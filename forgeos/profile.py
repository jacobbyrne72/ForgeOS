"""Model performance profiling — measure actual cost and latency per model."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ModelProfile:
    model: str
    provider: str = ""
    calls: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cost_usd_micros: int = 0
    total_latency_ms: float = 0.0
    successes: int = 0
    failures: int = 0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.calls, 1)

    @property
    def cost_per_1k_tokens(self) -> float:
        total = self.total_tokens_in + self.total_tokens_out
        return (self.total_cost_usd_micros / 1000) / max(total, 1)

    @property
    def success_rate(self) -> float:
        n = self.successes + self.failures
        return self.successes / n if n > 0 else 0.0

    @property
    def cost_per_task(self) -> float:
        return self.total_cost_usd_micros / 1_000_000 / max(self.calls, 1)

class ModelProfiler:
    """Accumulates per-model performance data from completed tasks."""

    def __init__(self):
        self._profiles: dict[str, ModelProfile] = {}

    def record(self, model, provider, tokens_in, tokens_out, cost_usd_micros, latency_ms, success):
        p = self._profiles.setdefault(model, ModelProfile(model=model, provider=provider))
        p.calls += 1
        p.total_tokens_in += tokens_in
        p.total_tokens_out += tokens_out
        p.total_cost_usd_micros += cost_usd_micros
        p.total_latency_ms += latency_ms
        if success:
            p.successes += 1
        else:
            p.failures += 1

    def profile(self, model):
        return self._profiles.get(model)

    def all_profiles(self):
        return dict(self._profiles)

    def cheapest(self, min_calls=3):
        cands = [p for p in self._profiles.values() if p.calls >= min_calls]
        if not cands:
            return None
        cands.sort(key=lambda p: p.cost_per_task)
        return cands[0]

    def fastest(self, min_calls=3):
        cands = [p for p in self._profiles.values() if p.calls >= min_calls]
        if not cands:
            return None
        cands.sort(key=lambda p: p.avg_latency_ms)
        return cands[0]

    def best_value(self, min_calls=3):
        cands = [p for p in self._profiles.values() if p.calls >= min_calls]
        if not cands:
            return None
        cands.sort(key=lambda p: p.cost_per_task / max(p.avg_latency_ms, 0.001))
        return cands[0]

    def summary(self):
        lines = ["Model profiles:", "=" * 50]
        for model, p in sorted(self._profiles.items(), key=lambda x: x[1].cost_per_task):
            lines.append(f"  {model:<30} ${p.cost_per_task:.4f}/task  {p.avg_latency_ms:.0f}ms  {p.success_rate:.0%} ok  ({p.calls} calls)")
        return chr(10).join(lines)
