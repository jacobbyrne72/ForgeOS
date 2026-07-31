"""Adaptive adapter selection — picks the cheapest adapter per task.

Profiles each adapter's cost and latency on completed tasks,
then uses that profile to route new tasks to the cheapest worker
that can actually finish them. The profile is maintained per-adapter
and recomputed on every completed task, so the system gets smarter
over time without any configuration.

Key insight: the cheapest model for one task type may be the most
expensive for another. This module makes per-task routing decisions.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class AdapterProfile:
    adapter_name: str
    total_tasks: int = 0
    total_cost_usd_micros: int = 0
    total_seconds: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    capabilities: set[str] = field(default_factory=set)

    @property
    def avg_cost_usd(self) -> float:
        return self.total_cost_usd_micros / 1_000_000 / max(self.total_tasks, 1)

    @property
    def avg_seconds(self) -> float:
        return self.total_seconds / max(self.total_tasks, 1)

    @property
    def success_rate(self) -> float:
        n = self.success_count + self.failure_count
        return self.success_count / n if n > 0 else 0.0

@dataclass
class AdapterDecision:
    adapter_name: str
    reason: str
    estimated_cost_usd: float
    estimated_seconds: float
    confidence: float  # 0-1 based on sample size

class AdapterProfiler:
    """Tracks per-adapter cost/performance and decides which to use."""

    def __init__(self) -> None:
        self._profiles: dict[str, AdapterProfile] = {}

    def record_task(
        self,
        adapter_name: str,
        cost_usd_micros: int,
        seconds: float,
        success: bool,
        capabilities: set[str] | None = None,
    ) -> None:
        prof = self._profiles.setdefault(adapter_name, AdapterProfile(adapter_name=adapter_name))
        prof.total_tasks += 1
        prof.total_cost_usd_micros += cost_usd_micros
        prof.total_seconds += seconds
        if success:
            prof.success_count += 1
        else:
            prof.failure_count += 1
        if capabilities:
            prof.capabilities.update(capabilities)

    def best_adapter(
        self,
        required_capabilities: set[str],
        budget_usd_micros: int | None = None,
        max_seconds: float | None = None,
    ) -> AdapterDecision | None:
        candidates = []
        for name, prof in self._profiles.items():
            if required_capabilities and not required_capabilities.issubset(prof.capabilities):
                continue
            if prof.success_rate < 0.3 and prof.total_tasks >= 5:
                continue  # skip adapters that consistently fail
            if budget_usd_micros is not None and prof.avg_cost_usd * 1_000_000 > budget_usd_micros:
                continue
            if max_seconds is not None and prof.avg_seconds > max_seconds:
                continue
            candidates.append((name, prof))

        if not candidates:
            return None

        # Rank by cost, with success rate as tiebreaker
        candidates.sort(key=lambda x: (x[1].avg_cost_usd, -x[1].success_rate))
        best_name, best_prof = candidates[0]
        confidence = min(best_prof.total_tasks / 20, 1.0)  # more tasks = more confidence

        return AdapterDecision(
            adapter_name=best_name,
            reason=f"cheapest capable: ${best_prof.avg_cost_usd:.4f}/task, {best_prof.success_rate:.0%} success rate",
            estimated_cost_usd=best_prof.avg_cost_usd,
            estimated_seconds=best_prof.avg_seconds,
            confidence=confidence,
        )

    def all_profiles(self) -> dict[str, AdapterProfile]:
        return dict(self._profiles)
