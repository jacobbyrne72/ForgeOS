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
from typing import Any

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

"""Re-run cost measurement on historical jobs to identify waste.

Replays completed benchmark jobs and computes:
- Total cost per job
- Cost per task
- Wasted cost (dead worker calls, retry storms, cold starts)
- Savings vs naive execution
"""

def replay_job(ledger, job_id: str) -> dict[str, Any]:
    """Analyze a completed job and return cost breakdown."""
    job = ledger.job(job_id)
    if job is None:
        return {"error": f"job {job_id} not found"}

    tasks = ledger.tasks_for_job(job_id)
    spends = [ledger.task_spend_micros(t.id) for t in tasks]

    total_cost_usd = sum(s / 1_000_000 for s in spends)
    total_tasks = len(tasks)

    # Identify waste patterns
    dead_worker_cost = 0
    retry_cost = 0
    for task in tasks:
        task_reports = ledger.reports_for_task(task.id)
        for r in task_reports:
            if r.get("worker_id") and isinstance(r.get("worker_id"), str):
                pass  # counted normally
            usd = r.get("usd_micros", 0)
            if usd > 0 and r.get("success") is False:
                dead_worker_cost += usd
            if r.get("attempt_number", 0) > 1:
                retry_cost += usd

    return {
        "job_id": job_id,
        "objective": job.objective[:100],
        "total_tasks": total_tasks,
        "total_cost_usd": round(total_cost_usd, 6),
        "cost_per_task_usd": round(total_cost_usd / max(total_tasks, 1), 6),
        "dead_worker_waste_usd": round(dead_worker_cost / 1_000_000, 6),
        "retry_waste_usd": round(retry_cost / 1_000_000, 6),
        "total_waste_usd": round((dead_worker_cost + retry_cost) / 1_000_000, 6),
        "waste_pct": round((dead_worker_cost + retry_cost) / max(total_cost_usd * 1_000_000, 1) * 100, 1),
    }

def replay_all(ledger) -> list[dict[str, Any]]:
    """Replay all completed jobs and return per-job analysis."""
    jobs = ledger.active_jobs()
    results = []
    for row in jobs:
        if isinstance(row, dict):
            jid = row.get("id") or row.get("job_id")
        else:
            jid = getattr(row, "id", None)
        if jid is None:
            continue
        r = replay_job(ledger, jid)
        results.append(r)
    return results
