"""Automatic cost optimizer — applies the right layer per task type."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass
class OptimizationPlan:
    steps: list[str]
    estimated_savings_usd: float
    estimated_latency_ms: float
    rationale: str

class CostOptimizer:
    """Chooses the cheapest optimization pipeline for a given task."""

    RULES = {
        "simple_edit":OptimizationPlan(
            steps=["cache_check", "compress_context"],
            estimated_savings_usd=0.008,
            estimated_latency_ms=5,
            rationale="Cache hit avoids model call; compression reduces tokens further",
        ),
        "code_gen":OptimizationPlan(
            steps=["compiler", "cache_check", "circuit_breaker", "model_select"],
            estimated_savings_usd=0.02,
            estimated_latency_ms=150,
            rationale="Compiler eliminates task decomposition call; model_select picks cheapest capable model",
        ),
        "security_scan":OptimizationPlan(
            steps=["diff_scan", "security_scan"],
            estimated_savings_usd=0.005,
            estimated_latency_ms=10,
            rationale="Diff scan only checks changed lines, not whole file",
        ),
        "review":OptimizationPlan(
            steps=["compiler", "cache_check", "compress_context", "model_select"],
            estimated_savings_usd=0.012,
            estimated_latency_ms=120,
            rationale="Review tasks benefit from compression + model_select + caching",
        ),
        "planning":OptimizationPlan(
            steps=["compiler", "model_select"],
            estimated_savings_usd=0.015,
            estimated_latency_ms=80,
            rationale="Planning uses compiler for task decomposition + cheap model for generation",
        ),
        "debug":OptimizationPlan(
            steps=["compiler", "circuit_breaker", "model_select"],
            estimated_savings_usd=0.01,
            estimated_latency_ms=100,
            rationale="Circit breaker saves wasted calls to unresponsive models",
        ),
        "refactor":OptimizationPlan(
            steps=["compiler", "cache_check", "compress_context", "model_select"],
            estimated_savings_usd=0.015,
            estimated_latency_ms=130,
            rationale="Refactor tasks are multi-step — compiler + compression + best model",
        ),
    }

    def plan_for(self, task_type: str) -> OptimizationPlan:
        return self.RULES.get(task_type, OptimizationPlan(
            steps=["compiler", "model_select"],
            estimated_savings_usd=0.01,
            estimated_latency_ms=50,
            rationale="Default: compiler + model_select for any unclassified task",
        ))

    def estimate_savings(self, task_type: str, daily_tasks: int = 100) -> dict[str, Any]:
        plan = self.plan_for(task_type)
        monthly = plan.estimated_savings_usd * daily_tasks * 22
        yearly = monthly * 12
        return {
            "task_type": task_type,
            "steps": plan.steps,
            "savings_per_task_usd": plan.estimated_savings_usd,
            "daily_tasks": daily_tasks,
            "monthly_savings_usd": round(monthly, 2),
            "yearly_savings_usd": round(yearly, 2),
            "rationale": plan.rationale,
        }
