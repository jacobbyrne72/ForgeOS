"""Model ranker - rank models by cost-effectiveness for tasks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelRanking:
    model: str
    provider: str
    cost_per_1k_tokens: float
    quality_score: float
    latency_ms: float


class ModelRanker:
    """Rank models by cost-effectiveness: quality per dollar."""

    def __init__(self):
        # Known model quality scores (1.0 = reference)
        self.model_quality = {
            "opus": 1.0,
            "sonnet": 0.92,
            "haiku": 0.78,
            "gpt-4": 1.0,
            "gpt-4o-mini": 0.85,
            "gpt-3.5-turbo": 0.72,
            "claude-3-sonnet": 0.92,
            "claude-3-haiku": 0.78,
        }
        self.model_costs = {
            "opus": 0.015,
            "sonnet": 0.003,
            "haiku": 0.00025,
            "gpt-4": 0.03,
            "gpt-4o-mini": 0.00015,
            "gpt-3.5-turbo": 0.0005,
            "claude-3-sonnet": 0.003,
            "claude-3-haiku": 0.00025,
        }

    def rank(self, task_complexity: str = "simple") -> list[dict]:
        """Rank models by cost-effectiveness for a task.
        
        Lower rank number = better value.
        quality_per_dollar = quality_score / cost_per_1k_tokens
        """
        results = []
        for model, quality in self.model_quality.items():
            cost = self.model_costs.get(model, 0.003)
            value = quality / max(cost, 0.00001)
            results.append(dict(
                model=model,
                provider="anthropic" if "claude" in model or "opus" in model or "sonnet" in model or "haiku" in model else "openai",
                cost_per_1k_tokens=cost,
                quality_score=quality,
                quality_per_dollar=round(value, 2),
            ))

        results.sort(key=lambda r: r["quality_per_dollar"], reverse=True)
        return results

    def recommend(self, task_complexity: str = "simple", max_cost: float = 0.01) -> Optional[dict]:
        """Recommend the cheapest model that meets quality requirements."""
        ranked = self.rank(task_complexity)
        for r in ranked:
            if r["cost_per_1k_tokens"] <= max_cost:
                return dict(
                    recommendation=r["model"],
                    reason=f"Best value at ${r['cost_per_1k_tokens']}/1k tokens",
                    quality_per_dollar=r["quality_per_dollar"],
                    estimated_cost_per_task=r["cost_per_1k_tokens"] * 0.5,
                )
        return None
