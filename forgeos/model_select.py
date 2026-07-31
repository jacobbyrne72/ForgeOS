"""Model-aware prompt optimization — pick the cheapest capable model."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass
class ModelRecommendation:
    model: str
    provider: str
    estimated_cost_usd: float
    capability_match: float
    reason: str

class ModelSelector:
    def __init__(self):
        self._tiers = {
            "ollama/llama3.2-mini": {"provider":"local","cost":0.0,"max_complexity":2,"capabilities":{"summarize","classify"}},
            "ollama/deepseek-coder-v2": {"provider":"local","cost":0.0,"max_complexity":3,"capabilities":{"edit","implement","python","refactor"}},
            "openrouter/hermes-3.1-mini": {"provider":"openrouter","cost":0.15,"max_complexity":3,"capabilities":{"edit","implement","python","typescript","summarize"}},
            "openrouter/deepseek-v3.2": {"provider":"openrouter","cost":0.14,"max_complexity":4,"capabilities":{"edit","implement","python","typescript","review"}},
            "openrouter/claude-sonnet-4": {"provider":"openrouter","cost":3.00,"max_complexity":5,"capabilities":{"edit","implement","python","typescript","review","architect"}},
        }

    def recommend(self, required_capabilities, task_complexity="simple", max_cost_usd=1.0):
        cands = []
        for model_id, t in self._tiers.items():
            if task_complexity == "complex" and t["max_complexity"] < 3:
                continue
            cap_match = len(required_capabilities & t["capabilities"]) / max(len(required_capabilities), 1)
            if cap_match < 0.3 and required_capabilities:
                continue
            if t["cost"] > max_cost_usd * 1000:
                continue
            cands.append((model_id, t, cap_match, t["cost"]))
        if not cands:
            return None
        cands.sort(key=lambda x: (x[3], -x[2]))
        best = cands[0]
        return ModelRecommendation(
            model=best[0], provider=best[1]["provider"],
            estimated_cost_usd=round(best[3] * 0.001, 6),
            capability_match=round(best[2], 2),
            reason=f"cheapest capable model at ${best[3]:.2f}/1M tokens",
        )
