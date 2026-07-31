"""Token-aware request router — pick cheapest model for the task."""
from __future__ import annotations


class TokenRouter:
    """Routes requests to the cheapest model that meets the requirement."""

    def __init__(self, models: dict[str, dict] | None = None):
        self.models = models or {}

    def register(self, name: str, cost_per_1k_input: float,
                 cost_per_1k_output: float, context_window: int,
                 max_output_tokens: int = 4096) -> None:
        self.models[name] = {
            "cost_per_1k_input": cost_per_1k_input,
            "cost_per_1k_output": cost_per_1k_output,
            "context_window": context_window,
            "max_output_tokens": max_output_tokens,
        }

    def cheapest(self, est_input_tokens: int, est_output_tokens: int,
                 min_context: int = 0) -> str | None:
        candidates = [
            (name, info) for name, info in self.models.items()
            if info["context_window"] >= est_input_tokens + est_output_tokens
            and info["max_output_tokens"] >= est_output_tokens
            and (min_context == 0 or info["context_window"] >= min_context)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda x: self._cost(x[1], est_input_tokens, est_output_tokens))[0]

    def _cost(self, info: dict, input_tokens: int, output_tokens: int) -> float:
        return (info["cost_per_1k_input"] * input_tokens / 1000 +
                info["cost_per_1k_output"] * output_tokens / 1000)

    def route(self, prompt: str, est_output_tokens: int = 512,
              min_context: int = 0) -> dict:
        est_input = len(prompt) // 4
        model = self.cheapest(est_input, est_output_tokens, min_context)
        if model is None:
            return {"model": None, "est_cost_usd": 0, "reason": "no_suitable_model"}
        info = self.models[model]
        cost = self._cost(info, est_input, est_output_tokens)
        return {"model": model, "est_cost_usd": round(cost, 6),
                "est_input_tokens": est_input, "est_output_tokens": est_output_tokens}

    def route_batch(self, prompts: list[str], est_output_tokens: int = 512,
                    min_context: int = 0) -> list[dict]:
        return [self.route(p, est_output_tokens, min_context) for p in prompts]

    def compare(self, prompt: str, est_output_tokens: int = 512) -> list[dict]:
        est_input = len(prompt) // 4
        results = []
        for name, info in self.models.items():
            if info["context_window"] < est_input + est_output_tokens:
                continue
            cost = self._cost(info, est_input, est_output_tokens)
            results.append({"model": name, "est_cost_usd": round(cost, 6),
                            "input_tokens": est_input, "output_tokens": est_output_tokens})
        results.sort(key=lambda x: x["est_cost_usd"])
        return results
