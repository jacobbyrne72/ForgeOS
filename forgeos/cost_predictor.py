"""Cost predictor - estimate API cost before calling."""
from __future__ import annotations

class CostPredictor:
    def __init__(self, cost_per_1k_input=0.0015, cost_per_1k_output=0.002):
        self.cost_per_1k_input = cost_per_1k_input
        self.cost_per_1k_output = cost_per_1k_output
        self._predictions = 0

    def predict_cost(self, input_tokens, output_tokens=0):
        self._predictions += 1
        input_cost = input_tokens * self.cost_per_1k_input / 1000
        output_cost = output_tokens * self.cost_per_1k_output / 1000
        return round(input_cost + output_cost, 6)

    def should_use_cache(self, input_tokens, cached_cost=0.0):
        predicted = self.predict_cost(input_tokens)
        return cached_cost < predicted * 0.5

    def stats(self):
        return dict(predictions=self._predictions)
