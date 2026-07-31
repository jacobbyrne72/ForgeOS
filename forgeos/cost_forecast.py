"""Cost forecast - predict future spend using historical data, and estimate
the cost of an individual upcoming call before making it.

Consolidates cost_forecast + budget_forecast + cost_predictor (2026-07-31):
CostForecast.forecast() is unchanged and still the one wired into
`forge forecast`; its new budget_ok() folds in budget_forecast.py's
ceiling-check idea, reusing this class's own regression forecast rather than
budget_forecast.py's separate (weaker) averaging algorithm — so
budget_forecast.py's class is gone rather than kept side by side.
CostPredictor is a genuinely different job — an a-priori per-call $ estimate
from a token count and a price table, no history involved — kept as its own
class here rather than as a fourth small orphaned module. Neither donor was
wired anywhere or used by any other module.
"""
from __future__ import annotations

from .cost_tracker import CostTracker


class CostForecast:
    def __init__(self, daily_budget: float | None = None):
        self.tracker = CostTracker()
        self.daily_budget = daily_budget

    def forecast(self, days: int = 30) -> dict:
        """Forecast cost for next N days using linear trend.

        Uses simple linear regression on historical daily spend.
        Falls back to average if insufficient data.
        """
        history = self.tracker.recent(limit=100)

        if len(history) < 2:
            return {
                "method": "insufficient_data",
                "days": days,
                "forecast_usd": 0.0,
                "confidence": "unknown",
            }

        # Simple linear regression
        n = len(history)
        sum_x = sum(range(n))
        sum_y = sum(h["saved_usd"] for h in history)
        sum_xy = sum(i * h["saved_usd"] for i, h in enumerate(history))
        sum_x2 = sum(i * i for i in range(n))

        denom = n * sum_x2 - sum_x * sum_x
        if denom == 0:
            avg = sum_y / n
            return {
                "method": "average",
                "days": days,
                "forecast_usd": round(avg * days, 4),
                "confidence": "low",
            }

        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n

        # Predict cumulative spend over next N days
        predicted = intercept * days + slope * (n + days) * days / 2
        predicted = max(0, predicted)

        # Confidence based on data size
        if n >= 30:
            confidence = "high"
        elif n >= 10:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "method": "linear_regression",
            "days": days,
            "forecast_usd": round(predicted, 4),
            "slope_per_day": round(slope, 6),
            "confidence": confidence,
            "data_points": n,
        }

    def budget_ok(self, days: int = 30) -> dict:
        """Is the forecasted spend over `days` within daily_budget?

        (from budget_forecast.py's BudgetForecast.budget_ok/forecast_days,
        reimplemented on top of forecast()'s regression instead of a
        separate average.)
        """
        if self.daily_budget is None:
            raise ValueError("CostForecast(daily_budget=...) was not set")
        result = self.forecast(days)
        ceiling = self.daily_budget * days
        remaining = ceiling - result["forecast_usd"]
        return dict(
            forecast_usd=result["forecast_usd"],
            budget_ceiling=round(ceiling, 2),
            budget_remaining=round(remaining, 2),
            ok=remaining >= 0,
        )


class CostPredictor:
    """Estimate the cost of a single upcoming call before making it.

    (from cost_predictor.py, unchanged — a pre-call price-table lookup, not
    a historical-trend forecast, which is why it stayed a separate class
    instead of being folded into CostForecast.)
    """

    def __init__(self, cost_per_1k_input: float = 0.0015, cost_per_1k_output: float = 0.002):
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
