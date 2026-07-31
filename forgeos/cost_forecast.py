"""Cost forecast - predict future spend using historical data."""
from __future__ import annotations

from .cost_tracker import CostTracker


class CostForecast:
    def __init__(self):
        self.tracker = CostTracker()

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
