"""Budget forecast - predict future spend."""
from __future__ import annotations


class BudgetForecast:
    def __init__(self, daily_budget=10.0):
        self.daily_budget = daily_budget
        self._expenses = []

    def record(self, cost_usd):
        self._expenses.append(cost_usd)

    def forecast_days(self, days=30):
        if not self._expenses:
            return dict(forecast=0.0, budget_remaining=self.daily_budget * days)
        avg_daily = sum(self._expenses) / len(self._expenses)
        forecast = avg_daily * days
        remaining = self.daily_budget * days - forecast
        return dict(
            forecast=round(forecast, 2),
            budget_remaining=round(max(0, remaining), 2),
            avg_daily=round(avg_daily, 4),
        )

    def budget_ok(self, days=30):
        f = self.forecast_days(days)
        return f["budget_remaining"] >= 0
