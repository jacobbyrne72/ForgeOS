"""Cost replay module."""
from __future__ import annotations

from .cost_tracker import CostTracker


class CostReplayer:
    def __init__(self):
        self.tracker = CostTracker()

    def replay(self):
        stats = self.tracker.total_saved()
        return dict(
            total_saved=stats.get("total_usd", 0),
            calls=stats.get("total_events", 0),
            avg_savings_per_call=round(
                stats.get("total_usd", 0) / max(1, stats.get("total_events", 1)), 4
            ),
        )

    def print_replay(self):
        info = self.replay()
        print("=== Cost Replay ===")
        print("Total saved: $%.4f" % info["total_saved"])
        print("Total calls:", info["calls"])
        print("Avg savings/call: $%.4f" % info["avg_savings_per_call"])
        return info
