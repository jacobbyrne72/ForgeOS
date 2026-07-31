"""Cost efficiency ratio tracker - measures cost per useful output."""
from __future__ import annotations

from .cost_tracker import CostTracker


class CostEfficiency:
    def __init__(self):
        self.tracker = CostTracker()

    def record_output(self, task_type: str, output_quality: float,
                      tokens_used: int) -> None:
        """Record the quality of output for a task.

        output_quality is a 0.0-1.0 score of how useful the output was.
        A perfect score means every token was worth spending.
        A score of 0 means wasteful output.
        """
        self.tracker.record(task_type, 0.0, tokens_used, "efficiency", {
            "quality": output_quality,
        })

    def efficiency_report(self) -> dict:
        """Calculate cost efficiency ratios."""
        by_type = self.tracker.by_task_type()
        report = []

        for bt in by_type:
            tt = bt["task_type"]
            # We use quality scores from metadata if available
            # For now, calculate as tokens_per_dollar
            tokens = bt["total_tokens"]
            usd = bt["total_usd"]
            if usd > 0:
                tokens_per_dollar = tokens / usd
                efficiency = min(1.0, tokens_per_dollar / 10000)
            else:
                tokens_per_dollar = float("inf")
                efficiency = 1.0

            report.append(dict(
                task_type=tt,
                total_usd=usd,
                total_tokens=tokens,
                tokens_per_dollar=round(tokens_per_dollar, 2) if tokens_per_dollar != float('inf') else 'N/A',
                efficiency_score=round(efficiency, 4) if tokens_per_dollar != float('inf') else 1.0,
            ))

        total_usd_all = sum(r["total_usd"] for r in report)
        weighted = sum(
            (r["tokens_per_dollar"] if isinstance(r["tokens_per_dollar"], (int, float)) else 0) * r["total_usd"]
            for r in report
        )
        overall = round(weighted / total_usd_all, 2) if total_usd_all > 0 else 0
        return dict(
            by_task_type=sorted(report, key=lambda r: r["efficiency_score"], reverse=True),
            overall_tokens_per_dollar=overall,
        )