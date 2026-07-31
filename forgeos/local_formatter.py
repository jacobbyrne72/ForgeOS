"""Local code formatter that uses ruff or black instead of LLM API calls."""
from __future__ import annotations

import subprocess
import shutil


class LocalFormatter:
    """Format Python code locally using existing tools - zero API cost."""

    def __init__(self, tool: str = "ruff"):
        self.tool = tool
        self.available = shutil.which(tool) is not None
        self.cost_per_format = 0.0  # local = $0

    def format_code(self, code: str) -> tuple[str, dict]:
        """Format code locally if possible, otherwise mark as deferred."""
        if not self.available:
            return code, dict(
                formatted=False,
                reason="tool_not_available",
                cost_saved=0, 
                would_cost_api=0.03,
            )

        if self.tool == "ruff":
            try:
                # Run ruff format on the code via stdin
                result = subprocess.run(
                    ["ruff", "format", "-"],
                    input=code.encode("utf-8"),
                    capture_output=True,
                    timeout=10,
                )
                formatted = result.stdout.decode("utf-8")
                if formatted.strip():
                    return formatted, dict(
                        formatted=True,
                        cost_saved=0.03,
                        would_cost_api=0.03,
                    )
            except Exception:
                pass

        return code, dict(
            formatted=False,
            reason="format_failed",
            cost_saved=0,
            would_cost_api=0.03,
        )


    def format_file(self, filepath: str) -> dict:
        """Format a file in-place locally."""
        if not self.available:
            return dict(success=False, reason="tool_not_available")
        try:
            result = subprocess.run(
                [self.tool, "format", filepath],
                capture_output=True,
                timeout=30,
            )
            return dict(success=result.returncode == 0, cost_saved=0.03)
        except Exception as e:
            return dict(success=False, reason=str(e))


    def save_savings_report(self) -> dict:
        return dict(
            tool=self.tool,
            available=self.available,
            cost_per_format=0.0,
            api_cost_avoided_per_call=0.03,
        )

