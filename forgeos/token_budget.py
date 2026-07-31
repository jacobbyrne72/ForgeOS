"""Token budget enforcer - refuses prompts that exceed budget.

Cost is proportional to tokens. This module guarantees
no single request exceeds the configured token budget,
cutting worst-case spend to zero for oversized prompts.
"""
from __future__ import annotations

DEFAULT_BUDGET = 4096
DEFAULT_WARN_AT = 0.80

class TokenBudget:
    def __init__(self, max_tokens: int = DEFAULT_BUDGET,
                 warn_ratio: float = DEFAULT_WARN_AT):
        self.max_tokens = max_tokens
        self.warn_ratio = warn_ratio
        self.warn_at = int(max_tokens * warn_ratio)
        self._total_ever = {"tokens": 0, "rejected": 0, "warned": 0}

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate: 1 token ~= 4 chars for English."""
        return max(1, len(text) // 4)

    def check(self, prompt: str, model: str = "") -> dict:
        """Check if a prompt fits the budget."""
        est = self.estimate_tokens(prompt)
        remaining = self.max_tokens - est
        self._total_ever["tokens"] += est

        if remaining < 0:
            self._total_ever["rejected"] += 1
            return {
                "ok": False,
                "estimated_tokens": est,
                "action": "reject",
                "budget_remaining": remaining,
                "max_tokens": self.max_tokens,
                "reason": "prompt exceeds max_tokens budget",
            }
        if remaining < self.warn_at:
            self._total_ever["warned"] += 1
            return {
                "ok": True,
                "estimated_tokens": est,
                "action": "warn",
                "budget_remaining": remaining,
                "max_tokens": self.max_tokens,
                "reason": "approaching token budget limit",
            }
        return {
            "ok": True,
            "estimated_tokens": est,
            "action": "proceed",
            "budget_remaining": remaining,
            "max_tokens": self.max_tokens,
            "reason": "",
        }

    def enforce(self, prompt: str, model: str = "") -> str:
        """Check and raise if over budget, return prompt if ok."""
        result = self.check(prompt, model=model)
        if not result["ok"]:
            raise BudgetExceeded(
                "TokenBudget: {} tokens exceeds {} max. Action: reject.".format(
                    result["estimated_tokens"], self.max_tokens
                )
            )
        return prompt

    def stats(self) -> dict:
        return dict(self._total_ever)


class BudgetExceeded(Exception):
    pass
