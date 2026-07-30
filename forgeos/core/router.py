"""The escalation ladder — start at the cheapest tier that can plausibly finish.

Two rules do most of the work here:

1. **Begin at the cheapest tier whose historical success rate clears the task's
   threshold.** Not the cheapest tier outright — a tier that fails costs its own
   tokens plus the retry plus the escalation, so "cheap and hopeless" is the most
   expensive option available.

2. **Escalate at phase boundaries, not every turn.** Per-turn escalation is how a
   harness ends up paying premium rates for a whole task because of one bad turn.

Tier 0 is ordinary code. It is genuinely free and it is checked first, because the
biggest saving available is never calling a model at all.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel

from ..contracts import FailureClass
from ..registry import Candidate, CostTier, Registry


class Tier(IntEnum):
    """Ordered by marginal cost. Never skip upward without a reason."""

    DETERMINISTIC = 0  # search, parse, dep graph, tests, formatting — no model
    LOCAL = 1          # local/small/free model: classify, summarise, dedupe
    CHEAP_API = 2      # docs, routine tests, narrow edits
    SUBSCRIPTION = 3   # subscription coding CLI: standard implementation & debugging
    STRONG = 4         # architecture, hard cross-module bugs, migration planning
    COUNCIL = 5        # parallel panel: only for irreducible high-risk disagreement


# Cost tiers map onto ladder rungs. Free-and-local is rung 1, premium is rung 4.
_TIER_BY_COST: dict[CostTier, Tier] = {
    CostTier.FREE: Tier.LOCAL,
    CostTier.CHEAP: Tier.CHEAP_API,
    CostTier.STANDARD: Tier.SUBSCRIPTION,
    CostTier.PREMIUM: Tier.STRONG,
}


class ReasoningEffort(str):
    """Provider-independent effort. Adapters translate to their own knob."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAX = "max"


# Effort is a property of the task, not of the model. Declaring it here keeps a
# worker from silently thinking hard (and expensively) about a trivial edit.
EFFORT_POLICY: dict[str, str] = {
    "trivial_edit": ReasoningEffort.NONE,
    "known_pattern": ReasoningEffort.LOW,
    "mechanical": ReasoningEffort.LOW,
    "narrow_bugfix": ReasoningEffort.MEDIUM,
    "test": ReasoningEffort.MEDIUM,
    "review": ReasoningEffort.MEDIUM,
    "cross_module_bugfix": ReasoningEffort.HIGH,
    "design": ReasoningEffort.HIGH,
    "architecture": ReasoningEffort.MAX,
    "security": ReasoningEffort.MAX,
}
DEFAULT_EFFORT = ReasoningEffort.MEDIUM

# How confident we need to be that a tier can finish, by task risk.
THRESHOLD_BY_RISK: dict[str, float] = {
    "low": 0.40,
    "medium": 0.60,
    "high": 0.75,
}
DEFAULT_RISK = "medium"


# Output-length ceilings per effort level. Routing the OUTPUT budget matters as
# much as routing the model: a strong model held to a short answer is frequently
# both cheaper and better than a weak model allowed to ramble, and output tokens
# are the expensive side of most price sheets.
MAX_OUTPUT_BY_EFFORT: dict[str, int] = {
    ReasoningEffort.NONE: 512,
    ReasoningEffort.LOW: 1_024,
    ReasoningEffort.MEDIUM: 2_048,
    ReasoningEffort.HIGH: 4_096,
    ReasoningEffort.MAX: 8_192,
}


class Route(BaseModel):
    """A routing decision, with the reasoning recorded for the ledger.

    Four things are chosen, not one: the worker, the tier, the reasoning effort and
    the output-length budget. Selecting only a model name leaves the two biggest
    cost levers unset.
    """

    tier: Tier
    worker_id: str = ""
    effort: str = DEFAULT_EFFORT
    max_output_tokens: int = MAX_OUTPUT_BY_EFFORT[DEFAULT_EFFORT]
    reason: str = ""
    measured: bool = False
    deterministic: bool = False
    escalated_from: Tier | None = None

    @property
    def costs_tokens(self) -> bool:
        return not self.deterministic


class Router:
    def __init__(self, registry: Registry):
        self.registry = registry

    # ------------------------------------------------------------- effort

    def effort_for(self, capabilities: list[str]) -> str:
        """Highest effort demanded by any capability the task needs."""
        order = [
            ReasoningEffort.NONE,
            ReasoningEffort.LOW,
            ReasoningEffort.MEDIUM,
            ReasoningEffort.HIGH,
            ReasoningEffort.MAX,
        ]
        best = ReasoningEffort.NONE
        seen = False
        for cap in capabilities:
            if cap in EFFORT_POLICY:
                seen = True
                if order.index(EFFORT_POLICY[cap]) > order.index(best):
                    best = EFFORT_POLICY[cap]
        return best if seen else DEFAULT_EFFORT

    # -------------------------------------------------------------- route

    def route(
        self,
        capabilities: list[str],
        *,
        stats: dict[str, dict] | None = None,
        risk: str = DEFAULT_RISK,
        needs_file_edits: bool = False,
        deterministic_available: bool = False,
        min_tier: Tier = Tier.LOCAL,
    ) -> Route | None:
        """Pick the cheapest rung whose expected success clears the risk threshold."""
        effort = self.effort_for(capabilities)

        # Rung 0 first, always. Free and exact beats cheap and probabilistic.
        if deterministic_available:
            return Route(
                tier=Tier.DETERMINISTIC,
                effort=ReasoningEffort.NONE,
                deterministic=True,
                reason="handled by ordinary code — no model call needed",
            )

        threshold = THRESHOLD_BY_RISK.get(risk, THRESHOLD_BY_RISK[DEFAULT_RISK])
        ranked = self.registry.rank(
            capabilities, stats=stats, needs_file_edits=needs_file_edits
        )
        if not ranked:
            return None

        # Cheapest rung first; take the first candidate clearing the threshold.
        by_tier = sorted(ranked, key=lambda c: (self.tier_of(c), c.usd_micros))
        for cand in by_tier:
            tier = self.tier_of(cand)
            if tier < min_tier:
                continue
            if cand.win_rate >= threshold:
                return Route(
                    tier=tier,
                    worker_id=cand.worker.worker_id,
                    effort=effort,
                    max_output_tokens=MAX_OUTPUT_BY_EFFORT[effort],
                    measured=cand.measured,
                    reason=f"cheapest tier clearing {threshold:.0%} for {risk} risk ({cand.reason})",
                )

        # Nothing clears the bar. Take the strongest available rather than the
        # cheapest — a task nobody is likely to pass should at least get the best
        # shot, not the cheapest failure followed by escalation anyway.
        best = max(ranked, key=lambda c: c.win_rate)
        return Route(
            tier=self.tier_of(best),
            worker_id=best.worker.worker_id,
            effort=effort,
            max_output_tokens=MAX_OUTPUT_BY_EFFORT[effort],
            measured=best.measured,
            reason=(
                f"no tier clears {threshold:.0%}; using best available "
                f"({best.win_rate:.0%}) rather than a cheap likely failure"
            ),
        )

    def tier_of(self, candidate: Candidate) -> Tier:
        return _TIER_BY_COST[candidate.worker.tier]

    # ----------------------------------------------------------- escalate

    def escalate(
        self,
        current: Route,
        failure: FailureClass,
        capabilities: list[str],
        *,
        stats: dict[str, dict] | None = None,
        risk: str = DEFAULT_RISK,
        needs_file_edits: bool = False,
        at_phase_boundary: bool = True,
    ) -> Route | None:
        """Move up a rung — but only when a rung is actually the problem.

        Returns None when escalation would be waste. Refusing to escalate is the
        valuable behaviour: three of the six failure classes cannot be fixed by any
        model at any price, and spending premium tokens on them is pure loss.
        """
        if not failure.escalate_model:
            return None
        if not at_phase_boundary:
            return None
        if current.tier >= Tier.COUNCIL:
            return None

        next_tier = Tier(min(int(current.tier) + 1, int(Tier.STRONG)))
        route = self.route(
            capabilities,
            stats=stats,
            risk=risk,
            needs_file_edits=needs_file_edits,
            min_tier=next_tier,
        )
        if route is None:
            return None
        route.escalated_from = current.tier
        route.reason = f"escalated from tier {int(current.tier)} on model failure; {route.reason}"
        return route

    def should_convene_council(self, risk: str, disagreement: bool, tests_can_decide: bool) -> bool:
        """A council is the most expensive tool available. Almost never the answer.

        When tests can adjudicate, running both candidates and letting the suite
        decide is cheaper AND more reliable than paying several models to argue —
        argument converges on the most confidently expressed answer, not the correct
        one.
        """
        if tests_can_decide:
            return False
        return risk == "high" and disagreement


__all__ = [
    "DEFAULT_EFFORT",
    "DEFAULT_RISK",
    "EFFORT_POLICY",
    "THRESHOLD_BY_RISK",
    "ReasoningEffort",
    "Route",
    "Router",
    "Tier",
]
