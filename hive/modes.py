"""Two operating profiles: how hard to look before touching anything.

`Mode` picks a verification posture, never a budget or an approval rule.
`Budget` (hard USD/seconds/iteration ceilings) and the human-approval gate for
irreversible actions (AGENTS.md rule 7) are governed entirely by
`contracts.Budget` and the adapter/governor layer — nothing in this file can
widen either. BLITZ buys speed by narrowing *verification breadth* during
iteration (targeted tests instead of the full suite, fewer review passes); it
never buys speed by raising a cap or skipping an approval. See
`ModeProfile` below: there is deliberately no budget-shaped or
approval-shaped field on it.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, field_validator

from .core.router import ReasoningEffort, Tier

# Capability/keyword stems that force DEEP regardless of how small the task
# looks or how low its measured risk is. Substring match on purpose: a
# capability tagged "database_migration" or "auth_flow" should still trip
# this, not just the exact literal "migration"/"auth".
_DEEP_STEMS: tuple[str, ...] = ("security", "auth", "architecture", "migrat")


class Mode(str, Enum):
    BLITZ = "blitz"  # small fixes, tests, docs, mechanical migrations, read-only audits
    DEEP = "deep"    # architecture, auth, migrations, broad refactors, hard prod failures


class ModeProfile(BaseModel):
    """How a mode runs. Deliberately silent on budget and approval — those are
    never mode-dependent; see module docstring.
    """

    parallel_readers: int
    initial_tier: Tier
    reasoning_effort: str
    targeted_tests_only_during_iteration: bool
    full_review_count: int
    max_manager_interventions: int
    parallel_hypotheses: int
    require_independent_review: bool
    require_security_checks: bool

    @field_validator(
        "parallel_readers",
        "full_review_count",
        "max_manager_interventions",
        "parallel_hypotheses",
    )
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("must be >= 1")
        return v


# BLITZ: throughput over depth. Many cheap parallel readers to canvas broadly
# and fast; targeted tests while iterating, full suite only at the merge
# boundary (one full_review_count pass); tight manager-intervention ceiling
# because the whole point is not paying a human/manager tax on small work.
BLITZ_PROFILE = ModeProfile(
    parallel_readers=4,
    initial_tier=Tier.CHEAP_API,
    reasoning_effort=ReasoningEffort.LOW,
    targeted_tests_only_during_iteration=True,
    full_review_count=1,
    max_manager_interventions=2,
    parallel_hypotheses=1,
    require_independent_review=False,
    require_security_checks=False,
)

# DEEP: depth over throughput. Fewer, more careful parallel readers (breadth
# is not the bottleneck for architecture/security work, coherent understanding
# is); every iteration runs real verification, never just the targeted slice;
# independent review and security checks are mandatory; more parallel
# hypotheses because hard bugs/migrations benefit from exploring competing
# fixes rather than committing to the first one.
DEEP_PROFILE = ModeProfile(
    parallel_readers=2,
    initial_tier=Tier.STRONG,
    reasoning_effort=ReasoningEffort.HIGH,
    targeted_tests_only_during_iteration=False,
    full_review_count=2,
    max_manager_interventions=8,
    parallel_hypotheses=3,
    require_independent_review=True,
    require_security_checks=True,
)

PROFILES: dict[Mode, ModeProfile] = {
    Mode.BLITZ: BLITZ_PROFILE,
    Mode.DEEP: DEEP_PROFILE,
}


def profile_for(mode: Mode) -> ModeProfile:
    return PROFILES[mode]


def suggest_mode(capabilities: list[str], risk: str) -> Mode:
    """Deterministic. No model call.

    Anything touching security/architecture/auth/migration is DEEP regardless
    of how small it looks or how low `risk` is measured at — those keywords
    override the risk class entirely, they are not merely one input to it.
    Otherwise, high measured risk alone is also enough to warrant DEEP; every
    other combination defaults to BLITZ.
    """
    caps_lower = [c.lower() for c in capabilities]
    if any(stem in cap for cap in caps_lower for stem in _DEEP_STEMS):
        return Mode.DEEP
    if risk == "high":
        return Mode.DEEP
    return Mode.BLITZ


__all__ = [
    "BLITZ_PROFILE",
    "DEEP_PROFILE",
    "PROFILES",
    "Mode",
    "ModeProfile",
    "profile_for",
    "suggest_mode",
]
