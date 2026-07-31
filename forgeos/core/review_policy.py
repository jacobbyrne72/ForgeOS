"""Who is strong enough to review this, and when is tiering up worth the tokens?

`core/verify.py`'s merge gate already refuses self-review and warns when the
reviewer shares the implementer's provider family (LLM evaluators favour their
own generations -- Panickssery et al., NeurIPS 2024, arXiv 2404.13076). What it
never checked is CAPABILITY. A different worker at a weaker tier is still a
rubber stamp: a model that could not have written the change is in no position
to find what is wrong with it, and "two workers agreed" reads in the receipt
exactly like real corroboration.

THE COST ARGUMENT, because "always have the strong model check" is the obvious
policy and it is wrong.

Reviewing is cheap in a way implementing is not. The implementer needs the
repo: context capsule, scope files, prior attempts. The reviewer needs the
DIFF, the criteria, and the evidence -- a fraction of the input tokens, and
short output ("pass", or what is wrong). So a tier-up review costs far less
than a tier-up implementation, and that asymmetry is what makes selective
escalation affordable at all.

But it is not free, and the expensive tier is the scarcest. Spending it on
every task re-buys verification that mechanical gates -- tests, scanners,
acceptance checks -- already did deterministically for nothing. The ladder that
actually pays:

    1. Mechanical gates.        Free. Always. Already enforced in verify.py.
    2. Same-tier, different worker, different family where possible.
    3. Tier UP, only on a trigger below.

The triggers are all "the cost of being wrong just went up" -- the same
precedence `core/effort.py` uses, for the same reason:

  - DEEP/COMPLEX difficulty. Under-reviewed hard work yields a confident wrong
    merge, which is worse than an expensive right one because it does not look
    like a failure.
  - The merge gate raised warnings. Something already smells.
  - The implementer's measured win rate is below par. A worker that keeps
    getting rejected does not get a cheaper reviewer.
  - Risky surface: security or migration capabilities in the task.

And one rule with no trigger attached, because it is never acceptable:
A REVIEWER MUST NEVER BE WEAKER THAN THE IMPLEMENTER. Not as an escalation
policy -- as a floor. Cheap-reviews-expensive is the shape that produces
"approved" with no information in it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .effort import Difficulty
from .router import Tier

# A worker rejected this often has earned a stronger reader, whatever the task
# looks like. Below this, "it usually works" is not evidence.
DEFAULT_MIN_WIN_RATE = 0.6

# Capabilities where a missed defect is not merely rework.
RISKY_CAPABILITIES = frozenset({"security", "migration", "release", "deploy", "auth"})

# Difficulties that buy a stronger reader on their own.
_TIER_UP_DIFFICULTIES = frozenset({Difficulty.COMPLEX, Difficulty.DEEP})


@dataclass(frozen=True)
class ReviewRequirement:
    """What this task's review must satisfy for a merge to mean anything."""

    min_tier: Tier
    reasons: tuple[str, ...] = ()
    prefer_different_family: bool = True

    @property
    def escalated(self) -> bool:
        """True when the policy asked for more than the implementer's own tier."""
        return bool(self.reasons)

    def render(self) -> str:
        if not self.reasons:
            return f"review at tier >= {self.min_tier.name} (floor: never weaker than the author)"
        return (f"review at tier >= {self.min_tier.name} because: "
                + "; ".join(self.reasons))


def required_review(
    *,
    implementer_tier: Tier,
    difficulty: Difficulty = Difficulty.STANDARD,
    win_rate: float | None = None,
    gate_warnings: int = 0,
    capabilities: frozenset[str] | set[str] | None = None,
    min_win_rate: float = DEFAULT_MIN_WIN_RATE,
) -> ReviewRequirement:
    """The minimum reviewer this task deserves.

    The floor is always the implementer's own tier -- never weaker -- and each
    trigger raises it by exactly one rung. One rung, not straight to the top:
    the point is a reader who could have done the work, not the most expensive
    model available. COUNCIL is reserved for irreducible disagreement and is
    never reached by this function.
    """
    caps = set(capabilities or ())
    reasons: list[str] = []

    if difficulty in _TIER_UP_DIFFICULTIES:
        reasons.append(f"{difficulty.value} work needs a reader who could have written it")
    if gate_warnings > 0:
        reasons.append(f"the merge gate raised {gate_warnings} warning(s)")
    if win_rate is not None and win_rate < min_win_rate:
        reasons.append(f"implementer win rate {win_rate:.0%} is below {min_win_rate:.0%}")
    risky = caps & RISKY_CAPABILITIES
    if risky:
        reasons.append(f"risky surface: {', '.join(sorted(risky))}")

    floor = implementer_tier
    if reasons:
        # One rung up, capped below COUNCIL -- a parallel panel is a different
        # decision with a different cost, not the top of this ladder.
        floor = Tier(min(int(implementer_tier) + 1, int(Tier.STRONG)))
    return ReviewRequirement(min_tier=floor, reasons=tuple(reasons))


def review_is_adequate(
    requirement: ReviewRequirement,
    *,
    reviewer_tier: Tier,
    implementer_tier: Tier,
    reviewer_family: str = "",
    implementer_family: str = "",
) -> tuple[bool, list[str], list[str]]:
    """`(ok, blocking_reasons, warnings)` for a proposed reviewer.

    Blocking and warning are deliberately different severities, matching
    `verify.py`: a reviewer too weak to have written the change carries no
    information and blocks; a same-family reviewer is weaker evidence rather
    than none, and blocking it outright would refuse every single-vendor fleet.
    """
    blocking: list[str] = []
    warnings: list[str] = []

    if reviewer_tier < implementer_tier:
        blocking.append(
            f"reviewer tier {reviewer_tier.name} is BELOW implementer "
            f"{implementer_tier.name} — a model that could not have written the "
            f"change cannot judge it, and 'approved' from it carries no information"
        )
    elif reviewer_tier < requirement.min_tier:
        blocking.append(
            f"reviewer tier {reviewer_tier.name} is below the required "
            f"{requirement.min_tier.name} — {'; '.join(requirement.reasons)}"
        )

    if (requirement.prefer_different_family and reviewer_family and implementer_family
            and reviewer_family == implementer_family):
        warnings.append(
            f"reviewer and implementer are both {reviewer_family} — an evaluator "
            f"favours its own generations (arXiv 2404.13076), so this is weaker "
            f"corroboration than a cross-family read"
        )

    return (not blocking), blocking, warnings


@dataclass
class ReviewPlan:
    """What to actually do, for a caller that wants one object."""

    requirement: ReviewRequirement
    escalate: bool
    estimated_extra_usd: float = 0.0
    notes: list[str] = field(default_factory=list)


def plan_review(
    *,
    implementer_tier: Tier,
    difficulty: Difficulty = Difficulty.STANDARD,
    win_rate: float | None = None,
    gate_warnings: int = 0,
    capabilities: frozenset[str] | set[str] | None = None,
) -> ReviewPlan:
    """Convenience wrapper: requirement plus whether it escalates."""
    requirement = required_review(
        implementer_tier=implementer_tier, difficulty=difficulty, win_rate=win_rate,
        gate_warnings=gate_warnings, capabilities=capabilities,
    )
    notes = []
    if not requirement.escalated:
        notes.append(
            "same-tier review is sufficient here; mechanical gates already cover "
            "what a stronger reader would mostly re-derive"
        )
    return ReviewPlan(
        requirement=requirement,
        escalate=requirement.min_tier > implementer_tier,
        notes=notes,
    )
