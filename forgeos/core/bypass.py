"""When orchestration costs more than the work it orchestrates.

Measured: a one-line docstring edit, submitted through the dashboard chat bar,
went through compile -> route -> execute -> review -> merge gate and drew four
model calls plus a tier prior. A single raw call answered the same request in
one round trip. The orchestration was not wrong -- it produced a reviewed result
where the raw call produced a truncated one -- but it was disproportionate, and
a user who tries ForgeOS on a small edit and watches it convene a committee will
not try it twice.

The lever is that some tasks are FULLY SPECIFIED by their own contract. "Rename
`_weaker` to `_less_certain` in one file" has no ambiguity to resolve, no design
to weigh and nothing for a planner to decompose. Everything an orchestrator adds
there is overhead: the decomposition finds one task, the router picks the only
capable worker, and the reviewer reads a diff a test already proved.

WHAT THIS DOES NOT SKIP, ever:

  - The merge gate. Tests, scanners and the acceptance check are deterministic
    and near-free; skipping them to save nothing is how a bypass turns into a
    hole. A bypassed task is still gated, still recorded, still refusable.
  - The ledger. Spend is recorded on the same path. An unrecorded call is
    invisible to the governor whatever route it took.
  - The budget. `CallRefused` still applies.

What it skips is the PLANNING and MULTI-WORKER COORDINATION around a task that
has one obvious shape: no decomposition pass, no capability negotiation, no
second worker convened to discuss a rename.

The eligibility test is deliberately narrow and deterministic. A bypass that
fires on ambiguous work trades a small saving for the expensive failure --
unreviewed wrong code merged fast -- and that trade is never worth it. Every
condition below has to hold; any doubt routes the normal way.
"""

from __future__ import annotations

from dataclasses import dataclass

from .effort import Difficulty, classify

# One file. The moment a change spans files, "which files" is part of the
# problem and a planner earns its keep.
MAX_BYPASS_SCOPE_PATHS = 1

# Difficulties whose work is fully specified by the contract itself.
BYPASS_DIFFICULTIES = frozenset({Difficulty.MECHANICAL, Difficulty.LOOKUP})

# Capabilities that always deserve the full path regardless of how small the
# change looks. A one-line auth edit is still an auth edit.
NEVER_BYPASS_CAPABILITIES = frozenset({
    "security", "migration", "release", "deploy", "auth", "review",
})


@dataclass(frozen=True)
class BypassDecision:
    """Whether to skip orchestration, and the reason either way.

    `reason` is populated in BOTH directions on purpose. A silent bypass and a
    silent refusal to bypass are equally hard to audit later, and this decision
    changes what a receipt means.
    """

    bypass: bool
    reason: str
    difficulty: Difficulty = Difficulty.STANDARD

    def render(self) -> str:
        verb = "bypassing orchestration" if self.bypass else "full orchestration"
        return f"{verb}: {self.reason}"


def should_bypass(
    objective: str,
    *,
    scope_paths: int = 1,
    capabilities: frozenset[str] | set[str] | None = None,
    acceptance: list[str] | tuple[str, ...] | None = None,
    depends_on: list[str] | tuple[str, ...] | None = None,
    force_full: bool = False,
) -> BypassDecision:
    """Deterministic. No model call -- paying one to decide whether to pay one
    is the trap this whole module exists inside of.

    Ordered so the strongest objection wins: an explicit override, then a
    dependency edge, then risky capabilities, then breadth, then difficulty,
    then the acceptance contract. Each returns the specific reason rather than
    a generic "not eligible", because the reason is what tells a maintainer
    whether the policy is behaving.
    """
    caps = set(capabilities or ())

    if force_full:
        return BypassDecision(False, "caller requested the full path")

    if depends_on:
        # A task with a predecessor is a graph node. Whatever it is on its own,
        # its ORDER matters, and ordering is precisely what the scheduler owns.
        return BypassDecision(False, f"has {len(depends_on)} dependency edge(s)")

    risky = caps & NEVER_BYPASS_CAPABILITIES
    if risky:
        return BypassDecision(False, f"capabilities require full review: {', '.join(sorted(risky))}")

    if scope_paths > MAX_BYPASS_SCOPE_PATHS:
        return BypassDecision(
            False, f"touches {scope_paths} paths; choosing among them is the planning"
        )

    difficulty = classify(objective, scope_paths=scope_paths, capabilities=caps)
    if difficulty not in BYPASS_DIFFICULTIES:
        return BypassDecision(
            False, f"{difficulty.value} work is not fully specified by its contract",
            difficulty=difficulty,
        )

    if not acceptance:
        # Without a stated acceptance criterion there is nothing for the merge
        # gate to check mechanically, and the bypass's whole safety argument is
        # that the gate still runs. No criterion, no bypass.
        return BypassDecision(
            False, "no acceptance criterion to check mechanically", difficulty=difficulty
        )

    return BypassDecision(
        True,
        f"{difficulty.value} work, one path, {len(acceptance)} mechanical "
        f"criterion(a) — orchestration would add cost, not certainty",
        difficulty=difficulty,
    )


def bypass_for_spec(spec, *, force_full: bool = False) -> BypassDecision:
    """`should_bypass` from a `TaskSpec`. Never raises: a bypass decision is an
    optimisation, and failing a real task because eligibility could not be
    computed would trade the work for a routing detail."""
    try:
        return should_bypass(
            getattr(spec, "subject", "") or "",
            scope_paths=len(getattr(getattr(spec, "scope", None), "paths", []) or []) or 1,
            capabilities=set(getattr(spec, "capabilities", []) or []),
            acceptance=list(getattr(spec, "acceptance", []) or []),
            depends_on=list(getattr(spec, "depends_on", []) or []),
            force_full=force_full,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return BypassDecision(False, f"eligibility unavailable ({type(exc).__name__})")
