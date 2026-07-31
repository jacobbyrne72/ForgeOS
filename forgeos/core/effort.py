"""How hard is this task, and therefore how much reasoning should it buy?

Model choice and reasoning effort are TWO knobs, and this project only ever
turned the first one. `Router` picks a tier (deterministic -> local -> cheap
API -> subscription -> strong); `GatewayRequest.reasoning_effort` then went out
at its `"medium"` default on every call ForgeOS has ever made, whatever the task
was.

That default is not free. Measured on ForgeBench, same model, same six
questions, the only change being effort:

    reasoning_effort="medium"   694 output tokens, 3/6 accepted
    reasoning_effort="none"     233 output tokens, 5/6 accepted

Output tokens fell 66% and acceptance ROSE. The tasks were "name the function
and the guard, under 30 words" -- lookups with nothing to reason about. The
budget went to chain-of-thought nobody reads, and on three of them the model
exhausted its output cap while still reasoning and returned an empty string,
which the suite then scored as the model getting the question wrong.

So: effort is worth routing, and routing it wrongly is expensive in both
directions. Too much burns tokens and can starve the answer; too little on a
genuinely hard task buys a cheap wrong answer, which is the more expensive
mistake because it is not obviously a mistake.

THE CLASSIFIER IS PLAIN CODE AND MUST STAY THAT WAY. Asking a model how hard a
task is means paying a model to decide what to pay a model -- the cost lands
before any work happens, on every task, including the ones that were trivial.
A wrong-but-free guess that a cheap deterministic signal can revise beats an
accurate one that bills. `classify` therefore reads the objective's own verbs,
its scope size, and its declared capabilities, and nothing else.
"""

from __future__ import annotations

import re
from enum import Enum

# Ordered easiest to hardest. The order is meaningful: `at_least` walks it.
class Difficulty(str, Enum):
    LOOKUP = "lookup"          # the answer is already in the context; find and state it
    MECHANICAL = "mechanical"  # the change is fully specified; apply it
    STANDARD = "standard"      # ordinary implementation with an obvious shape
    COMPLEX = "complex"        # multi-file, or the shape has to be decided
    DEEP = "deep"              # unknown cause, or a design with real trade-offs


_ORDER: tuple[Difficulty, ...] = (
    Difficulty.LOOKUP,
    Difficulty.MECHANICAL,
    Difficulty.STANDARD,
    Difficulty.COMPLEX,
    Difficulty.DEEP,
)

# Effort levels are `GatewayRequest.reasoning_effort`'s own vocabulary, so this
# maps straight onto the request with no translation layer to drift.
_EFFORT: dict[Difficulty, str] = {
    Difficulty.LOOKUP: "none",
    Difficulty.MECHANICAL: "none",
    Difficulty.STANDARD: "low",
    Difficulty.COMPLEX: "medium",
    Difficulty.DEEP: "high",
}

# Signals are matched as whole words against a lower-cased objective. Substring
# matching would fire "find" inside "undefined" and "why" inside "anywhere".
def _words(pattern: str) -> re.Pattern[str]:
    return re.compile(rf"\b(?:{pattern})\b")


# A question whose answer already exists somewhere in the provided context.
_LOOKUP = _words(
    r"name|names|which|what|where|list|find|locate|identify|state|report|"
    r"summarize|summarise|describe|explain what|read"
)
# A fully-specified edit. The decision has been made; only application remains.
_MECHANICAL = _words(
    r"rename|reformat|format|reindent|typo|spelling|bump|sort|alphabetise|"
    r"alphabetize|delete|remove|inline|extract|move"
)
# Genuinely open work: cause unknown, or trade-offs to weigh.
_DEEP = _words(
    r"why|debug|diagnose|root cause|investigate|design|architect|architecture|"
    r"refactor|redesign|optimise|optimize|profile|race|deadlock|flaky|"
    r"intermittent|security|migrate|migration"
)
# Ordinary build work.
_STANDARD = _words(r"add|implement|write|create|build|support|handle|wire|fix|patch|update")

# Above this many files, "which files" is itself part of the problem.
_WIDE_SCOPE = 4


def at_least(a: Difficulty, b: Difficulty) -> Difficulty:
    """The harder of two difficulties.

    Used to combine independent signals. Combining by taking the MAXIMUM rather
    than by averaging is deliberate: a task that is hard along any one axis is
    hard, and averaging would let three easy signals talk one real one down.
    """
    return a if _ORDER.index(a) >= _ORDER.index(b) else b


def classify(
    objective: str,
    *,
    scope_paths: int = 1,
    capabilities: frozenset[str] | set[str] | None = None,
) -> Difficulty:
    """Deterministic difficulty for one task. No model call, no network, no I/O.

    Precedence is by cost of being wrong, not by how confident the signal looks.
    A DEEP verb wins outright even in a one-file scope, because under-reasoning
    a root-cause question yields a confident wrong answer that reads exactly
    like a right one. LOOKUP has to survive every other check to hold, since it
    is the only class that buys no reasoning at all.
    """
    text = (objective or "").lower()
    caps = set(capabilities or ())

    if _DEEP.search(text):
        return Difficulty.DEEP
    if "review" in caps or "security" in caps:
        # A reviewer that does not reason is a rubber stamp, and the merge gate
        # depends on this one being real.
        return at_least(Difficulty.COMPLEX, Difficulty.STANDARD)

    wide = scope_paths > _WIDE_SCOPE
    if _MECHANICAL.search(text) and not wide:
        return Difficulty.MECHANICAL
    if _LOOKUP.search(text) and not _STANDARD.search(text) and not wide:
        return Difficulty.LOOKUP
    if wide:
        return Difficulty.COMPLEX
    if _STANDARD.search(text):
        return Difficulty.STANDARD
    # Nothing matched. STANDARD rather than LOOKUP: an unrecognised objective is
    # an unknown, and the cheap-wrong-answer failure costs more than the tokens
    # a middling effort setting spends.
    return Difficulty.STANDARD


def effort_for(difficulty: Difficulty) -> str:
    """`reasoning_effort` for a difficulty, in the gateway's own vocabulary."""
    return _EFFORT[difficulty]


def route_effort(
    objective: str,
    *,
    scope_paths: int = 1,
    capabilities: frozenset[str] | set[str] | None = None,
) -> tuple[Difficulty, str]:
    """`(difficulty, reasoning_effort)` for a task. The one call sites want."""
    difficulty = classify(objective, scope_paths=scope_paths, capabilities=capabilities)
    return difficulty, effort_for(difficulty)
