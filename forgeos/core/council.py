"""Running the panel that `should_convene_council` decides to convene.

`Router.should_convene_council()` returns a boolean and nothing anywhere acts
on it. The decision to spend the most expensive tool in the system was
implemented; the tool was not. This is that tool.

WHEN A COUNCIL IS THE RIGHT ANSWER, which is rarely. The router's own gate says
it: never when tests can adjudicate, and only on high risk with genuine
disagreement. Running both candidates and letting the suite decide is cheaper
AND more reliable than paying several models to argue, because argument
converges on the most confidently expressed answer rather than the correct one.
That is the whole reason this module is hard to reach.

WHAT MAKES THIS DIFFERENT FROM ASKING N MODELS AND TAKING THE MAJORITY:

1. ABSTENTION COUNTS AS INFORMATION, NOT A VOTE. A member that says "I cannot
   tell from this" is more useful than one that guesses, and counting it as a
   vote for whatever it happened to say destroys exactly that signal. High
   abstention means the question was underspecified -- a fact worth returning
   rather than averaging away.

2. AGREEMENT AMONG THE SAME FAMILY IS WEAKER EVIDENCE. LLM evaluators favour
   their own generations (Panickssery et al., NeurIPS 2024, arXiv 2404.13076),
   so three members from one vendor agreeing is not three independent
   confirmations. The verdict records family diversity so a reader can discount
   accordingly, and `verify.py` already applies the same reasoning to reviewers.

3. A TIE IS A RESULT. Returning the first option on a split is how a coin flip
   gets recorded as a decision. `Outcome.SPLIT` escalates to a human, which is
   what the caller wanted the council to avoid -- and knowing that it failed to
   avoid it is the point.

Deterministic given the same votes: no random tie-breaks, so a decision can be
replayed and argued with.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

# Below this share of non-abstaining members agreeing, there is no majority
# worth acting on. Two of three is a majority; two of five is not.
DEFAULT_MAJORITY = 0.5

# Above this share abstaining, the question is the problem, not the answer.
ABSTENTION_LIMIT = 0.5


class Outcome(str, Enum):
    DECIDED = "decided"          # a clear majority of those who answered
    SPLIT = "split"              # no majority -- a human decides
    ABSTAINED = "abstained"      # too few members could answer at all
    EMPTY = "empty"              # nobody voted


@dataclass(frozen=True)
class Vote:
    """One member's answer. `choice=None` is an abstention, deliberately not a
    sentinel string -- "unknown" as a choice would be counted as agreement by
    anything that groups on the value."""

    member: str
    choice: str | None
    reason: str = ""
    family: str = ""          # provider family, for the diversity discount
    confidence: float | None = None

    @property
    def abstained(self) -> bool:
        return self.choice is None


@dataclass
class Verdict:
    outcome: Outcome
    choice: str | None = None
    votes: list[Vote] = field(default_factory=list)
    tally: dict[str, int] = field(default_factory=dict)
    abstentions: int = 0
    families: set[str] = field(default_factory=set)
    reason: str = ""

    @property
    def answered(self) -> int:
        return len(self.votes) - self.abstentions

    @property
    def unanimous(self) -> bool:
        """Among those who ANSWERED. Unanimity including abstainers would be a
        different and much weaker claim."""
        return self.answered > 0 and len(self.tally) == 1

    @property
    def single_family(self) -> bool:
        """Every answering member came from one provider. Agreement here is
        weaker evidence than the count suggests."""
        return len(self.families) <= 1 and self.answered > 1

    @property
    def needs_human(self) -> bool:
        return self.outcome in (Outcome.SPLIT, Outcome.ABSTAINED, Outcome.EMPTY)

    def render(self) -> str:
        if self.outcome is Outcome.DECIDED:
            counts = ", ".join(f"{k}={v}" for k, v in sorted(self.tally.items()))
            caveat = ""
            if self.single_family:
                caveat = ("  (all from one provider family -- an evaluator favours "
                          "its own generations, arXiv 2404.13076)")
            return (f"decided: {self.choice}  [{counts}]"
                    f"{f', {self.abstentions} abstained' if self.abstentions else ''}"
                    f"{caveat}")
        return f"{self.outcome.value}: {self.reason}"


def tally(votes: list[Vote]) -> Verdict:
    """Count votes into a verdict. Pure -- no model call, no I/O.

    Separated from execution so the counting rules can be tested exhaustively
    without a provider, and so a caller can re-tally recorded votes later and
    get the identical answer.
    """
    if not votes:
        return Verdict(Outcome.EMPTY, reason="no members voted")

    abstentions = sum(1 for v in votes if v.abstained)
    answered = [v for v in votes if not v.abstained]
    families = {v.family for v in answered if v.family}

    if not answered:
        return Verdict(Outcome.ABSTAINED, votes=votes, abstentions=abstentions,
                       reason="every member abstained -- the question is underspecified")

    if abstentions / len(votes) > ABSTENTION_LIMIT:
        # More members could not answer than could. Whatever the minority
        # decided, the honest report is that the question did not admit one.
        return Verdict(
            Outcome.ABSTAINED, votes=votes, abstentions=abstentions, families=families,
            # Plain dict(Counter(...)). A comprehension of the form
            # `{v.choice: c for v.choice, c in ...}` parses, but its loop target
            # ASSIGNS to `v.choice` -- and `Vote` is frozen, so it raises at
            # runtime on every call that reaches this branch.
            tally=dict(Counter(v.choice for v in answered if v.choice is not None)),
            reason=f"{abstentions} of {len(votes)} abstained -- too few could answer",
        )

    counts: dict[str, int] = Counter(v.choice for v in answered if v.choice is not None)
    ranked = Counter(counts).most_common()
    top_choice, top_count = ranked[0]

    # A tie is a result, not something to break. Returning the first option
    # would record a coin flip as a decision.
    if len(ranked) > 1 and ranked[1][1] == top_count:
        tied = sorted(c for c, n in ranked if n == top_count and c is not None)
        return Verdict(
            Outcome.SPLIT, votes=votes, tally=dict(counts), abstentions=abstentions,
            families=families,
            reason=f"tied {top_count}-{top_count} between {' and '.join(tied)}",
        )

    if top_count / len(answered) <= DEFAULT_MAJORITY:
        return Verdict(
            Outcome.SPLIT, votes=votes, tally=dict(counts), abstentions=abstentions,
            families=families,
            reason=f"leading option has {top_count}/{len(answered)}, not a majority",
        )

    return Verdict(Outcome.DECIDED, choice=top_choice, votes=votes,
                   tally=dict(counts), abstentions=abstentions, families=families)


def convene(ask, members, *, question: str) -> Verdict:
    """Put `question` to every member and tally the answers.

    `ask(member, question) -> Vote` is supplied by the caller: this module does
    not know how to reach a model and must not learn. Keeping execution out
    means the counting rules above are testable without a provider, and that a
    council cannot quietly acquire the ability to spend money.

    A member that raises is recorded as an ABSTENTION rather than dropped. A
    council of five where two crashed is not a council of three -- the tally
    needs to know the difference, because it changes whether the majority was
    real.
    """
    votes: list[Vote] = []
    for member in members:
        try:
            vote = ask(member, question)
        except Exception as exc:
            votes.append(Vote(member=str(member), choice=None,
                              reason=f"failed: {type(exc).__name__}"))
            continue
        votes.append(vote if vote is not None
                     else Vote(member=str(member), choice=None, reason="no answer"))
    return tally(votes)
