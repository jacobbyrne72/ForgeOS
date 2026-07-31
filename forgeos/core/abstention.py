"""Stopping a worker that is not going to succeed, before it spends the proof.

The escalation boundary is where a harness wastes money twice. A worker that
cannot do the task keeps going until something trips -- a budget, an iteration
cap, a timeout -- and only then does the tier above get called. So the bill is
the failed attempt IN FULL, plus the successful one, plus whatever the failed
attempt's output cost to review.

The cheap fix is not a better model. It is asking earlier whether this attempt
is going anywhere, and two signals answer that without a model call:

  ABSTENTION. The worker says so. "I don't have access to X", "I can't find Y",
  "this needs a decision about Z" is information, and it arrives long before the
  budget does. A harness that treats it as ordinary output and lets the loop run
  is ignoring the one honest thing the worker said. Abstention is a SUCCESS of
  the worker's calibration and should be rewarded with a fast handoff, never
  punished with a retry that burns the rest of its budget.

  STALL. The worker does not say so, but nothing is happening: spend rising with
  no evidence recorded, or the same failure repeating verbatim. The second case
  matters most -- an identical error three times is not perseverance, it is a
  loop, and the next attempt costs exactly as much as the last one and produces
  the same thing.

WHAT THIS DELIBERATELY DOES NOT DO: escalate on its own. It returns a
recommendation and the reason. Escalation policy lives in `core/router.py`, and
a module that both detects trouble and spends money on it is one nobody can
reason about. `core/manager.py` already owns which failure classes may escalate
a tier at all -- SPECIFICATION and POLICY failures never should, because a
stronger model cannot fix an underspecified task and will happily bill you to
prove it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

# Verbatim repeats of the same failure. Two can be a flake; three is a loop, and
# the next attempt costs what the last one did for the same result.
DEFAULT_REPEAT_LIMIT = 3

# Spend past which "nothing to show yet" stops being a normal early phase.
DEFAULT_EVIDENCE_GRACE_MICROS = 50_000  # $0.05


class Recommendation(str, Enum):
    CONTINUE = "continue"          # nothing wrong; do not spend attention on it
    HANDOFF = "handoff"            # the worker abstained: reroute, do not retry
    STOP_LOOPING = "stop_looping"  # same failure repeating; another try buys nothing
    ASK_HUMAN = "ask_human"        # the blocker is a decision, not a capability


# Phrases where a worker is telling you it cannot proceed. Matched as whole
# phrases, case-insensitively, against the worker's own text -- never inferred
# from a low confidence score, which is a different and much noisier signal.
_ABSTAIN_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.I) for p in (
        r"\bi (?:do not|don't|cannot|can't) have (?:access|permission)\b",
        r"\bi (?:cannot|can't|am unable to) (?:find|locate|see)\b",
        r"\bno (?:such file|access to)\b",
        r"\bnot enough (?:context|information)\b",
        r"\bi need (?:more information|clarification|a decision)\b",
        r"\bthis (?:requires|needs) a (?:decision|human|choice)\b",
        r"\bambiguous\b.{0,40}\b(?:cannot|can't|unable)\b",
        r"\bi am not able to\b",
    )
)

# Abstentions that are a QUESTION rather than a capability gap. A stronger model
# cannot answer "which of these two behaviours did you want" -- only the person
# who asked can, and routing it upward buys an expensive guess.
_NEEDS_HUMAN = tuple(
    re.compile(p, re.I) for p in (
        r"\bneed (?:a decision|clarification|more information)\b",
        r"\brequires a (?:decision|human|choice)\b",
        r"\bwhich (?:one )?(?:do you|should i)\b",
    )
)


@dataclass(frozen=True)
class Judgement:
    recommendation: Recommendation
    reason: str
    quote: str = ""

    @property
    def should_stop(self) -> bool:
        return self.recommendation is not Recommendation.CONTINUE

    def render(self) -> str:
        quoted = f' — worker said: "{self.quote}"' if self.quote else ""
        return f"{self.recommendation.value}: {self.reason}{quoted}"


def _first_match(text: str, patterns) -> str:
    for pattern in patterns:
        found = pattern.search(text)
        if found:
            line = text[max(0, found.start() - 40): found.end() + 40].strip()
            return " ".join(line.split())[:140]
    return ""


def detect_abstention(text: str) -> Judgement | None:
    """The worker said it cannot proceed. `None` when it did not.

    Reads the worker's WORDS, never a confidence number. A low score means the
    model is unsure of its answer; an abstention means it is sure it cannot
    answer, and those call for opposite responses -- one wants another attempt,
    the other wants a different worker.
    """
    if not text or not text.strip():
        return None
    human_quote = _first_match(text, _NEEDS_HUMAN)
    if human_quote:
        return Judgement(
            Recommendation.ASK_HUMAN,
            "the blocker is a decision, and a stronger model can only guess at it",
            human_quote,
        )
    quote = _first_match(text, _ABSTAIN_PATTERNS)
    if quote:
        return Judgement(
            Recommendation.HANDOFF,
            "the worker stated it cannot proceed; rerouting now costs less than "
            "letting it spend the rest of its budget proving it",
            quote,
        )
    return None


def detect_repeat_failure(
    failures: list[str] | tuple[str, ...], *, limit: int = DEFAULT_REPEAT_LIMIT
) -> Judgement | None:
    """The same failure, verbatim, `limit` times running.

    Compared after whitespace normalisation only -- NOT fuzzily. A near-match
    heuristic would eventually stop a worker that was genuinely making progress
    through similar-looking errors, and stopping real work to save a call is the
    expensive mistake in this direction.
    """
    if len(failures) < limit:
        return None
    recent = [" ".join(f.split()) for f in failures[-limit:] if f and f.strip()]
    if len(recent) < limit or len(set(recent)) != 1:
        return None
    return Judgement(
        Recommendation.STOP_LOOPING,
        f"the same failure {limit} times running; the next attempt costs what the "
        f"last one did and produces the same thing",
        recent[-1][:140],
    )


def judge(
    *,
    text: str = "",
    failures: list[str] | tuple[str, ...] | None = None,
    spend_micros: int = 0,
    evidence_count: int = 0,
    repeat_limit: int = DEFAULT_REPEAT_LIMIT,
    evidence_grace_micros: int = DEFAULT_EVIDENCE_GRACE_MICROS,
) -> Judgement:
    """One verdict from every signal. Deterministic; no model call.

    Ordered by how much the signal is worth trusting: an explicit statement from
    the worker outranks an inference drawn about it.
    """
    stated = detect_abstention(text)
    if stated is not None:
        return stated

    looping = detect_repeat_failure(list(failures or []), limit=repeat_limit)
    if looping is not None:
        return looping

    if spend_micros > evidence_grace_micros and evidence_count == 0:
        return Judgement(
            Recommendation.STOP_LOOPING,
            f"${spend_micros / 1e6:.4f} spent with no evidence recorded — the "
            f"signature of a worker burning budget looking busy",
        )

    return Judgement(Recommendation.CONTINUE, "no abstention, loop or stall signal")
