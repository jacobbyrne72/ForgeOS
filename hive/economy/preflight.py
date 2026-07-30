"""Count and price a call BEFORE it is made, so the call can be refused.

A ledger check after the fact can only report an overspend; it can never
prevent one — the tokens are already gone. That asymmetry is the whole reason
this module exists. The preflight estimate is the one place where "no" still
costs nothing.

Counts are labelled honestly. `exact=True` means the target model's own
tokeniser counted the text; anything else is an estimate and is marked as
such, because a budget decision made on a number that pretends to be measured
is a budget decision made on a lie.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from ..catalog import ModelCard

try:  # tiktoken is expected, but its absence must degrade to estimates, not crash.
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None  # type: ignore[assignment]

# Rough rule of thumb for English/code when no tokeniser is available.
_CHARS_PER_TOKEN = 4
ESTIMATE_ENCODING = "estimate:chars/4"


class TokenCount(BaseModel):
    """A token count that admits how it was produced."""

    tokens: int = Field(ge=0)
    exact: bool
    encoding_used: str


def count_tokens(text: str, model: str = "") -> TokenCount:
    """Count tokens with the model's own tokeniser when tiktoken has it.

    Never raises: an unknown model, an empty model name, or a missing tiktoken
    all degrade to a ceil(chars/4) estimate flagged `exact=False`.
    """
    if tiktoken is not None and model:
        try:
            enc = tiktoken.encoding_for_model(model)
        except (KeyError, ValueError):
            enc = None
        if enc is not None:
            # disallowed_special=() so text QUOTING a special token (e.g. a doc
            # containing "<|endoftext|>") is counted, not crashed on.
            return TokenCount(
                tokens=len(enc.encode(text, disallowed_special=())),
                exact=True,
                encoding_used=enc.name,
            )
    return TokenCount(
        tokens=-(-len(text) // _CHARS_PER_TOKEN),  # ceil; 0 only for empty text
        exact=False,
        encoding_used=ESTIMATE_ENCODING,
    )


class CallEstimate(BaseModel):
    """What one prospective call is expected to cost, computed before it is made."""

    model_ref: str
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    usd_micros: int = Field(ge=0)
    fits_context: bool
    exact: bool  # was tokens_in counted by the model's tokeniser, or estimated?


def estimate_call(prompt: str, expected_output_tokens: int, card: ModelCard) -> CallEstimate:
    """Price a prospective call against a ModelCard.

    `expected_output_tokens` should be the caller's declared output ceiling
    (max_tokens), not a hope — pricing the cap is what makes the estimate an
    upper bound rather than wishful thinking.
    """
    counted = count_tokens(prompt, card.model_id)
    return CallEstimate(
        model_ref=card.ref,
        tokens_in=counted.tokens,
        tokens_out=expected_output_tokens,
        usd_micros=card.cost_micros(counted.tokens, expected_output_tokens),
        fits_context=card.fits(counted.tokens),
        exact=counted.exact,
    )


class Decision(str, Enum):
    ALLOW = "allow"
    REFUSE_BUDGET = "refuse_budget"
    REFUSE_CONTEXT = "refuse_context"


class CallRefused(RuntimeError):
    """Raised by PreflightVerdict.raise_if_refused(). A refusal is a brake, not advice."""

    def __init__(self, verdict: PreflightVerdict):
        super().__init__(verdict.reason)
        self.verdict = verdict


class PreflightVerdict(BaseModel):
    decision: Decision
    reason: str
    estimate: CallEstimate

    @property
    def allowed(self) -> bool:
        return self.decision is Decision.ALLOW

    def raise_if_refused(self) -> None:
        """Turn a refusal into an exception so a call site cannot quietly ignore it."""
        if not self.allowed:
            raise CallRefused(self)


def check(estimate: CallEstimate, remaining_micros: int, max_context: int = 0) -> PreflightVerdict:
    """ALLOW or REFUSE a prospective call. A refusal means the call must not be made.

    Context is checked before budget — a payload that cannot physically fit is
    refused regardless of how much money remains. Spending exactly to the cap
    is allowed; exceeding it by one microdollar is not.
    """
    tag = "" if estimate.exact else " [token count estimated]"
    if not estimate.fits_context:
        return PreflightVerdict(
            decision=Decision.REFUSE_CONTEXT,
            reason=f"{estimate.tokens_in} input tokens exceed {estimate.model_ref}'s context window{tag}",
            estimate=estimate,
        )
    if max_context and estimate.tokens_in > max_context:
        return PreflightVerdict(
            decision=Decision.REFUSE_CONTEXT,
            reason=f"{estimate.tokens_in} input tokens exceed the {max_context}-token context cap{tag}",
            estimate=estimate,
        )
    if estimate.usd_micros > remaining_micros:
        return PreflightVerdict(
            decision=Decision.REFUSE_BUDGET,
            reason=(
                f"estimated cost {estimate.usd_micros} usd_micros exceeds remaining budget "
                f"{remaining_micros} usd_micros{tag}"
            ),
            estimate=estimate,
        )
    return PreflightVerdict(
        decision=Decision.ALLOW,
        reason=(
            f"estimated {estimate.usd_micros} usd_micros ({estimate.tokens_in} in / "
            f"{estimate.tokens_out} out) within remaining {remaining_micros} usd_micros{tag}"
        ),
        estimate=estimate,
    )


__all__ = [
    "CallEstimate",
    "CallRefused",
    "Decision",
    "ESTIMATE_ENCODING",
    "PreflightVerdict",
    "TokenCount",
    "check",
    "count_tokens",
    "estimate_call",
]
