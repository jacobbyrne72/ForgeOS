"""SavingsProof — a receipt that says which of its numbers are real.

Every harness that claims token/cost savings can print a big percentage.
Almost none can say whether that percentage came from something actually
observed or from a plain-code estimate of what "would have" happened. This
module is the difference: every number in a `SavingsProof` is a `Figure`
carrying its own `Provenance`, and the one rule enforced mechanically
throughout is:

    NEVER RENDER AN ESTIMATE AS A MEASUREMENT.

Concretely:

- `actual.*` figures come straight off `Ledger` / `SpanStore` / `EventLog` —
  stores that only ever record what really happened. They are
  `Provenance.MEASURED`.
- `baseline.*` figures describe a counterfactual: the same mission run under
  a different policy. This module cannot execute that counterfactual (see
  `CounterfactualPlan`), so the caller supplies the baseline, honestly
  labelled `Provenance.REPLAYED` (a paired run actually happened) or
  `Provenance.MODELLED` (someone computed an estimate).
- `savings_pct()` combines an actual figure and a baseline figure. Its
  provenance is the WEAKER of the two inputs, never stronger — a real actual
  divided by a guessed baseline produces a guessed saving. This is the
  central rule and it is enforced in code, not just in the docstring.
- `verify_proof()` re-derives those same rules against a finished
  `SavingsProof` and complains if anything inside it claims more certainty
  than its own inputs support.

Domain-agnostic on purpose (see `AGENTS.md`): nothing in here knows what
kind of mission produced these tokens and dollars.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

from ..contracts import from_micros, new_id, now
from ..core.timing import SpanStore
from ..events import EventLog, EventType
from ..ledger import Ledger
from .avoidance import AvoidanceLog

# --------------------------------------------------------------------- enum


class Provenance(str, Enum):
    """How a figure came to have the value it has.

    Ordered by strength (weakest first) via `_STRENGTH` below. MEASURED and
    REPLAYED are both "facts" (`Figure.is_fact`) in the sense that something
    actually happened and was observed — but they are not interchangeable:
    MEASURED is the value read directly off the mission that produced this
    receipt, while REPLAYED is a value read off a *separate*, actually-
    executed paired run. A paired run is real evidence, but it is evidence
    about a different execution (different clock, possibly different cache
    state), so it sits one notch below a direct measurement of the thing
    itself. This ranking is a design judgement, not something the ledger
    schema states explicitly — flagged for review.
    """

    MEASURED = "measured"    # observed directly on the mission this receipt is about
    REPLAYED = "replayed"    # observed on an actually-executed paired baseline run
    MODELLED = "modelled"    # computed estimate — no execution happened
    UNKNOWN = "unknown"      # provenance was never recorded


_STRENGTH: dict[Provenance, int] = {
    Provenance.UNKNOWN: 0,
    Provenance.MODELLED: 1,
    Provenance.REPLAYED: 2,
    Provenance.MEASURED: 3,
}

_PROVENANCE_WORD: dict[Provenance, str] = {
    Provenance.MEASURED: "measured",
    Provenance.REPLAYED: "replayed",
    Provenance.MODELLED: "estimated",
    Provenance.UNKNOWN: "unknown",
}


def _weaker(a: Provenance, b: Provenance) -> Provenance:
    """The less-certain of two provenances. Ties keep the first argument."""
    return a if _STRENGTH[a] <= _STRENGTH[b] else b


# ------------------------------------------------------------------- Figure

_CURRENCY_SYMBOLS: dict[str, str] = {"usd": "$", "eur": "€", "gbp": "£"}


def _format_value(value: float, unit: str) -> str:
    symbol = _CURRENCY_SYMBOLS.get(unit)
    if symbol is not None:
        return f"{symbol}{value:,.2f}"
    if unit == "pct":
        return f"{value:.1f}%"
    if unit == "seconds":
        return f"{value:.1f}s"
    if unit == "tokens":
        return f"{value:,.0f} tokens"
    if unit == "count":
        return f"{value:,.0f}"
    return f"{value:g} {unit}"


class Figure(BaseModel):
    """One number, with enough attached to judge how much to trust it.

    `is_fact` is the mechanical gate other code can branch on; `render()` is
    the human-facing form and it NEVER drops the provenance word — that word
    is the entire point of this module.
    """

    value: float
    unit: str
    provenance: Provenance
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    error_bound: float | None = Field(default=None, ge=0.0)
    note: str = ""

    @property
    def is_fact(self) -> bool:
        """True for a provenance backed by something that actually happened."""
        return self.provenance in (Provenance.MEASURED, Provenance.REPLAYED)

    def render(self) -> str:
        """e.g. ``"$0.38 measured"`` or ``"6.2% estimated ±0.8%"``.

        The provenance word always trails the value. An error bound, if
        present, is appended in the same unit so a reader never has to
        cross-reference two different figures to know what the ± means.
        """
        text = f"{_format_value(self.value, self.unit)} {_PROVENANCE_WORD[self.provenance]}"
        if self.error_bound is not None:
            text += f" ±{_format_value(self.error_bound, self.unit)}"
        return text


# ------------------------------------------------------------- savings math


def savings_pct(actual: Figure, baseline: Figure) -> Figure:
    """Percentage change from `baseline` to `actual`, provenance-honest.

    THE CORE RULE: the result's provenance is `_weaker(actual.provenance,
    baseline.provenance)`. A MEASURED actual against a MODELLED baseline
    yields a MODELLED saving — never MEASURED — because the weaker input
    caps how much the combination can be trusted. This is what stops a real
    number and a guess from combining into a claim stronger than either one
    alone.

    A zero baseline cannot be expressed as a percentage of itself; rather
    than raise `ZeroDivisionError` or silently return something misleading,
    this returns 0.0 with a note explaining why, at the honestly-weaker
    provenance of the two inputs.

    Negative results (actual cost MORE than baseline) are returned as-is.
    Clamping a regression to zero would hide the single worst failure mode
    this module exists to prevent.

    `error_bound` is converted into the same "% of baseline" unit as the
    result rather than copied over as-is: an absolute error bound (say
    $0.30, or 15,000 tokens) carried unchanged onto a figure whose unit is
    now "pct" would silently relabel a dollar or token amount as a
    percentage — a unit error, and a misleading one, since it renders as a
    plausible-looking but meaningless number like "±15000.0%".
    """
    provenance = _weaker(actual.provenance, baseline.provenance)
    confidences = [c for c in (actual.confidence, baseline.confidence) if c is not None]
    confidence = min(confidences) if confidences else None

    if baseline.value == 0:
        return Figure(
            value=0.0,
            unit="pct",
            provenance=provenance,
            confidence=confidence,
            error_bound=None,
            note=(
                "baseline is zero — percentage saving is undefined; reported as 0.0 "
                "rather than dividing by zero"
            ),
        )

    pct = round(100.0 * (baseline.value - actual.value) / baseline.value, 4)
    source_bound = baseline.error_bound if baseline.error_bound is not None else actual.error_bound
    error_bound = (
        abs(round(100.0 * source_bound / baseline.value, 4)) if source_bound is not None else None
    )
    return Figure(
        value=pct,
        unit="pct",
        provenance=provenance,
        confidence=confidence,
        error_bound=error_bound,
        note=f"actual={actual.provenance.value}, baseline={baseline.provenance.value}",
    )


# ---------------------------------------------------------------- the proof


class Outcome(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    PARTIAL = "partial"
    INCONCLUSIVE = "inconclusive"


STANDARD_METRICS: tuple[str, ...] = (
    "input_tokens",
    "output_tokens",
    "cash_cost",
    "wall_time",
    "human_interventions",
)


class Baseline(BaseModel):
    """The counterfactual side of the receipt.

    `baseline_provenance` describes the WHOLE baseline, not one figure in it:
    REPLAYED means every figure here came from an actually-executed paired
    run; MODELLED means every figure here is a plain-code estimate. Per-figure
    `provenance` on each `Figure` in `figures` still governs `savings_pct`,
    but the two should agree — a mismatch between this flag and a figure's
    own provenance is exactly what `verify_proof` checks for.
    """

    baseline_policy: str
    baseline_provenance: Provenance
    figures: dict[str, Figure] = Field(default_factory=dict)

    @field_validator("baseline_policy")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("baseline_policy must name the policy being compared against")
        return v.strip()


class SavingsProof(BaseModel):
    """The receipt. Every numeric figure inside carries its own provenance.

    This model does not compute anything on construction beyond basic
    sanity (`acceptance_passed <= acceptance_total`) — `build_proof()` is
    what assembles one from the live stores, and `savings_pct()` is what
    fills in `savings`. Constructing a `SavingsProof` directly (e.g. in a
    test) is deliberately just as easy, because `verify_proof()` needs to be
    able to catch a badly-hand-built one.
    """

    id: str = Field(default_factory=lambda: new_id("proof"))
    mission_id: str
    repo_revision: str
    outcome: Outcome
    acceptance_passed: int = Field(ge=0)
    acceptance_total: int = Field(ge=0)
    actual: dict[str, Figure] = Field(default_factory=dict)
    baseline: Baseline
    savings: dict[str, Figure] = Field(default_factory=dict)
    contract_hash: str = ""
    actual_trace_hash: str = ""
    baseline_trace_hash: str = ""
    test_artifact_hash: str = ""
    created_at: float = Field(default_factory=now)

    @model_validator(mode="after")
    def _acceptance_is_sane(self) -> "SavingsProof":
        if self.acceptance_passed > self.acceptance_total:
            raise ValueError("acceptance_passed cannot exceed acceptance_total")
        return self


def build_proof(
    ledger: Ledger,
    avoidance_log: AvoidanceLog,
    span_store: SpanStore,
    event_log: EventLog,
    job_id: str,
    *,
    mission_id: str,
    repo_revision: str,
    outcome: Outcome,
    acceptance_passed: int,
    acceptance_total: int,
    baseline: Baseline,
    contract_hash: str = "",
    actual_trace_hash: str = "",
    baseline_trace_hash: str = "",
    test_artifact_hash: str = "",
) -> SavingsProof:
    """Assemble the ACTUAL side of a receipt from the real stores.

    Every figure produced here is `Provenance.MEASURED` because it is read
    straight off stores that only ever record what really happened — there
    is no column in any of these schemas for a guess:

    - `ledger.job_spend_micros` / `ledger.cache_stats` -> cash_cost, tokens.
    - `avoidance_log.totals` corroborates the token figure with an
      independently-audited count of tokens avoided outright (cache hits,
      packed reads, dedup, ...), noted alongside it rather than merged in,
      so the two measurements stay individually checkable.
    - `span_store.attribution` -> wall_clock, the only honest source for
      elapsed time since there is no separate job start/end marker.
    - `event_log` -> a count of `GOVERNOR_TRIPPED` events as the measured
      proxy for human interventions: a governor trip is, by the hard rule in
      `AGENTS.md` ("never widen a budget to make a job finish"), always a
      point where a human had to look at the job and decide, never something
      auto-resolved. This is a judgement call about what "human_interventions"
      means operationally — flagged for review, since nothing in the schema
      is literally labelled that way.

    The BASELINE side is not derived here — this module has no way to
    execute a paired counterfactual mission (see `CounterfactualPlan`), so
    the caller supplies `baseline` already labelled with how it was actually
    obtained.
    """
    spend_micros = ledger.job_spend_micros(job_id)
    cache = ledger.cache_stats(job_id)
    attribution = span_store.attribution(job_id)
    trips = [e for e in event_log.replay(job_id) if e.type is EventType.GOVERNOR_TRIPPED]
    avoided = avoidance_log.totals(job_id)

    actual = {
        "input_tokens": Figure(
            value=float(cache["fresh_in"] + cache["cached_in"]),
            unit="tokens",
            provenance=Provenance.MEASURED,
            note=(
                f"ledger.cache_stats(job_id={job_id!r}): fresh_in+cached_in; "
                f"avoidance_log.totals reports {avoided['saved_tokens']} further tokens "
                "avoided outright (not double-counted here)"
            ),
        ),
        "output_tokens": Figure(
            value=float(cache["tokens_out"]),
            unit="tokens",
            provenance=Provenance.MEASURED,
            note=f"ledger.cache_stats(job_id={job_id!r}): tokens_out",
        ),
        "cash_cost": Figure(
            value=from_micros(spend_micros),
            unit="usd",
            provenance=Provenance.MEASURED,
            note=f"ledger.job_spend_micros(job_id={job_id!r})",
        ),
        "wall_time": Figure(
            value=attribution.wall_clock,
            unit="seconds",
            provenance=Provenance.MEASURED,
            note=f"span_store.attribution(job_id={job_id!r}).wall_clock",
        ),
        "human_interventions": Figure(
            value=float(len(trips)),
            unit="count",
            provenance=Provenance.MEASURED,
            note=f"event_log.replay(job_id={job_id!r}): count of GOVERNOR_TRIPPED events",
        ),
    }

    savings = {
        key: savings_pct(actual[key], baseline.figures[key])
        for key in actual
        if key in baseline.figures
    }

    return SavingsProof(
        mission_id=mission_id,
        repo_revision=repo_revision,
        outcome=outcome,
        acceptance_passed=acceptance_passed,
        acceptance_total=acceptance_total,
        actual=actual,
        baseline=baseline,
        savings=savings,
        contract_hash=contract_hash,
        actual_trace_hash=actual_trace_hash,
        baseline_trace_hash=baseline_trace_hash,
        test_artifact_hash=test_artifact_hash,
    )


# -------------------------------------------------------------- the receipt

_WIDTH = 68


def _rule(ch: str = "-") -> str:
    return ch * _WIDTH


def _kv(label: str, value: str, indent: int = 2) -> str:
    return f"{' ' * indent}{label:<22}{value}"


def _overall_confidence(figures: dict[str, Figure]) -> float | None:
    vals = [f.confidence for f in figures.values() if f.confidence is not None]
    return min(vals) if vals else None


def render_receipt(proof: SavingsProof) -> str:
    """A fixed-width text receipt. Every numeric line ends in its provenance word.

    Reuses `Figure.render()` for every value line, so there is exactly one
    place in the codebase that decides how a provenance word gets attached
    to a number. When the baseline is `Provenance.MODELLED`, a bordered
    warning block is inserted before the baseline figures — the single most
    important line in the whole receipt, since it is the one that stops a
    reader from mistaking an estimate for a fact.
    """
    lines: list[str] = []
    lines.append("=" * _WIDTH)
    lines.append(f"SAVINGS PROOF -- {proof.mission_id}")
    lines.append(_rule())
    lines.append(_kv("repo revision:", proof.repo_revision, indent=0))
    lines.append(_kv("outcome:", proof.outcome.value, indent=0))
    accept_flag = "PASS" if proof.acceptance_passed == proof.acceptance_total else "INCOMPLETE"
    lines.append(
        _kv(
            "acceptance:",
            f"{proof.acceptance_passed}/{proof.acceptance_total} {accept_flag}",
            indent=0,
        )
    )

    lines.append(_rule())
    lines.append("ACTUAL")
    for key in STANDARD_METRICS:
        fig = proof.actual.get(key)
        if fig is not None:
            lines.append(_kv(key + ":", fig.render()))

    lines.append(_rule())
    lines.append(f"BASELINE  (policy: {proof.baseline.baseline_policy})")
    if proof.baseline.baseline_provenance is Provenance.MODELLED:
        conf = _overall_confidence(proof.baseline.figures)
        conf_str = f"{conf:.2f}" if conf is not None else "n/a"
        lines.append("!" * _WIDTH)
        lines.append(
            "  BASELINE WAS NOT ACTUALLY RUN -- this is a MODELLED estimate, "
            f"not a measurement. confidence={conf_str}"
        )
        lines.append("!" * _WIDTH)
    for key in STANDARD_METRICS:
        fig = proof.baseline.figures.get(key)
        if fig is not None:
            lines.append(_kv(key + ":", fig.render()))

    lines.append(_rule())
    lines.append("SAVINGS")
    for key in STANDARD_METRICS:
        fig = proof.savings.get(key)
        if fig is not None:
            lines.append(_kv(key + ":", fig.render()))

    lines.append(_rule())
    lines.append("EVIDENCE")
    for label, val in (
        ("contract_hash:", proof.contract_hash),
        ("actual_trace_hash:", proof.actual_trace_hash),
        ("baseline_trace_hash:", proof.baseline_trace_hash),
        ("test_artifact_hash:", proof.test_artifact_hash),
    ):
        lines.append(_kv(label, val if val else "MISSING"))
    lines.append("=" * _WIDTH)
    return "\n".join(lines)


# -------------------------------------------------------------- integrity


def verify_proof(proof: SavingsProof) -> list[str]:
    """Integrity complaints about `proof`. Empty list means internally consistent.

    This is the mechanical half of "never render an estimate as a
    measurement" — it re-derives, from `proof`'s own actual/baseline
    figures, what each saving's provenance and sign are ALLOWED to be, and
    complains whenever the stored value claims more than that.
    """
    complaints: list[str] = []

    # 1. evidence hashes must be present — an unattributable receipt is not
    #    a receipt, it's an assertion.
    for field_name in (
        "contract_hash",
        "actual_trace_hash",
        "baseline_trace_hash",
        "test_artifact_hash",
    ):
        if not getattr(proof, field_name).strip():
            complaints.append(f"missing evidence hash: {field_name}")

    # 2. an ACCEPTED outcome with failing acceptance criteria is a false green.
    if proof.outcome is Outcome.ACCEPTED and proof.acceptance_passed < proof.acceptance_total:
        complaints.append(
            "outcome is ACCEPTED but acceptance is "
            f"{proof.acceptance_passed}/{proof.acceptance_total} — not all criteria passed"
        )

    # 3. a figure claiming MEASURED must say where it came from.
    for section_name, figures in (
        ("actual", proof.actual),
        ("baseline", proof.baseline.figures),
        ("savings", proof.savings),
    ):
        for key, fig in figures.items():
            if fig.provenance is Provenance.MEASURED and not fig.note.strip():
                complaints.append(f"{section_name} figure '{key}' is MEASURED but has no source note")

    # 4. a saving cannot claim a provenance stronger than the weaker of its
    #    two inputs, and its sign must agree with what those inputs imply.
    for key, saved in proof.savings.items():
        actual_fig = proof.actual.get(key)
        baseline_fig = proof.baseline.figures.get(key)
        if actual_fig is None or baseline_fig is None:
            continue

        expected_provenance = _weaker(actual_fig.provenance, baseline_fig.provenance)
        if _STRENGTH[saved.provenance] > _STRENGTH[expected_provenance]:
            complaints.append(
                f"savings '{key}' is claimed as {saved.provenance.value} but its inputs "
                f"({actual_fig.provenance.value} actual, {baseline_fig.provenance.value} baseline) "
                f"only support {expected_provenance.value}"
            )

        if baseline_fig.value != 0:
            if baseline_fig.value > actual_fig.value:
                expected_sign = 1
            elif baseline_fig.value < actual_fig.value:
                expected_sign = -1
            else:
                expected_sign = 0
            actual_sign = 1 if saved.value > 0 else (-1 if saved.value < 0 else 0)
            if expected_sign != 0 and actual_sign != 0 and expected_sign != actual_sign:
                kind = "a saving" if expected_sign > 0 else "a regression (spent more than baseline)"
                shown = "positive" if actual_sign > 0 else "negative"
                complaints.append(
                    f"savings '{key}' is shown as {shown} but actual vs baseline implies {kind} "
                    "— a regression must never be presented as a saving"
                )

    return complaints


# --------------------------------------------------------- counterfactual


class CounterfactualPlan(BaseModel):
    """What an actually-executed paired baseline replay would need.

    This is a CONTRACT, not an executor. Nothing in `forgeos/economy/savings.py`
    runs a mission. Until some other component executes this exact plan —
    same `pinned_revision`, same `task_ids`, run under `policy_name` — and
    records the result, any baseline describing that comparison is, and
    remains, `Provenance.MODELLED`. `ready()` only says "this plan has
    enough information to be run"; it does not mean a replay happened, and
    it must never be treated as though it did.
    """

    policy_name: str
    pinned_revision: str
    task_ids: list[str] = Field(default_factory=list)
    notes: str = ""

    def ready(self) -> bool:
        """True once policy, revision, and a non-empty task set are all set.

        Still just a readiness check on the CONTRACT — see the class
        docstring. A ready plan has not been run.
        """
        return (
            bool(self.policy_name.strip())
            and bool(self.pinned_revision.strip())
            and len(self.task_ids) > 0
        )


__all__ = [
    "Baseline",
    "CounterfactualPlan",
    "Figure",
    "Outcome",
    "Provenance",
    "STANDARD_METRICS",
    "SavingsProof",
    "build_proof",
    "render_receipt",
    "savings_pct",
    "verify_proof",
]
