"""Model-trait registry — how to prompt each model for the countermeasure its
own failure mode needs.

The concrete problem: a model can be so capable that it works for a long
stretch, over-engineers, and delivers far more than the task asked for — a
real, expensive failure mode, because capability without a stop condition
just burns budget producing more than was requested. Other models fail the
opposite way: terse to the point of skipping requirements, or ignoring an
output schema entirely. Different models need different countermeasures, so
this module records, per model, HOW TO PROMPT IT and what to do about its
specific tendency.

CRITICAL EPISTEMIC RULE, mirrored from `forgeos.knowledge.claims`: a trait is a
CLAIM, not a fact. Vendor benchmark numbers are marketing; a colleague's (or
this operator's own) impression is anecdote. So every `Trait` carries
`TraitEvidence`:

    MEASURED     from this harness's own ledger outcomes
    DOCUMENTED   vendor/published, cited in `source`
    OBSERVED     an impression, explicitly unverified

MEASURED always overrides DOCUMENTED and OBSERVED for the same tendency —
`ModelProfile.dominant_trait` enforces that ordering, not the caller. An
OBSERVED trait is marked unverified everywhere it surfaces (`Trait.unverified`,
`Trait.label`, `ModelProfile.describe`) and it must never become a silent,
permanent hard rule: `record_outcome` is the unlearning path, exactly the
gate `forgeos/knowledge/claims.py` enforces for outside claims — this harness's
own contradicting results downgrade or remove a trait rather than piling on
top of it. A registry that cannot unlearn is worse than none, because a wrong
trait permanently mis-prompts a model that may be perfectly fine.
"""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, Field

from ..core.router import DEFAULT_EFFORT, MAX_OUTPUT_BY_EFFORT

# ~4 chars/token, the same rough approximation forgeos.prompts.prefix uses to
# size a prefix for a sanity check rather than to bill it.
_CHARS_PER_TOKEN = 4

# An over-engineering model gets its output ceiling cut, not just capped at
# the normal rung -- the whole point is a HARD cap, tighter than usual.
_OVER_ENGINEER_CAP_FRACTION = 0.5

# How much a confirming/contradicting outcome moves confidence per call.
# Contradiction moves faster than confirmation: our own harness disagreeing
# with a trait is stronger signal than one more data point agreeing with it,
# because the trait was already assumed true going in.
_CONFIRM_CONFIDENCE_STEP = 0.15
_CONTRADICT_CONFIDENCE_STEP = 0.3


class Tendency(str, Enum):
    """A named failure or success mode a model tends toward."""

    OVER_ENGINEERS = "over_engineers"
    UNDER_DELIVERS = "under_delivers"
    IGNORES_SCHEMA = "ignores_schema"
    VERBOSE_PROSE = "verbose_prose"
    STOPS_EARLY = "stops_early"
    NEEDS_EXPLICIT_STOP = "needs_explicit_stop"
    STRONG_TOOL_USE = "strong_tool_use"
    WEAK_TOOL_USE = "weak_tool_use"
    LITERAL_INSTRUCTION_FOLLOWER = "literal_instruction_follower"
    DRIFTS_OFF_TASK = "drifts_off_task"


class TraitEvidence(str, Enum):
    """How much a trait may be trusted. Order is precedence, not chronology."""

    MEASURED = "measured"      # this harness's own ledger outcomes
    DOCUMENTED = "documented"  # vendor/published, cited in `source`
    OBSERVED = "observed"      # impression, explicitly unverified


# Index = precedence. Higher wins when the same tendency is claimed twice
# with different evidence -- this is the ONE place that ordering is decided.
_EVIDENCE_RANK: dict[TraitEvidence, int] = {
    TraitEvidence.OBSERVED: 0,
    TraitEvidence.DOCUMENTED: 1,
    TraitEvidence.MEASURED: 2,
}
# Inverse of _EVIDENCE_RANK, used by record_outcome to step evidence down a
# rank on contradiction rather than only touching confidence.
_EVIDENCE_BY_RANK: list[TraitEvidence] = [
    TraitEvidence.OBSERVED,
    TraitEvidence.DOCUMENTED,
    TraitEvidence.MEASURED,
]


class Trait(BaseModel):
    """One claim about one model's behavior."""

    tendency: Tendency
    evidence: TraitEvidence
    note: str
    source: str = ""
    observations: int = 0
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @property
    def unverified(self) -> bool:
        """True for an impression that has not been corroborated or measured.

        This must stay a live property, not a cached flag set at construction,
        so that promoting `evidence` (via `record_outcome`) is the only way to
        clear it -- there is no back door that marks a trait verified without
        the harness actually having measured it.
        """
        return self.evidence is TraitEvidence.OBSERVED

    @property
    def label(self) -> str:
        """The tag every surface (report, dashboard, log) must show for this trait."""
        tag = " [UNVERIFIED]" if self.unverified else ""
        return f"{self.tendency.value}{tag}"


class PromptDiscipline(BaseModel):
    """The countermeasures to append to a prompt for a given model + task."""

    max_output_tokens: int | None = None
    require_explicit_stop_condition: bool = False
    require_schema_echo: bool = False
    forbid_speculative_work: bool = False
    demand_smallest_change: bool = False
    checkpoint_every_n_steps: int | None = None
    extra_instructions: list[str] = Field(default_factory=list)


class ModelProfile(BaseModel):
    """Everything on record for one model: its traits and derived discipline."""

    model_ref: str
    traits: list[Trait] = Field(default_factory=list)
    discipline: PromptDiscipline = Field(default_factory=PromptDiscipline)
    notes: str = ""

    def dominant_trait(self, tendency: Tendency) -> Trait | None:
        """The highest-evidence trait on record for `tendency`, or None.

        When the same tendency has been claimed more than once (say an
        OBSERVED impression and a later MEASURED result) this is the one
        place that decides which wins -- MEASURED, always, regardless of
        which was recorded first.
        """
        matches = [t for t in self.traits if t.tendency is tendency]
        if not matches:
            return None
        return max(matches, key=lambda t: (_EVIDENCE_RANK[t.evidence], t.confidence))

    def has(self, tendency: Tendency) -> bool:
        return self.dominant_trait(tendency) is not None

    def describe(self) -> str:
        """Human-readable summary for dashboards/logs.

        An unverified trait is labelled here too -- "everywhere it surfaces"
        includes the plain-text summary a human reviewing routing decisions
        would read, not just the `Trait` object itself.
        """
        if not self.traits:
            return f"{self.model_ref}: no traits recorded"
        lines = [f"{self.model_ref}:"]
        for t in self.traits:
            lines.append(
                f"  - {t.label} ({t.evidence.value}, confidence={t.confidence:.2f}): {t.note}"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Seed data. Every seed trait is an OBSERVED impression -- including the
# operator's own, about a model that works for extended stretches and
# over-builds when left unchecked. No benchmark figure is invented here and
# no claim is attributed to a vendor document nobody has read; that is what
# DOCUMENTED evidence is for, once such a source is actually cited.
# --------------------------------------------------------------------------

SEED_MODEL_PROFILES: list[ModelProfile] = [
    ModelProfile(
        model_ref="codex/sol-5.6",
        traits=[
            Trait(
                tendency=Tendency.OVER_ENGINEERS,
                evidence=TraitEvidence.OBSERVED,
                note=(
                    "Operator's own impression: this model is capable enough to keep "
                    "working for extended stretches, and left unchecked it over-builds, "
                    "delivering far more than the task asked for. Not yet measured by "
                    "this harness's own ledger -- treat as an unverified impression, "
                    "not a rule."
                ),
                source="operator, direct session impression",
                observations=1,
                confidence=0.5,
            ),
        ],
        notes="Seed entry from an operator observation. See trait.unverified before trusting it.",
    ),
]


def _fresh_registry() -> dict[str, ModelProfile]:
    """Deep copies, so mutating the working registry never touches the seed constants."""
    return {p.model_ref: p.model_copy(deep=True) for p in SEED_MODEL_PROFILES}


_REGISTRY: dict[str, ModelProfile] = _fresh_registry()


def reset_registry() -> None:
    """Restore the working registry to its seed state. For test isolation."""
    _REGISTRY.clear()
    _REGISTRY.update(_fresh_registry())


def register_profile(profile: ModelProfile) -> None:
    _REGISTRY[profile.model_ref] = profile


def get_profile(model_ref: str) -> ModelProfile | None:
    return _REGISTRY.get(model_ref)


def all_profiles() -> list[ModelProfile]:
    return list(_REGISTRY.values())


def _profile_for(model_ref: str) -> ModelProfile:
    """A registered profile, or an empty one. Never a KeyError.

    An unknown model is the common case for a harness that reaches new
    providers constantly -- it must produce safe defaults, not a crash.
    """
    return _REGISTRY.get(model_ref) or ModelProfile(
        model_ref=model_ref,
        notes="no traits recorded for this model -- conservative defaults apply",
    )


# --------------------------------------------------------------------- derive


def discipline_for(model_ref: str, task_effort: str = DEFAULT_EFFORT) -> PromptDiscipline:
    """Derive the countermeasures from whatever traits are on record.

    An unregistered model gets the router's normal output ceiling for this
    effort level plus a stop condition -- the one countermeasure cheap enough
    to apply blind to a model we know nothing about. A model on record as
    OVER_ENGINEERS gets a genuinely hard cap (tighter than the normal rung),
    a speculative-work ban, a smallest-change demand, and an explicit stop
    condition; the other tendencies each drive the one countermeasure that
    actually addresses them.
    """
    profile = _profile_for(model_ref)
    baseline_cap = MAX_OUTPUT_BY_EFFORT.get(task_effort, MAX_OUTPUT_BY_EFFORT[DEFAULT_EFFORT])

    discipline = PromptDiscipline(
        max_output_tokens=baseline_cap,
        # No traits on record at all -- an unknown model -- is exactly the
        # case with the least information, so ask for a stop condition
        # regardless. A model that already has SOME trait on record gets a
        # more specific answer from the checks below.
        require_explicit_stop_condition=not profile.traits,
    )

    if profile.dominant_trait(Tendency.OVER_ENGINEERS) is not None:
        discipline.max_output_tokens = int(baseline_cap * _OVER_ENGINEER_CAP_FRACTION)
        discipline.forbid_speculative_work = True
        discipline.demand_smallest_change = True
        discipline.require_explicit_stop_condition = True

    if profile.dominant_trait(Tendency.NEEDS_EXPLICIT_STOP) is not None:
        discipline.require_explicit_stop_condition = True

    if profile.dominant_trait(Tendency.IGNORES_SCHEMA) is not None:
        discipline.require_schema_echo = True

    if (
        profile.dominant_trait(Tendency.STOPS_EARLY) is not None
        or profile.dominant_trait(Tendency.UNDER_DELIVERS) is not None
    ):
        discipline.checkpoint_every_n_steps = discipline.checkpoint_every_n_steps or 1

    if profile.dominant_trait(Tendency.DRIFTS_OFF_TASK) is not None:
        discipline.checkpoint_every_n_steps = discipline.checkpoint_every_n_steps or 2
        discipline.require_explicit_stop_condition = True

    if profile.dominant_trait(Tendency.VERBOSE_PROSE) is not None:
        discipline.extra_instructions.append("No filler prose; answer, then stop.")

    if profile.dominant_trait(Tendency.WEAK_TOOL_USE) is not None:
        discipline.extra_instructions.append("Use real tool calls, not descriptions of them.")

    return discipline


def discipline_instructions(discipline: PromptDiscipline) -> list[str]:
    """The imperative lines to append to a prompt tail.

    Kept deliberately short: this rides on every call to a model with this
    discipline, cache hit or not, so verbosity here is a permanent, repeating
    tax rather than a one-time cost.
    """
    lines: list[str] = []
    if discipline.require_explicit_stop_condition:
        lines.append("Stop at acceptance criteria met. No over-polishing.")
    if discipline.forbid_speculative_work:
        lines.append("No speculative work. Do only what was asked.")
    if discipline.demand_smallest_change:
        lines.append("Smallest change that satisfies the ask.")
    if discipline.require_schema_echo:
        lines.append("Echo the output schema before answering.")
    if discipline.checkpoint_every_n_steps:
        lines.append(f"Checkpoint every {discipline.checkpoint_every_n_steps} step(s).")
    if discipline.max_output_tokens:
        lines.append(f"Output cap: {discipline.max_output_tokens} tokens.")
    lines.extend(discipline.extra_instructions)
    return lines


def instructions_token_estimate(lines: list[str]) -> int:
    """Rough size of a discipline_instructions() block, ~4 chars/token."""
    return sum(len(line) for line in lines) // _CHARS_PER_TOKEN


# ---------------------------------------------------------------- unlearning


def record_outcome(model_ref: str, tendency: Tendency, *, confirmed: bool) -> Trait | None:
    """Update the registry from this harness's own results.

    `confirmed=True` is this harness observing the tendency itself: the trait
    (creating one if none existed) is promoted to MEASURED and its confidence
    rises. `confirmed=False` is this harness's own result CONTRADICTING the
    trait -- confidence falls and evidence steps down a rank (MEASURED ->
    DOCUMENTED -> OBSERVED); once confidence bottoms out the trait is removed
    entirely. That removal path is the point: a trait this harness's own data
    no longer supports must not keep mis-prompting the model forever.
    """
    profile = _REGISTRY.setdefault(model_ref, ModelProfile(model_ref=model_ref))
    existing = profile.dominant_trait(tendency)

    if confirmed:
        if existing is None:
            trait = Trait(
                tendency=tendency,
                evidence=TraitEvidence.MEASURED,
                note="confirmed by this harness's own ledger outcomes",
                source="forgeos ledger",
                observations=1,
                confidence=0.6,
            )
            profile.traits.append(trait)
            return trait
        existing.evidence = TraitEvidence.MEASURED
        existing.observations += 1
        existing.confidence = min(1.0, existing.confidence + _CONFIRM_CONFIDENCE_STEP)
        existing.source = existing.source or "forgeos ledger"
        return existing

    if existing is None:
        return None

    existing.observations += 1
    existing.confidence -= _CONTRADICT_CONFIDENCE_STEP
    if existing.confidence <= 0.0:
        profile.traits = [t for t in profile.traits if t is not existing]
        return None

    rank = _EVIDENCE_RANK[existing.evidence]
    if rank > 0:
        existing.evidence = _EVIDENCE_BY_RANK[rank - 1]
    return existing


# Digit-heavy, benchmark-shaped text (a percentage, or a run of 3+ digits) has
# no business in a seed trait's note/source -- those fields are impressions,
# not marketing copy. Exposed so tests (and any future seed addition) can
# check new entries the same way.
BENCHMARK_SHAPED_PATTERN = re.compile(r"\d{3,}|\d+(\.\d+)?%")


__all__ = [
    "BENCHMARK_SHAPED_PATTERN",
    "SEED_MODEL_PROFILES",
    "ModelProfile",
    "PromptDiscipline",
    "Tendency",
    "Trait",
    "TraitEvidence",
    "all_profiles",
    "discipline_for",
    "discipline_instructions",
    "get_profile",
    "instructions_token_estimate",
    "record_outcome",
    "register_profile",
    "reset_registry",
]
