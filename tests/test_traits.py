"""Tests for the model-trait registry.

The premise this file exists to protect: a trait is a CLAIM, not a fact, and
the countermeasures derived from it must behave accordingly -- MEASURED
evidence always wins over an impression, an OBSERVED trait must stay visibly
unverified everywhere it surfaces, and the registry must be able to unlearn a
trait its own results contradict. That last property is the one that keeps a
wrong impression from permanently mis-prompting a model that turns out fine.
"""

from __future__ import annotations

import re

import pytest

from forgeos.core.router import DEFAULT_EFFORT, MAX_OUTPUT_BY_EFFORT, ReasoningEffort
from forgeos.models.traits import (
    SEED_MODEL_PROFILES,
    ModelProfile,
    Tendency,
    Trait,
    TraitEvidence,
    all_profiles,
    discipline_for,
    discipline_instructions,
    get_profile,
    instructions_token_estimate,
    record_outcome,
    register_profile,
    reset_registry,
)


@pytest.fixture(autouse=True)
def _isolated_registry():
    """Every test starts from the pristine seed state and leaves it that way.

    record_outcome and register_profile mutate module-level state; without
    this, one test's record_outcome calls would leak into the next test's
    assertions purely because of run order.
    """
    reset_registry()
    yield
    reset_registry()


# --------------------------------------------------------------------- seeds


def test_seed_profiles_are_all_observed():
    """Seed traits are impressions (including the operator's own about
    over-engineering), never anything stronger -- MEASURED/DOCUMENTED must be
    earned by this harness, not asserted at seed time.
    """
    assert SEED_MODEL_PROFILES, "expected at least one seed profile"
    for profile in SEED_MODEL_PROFILES:
        assert profile.traits, f"{profile.model_ref} has no traits to seed"
        for trait in profile.traits:
            assert trait.evidence is TraitEvidence.OBSERVED
            assert trait.unverified is True


def test_seed_data_claims_no_benchmark_figure():
    """Scan every seed trait's note/source for digit-heavy, benchmark-shaped
    text (a percentage, or 3+ consecutive digits). None should appear --
    seeds are impressions, not marketing copy, and inventing a number here
    would be indistinguishable from an invented vendor benchmark.
    """
    benchmark_shaped = re.compile(r"\d{3,}|\d+(\.\d+)?%")
    for profile in SEED_MODEL_PROFILES:
        for trait in profile.traits:
            assert not benchmark_shaped.search(trait.note), trait.note
            assert not benchmark_shaped.search(trait.source), trait.source


def test_reset_registry_restores_seed_state():
    seed_ref = SEED_MODEL_PROFILES[0].model_ref
    record_outcome(seed_ref, Tendency.OVER_ENGINEERS, confirmed=True)
    assert get_profile(seed_ref).dominant_trait(Tendency.OVER_ENGINEERS).evidence is (
        TraitEvidence.MEASURED
    )
    reset_registry()
    assert get_profile(seed_ref).dominant_trait(Tendency.OVER_ENGINEERS).evidence is (
        TraitEvidence.OBSERVED
    )


# ------------------------------------------------------------ evidence order


def test_measured_overrides_documented_for_same_tendency():
    profile = ModelProfile(
        model_ref="test/measured-vs-documented",
        traits=[
            Trait(
                tendency=Tendency.IGNORES_SCHEMA,
                evidence=TraitEvidence.DOCUMENTED,
                note="vendor changelog says so",
                source="vendor docs",
            ),
            Trait(
                tendency=Tendency.IGNORES_SCHEMA,
                evidence=TraitEvidence.MEASURED,
                note="this harness measured it directly",
                source="forgeos ledger",
            ),
        ],
    )
    dominant = profile.dominant_trait(Tendency.IGNORES_SCHEMA)
    assert dominant is not None
    assert dominant.evidence is TraitEvidence.MEASURED


def test_measured_overrides_observed_regardless_of_list_order():
    """Order in the list must not matter -- only evidence rank does."""
    profile = ModelProfile(
        model_ref="test/order-independent",
        traits=[
            Trait(
                tendency=Tendency.OVER_ENGINEERS,
                evidence=TraitEvidence.MEASURED,
                note="measured first in the list",
            ),
            Trait(
                tendency=Tendency.OVER_ENGINEERS,
                evidence=TraitEvidence.OBSERVED,
                note="an impression, listed second",
            ),
        ],
    )
    assert profile.dominant_trait(Tendency.OVER_ENGINEERS).evidence is TraitEvidence.MEASURED


# ---------------------------------------------------------- unverified marker


def test_observed_trait_is_flagged_unverified_on_the_trait_itself():
    trait = Trait(tendency=Tendency.OVER_ENGINEERS, evidence=TraitEvidence.OBSERVED, note="impression")
    assert trait.unverified is True
    assert "UNVERIFIED" in trait.label


def test_measured_trait_is_not_flagged_unverified():
    trait = Trait(tendency=Tendency.OVER_ENGINEERS, evidence=TraitEvidence.MEASURED, note="measured")
    assert trait.unverified is False
    assert "UNVERIFIED" not in trait.label


def test_observed_trait_stays_flagged_in_profile_describe():
    """The marker must survive into the human-readable summary too --
    "everywhere it surfaces" is not just the Trait object in isolation.
    """
    profile = SEED_MODEL_PROFILES[0]
    summary = profile.describe()
    assert "UNVERIFIED" in summary


# --------------------------------------------------------- discipline derive


def test_over_engineers_model_gets_hard_cap_and_speculation_ban():
    seed_ref = SEED_MODEL_PROFILES[0].model_ref  # seeded OVER_ENGINEERS, OBSERVED
    discipline = discipline_for(seed_ref, ReasoningEffort.MEDIUM)

    baseline = MAX_OUTPUT_BY_EFFORT[ReasoningEffort.MEDIUM]
    assert discipline.max_output_tokens is not None
    assert discipline.max_output_tokens < baseline
    assert discipline.forbid_speculative_work is True
    assert discipline.demand_smallest_change is True
    assert discipline.require_explicit_stop_condition is True


def test_needs_explicit_stop_sets_the_stop_condition_flag():
    register_profile(
        ModelProfile(
            model_ref="test/needs-stop",
            traits=[
                Trait(
                    tendency=Tendency.NEEDS_EXPLICIT_STOP,
                    evidence=TraitEvidence.DOCUMENTED,
                    note="published note that this model runs on without a stop condition",
                    source="vendor docs",
                )
            ],
        )
    )
    discipline = discipline_for("test/needs-stop")
    assert discipline.require_explicit_stop_condition is True


def test_ignores_schema_sets_the_schema_echo_flag():
    register_profile(
        ModelProfile(
            model_ref="test/ignores-schema",
            traits=[
                Trait(
                    tendency=Tendency.IGNORES_SCHEMA,
                    evidence=TraitEvidence.MEASURED,
                    note="measured dropping the required output schema",
                )
            ],
        )
    )
    discipline = discipline_for("test/ignores-schema")
    assert discipline.require_schema_echo is True


def test_unknown_model_gets_conservative_defaults_not_a_crash():
    discipline = discipline_for("totally/unknown-model-nobody-registered", ReasoningEffort.HIGH)
    assert discipline.max_output_tokens == MAX_OUTPUT_BY_EFFORT[ReasoningEffort.HIGH]
    assert discipline.require_explicit_stop_condition is True
    # No evidence of over-engineering -- must not invent a countermeasure for
    # a tendency nobody has observed.
    assert discipline.forbid_speculative_work is False
    assert discipline.demand_smallest_change is False


def test_unknown_model_default_effort_also_does_not_crash():
    discipline = discipline_for("another/unknown-model")
    assert discipline.max_output_tokens == MAX_OUTPUT_BY_EFFORT[DEFAULT_EFFORT]


# ------------------------------------------------------------------ brevity


def test_discipline_instructions_are_brief():
    """These lines ride on every call; assert the realistic worst case (an
    over-engineering model that also ignores schemas) stays well under the
    ~60 token budget a permanent per-call tax has to respect.
    """
    register_profile(
        ModelProfile(
            model_ref="test/over-engineer-and-schema-ignorer",
            traits=[
                Trait(tendency=Tendency.OVER_ENGINEERS, evidence=TraitEvidence.OBSERVED, note="x"),
                Trait(tendency=Tendency.IGNORES_SCHEMA, evidence=TraitEvidence.MEASURED, note="y"),
            ],
        )
    )
    discipline = discipline_for("test/over-engineer-and-schema-ignorer", ReasoningEffort.LOW)
    lines = discipline_instructions(discipline)
    assert lines  # something must actually be emitted
    assert instructions_token_estimate(lines) < 60


def test_discipline_instructions_empty_when_nothing_is_flagged():
    from forgeos.models.traits import PromptDiscipline

    bare = PromptDiscipline()
    assert discipline_instructions(bare) == []


# ------------------------------------------------------------------ unlearn


def test_record_outcome_confirmed_true_promotes_toward_measured():
    seed_ref = SEED_MODEL_PROFILES[0].model_ref
    before = get_profile(seed_ref).dominant_trait(Tendency.OVER_ENGINEERS)
    assert before.evidence is TraitEvidence.OBSERVED
    # Snapshot the float: `before` aliases the same object record_outcome
    # mutates in place, so the reference itself would reflect the
    # post-mutation value too and this assertion would be comparing a value
    # to itself.
    confidence_before = before.confidence

    updated = record_outcome(seed_ref, Tendency.OVER_ENGINEERS, confirmed=True)
    assert updated.evidence is TraitEvidence.MEASURED
    assert updated.confidence > confidence_before


def test_record_outcome_confirmed_true_creates_a_trait_if_none_existed():
    trait = record_outcome("test/brand-new-model", Tendency.WEAK_TOOL_USE, confirmed=True)
    assert trait is not None
    assert trait.evidence is TraitEvidence.MEASURED
    assert get_profile("test/brand-new-model").has(Tendency.WEAK_TOOL_USE)


def test_record_outcome_confirmed_false_eventually_removes_the_trait():
    """Repeated contradiction must be able to unlearn a trait entirely --
    a registry that only ever accumulates traits is worse than none, because
    a wrong one permanently mis-prompts a model that may be perfectly fine.
    """
    seed_ref = SEED_MODEL_PROFILES[0].model_ref
    assert get_profile(seed_ref).has(Tendency.OVER_ENGINEERS)

    for _ in range(10):
        record_outcome(seed_ref, Tendency.OVER_ENGINEERS, confirmed=False)
        if not get_profile(seed_ref).has(Tendency.OVER_ENGINEERS):
            break

    assert not get_profile(seed_ref).has(Tendency.OVER_ENGINEERS)


def test_record_outcome_confirmed_false_downgrades_before_it_removes():
    """A single contradiction against a MEASURED trait should not vanish it
    outright -- it should be demoted a step, giving the registry a chance to
    recover from one noisy result rather than discarding history instantly.
    """
    register_profile(
        ModelProfile(
            model_ref="test/measured-then-contradicted",
            traits=[
                Trait(
                    tendency=Tendency.DRIFTS_OFF_TASK,
                    evidence=TraitEvidence.MEASURED,
                    note="previously measured",
                    confidence=1.0,
                )
            ],
        )
    )
    updated = record_outcome("test/measured-then-contradicted", Tendency.DRIFTS_OFF_TASK, confirmed=False)
    assert updated is not None
    assert updated.evidence is TraitEvidence.DOCUMENTED
    assert updated.confidence < 1.0


def test_record_outcome_confirmed_false_on_untracked_tendency_is_a_noop():
    result = record_outcome("test/no-such-trait-model", Tendency.STOPS_EARLY, confirmed=False)
    assert result is None


# ------------------------------------------------------------------- registry


def test_register_and_get_profile_roundtrip():
    profile = ModelProfile(model_ref="test/roundtrip")
    register_profile(profile)
    assert get_profile("test/roundtrip") is profile


def test_all_profiles_includes_seed_and_newly_registered():
    register_profile(ModelProfile(model_ref="test/newcomer"))
    refs = {p.model_ref for p in all_profiles()}
    assert "test/newcomer" in refs
    assert SEED_MODEL_PROFILES[0].model_ref in refs
