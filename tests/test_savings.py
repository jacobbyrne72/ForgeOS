"""SavingsProof tests: provenance honesty is the whole point of this module.

No LLM calls, no network — everything here is either pure computation or
reads/writes against the real SQLite-backed stores (`Ledger`, `AvoidanceLog`,
`SpanStore`, `EventLog`) opened at ``:memory:``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from forgeos.core.timing import Phase, SpanStore
from forgeos.economy.avoidance import AvoidanceLog, AvoidanceMethod
from forgeos.economy.savings import (
    Baseline,
    CounterfactualPlan,
    Figure,
    Outcome,
    Provenance,
    SavingsProof,
    build_proof,
    render_receipt,
    savings_pct,
    verify_proof,
)
from forgeos.events import EventLog, EventType
from forgeos.ledger import Ledger

# --------------------------------------------------------------- fixtures


@pytest.fixture
def ledger() -> Ledger:
    lg = Ledger(":memory:")
    yield lg
    lg.close()


@pytest.fixture
def avoidance_log() -> AvoidanceLog:
    log = AvoidanceLog(":memory:")
    yield log
    log.close()


@pytest.fixture
def span_store() -> SpanStore:
    store = SpanStore(":memory:")
    yield store
    store.close()


@pytest.fixture
def event_log() -> EventLog:
    log = EventLog(":memory:")
    yield log
    log.close()


def _measured(value: float, unit: str = "usd", note: str = "src") -> Figure:
    return Figure(value=value, unit=unit, provenance=Provenance.MEASURED, note=note)


def _modelled(value: float, unit: str = "usd", confidence: float | None = 0.7) -> Figure:
    return Figure(value=value, unit=unit, provenance=Provenance.MODELLED, confidence=confidence)


def _replayed(value: float, unit: str = "usd", note: str = "replay run") -> Figure:
    return Figure(value=value, unit=unit, provenance=Provenance.REPLAYED, note=note)


def _full_hashes() -> dict:
    return {
        "contract_hash": "c" * 8,
        "actual_trace_hash": "a" * 8,
        "baseline_trace_hash": "b" * 8,
        "test_artifact_hash": "t" * 8,
    }


# ------------------------------------------------------------- Provenance


def test_is_fact_true_for_measured_and_replayed():
    assert _measured(1.0).is_fact is True
    assert _replayed(1.0).is_fact is True


def test_is_fact_false_for_modelled_and_unknown():
    assert _modelled(1.0).is_fact is False
    assert Figure(value=1.0, unit="usd", provenance=Provenance.UNKNOWN).is_fact is False


# ------------------------------------------------------------------ Figure


def test_render_measured_matches_spec_example():
    fig = Figure(value=0.38, unit="eur", provenance=Provenance.MEASURED)
    assert fig.render() == "€0.38 measured"


def test_render_modelled_with_error_bound_matches_spec_example():
    fig = Figure(value=6.2, unit="pct", provenance=Provenance.MODELLED, error_bound=0.8)
    assert fig.render() == "6.2% estimated ±0.8%"


def test_render_always_ends_with_provenance_word():
    for prov, word in (
        (Provenance.MEASURED, "measured"),
        (Provenance.REPLAYED, "replayed"),
        (Provenance.MODELLED, "estimated"),
        (Provenance.UNKNOWN, "unknown"),
    ):
        fig = Figure(value=5, unit="tokens", provenance=prov)
        assert fig.render().endswith(word)


def test_confidence_out_of_range_rejected():
    with pytest.raises(ValidationError):
        Figure(value=1.0, unit="usd", provenance=Provenance.MODELLED, confidence=1.5)


# -------------------------------------------------------------- savings_pct


def test_measured_actual_modelled_baseline_yields_modelled_saving():
    """THE CORE RULE: a real actual against a guessed baseline is a guess, not a fact."""
    actual = _measured(8.0)
    baseline = _modelled(10.0)
    saved = savings_pct(actual, baseline)
    assert saved.provenance is Provenance.MODELLED
    assert saved.is_fact is False
    assert saved.value == pytest.approx(20.0)  # (10-8)/10 * 100


def test_measured_actual_replayed_baseline_yields_replayed_saving():
    """Both inputs are facts, but REPLAYED is the weaker fact of the two."""
    actual = _measured(8.0)
    baseline = _replayed(10.0)
    saved = savings_pct(actual, baseline)
    assert saved.provenance is Provenance.REPLAYED
    assert saved.is_fact is True


def test_measured_actual_measured_baseline_yields_measured_saving():
    actual = _measured(8.0)
    baseline = _measured(10.0)
    saved = savings_pct(actual, baseline)
    assert saved.provenance is Provenance.MEASURED


def test_unknown_baseline_drags_saving_to_unknown():
    actual = _measured(8.0)
    baseline = Figure(value=10.0, unit="usd", provenance=Provenance.UNKNOWN)
    saved = savings_pct(actual, baseline)
    assert saved.provenance is Provenance.UNKNOWN


def test_zero_baseline_does_not_divide_by_zero():
    actual = _measured(5.0)
    baseline = _measured(0.0)
    saved = savings_pct(actual, baseline)  # must not raise ZeroDivisionError
    assert saved.value == 0.0
    assert "zero" in saved.note.lower()


def test_negative_savings_reported_honestly_not_clamped_or_hidden():
    """Spending MORE than baseline is a regression. It must show up negative."""
    actual = _measured(15.0)
    baseline = _measured(10.0)
    saved = savings_pct(actual, baseline)
    assert saved.value < 0
    assert saved.value == pytest.approx(-50.0)  # (10-15)/10 * 100
    assert "-" in saved.render()


def test_savings_pct_confidence_is_weakest_of_inputs():
    actual = _modelled(8.0, confidence=0.9)
    baseline = _modelled(10.0, confidence=0.4)
    saved = savings_pct(actual, baseline)
    assert saved.confidence == pytest.approx(0.4)


def test_savings_pct_error_bound_is_converted_to_pct_not_copied_raw():
    """A dollar/token error bound must not be silently relabelled as a percent.

    Regression test: `savings_pct` used to copy `baseline.error_bound`
    straight onto the result, whose unit is "pct" — so a baseline of
    "210,000 tokens ±15,000" produced a savings figure that rendered as
    "±15000.0%", a meaningless, unit-mismatched number.
    """
    actual = Figure(value=1.5, unit="usd", provenance=Provenance.MEASURED, note="src")
    baseline = Figure(
        value=1.85, unit="usd", provenance=Provenance.MODELLED, error_bound=0.30,
    )
    saved = savings_pct(actual, baseline)
    assert saved.error_bound == pytest.approx(100.0 * 0.30 / 1.85)
    assert saved.error_bound != 0.30  # must be converted, not copied raw
    rendered = saved.render()
    assert "16." in rendered  # ~16.2%, not "±0.3%" and not "±15000.0%"


def test_savings_pct_zero_baseline_has_no_error_bound():
    actual = _measured(5.0, note="src")
    baseline = Figure(
        value=0.0, unit="usd", provenance=Provenance.MEASURED, error_bound=1.0, note="src",
    )
    saved = savings_pct(actual, baseline)
    assert saved.error_bound is None


# --------------------------------------------------------- model validation


def test_baseline_policy_must_not_be_blank():
    with pytest.raises(ValidationError):
        Baseline(baseline_policy="   ", baseline_provenance=Provenance.MODELLED)


def test_acceptance_passed_cannot_exceed_total():
    with pytest.raises(ValidationError):
        SavingsProof(
            mission_id="m1",
            repo_revision="deadbeef",
            outcome=Outcome.ACCEPTED,
            acceptance_passed=6,
            acceptance_total=5,
            baseline=Baseline(baseline_policy="naive", baseline_provenance=Provenance.MODELLED),
        )


# ------------------------------------------------------------- build_proof


def test_build_proof_assembles_measured_actuals_from_real_stores(
    ledger, avoidance_log, span_store, event_log
):
    job_id = "job_abc"

    ledger.record_spend(
        job_id, "worker_1", "gpt-x", 380_000,  # $0.38
        tokens_in=900, tokens_out=300, tokens_cached_in=100,
    )
    avoidance_log.record(
        job_id, AvoidanceMethod.CACHE_HIT,
        baseline_tokens=500, actual_tokens=100, baseline_source="tiktoken count of full file",
    )
    s1 = span_store.start(job_id, Phase.MODEL_REASONING, started_at=1000.0)
    span_store.end(s1.id, at=1030.0)
    event_log.append(job_id, EventType.GOVERNOR_TRIPPED, reason="budget")
    event_log.append(job_id, EventType.TASK_ACCEPTED)  # not a trip, must not be counted

    baseline = Baseline(
        baseline_policy="always-claude-opus",
        baseline_provenance=Provenance.MODELLED,
        figures={
            "cash_cost": _modelled(1.20),
            "input_tokens": _modelled(2000, unit="tokens"),
        },
    )

    proof = build_proof(
        ledger, avoidance_log, span_store, event_log, job_id,
        mission_id="mission_1",
        repo_revision="deadbeef",
        outcome=Outcome.ACCEPTED,
        acceptance_passed=3,
        acceptance_total=3,
        baseline=baseline,
        **_full_hashes(),
    )

    assert proof.actual["cash_cost"].value == pytest.approx(0.38)
    assert proof.actual["cash_cost"].provenance is Provenance.MEASURED
    assert proof.actual["cash_cost"].note  # must cite its source

    assert proof.actual["input_tokens"].value == pytest.approx(1000)  # 900 fresh + 100 cached
    assert proof.actual["output_tokens"].value == pytest.approx(300)
    assert proof.actual["wall_time"].value == pytest.approx(30.0)
    assert proof.actual["human_interventions"].value == pytest.approx(1.0)  # one trip, not two events

    # savings computed only for keys present in both actual and baseline
    assert set(proof.savings) == {"cash_cost", "input_tokens"}
    assert proof.savings["cash_cost"].provenance is Provenance.MODELLED  # weakest-of-two rule
    assert "avoidance_log" in proof.actual["input_tokens"].note


def test_build_proof_with_no_governor_trips_reports_zero_interventions(
    ledger, avoidance_log, span_store, event_log
):
    job_id = "job_clean"
    ledger.record_spend(job_id, "w", "m", 10_000)
    baseline = Baseline(baseline_policy="naive", baseline_provenance=Provenance.MODELLED)
    proof = build_proof(
        ledger, avoidance_log, span_store, event_log, job_id,
        mission_id="m2", repo_revision="rev", outcome=Outcome.ACCEPTED,
        acceptance_passed=1, acceptance_total=1, baseline=baseline,
    )
    assert proof.actual["human_interventions"].value == 0.0
    assert proof.actual["human_interventions"].provenance is Provenance.MEASURED


# ----------------------------------------------------------- render_receipt


def _sample_proof(baseline_provenance: Provenance = Provenance.MODELLED) -> SavingsProof:
    actual = {
        "cash_cost": _measured(0.38, note="ledger.job_spend_micros"),
        "input_tokens": _measured(1000, unit="tokens", note="ledger.cache_stats"),
    }
    baseline_figures = {
        "cash_cost": Figure(
            value=1.20, unit="usd", provenance=baseline_provenance, confidence=0.6,
        ),
        "input_tokens": Figure(
            value=2000, unit="tokens", provenance=baseline_provenance, confidence=0.8,
        ),
    }
    baseline = Baseline(
        baseline_policy="always-claude-opus",
        baseline_provenance=baseline_provenance,
        figures=baseline_figures,
    )
    savings = {k: savings_pct(actual[k], baseline_figures[k]) for k in actual}
    return SavingsProof(
        mission_id="mission_receipt",
        repo_revision="deadbeef",
        outcome=Outcome.ACCEPTED,
        acceptance_passed=4,
        acceptance_total=4,
        actual=actual,
        baseline=baseline,
        savings=savings,
        **_full_hashes(),
    )


def test_receipt_visibly_marks_every_estimate():
    proof = _sample_proof(Provenance.MODELLED)
    receipt = render_receipt(proof)

    assert "measured" in receipt          # the actual figures
    assert "estimated" in receipt         # the modelled baseline + derived savings
    assert "BASELINE WAS NOT ACTUALLY RUN" in receipt
    assert "confidence=0.60" in receipt   # min(0.6, 0.8)


def test_receipt_no_warning_when_baseline_is_replayed():
    proof = _sample_proof(Provenance.REPLAYED)
    receipt = render_receipt(proof)
    assert "BASELINE WAS NOT ACTUALLY RUN" not in receipt
    assert "replayed" in receipt


def test_receipt_shows_missing_hashes_explicitly():
    proof = _sample_proof(Provenance.MODELLED)
    proof.contract_hash = ""
    receipt = render_receipt(proof)
    assert "MISSING" in receipt


# -------------------------------------------------------------- verify_proof


def test_verify_proof_empty_for_consistent_receipt():
    proof = _sample_proof(Provenance.MODELLED)
    assert verify_proof(proof) == []


def test_verify_proof_catches_saving_claimed_stronger_than_its_inputs():
    """A MEASURED actual + MODELLED baseline must not be reported as a MEASURED saving."""
    proof = _sample_proof(Provenance.MODELLED)
    # Tamper: claim the cash_cost saving is MEASURED even though the baseline is MODELLED.
    proof.savings["cash_cost"] = Figure(
        value=proof.savings["cash_cost"].value,
        unit="pct",
        provenance=Provenance.MEASURED,
        note="tampered",
    )
    complaints = verify_proof(proof)
    assert any("claimed as measured" in c for c in complaints)


def test_verify_proof_catches_accepted_with_failing_acceptance():
    proof = _sample_proof(Provenance.MODELLED)
    proof.acceptance_passed = 2
    proof.acceptance_total = 4
    complaints = verify_proof(proof)
    assert any("ACCEPTED" in c and "2/4" in c for c in complaints)


def test_verify_proof_catches_missing_evidence_hashes():
    proof = _sample_proof(Provenance.MODELLED)
    proof.contract_hash = ""
    proof.test_artifact_hash = "  "
    complaints = verify_proof(proof)
    assert any("contract_hash" in c for c in complaints)
    assert any("test_artifact_hash" in c for c in complaints)
    # the other two are still present, so no complaint about them
    assert not any("actual_trace_hash" in c for c in complaints)
    assert not any("baseline_trace_hash" in c for c in complaints)


def test_verify_proof_catches_measured_figure_with_no_source():
    proof = _sample_proof(Provenance.MODELLED)
    proof.actual["cash_cost"] = Figure(value=0.38, unit="usd", provenance=Provenance.MEASURED, note="")
    complaints = verify_proof(proof)
    assert any("MEASURED but has no source note" in c for c in complaints)


def test_verify_proof_catches_hidden_regression_shown_as_positive():
    """We spent MORE than baseline, but the stored figure lies and shows a gain."""
    proof = _sample_proof(Provenance.MODELLED)
    actual_cost = proof.actual["cash_cost"].value       # 0.38
    baseline_cost = proof.baseline.figures["cash_cost"].value  # 1.20
    # Force a regression: actual now costs more than baseline.
    proof.actual["cash_cost"] = Figure(
        value=baseline_cost + 5.0, unit="usd", provenance=Provenance.MEASURED, note="src"
    )
    # But the stored saving is falsely reported as a positive saving.
    proof.savings["cash_cost"] = Figure(
        value=10.0, unit="pct", provenance=Provenance.MODELLED, note="tampered positive"
    )
    complaints = verify_proof(proof)
    assert any("regression" in c and "cash_cost" in c for c in complaints)
    assert actual_cost != baseline_cost  # sanity: fixture actually differs


def test_verify_proof_does_not_flag_an_honest_negative_saving():
    """A correctly-computed regression must NOT itself be treated as an error."""
    actual = _measured(15.0, note="src")
    baseline = _measured(10.0, note="src")
    saved = savings_pct(actual, baseline)
    assert saved.value < 0

    proof = SavingsProof(
        mission_id="m3",
        repo_revision="rev",
        outcome=Outcome.REJECTED,
        acceptance_passed=1,
        acceptance_total=2,
        actual={"cash_cost": actual},
        baseline=Baseline(
            baseline_policy="naive",
            baseline_provenance=Provenance.MEASURED,
            figures={"cash_cost": baseline},
        ),
        savings={"cash_cost": saved},
        **_full_hashes(),
    )
    complaints = verify_proof(proof)
    assert not any("regression" in c for c in complaints)
    assert not any("claimed as" in c for c in complaints)


# ----------------------------------------------------------- CounterfactualPlan


def test_counterfactual_plan_ready_when_fully_specified():
    plan = CounterfactualPlan(
        policy_name="cheap-tier-router",
        pinned_revision="deadbeef",
        task_ids=["task_1", "task_2"],
    )
    assert plan.ready() is True


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(policy_name="", pinned_revision="deadbeef", task_ids=["t1"]),
        dict(policy_name="p", pinned_revision="", task_ids=["t1"]),
        dict(policy_name="p", pinned_revision="deadbeef", task_ids=[]),
    ],
)
def test_counterfactual_plan_not_ready_when_incomplete(kwargs):
    plan = CounterfactualPlan(**kwargs)
    assert plan.ready() is False


# ------------------------------------------------- small amounts stay visible


def test_a_fraction_of_a_cent_is_never_rendered_as_zero():
    """The bug this pins: at two decimal places every sub-cent figure rendered
    "$0.00", so a receipt reported real spend as free. In this system a single
    model call routinely costs less than a cent, so that was not an edge case,
    it was the common case -- and it always erred toward looking cheaper, the
    direction nobody double-checks.
    """
    f = Figure(value=0.004, unit="usd", provenance=Provenance.MEASURED)
    assert "$0.00 " not in f.render()
    assert "0.004" in f.render()


def test_exact_zero_still_renders_as_zero():
    """Zero really is zero; widening precision there would only add noise."""
    f = Figure(value=0.0, unit="usd", provenance=Provenance.MEASURED)
    assert f.render().startswith("$0.00")


def test_ordinary_amounts_keep_two_decimals():
    """No regression in the normal case -- dollars still read like dollars."""
    assert Figure(value=12.5, unit="usd",
                  provenance=Provenance.MEASURED).render().startswith("$12.50")
    assert Figure(value=1234.5, unit="usd",
                  provenance=Provenance.MEASURED).render().startswith("$1,234.50")


def test_a_sub_cent_regression_is_still_visible_as_a_regression():
    """Costing MORE is the failure this module exists to surface. Rounding a
    small overspend to "$0.00" would hide exactly that."""
    f = Figure(value=-0.003, unit="usd", provenance=Provenance.MEASURED)
    rendered = f.render()
    assert "0.003" in rendered and "-" in rendered


def test_precision_never_runs_away():
    """A denormal-ish value must not produce a fifty-character number."""
    f = Figure(value=1e-12, unit="usd", provenance=Provenance.MEASURED)
    assert len(f.render()) < 30


def test_a_sub_cent_error_bound_is_also_visible():
    """The bound shares the formatter, so it shared the bug: '$0.0042 measured
    ±$0.00' states a precision the number does not have."""
    f = Figure(value=0.0042, unit="usd", provenance=Provenance.MEASURED,
               error_bound=0.0008)
    assert "±$0.00 " not in f.render() and "0.0008" in f.render()
