"""Billed-cost breakdown tests.

arXiv:2607.12161v2 measured 2,908 real provider-billed Claude Code runs and
found prompt-cache traffic is ~87% of cost composition, and that cutting raw
tokens can silently raise billed cost while lowering task success. Two
consequences enforced here:

1. A single `usd_micros` total hides the thing that matters — it must
   decompose into cache-write / cache-read / fresh-input / output
   (`billed_cost_breakdown`), and "we have no data for this" must never
   render as "$0.00" (`Provenance.UNKNOWN`, not a measured/modelled zero).
2. Savings must be expressed as billed cost per accepted unit of work
   (`cost_per_accepted_task`), never as a raw token count.

No LLM calls, no network — everything here is either pure computation or
reads against a real SQLite-backed `Ledger` opened at ``:memory:``.
"""

from __future__ import annotations

import pytest

from forgeos.catalog import Catalog, ModelCard
from forgeos.contracts import to_micros
from forgeos.core.timing import SpanStore
from forgeos.economy.avoidance import AvoidanceLog
from forgeos.economy.savings import (
    Baseline,
    Figure,
    Outcome,
    Provenance,
    SavingsProof,
    billed_cost_breakdown,
    build_proof,
    cost_per_accepted_task,
    render_receipt,
    verify_proof,
)
from forgeos.events import EventLog
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


PRICED = Catalog(
    [
        ModelCard(
            model_id="m",
            provider="p",
            input_cost_per_1m=3.0,
            output_cost_per_1m=15.0,
            cache_read_cost_per_1m=0.3,
        ),
    ]
)


def _sum_parts_micros(b) -> int:
    return (
        to_micros(b.fresh_input.value)
        + to_micros(b.cache_read.value)
        + to_micros(b.cache_write.value)
        + to_micros(b.output.value)
        + to_micros(b.unpriced.value)
    )


# --------------------------------------------------------- billed_cost_breakdown


def test_four_components_sum_to_recorded_total(ledger):
    """The whole point: cache-write + cache-read + fresh-input + output (plus
    the unpriced honesty valve, zero here) must reconcile to the ledger's own
    measured total, exactly, in integer micros — not approximately."""
    job = "job1"
    card = PRICED.get("p/m")
    micros = card.cost_micros(tokens_in=1000, tokens_out=200, tokens_cached_in=500)
    ledger.record_spend(
        job, "w", "p/m", micros, tokens_in=1000, tokens_out=200, tokens_cached_in=500
    )

    b = billed_cost_breakdown(ledger, job, catalog=PRICED)

    assert b.unpriced.value == 0.0  # every row priced, nothing left unattributed
    assert _sum_parts_micros(b) == to_micros(b.total.value)
    assert _sum_parts_micros(b) == micros
    assert b.sums_to_total() is True


def test_sum_holds_exactly_across_many_odd_shaped_rows(ledger):
    """Regression guard for the largest-remainder allocator: independent
    per-share rounding could drift the sum by a microdollar or two across many
    rows with non-round token counts. It must not — a budget ledger has no
    tolerance for "off by a microdollar"."""
    job = "job_many"
    card = PRICED.get("p/m")
    total_expected = 0
    for i in range(37):  # an odd count of odd-shaped calls, on purpose
        tin, tout, tcached = 137 + i, 41 + i, 19 + i
        micros = card.cost_micros(tokens_in=tin, tokens_out=tout, tokens_cached_in=tcached)
        ledger.record_spend(
            job, "w", "p/m", micros, tokens_in=tin, tokens_out=tout, tokens_cached_in=tcached
        )
        total_expected += micros

    b = billed_cost_breakdown(ledger, job, catalog=PRICED)
    assert to_micros(b.total.value) == total_expected
    assert _sum_parts_micros(b) == total_expected
    assert b.sums_to_total() is True


def test_cache_write_is_unknown_not_zero(ledger):
    """The spend schema has never recorded a cache-write token count for any
    model — cache_write must read as UNKNOWN, never a measured/modelled zero,
    even for a model whose card carries a cache_write_cost_per_1m price."""
    job = "job2"
    priced_with_write = Catalog(
        [
            ModelCard(
                model_id="m",
                provider="p",
                input_cost_per_1m=3.0,
                output_cost_per_1m=15.0,
                cache_read_cost_per_1m=0.3,
                cache_write_cost_per_1m=3.75,
            ),
        ]
    )
    card = priced_with_write.get("p/m")
    micros = card.cost_micros(tokens_in=500, tokens_out=100, tokens_cached_in=200)
    ledger.record_spend(
        job, "w", "p/m", micros, tokens_in=500, tokens_out=100, tokens_cached_in=200
    )

    b = billed_cost_breakdown(ledger, job, catalog=priced_with_write)

    assert b.cache_write.provenance is Provenance.UNKNOWN
    assert b.cache_write.value == 0.0
    assert b.cache_write.is_fact is False


def test_model_with_no_catalog_pricing_reports_unpriced_not_zero(ledger):
    """A spend row whose 'model' has no catalog entry at all (e.g. a flat-rate
    tier-prior estimate row keyed by worker id — see forge.py's unmetered-
    worker fallback) cannot be split into fresh/cache-read/output. It must
    land in `unpriced`, not silently vanish or get guessed into one of the
    four named components."""
    job = "job3"
    empty_catalog = Catalog([])
    ledger.record_spend(
        job, "w", "omc.executor", 42_000, tokens_in=10, tokens_out=5, tokens_cached_in=0
    )

    b = billed_cost_breakdown(ledger, job, catalog=empty_catalog)

    assert b.unpriced.value == pytest.approx(0.042)
    assert b.unpriced.provenance is Provenance.MEASURED
    assert b.fresh_input.value == 0.0
    assert b.cache_read.value == 0.0
    assert b.output.value == 0.0
    assert _sum_parts_micros(b) == to_micros(b.total.value)


def test_mixed_priced_and_unpriced_rows_still_reconcile(ledger):
    job = "job4"
    card = PRICED.get("p/m")
    priced_micros = card.cost_micros(tokens_in=1000, tokens_out=200, tokens_cached_in=500)
    ledger.record_spend(
        job, "w1", "p/m", priced_micros, tokens_in=1000, tokens_out=200, tokens_cached_in=500
    )
    ledger.record_spend(
        job, "w2", "omc.executor", 50_000, tokens_in=10, tokens_out=5, tokens_cached_in=0
    )

    b = billed_cost_breakdown(ledger, job, catalog=PRICED)

    assert to_micros(b.unpriced.value) == 50_000
    assert to_micros(b.total.value) == priced_micros + 50_000
    assert b.sums_to_total() is True


def test_fresh_cache_read_output_are_labelled_modelled_not_measured(ledger):
    """These three are derived from token counts and catalog price WEIGHTS —
    not read verbatim off a ledger column — so per the provenance-honesty
    rule they must claim MODELLED, never MEASURED. Only `total` (a literal
    `job_spend_micros` read) and `unpriced` (a literal sum, no allocation
    guess involved) earn MEASURED."""
    job = "job5"
    card = PRICED.get("p/m")
    micros = card.cost_micros(tokens_in=1000, tokens_out=200, tokens_cached_in=500)
    ledger.record_spend(
        job, "w", "p/m", micros, tokens_in=1000, tokens_out=200, tokens_cached_in=500
    )

    b = billed_cost_breakdown(ledger, job, catalog=PRICED)

    assert b.fresh_input.provenance is Provenance.MODELLED
    assert b.cache_read.provenance is Provenance.MODELLED
    assert b.output.provenance is Provenance.MODELLED
    assert b.total.provenance is Provenance.MEASURED
    assert b.unpriced.provenance is Provenance.MEASURED


def test_no_spend_returns_all_zero_and_still_reconciles(ledger):
    b = billed_cost_breakdown(ledger, "job_empty", catalog=PRICED)
    assert b.total.value == 0.0
    assert b.sums_to_total() is True


def test_default_catalog_used_when_none_passed(ledger):
    """Must not crash when the caller does not supply a catalog — falls back
    to `default_catalog()`, which is safe (possibly empty) on a machine with
    no local models.dev cache."""
    job = "job_default_cat"
    ledger.record_spend(job, "w", "omc.executor", 1_000, tokens_in=1, tokens_out=1)
    b = billed_cost_breakdown(ledger, job)  # no catalog kwarg
    assert b.sums_to_total() is True


# --------------------------------------------------------- cost_per_accepted_task


def test_cost_per_accepted_task_is_none_when_nothing_accepted():
    cash_cost = Figure(value=12.0, unit="usd", provenance=Provenance.MEASURED, note="src")
    result = cost_per_accepted_task(cash_cost, 0)
    assert result is None  # not 0.0, not inf


def test_cost_per_accepted_task_rejects_negative_acceptance_too():
    cash_cost = Figure(value=5.0, unit="usd", provenance=Provenance.MEASURED, note="src")
    assert cost_per_accepted_task(cash_cost, -1) is None


def test_cost_per_accepted_task_divides_measured_cash_cost():
    cash_cost = Figure(
        value=12.0, unit="usd", provenance=Provenance.MEASURED, note="ledger.job_spend_micros"
    )
    result = cost_per_accepted_task(cash_cost, 4)
    assert result is not None
    assert result.value == pytest.approx(3.0)
    assert result.provenance is Provenance.MEASURED
    assert "4 accepted" in result.note


def test_cost_per_accepted_task_inherits_modelled_provenance():
    """A projected cash_cost (MODELLED) must not launder into a MEASURED
    per-task figure just because the division itself is exact arithmetic."""
    cash_cost = Figure(value=10.0, unit="usd", provenance=Provenance.MODELLED, confidence=0.5)
    result = cost_per_accepted_task(cash_cost, 2)
    assert result is not None
    assert result.provenance is Provenance.MODELLED
    assert result.confidence == pytest.approx(0.5)


# --------------------------------------------------- SavingsProof integration


def test_build_proof_leads_receipt_with_cost_per_accepted_task(
    ledger, avoidance_log, span_store, event_log
):
    job = "job_receipt"
    ledger.record_spend(
        job, "w", "gpt-x", 400_000, tokens_in=900, tokens_out=300, tokens_cached_in=100
    )
    baseline = Baseline(baseline_policy="naive", baseline_provenance=Provenance.MODELLED)

    proof = build_proof(
        ledger,
        avoidance_log,
        span_store,
        event_log,
        job,
        mission_id="m",
        repo_revision="rev",
        outcome=Outcome.ACCEPTED,
        acceptance_passed=4,
        acceptance_total=4,
        baseline=baseline,
        contract_hash="c" * 8,
        actual_trace_hash="a" * 8,
        baseline_trace_hash="b" * 8,
        test_artifact_hash="t" * 8,
    )

    assert proof.cost_per_accepted_task is not None
    assert proof.cost_per_accepted_task.value == pytest.approx(0.1)  # $0.40 / 4
    assert proof.cost_per_accepted_task.provenance is Provenance.MEASURED

    receipt = render_receipt(proof)
    lines = receipt.splitlines()
    cost_line_idx = next(i for i, ln in enumerate(lines) if "cost / accepted task" in ln)
    actual_idx = next(i for i, ln in enumerate(lines) if ln.strip() == "ACTUAL")
    assert cost_line_idx < actual_idx  # it leads: appears before the ACTUAL section

    assert verify_proof(proof) == []


def test_build_proof_cost_per_accepted_task_none_when_nothing_accepted(
    ledger, avoidance_log, span_store, event_log
):
    job = "job_rejected"
    ledger.record_spend(job, "w", "gpt-x", 100_000, tokens_in=100, tokens_out=50)
    baseline = Baseline(baseline_policy="naive", baseline_provenance=Provenance.MODELLED)

    proof = build_proof(
        ledger,
        avoidance_log,
        span_store,
        event_log,
        job,
        mission_id="m2",
        repo_revision="rev",
        outcome=Outcome.REJECTED,
        acceptance_passed=0,
        acceptance_total=3,
        baseline=baseline,
    )

    assert proof.cost_per_accepted_task is None
    receipt = render_receipt(proof)
    assert "n/a (0 accepted)" in receipt
    assert verify_proof(proof) == [
        "missing evidence hash: contract_hash",
        "missing evidence hash: actual_trace_hash",
        "missing evidence hash: baseline_trace_hash",
        "missing evidence hash: test_artifact_hash",
    ]


def test_verify_proof_catches_cost_per_accepted_task_set_with_zero_accepted():
    proof = SavingsProof(
        mission_id="m",
        repo_revision="rev",
        outcome=Outcome.REJECTED,
        acceptance_passed=0,
        acceptance_total=3,
        actual={"cash_cost": Figure(value=1.0, unit="usd", provenance=Provenance.MEASURED, note="src")},
        cost_per_accepted_task=Figure(
            value=1.0, unit="usd", provenance=Provenance.MEASURED, note="bad"
        ),
        baseline=Baseline(baseline_policy="naive", baseline_provenance=Provenance.MODELLED),
    )
    complaints = verify_proof(proof)
    assert any("acceptance_passed is 0" in c for c in complaints)


def test_verify_proof_catches_cost_per_accepted_task_inconsistent_with_cash_cost():
    proof = SavingsProof(
        mission_id="m",
        repo_revision="rev",
        outcome=Outcome.ACCEPTED,
        acceptance_passed=2,
        acceptance_total=2,
        actual={"cash_cost": Figure(value=10.0, unit="usd", provenance=Provenance.MEASURED, note="src")},
        cost_per_accepted_task=Figure(
            value=999.0, unit="usd", provenance=Provenance.MEASURED, note="tampered"
        ),
        baseline=Baseline(baseline_policy="naive", baseline_provenance=Provenance.MODELLED),
    )
    complaints = verify_proof(proof)
    assert any("cash_cost/acceptance_passed implies" in c for c in complaints)


def test_verify_proof_catches_cost_per_accepted_task_overclaiming_provenance():
    proof = SavingsProof(
        mission_id="m",
        repo_revision="rev",
        outcome=Outcome.ACCEPTED,
        acceptance_passed=2,
        acceptance_total=2,
        actual={
            "cash_cost": Figure(value=10.0, unit="usd", provenance=Provenance.MODELLED, confidence=0.5)
        },
        cost_per_accepted_task=Figure(
            value=5.0, unit="usd", provenance=Provenance.MEASURED, note="tampered"
        ),
        baseline=Baseline(baseline_policy="naive", baseline_provenance=Provenance.MODELLED),
    )
    complaints = verify_proof(proof)
    assert any("claims stronger provenance" in c for c in complaints)


def test_verify_proof_silent_when_cost_per_accepted_task_absent():
    """Existing SavingsProof construction without the new field must stay clean —
    the field is optional and defaults to None."""
    proof = SavingsProof(
        mission_id="m",
        repo_revision="rev",
        outcome=Outcome.ACCEPTED,
        acceptance_passed=1,
        acceptance_total=1,
        actual={"cash_cost": Figure(value=1.0, unit="usd", provenance=Provenance.MEASURED, note="src")},
        baseline=Baseline(baseline_policy="naive", baseline_provenance=Provenance.MODELLED),
    )
    assert proof.cost_per_accepted_task is None
    complaints = verify_proof(proof)
    assert not any("cost_per_accepted_task" in c for c in complaints)
