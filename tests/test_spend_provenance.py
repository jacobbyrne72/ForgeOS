"""Measured spend and modelled tier priors are different currencies.

`kind='call'` is a provider-billed call with real usage. `kind='estimate'` is a
tier prior -- what a flat-rate subscription seat is *reckoned* to have cost, for
a worker that reports no tokens. They were summed into one figure everywhere a
total appeared.

Measured on a real job run from the dashboard chat bar: four billed calls
totalling $0.0008 sat under one $0.06 tier prior, and the page reported
"$0.06 spend" as though a provider had invoiced it. A 75x overstatement, in the
one product whose stated claim is that a dollar figure carries its provenance.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from forgeos.contracts import Budget, JobSpec
from forgeos.ledger import open_ledger


@pytest.fixture
def ledger(tmp_path):
    return open_ledger(tmp_path / "ledger.db")


def _job(ledger) -> str:
    return ledger.open_job(JobSpec(objective="o", cwd=".", budget=Budget(max_usd=5.0)))


def _spend(ledger, job_id, *, kind, micros, tokens=0):
    return ledger.record_spend(
        job_id, worker_id="w", model="m", usd_micros=micros, kind=kind,
        tokens_in=tokens, tokens_out=tokens,
    )


# ------------------------------------------------------------- the split


def test_measured_only_excludes_a_modelled_tier_prior(ledger):
    """THE regression, with the real numbers from the job that exposed it."""
    job = _job(ledger)
    _spend(ledger, job, kind="call", micros=323, tokens=61)
    _spend(ledger, job, kind="call", micros=182, tokens=61)
    _spend(ledger, job, kind="call", micros=174, tokens=61)
    _spend(ledger, job, kind="call", micros=109, tokens=61)
    _spend(ledger, job, kind="estimate", micros=60_000)

    assert ledger.job_spend_micros(job, measured_only=True) == 788
    assert ledger.job_spend_micros(job) == 60_788, "default must stay total"


def test_the_split_reports_both_halves(ledger):
    job = _job(ledger)
    _spend(ledger, job, kind="call", micros=500)
    _spend(ledger, job, kind="estimate", micros=60_000)
    measured, modelled = ledger.job_spend_split(job)
    assert (measured, modelled) == (500, 60_000)


def test_a_job_with_only_estimates_reports_zero_measured(ledger):
    """A flat-rate seat that billed nothing must not read as money spent."""
    job = _job(ledger)
    _spend(ledger, job, kind="estimate", micros=60_000)
    assert ledger.job_spend_micros(job, measured_only=True) == 0
    assert ledger.job_spend_split(job) == (0, 60_000)


def test_a_job_with_only_calls_reports_zero_modelled(ledger):
    job = _job(ledger)
    _spend(ledger, job, kind="call", micros=1_234, tokens=10)
    assert ledger.job_spend_split(job) == (1_234, 0)


def test_an_empty_job_splits_to_zero_zero(ledger):
    assert ledger.job_spend_split(_job(ledger)) == (0, 0)


def test_task_spend_honours_the_same_distinction(ledger):
    job = _job(ledger)
    ledger.record_spend(job, worker_id="w", model="m", usd_micros=300,
                        kind="call", task_id="t1", tokens_in=5, tokens_out=5)
    ledger.record_spend(job, worker_id="w", model="m", usd_micros=60_000,
                        kind="estimate", task_id="t1")
    assert ledger.task_spend_micros("t1", measured_only=True) == 300
    assert ledger.task_spend_micros("t1") == 60_300


def test_an_unknown_kind_counts_as_modelled_not_measured(ledger):
    """Only `call` is a billed fact. Anything else must fall to the cautious
    side -- a new row kind appearing later must not silently join the measured
    total and inflate a receipt."""
    job = _job(ledger)
    _spend(ledger, job, kind="prior", micros=999)
    measured, modelled = ledger.job_spend_split(job)
    assert (measured, modelled) == (0, 999)


def test_the_default_is_unchanged_so_no_budget_check_silently_moves(ledger):
    """Every historical caller summed both. Changing that default under them
    would move a governor threshold without anyone asking for it."""
    job = _job(ledger)
    _spend(ledger, job, kind="call", micros=100)
    _spend(ledger, job, kind="estimate", micros=900)
    assert ledger.job_spend_micros(job) == 1_000
