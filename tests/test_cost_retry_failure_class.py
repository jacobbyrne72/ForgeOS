"""Retry decisions driven by the failure taxonomy, not by error wording.

`cost_retry.py` decided on budget arithmetic plus substring-matching the error
string, while `contracts.FailureClass` -- a considered taxonomy that
`circuit_breaker.py` already consumed correctly -- sat unused in the same
package. Two classification systems, and the weaker one was wired up.

Worse than redundant: the waste list contained "rate limit" and "quota
exceeded". Both are TRANSIENT, the definitively retryable class, so the module
was giving up on exactly the failures most likely to succeed next attempt.
"""

from __future__ import annotations

import pytest

from forgeos.contracts import FailureClass
from forgeos.cost_retry import CostRetry


def _retry(**kw):
    return CostRetry(max_retries=3, base_cost_budget=10.0, **kw)


# ------------------------------------------------ the taxonomy decides


@pytest.mark.parametrize("fc", [FailureClass.SPECIFICATION, FailureClass.POLICY])
def test_a_failure_needing_a_human_is_never_retried(fc):
    """A wrong task and a denied permission do not improve by being asked
    again, however much budget is left."""
    ok, info = _retry().should_retry(0, retry_cost=0.001, failure_class=fc)
    assert not ok
    assert info["reason"] == "needs_human"
    assert info["failure_class"] == fc.value


@pytest.mark.parametrize("fc", [FailureClass.TRANSIENT, FailureClass.CONTEXT,
                                FailureClass.ENVIRONMENT, FailureClass.MODEL])
def test_every_other_class_is_allowed_to_retry_on_budget(fc):
    ok, info = _retry().should_retry(0, retry_cost=0.001, failure_class=fc)
    assert ok, info


def test_a_rate_limit_is_retried_when_classified_transient():
    """THE regression. "rate limit" was in the waste list, so the most
    retryable failure there is was treated as unretryable."""
    ok, _ = _retry().should_retry(
        0, last_error="429 rate limit exceeded", retry_cost=0.001,
        failure_class=FailureClass.TRANSIENT,
    )
    assert ok


def test_rate_limit_wording_alone_no_longer_blocks_a_retry():
    """Even with no failure class, the two wrong entries are gone."""
    ok, _ = _retry().should_retry(0, last_error="429 rate limit", retry_cost=0.001)
    assert ok


def test_quota_exceeded_wording_alone_no_longer_blocks_a_retry():
    ok, _ = _retry().should_retry(0, last_error="quota exceeded", retry_cost=0.001)
    assert ok


# ------------------------------------------------------- precedence


def test_the_taxonomy_outranks_the_error_text():
    """A definite answer beats a guess about someone else's wording."""
    ok, info = _retry().should_retry(
        0, last_error="invalid prompt", retry_cost=0.001,
        failure_class=FailureClass.TRANSIENT,
    )
    assert ok, info


def test_substring_matching_only_runs_without_a_failure_class():
    ok, info = _retry().should_retry(0, last_error="invalid prompt", retry_cost=0.001)
    assert not ok and info["reason"] == "waste_error"
    assert "FailureClass" in info["detail"]


def test_unretryable_is_decided_before_affordability():
    """Whether retrying CAN work is a different question from whether it is
    affordable, and the first answers no for free."""
    ok, info = _retry().should_retry(
        0, retry_cost=999_999.0, failure_class=FailureClass.POLICY,
    )
    assert not ok
    assert info["reason"] == "needs_human", "budget answered before the taxonomy"


# ------------------------------------------------ existing behaviour intact


def test_the_budget_ceiling_still_stops_a_retry():
    ok, info = _retry().should_retry(0, retry_cost=999.0,
                                     failure_class=FailureClass.TRANSIENT)
    assert not ok and info["reason"] == "budget_exceeded"


def test_max_retries_still_stops_a_retry():
    ok, info = _retry().should_retry(3, failure_class=FailureClass.TRANSIENT)
    assert not ok and info["reason"] == "max_retries_reached"


def test_the_cumulative_cap_still_applies():
    r = CostRetry(max_retries=5, base_cost_budget=10.0, max_total_retry_spend=0.05)
    r.retry_spend = 0.05
    ok, info = r.should_retry(0, retry_cost=0.01, failure_class=FailureClass.TRANSIENT)
    assert not ok and info["reason"] == "total_retry_budget_exceeded"


def test_callers_passing_no_failure_class_still_work():
    """Every existing call site invokes this without the new argument."""
    ok, _ = _retry().should_retry(0, retry_cost=0.001)
    assert ok
