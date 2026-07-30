"""Economy tests: preflight refusal and the avoidance counterfactual ledger.

No LLM calls, no network. tiktoken runs locally.
"""

from __future__ import annotations

import pytest

from forgeos.catalog import ModelCard
from forgeos.economy import (
    Avoidance,
    AvoidanceLog,
    AvoidanceMethod,
    CallRefused,
    Decision,
    check,
    count_tokens,
    estimate_call,
)
from forgeos.economy.preflight import ESTIMATE_ENCODING


def _card(**kw) -> ModelCard:
    base = dict(
        model_id="made-up-model",
        provider="testprov",
        input_cost_per_1m=1.0,
        output_cost_per_1m=2.0,
        context=100_000,
    )
    base.update(kw)
    return ModelCard(**base)


# ---------------------------------------------------------------- counting


def test_known_model_count_is_exact():
    tc = count_tokens("hello world, this is a test", "gpt-4o")
    assert tc.exact is True
    assert tc.encoding_used != ESTIMATE_ENCODING
    assert tc.tokens > 0


def test_unknown_model_estimates_and_says_so():
    tc = count_tokens("x" * 401, "no-such-model-xyz")
    assert tc.exact is False
    assert tc.encoding_used == ESTIMATE_ENCODING
    assert tc.tokens == 101  # ceil(401 / 4)


def test_empty_model_falls_back_to_estimate():
    assert count_tokens("abcd").exact is False


def test_empty_text_is_zero_tokens():
    assert count_tokens("", "gpt-4o").tokens == 0
    assert count_tokens("").tokens == 0


def test_special_token_literals_do_not_raise():
    # tiktoken raises on special tokens by default; a doc QUOTING one must still count.
    tc = count_tokens("the marker <|endoftext|> appears in this doc", "gpt-4o")
    assert tc.tokens > 0
    assert tc.exact is True


# ---------------------------------------------------------------- estimate


def test_estimate_prices_against_the_card():
    est = estimate_call("x" * 400, 50, _card())
    # 100 in @ $1/1M + 50 out @ $2/1M = $0.0002 = 200 micros
    assert est.tokens_in == 100
    assert est.tokens_out == 50
    assert est.usd_micros == 200
    assert est.fits_context is True
    assert est.exact is False  # unknown model → estimated count, and labelled so


def test_estimate_flags_payload_bigger_than_model_context():
    est = estimate_call("x" * 400, 10, _card(context=50))
    assert est.fits_context is False


# ------------------------------------------------------------------- check


def test_allows_when_it_fits():
    est = estimate_call("x" * 400, 50, _card())
    v = check(est, remaining_micros=1_000)
    assert v.decision is Decision.ALLOW
    assert v.allowed


def test_refuses_when_estimate_exceeds_remaining_budget():
    est = estimate_call("x" * 400, 50, _card())
    v = check(est, remaining_micros=est.usd_micros - 1)
    assert v.decision is Decision.REFUSE_BUDGET
    assert not v.allowed
    assert "budget" in v.reason


def test_spending_exactly_to_the_cap_is_allowed():
    est = estimate_call("x" * 400, 50, _card())
    assert check(est, remaining_micros=est.usd_micros).allowed


def test_refuses_when_payload_exceeds_model_context():
    est = estimate_call("x" * 400, 10, _card(context=50))
    v = check(est, remaining_micros=10**9)
    assert v.decision is Decision.REFUSE_CONTEXT
    assert not v.allowed


def test_refuses_when_payload_exceeds_context_cap():
    est = estimate_call("x" * 400, 10, _card())
    v = check(est, remaining_micros=10**9, max_context=99)
    assert v.decision is Decision.REFUSE_CONTEXT


def test_context_refusal_outranks_budget_refusal():
    # Both violated: the physical impossibility wins the reason string.
    est = estimate_call("x" * 400, 10, _card(context=50))
    assert check(est, remaining_micros=0).decision is Decision.REFUSE_CONTEXT


def test_free_call_passes_an_exhausted_budget():
    est = estimate_call("x" * 400, 50, _card(input_cost_per_1m=0.0, output_cost_per_1m=0.0))
    assert check(est, remaining_micros=0).allowed


def test_estimated_counts_are_labelled_in_the_reason():
    est = estimate_call("x" * 400, 50, _card())  # unknown model → estimated
    assert "estimated" in check(est, remaining_micros=1_000).reason


def test_refusal_is_a_hard_signal():
    est = estimate_call("x" * 400, 50, _card())
    with pytest.raises(CallRefused):
        check(est, remaining_micros=0).raise_if_refused()
    check(est, remaining_micros=10**9).raise_if_refused()  # allow → no raise


# --------------------------------------------------------------- avoidance


@pytest.fixture()
def avd():
    log = AvoidanceLog(":memory:")
    yield log
    log.close()


def test_record_returns_id_and_persists(avd):
    rid = avd.record(
        "job1",
        AvoidanceMethod.PACK,
        baseline_tokens=10_000,
        actual_tokens=1_500,
        baseline_source="tiktoken count of full file",
    )
    rows = avd.for_job("job1")
    assert [r.id for r in rows] == [rid]
    assert rows[0].saved_tokens == 8_500
    assert rows[0].method is AvoidanceMethod.PACK


def test_saved_tokens_never_negative_when_actual_exceeds_baseline():
    row = Avoidance(
        job_id="j",
        method=AvoidanceMethod.CACHE_HIT,
        baseline_tokens=100,
        actual_tokens=150,
        baseline_source="ledger row of the original call",
    )
    assert row.saved_tokens == 0


def test_blank_baseline_source_is_inadmissible(avd):
    # A guessed baseline is not evidence; the store must refuse it at the edge.
    with pytest.raises(ValueError):
        avd.record("j", AvoidanceMethod.DEDUP, baseline_tokens=10, actual_tokens=1, baseline_source="   ")


def test_negative_token_counts_rejected(avd):
    with pytest.raises(ValueError):
        avd.record("j", AvoidanceMethod.PACK, baseline_tokens=-5, actual_tokens=0, baseline_source="s")
    with pytest.raises(ValueError):
        avd.record("j", AvoidanceMethod.PACK, baseline_tokens=5, actual_tokens=-1, baseline_source="s")


def test_empty_store_totals_are_zero_not_an_error(avd):
    assert avd.totals() == {
        "baseline_tokens": 0,
        "actual_tokens": 0,
        "saved_tokens": 0,
        "saved_pct": 0.0,
        "by_method": {},
    }


def test_totals_maths(avd):
    avd.record(
        "j",
        AvoidanceMethod.PACK,
        baseline_tokens=1_000,
        actual_tokens=200,
        baseline_source="tiktoken count of full file",
    )
    avd.record(
        "j",
        AvoidanceMethod.CACHE_HIT,
        baseline_tokens=500,
        actual_tokens=0,
        baseline_source="ledger row of the original call",
    )
    t = avd.totals()
    assert t["baseline_tokens"] == 1_500
    assert t["actual_tokens"] == 200
    assert t["saved_tokens"] == 1_300
    assert t["saved_pct"] == pytest.approx(86.67, abs=0.01)
    assert t["by_method"]["pack"]["saved_tokens"] == 800
    assert t["by_method"]["cache_hit"] == {
        "count": 1,
        "baseline_tokens": 500,
        "actual_tokens": 0,
        "saved_tokens": 500,
        "saved_pct": 100.0,
    }


def test_overspend_row_cannot_cancel_a_real_saving(avd):
    avd.record(
        "j",
        AvoidanceMethod.PACK,
        baseline_tokens=1_000,
        actual_tokens=900,
        baseline_source="tiktoken count of full file",
    )  # saved 100
    avd.record(
        "j",
        AvoidanceMethod.DETERMINISTIC,
        baseline_tokens=100,
        actual_tokens=150,
        baseline_source="ledger row of a prior equivalent call",
    )  # overspent by 50
    t = avd.totals()
    assert t["saved_tokens"] == 100  # per-row clamp: 100 + 0, not 100 - 50
    assert t["by_method"]["deterministic"]["saved_tokens"] == 0


def test_totals_filter_by_job(avd):
    avd.record("a", AvoidanceMethod.PACK, baseline_tokens=100, actual_tokens=10, baseline_source="tiktoken count")
    avd.record("b", AvoidanceMethod.PACK, baseline_tokens=999, actual_tokens=0, baseline_source="tiktoken count")
    assert avd.totals(job_id="a")["baseline_tokens"] == 100
    assert avd.totals(job_id="a")["saved_tokens"] == 90
    assert avd.totals()["baseline_tokens"] == 1_099


def test_zero_baseline_row_does_not_divide_by_zero(avd):
    avd.record(
        "j",
        AvoidanceMethod.VERIFY_GATE,
        baseline_tokens=0,
        actual_tokens=0,
        baseline_source="verified green in ledger",
    )
    t = avd.totals()
    assert t["saved_pct"] == 0.0
    assert t["by_method"]["verify_gate"]["saved_pct"] == 0.0
