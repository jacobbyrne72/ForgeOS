"""Catching a doomed attempt before it spends the budget proving it is doomed.

The escalation boundary bills twice: the failed attempt in full, then the
successful one. These signals cut the first half short — and the risk running
the other way is worse, so most of this file is about NOT stopping work that was
going somewhere.
"""

from __future__ import annotations

import pytest

from forgeos.core.abstention import (
    DEFAULT_EVIDENCE_GRACE_MICROS,
    Recommendation,
    detect_abstention,
    detect_repeat_failure,
    judge,
)


# ------------------------------------------------------------- abstention


@pytest.mark.parametrize("text", [
    "I don't have access to the deployment credentials.",
    "I cannot find the module you referenced anywhere in the repo.",
    "There is not enough context to make this change safely.",
    "I am not able to complete this without the schema.",
])
def test_a_stated_inability_is_a_handoff_not_a_retry(text):
    """Abstention is the worker's calibration WORKING. Retrying it burns the
    rest of its budget to re-learn what it already told you."""
    j = detect_abstention(text)
    assert j is not None and j.recommendation is Recommendation.HANDOFF
    assert j.quote


def test_a_request_for_a_decision_goes_to_a_human_not_a_bigger_model():
    """A stronger model cannot answer 'which behaviour did you want'. Routing
    it upward buys an expensive guess."""
    j = detect_abstention("This requires a decision about whether to keep the old API.")
    assert j is not None and j.recommendation is Recommendation.ASK_HUMAN


def test_asking_which_option_goes_to_a_human():
    j = detect_abstention("Two designs are possible — which one should I build?")
    assert j is not None and j.recommendation is Recommendation.ASK_HUMAN


@pytest.mark.parametrize("text", [
    "Done. Added the retry helper and a test; both pass.",
    "The function is `record_spend` and the guard is the inflight dedup.",
    "I cannot overstate how important this cache is.",
    "",
    "   ",
])
def test_ordinary_output_is_not_an_abstention(text):
    """False positives here stop work that was succeeding — the expensive
    direction. Note the third: 'cannot' appears, but not as an inability."""
    assert detect_abstention(text) is None


def test_the_worker_is_quoted_so_a_human_can_check_the_call():
    j = detect_abstention("Sorry, I do not have access to the production database.")
    assert "access" in j.quote
    assert "worker said" in j.render()


# ----------------------------------------------------------- repeat failure


def test_the_same_failure_three_times_stops():
    err = "ModuleNotFoundError: no module named 'widget'"
    j = detect_repeat_failure([err, err, err])
    assert j is not None and j.recommendation is Recommendation.STOP_LOOPING


def test_two_repeats_are_not_yet_a_loop():
    """Two can be a flake. Stopping there costs a real attempt."""
    err = "boom"
    assert detect_repeat_failure([err, err]) is None


def test_different_failures_are_progress_not_a_loop():
    """Moving through different errors is what debugging looks like."""
    assert detect_repeat_failure(["error A", "error B", "error C"]) is None


def test_only_the_most_recent_attempts_count():
    """An old repeat that the worker has since moved past must not stop it."""
    assert detect_repeat_failure(["x", "x", "x", "different", "another"]) is None


def test_whitespace_differences_still_count_as_the_same_failure():
    j = detect_repeat_failure(["a  b", "a b", " a   b "])
    assert j is not None


def test_matching_is_exact_not_fuzzy():
    """A near-match heuristic would eventually stop a worker genuinely making
    progress through similar-looking errors."""
    assert detect_repeat_failure([
        "failed on line 1", "failed on line 2", "failed on line 3",
    ]) is None


def test_an_empty_failure_list_is_not_a_loop():
    assert detect_repeat_failure([]) is None


# ------------------------------------------------------------------ stall


def test_spend_with_no_evidence_is_caught():
    j = judge(spend_micros=DEFAULT_EVIDENCE_GRACE_MICROS + 1, evidence_count=0)
    assert j.recommendation is Recommendation.STOP_LOOPING
    assert "no evidence" in j.reason


def test_early_spend_with_no_evidence_yet_is_normal():
    """Nothing to show in the first moments is a phase, not a failure."""
    j = judge(spend_micros=DEFAULT_EVIDENCE_GRACE_MICROS - 1, evidence_count=0)
    assert j.recommendation is Recommendation.CONTINUE


def test_spend_with_evidence_is_fine_however_large():
    j = judge(spend_micros=10_000_000, evidence_count=3)
    assert j.recommendation is Recommendation.CONTINUE


# -------------------------------------------------------------- precedence


def test_what_the_worker_said_outranks_what_was_inferred_about_it():
    """An explicit statement beats an inference drawn from counters."""
    j = judge(text="I do not have access to that repo.",
              failures=["e", "e", "e"], spend_micros=999_999, evidence_count=0)
    assert j.recommendation is Recommendation.HANDOFF


def test_a_loop_outranks_a_bare_spend_signal():
    j = judge(failures=["same", "same", "same"],
              spend_micros=999_999, evidence_count=0)
    assert j.recommendation is Recommendation.STOP_LOOPING


def test_a_healthy_attempt_draws_no_recommendation():
    j = judge(text="Working on it; two tests pass so far.",
              failures=["one thing"], spend_micros=1_000, evidence_count=2)
    assert j.recommendation is Recommendation.CONTINUE
    assert not j.should_stop


def test_judge_with_no_signals_at_all_continues():
    assert judge().recommendation is Recommendation.CONTINUE


# --------------------------------------------------------------- mechanics


def test_this_module_never_escalates_by_itself():
    """It returns a recommendation. A module that both detects trouble and
    spends money on it is one nobody can reason about."""
    import inspect

    from forgeos.core import abstention

    src = inspect.getsource(abstention)
    for forbidden in ("Gateway", "complete(", "record_spend", "escalate("):
        assert forbidden not in src, f"{forbidden} — this module must only advise"


def test_judgement_is_deterministic():
    args = dict(text="I cannot find the file.", failures=["a"], spend_micros=10)
    first = judge(**args)
    for _ in range(5):
        again = judge(**args)
        assert (again.recommendation, again.reason) == (first.recommendation, first.reason)


def test_no_model_call_is_made(monkeypatch):
    import socket

    def _boom(*a, **k):  # pragma: no cover
        raise AssertionError("judge attempted network I/O")

    monkeypatch.setattr(socket, "socket", _boom)
    monkeypatch.setattr(socket, "create_connection", _boom)
    assert judge(text="I cannot find it.").should_stop
