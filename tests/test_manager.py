"""Manager tests.

The manager runs on a free-tier model, so the whole point of this module is
proving the harness never trusts that model more than it has to: a clean
heartbeat and every classified `FailureClass` must be decided without ever
calling `complete()`, and when the model genuinely is needed, garbage output
must retry once and then fail safe -- never to CONTINUE, because a manager
that guesses "carry on" after failing to parse twice is how a runaway loop
keeps running.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from forgeos.contracts import FailureClass, Scope, TaskSpec
from forgeos.core.manager import (
    Manager,
    ManagerDecision,
    ManagerVerdict,
    SafetyVerdict,
    parallel_safety,
    wake_triggers,
)


def clean_heartbeat(**overrides) -> dict:
    hb = {"task_id": "t1", "state": "running", "needs_manager": False}
    hb.update(overrides)
    return hb


class ScriptedComplete:
    """Fake `complete()` -- no network, no real model. Returns replies from a
    fixed script, one per call, and records every prompt it was given."""

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        if not self._replies:
            raise AssertionError("ScriptedComplete exhausted -- more calls than scripted replies")
        return self._replies.pop(0)


def verdict_json(**overrides) -> str:
    payload = {
        "decision": "follow_up",
        "reason_code": "test",
        "instructions": ["do the thing"],
        "context_additions": [],
        "escalate_after_failed_attempts": 1,
        "confidence": 0.7,
    }
    payload.update(overrides)
    return json.dumps(payload)


# ------------------------------------------------------------- wake_triggers


def test_clean_heartbeat_has_no_wake_triggers():
    assert wake_triggers(clean_heartbeat()) == []


def test_needs_manager_flag_wakes():
    assert wake_triggers(clean_heartbeat(needs_manager=True)) == ["needs_manager"]


def test_escalations_wake():
    hb = clean_heartbeat(escalations=["stuck", "loop"])
    assert wake_triggers(hb) == ["escalation:stuck", "escalation:loop"]


def test_blocker_wakes():
    assert wake_triggers(clean_heartbeat(blocker="missing dependency")) == ["blocker"]


def test_failed_tests_wake():
    hb = clean_heartbeat(tests={"passed": 2, "failed": 1})
    assert wake_triggers(hb) == ["tests_failed"]


def test_passing_tests_do_not_wake():
    hb = clean_heartbeat(tests={"passed": 5, "failed": 0})
    assert wake_triggers(hb) == []


def test_trouble_state_wakes():
    assert wake_triggers(clean_heartbeat(state="blocked")) == ["state:blocked"]


# ------------------------------------------------------------------- clean


def test_clean_heartbeat_decides_continue_with_zero_model_calls():
    complete = ScriptedComplete([])
    mgr = Manager(complete)
    verdict = mgr.decide(clean_heartbeat(), failure=None, attempts=0, max_attempts=5)
    assert verdict.decision is ManagerDecision.CONTINUE
    assert verdict.instructions == []
    assert mgr.model_calls_made == 0
    assert complete.calls == []


# ------------------------------------------------------- deterministic failure


@pytest.mark.parametrize(
    "failure,expected_decision",
    [
        (FailureClass.ENVIRONMENT, ManagerDecision.FOLLOW_UP),
        (FailureClass.CONTEXT, ManagerDecision.ADD_CONTEXT),
        (FailureClass.TRANSIENT, ManagerDecision.FOLLOW_UP),
        (FailureClass.MODEL, ManagerDecision.ESCALATE),
        (FailureClass.SPECIFICATION, ManagerDecision.ASK_HUMAN),
        (FailureClass.POLICY, ManagerDecision.ASK_HUMAN),
    ],
)
def test_every_failure_class_is_decided_deterministically(failure, expected_decision):
    complete = ScriptedComplete([])
    mgr = Manager(complete)
    verdict = mgr.decide(clean_heartbeat(), failure=failure, attempts=0, max_attempts=5)
    assert verdict.decision is expected_decision
    assert mgr.model_calls_made == 0
    assert complete.calls == []


def test_environment_failure_deterministic_with_zero_model_calls():
    complete = ScriptedComplete([])
    mgr = Manager(complete)
    verdict = mgr.decide(
        clean_heartbeat(blocker="pip install failed"),
        failure=FailureClass.ENVIRONMENT,
        attempts=1,
        max_attempts=5,
    )
    assert verdict.decision is ManagerDecision.FOLLOW_UP
    assert verdict.instructions  # non-empty, required by validation for FOLLOW_UP
    assert mgr.model_calls_made == 0


def test_context_failure_surfaces_blocker_as_context_addition():
    complete = ScriptedComplete([])
    mgr = Manager(complete)
    verdict = mgr.decide(
        clean_heartbeat(blocker="missing symbol Foo.bar"),
        failure=FailureClass.CONTEXT,
        attempts=0,
        max_attempts=5,
    )
    assert verdict.decision is ManagerDecision.ADD_CONTEXT
    assert verdict.context_additions == ["missing symbol Foo.bar"]
    assert mgr.model_calls_made == 0


# ------------------------------------------------------------- max attempts


def test_max_attempts_routes_to_human_even_on_model_failure():
    """A MODEL failure would normally ESCALATE -- but out of attempts means a
    human decides, never another rung of the ladder."""
    complete = ScriptedComplete([])
    mgr = Manager(complete)
    verdict = mgr.decide(clean_heartbeat(), failure=FailureClass.MODEL, attempts=5, max_attempts=5)
    assert verdict.decision is ManagerDecision.ASK_HUMAN
    assert mgr.model_calls_made == 0


def test_max_attempts_routes_to_human_without_governor_failure():
    complete = ScriptedComplete([])
    mgr = Manager(complete)
    hb = clean_heartbeat(escalations=["stuck"], needs_manager=True)
    verdict = mgr.decide(hb, failure=None, attempts=3, max_attempts=3)
    assert verdict.decision is ManagerDecision.ASK_HUMAN
    assert mgr.model_calls_made == 0


# --------------------------------------------------------- model-backed path


def test_worker_escalation_without_failure_calls_model_once_when_valid():
    complete = ScriptedComplete([verdict_json(decision="split", instructions=[])])
    mgr = Manager(complete)
    hb = clean_heartbeat(escalations=["unclear"], needs_manager=True)
    verdict = mgr.decide(hb, failure=None, attempts=0, max_attempts=5)
    assert verdict.decision is ManagerDecision.SPLIT
    assert mgr.model_calls_made == 1


def test_prose_wrapped_json_is_extracted_and_parsed():
    payload = verdict_json(decision="add_context", instructions=[], context_additions=["ref1"])
    reply = f"Sure thing, here is my decision:\n{payload}\nHope that helps!"
    complete = ScriptedComplete([reply])
    mgr = Manager(complete)
    hb = clean_heartbeat(escalations=["low_confidence"], needs_manager=True)
    verdict = mgr.decide(hb, failure=None, attempts=0, max_attempts=5)
    assert verdict.decision is ManagerDecision.ADD_CONTEXT
    assert verdict.context_additions == ["ref1"]
    assert mgr.model_calls_made == 1  # parsed on the first attempt, no retry needed


def test_empty_string_reply_retries_once_then_falls_back():
    complete = ScriptedComplete(["", ""])
    mgr = Manager(complete)
    hb = clean_heartbeat(escalations=["stuck"], needs_manager=True)
    verdict = mgr.decide(hb, failure=None, attempts=0, max_attempts=5)
    assert verdict.decision in (ManagerDecision.BLOCK, ManagerDecision.ASK_HUMAN)
    assert verdict.decision is not ManagerDecision.CONTINUE
    assert mgr.model_calls_made == 2
    assert len(complete.calls) == 2
    assert complete.calls[0] != complete.calls[1]  # second prompt is the terser retry


def test_malformed_json_retries_once_then_falls_back():
    complete = ScriptedComplete(["{not valid json", "{also not valid"])
    mgr = Manager(complete)
    hb = clean_heartbeat(escalations=["loop"], needs_manager=True)
    verdict = mgr.decide(hb, failure=None, attempts=0, max_attempts=5)
    assert verdict.decision in (ManagerDecision.BLOCK, ManagerDecision.ASK_HUMAN)
    assert verdict.decision is not ManagerDecision.CONTINUE
    assert mgr.model_calls_made == 2


def test_garbage_prose_with_no_json_retries_then_falls_back_never_continue():
    complete = ScriptedComplete(["I think we should just keep going, all good!", "yeah still fine"])
    mgr = Manager(complete)
    hb = clean_heartbeat(escalations=["stuck"], needs_manager=True)
    verdict = mgr.decide(hb, failure=None, attempts=0, max_attempts=5)
    assert verdict.decision is not ManagerDecision.CONTINUE
    assert verdict.decision in (ManagerDecision.BLOCK, ManagerDecision.ASK_HUMAN)
    assert mgr.model_calls_made == 2


def test_follow_up_missing_instructions_from_model_is_rejected_then_retried():
    """The model claims FOLLOW_UP but forgets instructions -- validation must
    reject that reply just like any other malformed output."""
    bad = verdict_json(decision="follow_up", instructions=[])
    complete = ScriptedComplete([bad, bad])
    mgr = Manager(complete)
    hb = clean_heartbeat(escalations=["low_confidence"], needs_manager=True)
    verdict = mgr.decide(hb, failure=None, attempts=0, max_attempts=5)
    assert verdict.decision in (ManagerDecision.BLOCK, ManagerDecision.ASK_HUMAN)
    assert mgr.model_calls_made == 2


def test_recovers_on_retry_after_one_bad_reply():
    complete = ScriptedComplete(["not json at all", verdict_json(decision="continue", instructions=[])])
    mgr = Manager(complete)
    hb = clean_heartbeat(escalations=["low_confidence"], needs_manager=True)
    verdict = mgr.decide(hb, failure=None, attempts=0, max_attempts=5)
    assert verdict.decision is ManagerDecision.CONTINUE
    assert mgr.model_calls_made == 2


# ----------------------------------------------------------------- validation


def test_follow_up_requires_instructions():
    with pytest.raises(ValidationError):
        ManagerVerdict(decision=ManagerDecision.FOLLOW_UP, reason_code="x", instructions=[])


def test_continue_forbids_instructions():
    with pytest.raises(ValidationError):
        ManagerVerdict(decision=ManagerDecision.CONTINUE, reason_code="x", instructions=["do it"])


def test_follow_up_with_instructions_is_valid():
    v = ManagerVerdict(decision=ManagerDecision.FOLLOW_UP, reason_code="x", instructions=["go"])
    assert v.instructions == ["go"]


def test_blank_reason_code_rejected():
    with pytest.raises(ValidationError):
        ManagerVerdict(decision=ManagerDecision.BLOCK, reason_code="   ")


def test_negative_escalate_after_failed_attempts_rejected():
    with pytest.raises(ValidationError):
        ManagerVerdict(decision=ManagerDecision.BLOCK, reason_code="x", escalate_after_failed_attempts=-1)


# ============================================== parallel-safety pre-flight
# Anthropic vs. Cognition on multi-agent decomposition: parallel subagents
# silently diverge when they share state neither one fully sees, even when
# their files never overlap. Every test here is one instance of that failure
# mode, or a pin that the conservative bias holds when there is nothing to
# flag.


def _spec(subject: str, description: str, *, acceptance=(), paths=("src/default.py",)) -> TaskSpec:
    return TaskSpec(job_id="j1", subject=subject, description=description,
                    acceptance=list(acceptance), scope=Scope(paths=list(paths)))


def test_disjoint_scope_and_no_shared_contract_is_safe():
    a = _spec("fix retry", "Fix the retry backoff bug in the scheduler.", paths=["src/a.py"])
    b = _spec("rename helper", "Rename the logging helper for clarity.", paths=["src/b.py"])
    verdict = parallel_safety(a, b)
    assert verdict.safe
    assert verdict.reasons == []
    assert verdict.introduces_before is None


def test_identical_scope_paths_are_not_safe():
    a = _spec("a", "Fix a bug.", paths=["src/shared.py"])
    b = _spec("b", "Fix another bug.", paths=["src/shared.py"])
    verdict = parallel_safety(a, b)
    assert not verdict.safe
    assert any("share scope" in r for r in verdict.reasons)


def test_glob_overlap_reuses_leases_patterns_overlap():
    """Overlap detection is the exact question a path lease already answers
    -- this pins that manager.py calls `patterns_overlap` rather than a
    second glob matcher that could silently drift from it."""
    a = _spec("a", "Fix a bug.", paths=["src/**"])
    b = _spec("b", "Fix another bug.", paths=["src/pkg/foo.py"])
    verdict = parallel_safety(a, b)
    assert not verdict.safe
    assert any("share scope" in r for r in verdict.reasons)


def test_unscoped_task_is_conservatively_unsafe():
    """An empty scope.paths means the task can touch anything -- non-overlap
    cannot be proven, so this must not read as 'nothing to check'."""
    a = _spec("a", "Fix a bug.", paths=[])
    b = _spec("b", "Fix another bug.", paths=["src/b.py"])
    verdict = parallel_safety(a, b)
    assert not verdict.safe
    assert any("no declared scope.paths" in r for r in verdict.reasons)


def test_shared_contract_blocks_even_with_disjoint_scope():
    a = _spec("build gateway",
              "Create the `PaymentGateway` class that exposes charge and refund.",
              paths=["src/gateway.py"])
    b = _spec("wire checkout",
              "Wire the checkout flow to call PaymentGateway.charge() on submit.",
              paths=["src/checkout.py"])
    verdict = parallel_safety(a, b)
    assert not verdict.safe
    assert any("PaymentGateway" in r for r in verdict.reasons)
    assert verdict.introduces_before == (a.id, b.id)


def test_shared_contract_direction_is_detected_either_way():
    """The introducer is whichever task's text uses the introduce-verb near
    the symbol, regardless of which one is passed as `a` or `b`."""
    a = _spec("wire checkout",
              "Wire the checkout flow to call PaymentGateway.charge() on submit.",
              paths=["src/checkout.py"])
    b = _spec("build gateway",
              "Create the `PaymentGateway` class that exposes charge and refund.",
              paths=["src/gateway.py"])
    verdict = parallel_safety(a, b)
    assert not verdict.safe
    assert verdict.introduces_before == (b.id, a.id)


def test_mentioning_an_existing_symbol_without_an_introduce_verb_is_not_flagged():
    """Referencing something that already exists is not a contract conflict --
    only text following create/add/introduce/define counts as a claim."""
    a = _spec("a", "Uses the existing Foo class to validate input.", paths=["src/a.py"])
    b = _spec("b", "Also touches the Foo class in a different way.", paths=["src/b.py"])
    verdict = parallel_safety(a, b)
    assert verdict.safe


def test_snake_case_symbol_is_detected():
    a = _spec("a", "Add the `reduce_pytest` helper for parsing test output.",
              paths=["src/a.py"])
    b = _spec("b", "Call reduce_pytest on the raw output before summarizing.",
              paths=["src/b.py"])
    verdict = parallel_safety(a, b)
    assert not verdict.safe
    assert any("reduce_pytest" in r for r in verdict.reasons)


def test_dotted_path_symbol_is_detected():
    a = _spec("a", "Introduce forgeos.core.newmod.Thing for the new pipeline.",
              paths=["src/a.py"])
    b = _spec("b", "Depends on forgeos.core.newmod.Thing to run.", paths=["src/b.py"])
    verdict = parallel_safety(a, b)
    assert not verdict.safe


def test_quoted_name_symbol_is_detected():
    a = _spec("a", 'Define "MAX_RETRY_WINDOW" as the new constant.', paths=["src/a.py"])
    b = _spec("b", "Read the MAX_RETRY_WINDOW value at startup.", paths=["src/b.py"])
    verdict = parallel_safety(a, b)
    assert not verdict.safe


def test_pure_scope_overlap_has_no_introduces_before():
    """A path-overlap-only violation has no natural direction -- either order
    sequences the pair correctly, so this must stay None rather than
    implying a resolution the check does not actually know."""
    a = _spec("a", "Fix a bug.", paths=["src/shared.py"])
    b = _spec("b", "Fix another bug.", paths=["src/shared.py"])
    verdict = parallel_safety(a, b)
    assert not verdict.safe
    assert verdict.introduces_before is None


def test_both_violations_are_reported_together():
    """One fix at a time would mean one re-check per problem -- both the
    scope overlap and the contract conflict should surface in one verdict."""
    a = _spec("build gateway",
              "Create the `PaymentGateway` class that exposes charge and refund.",
              paths=["src/shared.py"])
    b = _spec("wire checkout",
              "Wire the checkout flow to call PaymentGateway.charge() on submit.",
              paths=["src/shared.py"])
    verdict = parallel_safety(a, b)
    assert not verdict.safe
    assert len(verdict.reasons) == 2
    assert any("share scope" in r for r in verdict.reasons)
    assert any("PaymentGateway" in r for r in verdict.reasons)


def test_a_genuinely_independent_pair_is_safe_not_flagged():
    """The check must not become unpassable -- a real independent pair with
    real scope and a real contract mention (but no introduce verb) must
    still be marked safe."""
    a = _spec("a", "Fix the retry backoff bug in the scheduler.", paths=["src/a.py"],
              acceptance=["scheduler tests pass"])
    b = _spec("b", "Add a docstring to the existing helper function.", paths=["src/b.py"],
              acceptance=["helper docstring present"])
    verdict = parallel_safety(a, b)
    assert verdict.safe
    assert verdict.reasons == []
