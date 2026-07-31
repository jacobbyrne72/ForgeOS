"""Auto-repair retry on schema/format validation failure, plus the inner
verify/repair loop's own prior-iteration compression.

Source: a tutorial on the `instructor` library (patch a client, validate
against a pydantic schema, retry with the exact error fed back so the model
can self-correct) and a Q&A on compressing "try again" loop history before
the next iteration -- see `forgeos/core/manager.py`'s module docstring and
the `ParseAttempt` / `RepairAttempt` / `compress_repair_history` docstrings
for the full citation trail.

Two guarantees matter more than the feature itself:

1. A schema/format violation is `FailureClass.SPECIFICATION`, never
   `.MODEL` -- `router.Router.escalate` only escalates on MODEL, so a
   formatting mistake any tier could make must never buy a pricier one.
2. Compression drops whole low-value iterations; it never rewords or trims
   the exact error text inside one it keeps (arXiv:2607.12161) -- stripping
   exactly that kind of anchor was measured to raise billed cost and lower
   task success, because the next attempt can no longer match its fix
   against the failure.

No LLM calls, no network -- `complete` is a scripted fake, exactly like
tests/test_manager.py.
"""

from __future__ import annotations

import json

from forgeos.contracts import FailureClass
from forgeos.core.manager import (
    Manager,
    ManagerDecision,
    RepairAttempt,
    SCHEMA_REPAIR_FAILURE_CLASS,
    compress_repair_history,
)
from forgeos.core.router import Route, Router, Tier
from forgeos.registry import Adapter, CostTier, Registry, WorkerProfile


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


def clean_heartbeat(**overrides) -> dict:
    hb = {"task_id": "t1", "state": "running", "needs_manager": False}
    hb.update(overrides)
    return hb


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


# ==================================================== retry once, error appended


def test_malformed_json_repair_prompt_appends_the_exact_json_error():
    """The repair prompt is the SAME prompt (byte-identical prefix) plus the
    exact JSON syntax error appended in the tail -- never a generic 'reply in
    JSON only' reprompt, and never a rebuild of the prefix."""
    bad = "{oops not json"
    complete = ScriptedComplete([bad, bad])
    mgr = Manager(complete)
    hb = clean_heartbeat(escalations=["loop"], needs_manager=True)
    mgr.decide(hb, failure=None, attempts=0, max_attempts=5)

    assert len(complete.calls) == 2
    prompt1, prompt2 = complete.calls
    assert prompt2.startswith(prompt1), "a repair retry appends to the SAME prompt, never edits it"
    appended = prompt2[len(prompt1):]

    try:
        json.loads(bad)
        raise AssertionError("fixture must actually be invalid JSON")
    except json.JSONDecodeError as exc:
        exact_json_error = str(exc)
    assert exact_json_error in appended, "the exact parser error must survive byte-for-byte"


def test_schema_violation_repair_prompt_appends_the_exact_pydantic_error():
    """Valid JSON, invalid schema (FOLLOW_UP with empty instructions) is a
    genuine pydantic ValidationError, not a JSON syntax error -- the repair
    prompt must carry ITS exact message, not a generic one."""
    bad = verdict_json(decision="follow_up", instructions=[])
    complete = ScriptedComplete([bad, bad])
    mgr = Manager(complete)
    hb = clean_heartbeat(escalations=["low_confidence"], needs_manager=True)
    verdict = mgr.decide(hb, failure=None, attempts=0, max_attempts=5)

    prompt1, prompt2 = complete.calls
    assert prompt2.startswith(prompt1)
    appended = prompt2[len(prompt1):]
    assert "FOLLOW_UP requires at least one instruction" in appended, (
        "the exact pydantic error text must survive verbatim so the model can "
        "self-correct against its own specific mistake"
    )
    assert verdict.decision is ManagerDecision.ASK_HUMAN


def test_repair_retry_recovers_without_being_treated_as_a_failure():
    """The point of the feature: fix on the repair turn and the caller never
    sees a failure at all -- not ASK_HUMAN, not ESCALATE, just the recovered
    verdict, exactly as if the model had gotten it right the first time."""
    bad = "{not valid json"
    good = verdict_json(decision="continue", instructions=[])
    complete = ScriptedComplete([bad, good])
    mgr = Manager(complete)
    hb = clean_heartbeat(escalations=["stuck"], needs_manager=True)
    verdict = mgr.decide(hb, failure=None, attempts=0, max_attempts=5)

    assert verdict.decision is ManagerDecision.CONTINUE
    assert mgr.model_calls_made == 2


def test_repair_retry_happens_at_most_once():
    """Exactly one retry -- never a third call, however bad the second reply
    still is. `instructor`'s pattern retries a bounded number of times; this
    harness's bound is one."""
    complete = ScriptedComplete(["nope", "still nope"])
    mgr = Manager(complete)
    hb = clean_heartbeat(escalations=["loop"], needs_manager=True)
    verdict = mgr.decide(hb, failure=None, attempts=0, max_attempts=5)

    assert mgr.model_calls_made == 2
    assert len(complete.calls) == 2
    assert verdict.decision is ManagerDecision.ASK_HUMAN


def test_first_attempt_success_never_triggers_a_repair_call():
    """No violation, no retry -- the repair machinery must not fire on a
    clean first reply."""
    complete = ScriptedComplete([verdict_json(decision="split", instructions=[])])
    mgr = Manager(complete)
    hb = clean_heartbeat(escalations=["unclear"], needs_manager=True)
    verdict = mgr.decide(hb, failure=None, attempts=0, max_attempts=5)

    assert verdict.decision is ManagerDecision.SPLIT
    assert mgr.model_calls_made == 1


# ============================================== SPECIFICATION, never MODEL


def test_schema_repair_failure_class_is_specification_not_model():
    assert SCHEMA_REPAIR_FAILURE_CLASS is FailureClass.SPECIFICATION
    assert SCHEMA_REPAIR_FAILURE_CLASS is not FailureClass.MODEL


def test_specification_failures_never_escalate_the_price_tier():
    """The CRITICAL guarantee, proven against the real router: a
    SPECIFICATION classification is refused escalation, unlike MODEL --
    escalating would buy premium tokens for a mistake any tier would make."""
    registry = Registry([
        WorkerProfile(worker_id="cheap.local", adapter=Adapter.OLLAMA, tier=CostTier.FREE,
                      capabilities={"edit"}, can_edit_files=True, prior_win_rate=0.9),
        WorkerProfile(worker_id="premium.cloud", adapter=Adapter.CLI_TEAM, tier=CostTier.PREMIUM,
                      capabilities={"edit"}, can_edit_files=True, prior_win_rate=0.95),
    ])
    router = Router(registry)
    current = Route(tier=Tier.LOCAL, worker_id="cheap.local")

    assert router.escalate(current, SCHEMA_REPAIR_FAILURE_CLASS, ["edit"]) is None
    assert router.escalate(current, FailureClass.MODEL, ["edit"]) is not None, (
        "sanity check: MODEL failures on the same setup DO escalate, so the "
        "SPECIFICATION refusal above is the schema failure class actually "
        "being honoured, not the router refusing to escalate at all"
    )


# ==================================================== compress_repair_history


def test_compress_repair_history_is_a_noop_for_a_single_iteration():
    """The manager's real usage never has more than one prior iteration to
    compress before building the repair prompt -- this pins that the no-op
    path leaves it completely untouched."""
    history = [RepairAttempt(iteration=1, raw_reply="{bad", error="boom")]
    compressed = compress_repair_history(history, trigger_tokens=1, clear_at_least_tokens=1)
    assert compressed == history


def test_compress_repair_history_drops_whole_older_entries_never_touches_the_newest():
    long_error = "X" * 2000
    history = [
        RepairAttempt(iteration=1, raw_reply="reply one", error="oldest " + long_error),
        RepairAttempt(iteration=2, raw_reply="reply two", error="middle " + long_error),
        RepairAttempt(iteration=3, raw_reply="reply three", error="newest " + long_error),
    ]

    compressed = compress_repair_history(
        history, keep_last_n=1, trigger_tokens=10, clear_at_least_tokens=50
    )

    assert len(compressed) == 3
    by_iteration = {r.iteration: r for r in history}
    # newest is protected by keep_last_n=1 and survives byte-for-byte
    assert compressed[-1].error == by_iteration[3].error
    assert compressed[-1].raw_reply == by_iteration[3].raw_reply
    # every surviving entry is EITHER byte-identical to its source OR a whole
    # placeholder -- never a reworded/trimmed variant of the original text
    for c in compressed:
        original = by_iteration[c.iteration].error
        assert c.error == original or c.error.startswith("[tool result cleared")
    at_least_one_cleared = any(c.error != by_iteration[c.iteration].error for c in compressed)
    assert at_least_one_cleared, "the tiny budget must clear at least one whole older iteration"


def test_compress_repair_history_exact_error_string_survives_byte_for_byte():
    """Load-bearing per arXiv:2607.12161: the newest (kept) iteration's exact
    error text -- the anchor the next attempt matches its fix against -- is
    never reworded or trimmed, even under a tight compression budget."""
    exact_error = (
        "schema validation failed: 1 validation error for ManagerVerdict\n"
        "instructions\n  Value error, FOLLOW_UP requires at least one instruction "
        "[type=value_error]"
    )
    history = [
        RepairAttempt(iteration=1, raw_reply="junk reply, ignore me", error="X" * 5000),
        RepairAttempt(iteration=2, raw_reply="{'decision': 'follow_up'}", error=exact_error),
    ]

    compressed = compress_repair_history(
        history, keep_last_n=1, trigger_tokens=10, clear_at_least_tokens=1
    )

    assert compressed[-1].error == exact_error, "byte-for-byte, not a paraphrase"


def test_compress_repair_history_below_trigger_leaves_everything_untouched():
    """Below the token trigger, compressing is pure waste -- a no-op, exactly
    like `economy.reducer.clear_tool_results` itself guarantees."""
    history = [
        RepairAttempt(iteration=1, raw_reply="a", error="b"),
        RepairAttempt(iteration=2, raw_reply="c", error="d"),
    ]
    compressed = compress_repair_history(history, trigger_tokens=1_000_000)
    assert compressed == history
