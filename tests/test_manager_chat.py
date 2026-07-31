"""The manager you talk to — and the gate that stops it running away.

The feature is "you never write a worker prompt". The safety property is that
**nothing executes without an approval bound to the exact plan you read**. Those
are different things, and only the second one can be attacked, so most of this
file attacks it.

The approval check is mechanical on purpose. A conversational gate ("did the
user say yes?") is exactly what a persuasive model routes around, and the person
paying is the one who eats the difference.
"""

from __future__ import annotations

import pytest

from forgeos.contracts_v2 import RiskClass
from forgeos.manager_chat import (
    ApprovalError,
    ManagerSession,
    Phase,
    Question,
    draft_blueprint,
)


def _bp(**kw):
    base = dict(
        objective="Add provider failover without changing the public API",
        plan=["locate the retry path", "add failover", "prove it with a test"],
        criteria=[("C1", "Failover triggers on 429", ["unit-test:test_failover"])],
        max_usd=2.0,
        non_goals=["do not change the public API"],
        write_scope=["src/retry.py"],
    )
    base.update(kw)
    return draft_blueprint(**base)


def _session() -> ManagerSession:
    return ManagerSession(objective="Add provider failover")


# ===================================================== the approval gate


def test_nothing_executes_without_approval():
    """The gate, not a formality."""
    s = _session()
    s.propose(_bp())
    with pytest.raises(ApprovalError):
        s.job_cards()


def test_job_cards_refuse_before_anything_is_even_proposed():
    with pytest.raises(ApprovalError):
        _session().job_cards()


def test_approval_requires_the_digest_of_what_was_shown():
    s = _session()
    s.propose(_bp())
    with pytest.raises(ApprovalError):
        s.approve("")
    with pytest.raises(ApprovalError):
        s.approve("0" * 64)


def test_a_revised_blueprint_voids_the_previous_approval():
    """The attack this exists to stop: get approval, then change the plan.

    A blueprint revised after you read it must not carry your approval forward,
    however reasonable the revision looks.
    """
    s = _session()
    s.propose(_bp())
    s.approve(s.blueprint.contract.digest())
    assert s.is_approved

    s.propose(_bp(objective="Also refactor the billing module", max_usd=50.0))
    assert not s.is_approved, "approval survived a plan change"
    assert s.phase is Phase.AWAITING_APPROVAL
    with pytest.raises(ApprovalError):
        s.job_cards()


def test_the_short_digest_shown_to_a_human_is_accepted():
    """The rendered blueprint shows 16 chars; a human retyping that must work,
    or the gate becomes something people paste around rather than read."""
    s = _session()
    s.propose(_bp())
    s.approve(s.blueprint.contract.digest()[:16])
    assert s.is_approved


def test_approving_a_different_sessions_digest_does_not_work():
    a, b = _session(), _session()
    a.propose(_bp())
    b.propose(_bp(objective="Something else entirely"))
    with pytest.raises(ApprovalError):
        b.approve(a.blueprint.contract.digest())


# ============================================ the human never writes a prompt


def test_job_cards_carry_scope_budget_and_stop_conditions_from_the_contract():
    """The actual feature: intent in, fully-specified worker contract out."""
    s = _session()
    s.propose(_bp())
    s.approve(s.blueprint.contract.digest())

    cards = s.job_cards()
    assert len(cards) == 1
    card = cards[0]
    assert card.write_scope == ["src/retry.py"]
    assert card.criterion_ids == ["C1"]
    assert "criteria_pass" in card.stop_conditions
    assert "budget_exhausted" in card.stop_conditions
    assert card.mission_id == s.blueprint.contract.mission_id


def test_roles_split_criteria_across_workers():
    s = _session()
    s.propose(_bp(criteria=[
        ("C1", "Failover triggers", ["unit-test:a"]),
        ("C2", "Reviewed independently", ["review:human"]),
    ]))
    s.approve(s.blueprint.contract.digest())

    cards = s.job_cards(roles={"implementer": ["C1"], "verifier": ["C2"]})
    assert {c.role for c in cards} == {"implementer", "verifier"}
    assert next(c for c in cards if c.role == "verifier").criterion_ids == ["C2"]


def test_every_job_card_is_answerable_against_the_contract():
    s = _session()
    s.propose(_bp())
    s.approve(s.blueprint.contract.digest())
    for card in s.job_cards():
        assert card.covers(s.blueprint.contract) == [], "card claims an unknown criterion"


# ============================================================ the blueprint


def test_a_blueprint_cannot_promise_what_nobody_can_check():
    """`Criterion` requires a proof route, so an uncheckable promise cannot even
    be drafted."""
    with pytest.raises(Exception):
        _bp(criteria=[("C1", "It will be good", [])])


def test_the_rendering_shows_what_a_human_needs_to_decide():
    text = _bp().render()
    for expected in ("GOAL", "PLAN", "WILL NOT", "PROOF", "CEILING", "DIGEST"):
        assert expected in text, f"a human cannot approve without {expected}"


def test_an_estimate_is_never_displayed_as_a_measurement():
    """At the moment someone decides to spend, the two must not look alike."""
    text = _bp(estimated_usd=0.42).render()
    assert "estimated" in text and "measured" not in text


def test_no_exclusions_is_stated_rather_than_left_blank():
    """Silence about scope is how 'it also did this' happens."""
    assert "nothing excluded" in _bp(non_goals=[]).render()


# =========================================================== clarification


def test_questions_are_additive_and_never_re_asked():
    s = _session()
    s.ask_questions([Question(text="Which provider first?", why="routing order")])
    q = s.open_questions()[0]
    s.answer(q.id, "Anthropic")

    remaining = s.ask_questions([
        Question(text="Which provider first?", why="duplicate"),
        Question(text="Retry how many times?", why="budget"),
    ])
    assert [x.text for x in remaining] == ["Retry how many times?"]


def test_a_question_must_say_why_it_is_being_asked():
    """Without a reason the human cannot tell which answers matter, so they
    over-answer or disengage."""
    with pytest.raises(Exception):
        Question(text="What framework?")


# ============================================================= supervision


def test_a_healthy_worker_draws_no_intervention():
    """The cheapest supervision is the one that does not happen."""
    s = _session()
    assert s.follow_up({"state": "running", "tokens_used": 900, "evidence_count": 2}) is None
    assert s.follow_up({}) is None


def test_spend_without_evidence_is_caught():
    """The drift signature that matters: burning budget looking busy."""
    out = _session().follow_up({"state": "running", "tokens_used": 5000, "evidence_count": 0})
    assert out is not None and "evidence" in out.lower()


def test_a_stated_blocker_is_answered_not_re_diagnosed():
    out = _session().follow_up({"state": "running", "blocker": "no write access to src/"})
    assert out is not None and "no write access to src/" in out


def test_blocked_with_no_reason_is_asked_for_one():
    out = _session().follow_up({"state": "blocked"})
    assert out is not None and "what" in out.lower()


def test_supervision_never_reads_a_transcript():
    """A manager that re-reads a worker's conversation costs more than the work
    it supervises. Heartbeat keys only — no message history."""
    s = _session()
    hb = {"state": "running", "tokens_used": 10, "evidence_count": 1,
          "messages": ["a" * 5000], "transcript": ["b" * 5000]}
    assert s.follow_up(hb) is None, "a transcript key changed the decision"


def test_the_human_log_is_never_worker_context():
    """`say` records for the person. If it fed a worker prompt, a long
    conversation would silently become an expensive one."""
    s = _session()
    for i in range(50):
        s.say("human", f"message {i}")
    s.propose(_bp())
    s.approve(s.blueprint.contract.digest())
    card = s.job_cards()[0]
    blob = card.model_dump_json()
    assert "message 0" not in blob and "message 49" not in blob


# ================================================================ lifecycle


def test_phases_move_forward_through_a_normal_run():
    s = _session()
    assert s.phase is Phase.GATHERING
    s.propose(_bp())
    assert s.phase is Phase.AWAITING_APPROVAL
    s.approve(s.blueprint.contract.digest())
    assert s.phase is Phase.EXECUTING
    s.close(accepted=True)
    assert s.phase is Phase.DONE


def test_revisions_are_counted_so_replanning_is_visible():
    s = _session()
    s.propose(_bp())
    assert s.revisions == 0
    s.propose(_bp(objective="narrower"))
    s.propose(_bp(objective="narrower still"))
    assert s.revisions == 2


@pytest.mark.parametrize("risk", list(RiskClass))
def test_every_risk_class_renders(risk):
    assert risk.value in _bp(risk=risk).render()
