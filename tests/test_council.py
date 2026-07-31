"""The panel `should_convene_council` decides to convene, and never could.

The router returned a boolean that nothing acted on: the decision to spend the
most expensive tool in the system was implemented, the tool was not.

Most of this file is about the three ways a naive "ask N models, take the
majority" gets it wrong.
"""

from __future__ import annotations

from forgeos.core.council import (
    ABSTENTION_LIMIT,
    Outcome,
    Verdict,
    Vote,
    convene,
    tally,
)


def _v(member, choice, family=""):
    return Vote(member=member, choice=choice, family=family)


# ------------------------------------------------------------- deciding


def test_a_clear_majority_decides():
    v = tally([_v("a", "yes"), _v("b", "yes"), _v("c", "no")])
    assert v.outcome is Outcome.DECIDED and v.choice == "yes"
    assert v.tally == {"yes": 2, "no": 1}


def test_unanimity_is_measured_among_those_who_answered():
    """Unanimity including abstainers would be a different, weaker claim."""
    v = tally([_v("a", "yes"), _v("b", "yes"), _v("c", None)])
    assert v.unanimous and v.answered == 2


# ------------------------------------------------------ a tie is a result


def test_a_tie_does_not_silently_pick_the_first_option():
    """Returning the leading option on a split records a coin flip as a
    decision."""
    v = tally([_v("a", "yes"), _v("b", "no")])
    assert v.outcome is Outcome.SPLIT
    assert v.choice is None
    assert "tied" in v.reason


def test_a_split_escalates_to_a_human():
    assert tally([_v("a", "x"), _v("b", "y")]).needs_human


def test_a_plurality_short_of_a_majority_is_a_split():
    """Two of five is the largest group and still not a majority."""
    v = tally([_v("a", "x"), _v("b", "x"), _v("c", "y"), _v("d", "z"), _v("e", "w")])
    assert v.outcome is Outcome.SPLIT
    assert "not a majority" in v.reason


def test_the_tie_report_names_every_tied_option():
    v = tally([_v("a", "x"), _v("b", "y")])
    assert "x" in v.reason and "y" in v.reason


# --------------------------------------------- abstention is information


def test_an_abstention_is_not_a_vote():
    """Counting an "I cannot tell" as a vote for whatever it happened to say
    destroys the one signal worth having."""
    v = tally([_v("a", "yes"), _v("b", "yes"), _v("c", None)])
    assert v.abstentions == 1
    assert v.tally == {"yes": 2}


def test_everyone_abstaining_reports_the_question_as_the_problem():
    v = tally([_v("a", None), _v("b", None)])
    assert v.outcome is Outcome.ABSTAINED
    assert "underspecified" in v.reason


def test_more_abstentions_than_answers_overrides_the_minority():
    """Whatever the two who answered decided, the honest report is that the
    question did not admit an answer."""
    votes = [_v("a", "yes"), _v("b", "yes"), _v("c", None), _v("d", None), _v("e", None)]
    v = tally(votes)
    assert v.outcome is Outcome.ABSTAINED
    assert v.abstentions / len(votes) > ABSTENTION_LIMIT


def test_a_few_abstentions_do_not_block_a_real_majority():
    v = tally([_v("a", "yes"), _v("b", "yes"), _v("c", "yes"), _v("d", None)])
    assert v.outcome is Outcome.DECIDED and v.choice == "yes"


def test_no_votes_at_all_is_its_own_outcome():
    assert tally([]).outcome is Outcome.EMPTY


# --------------------------------------------------- family diversity


def test_agreement_within_one_family_is_flagged():
    """Three members from one vendor agreeing is not three independent
    confirmations -- an evaluator favours its own generations."""
    v = tally([_v("a", "yes", "anthropic"), _v("b", "yes", "anthropic"),
               _v("c", "yes", "anthropic")])
    assert v.outcome is Outcome.DECIDED
    assert v.single_family
    assert "2404.13076" in v.render()


def test_cross_family_agreement_is_not_flagged():
    v = tally([_v("a", "yes", "anthropic"), _v("b", "yes", "deepseek"),
               _v("c", "yes", "openai")])
    assert not v.single_family
    assert "2404.13076" not in v.render()


def test_a_single_answering_member_is_not_called_single_family():
    """One voice is not a consensus of any kind, flagged or otherwise."""
    assert not tally([_v("a", "yes", "anthropic")]).single_family


# ------------------------------------------------------------ convening


def test_convene_collects_and_tallies():
    def ask(member, question):
        return Vote(member=member, choice="yes" if member != "c" else "no")

    v = convene(ask, ["a", "b", "c"], question="ship it?")
    assert v.outcome is Outcome.DECIDED and v.choice == "yes"


def test_a_member_that_raises_becomes_an_abstention_not_a_disappearance():
    """A council of five where two crashed is not a council of three. The tally
    needs the difference, because it changes whether the majority was real."""
    def ask(member, question):
        if member == "c":
            raise RuntimeError("provider down")
        return Vote(member=member, choice="yes")

    v = convene(ask, ["a", "b", "c"], question="q")
    assert len(v.votes) == 3
    assert v.abstentions == 1
    assert any("failed" in x.reason for x in v.votes)


def test_a_member_returning_nothing_is_an_abstention():
    v = convene(lambda m, q: None, ["a", "b"], question="q")
    assert v.outcome is Outcome.ABSTAINED


def test_this_module_cannot_spend_money():
    """Execution is injected. Keeping it out means the counting rules are
    testable without a provider, and a council cannot quietly acquire the
    ability to bill."""
    import inspect

    from forgeos.core import council

    src = inspect.getsource(council)
    for forbidden in ("Gateway", "record_spend", "httpx", "requests"):
        assert forbidden not in src


def test_tallying_is_deterministic():
    """A decision that varies between runs cannot be replayed or argued with."""
    votes = [_v("a", "x"), _v("b", "x"), _v("c", "y")]
    first = tally(votes)
    for _ in range(5):
        again = tally(votes)
        assert (again.outcome, again.choice, again.tally) == \
               (first.outcome, first.choice, first.tally)


def test_a_verdict_renders_its_reasoning():
    assert "decided" in tally([_v("a", "x"), _v("b", "x")]).render()
    assert isinstance(Verdict(Outcome.EMPTY, reason="none").render(), str)
