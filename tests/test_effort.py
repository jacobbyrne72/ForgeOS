"""Difficulty -> reasoning effort, the second routing knob.

The measurement behind this module: on the same six ForgeBench questions with
the same model, `reasoning_effort="medium"` produced 694 output tokens and 3/6
acceptance; `"none"` produced 233 and 5/6. Three of the six had returned an
EMPTY string after spending their whole output cap on chain-of-thought, which
the suite scored as the model being wrong.

So the tests that matter here are not "does the enum map" -- they are:
lookups must buy no reasoning, hard things must not be under-reasoned, and the
classifier must never call a model.
"""

from __future__ import annotations

import pytest

from forgeos.core.effort import (
    Difficulty,
    at_least,
    classify,
    effort_for,
    route_effort,
)


# ------------------------------------------------------- the measured case


def test_a_lookup_buys_no_reasoning():
    """The exact shape that cost 66% of output tokens for nothing."""
    difficulty, effort = route_effort(
        "In this codebase, which function records spend to the ledger, and what "
        "prevents the same model call being charged twice? Name the function and "
        "the guard. Answer in under 80 words."
    )
    assert difficulty is Difficulty.LOOKUP
    assert effort == "none"


@pytest.mark.parametrize("objective", [
    "Name the exact exception class raised when a call is refused",
    "Which property prices a cached input token?",
    "List the two functions that convert dollars to microdollars",
    "Where is the merge gate implemented?",
])
def test_question_shaped_objectives_are_lookups(objective):
    assert classify(objective) is Difficulty.LOOKUP


# ------------------------------------------- not under-reasoning hard work


@pytest.mark.parametrize("objective", [
    "Why does the lease occasionally grant two workers the same path?",
    "Debug the intermittent test failure in the scheduler",
    "Find the root cause of the double-charge",
    "Design the retry policy for provider failover",
    "Investigate the deadlock under concurrent writes",
])
def test_open_ended_work_gets_real_reasoning(objective):
    """Under-reasoning these buys a confident wrong answer, which is worse than
    an expensive right one because it does not look like a failure."""
    difficulty, effort = route_effort(objective)
    assert difficulty is Difficulty.DEEP
    assert effort == "high"


def test_a_deep_verb_beats_a_lookup_verb_in_the_same_sentence():
    """"Find the root cause" contains "find". Precedence is by cost of being
    wrong, not by which pattern matched first."""
    assert classify("Find the root cause of the flaky merge gate") is Difficulty.DEEP


def test_a_deep_verb_wins_even_in_a_single_file_scope():
    assert classify("Why is this slow?", scope_paths=1) is Difficulty.DEEP


def test_review_work_is_never_cheap():
    """A reviewer that does not reason is a rubber stamp, and the merge gate
    depends on this one being real."""
    difficulty, effort = route_effort(
        "Check the diff", capabilities={"review"}
    )
    assert difficulty is Difficulty.COMPLEX
    assert effort != "none"


def test_security_work_is_never_cheap():
    assert classify("Look at the input handling", capabilities={"security"}) is Difficulty.COMPLEX


# ------------------------------------------------------------ scope signal


def test_a_wide_scope_is_complex_even_for_a_simple_verb():
    """Past a handful of files, 'which files' is itself part of the problem."""
    assert classify("Add a docstring", scope_paths=12) is Difficulty.COMPLEX


def test_a_wide_scope_stops_a_lookup_from_staying_cheap():
    assert classify("Which function does this?", scope_paths=30) is Difficulty.COMPLEX


def test_narrow_scope_keeps_mechanical_work_cheap():
    difficulty, effort = route_effort("Rename the helper to _weaker", scope_paths=1)
    assert difficulty is Difficulty.MECHANICAL
    assert effort == "none"


# ------------------------------------------------------------- ordinary work


@pytest.mark.parametrize("objective", [
    "Add provider failover to the gateway",
    "Implement the budget cap",
    "Write a test for the budget cap",
])
def test_ordinary_build_work_is_standard(objective):
    difficulty, effort = route_effort(objective)
    assert difficulty is Difficulty.STANDARD
    assert effort == "low"


def test_writing_a_test_for_a_race_is_not_ordinary_work():
    """Caught by this suite's own first run, and the classifier was right:
    "write a test for the lease race" reads as STANDARD by its verb, but
    reproducing a race is exactly the work that needs reasoning. The concurrency
    signal outranking the build verb is the intended behaviour."""
    assert classify("Write a test for the lease race") is Difficulty.DEEP


def test_an_unrecognised_objective_does_not_fall_to_the_cheapest_setting():
    """An unknown is not a lookup. Defaulting to "none" would silently strip
    reasoning from every objective whose phrasing this module has not seen."""
    difficulty = classify("Zorble the frobnicator")
    assert difficulty is Difficulty.STANDARD
    assert effort_for(difficulty) != "none"


def test_an_empty_objective_is_handled():
    assert classify("") is Difficulty.STANDARD


# ---------------------------------------------------------------- mechanics


def test_at_least_takes_the_harder_of_two():
    assert at_least(Difficulty.LOOKUP, Difficulty.DEEP) is Difficulty.DEEP
    assert at_least(Difficulty.DEEP, Difficulty.LOOKUP) is Difficulty.DEEP
    assert at_least(Difficulty.STANDARD, Difficulty.STANDARD) is Difficulty.STANDARD


def test_every_difficulty_maps_to_an_effort():
    """A new difficulty with no mapping would raise at routing time, in
    production, on whichever task first hit it."""
    for difficulty in Difficulty:
        assert effort_for(difficulty) in {"none", "low", "medium", "high", "max"}


def test_effort_is_monotonic_in_difficulty():
    """Harder must never buy strictly less reasoning than easier."""
    rank = {"none": 0, "low": 1, "medium": 2, "high": 3, "max": 4}
    order = [Difficulty.LOOKUP, Difficulty.MECHANICAL, Difficulty.STANDARD,
             Difficulty.COMPLEX, Difficulty.DEEP]
    efforts = [rank[effort_for(d)] for d in order]
    assert efforts == sorted(efforts)


def test_classification_is_deterministic():
    """Same input, same answer, every time -- routing that varies between calls
    makes a cost regression impossible to attribute."""
    args = ("Add failover to the gateway", 3, frozenset({"edit"}))
    first = classify(args[0], scope_paths=args[1], capabilities=args[2])
    for _ in range(5):
        assert classify(args[0], scope_paths=args[1], capabilities=args[2]) is first


def test_the_classifier_makes_no_network_or_model_call(monkeypatch):
    """The rule this module exists under: never pay a model to decide what to
    pay a model. Any socket use here would mean the cost lands on every task
    before any work happens."""
    import socket

    def _boom(*a, **k):  # pragma: no cover - only runs if the rule is broken
        raise AssertionError("classify() attempted network I/O")

    monkeypatch.setattr(socket, "socket", _boom)
    monkeypatch.setattr(socket, "create_connection", _boom)
    assert classify("Why does this deadlock?") is Difficulty.DEEP


# ------------------------------------------------------- wiring, not dead code


def test_the_gateway_request_actually_carries_the_routed_effort():
    """A classifier nothing calls saves nothing. This asserts the value reaches
    the request object, which is the only place it can affect a bill."""
    import asyncio

    from forgeos.adapters.gateway_worker import _Session

    sess = _Session(task_id="t", cwd=".", model_ref="p/m", reasoning_effort="none")
    assert sess.reasoning_effort == "none"
    del asyncio


def test_an_adapter_on_the_old_signature_is_not_broken_by_the_new_argument():
    """`WorkerAdapter` is a Protocol, so third-party and test adapters may still
    take three arguments. Passing a fourth unconditionally raises inside the
    stream and surfaces as 'the worker produced nothing' -- a silent, badly
    mislabelled failure."""
    from forgeos.adapters.executor import _accepts_effort

    class Old:
        async def start(self, task_id, cwd, model_profile):  # noqa: D102
            return "s"

    class New:
        async def start(self, task_id, cwd, model_profile, reasoning_effort=""):  # noqa: D102
            return "s"

    assert _accepts_effort(Old) is False
    assert _accepts_effort(New) is True


def test_effort_classification_never_fails_a_task():
    """An effort setting is an optimisation. Failing a real job because its
    difficulty could not be guessed trades the work for a routing detail."""
    from forgeos.adapters.executor import _effort_for

    class Broken:
        @property
        def subject(self):
            raise RuntimeError("boom")

    assert _effort_for(Broken()) == ""
