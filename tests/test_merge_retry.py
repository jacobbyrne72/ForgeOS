"""Which merge-gate refusals are worth paying to retry.

The decision is conditional on purpose. Both blanket answers are wrong, and each
is wrong in a way that costs real money:

- **Always retry** triples the bill for a guaranteed second refusal. Observed
  live: a text-only gateway worker refused for "no tests passed; no commands
  run" cannot ever satisfy those, because an HTTP completion has no filesystem
  and runs no commands. Three attempts, three identical refusals, three charges.
- **Never retry** wastes the case the retry context exists for. The gate's
  reasons are carried into the next prompt, so "you did not run the tests" is
  actionable feedback — refusing to act on it throws away a cheap fix.

So: retry when the worker could plausibly clear the complaint, stop when the
complaint is a property of the machine or the fleet rather than of the attempt.
"""

from __future__ import annotations

import pytest

from forgeos.forge import _refusal_is_fixable

# Verbatim from MergeGate.evaluate, so a reworded reason string breaks this test
# rather than silently changing which refusals cost money to retry.
NO_TESTS = "no tests passed — nothing was actually verified"
NO_EVIDENCE = "no evidence recorded"
NO_COMMAND = "no command was run to produce the evidence"
TESTS_FAILING = "3 test(s) failing"
SECURITY_FAILED = "security failed (2 finding(s))"

NO_SCANNER = "no security gate was run — absence is not a pass"
UNAVAILABLE = "security could not be checked — not treated as clean"
NO_REVIEW = "no independent review"
NO_REVIEWER_ID = "reviewer identity missing — independence unverifiable"
SELF_REVIEW = "reviewer must not be the implementer"
DERIVED = "reviewer id 'reviewer::a' is derived from implementer 'a' — that is the same worker relabelled"


# ------------------------------------------------------------------ fixable


@pytest.mark.parametrize("reason", [NO_TESTS, NO_EVIDENCE, NO_COMMAND,
                                    TESTS_FAILING, SECURITY_FAILED])
def test_a_refusal_the_worker_could_clear_is_retried(reason):
    assert _refusal_is_fixable([reason]) is True


def test_several_fixable_reasons_together_are_still_fixable():
    assert _refusal_is_fixable([NO_TESTS, NO_EVIDENCE, NO_COMMAND]) is True


# --------------------------------------------------------------- structural


@pytest.mark.parametrize("reason", [NO_SCANNER, UNAVAILABLE, NO_REVIEW,
                                    NO_REVIEWER_ID, SELF_REVIEW, DERIVED])
def test_a_refusal_no_retry_can_clear_is_not_retried(reason):
    """A missing scanner and a one-worker fleet do not become fixable by
    spending more money on the same task."""
    assert _refusal_is_fixable([reason]) is False


def test_the_live_text_only_worker_case_is_not_retried():
    """The exact refusal seen from tools/live_job.py. An HTTP completion has no
    filesystem, so it can never run a test or a command — three attempts would
    have produced three identical refusals at triple the price."""
    assert _refusal_is_fixable([NO_TESTS, NO_COMMAND, NO_SCANNER]) is False


# ------------------------------------------------------- the expensive edge


def test_one_structural_reason_poisons_an_otherwise_fixable_refusal():
    """The load-bearing case. A task refused for BOTH missing tests and missing
    review would still be refused after the worker adds tests, because nothing
    the worker does conjures a second reviewer. Retrying is certain waste, so
    any structural reason wins over every fixable one.
    """
    assert _refusal_is_fixable([NO_TESTS, NO_REVIEW]) is False
    assert _refusal_is_fixable([TESTS_FAILING, SELF_REVIEW]) is False


def test_no_reasons_is_not_fixable():
    """An allowed verdict has no reasons; there is nothing to retry."""
    assert _refusal_is_fixable([]) is False


def test_an_unrecognised_reason_is_not_retried():
    """Unknown means unknown. Defaulting to retry would spend real money on a
    refusal nobody has shown a worker can clear."""
    assert _refusal_is_fixable(["something nobody has classified yet"]) is False


def test_matching_is_case_insensitive():
    assert _refusal_is_fixable(["NO EVIDENCE RECORDED"]) is True
    assert _refusal_is_fixable(["NO INDEPENDENT REVIEW"]) is False
