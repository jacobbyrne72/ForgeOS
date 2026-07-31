"""Compression that returns whole symbols, and says so honestly.

This module's docstring used to claim "AST-aware" while its own next sentence
said "Uses string matching". It also claimed 60-90% reduction with nothing
measuring it. Both are fixed; these tests stop either coming back.
"""

from __future__ import annotations

from forgeos.context_compress import (
    Method,
    compress,
    compress_context,
    objective_terms,
)

SOURCE = '''\
import os


def unrelated_helper(x):
    """Nothing to do with the question."""
    return x + 1


@property
def record_spend(self, job, usd):
    """Write one call's cost to the ledger."""
    total = usd
    return total


class Ledger:
    """Storage."""

    def bump_generation(self, task_id):
        return task_id
'''


def test_a_returned_symbol_is_syntactically_complete():
    """THE point. A line window returns a `def` without its body; a model then
    guesses or asks for the file again, and the second costs more than sending
    the function whole."""
    result = compress("which function records spend", [("l.py", SOURCE)])
    body = result.files[0][1]
    assert "def record_spend" in body
    assert "return total" in body, "the body was cut off from its signature"
    compile(body, "<kept>", "exec")


def test_decorators_travel_with_their_function():
    """A method delivered without its @property reads as a plain method, and
    the model reasons about something that does not exist."""
    body = compress("record spend", [("l.py", SOURCE)]).files[0][1]
    assert "@property" in body


def test_irrelevant_symbols_are_dropped():
    body = compress("record spend", [("l.py", SOURCE)]).files[0][1]
    assert "unrelated_helper" not in body


def test_the_method_actually_used_is_reported():
    """Silently degrading to fragments while claiming symbol extraction is how
    the original docstring came to be wrong."""
    assert compress("record spend", [("l.py", SOURCE)]).method is Method.SYMBOLS


def test_unparseable_source_falls_back_and_says_so():
    broken = "def oops(:\n    this is not python\n  record spend here\n"
    result = compress("record spend", [("b.py", broken)])
    assert result.method is Method.LINE_WINDOW
    assert any("would not parse" in n for n in result.notes)


def test_a_non_python_language_falls_back_and_says_so():
    js = "function recordSpend() { return 1; }\nfunction other() {}\n"
    result = compress("record spend", [("a.js", js)], language="javascript")
    assert result.method is Method.LINE_WINDOW


def test_nothing_matching_returns_the_input_untouched():
    """Sending nothing is not compression, it is losing the task's context."""
    result = compress("quantum chromodynamics", [("l.py", SOURCE)])
    assert result.method is Method.UNCHANGED
    assert result.files == [("l.py", SOURCE)]


def test_an_empty_objective_returns_the_input_untouched():
    result = compress("", [("l.py", SOURCE)])
    assert result.method is Method.UNCHANGED


def test_a_matched_method_is_not_sent_twice_inside_its_class():
    src = "class A:\n    def spend(self):\n        return 1\n"
    body = compress("spend", [("a.py", src)]).files[0][1]
    assert body.count("def spend") == 1


def test_reduction_is_measured_on_this_input_not_asserted():
    result = compress("record spend", [("l.py", SOURCE)])
    assert result.chars_after < result.chars_before
    assert 0 < result.reduction_pct < 100
    assert f"{result.chars_before:,}" in result.render()


def test_reduction_of_an_empty_input_does_not_divide_by_zero():
    assert compress("x", []).reduction_pct == 0.0


def test_the_old_entry_point_still_works_for_existing_callers():
    out = compress_context("record spend", [("l.py", SOURCE)])
    assert isinstance(out, list) and isinstance(out[0], tuple)


def test_objective_terms_drop_scaffolding_and_split_identifiers():
    terms = objective_terms("Add a fix to record_spend in the ledger")
    assert "record_spend" in terms and "spend" in terms and "ledger" in terms
    for noise in ("add", "the", "fix", "to"):
        assert noise not in terms


def test_a_name_match_outranks_a_body_mention():
    src = ('def target_symbol():\n    return 1\n\n\n'
           'def other():\n    # target_symbol is mentioned here\n    return 2\n')
    body = compress("target_symbol", [("a.py", src)], min_score=5).files[0][1]
    assert "def target_symbol" in body and "def other" not in body
