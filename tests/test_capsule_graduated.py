"""Graduated capsule inclusion: trim to whole lines instead of dropping.

`add()` is a cliff — an item one token over budget is lost entirely, however
relevant it was. This repo's own A/B benchmark showed the cost: of 31 ranked
blocks, 3 were sent and 28 were dropped at the boundary, several of them
high-scoring blocks that merely arrived after the budget was spent.

`fit()` keeps the head of a good block instead of losing all of it, subject to
two hard rules that are the whole reason this is safe:

1. **Whole lines only.** arXiv 2607.12161 measured trimming that severed
   verbatim anchors: billed cost rose 6.8% and success fell from 27/40 to
   15/40, because the model could no longer match its fix against the failure.
   A half-identifier is worse than an absent one — it looks usable and is not.
2. **The ref describes what was actually sent.** Hash and token count are built
   from the admitted text, so the capsule's accounting cannot overstate what it
   carried.
"""

from __future__ import annotations

from forgeos.economy.capsule import CapsuleBuilder, RefKind, make_ref
from forgeos.economy.preflight import count_tokens


def _lines(n: int, width: int = 60) -> str:
    """n distinct, identifier-shaped lines — so a mid-line cut would be visible."""
    return "\n".join(f"def function_number_{i:03d}():  # {'x' * width}" for i in range(n))


# ------------------------------------------------------------- whole item


def test_an_item_that_fits_is_admitted_whole():
    b = CapsuleBuilder(budget=10_000)
    body = _lines(5)
    got = b.fit(RefKind.FILE_SLICE, "file_slice://a.py#L1-5", body, "relevant")
    assert got == body
    assert len(b.items) == 1
    assert b.partial == {}


def test_a_whole_item_is_not_recorded_as_partial():
    b = CapsuleBuilder(budget=10_000)
    b.fit(RefKind.FILE_SLICE, "file_slice://a.py#L1-5", _lines(5), "relevant")
    assert b.excluded == {}
    assert b.partial == {}


# ----------------------------------------------------------------- trimming


def test_an_oversized_item_is_trimmed_rather_than_dropped():
    """The behaviour the benchmark asked for: keep the head, not nothing."""
    big = _lines(80)
    budget = count_tokens(big, "").tokens // 4
    b = CapsuleBuilder(budget=budget)
    got = b.fit(RefKind.FILE_SLICE, "file_slice://big.py#L1-80", big, "ranked first")

    assert got is not None, "a high-ranked block was dropped at the cliff"
    assert len(got) < len(big)
    assert len(b.items) == 1
    assert "file_slice://big.py#L1-80" in b.partial


def test_trimming_cuts_only_at_line_boundaries():
    """The rule that keeps verbatim anchors intact. A truncated identifier is
    worse than an omitted one."""
    big = _lines(80)
    b = CapsuleBuilder(budget=count_tokens(big, "").tokens // 3)
    got = b.fit(RefKind.FILE_SLICE, "file_slice://big.py#L1-80", big, "r")

    original = big.splitlines()
    for line in got.splitlines():
        assert line in original, f"line was cut mid-way: {line!r}"


def test_the_trimmed_prefix_is_the_head_in_order():
    big = _lines(60)
    b = CapsuleBuilder(budget=count_tokens(big, "").tokens // 3)
    got = b.fit(RefKind.FILE_SLICE, "file_slice://big.py#L1-60", big, "r")
    assert big.startswith(got), "trim must keep the head, not an arbitrary slice"


def test_the_admitted_ref_is_priced_on_what_was_sent_not_the_original():
    """Pricing the original while sending a trimmed copy would make the
    capsule's own token accounting a fiction."""
    big = _lines(80)
    full_tokens = count_tokens(big, "").tokens
    b = CapsuleBuilder(budget=full_tokens // 3)
    got = b.fit(RefKind.FILE_SLICE, "file_slice://big.py#L1-80", big, "r")

    assert b.items[0].token_estimate == count_tokens(got, "").tokens
    assert b.items[0].token_estimate < full_tokens
    # And the hash must be of the admitted text, not the original.
    assert b.items[0].sha256 == make_ref(RefKind.FILE_SLICE, "x", got, "y").sha256


def test_the_budget_is_never_exceeded_by_a_trimmed_item():
    big = _lines(200)
    budget = count_tokens(big, "").tokens // 5
    b = CapsuleBuilder(budget=budget)
    b.fit(RefKind.FILE_SLICE, "file_slice://big.py#L1-200", big, "r")
    assert b.total_tokens <= budget


def test_the_partial_note_records_how_much_was_lost():
    """Silent trimming would let a reader believe the capsule carried the whole
    thing."""
    big = _lines(80)
    b = CapsuleBuilder(budget=count_tokens(big, "").tokens // 4)
    b.fit(RefKind.FILE_SLICE, "file_slice://big.py#L1-80", big, "r")
    note = b.partial["file_slice://big.py#L1-80"]
    assert "of 80 lines" in note


def test_the_reason_on_the_ref_says_it_is_partial():
    big = _lines(80)
    b = CapsuleBuilder(budget=count_tokens(big, "").tokens // 4)
    b.fit(RefKind.FILE_SLICE, "file_slice://big.py#L1-80", big, "ranked first")
    assert "partial" in b.items[0].reason


# ------------------------------------------------------------------ floors


def test_a_stub_too_small_to_be_useful_is_dropped_not_sent():
    """Spending the last of the budget on four lines of a 200-line block buys
    nothing; dropping it and saying so is the honest outcome."""
    big = _lines(200)
    b = CapsuleBuilder(budget=10)          # room for roughly nothing
    got = b.fit(RefKind.FILE_SLICE, "file_slice://big.py#L1-200", big, "r")
    assert got is None
    assert "file_slice://big.py#L1-200" in b.excluded
    assert b.items == []


def test_a_short_item_that_does_not_fit_is_dropped_rather_than_trimmed():
    b = CapsuleBuilder(budget=5)
    got = b.fit(RefKind.FILE_SLICE, "file_slice://s.py#L1-2", _lines(2), "r")
    assert got is None
    assert "too few to trim" in b.excluded["file_slice://s.py#L1-2"]


# ------------------------------------------------------- graduated in practice


def test_a_late_high_value_block_survives_where_add_would_have_lost_it():
    """The benchmark case, end to end: fill most of the budget, then offer one
    more good block. `add` drops it whole; `fit` keeps its head."""
    filler = _lines(20)
    late = _lines(40)
    # Budget = the filler plus room for a useful slice of the late block, but
    # nowhere near all of it. Derived rather than guessed: a hardcoded number
    # silently becomes either "everything fits" or "nothing does" the moment the
    # tokeniser or the fixture changes, and both make this test vacuous.
    room = count_tokens(_lines(10), "").tokens
    budget = count_tokens(filler, "").tokens + room
    b_add = CapsuleBuilder(budget=budget)
    b_fit = CapsuleBuilder(budget=budget)

    b_add.add(make_ref(RefKind.FILE_SLICE, "file_slice://f.py#L1-20", filler, "filler"))
    b_fit.fit(RefKind.FILE_SLICE, "file_slice://f.py#L1-20", filler, "filler")

    dropped = not b_add.add(
        make_ref(RefKind.FILE_SLICE, "file_slice://late.py#L1-40", late, "late but good")
    )
    kept = b_fit.fit(RefKind.FILE_SLICE, "file_slice://late.py#L1-40", late, "late but good")

    assert dropped, "precondition: add() should hit the cliff here"
    assert kept, "fit() lost the block add() also lost — no improvement"
    assert b_fit.total_tokens <= budget
