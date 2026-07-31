"""Two-stage retrieval: retrieve broad, then rerank narrow.

The failure single-stage retrieval produces is specific -- it fills the budget
with chunks that look most like the question rather than the ones that answer
it. Measured on this repo: the chunk defining `record_spend` ranked #53 of 302
while module docstrings *discussing* double-charging took every top slot. Every
word of the question was in those docstrings; none of the answer was.
"""

from __future__ import annotations

from forgeos.economy.rerank import (
    DEFAULT_CANDIDATE_LIMIT,
    idf,
    precision_stage,
    recall_stage,
    rerank_report,
    two_stage_rank,
)

TERMS = ("record", "spend", "ledger", "guard", "twice")

PROSE = (
    # Hits EVERY question term, so it strictly outscores the definition on
    # stage-1 recall. The fixture has to reproduce the failure by score, not by
    # winning an alphabetical tiebreak -- an earlier version tied at 4.135 and
    # "passed" only because `ledger.py` sorts before `notes.py`, which would
    # have proved nothing about reranking at all.
    "notes.py:1-40",
    '"""Discussion of how the ledger records spend, and what guard stops a\n'
    "call being charged twice. The ledger records spend. Spend, ledger and\n"
    'guard, twice over. Recording spend in the ledger twice is avoidable."""\n',
)
DEFINITION = (
    "ledger.py:380-392",
    "def record_spend(self, job, usd, guard=None):\n"
    "    # the inflight guard stops the same call being charged twice\n"
    "    return job\n",
)
IRRELEVANT = ("util.py:1-10", "def helper(x):\n    return x + 1\n")


# --------------------------------------------------------------- the point


def test_a_definition_outranks_prose_that_merely_discusses_it():
    """THE regression. Reranking exists for exactly this."""
    ranked = two_stage_rank([PROSE, DEFINITION, IRRELEVANT], TERMS)
    assert ranked[0].ref == DEFINITION[0], (
        f"prose beat the definition: {[c.ref for c in ranked]}"
    )


def test_single_stage_recall_alone_gets_it_wrong():
    """Establishes that stage 2 is doing real work rather than agreeing with
    stage 1. If recall already ranked correctly, reranking would be dead
    weight worth deleting."""
    candidates, _weights = recall_stage([PROSE, DEFINITION, IRRELEVANT], TERMS)
    assert candidates[0].ref == PROSE[0], "fixture no longer reproduces the failure"


def test_reranking_records_the_movement():
    ranked = two_stage_rank([PROSE, DEFINITION, IRRELEVANT], TERMS)
    winner = next(c for c in ranked if c.ref == DEFINITION[0])
    assert winner.moved > 0
    assert winner.rank_before > winner.rank_after


# ------------------------------------------------------------ recall stage


def test_recall_keeps_anything_with_signal():
    """Being wrong here is unrecoverable: a chunk not retrieved cannot be
    reranked back in."""
    candidates, _ = recall_stage([PROSE, DEFINITION, IRRELEVANT], TERMS)
    refs = {c.ref for c in candidates}
    assert PROSE[0] in refs and DEFINITION[0] in refs


def test_recall_drops_chunks_with_no_signal_at_all():
    candidates, _ = recall_stage([PROSE, DEFINITION, IRRELEVANT], TERMS)
    assert IRRELEVANT[0] not in {c.ref for c in candidates}


def test_recall_is_bounded_by_the_candidate_limit():
    chunks = [(f"f.py:{i}", f"the ledger records spend number {i}") for i in range(200)]
    candidates, _ = recall_stage(chunks, TERMS, candidate_limit=25)
    assert len(candidates) == 25


def test_the_default_candidate_limit_is_sane():
    assert 10 <= DEFAULT_CANDIDATE_LIMIT <= 200


# --------------------------------------------------------- precision stage


def test_coverage_beats_repetition():
    """A chunk touching three different parts of the question answers more than
    one repeating a single part nine times."""
    repeated = ("a.py:1", "spend spend spend spend spend spend spend spend spend\n")
    covering = ("b.py:1", "the ledger guard for spend\n")
    ranked = two_stage_rank([repeated, covering], TERMS)
    assert ranked[0].ref == "b.py:1"


def test_a_long_diluted_chunk_does_not_win_on_bulk_alone():
    dense = ("small.py:1", "record spend ledger guard twice\n")
    diluted = ("big.py:1", "record spend ledger guard twice\n" + ("filler\n" * 200))
    ranked = two_stage_rank([dense, diluted], TERMS)
    assert ranked[0].ref == "small.py:1"


def test_a_def_line_matching_nothing_asked_gets_no_boost():
    """The boost must key on the DEFINED NAME, not on a chunk containing a
    `def` line at all."""
    unrelated_def = ("u.py:1", "def helper(x):\n    # spend\n    return x\n")
    plain = ("p.py:1", "# spend\n")
    ranked = two_stage_rank([unrelated_def, plain], ("spend",))
    scores = {c.ref: round(c.precision_score, 6) for c in ranked}
    assert scores["u.py:1"] <= scores["p.py:1"] * 1.5


# ----------------------------------------------------------------- idf


def test_a_term_in_every_chunk_carries_little_weight():
    bodies = ["the ledger a", "the ledger b", "the ledger c"]
    w = idf(("ledger", "rare"), bodies + ["rare thing"])
    assert w["rare"] > w["ledger"]


def test_a_term_absent_from_the_corpus_is_dropped():
    assert "absent" not in idf(("absent",), ["nothing here"])


def test_idf_handles_an_empty_corpus():
    assert idf(("x",), []) == {}


# --------------------------------------------------------------- mechanics


def test_ranking_is_deterministic():
    chunks = [PROSE, DEFINITION, IRRELEVANT]
    first = [c.ref for c in two_stage_rank(chunks, TERMS)]
    for _ in range(5):
        assert [c.ref for c in two_stage_rank(chunks, TERMS)] == first


def test_an_empty_corpus_returns_nothing_rather_than_raising():
    assert two_stage_rank([], TERMS) == []


def test_no_terms_returns_nothing():
    assert two_stage_rank([PROSE, DEFINITION], ()) == []


def test_precision_stage_on_an_empty_candidate_set():
    assert precision_stage([], TERMS, {}) == []


def test_the_report_shows_what_moved():
    ranked = two_stage_rank([PROSE, DEFINITION, IRRELEVANT], TERMS)
    text = rerank_report(ranked)
    assert "reranked" in text and DEFINITION[0] in text


def test_the_report_handles_no_candidates():
    assert rerank_report([]) == "no candidates"


def test_no_model_call_is_made(monkeypatch):
    """Both stages are plain code. Paying a model to choose context is the cost
    this module exists to avoid."""
    import socket

    def _boom(*a, **k):  # pragma: no cover
        raise AssertionError("two_stage_rank attempted network I/O")

    monkeypatch.setattr(socket, "socket", _boom)
    monkeypatch.setattr(socket, "create_connection", _boom)
    assert two_stage_rank([PROSE, DEFINITION], TERMS)
