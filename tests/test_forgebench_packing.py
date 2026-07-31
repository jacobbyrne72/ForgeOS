"""The packer's own preconditions, checked without spending a penny.

Every one of these caught a real defect that had shipped:

- The ForgeOS arm was handed a context that could not answer its own question
  (`ledger-dedup-guard`: the chunk defining `record_spend` ranked #53 of 302
  while five docstrings *about* double-charging took every slot). The suite
  scored that as ForgeOS being worse than the baseline. It was a packing bug.
- The packed capsule came out LARGER than simply sending the file
  (`preflight-refusal-types`, 108% of baseline). A wrapper that costs more than
  no wrapper has no reason to exist, and no test said so.
- The budget was computed from overlapping chunk bodies, so every task silently
  received about twice its intended allowance.

These run offline against the real repo. A model is never consulted -- the
question is only ever "could ANY model answer from this context", which is a
property of the packer alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from forgeos import forgebench
from forgeos.forgebench import (
    CAPSULE_BUDGET_CAP_TOKENS,
    DEFAULT_SUITE,
    build_baseline_prompt,
    build_forgeos_prompt,
    capsule_budget_for,
    objective_terms,
    rank_chunk,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def repo_root() -> Path:
    """The real checkout. These tests read the actual source the suite pins --
    a synthetic fixture repo would pass while the shipped packer stayed broken.
    """
    return REPO_ROOT

# What each task's acceptance check demands, in the same any-of-per-group shape
# `keyword_groups` uses. Kept beside the suite rather than introspected out of
# the closures: a test that derived these from the code under test could not
# catch the code under test changing them.
ACCEPTANCE_GROUPS: dict[str, list[tuple[str, ...]]] = {
    "ledger-dedup-guard": [
        ("record_spend",),
        ("inflight", "in-flight", "in flight", "dedup", "coalesce", "coalesced"),
    ],
    "preflight-refusal-types": [("CallRefused",), ("Decision",)],
    "catalog-cache-pricing": [("cache_read_rate",), ("input_cost_per_1m",)],
    "ledger-generation-fencing": [("bump_generation",), ("record_report_if_current",)],
    "contracts-micros-json": [("to_micros",), ("from_micros",)],
    "savings-provenance-rule": [("_weaker",), ("weaker",)],
}


def _ids(tasks):
    return [t.id for t in tasks]


def test_every_pinned_task_has_declared_acceptance_groups():
    """A task added to the suite without declaring what its answer needs would
    silently skip every check in this file."""
    assert sorted(ACCEPTANCE_GROUPS) == sorted(_ids(DEFAULT_SUITE))


@pytest.mark.parametrize("task", DEFAULT_SUITE, ids=_ids(DEFAULT_SUITE))
def test_forgeos_context_can_support_the_acceptance_check(task, repo_root):
    """THE regression. If no member of a required group appears anywhere in the
    packed prompt, the task is unanswerable in that arm and the benchmark is
    measuring the packer's blind spot, not the model's."""
    packed, _stats = build_forgeos_prompt(repo_root, task)
    low = packed.lower()
    missing = [g for g in ACCEPTANCE_GROUPS[task.id]
               if not any(term.lower() in low for term in g)]
    assert not missing, (
        f"{task.id}: the packed context contains no member of {missing}, so no "
        f"model could satisfy the acceptance check from it. The ForgeOS arm "
        f"would fail a task the baseline arm passes, and the suite would report "
        f"that as a quality difference rather than the packing bug it is."
    )


@pytest.mark.parametrize("task", DEFAULT_SUITE, ids=_ids(DEFAULT_SUITE))
def test_baseline_context_can_support_the_acceptance_check(task, repo_root):
    """The baseline sends whole files, so a miss here means the acceptance check
    demands something that is not in the task's declared scope at all -- a
    broken task, not a broken packer."""
    prompt = build_baseline_prompt(repo_root, task).lower()
    missing = [g for g in ACCEPTANCE_GROUPS[task.id]
               if not any(term.lower() in prompt for term in g)]
    assert not missing, f"{task.id}: {missing} is absent from the task's own scope files"


@pytest.mark.parametrize("task", DEFAULT_SUITE, ids=_ids(DEFAULT_SUITE))
def test_the_packed_arm_never_costs_more_than_the_naive_arm(task, repo_root):
    """The floor under the entire premise. ForgeOS may be worse at some things,
    but sending MORE context than 'just paste the files in' means the wrapper is
    strictly negative value on that task."""
    packed, _ = build_forgeos_prompt(repo_root, task)
    baseline = build_baseline_prompt(repo_root, task)
    assert len(packed) < len(baseline), (
        f"{task.id}: packed {len(packed)} chars vs baseline {len(baseline)} -- "
        f"the capsule costs more than sending the raw files"
    )


@pytest.mark.parametrize("task", DEFAULT_SUITE, ids=_ids(DEFAULT_SUITE))
def test_the_capsule_respects_its_budget(task, repo_root):
    """A budget that does not bind is not a budget. This is the check that would
    have caught the scope being measured off overlapping chunks."""
    _packed, stats = build_forgeos_prompt(repo_root, task)
    scope_chars = sum(
        (repo_root / p).read_text(encoding="utf-8", errors="replace").__len__()
        for p in task.scope.paths
        if (repo_root / p).exists()
    )
    budget = capsule_budget_for(scope_chars)
    assert stats["capsule_tokens"] <= budget, (
        f"{task.id}: capsule holds {stats['capsule_tokens']} tokens against a "
        f"{budget}-token budget"
    )


@pytest.mark.parametrize("task", DEFAULT_SUITE, ids=_ids(DEFAULT_SUITE))
def test_packing_is_deterministic(task, repo_root):
    """Two runs over an unchanged repo must produce byte-identical context. A
    benchmark whose own input varies between runs cannot attribute a cost
    difference to anything it changed."""
    first, s1 = build_forgeos_prompt(repo_root, task)
    second, s2 = build_forgeos_prompt(repo_root, task)
    assert first == second and s1 == s2


# ------------------------------------------------------------------ budget rule


def test_budget_scales_with_scope():
    small = capsule_budget_for(6_000)
    large = capsule_budget_for(600_000)
    assert large > small, "a five-file question gets the same allowance as a one-file question"


def test_budget_never_exceeds_the_cap():
    assert capsule_budget_for(100_000_000) <= CAPSULE_BUDGET_CAP_TOKENS


def test_budget_never_exceeds_the_scope_it_is_drawn_from():
    """The rule that makes 'costs more than baseline' structurally impossible
    rather than merely unlikely."""
    for chars in (200, 1_000, 6_000, 40_000, 500_000):
        assert capsule_budget_for(chars) <= max(chars // 4, 1)


def test_budget_is_always_positive():
    for chars in (0, 1, 5):
        assert capsule_budget_for(chars) >= 1


# ------------------------------------------------------------ ranking property


def _synthetic_task(tmp_path: Path):
    """A scope where prose ABOUT the answer is far bulkier than the answer.

    This is the shape that broke the real suite: docstrings discussing a concept
    repeat the question's vocabulary many times over, while the definition that
    actually answers it is two lines. Built synthetically so the test pins the
    ranking RULE rather than an accident of this repo's current source.
    """
    from forgeos.contracts import Scope
    from forgeos.forgebench import AcceptanceCheck, PinnedTask, keyword_groups

    prose = tmp_path / "notes.py"
    prose.write_text(
        '"""\n'
        + ("Discussion of how the ledger records spend, and what prevents the "
           "same call being charged twice. Spend recording matters. Charged "
           "twice is bad. The ledger records spend.\n" * 40)
        + '"""\n',
        encoding="utf-8",
    )
    impl = tmp_path / "impl.py"
    impl.write_text(
        "import os\n\n\n"
        + "def unrelated_helper(x):\n    return x + 1\n\n\n" * 6
        + "def record_spend(job, usd):\n"
        "    return job\n",
        encoding="utf-8",
    )
    task = PinnedTask(
        id="synthetic",
        objective=(
            "Which function records spend to the ledger, and what prevents the "
            "same call being charged twice?"
        ),
        scope=Scope(paths=["notes.py", "impl.py"]),
        acceptance=AcceptanceCheck(
            description="names record_spend",
            predicate=keyword_groups(("record_spend",)),
        ),
    )
    return task


def test_a_definition_outranks_bulk_prose_that_merely_discusses_it(tmp_path):
    """The defect this encodes: forty lines of docstring repeating "records
    spend" buried `def record_spend` at rank #53 of 302, and the packed context
    could not answer its own question. Symbol names are the title field; prose
    is the body. The title must win.
    """
    task = _synthetic_task(tmp_path)
    packed, _stats = build_forgeos_prompt(tmp_path, task)
    assert "record_spend" in packed, (
        "the function that answers the question lost to prose about the question"
    )


def test_a_defined_symbol_outweighs_the_same_word_in_prose():
    """The ranking rule itself, stated without a fixture.

    Both chunks mention "spend" the same number of times. One DEFINES it. Code
    retrieval that cannot tell those apart is what let five docstrings about
    double-charging outrank the function that does the charging.
    """
    weights = {"spend": 1.0, "record": 1.0}
    defines = "def record_spend(job):\n    return job\n"
    mentions = "# we record spend here somewhere, see the spend record docs\n"
    assert rank_chunk(defines, weights) > rank_chunk(mentions, weights)


def test_the_weighting_is_live_not_dead_code():
    """If neutralising `_DEF_FIELD_WEIGHT` changes nothing, the constant is
    decorative and every claim made for it is false."""
    weights = {"spend": 1.0}
    body = "def record_spend(job):\n    return job\n"
    with_weight = rank_chunk(body, weights)
    original = forgebench._DEF_FIELD_WEIGHT
    try:
        forgebench._DEF_FIELD_WEIGHT = 0.0
        without = rank_chunk(body, weights)
    finally:
        forgebench._DEF_FIELD_WEIGHT = original
    assert with_weight > without


def test_a_chunk_defining_nothing_relevant_gets_no_boost():
    """The boost must key on the DEFINED NAME, not merely on a chunk happening
    to contain a `def` line."""
    weights = {"spend": 1.0}
    unrelated_def = "def helper(x):\n    # spend\n    return x\n"
    plain = "# spend\n"
    assert rank_chunk(unrelated_def, weights) == rank_chunk(plain, weights)


# --------------------------------------------------------------- term extraction


def test_objective_terms_drop_question_scaffolding():
    """'name the function' words appear in every objective, so they separate
    nothing and would only dilute the IDF weighting."""
    terms = objective_terms("Name the function that records spend to the ledger.")
    assert "record" in terms and "ledger" in terms
    for noise in ("name", "the", "function", "that"):
        assert noise not in terms


def test_objective_terms_split_snake_case_identifiers():
    terms = objective_terms("What does raise_if_refused do?")
    assert "raise_if_refused" in terms
    assert "refused" in terms, "a question naming a symbol should also match its parts"


def test_objective_terms_are_not_the_answer_key():
    """The ranking terms must come from the QUESTION. If they came from the
    acceptance check, the retriever would be tuned on the test set and would
    measure nothing about a repo it had not seen."""
    task = next(t for t in DEFAULT_SUITE if t.id == "ledger-dedup-guard")
    terms = objective_terms(task.objective)
    assert "record_spend" not in terms
    assert "inflight" not in terms


# ------------------------------------------------------------------ exit gate


def _totals(accepted: int, usd_micros: int):
    from forgeos.forgebench import ArmTotals

    return ArmTotals(attempted_count=6, accepted_count=accepted, tokens_in=0,
                     tokens_out=0, tokens_cached_in=0, usd_micros=usd_micros,
                     seconds=0.0)


def test_gate_fails_when_forgeos_gets_less_right_however_cheap():
    """Non-inferiority. The gate must never be purchasable with quality."""
    assert forgebench._exit_gate(_totals(2, 1), _totals(4, 10_000)) is False


def test_gate_fails_when_nothing_was_accepted_by_either_arm():
    """No correct work happened, so there is nothing to be cheaper per unit of.
    Spending almost nothing to achieve nothing is not a pass."""
    assert forgebench._exit_gate(_totals(0, 1), _totals(0, 10_000)) is False


def test_gate_passes_on_equal_acceptance_at_lower_cost():
    """The original case, unchanged: equal denominators cancel, so this is the
    same comparison the gate always made."""
    assert forgebench._exit_gate(_totals(4, 2_000), _totals(4, 7_000)) is True
    assert forgebench._exit_gate(_totals(4, 7_000), _totals(4, 2_000)) is False


def test_gate_passes_when_forgeos_is_both_better_and_cheaper():
    """The measured result that exposed the bug: 4/6 vs 2/6 at a third of the
    cost was being reported as FAIL because acceptance differed at all."""
    assert forgebench._exit_gate(_totals(4, 2_179), _totals(2, 7_121)) is True


def test_gate_fails_when_extra_acceptance_cost_more_per_task():
    """Accepting more does not excuse spending more per accepted task -- that
    is the whole reason the metric is per-task rather than a raw count."""
    assert forgebench._exit_gate(_totals(3, 90_000), _totals(2, 10_000)) is False


def test_gate_passes_when_the_baseline_got_nothing_right():
    """No ratio exists against zero correct work; any correct work wins."""
    assert forgebench._exit_gate(_totals(1, 5_000), _totals(0, 100)) is True
