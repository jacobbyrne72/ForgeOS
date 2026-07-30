"""Compressed retry context: a failed attempt's compact history fed into the
next attempt's prompt (`forgeos/forge.py` `_attempt_summary`/`_cap_attempt_history`,
`forgeos/adapters/executor.py` `_build_prompt`).

Two guarantees matter more than the feature itself:

1. A first attempt renders BYTE-IDENTICAL to a task with no retry history at
   all -- the retry block is appended strictly in the tail, never touching the
   prefix, because that prefix stability is exactly what `Ledger.cache_health`
   (see test_cache_health.py) measures. Get this wrong and Feature 1 reports a
   permanent, self-inflicted "regression".
2. Compression drops whole low-value entries; it never rewords or trims the
   exact error text / test node ids inside a kept one. arXiv:2607.12161
   measured that stripping exactly this kind of anchor text raised billed
   cost and lowered task success, because the next attempt could no longer
   match its fix against the failure.

No LLM calls, no network -- executors and reviewers are fakes, exactly like
tests/test_forge.py.
"""

from __future__ import annotations

import pytest

from forgeos.adapters.executor import _build_prompt
from forgeos.contracts import (
    AttemptSummary,
    Budget,
    FailureClass,
    Scope,
    TaskSpec,
    TaskState,
    TestResults,
)
from forgeos.economy.reducer import reduce_pytest
from forgeos.forge import (
    ATTEMPT_HISTORY_MAX_TOKENS,
    ExecutionResult,
    Forge,
    _attempt_summary,
    _cap_attempt_history,
)
from forgeos.registry import Adapter, CostTier, Registry, WorkerProfile


def _fleet() -> Registry:
    # Two workers, not one: the merge gate requires an independent reviewer
    # (a genuinely different worker_id), and a one-worker fleet can never
    # supply one -- see forgeos.forge.Forge._pick_reviewer.
    return Registry([
        WorkerProfile(worker_id="free.local", adapter=Adapter.OLLAMA, tier=CostTier.FREE,
                      capabilities={"edit", "python", "mechanical"}, can_edit_files=True,
                      prior_win_rate=0.85),
        WorkerProfile(worker_id="premium.cloud", adapter=Adapter.CLI_TEAM,
                      tier=CostTier.PREMIUM,
                      capabilities={"edit", "python", "mechanical", "architecture"},
                      can_edit_files=True, prior_win_rate=0.95),
    ])


def _task(subject="fix the flaky retry test") -> TaskSpec:
    return TaskSpec(job_id="pending", subject=subject, description="make it pass reliably",
                    capabilities=["edit", "python"], scope=Scope(paths=["src/thing.py"]),
                    acceptance=["targeted tests pass"], budget=Budget(max_usd=2.0))


@pytest.fixture()
def forge(tmp_path):
    f = Forge(home=tmp_path / "forgeos", registry=_fleet(), max_attempts=3)
    yield f
    f.close()


def _green(**kw) -> ExecutionResult:
    base = dict(
        state=TaskState.DONE, evidence="fixed", commands_run=["pytest"],
        files_touched=["src/thing.py"], tests=TestResults(passed=1, failed=0),
        usd_micros=1000,
    )
    base.update(kw)
    return ExecutionResult(**base)


def _pass_review(spec, worker):  # noqa: ARG001
    return ExecutionResult(state=TaskState.DONE)


# ============================================================ cache-safety:
# byte-identical first attempt, tail-only append


def test_build_prompt_with_empty_history_is_identical_to_no_history_field():
    base = _task()
    with_explicit_empty = base.model_copy(update={"attempt_history": []})
    assert _build_prompt(base) == _build_prompt(with_explicit_empty)


def test_first_attempt_prompt_is_byte_identical_to_a_task_with_no_retry_history(forge):
    """The literal requirement: attempt 1 must cost the same cache-wise as a
    task built with no retry machinery at all. Compared against an
    INDEPENDENT baseline that never goes through Forge."""
    captured_specs: list[TaskSpec] = []

    def flaky(spec, worker):  # noqa: ARG001
        captured_specs.append(spec)
        if len(captured_specs) == 1:
            return ExecutionResult(state=TaskState.FAILED, blocker="boom",
                                   failure=FailureClass.MODEL, usd_micros=1000)
        return _green()

    control_prompt = _build_prompt(_task())  # independent of the run below

    result = forge.run("ship", [_task()], executor=flaky, reviewer=_pass_review)

    assert len(captured_specs) == 2, "setup should fail once then succeed"
    assert captured_specs[0].attempt_history == []
    assert _build_prompt(captured_specs[0]) == control_prompt
    assert result.accepted == 1


def test_retry_prompt_is_the_first_attempts_bytes_plus_an_appended_tail(forge):
    exact_error = "AssertionError: assert result.accepted == 2"
    captured_specs: list[TaskSpec] = []

    def flaky(spec, worker):  # noqa: ARG001
        captured_specs.append(spec)
        if len(captured_specs) == 1:
            return ExecutionResult(state=TaskState.FAILED, blocker=exact_error,
                                   failure=FailureClass.MODEL, usd_micros=1000)
        return _green()

    forge.run("ship", [_task()], executor=flaky, reviewer=_pass_review)

    assert len(captured_specs) == 2
    prompt1 = _build_prompt(captured_specs[0])
    prompt2 = _build_prompt(captured_specs[1])

    assert prompt2.startswith(prompt1), (
        "a retry must APPEND to the prior attempt's bytes, never edit or "
        "reorder them, or the provider's prefix cache silently stops hitting"
    )
    assert prompt2 != prompt1
    appended = prompt2[len(prompt1):]
    assert exact_error in appended, "the failure reason belongs in the tail"
    assert "attempt 1" in prompt2


def test_reviewer_sees_the_canonical_spec_not_a_retry_history_copy(forge):
    """The reviewer is never retried -- it reviews whatever the implementer's
    CURRENT attempt produced. Confirms the retry-history plumbing only
    touches the executor call, not the (unrelated) reviewer call."""
    reviewer_specs: list[TaskSpec] = []
    captured_specs: list[TaskSpec] = []

    def flaky(spec, worker):  # noqa: ARG001
        captured_specs.append(spec)
        if len(captured_specs) == 1:
            return ExecutionResult(state=TaskState.FAILED, blocker="boom",
                                   failure=FailureClass.MODEL, usd_micros=1000)
        return _green()

    def reviewer(spec, worker):  # noqa: ARG001
        reviewer_specs.append(spec)
        return ExecutionResult(state=TaskState.DONE)

    forge.run("ship", [_task()], executor=flaky, reviewer=reviewer)

    assert len(reviewer_specs) == 1
    assert reviewer_specs[0].attempt_history == []


# ============================================================ accumulation
# across every attempt, newest-first


def test_history_accumulates_across_every_attempt_newest_first(forge):
    captured_specs: list[TaskSpec] = []

    def always_fails(spec, worker):  # noqa: ARG001
        captured_specs.append(spec)
        return ExecutionResult(state=TaskState.FAILED, blocker=f"boom {len(captured_specs)}",
                               failure=FailureClass.MODEL, usd_micros=1000)

    result = forge.run("ship", [_task()], executor=always_fails, reviewer=_pass_review)

    assert len(captured_specs) == 3  # forge fixture's max_attempts=3
    assert captured_specs[0].attempt_history == []
    assert [h.attempt for h in captured_specs[1].attempt_history] == [1]
    assert [h.attempt for h in captured_specs[2].attempt_history] == [2, 1]  # newest first
    assert "boom 1" in captured_specs[1].attempt_history[0].reason
    assert "boom 2" in captured_specs[2].attempt_history[0].reason
    assert "boom 1" in captured_specs[2].attempt_history[1].reason
    assert "exhausted" in result.outcomes[0].reason


# ============================================================ verbatim
# preservation: the coordinator's required test


_EXACT_ASSERTION = "AssertionError: assert result.accepted == 2"
_EXACT_NODEID = "tests/test_forge_retry.py::test_two_tasks_both_complete"
_EXACT_FAILED_LINE = f"FAILED {_EXACT_NODEID} - {_EXACT_ASSERTION}"

_PYTEST_LOG = (
    "============================= test session starts ==============================\n"
    "collected 2 items\n"
    "\n"
    "=========================== short test summary info ============================\n"
    f"{_EXACT_FAILED_LINE}\n"
    "================= 1 failed, 1 passed in 0.42s ==================\n"
)


def test_exact_error_text_and_test_node_id_survive_verbatim_into_the_retry_prompt():
    """Load-bearing: compression drops whole items, it never reworks or
    trims the anchor text inside one it keeps (arXiv:2607.12161). A summary
    that paraphrases "assertion failed in the retry test" instead of this
    exact line is strictly worse than sending nothing."""
    reduced = reduce_pytest(_PYTEST_LOG)
    assert _EXACT_FAILED_LINE in reduced.kept_lines

    result = ExecutionResult(state=TaskState.FAILED, failure=FailureClass.MODEL,
                             raw_output=_PYTEST_LOG, usd_micros=1000)
    summary = _attempt_summary(1, result, reduced.summary, reduced.kept_lines)

    assert _EXACT_ASSERTION in summary.reason
    assert _EXACT_NODEID in summary.reason
    assert _EXACT_FAILED_LINE in summary.reason

    task = _task().model_copy(update={"attempt_history": [summary]})
    prompt = _build_prompt(task)

    assert _EXACT_FAILED_LINE in prompt, "byte-for-byte, not a paraphrase"


def test_attempt_summary_preserves_blocker_and_evidence_verbatim():
    blocker = "ModuleNotFoundError: No module named 'nonexistent_thing_xyz'"
    evidence = "3 passed, 2 failed in 1.10s"
    result = ExecutionResult(state=TaskState.FAILED, failure=FailureClass.MODEL,
                             blocker=blocker, usd_micros=1000,
                             tests=TestResults(passed=3, failed=2))

    summary = _attempt_summary(1, result, evidence, [])

    assert blocker in summary.reason
    assert evidence in summary.reason
    assert "tests: 3p/2f" in summary.reason
    assert summary.failure_class == "model"
    assert summary.attempt == 1


# ============================================================ hard cap:
# drop whole entries, never truncate one


def test_cap_keeps_everything_under_a_generous_budget():
    history = [
        AttemptSummary(attempt=2, failure_class="model", reason="short b"),
        AttemptSummary(attempt=1, failure_class="model", reason="short a"),
    ]
    capped = _cap_attempt_history(history, max_tokens=ATTEMPT_HISTORY_MAX_TOKENS)
    assert capped == history


def test_cap_drops_whole_oldest_entries_and_never_truncates_a_kept_one():
    long_reason = "X" * 2000  # forces the cap to actually bind
    history = [
        AttemptSummary(attempt=3, failure_class="model", reason="newest " + long_reason),
        AttemptSummary(attempt=2, failure_class="model", reason="middle " + long_reason),
        AttemptSummary(attempt=1, failure_class="model", reason="oldest " + long_reason),
    ]

    capped = _cap_attempt_history(history, max_tokens=50)

    assert 1 <= len(capped) < len(history), "the tiny budget must drop at least one whole entry"
    assert capped[0].attempt == 3, "newest is always kept first"
    by_attempt = {h.attempt: h for h in history}
    for c in capped:
        # Whatever is kept is BYTE-IDENTICAL to its source -- there is no
        # partially-cut variant anywhere in this pipeline to produce one.
        assert c.reason == by_attempt[c.attempt].reason
        assert c.reason.endswith(long_reason)


def test_cap_always_keeps_the_single_newest_entry_even_if_it_alone_is_oversized():
    """A truncated anchor is worse than one large but intact entry -- if
    forced to choose, keep the anchor and spend the tokens."""
    huge = AttemptSummary(attempt=1, failure_class="model", reason="Y" * 5000)
    capped = _cap_attempt_history([huge], max_tokens=1)
    assert capped == [huge]
