"""Tests for the tool-output reducer and the incremental test-selection ladder.

No LLM calls, no network, no subprocess. Everything here is deterministic
parsing of hand-built (but structurally realistic) pytest output.
"""

from __future__ import annotations

import pytest

from hive.economy.avoidance import AvoidanceLog, AvoidanceMethod
from hive.economy.reducer import Failure, ReducedOutput, reduce_generic, reduce_pytest
from hive.economy.testselect import Level, Selection, TestGraph, next_level

# ------------------------------------------------------------- pytest fixture


def _build_realistic_pytest_log(n_dup: int = 400) -> tuple[str, list[str]]:
    """A realistic multi-failure pytest log: N identical failures from one bug
    in a loop, plus one genuinely distinct failure that crosses into a library
    frame, plus some passing tests. Returns (log text, every FAILED node id).
    """
    lines: list[str] = []
    lines.append("============================= test session starts ==============================")
    lines.append("platform win32 -- Python 3.11.9, pytest-9.0.3, pluggy-1.4.0")
    lines.append("rootdir: C:/Users/byrne/Downloads/hive")
    lines.append(f"collected {n_dup + 6} items")
    lines.append("")

    for i in range(5):
        lines.append(f"tests/test_ok.py::test_ok_{i} PASSED                                     [ 10%]")
    for i in range(n_dup):
        lines.append(f"tests/test_loop.py::test_dup_{i} FAILED                                   [ 50%]")
    lines.append("tests/test_other.py::test_unique_bug FAILED                                    [ 99%]")
    lines.append("")

    lines.append("=================================== FAILURES ===================================")
    for i in range(n_dup):
        lines.append(f"______________________________ test_dup_{i} ______________________________________")
        lines.append("")
        lines.append(f"    def test_dup_{i}():")
        lines.append('>       assert helper() == "expected"')
        lines.append("E       AssertionError: assert 'actual' == 'expected'")
        lines.append("")
        lines.append("tests/test_loop.py:12: AssertionError")
    lines.append("______________________________ test_unique_bug ______________________________________")
    lines.append("")
    lines.append("    def test_unique_bug():")
    lines.append(">       parse_config(bad_input)")
    lines.append("")
    lines.append("tests/test_other.py:20: in test_unique_bug")
    lines.append("    parse_config(bad_input)")
    lines.append(".venv/lib/python3.11/site-packages/somelib/parser.py:88: in parse_config")
    lines.append('    raise ValueError("bad config")')
    lines.append("E       ValueError: bad config")
    lines.append("")

    lines.append("=========================== short test summary info ============================")
    failed_ids: list[str] = []
    for i in range(n_dup):
        nid = f"tests/test_loop.py::test_dup_{i}"
        failed_ids.append(nid)
        lines.append(f"FAILED {nid} - AssertionError: assert 'actual' == 'expected'")
    unique_id = "tests/test_other.py::test_unique_bug"
    failed_ids.append(unique_id)
    lines.append(f"FAILED {unique_id} - ValueError: bad config")
    lines.append(f"================= {n_dup + 1} failed, 5 passed in 12.34s ==================")

    return "\n".join(lines), failed_ids


_ALL_PASSED_LOG = """\
============================= test session starts ==============================
platform linux -- Python 3.11.9, pytest-9.0.3
collected 3 items

tests/test_a.py::test_1 PASSED                                          [ 33%]
tests/test_a.py::test_2 PASSED                                          [ 66%]
tests/test_a.py::test_3 PASSED                                          [100%]

============================== 3 passed in 0.05s ===============================
"""


# ---------------------------------------------------------------- reduce_pytest


def test_reduces_by_more_than_80_percent_on_a_realistic_log():
    output, _ = _build_realistic_pytest_log(n_dup=400)
    reduced = reduce_pytest(output)
    assert reduced.saved_pct > 80.0
    assert reduced.reduced_tokens_estimate < reduced.original_tokens_estimate


def test_never_drops_a_failure_every_failed_id_survives():
    output, failed_ids = _build_realistic_pytest_log(n_dup=400)
    reduced = reduce_pytest(output)
    joined = "\n".join(reduced.kept_lines)
    missing = [fid for fid in failed_ids if fid not in joined]
    assert missing == []
    assert reduced.failed == len(failed_ids)


def test_400_duplicate_errors_collapse_to_one_plus_a_count():
    output, failed_ids = _build_realistic_pytest_log(n_dup=400)
    reduced = reduce_pytest(output)
    assert len(reduced.failures) == 2  # one duplicated root cause + one unique
    dup_group = next(f for f in reduced.failures if f.count > 1)
    assert dup_group.count == 400
    assert len(dup_group.also) == 399
    unique_group = next(f for f in reduced.failures if f.count == 1)
    assert "ValueError" in unique_group.error


def test_project_frame_preferred_over_library_frame():
    output, _ = _build_realistic_pytest_log(n_dup=1)
    reduced = reduce_pytest(output)
    unique = next(f for f in reduced.failures if "ValueError" in f.error)
    assert "tests/test_other.py" in unique.frame
    assert "site-packages" not in unique.frame


def test_all_passed_log_has_no_failures():
    reduced = reduce_pytest(_ALL_PASSED_LOG)
    assert reduced.exit_code == 0
    assert reduced.passed == 3
    assert reduced.failed == 0
    assert reduced.failures == []


def test_empty_pytest_output_does_not_crash():
    reduced = reduce_pytest("")
    assert reduced.original_lines == 0
    assert reduced.failures == []
    assert reduced.summary  # some honest label, not a crash


def test_malformed_pytest_output_does_not_crash():
    reduced = reduce_pytest("this is not pytest output\njust some noise\n")
    assert reduced.failures == []
    assert reduced.exit_code == 5  # unknown state — never claim success on a guess


def test_reduced_output_is_a_pydantic_model_with_expected_shape():
    reduced = reduce_pytest(_ALL_PASSED_LOG)
    assert isinstance(reduced, ReducedOutput)
    assert isinstance(reduced.artifact_ref, str) and len(reduced.artifact_ref) == 64  # sha256 hex
    assert reduced.saved_tokens == max(0, reduced.original_tokens_estimate - reduced.reduced_tokens_estimate)


def test_failure_model_holds_test_frame_error():
    f = Failure(test="tests/x.py::test_a", frame="tests/x.py:1", error="boom")
    assert f.test == "tests/x.py::test_a"
    assert f.frame == "tests/x.py:1"
    assert f.error == "boom"
    assert f.count == 1
    assert f.also == []


# --------------------------------------------------------------- reduce_generic


def test_reduce_generic_keeps_head_tail_and_marks_truncation():
    lines = [f"line {i}" for i in range(200)]
    output = "\n".join(lines)
    reduced = reduce_generic(output, max_lines=40)
    joined = "\n".join(reduced.kept_lines)
    assert "line 0" in joined
    assert "line 199" in joined
    assert any("omitted" in ln or "withheld" in ln for ln in reduced.kept_lines)
    assert reduced.original_lines == 200


def test_reduce_generic_dedups_repeated_error_lines():
    lines = [f"build step {i}" for i in range(60)]
    for _ in range(50):
        lines.insert(30, "ERROR: disk full")
    output = "\n".join(lines)
    reduced = reduce_generic(output, max_lines=40)
    joined = "\n".join(reduced.kept_lines)
    assert joined.count("ERROR: disk full") == 1
    assert any("omitted" in ln or "withheld" in ln for ln in reduced.kept_lines)


def test_reduce_generic_short_output_is_not_truncated():
    output = "\n".join(f"line {i}" for i in range(10))
    reduced = reduce_generic(output, max_lines=40)
    assert reduced.kept_lines == [f"line {i}" for i in range(10)]
    assert not any("omitted" in ln or "withheld" in ln for ln in reduced.kept_lines)


def test_reduce_generic_empty_output_does_not_crash():
    reduced = reduce_generic("", max_lines=40)
    assert reduced.original_lines == 0
    assert reduced.kept_lines == []
    assert reduced.exit_code == 0


def test_reduce_generic_flags_error_lines_in_failed_count():
    output = "\n".join(["all good", "ERROR: boom", "still fine"])
    reduced = reduce_generic(output, max_lines=40)
    assert reduced.failed == 1
    assert reduced.exit_code == 1


# ------------------------------------------------------------------- record_to


@pytest.fixture()
def avd():
    log = AvoidanceLog(":memory:")
    yield log
    log.close()


def test_record_to_writes_deterministic_avoidance_row(avd):
    output, _ = _build_realistic_pytest_log(n_dup=5)
    reduced = reduce_pytest(output)
    rid = reduced.record_to(avd, "job1", task_id="task1")
    rows = avd.for_job("job1")
    assert [r.id for r in rows] == [rid]
    row = rows[0]
    assert row.method is AvoidanceMethod.DETERMINISTIC
    assert row.baseline_tokens == reduced.original_tokens_estimate
    assert row.actual_tokens == reduced.reduced_tokens_estimate
    assert row.saved_tokens == reduced.saved_tokens
    assert "sha256" in row.baseline_source
    assert row.task_id == "task1"


# ============================================================== test selection


def test_level_values_match_the_specified_order():
    assert [int(x) for x in (Level.SYNTAX, Level.LINT, Level.DIRECT, Level.PACKAGE, Level.INTEGRATION, Level.FULL)] == [
        0,
        1,
        2,
        3,
        4,
        5,
    ]


def test_next_level_climbs_one_rung_at_a_time_on_pass():
    assert next_level(Level.SYNTAX, True) == Level.LINT
    assert next_level(Level.LINT, True) == Level.DIRECT
    assert next_level(Level.DIRECT, True) == Level.PACKAGE
    assert next_level(Level.PACKAGE, True) == Level.INTEGRATION
    assert next_level(Level.INTEGRATION, True) == Level.FULL


def test_next_level_refuses_to_skip_a_rung():
    # Even if someone hopes to jump straight to FULL, only the immediate next rung is returned.
    for lvl in (Level.SYNTAX, Level.LINT, Level.DIRECT, Level.PACKAGE, Level.INTEGRATION):
        assert next_level(lvl, True) == Level(lvl + 1)


def test_next_level_at_full_after_pass_means_fully_verified():
    assert next_level(Level.FULL, True) is None


def test_next_level_does_not_climb_on_failure():
    assert next_level(Level.DIRECT, False) == Level.DIRECT
    assert next_level(Level.SYNTAX, False) == Level.SYNTAX
    assert next_level(Level.FULL, False) == Level.FULL


def test_level_for_lockfile_demands_full():
    g = TestGraph()
    assert g.level_for(["poetry.lock"]) == Level.FULL
    assert g.level_for(["uv.lock"]) == Level.FULL


def test_level_for_config_demands_integration():
    g = TestGraph()
    assert g.level_for(["pyproject.toml"]) == Level.INTEGRATION


def test_level_for_ordinary_source_edit_is_direct():
    g = TestGraph()
    assert g.level_for(["hive/economy/reducer.py"]) == Level.DIRECT


def test_level_for_no_changes_is_syntax():
    g = TestGraph()
    assert g.level_for([]) == Level.SYNTAX


def test_level_for_lockfile_outranks_a_single_function_edit():
    g = TestGraph()
    assert g.level_for(["hive/economy/reducer.py"]) < g.level_for(["uv.lock"])


def test_tests_for_maps_files_and_symbols():
    g = TestGraph()
    g.register_file("hive/economy/reducer.py", ["tests/test_reducer.py::test_a"])
    g.register_symbol("reduce_pytest", "tests/test_reducer.py::test_b")
    assert g.tests_for(["hive/economy/reducer.py"], ["reduce_pytest"]) == [
        "tests/test_reducer.py::test_a",
        "tests/test_reducer.py::test_b",
    ]


def test_tests_for_normalizes_path_separators():
    g = TestGraph()
    g.register_file("hive/economy/reducer.py", "t1")
    assert g.tests_for(["hive\\economy\\reducer.py"]) == ["t1"]


def test_select_direct_returns_mapped_tests():
    g = TestGraph()
    g.register_file("hive/economy/reducer.py", ["tests/test_reducer.py::test_a"])
    sel = g.select(["hive/economy/reducer.py"], level=Level.DIRECT)
    assert isinstance(sel, Selection)
    assert sel.level == Level.DIRECT
    assert sel.tests == ["tests/test_reducer.py::test_a"]
    assert not sel.escalated
    assert not sel.run_all


def test_unknown_file_escalates_to_package_using_sibling_tests():
    g = TestGraph()
    g.register_file("hive/economy/preflight.py", ["tests/test_economy.py::test_x"])
    sel = g.select(["hive/economy/reducer.py"], level=Level.DIRECT)  # same package, no direct mapping
    assert sel.escalated
    assert sel.level == Level.PACKAGE
    assert sel.tests == ["tests/test_economy.py::test_x"]
    assert not sel.run_all


def test_completely_unknown_change_escalates_and_runs_everything_known():
    g = TestGraph()
    sel = g.select(["some/brand_new/module.py"], level=Level.DIRECT)
    assert sel.escalated
    assert sel.level == Level.PACKAGE
    assert sel.run_all
    assert sel.tests == []


def test_a_change_with_no_known_tests_never_silently_selects_nothing():
    # AGENTS.md rule 8: "no tests selected" reported as green is a false green.
    g = TestGraph()
    sel = g.select(["totally/unmapped/file.py"], level=Level.DIRECT)
    assert sel.tests or sel.run_all
    assert sel.escalated


def test_no_change_at_all_does_not_falsely_escalate():
    g = TestGraph()
    sel = g.select([], level=Level.DIRECT)
    assert sel.tests == []
    assert not sel.escalated
    assert not sel.run_all


def test_integration_and_full_run_everything_by_definition():
    g = TestGraph()
    assert g.select(["x.py"], level=Level.INTEGRATION).run_all
    assert g.select(["x.py"], level=Level.FULL).run_all


def test_syntax_and_lint_are_not_pytest_selections():
    g = TestGraph()
    for lvl in (Level.SYNTAX, Level.LINT):
        sel = g.select(["x.py"], level=lvl)
        assert sel.tests == []
        assert not sel.run_all


def test_select_defaults_level_from_level_for_when_omitted():
    g = TestGraph()
    sel = g.select(["poetry.lock"])
    assert sel.level == Level.FULL
    assert sel.run_all
