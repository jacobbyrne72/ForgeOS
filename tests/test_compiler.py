"""Tests for forgeos.compiler — natural language -> decomposed TaskSpecs.

No dedicated test file existed for this module before this one (the similarly
named `forgeos.core.mission.compile_mission`, covered by test_mission.py, is a
different, unrelated function that happens to share a name). Scope here is
narrow: the `manager.parallel_safety` pre-flight wired into `compile_mission`
just before it hands back a `Mission` (compiler.py's `_sequence_unsafe_pairs`),
plus enough of the surrounding decomposition behavior to prove the wiring
does not disturb it.

`compile_mission`'s only two branches that ever produce more than one task
(bugfix, refactor) always construct their pair pre-sequenced (`diagnose ->
fix`), so the public function alone cannot exercise "two genuinely
independent tasks get a NEW dependency added" -- there is no combination of
inputs that produces two unrelated tasks today. `_sequence_unsafe_pairs` is
tested directly for that behavior; the tests through `compile_mission` itself
confirm the existing pre-sequenced pairs are left alone, not double-processed
or given a redundant/duplicate edge.
"""

from __future__ import annotations

from forgeos.compiler import Mission, _sequence_unsafe_pairs, compile_mission
from forgeos.contracts import Scope, TaskSpec


def _spec(subject: str, description: str, *, paths=("src/default.py",),
          depends_on=()) -> TaskSpec:
    return TaskSpec(job_id="j", subject=subject, description=description,
                    scope=Scope(paths=list(paths)), depends_on=list(depends_on))


# ------------------------------------------- _sequence_unsafe_pairs, direct


def test_no_violation_leaves_depends_on_and_deps_untouched():
    a = _spec("fix retry", "Fix the retry backoff bug in the scheduler.", paths=["src/a.py"])
    b = _spec("rename helper", "Rename the logging helper for clarity.", paths=["src/b.py"])
    deps: dict[str, list[str]] = {}

    _sequence_unsafe_pairs([a, b], deps)

    assert a.depends_on == []
    assert b.depends_on == []
    assert deps == {}


def test_contract_violation_adds_depends_on_with_introducer_first():
    a = _spec("build gateway",
              "Create the `PaymentGateway` class that exposes charge and refund.",
              paths=["src/gateway.py"])
    b = _spec("wire checkout",
              "Wire the checkout flow to call PaymentGateway.charge() on submit.",
              paths=["src/checkout.py"])
    deps: dict[str, list[str]] = {}

    _sequence_unsafe_pairs([a, b], deps)

    assert b.depends_on == [a.id]
    assert a.depends_on == []
    assert deps[b.id] == [a.id]


def test_pure_scope_overlap_falls_back_to_a_deterministic_id_tiebreak():
    """No contract violation here -- only a shared path -- so there is no
    natural introducer. The pair must still be sequenced, and the same way
    every time, not by whichever happened to be first in the list."""
    a = _spec("a", "Fix a bug.", paths=["src/shared.py"])
    b = _spec("b", "Fix another bug.", paths=["src/shared.py"])
    deps: dict[str, list[str]] = {}

    _sequence_unsafe_pairs([a, b], deps)

    first_id, second_id = sorted((a.id, b.id))
    dependent = a if second_id == a.id else b
    introducer = b if dependent is a else a
    assert dependent.depends_on == [first_id]
    assert introducer.depends_on == []
    assert deps[second_id] == [first_id]


def test_a_pair_already_sequenced_forward_is_left_alone():
    a = _spec("diagnose", "Investigate the bug.", paths=["src/shared.py"])
    b = _spec("fix", "Apply the fix.", paths=["src/shared.py"], depends_on=[a.id])
    deps = {b.id: [a.id]}

    _sequence_unsafe_pairs([a, b], deps)

    assert b.depends_on == [a.id]  # unchanged -- no duplicate entry
    assert a.depends_on == []
    assert deps[b.id] == [a.id]


def test_a_pair_already_sequenced_in_reverse_is_also_left_alone():
    """The skip check looks both directions -- an existing edge either way
    means the pair is already sequenced, regardless of which task the caller
    happened to pass as `a`."""
    a = _spec("a", "Do the second half.", paths=["src/shared.py"])
    b = _spec("b", "Do the first half.", paths=["src/shared.py"])
    a.depends_on.append(b.id)
    deps: dict[str, list[str]] = {a.id: [b.id]}

    _sequence_unsafe_pairs([a, b], deps)

    assert a.depends_on == [b.id]
    assert b.depends_on == []
    assert deps == {a.id: [b.id]}


def test_calling_it_twice_does_not_duplicate_the_edge():
    a = _spec("build gateway", "Create the `PaymentGateway` class.", paths=["src/gateway.py"])
    b = _spec("wire checkout", "Call PaymentGateway.charge() from checkout.",
              paths=["src/checkout.py"])
    deps: dict[str, list[str]] = {}

    _sequence_unsafe_pairs([a, b], deps)
    _sequence_unsafe_pairs([a, b], deps)

    assert b.depends_on == [a.id]
    assert deps[b.id] == [a.id]


def test_three_tasks_only_the_conflicting_pair_gets_an_edge():
    a = _spec("build gateway", "Create the `PaymentGateway` class.", paths=["src/gateway.py"])
    b = _spec("wire checkout", "Call PaymentGateway.charge() from checkout.",
              paths=["src/checkout.py"])
    c = _spec("update docs", "Document the release notes.", paths=["docs/release.md"])
    deps: dict[str, list[str]] = {}

    _sequence_unsafe_pairs([a, b, c], deps)

    assert b.depends_on == [a.id]
    assert a.depends_on == []
    assert c.depends_on == []
    assert deps == {b.id: [a.id]}


# --------------------------------------------- compile_mission, integration


def test_bugfix_objective_pair_stays_pre_sequenced_with_no_extra_edges(tmp_path):
    """The diagnose/fix pair is already `depends_on`-linked at construction --
    the parallel-safety post-flight must recognize that and add nothing on
    top of it, even though both tasks share the same inferred scope."""
    mission = compile_mission("fix the login timeout bug", cwd=str(tmp_path), max_tasks=6)

    assert len(mission.tasks) == 2
    t1, t2 = mission.tasks
    assert t2.depends_on == [t1.id]
    assert t1.depends_on == []
    assert mission.dependencies == {t2.id: [t1.id]}


def test_refactor_objective_pair_stays_pre_sequenced_with_no_extra_edges(tmp_path):
    mission = compile_mission("refactor the scheduler module", cwd=str(tmp_path), max_tasks=6)

    assert len(mission.tasks) == 2
    t1, t2 = mission.tasks
    assert t2.depends_on == [t1.id]
    assert t1.depends_on == []


def test_single_task_objective_has_no_dependencies(tmp_path):
    mission = compile_mission("write documentation for the api", cwd=str(tmp_path))

    assert len(mission.tasks) == 1
    assert mission.tasks[0].depends_on == []
    assert mission.dependencies == {}


def test_compile_mission_returns_a_mission_with_a_task_map(tmp_path):
    mission = compile_mission("write documentation for the api", cwd=str(tmp_path))

    assert isinstance(mission, Mission)
    assert mission.task_map == {t.id: t.subject for t in mission.tasks}
