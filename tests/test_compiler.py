"""Tests for forgeos.compiler — natural language -> decomposed TaskSpecs.

No dedicated test file existed for this module before this one (the similarly
named `forgeos.core.mission.compile_mission`, covered by test_mission.py, is a
different, unrelated function that happens to share a name). Scope here is
narrow: the `manager.parallel_safety` pre-flight wired into `compile_mission`
just before it hands back a `Mission` (compiler.py's `_sequence_unsafe_pairs`
and its `_reaches` reachability helper), plus enough of the surrounding
decomposition behavior to prove the wiring does not disturb it.

`compile_mission`'s only two branches that ever produce more than one task
(bugfix, refactor) always construct their pair pre-sequenced (`diagnose ->
fix`), so the public function alone cannot exercise "two genuinely independent
tasks get a NEW dependency added" -- there is no combination of inputs that
produces two unrelated tasks today. `_sequence_unsafe_pairs` and `_reaches`
are tested directly for that behavior; the tests through `compile_mission`
itself confirm the existing pre-sequenced pairs are left alone and that
repeated compiles of the same objective are structurally stable.
"""

from __future__ import annotations

from forgeos.compiler import Mission, _reaches, _sequence_unsafe_pairs, compile_mission
from forgeos.contracts import Scope, TaskSpec


def _spec(subject: str, description: str, *, paths=("src/default.py",),
          depends_on=()) -> TaskSpec:
    return TaskSpec(job_id="j", subject=subject, description=description,
                    scope=Scope(paths=list(paths)), depends_on=list(depends_on))


def _dependency_shape(mission: Mission) -> list[tuple[int, int]]:
    """(dependent_index, prerequisite_index) pairs by position in
    mission.tasks -- the only way to compare two separately-compiled missions
    structurally, since TaskSpec.id is a fresh uuid4 every compile."""
    index_of = {t.id: i for i, t in enumerate(mission.tasks)}
    pairs = [
        (index_of[t.id], index_of[dep_id])
        for t in mission.tasks
        for dep_id in t.depends_on
    ]
    return sorted(pairs)


# ------------------------------------------- _sequence_unsafe_pairs, direct


def test_no_violation_leaves_depends_on_and_deps_untouched():
    a = _spec("fix retry", "Fix the retry backoff bug in the scheduler.", paths=["src/a.py"])
    b = _spec("rename helper", "Rename the logging helper for clarity.", paths=["src/b.py"])
    deps: dict[str, list[str]] = {}

    conflicts = _sequence_unsafe_pairs([a, b], deps)

    assert a.depends_on == []
    assert b.depends_on == []
    assert deps == {}
    assert conflicts == []


def test_contract_violation_adds_depends_on_with_introducer_first():
    a = _spec("build gateway",
              "Create the `PaymentGateway` class that exposes charge and refund.",
              paths=["src/gateway.py"])
    b = _spec("wire checkout",
              "Wire the checkout flow to call PaymentGateway.charge() on submit.",
              paths=["src/checkout.py"])
    deps: dict[str, list[str]] = {}

    conflicts = _sequence_unsafe_pairs([a, b], deps)

    assert b.depends_on == [a.id]
    assert a.depends_on == []
    assert deps[b.id] == [a.id]
    assert conflicts == []


def test_pure_scope_overlap_falls_back_to_compiled_order_tiebreak():
    """No contract violation here -- only a shared path -- so there is no
    natural introducer. The earlier task in the COMPILED (list) order becomes
    the prerequisite, never a sort on `TaskSpec.id` -- ids are a fresh uuid4
    every compile, so an id-based tiebreak would not be stable across two
    compiles of the same mission."""
    a = _spec("a", "Fix a bug.", paths=["src/shared.py"])
    b = _spec("b", "Fix another bug.", paths=["src/shared.py"])
    deps: dict[str, list[str]] = {}

    conflicts = _sequence_unsafe_pairs([a, b], deps)

    assert b.depends_on == [a.id]
    assert a.depends_on == []
    assert deps[b.id] == [a.id]
    assert conflicts == []


def test_tiebreak_direction_follows_the_compiled_list_order_not_the_tasks_themselves():
    """Swapping which task is passed first flips the direction -- proving the
    tiebreak is about list position, not some inherent property of `a`/`b`."""
    a = _spec("a", "Fix a bug.", paths=["src/shared.py"])
    b = _spec("b", "Fix another bug.", paths=["src/shared.py"])
    deps: dict[str, list[str]] = {}

    _sequence_unsafe_pairs([b, a], deps)  # b listed first this time

    assert a.depends_on == [b.id]
    assert b.depends_on == []


def test_a_pair_already_sequenced_forward_is_left_alone():
    a = _spec("diagnose", "Investigate the bug.", paths=["src/shared.py"])
    b = _spec("fix", "Apply the fix.", paths=["src/shared.py"], depends_on=[a.id])
    deps = {b.id: [a.id]}

    conflicts = _sequence_unsafe_pairs([a, b], deps)

    assert b.depends_on == [a.id]  # unchanged -- no duplicate entry
    assert a.depends_on == []
    assert deps[b.id] == [a.id]
    assert conflicts == []


def test_a_pair_already_sequenced_in_reverse_is_also_left_alone():
    """The ordered check looks both directions -- an existing edge either way
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

    conflicts = _sequence_unsafe_pairs([a, b, c], deps)

    assert b.depends_on == [a.id]
    assert a.depends_on == []
    assert c.depends_on == []
    assert deps == {b.id: [a.id]}
    assert conflicts == []


def test_a_transitively_implied_ordering_added_within_the_same_pass_is_recognized():
    """x introduces `Core`, referenced by y -> edge y depends_on x. x also
    references `Base`, introduced by z -> edge x depends_on z. By the time
    the (y, z) pair is checked, y already transitively depends on z through
    x -- even though y and z's OWN scopes overlap, which parallel_safety
    would otherwise flag. A direct-membership check (only reading each
    depends_on list, never walking it) would miss this two-hop path and add
    a second, redundant edge; the live `_reaches` walk must not.
    """
    x = _spec("build core", "Create the `Core` class that calls Base.setup() during init.",
              paths=["src/core.py"])
    y = _spec("wire consumer", "Call Core.run() from the consumer.", paths=["src/shared.py"])
    z = _spec("provide base", "Define the `Base` class used everywhere.", paths=["src/shared.py"])
    deps: dict[str, list[str]] = {}

    conflicts = _sequence_unsafe_pairs([x, y, z], deps)

    assert y.depends_on == [x.id]
    assert x.depends_on == [z.id]
    assert z.depends_on == []  # nothing added between y and z -- already implied
    assert conflicts == []


# ------------------------------------------------------------- _reaches


def test_reaches_follows_a_multi_hop_chain():
    x = _spec("x", "Task x.", paths=["src/x.py"])
    y = _spec("y", "Task y.", paths=["src/y.py"])
    z = _spec("z", "Task z.", paths=["src/z.py"])
    x.depends_on = [y.id]
    y.depends_on = [z.id]
    by_id = {t.id: t for t in (x, y, z)}

    assert _reaches(by_id, x.id, z.id) is True    # x -> y -> z
    assert _reaches(by_id, z.id, x.id) is False   # no path the other way
    assert _reaches(by_id, x.id, y.id) is True    # direct hop too


def test_reaches_detects_a_would_be_cycle_directly():
    """Unit-level proof of the primitive `_sequence_unsafe_pairs`'s cycle
    guard relies on: if x already (transitively) depends on z, adding the
    reverse edge (z depends_on x) would close a loop, and `_reaches` must say
    so. `_sequence_unsafe_pairs`'s own live ordered-pair check (built on this
    same `_reaches`) means this situation cannot actually arise while going
    through that function's per-pair loop -- the ordered-pair check catches
    it first, as "already ordered", before a conflicting edge is ever
    attempted (see its docstring). This test verifies the guard's own logic
    in isolation rather than resting on that argument alone.
    """
    x = _spec("x", "Task x.", paths=["src/x.py"])
    y = _spec("y", "Task y.", paths=["src/y.py"])
    z = _spec("z", "Task z.", paths=["src/z.py"])
    x.depends_on = [y.id]
    y.depends_on = [z.id]
    by_id = {t.id: t for t in (x, y, z)}

    # x already depends on z -- adding "z depends_on x" would be this cycle.
    assert _reaches(by_id, x.id, z.id) is True


def test_a_pre_existing_cycle_in_the_input_does_not_crash_or_hang():
    """`_reaches`'s visited-set guards against a pathological pre-existing
    cycle in depends_on (never produced by compile_mission's own logic, but
    nothing stops a caller from handing this function anything). Must
    terminate, must not raise, and must not try to "fix" the cycle -- it just
    adds nothing new on top of it."""
    x = _spec("x", "Task x.", paths=["src/x.py"])
    y = _spec("y", "Task y.", paths=["src/y.py"])
    z = _spec("z", "Task z.", paths=["src/z.py"])
    x.depends_on = [y.id]
    y.depends_on = [z.id]
    z.depends_on = [x.id]  # a pre-existing 3-cycle, not created by this function
    deps: dict[str, list[str]] = {y.id: [x.id], z.id: [y.id], x.id: [z.id]}

    conflicts = _sequence_unsafe_pairs([x, y, z], deps)  # must not hang or raise

    assert isinstance(conflicts, list)
    assert x.depends_on == [y.id]
    assert y.depends_on == [z.id]
    assert z.depends_on == [x.id]


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
    assert mission.dependency_conflicts == []


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
    assert mission.dependency_conflicts == []


def test_compile_mission_returns_a_mission_with_a_task_map(tmp_path):
    mission = compile_mission("write documentation for the api", cwd=str(tmp_path))

    assert isinstance(mission, Mission)
    assert mission.task_map == {t.id: t.subject for t in mission.tasks}


def test_compiling_the_same_bugfix_objective_twice_yields_the_same_dependency_shape(tmp_path):
    """Same mission, same graph -- structurally, since `TaskSpec.id` is a
    fresh uuid4 every compile and cannot be compared directly."""
    m1 = compile_mission("fix the login timeout bug", cwd=str(tmp_path), max_tasks=6)
    m2 = compile_mission("fix the login timeout bug", cwd=str(tmp_path), max_tasks=6)

    assert _dependency_shape(m1) == _dependency_shape(m2) == [(1, 0)]


def test_sequence_unsafe_pairs_tiebreak_is_stable_across_fresh_random_ids():
    """`compile_mission` cannot currently produce two genuinely independent
    tasks (its only >1-task branches always pre-sequence their pair), so this
    exercises `_sequence_unsafe_pairs` directly, simulating what two separate
    `compile_mission` calls would feed it: the same task CONTENT, but a fresh
    random id each time (`TaskSpec.id` is uuid4). The tiebreak must depend
    only on list position, never on the id values, or "compile twice ->
    identical graph" would not hold once real decomposition produces
    independent pairs.
    """
    def _pair() -> tuple[TaskSpec, TaskSpec]:
        a = _spec("shared bug 1", "Fix a bug.", paths=["src/shared.py"])
        b = _spec("shared bug 2", "Fix another bug.", paths=["src/shared.py"])
        return a, b

    outcomes = []
    for _ in range(5):
        a, b = _pair()
        _sequence_unsafe_pairs([a, b], {})
        outcomes.append((b.depends_on == [a.id], a.depends_on == []))

    assert all(outcome == (True, True) for outcome in outcomes)
