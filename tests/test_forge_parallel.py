"""Parallel-wave Forge tests — proof the concurrency is real and still safe.

`test_forge.py` proves the machine runs. This file proves it runs CONCURRENTLY
without giving up any of the guarantees that made the sequential loop safe:

    - disjoint tasks genuinely overlap in time (measured, not asserted);
    - overlapping paths are serialised by the lease store, not by luck;
    - a governor trip observed by one thread stops the whole job;
    - an exception in a worker thread surfaces as a FAILED outcome;
    - resource-pressure refusals shrink concurrency instead of failing work;
    - results are deterministic across runs;
    - max_parallel=1 degrades to exactly the old sequential behaviour.

Executors are fake throughout — no network, no model, no subscription. Live
machine pressure is neutralised per-test so this suite passes identically on a
loaded workstation and an idle one; the concurrency being tested is the
scheduler's and the leases', not psutil's mood.
"""

from __future__ import annotations

import random
import threading
import time

from hive.contracts import Budget, Scope, TaskSpec, TaskState, TestResults
from hive.forge import ExecutionResult, Forge
from hive.registry import Adapter, CostTier, Registry, WorkerProfile


def _fleet() -> Registry:
    return Registry([
        WorkerProfile(worker_id="free.local", adapter=Adapter.GATEWAY, tier=CostTier.FREE,
                      capabilities={"edit", "python", "mechanical"}, can_edit_files=True,
                      prior_win_rate=0.85),
        WorkerProfile(worker_id="premium.cloud", adapter=Adapter.CLI_TEAM,
                      tier=CostTier.PREMIUM,
                      capabilities={"edit", "python", "mechanical", "architecture"},
                      can_edit_files=True, prior_win_rate=0.95),
    ])


def _forge(tmp_path, *, max_parallel: int, name: str = "hive") -> Forge:
    f = Forge(home=tmp_path / name, registry=_fleet(), max_attempts=3)
    # Pin the concurrency under test. The constructor sizes it to THIS machine,
    # which is correct in production and nondeterministic in a test.
    f.scheduler.max_parallel = max_parallel
    # Neutralise live pressure: admission always says yes, so concurrency is
    # bounded by the scheduler and the leases alone — the things under test.
    f.resources.may_start = lambda kind, running, pressure=None: True  # type: ignore[method-assign]
    return f


def _task(subject: str, paths: list[str], task_id: str | None = None) -> TaskSpec:
    kw = {"id": task_id} if task_id else {}
    return TaskSpec(job_id="pending", subject=subject, description="d",
                    capabilities=["edit", "python"], scope=Scope(paths=paths),
                    acceptance=["targeted tests pass"], budget=Budget(max_usd=2.0), **kw)


def _green(**kw) -> ExecutionResult:
    base = dict(
        state=TaskState.DONE, evidence="13 passed in 1.2s",
        commands_run=["python -m pytest -q"], files_touched=["src/retry.py"],
        tests=TestResults(passed=13, failed=0),
        tokens_in=1200, tokens_out=300, tokens_cached_in=4800,
        usd_micros=4200, seconds=42.0,
    )
    base.update(kw)
    return ExecutionResult(**base)


def _pass_review(spec, worker):  # noqa: ARG001
    return ExecutionResult(state=TaskState.DONE, evidence="no unresolved risk",
                           commands_run=["git diff --stat"])


# ================================================= genuine time overlap


def test_disjoint_tasks_genuinely_overlap_in_time(tmp_path):
    """Both executors must be inside their calls AT THE SAME MOMENT.

    The barrier is the proof, not a convenience: it only releases when both
    threads are simultaneously in flight. If execution were still serial, the
    first executor would sit at the barrier until its timeout, break it, and
    fail the run loudly — this test cannot pass by accident. The recorded
    windows then confirm the same fact by measurement.
    """
    barrier = threading.Barrier(2, timeout=10.0)
    windows: dict[str, tuple[float, float]] = {}
    guard = threading.Lock()

    def executor(spec, worker):  # noqa: ARG001
        start = time.monotonic()
        barrier.wait()
        time.sleep(0.05)
        with guard:
            windows[spec.subject] = (start, time.monotonic())
        return _green()

    f = _forge(tmp_path, max_parallel=2)
    try:
        result = f.run("parallel", [_task("a", ["src/a/"]), _task("b", ["src/b/"])],
                       executor=executor, reviewer=_pass_review)
    finally:
        f.close()

    assert result.accepted == 2, result.outcomes
    (sa, ea), (sb, eb) = windows["a"], windows["b"]
    assert sa < eb and sb < ea, f"intervals must intersect: a=({sa},{ea}) b=({sb},{eb})"


def test_overlapping_paths_never_run_concurrently(tmp_path):
    """The lease is the safety mechanism and concurrency must not bypass it.

    Two tasks claim the same file. If the wave ran them together the recorded
    windows would overlap (each executor is inside its call for 0.15s); the
    write lease must instead force one to wait for a later wave. Both still
    complete — serialised, never failed.
    """
    windows: dict[str, tuple[float, float]] = {}
    guard = threading.Lock()

    def executor(spec, worker):  # noqa: ARG001
        start = time.monotonic()
        time.sleep(0.15)
        with guard:
            windows[spec.subject] = (start, time.monotonic())
        return _green(files_touched=["src/shared.py"])

    f = _forge(tmp_path, max_parallel=2)
    try:
        result = f.run("contended",
                       [_task("a", ["src/shared.py"]), _task("b", ["src/shared.py"])],
                       executor=executor, reviewer=_pass_review)
    finally:
        f.close()

    assert result.accepted == 2, result.outcomes
    (sa, ea), (sb, eb) = windows["a"], windows["b"]
    assert ea <= sb or eb <= sa, (
        f"write-lease violated: both executors ran at once a=({sa},{ea}) b=({sb},{eb})"
    )


# ==================================================== the governor still rules


def test_a_governor_trip_mid_wave_stops_the_whole_job(tmp_path):
    """One thread observes the trip; the JOB stops, not just that task.

    `burn` blows its task budget while `slow` is provably in flight (the barrier
    guarantees simultaneity). `slow` is drained, not orphaned. `after` — ready
    only once `slow` completes — must never start: a halted job schedules no
    next wave.
    """
    barrier = threading.Barrier(2, timeout=10.0)
    ran: list[str] = []
    guard = threading.Lock()

    def executor(spec, worker):  # noqa: ARG001
        with guard:
            ran.append(spec.subject)
        barrier.wait()
        if spec.subject == "burn":
            return ExecutionResult(state=TaskState.RUNNING, usd_micros=9_000_000)
        time.sleep(0.2)  # still in flight when the trip lands
        return _green()

    burn = _task("burn", ["src/a/"])
    slow = _task("slow", ["src/b/"])
    after = _task("after", ["src/c/"])
    f = _forge(tmp_path, max_parallel=2)
    try:
        result = f.run("trip", [burn, slow, after], executor=executor,
                       reviewer=_pass_review, dependencies={after.id: [slow.id]})
    finally:
        f.close()

    assert "after" not in ran, "a tripped job must not start another wave"
    assert result.halted_reason.startswith("governor:"), result.halted_reason
    by_subject = {o.subject: o for o in result.outcomes}
    assert "governor" in by_subject["burn"].reason
    assert "slow" in by_subject, "an in-flight task must be drained, not vanish"
    assert "after" not in by_subject


# ======================================================== failure containment


def test_a_raising_executor_surfaces_as_failed_not_vanished(tmp_path):
    """A crashed worker thread is a FAILED outcome with the reason attached.

    The sibling task on the disjoint path must be untouched — one adapter crash
    cannot take the wave down with it, and the crashed task's leases must be
    released rather than wedging later waves.
    """
    def executor(spec, worker):  # noqa: ARG001
        if spec.subject == "boom":
            raise RuntimeError("adapter fell over")
        return _green()

    f = _forge(tmp_path, max_parallel=2)
    try:
        result = f.run("mixed", [_task("boom", ["src/a/"]), _task("ok", ["src/b/"])],
                       executor=executor, reviewer=_pass_review)
    finally:
        f.close()

    assert {o.subject for o in result.outcomes} == {"boom", "ok"}
    boom = next(o for o in result.outcomes if o.subject == "boom")
    assert not boom.accepted
    assert "RuntimeError" in boom.reason and "adapter fell over" in boom.reason
    assert next(o for o in result.outcomes if o.subject == "ok").accepted
    assert result.accepted == 1 and result.rejected == 1


def test_pressure_refusal_shrinks_concurrency_without_failing_tasks(tmp_path):
    """`may_start` saying no means "fewer at once", never "fail the task".

    The gate deterministically refuses the first two admission calls — timing
    cannot decide whether the refusal path is exercised. Refused tasks defer to
    later waves (or take the nothing-running deadlock escape) and every one of
    them still completes; a refusal must never surface as a failed outcome.
    """
    calls = {"n": 0, "refused": 0}
    guard = threading.Lock()

    def stingy_gate(kind, running, pressure=None):  # noqa: ARG001
        with guard:
            calls["n"] += 1
            if calls["n"] <= 2:
                calls["refused"] += 1
                return False
            return True

    def executor(spec, worker):  # noqa: ARG001
        time.sleep(0.05)
        return _green()

    f = _forge(tmp_path, max_parallel=3)
    f.resources.may_start = stingy_gate  # type: ignore[method-assign]
    try:
        result = f.run("pressured",
                       [_task("t1", ["src/a/"]), _task("t2", ["src/b/"]),
                        _task("t3", ["src/c/"])],
                       executor=executor, reviewer=_pass_review)
    finally:
        f.close()

    assert result.accepted == 3, result.outcomes
    assert result.rejected == 0, "a pressure refusal must never fail a task"
    assert calls["refused"] == 2, "the refusal path was never exercised"


# ============================================================== determinism


def test_outcome_order_is_deterministic_across_runs(tmp_path):
    """Completion order is scheduling noise; the result list must not carry it.

    Executors sleep a random jitter so threads finish in a different order every
    run. Outcomes are collected in submission order, so two identical jobs must
    produce identical result lists anyway.
    """
    def build_tasks() -> list[TaskSpec]:
        return [_task(f"t{i}", [f"src/m{i}/"], task_id=f"task_fixed{i}")
                for i in range(5)]

    def run_once(name: str):
        def executor(spec, worker):  # noqa: ARG001
            time.sleep(random.uniform(0, 0.05))
            return _green()

        f = _forge(tmp_path, max_parallel=3, name=name)
        try:
            r = f.run("det", build_tasks(), executor=executor, reviewer=_pass_review)
        finally:
            f.close()
        return [(o.task_id, o.accepted, o.reason) for o in r.outcomes]

    assert run_once("one") == run_once("two")


def test_max_parallel_one_is_exactly_sequential(tmp_path):
    """The degenerate pool must reproduce the old loop: one task at a time, in
    ready order, dependencies honoured, outcomes in execution order."""
    active = {"now": 0, "peak": 0}
    order: list[str] = []
    guard = threading.Lock()

    def executor(spec, worker):  # noqa: ARG001
        with guard:
            active["now"] += 1
            active["peak"] = max(active["peak"], active["now"])
            order.append(spec.subject)
        time.sleep(0.03)
        with guard:
            active["now"] -= 1
        return _green()

    first = _task("first", ["src/a/"], task_id="task_a")
    second = _task("second", ["src/b/"], task_id="task_b")
    third = _task("third", ["src/c/"], task_id="task_c")
    f = _forge(tmp_path, max_parallel=1)
    try:
        result = f.run("seq", [first, second, third], executor=executor,
                       reviewer=_pass_review, dependencies={third.id: [second.id]})
    finally:
        f.close()

    assert active["peak"] == 1, "max_parallel=1 must never overlap executions"
    assert order == ["first", "second", "third"]
    assert result.accepted == 3
    assert [o.task_id for o in result.outcomes] == ["task_a", "task_b", "task_c"]


# ============================================= one decision, one record


def test_recorded_worker_and_tier_come_from_the_same_decision(tmp_path):
    """The router picks (worker, tier) ONCE; the record must be that pair.

    This fleet forces the two selectors apart on purpose: the free worker's
    prior (0.55) misses the medium-risk threshold (0.60), so the router
    escalates to the premium worker — while the registry's own scorer, which
    knows nothing about risk thresholds, prefers the free one. Before the fix,
    `Scheduler.assign` ran that second selection and the ledger recorded a FREE
    worker against a paid tier: a pair no decision-maker ever produced, and
    exactly what a cost-attribution product cannot survive.
    """
    registry = Registry([
        WorkerProfile(worker_id="free.local", adapter=Adapter.GATEWAY,
                      tier=CostTier.FREE, capabilities={"edit", "python"},
                      can_edit_files=True, prior_win_rate=0.55, est_seconds=30.0),
        WorkerProfile(worker_id="premium.cloud", adapter=Adapter.CLI_TEAM,
                      tier=CostTier.PREMIUM, capabilities={"edit", "python"},
                      can_edit_files=True, prior_win_rate=0.95, est_seconds=60.0),
    ])
    f = Forge(home=tmp_path / "hive", registry=registry, max_attempts=3)
    f.scheduler.max_parallel = 1
    f.resources.may_start = lambda kind, running, pressure=None: True  # type: ignore[method-assign]

    routed: list = []
    real_route = f.router.route

    def spying_route(*args, **kw):
        r = real_route(*args, **kw)
        routed.append(r)
        return r

    f.router.route = spying_route  # type: ignore[method-assign]
    try:
        result = f.run("one decision", [_task("t", ["src/x.py"])],
                       executor=lambda s, w: _green(), reviewer=_pass_review)
    finally:
        f.close()

    assert routed and routed[0].worker_id == "premium.cloud", (
        "precondition broken: the router was supposed to escalate past free.local"
    )
    out = result.outcomes[0]
    assert out.accepted, out.reason
    assert out.worker_id == routed[0].worker_id
    assert out.tier == int(routed[0].tier)
