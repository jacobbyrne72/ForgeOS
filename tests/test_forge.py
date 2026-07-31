"""End-to-end Forge tests — proof the parts are actually wired together.

Every other test file proves one module works in isolation. These prove the machine
runs: a task goes in, the router picks a tier, leases are taken, output is reduced,
the merge gate rules, and a receipt comes out. Fourteen of fifteen modules had zero
importers before this file existed, which is exactly the failure it guards against.

The executor is injected and fake throughout — no network, no model, no subscription.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from forgeos.contracts import Budget, FailureClass, Scope, TaskSpec, TaskState, TestResults
from forgeos.core.market import CapacityMarket, Entitlement, Forecast, TelemetrySource
from forgeos.events import EventType
from forgeos.forge import ExecutionResult, Forge, current_task_cwd
from forgeos.leases import LeaseType
from forgeos.registry import Adapter, CostTier, Registry, WorkerProfile
from forgeos.worktrees import list_task_worktrees


def _fleet() -> Registry:
    return Registry([
        WorkerProfile(worker_id="free.local", adapter=Adapter.OLLAMA, tier=CostTier.FREE,
                      capabilities={"edit", "python", "mechanical"}, can_edit_files=True,
                      prior_win_rate=0.85),
        WorkerProfile(worker_id="premium.cloud", adapter=Adapter.CLI_TEAM,
                      tier=CostTier.PREMIUM,
                      capabilities={"edit", "python", "mechanical", "architecture"},
                      can_edit_files=True, prior_win_rate=0.95),
    ])


@pytest.fixture()
def forge(tmp_path):
    f = Forge(home=tmp_path / "forgeos", registry=_fleet(), max_attempts=3)
    yield f
    f.close()


def _task(subject="normalise retry parsing", paths=("src/retry.py",),
          caps=("edit", "python")) -> TaskSpec:
    return TaskSpec(job_id="pending", subject=subject, description="d",
                    capabilities=list(caps), scope=Scope(paths=list(paths)),
                    acceptance=["targeted tests pass"], budget=Budget(max_usd=2.0))


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


# ============================================================ happy path


def test_a_green_task_with_review_is_accepted(forge):
    result = forge.run("ship failover", [_task()],
                       executor=lambda s, w: _green(), reviewer=_pass_review)
    assert result.accepted == 1, result.outcomes
    assert result.rejected == 0
    assert result.all_accepted
    assert result.outcomes[0].reason == "merged"


def test_spend_and_cache_are_recorded_end_to_end(forge):
    result = forge.run("ship", [_task()], executor=lambda s, w: _green(),
                       reviewer=_pass_review)
    assert result.spend_usd == pytest.approx(0.0042)
    assert result.cache_hit_pct == pytest.approx(80.0)
    assert result.cost_per_accepted == pytest.approx(0.0042)


def test_cheapest_capable_worker_is_chosen(forge):
    """The cost thesis, end to end: free-and-capable must beat premium."""
    result = forge.run("ship", [_task()], executor=lambda s, w: _green(),
                       reviewer=_pass_review)
    assert result.outcomes[0].worker_id == "free.local"


def test_two_tasks_on_disjoint_paths_both_complete(forge):
    a = _task("task a", paths=["src/a/"])
    b = _task("task b", paths=["src/b/"])
    result = forge.run("ship", [a, b], executor=lambda s, w: _green(),
                       reviewer=_pass_review)
    assert result.accepted == 2


def test_worker_prompt_carries_teammate_board_context(forge):
    """`TeamBoard.context_for` (forgeos/core/awareness.py) is wired into the
    attempt loop's prompt assembly, not just unit-tested in isolation -- a
    live teammate holding a lease elsewhere must actually reach the prompt
    the executor receives."""
    forge.events.append("job-other", EventType.TASK_CREATED, task_id="T-other", subject="other work")
    forge.events.append("job-other", EventType.WORKER_ASSIGNED, task_id="T-other", worker="teammate.worker")
    forge.events.append("job-other", EventType.SESSION_STARTED, task_id="T-other", worker="teammate.worker")
    forge.leases.acquire("T-other", "default", "src/teammate/**", LeaseType.WRITE, ttl_seconds=1800)

    captured = {}

    def _executor(spec, worker):  # noqa: ARG001
        captured["description"] = spec.description
        return _green()

    result = forge.run("ship", [_task()], executor=_executor, reviewer=_pass_review)

    assert result.accepted == 1
    assert "T-other" in captured["description"]
    assert "src/teammate" in captured["description"]


# ====================================================== the gates hold


def test_a_task_with_no_tests_is_refused_by_the_merge_gate(forge):
    """Zero tests passed is not green — 'nothing ran' reported as success is the
    most common false green in an agent pipeline."""
    result = forge.run("ship", [_task()],
                       executor=lambda s, w: _green(tests=TestResults(passed=0, failed=0)),
                       reviewer=_pass_review)
    assert result.accepted == 0
    assert any("nothing was actually verified" in r
               for r in result.outcomes[0].merge_reasons)


def test_a_task_without_evidence_is_refused(forge):
    result = forge.run("ship", [_task()],
                       executor=lambda s, w: _green(evidence="", commands_run=[]),
                       reviewer=_pass_review)
    assert result.accepted == 0
    assert result.outcomes[0].merge_reasons


def test_missing_independent_review_blocks_the_merge(forge):
    """No reviewer passed in means no independent review happened."""
    result = forge.run("ship", [_task()], executor=lambda s, w: _green())
    assert result.accepted == 0
    assert any("independent review" in r for r in result.outcomes[0].merge_reasons)


def test_a_rejecting_reviewer_blocks_the_merge(forge):
    def reject(spec, worker):  # noqa: ARG001
        return ExecutionResult(state=TaskState.FAILED, evidence="unsafe pattern")

    result = forge.run("ship", [_task()], executor=lambda s, w: _green(),
                       reviewer=reject)
    assert result.accepted == 0


# ================================================ failure classification


def test_an_environment_failure_does_not_burn_retries_on_a_stronger_model(forge):
    """A broken venv fails identically at every price tier. Escalating it converts
    a fixable problem into an expensive one."""
    calls = {"n": 0}

    def broken_env(spec, worker):  # noqa: ARG001
        calls["n"] += 1
        return ExecutionResult(state=TaskState.FAILED, blocker="ModuleNotFoundError: pytest",
                               failure=FailureClass.ENVIRONMENT, usd_micros=1000)

    result = forge.run("ship", [_task()], executor=broken_env, reviewer=_pass_review)
    assert result.accepted == 0
    assert calls["n"] == 1, "must not retry an environment failure"
    assert "environment" in result.outcomes[0].reason


def test_a_model_failure_does_retry(forge):
    calls = {"n": 0}

    def flaky(spec, worker):  # noqa: ARG001
        calls["n"] += 1
        if calls["n"] < 2:
            return ExecutionResult(state=TaskState.FAILED,
                                   failure=FailureClass.MODEL, usd_micros=1000)
        return _green()

    result = forge.run("ship", [_task()], executor=flaky, reviewer=_pass_review)
    assert calls["n"] == 2
    assert result.accepted == 1


def test_attempts_are_bounded(forge):
    calls = {"n": 0}

    def always_fails(spec, worker):  # noqa: ARG001
        calls["n"] += 1
        return ExecutionResult(state=TaskState.FAILED, failure=FailureClass.MODEL,
                               usd_micros=1000)

    result = forge.run("ship", [_task()], executor=always_fails, reviewer=_pass_review)
    assert calls["n"] == 3
    assert "exhausted" in result.outcomes[0].reason


# ================================================== budget and dependencies


def test_the_governor_halts_a_runaway_task(forge):
    def expensive(spec, worker):  # noqa: ARG001
        return ExecutionResult(state=TaskState.RUNNING, usd_micros=9_000_000)

    result = forge.run("ship", [_task()], executor=expensive, reviewer=_pass_review)
    assert result.accepted == 0
    assert "governor" in result.outcomes[0].reason


def test_dependency_order_is_honoured(forge):
    order: list[str] = []
    first, second = _task("first", paths=["src/a/"]), _task("second", paths=["src/b/"])

    def record(spec, worker):  # noqa: ARG001
        order.append(spec.subject)
        return _green()

    forge.run("ship", [first, second], executor=record, reviewer=_pass_review,
              dependencies={second.id: [first.id]})
    assert order == ["first", "second"]


def test_a_task_needing_an_absent_capability_is_reported_not_crashed(forge):
    result = forge.run("ship", [_task(caps=["kubernetes"])],
                       executor=lambda s, w: _green(), reviewer=_pass_review)
    assert result.accepted == 0
    assert "capabilit" in result.outcomes[0].reason


def test_an_empty_task_list_completes_without_crashing(forge):
    result = forge.run("nothing to do", [], executor=lambda s, w: _green())
    assert result.accepted == 0 and result.rejected == 0


# ===================================================== output reduction


def test_raw_tool_output_is_reduced_before_it_can_reach_a_model(forge):
    """A 3,000-line log must never be carried forward whole."""
    noisy = "\n".join(f"tests/test_x.py::test_{i} PASSED" for i in range(1500))
    noisy += "\n=================== 1500 passed in 4.20s ===================\n"

    result = forge.run("ship", [_task()],
                       executor=lambda s, w: _green(raw_output=noisy),
                       reviewer=_pass_review)
    assert result.accepted == 1
    assert result.avoided_tokens > 0, "reduction must be recorded as a measured saving"


# ============================================================ integration


def test_forge_imports_and_wires_the_whole_stack():
    """Guard against regression to a pile of unwired modules.

    Fourteen of fifteen modules had zero importers before forge.py; this asserts the
    spine keeps referencing them so a part cannot silently fall out of the machine.
    """
    import pathlib

    src = pathlib.Path("forgeos/forge.py").read_text(encoding="utf-8")
    for part in ("awareness", "governor", "market", "resources", "router",
                 "scheduler", "timing", "verify", "avoidance", "reducer",
                 "events", "leases", "ledger", "registry", "settings"):
        assert part in src, f"forge.py no longer wires {part}"


def test_doctor_reports_the_real_machine(forge):
    text = forge.doctor()
    assert "forgeos doctor" in text
    assert "pools" in text and "providers" in text
    assert "reasoning=" in text


def test_doctor_includes_the_capacity_market_when_priced(tmp_path):
    market = CapacityMarket()
    market.record(Entitlement(resource_id="claude.sub", remaining=0.9,
                              resets_at=None, source=TelemetrySource.OFFICIAL_CLI))
    market.record(Entitlement(resource_id="openrouter", remaining=1.0,
                              source=TelemetrySource.OFFICIAL_API, cash_cost=0.5))
    market.forecast(Forecast(resource_id="claude.sub", expected_demand=0.1,
                             confidence=0.9))
    f = Forge(home=tmp_path / "h", registry=_fleet(), market=market)
    try:
        assert "claude.sub" in f.doctor()
    finally:
        f.close()


def test_max_parallel_is_sized_to_this_machine(forge):
    """A fixed worker count either starves a workstation or thrashes a laptop."""
    assert forge.scheduler.max_parallel >= 1
    assert forge.scheduler.max_parallel <= 8


# ================================================ lowering (rung 0 in the spine)


def _bulk_op(count=20_000, desc="find all import statements across the repository"):
    from forgeos.economy.lowerer import Operation

    return Operation(description=desc, item_count=count,
                     per_item_tokens_estimate=120,
                     has_deterministic_structure=True,
                     sample_items=['import os', 'import sys'])


def test_lowering_never_short_circuits_the_work(forge):
    """Lowering is advice about HOW to do the work cheaply, never a substitute for
    doing it. An accepted task with nothing run and no evidence would score a
    perfect cost-per-accepted-task by performing no work at all."""
    ran = {"n": 0}

    def normal(spec, worker):  # noqa: ARG001
        ran["n"] += 1
        return _green()

    spec = _task("extract fields", paths=["src/"])
    result = forge.run("extract", [spec], executor=normal, reviewer=_pass_review,
                       operations={spec.id: _bulk_op(count=50_000,
                                                     desc="extract seven fields from each record")})
    assert ran["n"] >= 1, "must fall back to the model path, not silently lower"
    assert result.accepted == 1


def test_a_reasoning_operation_is_never_lowered(forge):
    """'Decide whether this architecture is sound' is not algorithmic."""
    from forgeos.economy.lowerer import Operation

    ran = {"n": 0}

    def normal(spec, worker):  # noqa: ARG001
        ran["n"] += 1
        return _green()

    spec = _task("architecture review", paths=["src/"])
    op = Operation(description="decide whether this architecture is sound",
                   item_count=1, per_item_tokens_estimate=4000,
                   has_deterministic_structure=False)
    forge.run("review", [spec], executor=normal, reviewer=_pass_review,
              operations={spec.id: op})
    assert ran["n"] == 1


def test_a_task_with_no_declared_operation_runs_normally(forge):
    result = forge.run("ship", [_task()], executor=lambda s, w: _green(),
                       reviewer=_pass_review, operations={})
    assert result.accepted == 1


# =================================== test-tampering gate, wired end to end
# forge.py now passes the real ExecutionResult.files_touched into the merge
# gate, so an agent that "fixes" a failing test by editing the test itself
# (rather than the code) is caught here, not just at the verify.py unit level.


def test_a_diff_that_edits_a_test_alongside_its_source_is_blocked(forge):
    result = forge.run(
        "ship", [_task()],
        executor=lambda s, w: _green(files_touched=["tests/test_retry.py", "src/retry.py"]),
        reviewer=_pass_review,
    )
    assert result.accepted == 0
    assert any("test-tampering:" in r for r in result.outcomes[0].merge_reasons)


def test_an_ordinary_source_only_diff_is_unaffected_by_the_tamper_gate(forge):
    """The default fixture already touches only `src/retry.py` — this pins that
    the new wiring does not regress the happy path it runs on every test above."""
    result = forge.run("ship", [_task()], executor=lambda s, w: _green(),
                       reviewer=_pass_review)
    assert result.accepted == 1
    assert result.outcomes[0].merge_reasons == []


# ============================== reviewer provider-family WARN, wired end to end
# forge._family_of normalizes a WorkerProfile's vendor/model/adapter to a
# provider family — no new registry field, since every case is already
# derivable from fields WorkerProfile already has.


def _same_vendor_fleet() -> Registry:
    """Both workers report the same `vendor` ("claude") — `_family_of` should
    normalize both to the same "anthropic" family, giving the merge gate's
    self-preference WARN something real to fire on."""
    return Registry([
        WorkerProfile(worker_id="free.local", adapter=Adapter.OLLAMA, tier=CostTier.FREE,
                      vendor="claude",
                      capabilities={"edit", "python", "mechanical"}, can_edit_files=True,
                      prior_win_rate=0.85),
        WorkerProfile(worker_id="premium.cloud", adapter=Adapter.CLI_TEAM,
                      tier=CostTier.PREMIUM, vendor="claude",
                      capabilities={"edit", "python", "mechanical", "architecture"},
                      can_edit_files=True, prior_win_rate=0.95),
    ])


def test_same_provider_family_reviewer_warns_without_blocking(tmp_path):
    f = Forge(home=tmp_path / "forgeos", registry=_same_vendor_fleet(), max_attempts=3)
    try:
        result = f.run("ship", [_task()], executor=lambda s, w: _green(),
                       reviewer=_pass_review)
    finally:
        f.close()
    assert result.accepted == 1, result.outcomes
    warnings = result.outcomes[0].merge_warnings
    assert any("self-preference" in w for w in warnings)
    assert any("anthropic" in w for w in warnings), "vendor 'claude' must normalize to 'anthropic'"


def test_default_fleet_has_no_vendor_set_so_no_family_warning(forge):
    """The existing `_fleet()` fixture never declares `vendor` or a recognizable
    `model` — pins that the new wiring stays silent when neither profile gives
    `_family_of` anything to go on, rather than two unknowns matching each other."""
    result = forge.run("ship", [_task()], executor=lambda s, w: _green(),
                       reviewer=_pass_review)
    assert result.accepted == 1
    assert result.outcomes[0].merge_warnings == []


def test_family_derives_from_model_when_vendor_is_unset(tmp_path):
    """No `vendor` on either profile, but both `model` strings name the same
    vendor — `_family_of`'s model fallback exists exactly for gateway/free-tier
    profiles that route by model string rather than a fixed CLI vendor."""
    fleet = Registry([
        WorkerProfile(worker_id="a", adapter=Adapter.GATEWAY, tier=CostTier.FREE,
                      model="deepseek-v3", capabilities={"edit", "python", "mechanical"},
                      can_edit_files=True, prior_win_rate=0.85),
        WorkerProfile(worker_id="b", adapter=Adapter.GATEWAY, tier=CostTier.PREMIUM,
                      model="deepseek-r1", capabilities={"edit", "python", "mechanical"},
                      can_edit_files=True, prior_win_rate=0.95),
    ])
    f = Forge(home=tmp_path / "forgeos", registry=fleet, max_attempts=3)
    try:
        result = f.run("ship", [_task()], executor=lambda s, w: _green(),
                       reviewer=_pass_review)
    finally:
        f.close()
    assert result.accepted == 1
    assert any("deepseek" in w for w in result.outcomes[0].merge_warnings)


def test_two_local_adapter_workers_share_the_local_family(tmp_path):
    """Neither profile sets `vendor` or a recognizable `model`, but both use the
    Ollama adapter — `_family_of` falls back to "local" so a local model
    reviewing its own local output is still caught, not silently waved through
    for lack of a vendor string."""
    fleet = Registry([
        WorkerProfile(worker_id="a", adapter=Adapter.OLLAMA, tier=CostTier.FREE,
                      capabilities={"edit", "python", "mechanical"}, can_edit_files=True,
                      prior_win_rate=0.85),
        WorkerProfile(worker_id="b", adapter=Adapter.OLLAMA, tier=CostTier.PREMIUM,
                      capabilities={"edit", "python", "mechanical", "architecture"},
                      can_edit_files=True, prior_win_rate=0.95),
    ])
    f = Forge(home=tmp_path / "forgeos", registry=fleet, max_attempts=3)
    try:
        result = f.run("ship", [_task()], executor=lambda s, w: _green(),
                       reviewer=_pass_review)
    finally:
        f.close()
    assert result.accepted == 1
    assert any("local" in w for w in result.outcomes[0].merge_warnings)


# ========================================= _pick_reviewer cross-family preference
# forge._pick_reviewer should prefer a reviewer whose provider family differs
# from the implementer's, when the fleet actually has a capable one.


def test_cross_family_reviewer_preferred_over_same_family(tmp_path):
    """Implementer routes to the free anthropic worker (cheapest tier clearing
    the win-rate threshold). Both remaining workers can review, one same
    family (claude) and one cross family (deepseek) -- the deepseek worker
    must be picked, and the merge gate must carry no self-preference WARN."""
    fleet = Registry([
        WorkerProfile(worker_id="free.local", adapter=Adapter.OLLAMA, tier=CostTier.FREE,
                      vendor="claude",
                      capabilities={"edit", "python", "mechanical"}, can_edit_files=True,
                      prior_win_rate=0.85),
        WorkerProfile(worker_id="premium.claude", adapter=Adapter.CLI_TEAM,
                      tier=CostTier.PREMIUM, vendor="claude",
                      capabilities={"edit", "python", "mechanical", "architecture", "review"},
                      can_edit_files=True, prior_win_rate=0.95),
        WorkerProfile(worker_id="standard.deepseek", adapter=Adapter.GATEWAY,
                      tier=CostTier.STANDARD, model="deepseek-v3",
                      capabilities={"review", "mechanical"}, prior_win_rate=0.9),
    ])
    f = Forge(home=tmp_path / "forgeos", registry=fleet, max_attempts=3)
    try:
        result = f.run("ship", [_task()], executor=lambda s, w: _green(),
                       reviewer=_pass_review)
    finally:
        f.close()
    assert result.accepted == 1, result.outcomes
    assert not any("self-preference" in w for w in result.outcomes[0].merge_warnings)


def test_cross_family_candidate_without_review_capability_falls_back_same_family(tmp_path):
    """The only cross-family worker in the fleet (deepseek) never declares the
    "review" capability, so it is dropped before the family preference ever
    runs -- the same-family worker is still chosen, and its WARN still fires."""
    fleet = Registry([
        WorkerProfile(worker_id="free.local", adapter=Adapter.OLLAMA, tier=CostTier.FREE,
                      vendor="claude",
                      capabilities={"edit", "python", "mechanical"}, can_edit_files=True,
                      prior_win_rate=0.85),
        WorkerProfile(worker_id="premium.claude", adapter=Adapter.CLI_TEAM,
                      tier=CostTier.PREMIUM, vendor="claude",
                      capabilities={"edit", "python", "mechanical", "architecture", "review"},
                      can_edit_files=True, prior_win_rate=0.95),
        WorkerProfile(worker_id="standard.deepseek", adapter=Adapter.GATEWAY,
                      tier=CostTier.STANDARD, model="deepseek-v3",
                      capabilities={"mechanical"}, prior_win_rate=0.9),
    ])
    f = Forge(home=tmp_path / "forgeos", registry=fleet, max_attempts=3)
    try:
        result = f.run("ship", [_task()], executor=lambda s, w: _green(),
                       reviewer=_pass_review)
    finally:
        f.close()
    assert result.accepted == 1, result.outcomes
    assert any("self-preference" in w for w in result.outcomes[0].merge_warnings)


# ===================================================== worktree isolation
# `isolate_worktrees=True` (forgeos/worktrees.py) gives each file-editing task
# its own git worktree instead of sharing `cwd` directly, so a task the merge
# gate allows still has to actually land on main before it counts as accepted.
# Off by default -- every test above this section runs with it unset and must
# stay exactly as green as it always was.


def _git(args: list[str], cwd) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)


def _commit_all(cwd, message: str) -> None:
    _git(["add", "-A"], cwd)
    _git(["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", message], cwd)


def _git_repo(tmp_path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["init", "-q", "-b", "main"], repo)
    # Mirrors this repo's own .gitignore: worktree dirs are never tracked.
    (repo / ".gitignore").write_text(".forgeos-worktrees/\n", encoding="utf-8")
    (repo / "shared.txt").write_text("base\n", encoding="utf-8")
    _commit_all(repo, "init")
    return repo


def _editing_executor(content: str):
    """A fake executor that writes and commits a REAL file into whichever cwd
    isolation handed it (`current_task_cwd()`) -- a canned `ExecutionResult`
    alone, like every other test in this file uses, can't exercise real git
    worktrees or merges at all."""
    def _exec(spec, worker):  # noqa: ARG001
        cwd = Path(current_task_cwd())
        (cwd / "shared.txt").write_text(content, encoding="utf-8")
        _commit_all(cwd, f"{spec.id} edit")
        return _green(files_touched=["shared.txt"])
    return _exec


@pytest.fixture
def passing_security(monkeypatch):
    """Fake the security gate to PASS for the worktree tests.

    These tests write REAL files, so the real gate really scans — and on a
    machine without gitleaks/semgrep (every CI runner) it correctly refuses
    with "could not be checked". Scanner presence is core/verify's contract,
    proven in test_verify.py; what these tests prove is worktree mechanics,
    so the gate is faked to isolate the claim.
    """
    import forgeos.forge as forge_module
    from forgeos.core.verify import Gate, GateResult, GateStatus

    monkeypatch.setattr(
        forge_module, "run_security",
        lambda *a, **k: GateResult(gate=Gate.SECURITY, status=GateStatus.PASS,
                                   command="faked: worktree tests prove merges, not scanners"),
    )


def test_isolate_worktrees_off_by_default_never_touches_the_worktree_module(forge, monkeypatch):
    """The opt-in flag's whole point: leave every existing caller untouched."""
    import forgeos.forge as forge_module

    called = []
    monkeypatch.setattr(forge_module, "create_worktree",
                        lambda *a, **k: called.append((a, k)))
    result = forge.run("ship", [_task()], executor=lambda s, w: _green(),
                       reviewer=_pass_review)
    assert result.accepted == 1
    assert called == []


@pytest.mark.slow
def test_isolate_worktrees_two_compatible_concurrent_edits_both_merge(forge, tmp_path, passing_security):
    """Non-overlapping declared scope so the two tasks run in the SAME wave --
    leases never contend -- but their executor actually writes the SAME
    physical file both times, identically. Worktree isolation is what keeps
    that real concurrent access from corrupting either edit; identical
    changes from a shared base are what makes the merge "compatible"."""
    repo = _git_repo(tmp_path)
    a = _task("task a", paths=["src/a/"])
    b = _task("task b", paths=["src/b/"])

    result = forge.run(
        "ship", [a, b], executor=_editing_executor("shared by both\n"),
        reviewer=_pass_review, cwd=str(repo), isolate_worktrees=True,
    )

    assert result.accepted == 2, result.outcomes
    assert result.rejected == 0
    assert (repo / "shared.txt").read_text(encoding="utf-8") == "shared by both\n"
    assert list_task_worktrees(str(repo)) == []  # both cleaned up after landing


@pytest.mark.slow
def test_isolate_worktrees_conflicting_pair_second_refused_main_left_clean(forge, tmp_path, passing_security):
    """Two tasks make genuinely different edits to the same file. Whichever
    lands first wins (thread scheduling decides which); the other must be
    refused with the conflict reason, and main must show no trace of a
    half-applied second merge."""
    repo = _git_repo(tmp_path)
    a = _task("task a", paths=["src/a/"])
    b = _task("task b", paths=["src/b/"])
    written: dict[str, str] = {}

    def _diverging(spec, worker):  # noqa: ARG001
        cwd = Path(current_task_cwd())
        text = f"from {spec.id}\n"
        (cwd / "shared.txt").write_text(text, encoding="utf-8")
        _commit_all(cwd, f"{spec.id} edit")
        written[spec.id] = text
        return _green(files_touched=["shared.txt"])

    result = forge.run(
        "ship", [a, b], executor=_diverging, reviewer=_pass_review,
        cwd=str(repo), isolate_worktrees=True,
    )

    assert result.accepted == 1, result.outcomes
    assert result.rejected == 1
    accepted = next(o for o in result.outcomes if o.accepted)
    refused = next(o for o in result.outcomes if not o.accepted)
    assert any("merge conflict against main" in r for r in refused.merge_reasons)

    # Main ended up with exactly the winner's content -- no half-applied merge
    # left behind by the refused task.
    assert (repo / "shared.txt").read_text(encoding="utf-8") == written[accepted.task_id]
    assert _git(["status", "--porcelain"], repo).stdout.strip() == ""
    assert not (repo / ".git" / "MERGE_HEAD").exists()
    assert list_task_worktrees(str(repo)) == []  # both torn down either way


@pytest.mark.slow
def test_isolate_worktrees_retry_reuses_the_same_worktree(forge, tmp_path, passing_security):
    """Attempt 2 must resume in the SAME worktree attempt 1 got -- that is the
    point of isolation surviving a model-failure retry, not starting over."""
    repo = _git_repo(tmp_path)
    calls = {"n": 0}
    seen_cwds: list[str] = []

    def _flaky_then_edits(spec, worker):  # noqa: ARG001
        calls["n"] += 1
        seen_cwds.append(current_task_cwd())
        if calls["n"] < 2:
            return ExecutionResult(state=TaskState.FAILED, failure=FailureClass.MODEL,
                                   usd_micros=1000)
        cwd = Path(current_task_cwd())
        (cwd / "shared.txt").write_text("second attempt\n", encoding="utf-8")
        _commit_all(cwd, "second attempt")
        return _green(files_touched=["shared.txt"])

    result = forge.run(
        "ship", [_task("retry task", paths=["src/x/"])], executor=_flaky_then_edits,
        reviewer=_pass_review, cwd=str(repo), isolate_worktrees=True,
    )

    assert result.accepted == 1, result.outcomes
    assert calls["n"] == 2
    assert seen_cwds[0] == seen_cwds[1]  # same worktree reused across the retry
    assert (repo / "shared.txt").read_text(encoding="utf-8") == "second attempt\n"
