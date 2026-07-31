"""`python -m forgeos` — `doctor` and `receipts`.

Both commands are read-only and must never touch the network. `doctor`
wraps three existing, already-audited pieces instead of re-deriving them:
`Forge.doctor()` (machine/provider report), `adapters.factory.runnable_workers`
(constructs the real adapter per profile and calls its local-only `.health()`),
and `Catalog.stale()`/`ModelCard.age_days` (price provenance). Those three are
swapped for lightweight fakes here so these tests stay fast and independent of
what is actually installed on the machine running them -- and so they never
construct a real `Forge()`, which opens five sqlite stores under
`DEFAULT_HOME` as a side effect of construction. `receipts` reads a real
on-disk ledger seeded directly through the public `Ledger` API, the same way
`tests/test_cache_health.py` seeds spend rows -- no LLM calls, no subprocess.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from forgeos import __main__ as cli
from forgeos.catalog import Catalog, ModelCard
from forgeos.contracts import JobSpec, TaskSpec, TaskState, Verdict, WorkerReport
from forgeos.forge import DEFAULT_HOME, ForgeResult, TaskOutcome
from forgeos.ledger import Ledger

# --------------------------------------------------------------------- doctor


class _FakeForge:
    """Stands in for `Forge()` in `cmd_doctor`."""

    def __init__(self, doctor_text: str, workers: list):
        self._doctor_text = doctor_text
        self.registry = SimpleNamespace(all=lambda: workers)
        self.closed = False

    def doctor(self) -> str:
        return self._doctor_text

    def close(self) -> None:
        self.closed = True


@pytest.fixture()
def patched_doctor(monkeypatch):
    """Swap doctor's three data sources for deterministic fakes."""

    def apply(*, doctor_text="forgeos doctor\nmachine  ok", workers=None,
              runnable_results=None, catalog=None):
        workers = workers if workers is not None else []
        fake = _FakeForge(doctor_text, workers)
        monkeypatch.setattr("forgeos.Forge", lambda **kw: fake)
        monkeypatch.setattr(
            "forgeos.adapters.factory.runnable_workers",
            lambda profiles: runnable_results if runnable_results is not None else {},
        )
        monkeypatch.setattr("forgeos.catalog.default_catalog", lambda: catalog or Catalog([]))
        return fake

    return apply


def test_doctor_prints_forge_doctor_report_verbatim(capsys, patched_doctor):
    patched_doctor(doctor_text="forgeos doctor\nmachine  desktop  8 cores")
    rc = cli.main(["doctor"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "forgeos doctor" in out
    assert "machine  desktop  8 cores" in out


def test_doctor_reports_runnable_workers_from_the_router_ground_truth(capsys, patched_doctor):
    fake = patched_doctor(
        workers=["w1", "w2", "w3"],
        runnable_results={
            "w1": "ok: omc entrypoint present",
            "w2": "unavailable: 'node' not found on PATH",
            "w3": "ok: ollama daemon reachable",
        },
    )
    rc = cli.main(["doctor"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Registry: 3 worker(s)" in out
    assert "Runnable now: 2/3" in out
    assert "w1" in out and "ok: omc entrypoint present" in out
    assert "w2" in out and "unavailable: 'node' not found on PATH" in out
    assert fake.closed is True  # Forge is always closed, even on the happy path


def test_doctor_closes_forge_even_if_a_later_section_raises(monkeypatch, patched_doctor):
    fake = patched_doctor()
    monkeypatch.setattr(
        "forgeos.adapters.factory.runnable_workers",
        lambda profiles: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(RuntimeError):
        cli.main(["doctor"])
    assert fake.closed is True


def test_doctor_reports_unknown_staleness_for_an_empty_catalog(capsys, patched_doctor):
    patched_doctor(catalog=Catalog([]))
    cli.main(["doctor"])
    out = capsys.readouterr().out
    assert "Price catalog: empty (no local cache found)" in out


def test_doctor_reports_the_oldest_stamped_price_age(capsys, patched_doctor):
    now = __import__("time").time()
    fresh = ModelCard(model_id="m1", provider="p", fetched_at=now - 3600)          # ~1h old, not stale
    old = ModelCard(model_id="m2", provider="p", fetched_at=now - 40 * 86400)      # ~40d old, stale
    unstamped = ModelCard(model_id="m3", provider="p")                            # fetched_at=0.0, stale
    patched_doctor(catalog=Catalog([fresh, old, unstamped]))
    cli.main(["doctor"])
    out = capsys.readouterr().out
    assert "Price catalog: 3 model(s), 2 stale or unstamped (>30d)" in out
    assert "39.9" in out or "40.0" in out  # ~40 day(s) old, timing-tolerant


def test_doctor_returns_1_and_names_the_fix_when_the_home_dir_is_unusable(capsys, monkeypatch):
    def _raise(**kw):
        raise OSError("Permission denied: '/root/.forgeos'")

    monkeypatch.setattr("forgeos.Forge", _raise)
    rc = cli.main(["doctor"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "Permission denied" in out
    assert "Fix:" in out


# ------------------------------------------------------------------- receipts


@pytest.fixture()
def seeded_state_dir(tmp_path):
    """A real on-disk ledger: one job, one accepted task, one failed task,
    spend split across two workers -- seeded through the public `Ledger` API
    exactly the way `Forge` records it."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    ledger = Ledger(state_dir / "ledger.db")
    job = JobSpec(objective="prove receipts sums spend correctly", cwd=".")
    ledger.open_job(job)
    done_task = TaskSpec(job_id=job.id, subject="implement the thing", description="d")
    failed_task = TaskSpec(job_id=job.id, subject="a task that never landed", description="d")
    ledger.add_task(done_task, state=TaskState.DONE)
    ledger.add_task(failed_task, state=TaskState.FAILED)
    ledger.record_spend(job.id, "forgeos.executor", "claude-opus", 1_500_000, task_id=done_task.id)
    ledger.record_spend(job.id, "forgeos.executor", "claude-opus", 500_000, task_id=failed_task.id)
    ledger.record_spend(job.id, "forgeos.verifier", "claude-haiku", 200_000, task_id=done_task.id)
    ledger.close()
    return state_dir


def test_receipts_with_no_ledger_names_the_path_and_the_fix_and_returns_1(tmp_path, capsys):
    empty_dir = tmp_path / "never-run"
    rc = cli.main(["receipts", "--state-dir", str(empty_dir)])
    out = capsys.readouterr().out
    assert rc == 1
    assert str(empty_dir) in out
    assert "Fix:" in out
    # Read-only: must not have created the state dir or a ledger file.
    assert not empty_dir.exists()


def test_receipts_with_an_unreadable_ledger_file_returns_1(tmp_path, capsys):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "ledger.db").write_text("not a sqlite database", encoding="utf-8")
    rc = cli.main(["receipts", "--state-dir", str(state_dir)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "Cannot open ledger" in out
    assert "Fix:" in out


def test_receipts_with_an_empty_but_valid_ledger_returns_0(tmp_path, capsys):
    """A ledger that opened fine and simply has no jobs yet is a usable state
    dir with nothing to report -- not the same as an unusable one."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    Ledger(state_dir / "ledger.db").close()  # create schema, no jobs
    rc = cli.main(["receipts", "--state-dir", str(state_dir)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "no jobs recorded" in out


def test_receipts_summarizes_spend_by_job_and_cost_per_accepted(seeded_state_dir, capsys):
    rc = cli.main(["receipts", "--state-dir", str(seeded_state_dir)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Jobs: 1" in out
    assert "tasks=2 done=1 failed=1" in out
    assert "spend=$2.2000" in out
    assert "$/accepted=$2.2000" in out


def test_receipts_summarizes_spend_by_worker(seeded_state_dir, capsys):
    cli.main(["receipts", "--state-dir", str(seeded_state_dir)])
    out = capsys.readouterr().out
    assert "Spend by worker:" in out
    assert "forgeos.executor" in out and "2 call(s)" in out and "$2.0000" in out
    assert "forgeos.verifier" in out and "1 call(s)" in out and "$0.2000" in out


def test_receipts_totals_across_jobs(seeded_state_dir, capsys):
    cli.main(["receipts", "--state-dir", str(seeded_state_dir)])
    out = capsys.readouterr().out
    assert "Total: $2.2000 across 1 job(s), 1 accepted task(s), $2.2000/accepted" in out


def test_receipts_json_reports_resume_candidates_and_spend_provenance(tmp_path, capsys):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    ledger = Ledger(state_dir / "ledger.db")
    job = JobSpec(objective="find resumable work", cwd=".")
    ledger.open_job(job)
    done_task = TaskSpec(job_id=job.id, subject="finished", description="d")
    pending_task = TaskSpec(job_id=job.id, subject="resume me", description="d")
    ledger.add_task(done_task, state=TaskState.DONE)
    ledger.add_task(pending_task, state=TaskState.PENDING)
    ledger.record_spend(job.id, "worker", "model", 300_000, task_id=done_task.id)
    ledger.record_spend(job.id, "worker", "subscription", 700_000,
                        task_id=pending_task.id, kind="estimate")
    ledger.close()

    rc = cli.main(["receipts", "--state-dir", str(state_dir), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["schema"] == "forgeos.receipts.v1"
    assert payload["ok"] is True
    receipt = payload["jobs"][0]
    assert receipt["unfinished_tasks"] == 1
    assert receipt["resume_available"] is True
    assert receipt["resume_command"].startswith("python -m forgeos resume ")
    assert receipt["isolate_worktrees"] is False
    assert receipt["base_ref"] is None
    assert receipt["measured_spend_usd"] == 0.3
    assert receipt["modelled_spend_usd"] == 0.7
    assert payload["totals"]["attributed_spend_usd"] == 1.0


def test_receipts_with_no_accepted_tasks_reports_cost_as_not_applicable(tmp_path, capsys):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    ledger = Ledger(state_dir / "ledger.db")
    job = JobSpec(objective="everything failed", cwd=".")
    ledger.open_job(job)
    t = TaskSpec(job_id=job.id, subject="doomed", description="d")
    ledger.add_task(t, state=TaskState.FAILED)
    ledger.record_spend(job.id, "forgeos.executor", "claude-opus", 100_000, task_id=t.id)
    ledger.close()

    rc = cli.main(["receipts", "--state-dir", str(state_dir)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "$/accepted=n/a" in out
    assert "0 accepted task(s), n/a/accepted" in out


def test_receipts_defaults_to_default_home_when_state_dir_is_omitted():
    assert cli._resolve_state_dir(None) == DEFAULT_HOME
    assert cli._resolve_state_dir("/tmp/explicit") == cli.Path("/tmp/explicit")


# --------------------------------------------------------------- preflight


def _write_preflight_task(path, *, description="new prose"):
    path.write_text(json.dumps({
        "subject": "Add retry logic to the gateway client",
        "description": description,
        "acceptance": ["python -m pytest tests/test_gateway.py -q passes"],
        "scope": {"paths": ["forgeos/gateway/client.py"]},
        "capabilities": ["python"],
    }), encoding="utf-8")


def test_preflight_json_allows_without_creating_a_missing_ledger(tmp_path, capsys):
    task_file = tmp_path / "task.json"
    _write_preflight_task(task_file)
    state_dir = tmp_path / "never-run"

    rc = cli.main([
        "preflight", str(task_file), "--state-dir", str(state_dir), "--json",
    ])
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["schema"] == "forgeos.preflight.v1"
    assert payload["decision"] == "allow"
    assert payload["ledger_present"] is False
    assert payload["fingerprint"].startswith("fp_")
    assert not state_dir.exists()


def test_preflight_json_refuses_an_exact_prior_contract(tmp_path, capsys):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    ledger = Ledger(state_dir / "ledger.db")
    job = JobSpec(objective="prior work", cwd=".")
    ledger.open_job(job)
    prior = TaskSpec(
        job_id=job.id,
        subject="Add retry logic to the gateway client",
        description="old prose is intentionally ignored by the fingerprint",
        acceptance=["python -m pytest tests/test_gateway.py -q passes"],
        scope={"paths": ["forgeos/gateway/client.py"]},
        capabilities=["python"],
    )
    ledger.add_task(prior, state=TaskState.DONE)
    ledger.record_spend(job.id, "worker", "model", 1_250_000, task_id=prior.id)
    ledger.record_report(WorkerReport(
        task_id=prior.id, worker_id="worker", state=TaskState.DONE,
        verdict=Verdict.PASS, evidence="merged and tested",
    ))
    ledger.close()
    task_file = tmp_path / "task.json"
    _write_preflight_task(task_file, description="new prose")

    rc = cli.main([
        "preflight", str(task_file), "--state-dir", str(state_dir), "--json",
    ])
    payload = json.loads(capsys.readouterr().out)

    assert rc == 2
    assert payload["decision"] == "refuse_duplicate"
    assert payload["allowed"] is False
    assert payload["prior"]["task_id"] == prior.id
    assert payload["prior"]["spend_usd"] == 1.25
    assert "already merged" in payload["reason"]


# --------------------------------------------------------- call preflight


def test_call_preflight_json_estimates_without_contacting_a_provider(tmp_path, monkeypatch, capsys):
    from forgeos import catalog as catalog_module

    card = ModelCard(
        model_id="local-model", provider="local", input_cost_per_1m=1.0,
        output_cost_per_1m=2.0, context=1_000,
    )
    monkeypatch.setattr(catalog_module, "default_catalog", lambda: Catalog([card]))
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Inspect this small prompt.", encoding="utf-8")

    rc = cli.main([
        "call-preflight", "--prompt-file", str(prompt_file), "--model", "local/local-model",
        "--expected-output-tokens", "10", "--remaining-usd", "1", "--json",
    ])
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["schema"] == "forgeos.call_preflight.v1"
    assert payload["decision"] == "allow"
    assert payload["estimate"]["tokens_in"] > 0
    assert payload["estimate"]["estimated_usd"] > 0
    assert payload["price_provenance"] == "unknown"


def test_call_preflight_json_refuses_when_remaining_budget_is_too_small(tmp_path, monkeypatch, capsys):
    from forgeos import catalog as catalog_module

    card = ModelCard(
        model_id="expensive", provider="local", input_cost_per_1m=10.0,
        output_cost_per_1m=20.0, context=1_000,
    )
    monkeypatch.setattr(catalog_module, "default_catalog", lambda: Catalog([card]))
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("A prompt that costs something.", encoding="utf-8")

    rc = cli.main([
        "call-preflight", "--prompt-file", str(prompt_file), "--model", "local/expensive",
        "--expected-output-tokens", "100", "--remaining-usd", "0.000001", "--json",
    ])
    payload = json.loads(capsys.readouterr().out)

    assert rc == 2
    assert payload["decision"] == "refuse_budget"
    assert payload["allowed"] is False
    assert "exceeds remaining budget" in payload["reason"]


# ------------------------------------------------------------------------ cli


def test_main_with_no_subcommand_prints_help_and_returns_nonzero(capsys):
    rc = cli.main([])
    out = capsys.readouterr().out
    assert rc != 0
    assert "doctor" in out and "receipts" in out


def test_main_dispatch_is_reachable_for_every_registered_subcommand():
    """The same class of bug `tests/test_cli_dispatch.py` guards against in
    `cli.py`: a subcommand that parses but has no handler wired to it."""
    import ast
    import inspect

    src = inspect.getsource(cli)
    tree = ast.parse(src)
    main_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")

    registered = {
        node.args[0].value
        for node in ast.walk(main_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_parser"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }
    dispatch_keys = next(
        {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
        for node in ast.walk(main_node)
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", "") == "dispatch" for t in node.targets)
        and isinstance(node.value, ast.Dict)
    )
    assert registered == {
        "doctor", "preflight", "call-preflight", "receipts", "watch", "queue-status", "team", "resume",
        "serve-mcp", "memory"
    }
    assert dispatch_keys == registered
    # And main() must actually call the dispatch table, not just build it.
    returns = [n for n in ast.walk(main_node) if isinstance(n, ast.Return)]
    assert any(
        isinstance(r.value, ast.Subscript) or isinstance(r.value, ast.Call)
        for r in returns
        if r.value is not None
    )


# ---------------------------------------------------------------------- team


class _FakeTeamForge:
    """Stands in for `Forge()` in `_run_team`, same shape as `test_watch.py`'s
    `_FakeForge`: `.run()` returns a REAL `ForgeResult`/`TaskOutcome` so the
    printing and exit-code logic below is exercised against the real
    contract, not a second approximation of it.

    `.registry`/`.ledger` stand in for the real `Forge`'s attributes: `_run_team`
    passes both to `routed_executor` to build the reviewer, and `routed_executor`
    never touches either until the returned callable is actually invoked -- which
    a fake `.run()` that just returns a canned result never does."""

    def __init__(self, result: ForgeResult):
        self._result = result
        self.calls: list[dict] = []
        self.closed = False
        self.registry = object()
        self.ledger = object()

    def run(self, objective, tasks, *, cwd=".", budget=None, dependencies=None, reviewer=None):
        self.calls.append(
            {"objective": objective, "tasks": tasks, "cwd": cwd, "budget": budget,
             "dependencies": dependencies, "reviewer": reviewer}
        )
        return self._result

    def close(self) -> None:
        self.closed = True


def _refusing_factory():
    raise AssertionError("forge_factory must not be called")


def test_team_dry_run_prints_the_task_graph_and_never_touches_forge(tmp_path, capsys):
    rc = cli._run_team(
        "add a retry helper",
        cwd=str(tmp_path),
        budget_usd=None,
        state_dir=None,
        dry_run=True,
        forge_factory=_refusing_factory,
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Mission: add a retry helper" in out
    assert "Tasks: 1" in out


def test_team_without_dry_run_refuses_a_missing_budget_and_returns_1(tmp_path, capsys):
    rc = cli._run_team(
        "add a retry helper",
        cwd=str(tmp_path),
        budget_usd=None,
        state_dir=None,
        dry_run=False,
        forge_factory=_refusing_factory,
    )
    out = capsys.readouterr().out
    assert rc == 1
    assert "--budget-usd" in out
    assert "Fix:" in out


def test_team_run_with_everything_accepted_returns_0_and_prints_outcomes(tmp_path, capsys):
    result = ForgeResult(
        job_id="j1",
        objective="add a retry helper",
        accepted=1,
        rejected=0,
        spend_usd=1.23,
        outcomes=[
            TaskOutcome(
                task_id="task-1", subject="add a retry helper", accepted=True,
                worker_id="forgeos.executor", tier=1, attempts=1, usd_micros=1_230_000,
            ),
        ],
    )
    fake = _FakeTeamForge(result)

    rc = cli._run_team(
        "add a retry helper",
        cwd=str(tmp_path),
        budget_usd=5.0,
        state_dir=None,
        dry_run=False,
        forge_factory=lambda: fake,
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert fake.closed is True
    assert fake.calls[0]["budget"].max_usd == 5.0
    assert fake.calls[0]["reviewer"] is not None
    assert "forgeos.executor" in out
    assert "accepted=1 rejected=0" in out
    assert "$1.2300" in out


def test_team_run_passes_a_routed_reviewer_so_the_merge_gate_can_pass(tmp_path):
    """Before this, `_run_team` never passed `reviewer=`, so `Forge.run` defaulted
    to `reviewer=None` and every task was refused at the merge gate for lack of
    independent review -- see `Forge._pick_reviewer` / `core.verify`'s
    `"no independent review"` reason. `reviewer` must be built (not left None) and
    handed to `forge.run`, the same `routed_executor` construction `Forge.run`
    itself uses for its default executor."""
    result = ForgeResult(job_id="j1", objective="x", accepted=1, rejected=0)
    fake = _FakeTeamForge(result)

    cli._run_team(
        "x", cwd=str(tmp_path), budget_usd=5.0, state_dir=None, dry_run=False,
        forge_factory=lambda: fake,
    )

    assert len(fake.calls) == 1
    assert fake.calls[0]["reviewer"] is not None
    assert callable(fake.calls[0]["reviewer"])


def test_team_run_with_a_rejection_returns_2(tmp_path, capsys):
    result = ForgeResult(
        job_id="j1",
        objective="x",
        accepted=1,
        rejected=1,
        spend_usd=2.0,
        outcomes=[
            TaskOutcome(task_id="task-1", subject="a", accepted=True, worker_id="w1", attempts=1),
            TaskOutcome(
                task_id="task-2", subject="b", accepted=False, reason="test(s) failing",
                attempts=2, merge_warnings=["worktree merge produced a conflict"],
            ),
        ],
    )
    fake = _FakeTeamForge(result)

    rc = cli._run_team(
        "x", cwd=str(tmp_path), budget_usd=5.0, state_dir=None, dry_run=False,
        forge_factory=lambda: fake,
    )
    out = capsys.readouterr().out

    assert rc == 2
    assert fake.closed is True
    assert "accepted=1 rejected=1" in out
    assert "test(s) failing" in out
    assert "worktree merge produced a conflict" in out


def test_team_run_prints_guidance_when_refused_for_lack_of_independent_review(tmp_path, capsys):
    """A single-capable-worker fleet still gets a real, honest merge-gate refusal
    (`Forge._pick_reviewer` returns "" on purpose -- see forge.py:1106) -- that
    refusal is correct behavior, not a bug this CLI should paper over. But the
    generic `TaskOutcome.reason` ("merge gate refused") doesn't say why, and the
    detail lives in `merge_reasons`, not `reason` -- see `core.verify.py`'s
    `"no independent review"` line and `forge.py`'s `_STRUCTURAL_REFUSALS`. The
    CLI should surface that specific cause with one extra guidance line."""
    result = ForgeResult(
        job_id="j1",
        objective="x",
        accepted=0,
        rejected=1,
        outcomes=[
            TaskOutcome(
                task_id="task-1", subject="a", accepted=False, worker_id="forgeos.executor",
                reason="merge gate refused", merge_reasons=["no independent review"], attempts=3,
            ),
        ],
    )
    fake = _FakeTeamForge(result)

    rc = cli._run_team(
        "x", cwd=str(tmp_path), budget_usd=5.0, state_dir=None, dry_run=False,
        forge_factory=lambda: fake,
    )
    out = capsys.readouterr().out

    assert rc == 2
    assert "review needs a second capable worker" in out
    assert "only one that can take this task" in out


def test_main_team_subcommand_parses_args_and_reaches_run_team(tmp_path, monkeypatch):
    captured = {}

    def _fake_run_team(objective, **kwargs):
        captured["objective"] = objective
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(cli, "_run_team", _fake_run_team)
    rc = cli.main([
        "team", "add a retry helper",
        "--cwd", str(tmp_path),
        "--budget-usd", "3.5",
        "--state-dir", str(tmp_path / "state"),
        "--dry-run",
    ])

    assert rc == 0
    assert captured["objective"] == "add a retry helper"
    assert captured["cwd"] == str(tmp_path)
    assert captured["budget_usd"] == 3.5
    assert captured["state_dir"] == str(tmp_path / "state")
    assert captured["dry_run"] is True
