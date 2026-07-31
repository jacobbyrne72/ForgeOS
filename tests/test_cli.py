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

from types import SimpleNamespace

import pytest

from forgeos import __main__ as cli
from forgeos.catalog import Catalog, ModelCard
from forgeos.contracts import JobSpec, TaskSpec, TaskState
from forgeos.forge import DEFAULT_HOME
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
    assert registered == {"doctor", "receipts"}
    assert dispatch_keys == registered
    # And main() must actually call the dispatch table, not just build it.
    returns = [n for n in ast.walk(main_node) if isinstance(n, ast.Return)]
    assert any(
        isinstance(r.value, ast.Subscript) or isinstance(r.value, ast.Call)
        for r in returns
        if r.value is not None
    )
