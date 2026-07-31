"""`forgeos.watch.watch_queue` — the unattended job-queue daemon.

`watch_queue` never constructs a real `Forge()` in these tests -- that opens
five sqlite stores under `state_dir` as a side effect of construction, which
a test must not depend on or leave behind. Instead every test injects a
`forge_factory` returning a small fake exposing `.run()`/`.close()`, and
`.run()` returns a REAL `ForgeResult` (a plain pydantic model, safe to
construct directly) so the receipt-writing and done/failed routing below is
exercised against the real contract, not a second approximation of it.
"""

from __future__ import annotations

import json
from pathlib import Path

from forgeos.forge import ForgeResult
from forgeos.watch import (
    WATCH_HALT_KEY,
    WatchStats,
    _QueueOwnership,
    _write_heartbeat,
    queue_status,
    watch_queue,
)


class _FakeForge:
    """Stands in for `Forge()`. `run_outcomes` is consumed in order, one
    per call to `.run()` -- an `Exception` instance is raised instead of
    returned, so a test can simulate `forge.run()` blowing up mid-batch."""

    def __init__(self, run_outcomes: list):
        self._outcomes = list(run_outcomes)
        self.calls: list[dict] = []
        self.closed = False

    def run(self, objective, tasks, *, cwd=".", budget=None):
        self.calls.append({"objective": objective, "tasks": tasks, "cwd": cwd, "budget": budget})
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def close(self) -> None:
        self.closed = True


def _job_spec(*, objective="do the thing", budget_usd=5.0, tasks=None) -> dict:
    return {
        "objective": objective,
        "cwd": ".",
        "budget_usd": budget_usd,
        "tasks": tasks if tasks is not None else [{"subject": "s", "description": "d"}],
    }


def _write_spec(queue: Path, name: str, spec: dict) -> None:
    (queue / "incoming" / name).write_text(json.dumps(spec), encoding="utf-8")


# --------------------------------------------------------------- happy path


def test_once_processes_a_good_job_into_done_with_a_receipt(tmp_path):
    queue = tmp_path / "queue"
    (queue / "incoming").mkdir(parents=True)
    _write_spec(queue, "job1.json", _job_spec())

    result = ForgeResult(job_id="j1", objective="do the thing", accepted=1, rejected=0, spend_usd=0.5)
    forge = _FakeForge([result])

    stats = watch_queue(queue, once=True, forge_factory=lambda: forge)

    assert stats == WatchStats(jobs_done=1, jobs_failed=0, halted=False)
    assert not (queue / "incoming" / "job1.json").exists()
    assert (queue / "done" / "job1.json").exists()
    receipt = json.loads((queue / "done" / "job1.receipt.json").read_text())
    assert receipt["accepted"] == 1
    assert receipt["spend_usd"] == 0.5
    assert forge.closed is True


def test_a_job_that_ran_but_was_not_all_accepted_lands_in_failed_with_a_receipt(tmp_path):
    """A failed job still gets a receipt -- that is the culture."""
    queue = tmp_path / "queue"
    (queue / "incoming").mkdir(parents=True)
    _write_spec(queue, "a.json", _job_spec())

    result = ForgeResult(job_id="j1", objective="x", accepted=1, rejected=1, spend_usd=1.0)
    forge = _FakeForge([result])

    stats = watch_queue(queue, once=True, forge_factory=lambda: forge)

    assert stats.jobs_done == 0
    assert stats.jobs_failed == 1
    assert (queue / "failed" / "a.json").exists()
    receipt = json.loads((queue / "failed" / "a.receipt.json").read_text())
    assert receipt["rejected"] == 1


# -------------------------------------------------------- malformed specs


def test_malformed_json_is_refused_to_failed_with_a_reason_and_never_reaches_forge(tmp_path):
    queue = tmp_path / "queue"
    (queue / "incoming").mkdir(parents=True)
    (queue / "incoming" / "bad.json").write_text("{not json", encoding="utf-8")

    forge = _FakeForge([])
    stats = watch_queue(queue, once=True, forge_factory=lambda: forge)

    assert stats.jobs_failed == 1
    assert (queue / "failed" / "bad.json").exists()
    reason = (queue / "failed" / "bad.reason.txt").read_text()
    assert "invalid JSON" in reason
    assert forge.calls == []


def test_missing_budget_usd_is_refused_rather_than_run_with_a_made_up_cap(tmp_path):
    queue = tmp_path / "queue"
    (queue / "incoming").mkdir(parents=True)
    spec = _job_spec()
    del spec["budget_usd"]
    _write_spec(queue, "nobudget.json", spec)

    forge = _FakeForge([])
    stats = watch_queue(queue, once=True, forge_factory=lambda: forge)

    assert stats.jobs_failed == 1
    reason = (queue / "failed" / "nobudget.reason.txt").read_text()
    assert "budget_usd" in reason
    assert forge.calls == []  # refused before ever calling Forge.run


def test_invalid_task_shape_is_refused_with_a_reason_naming_the_task(tmp_path):
    queue = tmp_path / "queue"
    (queue / "incoming").mkdir(parents=True)
    spec = _job_spec(tasks=[{"subject": ""}])  # blank subject, missing description
    _write_spec(queue, "badtask.json", spec)

    forge = _FakeForge([])
    stats = watch_queue(queue, once=True, forge_factory=lambda: forge)

    assert stats.jobs_failed == 1
    reason = (queue / "failed" / "badtask.reason.txt").read_text()
    assert "tasks[0]" in reason


def test_forge_run_raising_refuses_that_job_without_crashing_the_rest_of_the_batch(tmp_path):
    queue = tmp_path / "queue"
    (queue / "incoming").mkdir(parents=True)
    _write_spec(queue, "a.json", _job_spec())
    _write_spec(queue, "b.json", _job_spec())

    forge = _FakeForge([
        RuntimeError("boom"),
        ForgeResult(job_id="j2", objective="x", accepted=1, rejected=0),
    ])
    stats = watch_queue(queue, once=True, forge_factory=lambda: forge)

    assert stats.jobs_failed == 1
    assert stats.jobs_done == 1
    reason = (queue / "failed" / "a.reason.txt").read_text()
    assert "forge.run raised" in reason and "boom" in reason
    assert (queue / "done" / "b.json").exists()


# ------------------------------------------------------------------- heartbeat


def test_heartbeat_is_written_after_a_once_pass_over_an_empty_backlog(tmp_path):
    queue = tmp_path / "queue"
    (queue / "incoming").mkdir(parents=True)
    forge = _FakeForge([])

    watch_queue(queue, once=True, forge_factory=lambda: forge)

    heartbeat = json.loads((queue / "heartbeat.json").read_text())
    assert heartbeat["state"] == "idle"
    assert heartbeat["current_job"] is None
    assert heartbeat["jobs_done"] == 0
    assert heartbeat["jobs_failed"] == 0
    assert "ts" in heartbeat


def test_heartbeat_reflects_counts_after_processing_jobs(tmp_path):
    queue = tmp_path / "queue"
    (queue / "incoming").mkdir(parents=True)
    _write_spec(queue, "a.json", _job_spec())

    forge = _FakeForge([ForgeResult(job_id="j1", objective="x", accepted=1, rejected=0)])
    watch_queue(queue, once=True, forge_factory=lambda: forge)

    heartbeat = json.loads((queue / "heartbeat.json").read_text())
    assert heartbeat["jobs_done"] == 1
    assert heartbeat["state"] == "idle"  # back to idle once the backlog drains


def test_heartbeat_publication_is_atomic_and_leaves_no_temp_snapshot(tmp_path):
    _write_heartbeat(tmp_path, state="running", current_job="job.json", stats=WatchStats())

    heartbeat = json.loads((tmp_path / "heartbeat.json").read_text(encoding="utf-8"))
    assert heartbeat["state"] == "running"
    assert heartbeat["current_job"] == "job.json"
    assert list(tmp_path.glob(".heartbeat-*.tmp")) == []


def test_queue_status_reports_fresh_heartbeat_and_idle_owner(tmp_path):
    _write_heartbeat(tmp_path, state="idle", current_job=None, stats=WatchStats(jobs_done=2))

    status = queue_status(tmp_path)

    assert status.owner_active is False
    assert status.heartbeat_valid is True
    assert status.stale is False
    assert status.jobs_done == 2


def test_queue_status_reports_stale_heartbeat_and_bad_json(tmp_path):
    import time

    (tmp_path / "heartbeat.json").write_text(
        json.dumps({"ts": time.time() - 100, "state": "running"}), encoding="utf-8"
    )
    stale = queue_status(tmp_path, stale_after_seconds=10)
    assert stale.heartbeat_valid is True and stale.stale is True

    (tmp_path / "heartbeat.json").write_text("{not-json", encoding="utf-8")
    broken = queue_status(tmp_path)
    assert broken.heartbeat_valid is False
    assert "JSONDecodeError" in (broken.heartbeat_error or "")


def test_second_queue_worker_refuses_before_constructing_forge(tmp_path):
    queue = tmp_path / "queue"
    queue.mkdir()
    owner = _QueueOwnership(queue)
    owner.acquire()
    factory_calls = []
    try:
        stats = watch_queue(
            queue,
            once=True,
            forge_factory=lambda: factory_calls.append("constructed"),
        )
        assert stats.ownership_conflict is True
        assert stats.jobs_done == 0 and stats.jobs_failed == 0
        assert factory_calls == []
    finally:
        owner.release()

    # OS-held ownership is crash-safe and reusable after release; no stale PID
    # marker or manual cleanup is needed for the next worker.
    stats = watch_queue(queue, once=True, forge_factory=lambda: _FakeForge([]))
    assert stats.ownership_conflict is False


def test_queue_ownership_is_enforced_across_processes(tmp_path):
    """The lock must protect the real deployment shape, not just threads."""
    import subprocess
    import sys
    import time

    queue = tmp_path / "queue"
    queue.mkdir()
    ready = tmp_path / "ready"
    stop = tmp_path / "stop"
    child_code = (
        "from pathlib import Path\n"
        "import sys, time\n"
        "from forgeos.watch import _QueueOwnership\n"
        "q=Path(sys.argv[1]); ready=Path(sys.argv[2]); stop=Path(sys.argv[3])\n"
        "owner=_QueueOwnership(q); owner.acquire(); ready.write_text('ready')\n"
        "while not stop.exists():\n    time.sleep(0.01)\n"
        "owner.release()\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", child_code, str(queue), str(ready), str(stop)],
        cwd=str(Path.cwd()),
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), "child never acquired the queue lock"
        stats = watch_queue(queue, once=True, forge_factory=lambda: _FakeForge([]))
        assert stats.ownership_conflict is True
    finally:
        stop.write_text("stop", encoding="utf-8")
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)
    assert child.returncode == 0


# ----------------------------------------------------------------------- halt


def test_halt_flag_set_before_the_run_stops_the_whole_batch(tmp_path):
    queue = tmp_path / "queue"
    (queue / "incoming").mkdir(parents=True)
    _write_spec(queue, "a.json", _job_spec())
    _write_spec(queue, "b.json", _job_spec())

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "halts.json").write_text(
        json.dumps({WATCH_HALT_KEY: {"halted": True, "reason": "operator stop"}}),
        encoding="utf-8",
    )

    forge = _FakeForge([ForgeResult(job_id="j", objective="x", accepted=1, rejected=0)])
    stats = watch_queue(queue, state_dir=state_dir, once=True, forge_factory=lambda: forge)

    assert stats.halted is True
    assert forge.calls == []
    assert (queue / "incoming" / "a.json").exists()
    assert (queue / "incoming" / "b.json").exists()


def test_halt_flag_set_mid_batch_stops_before_the_next_job(tmp_path):
    """The operator halts between jobs, not mid-job -- Forge.run's own
    per-job `_halted(job.id)` check already owns mid-job halting."""
    queue = tmp_path / "queue"
    (queue / "incoming").mkdir(parents=True)
    _write_spec(queue, "a.json", _job_spec())
    _write_spec(queue, "b.json", _job_spec())

    state_dir = tmp_path / "state"
    state_dir.mkdir()

    class _HaltingForge(_FakeForge):
        def run(self, objective, tasks, *, cwd=".", budget=None):
            result = super().run(objective, tasks, cwd=cwd, budget=budget)
            # Simulate an operator halting the daemon right after job "a" finishes.
            (state_dir / "halts.json").write_text(
                json.dumps({WATCH_HALT_KEY: {"halted": True}}), encoding="utf-8"
            )
            return result

    forge = _HaltingForge([
        ForgeResult(job_id="j1", objective="x", accepted=1, rejected=0),
        ForgeResult(job_id="j2", objective="x", accepted=1, rejected=0),
    ])
    stats = watch_queue(queue, state_dir=state_dir, once=True, forge_factory=lambda: forge)

    assert stats.halted is True
    assert len(forge.calls) == 1  # only "a" ran; "b" was never started
    assert stats.jobs_done == 1
    assert (queue / "done" / "a.json").exists()
    assert (queue / "incoming" / "b.json").exists()


# -------------------------------------------------------------------------- cli


def test_cli_watch_subcommand_reaches_watch_queue_and_reports_its_stats(tmp_path, monkeypatch, capsys):
    import importlib

    from forgeos import __main__ as cli

    # `forgeos/__init__.py`'s pre-existing `from .watch import watch` shadows
    # the `forgeos.watch` PACKAGE ATTRIBUTE with that function (`import X.Y`
    # binds `Y` by attribute traversal on the already-imported `X`, not via
    # `sys.modules`, so even `import forgeos.watch as m` would return the
    # function here). `importlib.import_module` goes straight to
    # `sys.modules['forgeos.watch']`, sidestepping the shadowed attribute --
    # the same reason `cmd_watch`'s own `from forgeos.watch import
    # watch_queue` (a fresh `from`-import, not an attribute read off an
    # already-bound `forgeos` object) is unaffected by any of this.
    watch_module = importlib.import_module("forgeos.watch")
    monkeypatch.setattr(
        watch_module, "watch_queue",
        lambda queue, **kw: WatchStats(jobs_done=2, jobs_failed=1, halted=False),
    )
    rc = cli.main(["watch", "--queue", str(tmp_path / "q"), "--once"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "2 done, 1 failed" in out


def test_cli_watch_returns_nonzero_for_an_ownership_conflict(tmp_path, monkeypatch, capsys):
    import importlib

    from forgeos import __main__ as cli

    watch_module = importlib.import_module("forgeos.watch")
    monkeypatch.setattr(
        watch_module, "watch_queue",
        lambda queue, **kw: WatchStats(ownership_conflict=True),
    )
    rc = cli.main(["watch", "--queue", str(tmp_path / "q"), "--once"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "another worker owns the queue" in out


def test_forge_cli_queue_status_forwards_machine_readable_monitor_state(tmp_path, capsys):
    from forgeos import cli

    _write_heartbeat(tmp_path, state="idle", current_job=None, stats=WatchStats())
    rc = cli.main(["queue-status", "--queue", str(tmp_path), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["heartbeat_valid"] is True
    assert payload["stale"] is False
