"""Tests for `hive/adapters/cli_team.py` (`OMCTeamAdapter`).

Entirely subprocess-driven and entirely mocked: `hive.adapters.cli_team._spawn`
is the exact seam this adapter uses to reach `node`, so every test substitutes
it directly (same isolation pattern as `tests/test_local_adapters.py`'s
`_spawn` seams for `jcode.py`/`ollama.py`, and `tests/test_adapters.py`'s
`_import_acp` seam for `acp.py`). No real `node` process is ever spawned, no
real omc team is ever started, no network call is ever made.

No `pytest-asyncio` in this environment (see `tests/test_adapters.py`), so
async adapter methods are driven with the same `run()`/`_collect()` helper.
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Any

import pytest

from hive.adapters import cli_team as cli_team_adapter
from hive.adapters.base import EventKind, WorkerAdapter, WorkerCapabilities, WorkerUsage
from hive.adapters.cli_team import OMCTeamAdapter


def run(coro):
    return asyncio.run(coro)


async def _collect(agen):
    out = []
    async for item in agen:
        out.append(item)
    return out


class FakeProcess:
    """Stands in for `asyncio.subprocess.Process`."""

    def __init__(self, returncode: int = 0, stdout: bytes = b"", stderr: bytes = b"") -> None:
        self.returncode: int | None = None
        self._final_returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    async def communicate(self) -> tuple[bytes, bytes]:
        self.returncode = self._final_returncode
        return self._stdout, self._stderr


def make_spawn(*processes: FakeProcess):
    """A fake `_spawn` that returns `processes` in order, one per call, and
    records every `cmd` list it was invoked with."""
    captured: list[list[str]] = []
    queue = list(processes)

    async def spawn(cmd: list[str]) -> FakeProcess:
        captured.append(cmd)
        return queue.pop(0) if queue else processes[-1]

    return spawn, captured


FAKE_BIN_PATH = str(Path("/fake/omc/bin/oh-my-claudecode.js"))


def make_adapter(**kwargs: Any) -> OMCTeamAdapter:
    kwargs.setdefault("bin_path", FAKE_BIN_PATH)
    kwargs.setdefault("poll_interval_seconds", 0.0)  # tests never actually sleep 2s
    return OMCTeamAdapter(**kwargs)


def start_result_json(job_id: str = "omc-abc123", pid: int = 4242) -> bytes:
    return f'{{"jobId": "{job_id}", "status": "running", "pid": {pid}}}'.encode()


def status_json(job_id: str = "omc-abc123", status: str = "completed", result: Any = "all done", **extra) -> bytes:
    payload = {"jobId": job_id, "status": status, "elapsedSeconds": "1.2", "result": result, **extra}
    import json as _json

    return _json.dumps(payload).encode()


# ============================================================== contract


def test_adapter_is_a_concrete_worker_adapter():
    """Instantiating at all proves every abstract method in `base.py` is
    implemented — Python's ABC machinery refuses to construct a subclass
    that leaves one out."""
    adapter = make_adapter()
    assert isinstance(adapter, WorkerAdapter)
    ok, reason = adapter.health()
    assert isinstance(ok, bool)
    assert isinstance(reason, str)
    assert isinstance(adapter.capabilities(), WorkerCapabilities)


def test_every_abstract_method_is_implemented_and_async_where_required():
    for name in ("start", "send", "cancel", "checkpoint", "resume", "usage", "close"):
        method = getattr(OMCTeamAdapter, name)
        assert inspect.iscoroutinefunction(method) or inspect.isasyncgenfunction(method), name
    assert callable(OMCTeamAdapter.health)
    assert callable(OMCTeamAdapter.capabilities)


# ================================================================= health


def test_health_false_when_node_not_on_path(monkeypatch):
    monkeypatch.setattr(cli_team_adapter.shutil, "which", lambda cmd: None)
    adapter = make_adapter()

    ok, reason = adapter.health()

    assert ok is False
    assert "node" in reason
    assert "not found on PATH" in reason


def test_health_false_when_plugin_bin_path_is_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_team_adapter.shutil, "which", lambda cmd: f"/usr/bin/{cmd}")
    missing = tmp_path / "does" / "not" / "exist" / "oh-my-claudecode.js"
    adapter = make_adapter(bin_path=str(missing))

    ok, reason = adapter.health()

    assert ok is False
    assert str(missing) in reason
    assert "not found" in reason


def test_health_true_when_node_and_bin_path_are_present(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_team_adapter.shutil, "which", lambda cmd: f"/usr/bin/{cmd}")
    bin_path = tmp_path / "oh-my-claudecode.js"
    bin_path.write_text("// fake plugin entrypoint", encoding="utf-8")
    adapter = make_adapter(bin_path=str(bin_path))

    ok, reason = adapter.health()

    assert ok is True
    assert str(bin_path) in reason


def test_health_never_raises_when_bin_path_is_unreadable(monkeypatch):
    """health() must never crash the harness (base.py contract) even if a
    weird bin_path value cannot be stat()'d cleanly."""
    monkeypatch.setattr(cli_team_adapter.shutil, "which", lambda cmd: f"/usr/bin/{cmd}")
    adapter = make_adapter(bin_path="\0invalid\0path")

    ok, reason = adapter.health()  # must not raise

    assert ok is False
    assert isinstance(reason, str)


# ============================================================ capabilities


def test_capabilities_shape_is_conservative_about_what_omc_cannot_report():
    caps = make_adapter().capabilities()
    assert caps.token_usage is False
    assert caps.tool_calls is False
    assert caps.structured_messages is False
    assert caps.file_diffs is True  # isolated git worktrees per worker
    assert caps.resumable_sessions is True
    assert caps.max_parallel_sessions == 4


def test_capabilities_max_parallel_sessions_is_configurable():
    caps = make_adapter(max_parallel_sessions=9).capabilities()
    assert caps.max_parallel_sessions == 9


# ======================================================== agentTypes: caller


def test_start_agent_type_comes_from_caller_model_profile_not_hardcoded(monkeypatch):
    """The router picks agentTypes; the adapter must forward whatever it is
    given as `--agent`, never substitute its own opinion."""
    spawn, captured = make_spawn(FakeProcess(returncode=0, stdout=start_result_json()))
    monkeypatch.setattr(cli_team_adapter, "_spawn", spawn)
    adapter = make_adapter()
    session_id = run(adapter.start("task-1", "/repo", "architect"))

    run(_collect(_first_event_only(adapter, session_id)))

    assert "--agent" in captured[0]
    idx = captured[0].index("--agent")
    assert captured[0][idx + 1] == "architect"


def test_start_agent_type_differs_across_two_sessions_when_caller_differs(monkeypatch):
    spawn1, captured1 = make_spawn(FakeProcess(returncode=0, stdout=start_result_json(job_id="omc-aaa11111")))
    monkeypatch.setattr(cli_team_adapter, "_spawn", spawn1)
    adapter = make_adapter()
    session_a = run(adapter.start("task-a", "/repo", "explore"))
    run(_collect(_first_event_only(adapter, session_a)))
    assert captured1[0][captured1[0].index("--agent") + 1] == "explore"

    spawn2, captured2 = make_spawn(FakeProcess(returncode=0, stdout=start_result_json(job_id="omc-bbb22222")))
    monkeypatch.setattr(cli_team_adapter, "_spawn", spawn2)
    session_b = run(adapter.start("task-b", "/repo", "critic"))
    run(_collect(_first_event_only(adapter, session_b)))
    assert captured2[0][captured2[0].index("--agent") + 1] == "critic"


async def _first_event_only(adapter: OMCTeamAdapter, session_id: str):
    agen = adapter.send(session_id, "do the task")
    yield await agen.__anext__()
    await agen.aclose()


def test_start_falls_back_to_default_agent_type_only_when_model_profile_is_empty(monkeypatch):
    spawn, captured = make_spawn(FakeProcess(returncode=0, stdout=start_result_json()))
    monkeypatch.setattr(cli_team_adapter, "_spawn", spawn)
    adapter = make_adapter(default_agent_type="codex")
    session_id = run(adapter.start("task-1", "/repo", ""))

    run(_collect(_first_event_only(adapter, session_id)))

    assert captured[0][captured[0].index("--agent") + 1] == "codex"


# ==================================================================== send


def test_send_starts_job_then_polls_status_to_completion(monkeypatch):
    start_proc = FakeProcess(returncode=0, stdout=start_result_json(job_id="omc-xyz99999"))
    status_proc = FakeProcess(returncode=0, stdout=status_json(job_id="omc-xyz99999", status="completed", result="ship it"))
    spawn, captured = make_spawn(start_proc, status_proc)
    monkeypatch.setattr(cli_team_adapter, "_spawn", spawn)
    adapter = make_adapter()
    session_id = run(adapter.start("task-1", "/repo", "executor"))

    events = run(_collect(adapter.send(session_id, "fix the failing tests")))

    assert captured[0][1:4] == [FAKE_BIN_PATH, "team", "start"]
    assert "--task" in captured[0]
    assert captured[0][captured[0].index("--task") + 1] == "fix the failing tests"
    assert captured[1][3:5] == ["status", "omc-xyz99999"]

    kinds = [e.kind for e in events]
    assert EventKind.META in kinds
    assert kinds[-1] is EventKind.DONE
    assert events[-1].status == "completed"
    message = next(e for e in events if e.kind is EventKind.MESSAGE)
    assert message.text == "ship it"


def test_send_polls_multiple_times_while_running_before_completing(monkeypatch):
    start_proc = FakeProcess(returncode=0, stdout=start_result_json(job_id="omc-poll00001"))
    running_proc_1 = FakeProcess(returncode=0, stdout=status_json(job_id="omc-poll00001", status="running"))
    running_proc_2 = FakeProcess(returncode=0, stdout=status_json(job_id="omc-poll00001", status="running"))
    done_proc = FakeProcess(returncode=0, stdout=status_json(job_id="omc-poll00001", status="completed", result="done"))
    spawn, captured = make_spawn(start_proc, running_proc_1, running_proc_2, done_proc)
    monkeypatch.setattr(cli_team_adapter, "_spawn", spawn)
    adapter = make_adapter(poll_interval_seconds=0.0)
    session_id = run(adapter.start("task-1", "/repo", "executor"))

    events = run(_collect(adapter.send(session_id, "long task")))

    # start + 3 status polls (2 running, 1 terminal)
    assert len(captured) == 4
    # "elapsedSeconds" only appears on poll-loop progress pings, not on the
    # initial "start" META event, so this counts the two running polls only.
    poll_events = [e for e in events if e.kind is EventKind.META and "elapsedSeconds" in e.data]
    assert len(poll_events) == 2
    assert events[-1].kind is EventKind.DONE
    assert events[-1].status == "completed"


def test_send_yields_error_event_when_job_fails(monkeypatch):
    start_proc = FakeProcess(returncode=0, stdout=start_result_json(job_id="omc-fail00001"))
    failed_proc = FakeProcess(
        returncode=0,
        stdout=status_json(job_id="omc-fail00001", status="failed", result=None, stderr="worker crashed"),
    )
    spawn, _ = make_spawn(start_proc, failed_proc)
    monkeypatch.setattr(cli_team_adapter, "_spawn", spawn)
    adapter = make_adapter()
    session_id = run(adapter.start("task-1", "/repo", "debugger"))

    events = run(_collect(adapter.send(session_id, "reproduce the bug")))

    error_event = next(e for e in events if e.kind is EventKind.ERROR)
    assert "worker crashed" in error_event.text
    assert events[-1].kind is EventKind.DONE
    assert events[-1].status == "failed"


def test_send_yields_error_event_instead_of_raising_when_start_exits_nonzero(monkeypatch):
    start_proc = FakeProcess(returncode=1, stdout=b"", stderr=b"Unsupported agent type: architect")
    spawn, _ = make_spawn(start_proc)
    monkeypatch.setattr(cli_team_adapter, "_spawn", spawn)
    adapter = make_adapter()
    session_id = run(adapter.start("task-1", "/repo", "architect"))

    events = run(_collect(adapter.send(session_id, "do it")))

    assert len(events) == 1
    assert events[0].kind is EventKind.ERROR
    assert "Unsupported agent type" in events[0].text


def test_send_yields_error_event_when_spawn_itself_raises(monkeypatch):
    async def boom(cmd: list[str]) -> Any:
        raise OSError("node vanished mid-race")

    monkeypatch.setattr(cli_team_adapter, "_spawn", boom)
    adapter = make_adapter()
    session_id = run(adapter.start("task-1", "/repo", "executor"))

    events = run(_collect(adapter.send(session_id, "do it")))

    assert len(events) == 1
    assert events[0].kind is EventKind.ERROR


def test_send_yields_error_event_when_start_output_is_not_parseable_json(monkeypatch):
    start_proc = FakeProcess(returncode=0, stdout=b"not json at all")
    spawn, _ = make_spawn(start_proc)
    monkeypatch.setattr(cli_team_adapter, "_spawn", spawn)
    adapter = make_adapter()
    session_id = run(adapter.start("task-1", "/repo", "executor"))

    events = run(_collect(adapter.send(session_id, "do it")))

    assert len(events) == 1
    assert events[0].kind is EventKind.ERROR


def test_second_send_on_an_already_started_job_repolls_instead_of_restarting(monkeypatch):
    start_proc = FakeProcess(returncode=0, stdout=start_result_json(job_id="omc-again0001"))
    # Terminal on the very first poll: the point of this test is what the
    # *second* send() does once job_id is already set, not how many polls
    # the first one takes.
    status_proc_1 = FakeProcess(returncode=0, stdout=status_json(job_id="omc-again0001", status="completed", result="first result"))
    spawn1, captured1 = make_spawn(start_proc, status_proc_1)
    monkeypatch.setattr(cli_team_adapter, "_spawn", spawn1)
    adapter = make_adapter()
    session_id = run(adapter.start("task-1", "/repo", "executor"))
    run(_collect(adapter.send(session_id, "first prompt")))

    status_proc_2 = FakeProcess(returncode=0, stdout=status_json(job_id="omc-again0001", status="completed", result="ok"))
    spawn2, captured2 = make_spawn(status_proc_2)
    monkeypatch.setattr(cli_team_adapter, "_spawn", spawn2)
    events = run(_collect(adapter.send(session_id, "a different second prompt")))

    # never a second "team start" — only ever "team status"
    assert all(cmd[3] == "status" for cmd in captured2)
    note_event = next(e for e in events if e.kind is EventKind.META and "note" in e.data)
    assert "one-shot" in note_event.data["note"]


# =================================================================== cancel


def test_cancel_marks_session_cancelled_and_invokes_cleanup(monkeypatch):
    start_proc = FakeProcess(returncode=0, stdout=start_result_json(job_id="omc-cancel0001"))
    spawn1, _ = make_spawn(start_proc)
    monkeypatch.setattr(cli_team_adapter, "_spawn", spawn1)
    adapter = make_adapter()
    session_id = run(adapter.start("task-1", "/repo", "executor"))
    run(_collect(_first_event_only(adapter, session_id)))

    cleanup_proc = FakeProcess(returncode=0, stdout=b'{"jobId": "omc-cancel0001", "message": "cleaned up"}')
    spawn2, captured2 = make_spawn(cleanup_proc)
    monkeypatch.setattr(cli_team_adapter, "_spawn", spawn2)

    run(adapter.cancel(session_id))

    assert adapter._sessions[session_id].cancelled is True
    assert captured2[0][3:5] == ["cleanup", "omc-cancel0001"]


def test_cancel_is_a_no_op_on_the_cli_when_job_never_started(monkeypatch):
    adapter = make_adapter()
    session_id = run(adapter.start("task-1", "/repo", "executor"))

    def must_not_be_called(cmd: list[str]) -> Any:
        raise AssertionError("must not shell out when no job was ever started")

    monkeypatch.setattr(cli_team_adapter, "_spawn", must_not_be_called)

    run(adapter.cancel(session_id))  # must not raise, must not spawn

    assert adapter._sessions[session_id].cancelled is True


def test_poll_loop_stops_promptly_once_cancelled(monkeypatch):
    start_proc = FakeProcess(returncode=0, stdout=start_result_json(job_id="omc-stop000001"))
    spawn, _ = make_spawn(start_proc)
    monkeypatch.setattr(cli_team_adapter, "_spawn", spawn)
    adapter = make_adapter()
    session_id = run(adapter.start("task-1", "/repo", "executor"))

    async def scenario():
        agen = adapter.send(session_id, "long running task")
        first = await agen.__anext__()  # the META "start" event
        # Setting the flag directly (rather than calling cancel()) isolates
        # what this test is about: the poll loop's own cancellation check,
        # not cancel()'s separate cleanup-CLI call (covered above).
        adapter._sessions[session_id].cancelled = True

        rest = [first]
        async for event in agen:
            rest.append(event)
        return rest

    events = run(scenario())
    assert events[-1].kind is EventKind.DONE
    assert events[-1].status == "cancelled"


# =================================================================== usage


def test_usage_marks_estimates_exact_false_before_any_send():
    adapter = make_adapter()
    session_id = run(adapter.start("task-1", "/repo", "executor"))

    usage = run(adapter.usage(session_id))

    assert usage == WorkerUsage()
    assert usage.exact is False


def test_usage_stays_exact_false_after_a_completed_job(monkeypatch):
    start_proc = FakeProcess(returncode=0, stdout=start_result_json(job_id="omc-usage00001"))
    status_proc = FakeProcess(returncode=0, stdout=status_json(job_id="omc-usage00001", status="completed", result="ok"))
    spawn, _ = make_spawn(start_proc, status_proc)
    monkeypatch.setattr(cli_team_adapter, "_spawn", spawn)
    adapter = make_adapter()
    session_id = run(adapter.start("task-1", "/repo", "executor"))
    run(_collect(adapter.send(session_id, "do it")))

    usage = run(adapter.usage(session_id))

    assert usage.exact is False
    assert usage.usd_micros == 0


# =============================================================== checkpoint


def test_checkpoint_has_no_transcript_shaped_keys(monkeypatch):
    start_proc = FakeProcess(returncode=0, stdout=start_result_json(job_id="omc-ckpt00001"))
    spawn, _ = make_spawn(start_proc)
    monkeypatch.setattr(cli_team_adapter, "_spawn", spawn)
    adapter = make_adapter()
    session_id = run(adapter.start("task-1", "/repo", "executor"))
    run(_collect(_first_event_only(adapter, session_id)))

    checkpoint = run(adapter.checkpoint(session_id))

    forbidden_substrings = ("message", "transcript", "history", "conversation", "chat", "prompt")
    for key in checkpoint:
        lowered = key.lower()
        assert not any(bad in lowered for bad in forbidden_substrings), key
    assert checkpoint["task_id"] == "task-1"
    assert checkpoint["cwd"] == "/repo"
    assert checkpoint["agent_type"] == "executor"
    assert checkpoint["job_id"] == "omc-ckpt00001"


def test_resume_restores_state_and_a_later_send_polls_the_same_job(monkeypatch):
    start_proc = FakeProcess(returncode=0, stdout=start_result_json(job_id="omc-resume0001"))
    spawn1, _ = make_spawn(start_proc)
    monkeypatch.setattr(cli_team_adapter, "_spawn", spawn1)
    adapter = make_adapter()
    session_id = run(adapter.start("task-1", "/repo", "executor"))
    run(_collect(_first_event_only(adapter, session_id)))
    checkpoint = run(adapter.checkpoint(session_id))
    run(adapter.close(session_id))  # consumes spawn1's one remaining call (its own cleanup)

    spawn2, captured2 = make_spawn(FakeProcess(returncode=0, stdout=status_json(job_id="omc-resume0001", status="completed", result="ok")))
    monkeypatch.setattr(cli_team_adapter, "_spawn", spawn2)

    resumed_id = run(adapter.resume(checkpoint))
    assert resumed_id != session_id  # a fresh hive-side session_id, honestly

    events = run(_collect(adapter.send(resumed_id, "continue")))

    assert captured2[0][3:5] == ["status", "omc-resume0001"]
    assert events[-1].kind is EventKind.DONE


# ===================================================================== close


def test_close_invokes_cleanup_when_a_job_was_started(monkeypatch):
    start_proc = FakeProcess(returncode=0, stdout=start_result_json(job_id="omc-close00001"))
    spawn1, _ = make_spawn(start_proc)
    monkeypatch.setattr(cli_team_adapter, "_spawn", spawn1)
    adapter = make_adapter()
    session_id = run(adapter.start("task-1", "/repo", "executor"))
    run(_collect(_first_event_only(adapter, session_id)))

    cleanup_proc = FakeProcess(returncode=0, stdout=b'{"jobId": "omc-close00001", "message": "cleaned up"}')
    spawn2, captured2 = make_spawn(cleanup_proc)
    monkeypatch.setattr(cli_team_adapter, "_spawn", spawn2)

    run(adapter.close(session_id))

    assert captured2[0][3:5] == ["cleanup", "omc-close00001"]
    assert session_id not in adapter._sessions


def test_close_does_not_shell_out_when_no_job_was_ever_started(monkeypatch):
    adapter = make_adapter()
    session_id = run(adapter.start("task-1", "/repo", "executor"))

    def must_not_be_called(cmd: list[str]) -> Any:
        raise AssertionError("must not shell out when no job exists to clean up")

    monkeypatch.setattr(cli_team_adapter, "_spawn", must_not_be_called)

    run(adapter.close(session_id))  # must not raise, must not spawn

    assert session_id not in adapter._sessions


def test_close_is_idempotent():
    adapter = make_adapter()
    session_id = run(adapter.start("task-1", "/repo", "executor"))

    run(adapter.close(session_id))
    assert session_id not in adapter._sessions
    run(adapter.close(session_id))  # closing twice must not raise

    with pytest.raises(KeyError):
        run(adapter.usage(session_id))
