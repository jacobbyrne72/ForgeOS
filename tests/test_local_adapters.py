"""Tests for the local/free worker adapter `ollama.py`.

The backend is entirely subprocess-driven and the `ollama` binary is never
invoked for real: `forgeos.adapters.ollama._spawn` and
`forgeos.adapters.ollama.subprocess.run` are the exact seams the adapter uses to
reach a subprocess, so tests substitute both directly (same isolation pattern
as `tests/test_adapters.py`'s `_import_acp` seam for `acp.py`). No real process
is ever spawned, no model is ever pulled, no network call is ever made.

No `pytest-asyncio` in this environment (see `tests/test_adapters.py`), so
async adapter methods are driven with the same `run()`/`_collect()` helper.
"""

from __future__ import annotations

import asyncio
import subprocess
from typing import Any

import pytest

from forgeos.adapters import ollama as ollama_adapter
from forgeos.adapters.base import EventKind, WorkerAdapter, WorkerCapabilities, WorkerUsage
from forgeos.adapters.ollama import OllamaAdapter


def run(coro):
    return asyncio.run(coro)


async def _collect(agen):
    out = []
    async for item in agen:
        out.append(item)
    return out


class FakeProcess:
    """Stands in for `asyncio.subprocess.Process`. `returncode` is `None`
    until `communicate()` completes, matching the real object's behaviour —
    the `cancel()` tests below depend on that to tell "still running" from
    "already finished"."""

    def __init__(self, returncode: int = 0, stdout: bytes = b"", stderr: bytes = b"") -> None:
        self.returncode: int | None = None
        self._final_returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self.terminated = False

    async def communicate(self) -> tuple[bytes, bytes]:
        self.returncode = self._final_returncode
        return self._stdout, self._stderr

    def terminate(self) -> None:
        self.terminated = True


def make_spawn(process: FakeProcess):
    captured: list[list[str]] = []

    async def spawn(cmd: list[str]) -> FakeProcess:
        captured.append(cmd)
        return process

    return spawn, captured


# ============================================================== contract


def test_adapter_is_a_concrete_worker_adapter():
    """Instantiating at all proves every abstract method in `base.py` is
    implemented — Python's ABC machinery refuses to construct a subclass
    that leaves one out."""
    adapter = OllamaAdapter()
    assert isinstance(adapter, WorkerAdapter)
    ok, reason = adapter.health()
    assert isinstance(ok, bool)
    assert isinstance(reason, str)
    assert isinstance(adapter.capabilities(), WorkerCapabilities)


# ================================================================= health


def test_ollama_health_false_when_binary_missing(monkeypatch):
    monkeypatch.setattr(ollama_adapter.shutil, "which", lambda cmd: None)

    def must_not_be_called(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("must not shell out when the binary itself is missing")

    monkeypatch.setattr(ollama_adapter.subprocess, "run", must_not_be_called)
    adapter = OllamaAdapter()

    ok, reason = adapter.health()

    assert ok is False
    assert "not found on PATH" in reason


def test_ollama_health_false_when_daemon_unreachable(monkeypatch):
    monkeypatch.setattr(ollama_adapter.shutil, "which", lambda cmd: "/usr/bin/ollama")

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(
            cmd, returncode=1, stdout="", stderr="could not connect to ollama app, is it running?"
        )

    monkeypatch.setattr(ollama_adapter.subprocess, "run", fake_run)
    adapter = OllamaAdapter()

    ok, reason = adapter.health()

    assert ok is False
    assert "not reachable" in reason
    assert "running" in reason


def test_ollama_health_false_when_model_not_pulled(monkeypatch):
    monkeypatch.setattr(ollama_adapter.shutil, "which", lambda cmd: "/usr/bin/ollama")

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        listing = "NAME             ID        SIZE    MODIFIED\nllama3:latest    abc123    4.7 GB  2 days ago\n"
        return subprocess.CompletedProcess(cmd, returncode=0, stdout=listing, stderr="")

    monkeypatch.setattr(ollama_adapter.subprocess, "run", fake_run)
    adapter = OllamaAdapter(model="llava-phi3:3.8b")

    ok, reason = adapter.health()

    assert ok is False
    assert "not pulled" in reason
    assert "ollama pull llava-phi3:3.8b" in reason


def test_ollama_health_never_auto_pulls(monkeypatch):
    """Health checking must never itself trigger a multi-GB download."""
    monkeypatch.setattr(ollama_adapter.shutil, "which", lambda cmd: "/usr/bin/ollama")
    seen_subcommands: list[str] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        seen_subcommands.append(cmd[1])
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ollama_adapter.subprocess, "run", fake_run)
    adapter = OllamaAdapter(model="never-pulled:1b")

    adapter.health()

    assert "pull" not in seen_subcommands


def test_ollama_health_true_when_daemon_up_and_model_present(monkeypatch):
    monkeypatch.setattr(ollama_adapter.shutil, "which", lambda cmd: "/usr/bin/ollama")

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        listing = "NAME               ID        SIZE    MODIFIED\nllava-phi3:3.8b    xyz789    2.9 GB  1 day ago\n"
        return subprocess.CompletedProcess(cmd, returncode=0, stdout=listing, stderr="")

    monkeypatch.setattr(ollama_adapter.subprocess, "run", fake_run)
    adapter = OllamaAdapter(model="llava-phi3:3.8b")

    ok, reason = adapter.health()

    assert ok is True
    assert "llava-phi3:3.8b" in reason


def test_ollama_health_never_raises_on_a_subprocess_error(monkeypatch):
    monkeypatch.setattr(ollama_adapter.shutil, "which", lambda cmd: "/usr/bin/ollama")

    def boom(cmd: list[str], **kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd, 10)

    monkeypatch.setattr(ollama_adapter.subprocess, "run", boom)
    adapter = OllamaAdapter()

    ok, reason = adapter.health()  # must not raise

    assert ok is False
    assert reason


# ========================================================== capabilities


def test_ollama_capabilities_declares_it_cannot_edit_files():
    caps = OllamaAdapter().capabilities()
    assert caps.file_diffs is False
    assert caps.tool_calls is False
    assert caps.resumable_sessions is False
    assert caps.token_usage is False


# ============================================================ ollama: send


def test_ollama_send_runs_model_and_prompt_and_reports_zero_cost(monkeypatch):
    process = FakeProcess(returncode=0, stdout=b"a friendly local summary")
    spawn, captured = make_spawn(process)
    monkeypatch.setattr(ollama_adapter, "_spawn", spawn)
    adapter = OllamaAdapter(model="llava-phi3:3.8b")
    session_id = run(adapter.start("task-1", "/repo", ""))

    events = run(_collect(adapter.send(session_id, "summarize this log")))

    assert captured[0][:2] == ["ollama", "run"]
    assert "llava-phi3:3.8b" in captured[0]
    assert "summarize this log" in captured[0]

    message = next(e for e in events if e.kind is EventKind.MESSAGE)
    assert message.text == "a friendly local summary"
    assert events[-1].kind is EventKind.DONE

    usage = run(adapter.usage(session_id))
    assert usage.usd_micros == 0
    assert usage.exact is True  # local inference: genuinely zero dollar cost


def test_ollama_send_yields_error_event_instead_of_raising_on_nonzero_exit(monkeypatch):
    process = FakeProcess(returncode=1, stdout=b"", stderr=b"Error: model requires more system memory")
    spawn, _ = make_spawn(process)
    monkeypatch.setattr(ollama_adapter, "_spawn", spawn)
    adapter = OllamaAdapter()
    session_id = run(adapter.start("task-1", "/repo", ""))

    events = run(_collect(adapter.send(session_id, "hi")))

    assert len(events) == 1
    assert events[0].kind is EventKind.ERROR
    assert "more system memory" in events[0].text


def test_ollama_send_yields_error_event_when_spawn_itself_raises(monkeypatch):
    async def boom(cmd: list[str]) -> Any:
        raise OSError("ollama vanished mid-race")

    monkeypatch.setattr(ollama_adapter, "_spawn", boom)
    adapter = OllamaAdapter()
    session_id = run(adapter.start("task-1", "/repo", ""))

    events = run(_collect(adapter.send(session_id, "hi")))

    assert len(events) == 1
    assert events[0].kind is EventKind.ERROR


def test_ollama_usage_default_is_not_exact_before_any_send():
    adapter = OllamaAdapter()
    session_id = run(adapter.start("task-1", "/repo", ""))

    usage = run(adapter.usage(session_id))

    assert usage == WorkerUsage()
    assert usage.exact is False


def test_ollama_cancel_terminates_a_tracked_in_flight_process():
    adapter = OllamaAdapter()
    session_id = run(adapter.start("task-1", "/repo", ""))
    process = FakeProcess()  # returncode still None: "in flight"
    adapter._sessions[session_id].current_process = process

    run(adapter.cancel(session_id))

    assert process.terminated is True


def test_ollama_cancel_is_a_no_op_when_nothing_is_in_flight():
    adapter = OllamaAdapter()
    session_id = run(adapter.start("task-1", "/repo", ""))

    run(adapter.cancel(session_id))  # must not raise


def test_ollama_checkpoint_has_no_transcript_shaped_keys():
    adapter = OllamaAdapter(model="llava-phi3:3.8b")
    session_id = run(adapter.start("task-1", "/repo", ""))

    checkpoint = run(adapter.checkpoint(session_id))

    forbidden_substrings = ("message", "transcript", "history", "conversation", "chat", "prompt")
    for key in checkpoint:
        lowered = key.lower()
        assert not any(bad in lowered for bad in forbidden_substrings), key
    assert checkpoint["task_id"] == "task-1"
    assert checkpoint["model"] == "llava-phi3:3.8b"


def test_ollama_resume_returns_a_fresh_session_honestly():
    adapter = OllamaAdapter()
    session_id = run(adapter.start("task-1", "/repo", ""))
    checkpoint = run(adapter.checkpoint(session_id))
    run(adapter.close(session_id))

    resumed_id = run(adapter.resume(checkpoint))

    assert resumed_id != session_id  # no backend session identity ever existed to carry over
    assert resumed_id in adapter._sessions


def test_ollama_close_is_idempotent_and_drops_the_session():
    adapter = OllamaAdapter()
    session_id = run(adapter.start("task-1", "/repo", ""))

    run(adapter.close(session_id))
    assert session_id not in adapter._sessions
    run(adapter.close(session_id))  # closing twice must not raise

    with pytest.raises(KeyError):
        run(adapter.usage(session_id))
