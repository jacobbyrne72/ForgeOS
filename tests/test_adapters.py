"""Adapter tests: the worker contract (base.py) and the ACP backend (acp.py).

The ACP SDK is always mocked here — no network, no real CLI spawn, no real
subprocess. `forgeos.adapters.acp._import_acp` is the one seam the adapter uses
to reach the SDK, so replacing it with a fake namespace (or a function that
raises `ImportError`) is enough to exercise every path without the vendor
package installed.

No `pytest-asyncio` in this environment (see pyproject.toml's `dev` extra),
so async adapter methods are driven with a small `run()`/`_collect()` helper
instead of `async def test_...`.
"""

from __future__ import annotations

import asyncio
import contextlib
import types
from dataclasses import dataclass, field
from typing import Any

import pytest

from forgeos.adapters import acp as acp_adapter
from forgeos.adapters.base import (
    EventKind,
    WorkerAdapter,
    WorkerCapabilities,
    WorkerEvent,
    WorkerUsage,
)
from forgeos.contracts import to_micros

ACPAdapter = acp_adapter.ACPAdapter


def run(coro):
    return asyncio.run(coro)


async def _collect(agen):
    out = []
    async for item in agen:
        out.append(item)
    return out


# ---------------------------------------------------------------------- base


def test_worker_capabilities_defaults_are_conservative():
    caps = WorkerCapabilities()
    assert caps.structured_messages is False
    assert caps.tool_calls is False
    assert caps.file_diffs is False
    assert caps.max_parallel_sessions == 1


def test_worker_usage_default_is_not_exact():
    usage = WorkerUsage()
    assert usage.exact is False
    assert usage.usd_micros == 0


def test_worker_adapter_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        WorkerAdapter()  # type: ignore[abstract]


def test_worker_adapter_requires_every_abstract_method():
    class Partial(WorkerAdapter):
        def health(self):
            return True, "ok"

        def capabilities(self):
            return WorkerCapabilities()

        async def start(self, task_id, cwd, model_profile):
            return "s"

        # send, cancel, checkpoint, resume, usage, close all missing

    with pytest.raises(TypeError):
        Partial()  # type: ignore[abstract]


class _DummyAdapter(WorkerAdapter):
    """Minimal concrete implementation, used only to prove the contract is
    satisfiable end to end (every abstract method implemented, sane types)."""

    def health(self) -> tuple[bool, str]:
        return True, "ok"

    def capabilities(self) -> WorkerCapabilities:
        return WorkerCapabilities()

    async def start(self, task_id: str, cwd: str, model_profile: str) -> str:
        return "sess-1"

    async def send(self, session_id: str, prompt: str):
        yield WorkerEvent(kind=EventKind.DONE, status="end_turn")

    async def cancel(self, session_id: str) -> None:
        return None

    async def checkpoint(self, session_id: str) -> dict:
        return {"task_id": "t"}

    async def resume(self, checkpoint: dict) -> str:
        return "sess-2"

    async def usage(self, session_id: str) -> WorkerUsage:
        return WorkerUsage()

    async def close(self, session_id: str) -> None:
        return None


def test_full_dummy_implementation_satisfies_the_contract():
    adapter = _DummyAdapter()
    assert adapter.health() == (True, "ok")
    session_id = run(adapter.start("t", ".", ""))
    events = run(_collect(adapter.send(session_id, "hi")))
    assert events[0].kind is EventKind.DONE
    assert run(adapter.usage(session_id)).exact is False


# ----------------------------------------------------------------- fakes

# Fake ACP schema objects. Field names mirror vendor/acp-python-sdk/src/acp/schema.py
# exactly (verified by reading the source) so a mismatch here would mean the
# adapter is reading fields no real update carries.


@dataclass
class FakeText:
    text: str
    type: str = "text"


@dataclass
class FakeAgentMessageChunk:
    content: Any
    session_update: str = "agent_message_chunk"


@dataclass
class FakeToolCallStart:
    tool_call_id: str
    title: str
    kind: str
    status: str
    session_update: str = "tool_call"


@dataclass
class FakeCost:
    amount: float
    currency: str


@dataclass
class FakeUsageUpdate:
    used: int
    size: int
    cost: Any = None
    session_update: str = "usage_update"


@dataclass
class FakeTurnUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int = 0


@dataclass
class FakePromptResponse:
    stop_reason: str
    usage: Any = None


@dataclass
class FakeAgentCapabilities:
    load_session: bool = False


@dataclass
class FakeInitializeResponse:
    agent_capabilities: FakeAgentCapabilities = field(default_factory=FakeAgentCapabilities)


@dataclass
class FakeNewSessionResponse:
    session_id: str


class FakePermissionOption:
    def __init__(self, option_id: str, kind: str) -> None:
        self.option_id = option_id
        self.kind = kind


class FakeConnection:
    """Records every call it receives; `prompt_script` scripts what `prompt()`
    does — a mix of session_update payloads, ("sleep",) markers that force a
    genuine event-loop suspension, and one ("response", PromptResponse)."""

    def __init__(self, client: Any, supports_load_session: bool = False) -> None:
        self.client = client
        self.supports_load_session = supports_load_session
        self.calls: list[tuple[str, dict]] = []
        self.next_session_id = "acp-sess-1"
        self.prompt_script: list[Any] = [("response", FakePromptResponse(stop_reason="end_turn"))]

    async def initialize(self, **kw: Any) -> FakeInitializeResponse:
        self.calls.append(("initialize", kw))
        return FakeInitializeResponse(agent_capabilities=FakeAgentCapabilities(self.supports_load_session))

    async def new_session(self, **kw: Any) -> FakeNewSessionResponse:
        self.calls.append(("new_session", kw))
        return FakeNewSessionResponse(session_id=self.next_session_id)

    async def load_session(self, **kw: Any) -> None:
        self.calls.append(("load_session", kw))
        return None

    async def prompt(self, **kw: Any) -> Any:
        self.calls.append(("prompt", kw))
        response = None
        for item in self.prompt_script:
            if isinstance(item, tuple) and item and item[0] == "sleep":
                await asyncio.sleep(0)
            elif isinstance(item, tuple) and item and item[0] == "response":
                response = item[1]
            else:
                await self.client.session_update(kw["session_id"], item)
        return response

    async def cancel(self, **kw: Any) -> None:
        self.calls.append(("cancel", kw))

    async def close_session(self, **kw: Any) -> None:
        self.calls.append(("close_session", kw))


def make_conn_factory(supports_load_session: bool):
    """Each spawn gets a fresh FakeConnection with its own session id, so a
    resumed session that falls back to `new_session` is provably a *different*
    session, not an accidental id collision with the original."""
    counter = {"n": 0}

    def factory(client: Any) -> FakeConnection:
        counter["n"] += 1
        conn = FakeConnection(client, supports_load_session=supports_load_session)
        conn.next_session_id = f"acp-sess-{counter['n']}"
        return conn

    return factory


def make_fake_acp(conn_factory):
    """A minimal fake `acp` module. Every attribute the adapter actually
    touches is verified present in vendor/acp-python-sdk/src/acp (see acp.py's
    module docstring) — `ClientCapabilities`/`Implementation`/`AllowedOutcome`/
    `DeniedOutcome` under `.schema`, everything else top-level."""

    captured_spawn_kwargs: list[dict] = []

    @contextlib.asynccontextmanager
    async def spawn_agent_process(client: Any, command: str, *args: str, **kwargs: Any):
        captured_spawn_kwargs.append({"command": command, "args": args, **kwargs})
        conn = conn_factory(client)
        process = object()
        yield conn, process

    class FakeRequestError(Exception):
        @classmethod
        def method_not_found(cls, method: str) -> "FakeRequestError":
            return cls(f"method not found: {method}")

    fake_schema = types.SimpleNamespace(
        ClientCapabilities=lambda **kw: types.SimpleNamespace(**kw),
        Implementation=lambda **kw: types.SimpleNamespace(**kw),
        AllowedOutcome=lambda **kw: types.SimpleNamespace(**kw),
        DeniedOutcome=lambda **kw: types.SimpleNamespace(**kw),
    )

    fake = types.SimpleNamespace(
        PROTOCOL_VERSION=1,
        text_block=lambda text: FakeText(text=text),
        spawn_agent_process=spawn_agent_process,
        RequestError=FakeRequestError,
        RequestPermissionResponse=lambda **kw: types.SimpleNamespace(**kw),
        DeclineElicitationResponse=lambda **kw: types.SimpleNamespace(**kw),
        schema=fake_schema,
    )
    return fake, captured_spawn_kwargs


# ------------------------------------------------------------------- health


def test_acp_health_false_when_sdk_missing(monkeypatch):
    def boom():
        raise ImportError("no module named acp")

    monkeypatch.setattr(acp_adapter, "_import_acp", boom)
    adapter = ACPAdapter("claude-agent-acp")

    ok, reason = adapter.health()

    assert ok is False
    assert "acp SDK not installed" in reason


def test_acp_health_false_when_cli_not_on_path(monkeypatch):
    fake, _ = make_fake_acp(make_conn_factory(False))
    monkeypatch.setattr(acp_adapter, "_import_acp", lambda: fake)
    monkeypatch.setattr(acp_adapter.shutil, "which", lambda cmd: None)
    adapter = ACPAdapter("nonexistent-cli")

    ok, reason = adapter.health()

    assert ok is False
    assert "nonexistent-cli" in reason


def test_acp_health_true_when_sdk_and_cli_are_available(monkeypatch):
    fake, _ = make_fake_acp(make_conn_factory(False))
    monkeypatch.setattr(acp_adapter, "_import_acp", lambda: fake)
    monkeypatch.setattr(acp_adapter.shutil, "which", lambda cmd: f"/usr/bin/{cmd}")
    adapter = ACPAdapter("claude-agent-acp")

    assert adapter.health() == (True, "acp SDK importable and CLI on PATH")


def test_acp_capabilities_declares_the_contract_shape():
    adapter = ACPAdapter("claude-agent-acp", max_parallel_sessions=7)

    caps = adapter.capabilities()

    assert caps.tool_calls is True
    assert caps.file_diffs is True
    assert caps.resumable_sessions is True
    assert caps.token_usage is True
    assert caps.reasoning_effort is False
    assert caps.max_parallel_sessions == 7


# --------------------------------------------------------------- start/send


def test_acp_start_never_constructs_or_forwards_an_env(monkeypatch):
    fake, captured = make_fake_acp(make_conn_factory(False))
    monkeypatch.setattr(acp_adapter, "_import_acp", lambda: fake)
    monkeypatch.setenv("SUPER_SECRET_TOKEN", "must-never-appear-anywhere")
    adapter = ACPAdapter("claude-agent-acp")

    run(adapter.start("task-1", "/repo", ""))

    assert len(captured) == 1
    call = captured[0]
    assert call.get("env") is None
    assert "must-never-appear-anywhere" not in repr(call)


def test_acp_send_translates_message_and_tool_call_updates(monkeypatch):
    def conn_factory(client):
        conn = FakeConnection(client)
        conn.prompt_script = [
            FakeAgentMessageChunk(content=FakeText(text="hello there")),
            ("sleep",),
            FakeToolCallStart(tool_call_id="t1", title="Reading file", kind="read", status="in_progress"),
            ("sleep",),
            ("response", FakePromptResponse(stop_reason="end_turn", usage=FakeTurnUsage(input_tokens=10, output_tokens=5))),
        ]
        return conn

    fake, _ = make_fake_acp(conn_factory)
    monkeypatch.setattr(acp_adapter, "_import_acp", lambda: fake)
    adapter = ACPAdapter("claude-agent-acp")
    session_id = run(adapter.start("task-1", "/repo", "sonnet"))
    assert session_id == "acp-sess-1"

    events = run(_collect(adapter.send(session_id, "do the thing")))

    kinds = [e.kind for e in events]
    assert EventKind.MESSAGE in kinds
    assert EventKind.TOOL_CALL in kinds
    assert kinds[-1] is EventKind.DONE
    assert events[-1].status == "end_turn"

    message_event = next(e for e in events if e.kind is EventKind.MESSAGE)
    assert message_event.text == "hello there"

    tool_event = next(e for e in events if e.kind is EventKind.TOOL_CALL)
    assert tool_event.tool_call_id == "t1"
    assert tool_event.tool_kind == "read"
    assert tool_event.status == "in_progress"
    assert tool_event.text == "Reading file"


def test_acp_send_surfaces_a_file_edit_as_a_canonical_diff(monkeypatch):
    @dataclass
    class FakeDiff:
        path: str
        new_text: str
        old_text: str | None = None

    @dataclass
    class FakeToolCallUpdate:
        tool_call_id: str
        content: Any
        session_update: str = "tool_call_update"

    def conn_factory(client):
        conn = FakeConnection(client)
        conn.prompt_script = [
            FakeToolCallUpdate(
                tool_call_id="t1",
                content=[FakeDiff(path="forgeos/foo.py", old_text="a = 1\n", new_text="a = 2\n")],
            ),
            ("response", FakePromptResponse(stop_reason="end_turn")),
        ]
        return conn

    fake, _ = make_fake_acp(conn_factory)
    monkeypatch.setattr(acp_adapter, "_import_acp", lambda: fake)
    adapter = ACPAdapter("claude-agent-acp")
    session_id = run(adapter.start("task-1", "/repo", ""))

    events = run(_collect(adapter.send(session_id, "edit it")))

    diff_event = next(e for e in events if e.kind is EventKind.FILE_DIFF)
    assert diff_event.path == "forgeos/foo.py"
    assert "-a = 1" in diff_event.diff
    assert "+a = 2" in diff_event.diff


def test_acp_send_reports_error_event_instead_of_raising(monkeypatch):
    class ExplodingConnection(FakeConnection):
        async def prompt(self, **kw):
            self.calls.append(("prompt", kw))
            raise RuntimeError("agent process crashed")

    fake, _ = make_fake_acp(lambda client: ExplodingConnection(client))
    monkeypatch.setattr(acp_adapter, "_import_acp", lambda: fake)
    adapter = ACPAdapter("claude-agent-acp")
    session_id = run(adapter.start("task-1", "/repo", ""))

    events = run(_collect(adapter.send(session_id, "do the thing")))

    assert len(events) == 1
    assert events[0].kind is EventKind.ERROR
    assert "agent process crashed" in events[0].text


# ------------------------------------------------------------------- usage


def test_acp_usage_is_inexact_before_any_turn_completes(monkeypatch):
    fake, _ = make_fake_acp(make_conn_factory(False))
    monkeypatch.setattr(acp_adapter, "_import_acp", lambda: fake)
    adapter = ACPAdapter("claude-agent-acp")
    session_id = run(adapter.start("task-1", "/repo", ""))

    usage = run(adapter.usage(session_id))

    assert usage == WorkerUsage()
    assert usage.exact is False


def test_acp_usage_reports_tokens_but_stays_inexact_without_a_reported_cost(monkeypatch):
    def conn_factory(client):
        conn = FakeConnection(client)
        conn.prompt_script = [
            ("response", FakePromptResponse(stop_reason="end_turn", usage=FakeTurnUsage(input_tokens=10, output_tokens=5))),
        ]
        return conn

    fake, _ = make_fake_acp(conn_factory)
    monkeypatch.setattr(acp_adapter, "_import_acp", lambda: fake)
    adapter = ACPAdapter("claude-agent-acp")
    session_id = run(adapter.start("task-1", "/repo", ""))
    run(_collect(adapter.send(session_id, "hi")))

    usage = run(adapter.usage(session_id))

    # Tokens are real, reported numbers — fine to carry. usd_micros is not:
    # nothing ever reported a dollar figure, so it must not claim exact=True.
    assert usage.input_tokens == 10
    assert usage.output_tokens == 5
    assert usage.usd_micros == 0
    assert usage.exact is False


def test_acp_usage_is_exact_once_a_real_cost_is_reported(monkeypatch):
    def conn_factory(client):
        conn = FakeConnection(client)
        conn.prompt_script = [
            FakeUsageUpdate(used=100, size=1000, cost=FakeCost(amount=0.0034, currency="USD")),
            ("response", FakePromptResponse(stop_reason="end_turn", usage=FakeTurnUsage(input_tokens=10, output_tokens=5))),
        ]
        return conn

    fake, _ = make_fake_acp(conn_factory)
    monkeypatch.setattr(acp_adapter, "_import_acp", lambda: fake)
    adapter = ACPAdapter("claude-agent-acp")
    session_id = run(adapter.start("task-1", "/repo", ""))

    events = run(_collect(adapter.send(session_id, "hi")))
    assert any(e.kind is EventKind.USAGE for e in events)

    usage = run(adapter.usage(session_id))

    assert usage.exact is True
    assert usage.usd_micros == to_micros(0.0034)


# --------------------------------------------------------- checkpoint/resume


def test_acp_checkpoint_has_no_transcript_shaped_keys(monkeypatch):
    fake, _ = make_fake_acp(make_conn_factory(False))
    monkeypatch.setattr(acp_adapter, "_import_acp", lambda: fake)
    adapter = ACPAdapter("claude-agent-acp")
    session_id = run(adapter.start("task-1", "/repo", "sonnet"))

    checkpoint = run(adapter.checkpoint(session_id))

    forbidden_substrings = ("message", "transcript", "history", "conversation", "chat", "prompt")
    for key in checkpoint:
        lowered = key.lower()
        assert not any(bad in lowered for bad in forbidden_substrings), key
    assert checkpoint["task_id"] == "task-1"
    assert checkpoint["cwd"] == "/repo"
    assert checkpoint["model_profile"] == "sonnet"
    assert checkpoint["acp_session_id"] == session_id
    assert checkpoint["command"] == "claude-agent-acp"


def test_acp_resume_uses_load_session_when_the_agent_supports_it(monkeypatch):
    fake, _ = make_fake_acp(make_conn_factory(True))
    monkeypatch.setattr(acp_adapter, "_import_acp", lambda: fake)
    adapter = ACPAdapter("claude-agent-acp")
    session_id = run(adapter.start("task-1", "/repo", "sonnet"))
    checkpoint = run(adapter.checkpoint(session_id))
    run(adapter.close(session_id))

    resumed_id = run(adapter.resume(checkpoint))

    assert resumed_id == session_id  # ACP session identity carried over
    resumed_conn = adapter._sessions[resumed_id].conn
    call_names = [name for name, _ in resumed_conn.calls]
    assert "load_session" in call_names
    assert "new_session" not in call_names


def test_acp_resume_falls_back_to_new_session_without_load_session_support(monkeypatch):
    fake, _ = make_fake_acp(make_conn_factory(False))
    monkeypatch.setattr(acp_adapter, "_import_acp", lambda: fake)
    adapter = ACPAdapter("claude-agent-acp")
    session_id = run(adapter.start("task-1", "/repo", "sonnet"))
    checkpoint = run(adapter.checkpoint(session_id))
    run(adapter.close(session_id))

    resumed_id = run(adapter.resume(checkpoint))

    assert resumed_id != session_id  # continuity honestly not claimed
    resumed_conn = adapter._sessions[resumed_id].conn
    call_names = [name for name, _ in resumed_conn.calls]
    assert "new_session" in call_names
    assert "load_session" not in call_names


# --------------------------------------------------------------- cancel/close


def test_acp_cancel_calls_the_underlying_cancel(monkeypatch):
    fake, _ = make_fake_acp(make_conn_factory(False))
    monkeypatch.setattr(acp_adapter, "_import_acp", lambda: fake)
    adapter = ACPAdapter("claude-agent-acp")
    session_id = run(adapter.start("task-1", "/repo", ""))

    run(adapter.cancel(session_id))

    conn = adapter._sessions[session_id].conn
    assert conn.calls[-1][0] == "cancel"


def test_acp_close_is_idempotent_and_drops_the_session(monkeypatch):
    fake, _ = make_fake_acp(make_conn_factory(False))
    monkeypatch.setattr(acp_adapter, "_import_acp", lambda: fake)
    adapter = ACPAdapter("claude-agent-acp")
    session_id = run(adapter.start("task-1", "/repo", ""))

    run(adapter.close(session_id))
    assert session_id not in adapter._sessions
    run(adapter.close(session_id))  # closing twice must not raise

    with pytest.raises(KeyError):
        run(adapter.usage(session_id))
