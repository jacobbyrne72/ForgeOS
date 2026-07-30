"""ACPAdapter — the Agent Client Protocol execution backend.

forgeos is always the ACP *client* side of the connection: it spawns a vendor
CLI that speaks ACP as the *agent* (`@agentclientprotocol/claude-agent-acp`,
`@agentclientprotocol/codex-acp`, or any other ACP-compatible agent binary),
drives the session lifecycle, and translates the agent's `session/update`
notifications into `base.WorkerEvent`s.

Real API surface, verified against `vendor/acp-python-sdk/src/acp` and the
two concrete agents at `vendor/claude-agent-acp` / `vendor/codex-acp`
(2026-07-30) — this is not guessed:

- `acp.spawn_agent_process(client, command, *args, cwd=..., env=...)` is an
  async context manager (`vendor/acp-python-sdk/src/acp/stdio.py`) that
  spawns `command` over stdio and yields `(ClientSideConnection, Process)`.
- `ClientSideConnection.initialize(protocol_version, client_capabilities,
  client_info) -> InitializeResponse` (`agent_capabilities.load_session`
  tells us whether `session/load` is available for `resume()`).
- `ClientSideConnection.new_session(cwd, mcp_servers=[]) ->
  NewSessionResponse(session_id, ...)`.
- `ClientSideConnection.load_session(cwd, session_id, mcp_servers=[])` —
  only called when `agent_capabilities.load_session` is true.
- `ClientSideConnection.prompt(session_id, prompt=[TextContentBlock, ...])
  -> PromptResponse(stop_reason, usage)`. The `Client.session_update`
  callback delivers `AgentMessageChunk` / `AgentThoughtChunk` /
  `ToolCallStart` / `ToolCallProgress` / `AgentPlanUpdate` / `UsageUpdate`
  notifications *while* `prompt()` is still in flight — the SDK's
  `Connection` runs its own receive-loop task from construction
  (`listening=True` by default, `acp/connection.py:Connection.__init__`),
  so callback and response race on the same event loop. `send()` below
  mirrors that by racing a queue fed from `session_update` against the
  `prompt()` coroutine.
- `ClientSideConnection.cancel(session_id)`, `.close_session(session_id)`.
- Every `SessionNotification.update` variant carries its own discriminator
  as a plain string field, `session_update` (wire alias `sessionUpdate`,
  e.g. `"agent_message_chunk"`, `"tool_call"`, `"usage_update"`). Dispatching
  on that string (duck-typed, via `getattr`) rather than importing and
  `isinstance`-checking every schema class keeps this adapter working across
  the protocol's still-`**UNSTABLE**`-tagged fields without pinning to one
  exact class set.
- `ClientCapabilities` and `Implementation` are only exported from
  `acp.schema`, not from the `acp` top-level package — confirmed by reading
  `acp/__init__.py`'s import list against `examples/client.py`'s imports.
  `PROTOCOL_VERSION`, `text_block`, `spawn_agent_process`, `RequestError`,
  `RequestPermissionResponse`, `DeclineElicitationResponse` *are* top-level.

Subscription CLIs own their own credentials (browser OAuth or their own
`*_API_KEY` env var — see `codex-acp`'s README). This adapter never reads,
copies, logs, or constructs an `env` mapping containing one: it always spawns
with `env=None`, so the subprocess inherits whatever the operator's own shell
already has, and authentication is entirely the vendor CLI's problem.
"""

from __future__ import annotations

import asyncio
import contextlib
import difflib
import shutil
from dataclasses import dataclass, field
from typing import Any

from collections.abc import AsyncIterator

from ..contracts import to_micros
from .base import EventKind, WorkerAdapter, WorkerCapabilities, WorkerEvent, WorkerUsage


def _import_acp() -> Any:
    """Import the ACP SDK, isolated in its own function.

    The harness has to run on a machine without ACP installed — `health()`
    must degrade to False with a clear reason, not blow up the process. Tests
    substitute this function directly so the SDK-absent and SDK-mocked paths
    are both exercised without a real ACP install or a real subprocess.
    """
    import acp  # noqa: PLC0415

    return acp


def _content_text(content: Any) -> str:
    """A `ContentBlock`'s displayable text, or a `<kind>` placeholder.

    Only `TextContentBlock` carries `.text`; image/audio/resource blocks
    don't, so this never assumes the content is text-shaped.
    """
    text = getattr(content, "text", None)
    if text is not None:
        return str(text)
    return f"<{getattr(content, 'type', 'content')}>"


def _first_file_diff(content_list: Any) -> Any | None:
    """The first `FileEditToolCallContent`-shaped item in a tool call's
    content list, found by duck-typing (`path` + `new_text`) rather than an
    `isinstance` check against one exact schema class."""
    for item in content_list or []:
        if hasattr(item, "path") and hasattr(item, "new_text"):
            return item
    return None


def _unified_diff(diff: Any) -> str:
    old = (getattr(diff, "old_text", None) or "").splitlines(keepends=True)
    new = (getattr(diff, "new_text", "") or "").splitlines(keepends=True)
    path = getattr(diff, "path", "")
    return "".join(difflib.unified_diff(old, new, fromfile=path, tofile=path))


def _tool_event(kind: EventKind, update: Any) -> WorkerEvent:
    return WorkerEvent(
        kind=kind,
        tool_call_id=getattr(update, "tool_call_id", "") or "",
        tool_kind=str(getattr(update, "kind", "") or ""),
        status=str(getattr(update, "status", "") or ""),
        text=getattr(update, "title", "") or "",
    )


def _usage_update_event(update: Any) -> WorkerEvent:
    data: dict[str, Any] = {
        "used": getattr(update, "used", 0),
        "size": getattr(update, "size", 0),
    }
    cost = getattr(update, "cost", None)
    if cost is not None:
        data["cost_amount"] = getattr(cost, "amount", None)
        data["cost_currency"] = getattr(cost, "currency", None)
    return WorkerEvent(kind=EventKind.USAGE, data=data)


def _translate_update(update: Any) -> WorkerEvent:
    """One `SessionNotification.update` -> one canonical `WorkerEvent`.

    Dispatches on the wire discriminator string (`session_update`) rather
    than importing every schema class, so an update shape this adapter
    doesn't specifically recognise still surfaces (as `EventKind.META`)
    instead of silently vanishing.
    """
    tag = getattr(update, "session_update", "")
    if tag in ("agent_message_chunk", "user_message_chunk"):
        return WorkerEvent(kind=EventKind.MESSAGE, text=_content_text(getattr(update, "content", None)))
    if tag == "agent_thought_chunk":
        return WorkerEvent(kind=EventKind.THOUGHT, text=_content_text(getattr(update, "content", None)))
    if tag == "tool_call":
        return _tool_event(EventKind.TOOL_CALL, update)
    if tag == "tool_call_update":
        diff = _first_file_diff(getattr(update, "content", None))
        if diff is not None:
            return WorkerEvent(
                kind=EventKind.FILE_DIFF,
                tool_call_id=getattr(update, "tool_call_id", "") or "",
                path=getattr(diff, "path", ""),
                diff=_unified_diff(diff),
            )
        return _tool_event(EventKind.TOOL_UPDATE, update)
    if tag in ("plan", "plan_update", "plan_removed"):
        entries = getattr(update, "entries", None)
        return WorkerEvent(
            kind=EventKind.PLAN,
            data={"entries": [e.model_dump(exclude_none=True) for e in entries]} if entries is not None else {},
        )
    if tag == "usage_update":
        return _usage_update_event(update)
    dump = update.model_dump(exclude_none=True) if hasattr(update, "model_dump") else {}
    return WorkerEvent(kind=EventKind.META, data={"session_update": tag, **dump})


def _apply_usage_update_event(prev: WorkerUsage, event: WorkerEvent) -> WorkerUsage:
    """Fold a real reported `Cost` into the running usage snapshot.

    Only a `usage_update` naming an actual USD amount ever flips `exact` to
    True — no guessing a dollar figure from token counts here.
    """
    amount = event.data.get("cost_amount")
    currency = event.data.get("cost_currency")
    if amount is None or currency != "USD":
        return prev
    return WorkerUsage(
        input_tokens=prev.input_tokens,
        output_tokens=prev.output_tokens,
        usd_micros=to_micros(float(amount)),
        exact=True,
    )


def _apply_turn_usage(prev: WorkerUsage, turn_usage: Any) -> WorkerUsage:
    """Fold `PromptResponse.usage` token counts into the snapshot.

    Token counts are real numbers, but they are not a dollar figure —
    `exact` is never derived from tokens alone, only ever from a reported
    `Cost` (see `_apply_usage_update_event`). Without that, presenting these
    tokens-implied-a-price as `exact=True` spend would be exactly the
    estimate-as-fact failure this contract exists to prevent.
    """
    if turn_usage is None:
        return prev
    return WorkerUsage(
        input_tokens=int(getattr(turn_usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(turn_usage, "output_tokens", 0) or 0),
        usd_micros=prev.usd_micros,
        exact=prev.exact,
    )


class _HiveClient:
    """forgeos's ACP `Client` half of the connection.

    forgeos is a headless harness, not an editor, so it declines the
    editor-mediated fs/terminal surfaces (`fs/*`, `terminal/*`) exactly like
    `acp-python-sdk`'s own `examples/client.py:ExampleClient` — the agent
    then falls back to touching disk and running commands directly inside
    its own worktree, which is what we want anyway. `session_update` is the
    one real piece of behaviour: forward every notification onto a queue for
    `ACPAdapter.send()` to drain. Tool-call permission requests are
    auto-approved: a worker running inside its own isolated git worktree
    (AGENTS.md) is exactly the case that does not need a human in the loop
    per call — the gated actions are push/merge/deploy/delete, which live
    above this adapter, not inside it.
    """

    def __init__(self, acp_module: Any, queue: "asyncio.Queue[WorkerEvent]") -> None:
        self._acp = acp_module
        self._queue = queue

    def on_connect(self, conn: Any) -> None:  # noqa: D102 - Client protocol hook, unused
        return None

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        await self._queue.put(_translate_update(update))

    async def request_permission(self, session_id: str, tool_call: Any, options: list[Any], **kwargs: Any) -> Any:
        for opt in options:
            if getattr(opt, "kind", "") in ("allow_once", "allow_always"):
                outcome = self._acp.schema.AllowedOutcome(outcome="selected", option_id=opt.option_id)
                return self._acp.RequestPermissionResponse(outcome=outcome)
        return self._acp.RequestPermissionResponse(outcome=self._acp.schema.DeniedOutcome(outcome="cancelled"))

    async def write_text_file(self, session_id: str, path: str, content: str, **kwargs: Any) -> Any:
        raise self._acp.RequestError.method_not_found("fs/write_text_file")

    async def read_text_file(
        self, session_id: str, path: str, line: int | None = None, limit: int | None = None, **kwargs: Any
    ) -> Any:
        raise self._acp.RequestError.method_not_found("fs/read_text_file")

    async def create_terminal(self, session_id: str, command: str, **kwargs: Any) -> Any:
        raise self._acp.RequestError.method_not_found("terminal/create")

    async def terminal_output(self, session_id: str, terminal_id: str, **kwargs: Any) -> Any:
        raise self._acp.RequestError.method_not_found("terminal/output")

    async def release_terminal(self, session_id: str, terminal_id: str, **kwargs: Any) -> Any:
        raise self._acp.RequestError.method_not_found("terminal/release")

    async def wait_for_terminal_exit(self, session_id: str, terminal_id: str, **kwargs: Any) -> Any:
        raise self._acp.RequestError.method_not_found("terminal/wait_for_exit")

    async def kill_terminal(self, session_id: str, terminal_id: str, **kwargs: Any) -> Any:
        raise self._acp.RequestError.method_not_found("terminal/kill")

    async def create_elicitation(self, message: str, mode: Any, **kwargs: Any) -> Any:
        return self._acp.DeclineElicitationResponse(action="decline")

    async def complete_elicitation(self, elicitation_id: str, **kwargs: Any) -> None:
        return None

    async def ext_method(self, method: str, params: dict) -> dict:
        raise self._acp.RequestError.method_not_found(method)

    async def ext_notification(self, method: str, params: dict) -> None:
        raise self._acp.RequestError.method_not_found(method)


@dataclass
class _Session:
    """Live state for one ACP session.

    Holds exactly what's needed to keep the connection open and to rebuild a
    checkpoint later — never message content. The transcript lives in the
    agent process (and, if it supports `session/load`, on its own disk); it
    is never forgeos's to carry.
    """

    task_id: str
    cwd: str
    model_profile: str
    command: str
    args: list[str]
    conn: Any
    process: Any
    exit_stack: contextlib.AsyncExitStack
    acp_session_id: str
    supports_load_session: bool
    queue: "asyncio.Queue[WorkerEvent]"
    last_usage: WorkerUsage = field(default_factory=WorkerUsage)


class ACPAdapter(WorkerAdapter):
    """Runs a task through any ACP-compatible agent CLI.

    `command`/`args` name the vendor CLI to spawn — e.g.
    `ACPAdapter("npx", ["-y", "@agentclientprotocol/claude-agent-acp"])` or
    `ACPAdapter("npx", ["-y", "@agentclientprotocol/codex-acp"])`. Nothing
    vendor-specific is hardcoded here; that choice belongs to
    `settings.Provider` / `registry.WorkerProfile`, not this adapter.
    """

    def __init__(self, command: str, args: list[str] | None = None, *, max_parallel_sessions: int = 4) -> None:
        self._command = command
        self._args = list(args or [])
        self._max_parallel_sessions = max_parallel_sessions
        self._sessions: dict[str, _Session] = {}

    # ------------------------------------------------------------- health

    def health(self) -> tuple[bool, str]:
        try:
            _import_acp()
        except ImportError as exc:
            return False, f"acp SDK not installed ({exc}); pip install the ACP Python SDK to enable this adapter"
        if not shutil.which(self._command):
            return False, f"'{self._command}' not found on PATH; install the vendor ACP CLI first"
        return True, "acp SDK importable and CLI on PATH"

    def capabilities(self) -> WorkerCapabilities:
        return WorkerCapabilities(
            structured_messages=True,
            resumable_sessions=True,
            tool_calls=True,
            file_diffs=True,
            reasoning_effort=False,
            token_usage=True,
            images=True,
            max_parallel_sessions=self._max_parallel_sessions,
        )

    # ------------------------------------------------------------ sessions

    def _get_session(self, session_id: str) -> _Session:
        try:
            return self._sessions[session_id]
        except KeyError:
            raise KeyError(f"no active ACP session: {session_id}") from None

    async def start(self, task_id: str, cwd: str, model_profile: str) -> str:
        acp = _import_acp()
        queue: asyncio.Queue[WorkerEvent] = asyncio.Queue()
        client = _HiveClient(acp, queue)
        stack = contextlib.AsyncExitStack()
        # env=None (the default) on purpose: the subprocess inherits the
        # operator's own environment rather than forgeos constructing one, so
        # forgeos never reads or copies whatever credential the CLI needs.
        conn, process = await stack.enter_async_context(
            acp.spawn_agent_process(client, self._command, *self._args, cwd=cwd)
        )
        init = await conn.initialize(
            protocol_version=acp.PROTOCOL_VERSION,
            client_capabilities=acp.schema.ClientCapabilities(),
            client_info=acp.schema.Implementation(name="forgeos", title="forgeos", version="0.1.0"),
        )
        supports_load_session = bool(getattr(getattr(init, "agent_capabilities", None), "load_session", False))

        meta: dict[str, Any] = {"hiveTaskId": task_id}
        if model_profile:
            meta["hiveModelProfile"] = model_profile
        new_sess = await conn.new_session(cwd=cwd, mcp_servers=[], **meta)
        acp_session_id = new_sess.session_id

        self._sessions[acp_session_id] = _Session(
            task_id=task_id,
            cwd=cwd,
            model_profile=model_profile,
            command=self._command,
            args=list(self._args),
            conn=conn,
            process=process,
            exit_stack=stack,
            acp_session_id=acp_session_id,
            supports_load_session=supports_load_session,
            queue=queue,
        )
        return acp_session_id

    async def send(self, session_id: str, prompt: str) -> AsyncIterator[WorkerEvent]:
        sess = self._get_session(session_id)
        acp = _import_acp()
        prompt_task: asyncio.Task = asyncio.ensure_future(
            sess.conn.prompt(session_id=sess.acp_session_id, prompt=[acp.text_block(prompt)])
        )
        try:
            while not prompt_task.done():
                get_task = asyncio.ensure_future(sess.queue.get())
                done, _pending = await asyncio.wait({prompt_task, get_task}, return_when=asyncio.FIRST_COMPLETED)
                if get_task in done:
                    event = get_task.result()
                    if event.kind is EventKind.USAGE:
                        sess.last_usage = _apply_usage_update_event(sess.last_usage, event)
                    yield event
                else:
                    get_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await get_task
            while not sess.queue.empty():
                event = sess.queue.get_nowait()
                if event.kind is EventKind.USAGE:
                    sess.last_usage = _apply_usage_update_event(sess.last_usage, event)
                yield event
            response = prompt_task.result()
        except BaseException as exc:  # noqa: BLE001 - surfaced as a canonical event, not swallowed
            prompt_task.cancel()
            yield WorkerEvent(kind=EventKind.ERROR, text=str(exc))
            return
        sess.last_usage = _apply_turn_usage(sess.last_usage, getattr(response, "usage", None))
        yield WorkerEvent(kind=EventKind.DONE, status=str(getattr(response, "stop_reason", "")))

    async def cancel(self, session_id: str) -> None:
        sess = self._get_session(session_id)
        await sess.conn.cancel(session_id=sess.acp_session_id)

    async def checkpoint(self, session_id: str) -> dict:
        """Canonical task state only — no message content of any kind."""
        sess = self._get_session(session_id)
        return {
            "adapter": "acp",
            "command": sess.command,
            "args": list(sess.args),
            "task_id": sess.task_id,
            "cwd": sess.cwd,
            "model_profile": sess.model_profile,
            "acp_session_id": sess.acp_session_id,
            "supports_load_session": sess.supports_load_session,
        }

    async def resume(self, checkpoint: dict) -> str:
        acp = _import_acp()
        command = checkpoint["command"]
        args = list(checkpoint.get("args", []))
        cwd = checkpoint["cwd"]
        task_id = checkpoint["task_id"]
        model_profile = checkpoint.get("model_profile", "")
        old_acp_session_id = checkpoint.get("acp_session_id", "")

        queue: asyncio.Queue[WorkerEvent] = asyncio.Queue()
        client = _HiveClient(acp, queue)
        stack = contextlib.AsyncExitStack()
        conn, process = await stack.enter_async_context(acp.spawn_agent_process(client, command, *args, cwd=cwd))
        init = await conn.initialize(
            protocol_version=acp.PROTOCOL_VERSION,
            client_capabilities=acp.schema.ClientCapabilities(),
            client_info=acp.schema.Implementation(name="forgeos", title="forgeos", version="0.1.0"),
        )
        agent_supports_load = bool(getattr(getattr(init, "agent_capabilities", None), "load_session", False))

        # A replacement worker resumes from canonical state, never from a
        # transcript. `session/load` is the one case where the *original*
        # ACP session identity itself is part of that state — the agent CLI
        # owns and replays its own history for that id. Anything else
        # (a different agent, or one that never advertised load_session)
        # starts a clean session against the same cwd instead of pretending
        # continuity it cannot actually provide.
        if agent_supports_load and old_acp_session_id:
            await conn.load_session(cwd=cwd, session_id=old_acp_session_id, mcp_servers=[])
            acp_session_id = old_acp_session_id
        else:
            meta: dict[str, Any] = {"hiveTaskId": task_id}
            if model_profile:
                meta["hiveModelProfile"] = model_profile
            new_sess = await conn.new_session(cwd=cwd, mcp_servers=[], **meta)
            acp_session_id = new_sess.session_id

        self._sessions[acp_session_id] = _Session(
            task_id=task_id,
            cwd=cwd,
            model_profile=model_profile,
            command=command,
            args=args,
            conn=conn,
            process=process,
            exit_stack=stack,
            acp_session_id=acp_session_id,
            supports_load_session=agent_supports_load,
            queue=queue,
        )
        return acp_session_id

    async def usage(self, session_id: str) -> WorkerUsage:
        return self._get_session(session_id).last_usage

    async def close(self, session_id: str) -> None:
        sess = self._sessions.pop(session_id, None)
        if sess is None:
            return
        with contextlib.suppress(Exception):
            await sess.conn.close_session(session_id=sess.acp_session_id)
        await sess.exit_stack.aclose()


__all__ = ["ACPAdapter"]
