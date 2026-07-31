"""ForgeOS as an MCP tool — Port #2 from `docs/research/harness-features-codex.md`
("MCP server surface for ForgeOS itself").

Codex's own `mcp-server` crate is explicitly a "prototype", which is the
justification here for implementing only the subset of the Model Context
Protocol a tools-only server needs: `initialize`, `tools/list`, `tools/call`,
plus graceful shutdown on stdin EOF. No new dependency -- the wire format is
JSON-RPC 2.0 over stdio, one message per line (newline-delimited, no
embedded newlines -- see the MCP stdio transport spec), which stdlib `json`
covers completely.

**stdout is the protocol channel.** Every response this module writes goes
to stdout and nowhere else; anything resembling a log line must go to
stderr instead, and none of the tool handlers below are allowed to let a
stray `print()` leak onto the real stdout -- see `_capture_stdout`.

Each tool reuses an existing, already-reviewed data path instead of
re-deriving it:

- `forgeos_doctor`   -> `forgeos.__main__.cmd_doctor` (machine/fleet/price report)
- `forgeos_receipts` -> `forgeos.__main__.cmd_receipts` (ledger summary)
- `forgeos_plan`     -> `forgeos.compiler.compile_mission` (dry-run, $0 --
                        never calls `Forge.run`, so nothing is ever spent)
- `forgeos_submit_job` -> `forgeos.watch._parse_job_spec` (the exact validator
                        `watch_queue` itself uses on a picked-up file), then
                        writes the same validated JSON into `--queue/incoming`
- `forgeos_queue_status` -> `forgeos.watch.queue_status` (read-only heartbeat
                        and OS-lock inspection; never starts Forge)

`forgeos_submit_job` refuses without an explicit `budget_usd`, same rule
`watch_queue` and `team` already enforce -- forgeos never invents a spending
cap. If the server was not started with `--queue`, the tool reports itself
unavailable rather than silently doing nothing.

Task text a caller sends (objective strings, task dicts) is DATA that flows
into pydantic validation, never into `eval`/`exec`/a shell.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import uuid
from pathlib import Path
from typing import Any

from . import __version__

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "forgeos"

# JSON-RPC 2.0 reserved error codes, used verbatim.
_PARSE_ERROR = -32700
_INVALID_REQUEST = -32600
_METHOD_NOT_FOUND = -32601
_INVALID_PARAMS = -32602


class _ToolError(Exception):
    """A tool ran but refused or failed -- becomes `isError: true` in the
    `tools/call` result. The *request* was well-formed; only the work was
    not, so this is never a JSON-RPC protocol-level error."""


class _UnknownTool(Exception):
    """`tools/call` named a tool outside the catalogue -- a malformed
    *request*, so this maps to a JSON-RPC `Invalid params` error instead."""


def _capture_stdout(fn, *args, **kwargs) -> str:
    """Run `fn`, returning whatever it printed instead of letting it reach
    the real stdout -- which here is reserved for the JSON-RPC stream."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*args, **kwargs)
    return buf.getvalue()


class McpServer:
    """Holds the two pieces of local configuration a tool call might need
    and dispatches `tools/list`/`tools/call` against the catalogue below."""

    def __init__(self, *, state_dir: str | None = None, queue_dir: str | None = None) -> None:
        self.state_dir = state_dir
        self.queue_dir = queue_dir

    def tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "forgeos_doctor",
                "description": (
                    "Read-only machine/fleet/price report: runnable workers, provider "
                    "readiness, price-catalog staleness. No network calls."
                ),
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "forgeos_receipts",
                "description": (
                    "Read-only ledger summary: spend by job/worker, cost per accepted task."
                ),
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "forgeos_plan",
                "description": (
                    "Compile a natural-language objective into a task graph. Dry-run "
                    "only -- never spends money, never calls Forge.run."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "objective": {"type": "string", "description": "Natural-language objective to decompose"},
                        "cwd": {"type": "string", "description": "Working directory to analyze (default: .)"},
                    },
                    "required": ["objective"],
                },
            },
            {
                "name": "forgeos_submit_job",
                "description": (
                    "Submit a validated job spec into the watch-mode queue's incoming/ "
                    "directory. Requires an explicit budget_usd -- forgeos never invents "
                    "a spending cap. Reports itself unavailable if the server was not "
                    "started with --queue."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "objective": {"type": "string"},
                        "budget_usd": {"type": "number"},
                        "cwd": {"type": "string"},
                        "tasks": {"type": "array", "items": {"type": "object"}},
                        "max_seconds": {"type": "integer"},
                        "max_iterations": {"type": "integer"},
                    },
                    "required": ["objective", "budget_usd"],
                },
            },
            {
                "name": "forgeos_queue_status",
                "description": (
                    "Read-only queue liveness: OS ownership lock, heartbeat validity, "
                    "age, current job, and completed/failed counts. Requires --queue; "
                    "never starts Forge or calls a provider."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "stale_after_seconds": {
                            "type": "number",
                            "description": "Heartbeat age threshold (default: 30 seconds)",
                        },
                    },
                },
            },
        ]

    # ------------------------------------------------------------- tool calls

    def call_tool(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        """Returns `(text, is_error)`. Any exception a handler raises becomes
        an error result, not a crashed server -- the same "one bad job must
        not stop the daemon" posture `watch_queue` already takes."""
        handler = self._HANDLERS.get(name)
        if handler is None:
            raise _UnknownTool(name)
        try:
            return handler(self, arguments or {}), False
        except _ToolError as exc:
            return str(exc), True
        except Exception as exc:  # noqa: BLE001 - a tool bug is an error result, never a crash
            return f"{exc.__class__.__name__}: {exc}", True

    def _call_doctor(self, arguments: dict[str, Any]) -> str:
        from .__main__ import cmd_doctor

        return _capture_stdout(cmd_doctor, argparse.Namespace())

    def _call_receipts(self, arguments: dict[str, Any]) -> str:
        from .__main__ import cmd_receipts

        return _capture_stdout(cmd_receipts, argparse.Namespace(state_dir=self.state_dir))

    def _call_plan(self, arguments: dict[str, Any]) -> str:
        from .__main__ import _print_mission_graph
        from .compiler import CompilerError, compile_mission

        objective = arguments.get("objective")
        if not isinstance(objective, str) or not objective.strip():
            raise _ToolError("objective must be a non-empty string")
        cwd = arguments.get("cwd") or "."

        try:
            mission = compile_mission(objective, cwd=cwd)
        except CompilerError as exc:
            raise _ToolError(f"Cannot compile objective: {exc}") from exc

        return _capture_stdout(_print_mission_graph, mission)

    def _call_submit_job(self, arguments: dict[str, Any]) -> str:
        if not self.queue_dir:
            raise _ToolError("submit unavailable: this server was not started with --queue")

        raw: dict[str, Any] = {
            "objective": arguments.get("objective", ""),
            "cwd": arguments.get("cwd", "."),
            "tasks": arguments.get("tasks", []),
        }
        if "budget_usd" in arguments:
            raw["budget_usd"] = arguments["budget_usd"]
        if "max_seconds" in arguments:
            raw["max_seconds"] = arguments["max_seconds"]
        if "max_iterations" in arguments:
            raw["max_iterations"] = arguments["max_iterations"]

        # Reuse `watch_queue`'s own validator so a spec this tool accepts is
        # guaranteed to be one `watch_queue` will later accept too -- the
        # queue file format is defined once, here as everywhere else.
        from .watch import _JobSpecError, _parse_job_spec

        try:
            _parse_job_spec(raw)
        except _JobSpecError as exc:
            raise _ToolError(str(exc)) from exc

        incoming = Path(self.queue_dir) / "incoming"
        incoming.mkdir(parents=True, exist_ok=True)
        name = f"mcp_{uuid.uuid4().hex[:12]}.json"
        (incoming / name).write_text(json.dumps(raw), encoding="utf-8")
        return json.dumps({"queued": name, "path": str(incoming / name)})

    def _call_queue_status(self, arguments: dict[str, Any]) -> str:
        if not self.queue_dir:
            raise _ToolError("queue status unavailable: this server was not started with --queue")
        stale_after = arguments.get("stale_after_seconds", 30.0)
        if (
            isinstance(stale_after, bool)
            or not isinstance(stale_after, (int, float))
            or not math.isfinite(stale_after)
            or stale_after <= 0
        ):
            raise _ToolError("stale_after_seconds must be a finite positive number")

        from .watch import queue_status

        return queue_status(
            self.queue_dir, stale_after_seconds=float(stale_after)
        ).model_dump_json()

    _HANDLERS = {
        "forgeos_doctor": _call_doctor,
        "forgeos_receipts": _call_receipts,
        "forgeos_plan": _call_plan,
        "forgeos_submit_job": _call_submit_job,
        "forgeos_queue_status": _call_queue_status,
    }


# --------------------------------------------------------------- JSON-RPC wire


def _response(msg_id: Any, *, result: Any = None, error: dict[str, Any] | None = None) -> dict[str, Any]:
    if error is not None:
        return {"jsonrpc": "2.0", "id": msg_id, "error": error}
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _error(code: int, message: str) -> dict[str, Any]:
    return {"code": code, "message": message}


def _handle_message(server: McpServer, msg: Any) -> dict[str, Any] | None:
    """Dispatch one already-JSON-decoded message. Returns a response dict,
    or `None` for a notification (no `id`, no response expected) -- unknown
    methods still get a proper JSON-RPC error, never a crash."""
    if not isinstance(msg, dict) or not isinstance(msg.get("method"), str):
        msg_id = msg.get("id") if isinstance(msg, dict) else None
        return _response(msg_id, error=_error(_INVALID_REQUEST, "Invalid Request"))

    method = msg["method"]
    is_notification = "id" not in msg
    msg_id = msg.get("id")

    if method == "initialize":
        result = {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": __version__},
        }
        return None if is_notification else _response(msg_id, result=result)

    if method in ("notifications/initialized", "initialized"):
        return None  # always a notification in practice; nothing to do

    if method == "tools/list":
        return None if is_notification else _response(msg_id, result={"tools": server.tools()})

    if method == "tools/call":
        params = msg.get("params") or {}
        name = params.get("name")
        arguments = params.get("arguments") or {}
        try:
            text, is_error = server.call_tool(name, arguments)
        except _UnknownTool:
            if is_notification:
                return None
            return _response(msg_id, error=_error(_INVALID_PARAMS, f"Unknown tool: {name}"))
        result = {"content": [{"type": "text", "text": text}], "isError": is_error}
        return None if is_notification else _response(msg_id, result=result)

    return None if is_notification else _response(msg_id, error=_error(_METHOD_NOT_FOUND, f"Method not found: {method}"))


def serve(*, state_dir: str | None = None, queue_dir: str | None = None, stdin=None, stdout=None) -> None:
    """Read newline-delimited JSON-RPC requests from `stdin`, write responses
    to `stdout`, one message per line, until EOF (graceful shutdown -- no
    separate `shutdown` request exists in MCP's stdio transport)."""
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    server = McpServer(state_dir=state_dir, queue_dir=queue_dir)

    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            stdout.write(json.dumps(_response(None, error=_error(_PARSE_ERROR, "Parse error"))) + "\n")
            stdout.flush()
            continue

        response = _handle_message(server, msg)
        if response is not None:
            stdout.write(json.dumps(response) + "\n")
            stdout.flush()


__all__ = ["McpServer", "PROTOCOL_VERSION", "serve"]
