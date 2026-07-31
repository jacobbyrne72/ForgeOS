"""`forgeos.mcp_server` driven as a real subprocess over real stdio.

Unlike most of this suite, these tests do not import `forgeos.mcp_server`
and call functions directly -- the whole point of an MCP server is the wire
protocol (newline-delimited JSON-RPC 2.0 on stdin/stdout), so these actually
spawn `python -m forgeos serve-mcp` and talk to it the way a real MCP client
would. That is also why every test here is `@pytest.mark.slow`.

`HOME`/`USERPROFILE` are pointed at a throwaway directory for every spawned
server: `forgeos_doctor` reuses `cmd_doctor`, which constructs a real
`Forge()` with no `--state-dir` override (matching the plain `doctor` CLI
subcommand, which has never had that flag), and a real `Forge()` opens sqlite
stores under `Path.home() / ".forgeos"` as a side effect of construction --
the same reason `tests/test_cli.py` never lets `cmd_doctor` touch the real
`Forge` in-process. Out here, across a subprocess boundary, redirecting HOME
is the equivalent guard.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from forgeos.watch import _parse_job_spec

pytestmark = pytest.mark.slow


def _start_server(tmp_path: Path, *, queue: Path | None = None) -> subprocess.Popen:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = dict(os.environ)
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)

    cmd = [sys.executable, "-m", "forgeos", "serve-mcp", "--state-dir", str(tmp_path / "state")]
    if queue is not None:
        cmd += ["--queue", str(queue)]

    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )


def _shutdown(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.stdin.close()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def _send(proc: subprocess.Popen, msg: dict) -> None:
    proc.stdin.write(json.dumps(msg) + "\n")
    proc.stdin.flush()


def _recv(proc: subprocess.Popen) -> dict:
    line = proc.stdout.readline()
    if not line:
        detail = proc.stderr.read() if proc.poll() is not None else "<process still running>"
        raise AssertionError(f"no response from serve-mcp (stderr: {detail!r})")
    return json.loads(line)


def _initialize(proc: subprocess.Popen, *, msg_id: int = 1) -> dict:
    _send(
        proc,
        {
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "0"},
            },
        },
    )
    resp = _recv(proc)
    _send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})
    return resp


@pytest.fixture
def server(tmp_path):
    proc = _start_server(tmp_path)
    yield proc
    _shutdown(proc)


# --------------------------------------------------------------- handshake


def test_initialize_handshake(server):
    resp = _initialize(server)
    assert resp["id"] == 1
    assert "error" not in resp
    result = resp["result"]
    assert result["protocolVersion"] == "2024-11-05"
    assert result["serverInfo"]["name"] == "forgeos"
    assert "tools" in result["capabilities"]


def test_tools_list_shape(server):
    _initialize(server)
    _send(server, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    resp = _recv(server)

    tools = {t["name"]: t for t in resp["result"]["tools"]}
    assert set(tools) == {"forgeos_doctor", "forgeos_receipts", "forgeos_plan", "forgeos_submit_job"}
    for tool in tools.values():
        assert isinstance(tool["description"], str) and tool["description"]
        assert tool["inputSchema"]["type"] == "object"
    assert tools["forgeos_plan"]["inputSchema"]["required"] == ["objective"]
    assert set(tools["forgeos_submit_job"]["inputSchema"]["required"]) == {"objective", "budget_usd"}


# ------------------------------------------------------------- read-only tools


def test_doctor_call_returns_a_sane_result(server):
    _initialize(server)
    _send(
        server,
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "forgeos_doctor", "arguments": {}}},
    )
    resp = _recv(server)

    assert resp["id"] == 3
    assert "error" not in resp
    result = resp["result"]
    assert result["isError"] is False
    text = result["content"][0]["text"]
    assert "forgeos doctor" in text
    assert "Runnable now:" in text


def test_plan_call_returns_a_task_graph(server, tmp_path):
    _initialize(server)
    _send(
        server,
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "forgeos_plan",
                "arguments": {"objective": "Fix the login bug", "cwd": str(tmp_path)},
            },
        },
    )
    resp = _recv(server)

    result = resp["result"]
    assert result["isError"] is False
    text = result["content"][0]["text"]
    assert "Mission:" in text
    assert "Tasks:" in text


def test_plan_call_with_blank_objective_is_a_tool_error(server):
    _initialize(server)
    _send(
        server,
        {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "forgeos_plan", "arguments": {}}},
    )
    resp = _recv(server)

    result = resp["result"]
    assert result["isError"] is True


# --------------------------------------------------------------- submit_job


def test_submit_unavailable_without_queue_configured(server):
    _initialize(server)
    _send(
        server,
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {"name": "forgeos_submit_job", "arguments": {"objective": "x", "budget_usd": 1.0}},
        },
    )
    resp = _recv(server)

    result = resp["result"]
    assert result["isError"] is True
    assert "unavailable" in result["content"][0]["text"]


def test_submit_without_budget_is_refused(tmp_path):
    queue = tmp_path / "queue"
    proc = _start_server(tmp_path, queue=queue)
    try:
        _initialize(proc)
        _send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/call",
                "params": {"name": "forgeos_submit_job", "arguments": {"objective": "do work"}},
            },
        )
        resp = _recv(proc)

        result = resp["result"]
        assert result["isError"] is True
        assert "budget_usd" in result["content"][0]["text"]
        incoming = queue / "incoming"
        if incoming.exists():
            assert list(incoming.glob("*.json")) == []
    finally:
        _shutdown(proc)


def test_submit_with_budget_lands_a_valid_file_in_incoming(tmp_path):
    queue = tmp_path / "queue"
    proc = _start_server(tmp_path, queue=queue)
    try:
        _initialize(proc)
        _send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 8,
                "method": "tools/call",
                "params": {
                    "name": "forgeos_submit_job",
                    "arguments": {
                        "objective": "do work",
                        "budget_usd": 5.0,
                        "tasks": [{"subject": "s", "description": "d"}],
                    },
                },
            },
        )
        resp = _recv(proc)

        result = resp["result"]
        assert result["isError"] is False
        payload = json.loads(result["content"][0]["text"])
        landed = Path(payload["path"])
        assert landed.parent == queue / "incoming"
        assert landed.exists()

        data = json.loads(landed.read_text(encoding="utf-8"))
        # Re-validate with the exact function `watch_queue` itself uses when
        # it later picks this file up -- the queue file format has one
        # definition, not a second one this tool invented independently.
        _parse_job_spec(data)
    finally:
        _shutdown(proc)


# ---------------------------------------------------------- protocol robustness


def test_malformed_json_gets_a_parse_error_and_the_server_keeps_running(server):
    server.stdin.write("not json at all\n")
    server.stdin.flush()
    resp = _recv(server)
    assert resp["error"]["code"] == -32700

    # Prove the loop is still alive after the bad line, not just that it
    # replied once before dying.
    resp2 = _initialize(server, msg_id=99)
    assert resp2["result"]["protocolVersion"] == "2024-11-05"


def test_request_missing_method_is_an_invalid_request(server):
    _send(server, {"jsonrpc": "2.0", "id": 10, "params": {}})
    resp = _recv(server)
    assert resp["error"]["code"] == -32600


def test_unknown_method_is_method_not_found(server):
    _initialize(server)
    _send(server, {"jsonrpc": "2.0", "id": 11, "method": "not/a/real/method"})
    resp = _recv(server)
    assert resp["error"]["code"] == -32601


def test_unknown_tool_name_is_invalid_params(server):
    _initialize(server)
    _send(
        server,
        {
            "jsonrpc": "2.0",
            "id": 12,
            "method": "tools/call",
            "params": {"name": "does_not_exist", "arguments": {}},
        },
    )
    resp = _recv(server)
    assert resp["error"]["code"] == -32602
