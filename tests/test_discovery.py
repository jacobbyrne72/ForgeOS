"""Tests for `hive.adapters.discovery`.

No CLI vendor binary is ever invoked for real: `discovery.subprocess.run` and
`discovery.shutil.which` are the exact seams this module uses to reach a
subprocess or check PATH, so every test substitutes them directly with
realistic captured `--help`/`--version` text (same isolation pattern as
`tests/test_local_adapters.py`'s `ollama_adapter.subprocess.run` seam).
"""

from __future__ import annotations

import json
import subprocess

import pytest

from hive.adapters import discovery
from hive.adapters.base import WorkerCapabilities
from hive.adapters.discovery import (
    McpServer,
    ProbeCache,
    ToolProbe,
    capabilities_from_probe,
    discover_all,
    discover_mcp_servers,
    probe_cli,
)

# ============================================================== fixtures

# Realistic `codex --help` output: a "Commands:" block including exec/review/
# login/mcp/resume/doctor, with `exec` described as running non-interactively
# — the literal headless marker this module looks for, not an assumption.
CODEX_HELP = """\
Codex CLI

Usage: codex [OPTIONS] [COMMAND]

Commands:
  exec      Run Codex non-interactively
  review    Review code changes
  login     Manage authentication
  mcp       Model Context Protocol server
  resume    Resume a previous session
  doctor    Diagnose environment and configuration
  help      Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
      --json     Emit machine-readable JSON output
  -p, --print    Alias for `exec`; run non-interactively and print the result
"""

CODEX_VERSION = "codex-cli 0.21.0\n"

# A CLI with no headless markers anywhere: no exec/run subcommand, no
# -p/--print/--non-interactive flag, no "non-interactively" phrase.
PLAIN_HELP = """\
plaintool - a simple interactive assistant

Usage: plaintool [OPTIONS] [COMMAND]

Commands:
  chat      Start an interactive chat session
  config    Edit configuration

Options:
  -h, --help     Print help
  -V, --version  Print version
"""

PLAIN_VERSION = "plaintool 3.4.0\n"


def _fake_run(routes: dict[str, subprocess.CompletedProcess]):
    """Build a `subprocess.run` stand-in keyed by `cmd[1]` (the sub-argument:
    `--help` or `--version`), matching the convention in
    `tests/test_local_adapters.py` (`fake_run` keyed on `cmd[1]`)."""

    def fake(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
        key = cmd[1] if len(cmd) > 1 else ""
        if key in routes:
            return routes[key]
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="unknown option")

    return fake


def _cli_present(monkeypatch, name: str, help_text: str, version_text: str) -> None:
    monkeypatch.setattr(discovery.shutil, "which", lambda cmd: f"/usr/bin/{name}")
    routes = {
        "--help": subprocess.CompletedProcess([name, "--help"], returncode=0, stdout=help_text, stderr=""),
        "--version": subprocess.CompletedProcess(
            [name, "--version"], returncode=0, stdout=version_text, stderr=""
        ),
    }
    monkeypatch.setattr(discovery.subprocess, "run", _fake_run(routes))


# =========================================================== probe_cli


def test_probe_cli_parses_codex_subcommands_and_version(monkeypatch):
    _cli_present(monkeypatch, "codex", CODEX_HELP, CODEX_VERSION)

    probe = probe_cli("codex")

    assert probe is not None
    assert probe.name == "codex"
    assert probe.version == "0.21.0"
    assert set(probe.subcommands) == {"exec", "review", "login", "mcp", "resume", "doctor", "help"}
    assert "--json" in probe.flags
    assert "-h" in probe.flags and "--help" in probe.flags
    assert probe.probe_source == "codex --help"


def test_probe_cli_has_headless_true_for_codex_fixture(monkeypatch):
    """`exec` is a subcommand AND its own description literally says
    'non-interactively' — the fixture the task description calls out."""
    _cli_present(monkeypatch, "codex", CODEX_HELP, CODEX_VERSION)

    probe = probe_cli("codex")

    assert probe.has_headless is True
    assert probe.headless_invocation == "codex exec"


def test_probe_cli_has_headless_false_when_no_marker_present(monkeypatch):
    _cli_present(monkeypatch, "plaintool", PLAIN_HELP, PLAIN_VERSION)

    probe = probe_cli("plaintool")

    assert probe.has_headless is False
    assert probe.headless_invocation == ""


def test_probe_cli_never_guesses_an_absent_capability(monkeypatch):
    """plaintool's help text has no 'mcp'/'resume' subcommand and no
    '--json'/'--resume'/'--mcp' flag anywhere: every derived capability must
    come back False, not a guess that "most CLIs probably support this"."""
    _cli_present(monkeypatch, "plaintool", PLAIN_HELP, PLAIN_VERSION)

    probe = probe_cli("plaintool")

    assert probe.supports_json is False
    assert probe.supports_resume is False
    assert probe.supports_mcp is False


def test_probe_cli_supports_resume_and_mcp_when_evidenced(monkeypatch):
    _cli_present(monkeypatch, "codex", CODEX_HELP, CODEX_VERSION)

    probe = probe_cli("codex")

    assert probe.supports_resume is True  # 'resume' subcommand
    assert probe.supports_mcp is True  # 'mcp' subcommand
    assert probe.supports_json is True  # '--json' flag


def test_probe_cli_returns_none_for_missing_binary(monkeypatch):
    monkeypatch.setattr(discovery.shutil, "which", lambda cmd: None)

    def must_not_be_called(*args, **kwargs):
        raise AssertionError("must not shell out when the binary itself is missing")

    monkeypatch.setattr(discovery.subprocess, "run", must_not_be_called)

    assert probe_cli("ghost-cli") is None


def test_probe_cli_never_raises_on_a_hung_help_call(monkeypatch):
    monkeypatch.setattr(discovery.shutil, "which", lambda cmd: "/usr/bin/codex")

    def boom(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 10)

    monkeypatch.setattr(discovery.subprocess, "run", boom)

    probe = probe_cli("codex")  # must not raise

    assert probe is not None
    assert probe.subcommands == []
    assert probe.has_headless is False
    assert probe.probe_source == "unavailable"


# ============================================================ discover_all


def test_discover_all_yields_none_for_missing_and_probe_for_present(monkeypatch):
    def which(cmd: str):
        return "/usr/bin/codex" if cmd == "codex" else None

    monkeypatch.setattr(discovery.shutil, "which", which)
    routes = {
        "--help": subprocess.CompletedProcess(["codex", "--help"], 0, CODEX_HELP, ""),
        "--version": subprocess.CompletedProcess(["codex", "--version"], 0, CODEX_VERSION, ""),
    }
    monkeypatch.setattr(discovery.subprocess, "run", _fake_run(routes))

    results = discover_all(["codex", "ghost-cli"])

    assert results["ghost-cli"] is None
    assert isinstance(results["codex"], ToolProbe)
    assert results["codex"].version == "0.21.0"


def test_discover_all_uses_cache_hit_without_a_help_call(monkeypatch, tmp_path):
    """On a cache hit, only the cheap `--version` call should run — a full
    `--help` re-parse would defeat the point of caching by version."""
    cache = ProbeCache(tmp_path / "probes.json")
    cached_probe = ToolProbe(
        name="codex", path="/usr/bin/codex", version="0.21.0",
        subcommands=["exec"], flags=["--json"], has_headless=True,
        headless_invocation="codex exec", supports_json=True,
        probed_at="then", probe_source="codex --help",
    )
    cache.put(cached_probe)

    monkeypatch.setattr(discovery.shutil, "which", lambda cmd: "/usr/bin/codex")

    calls: list[str] = []

    def fake(cmd, **kwargs):
        calls.append(cmd[1])
        if cmd[1] == "--help":
            raise AssertionError("cache hit must not re-run --help")
        return subprocess.CompletedProcess(cmd, 0, CODEX_VERSION, "")

    monkeypatch.setattr(discovery.subprocess, "run", fake)

    results = discover_all(["codex"], cache=cache)

    assert results["codex"] == cached_probe
    assert calls == ["--version"]


def test_discover_all_reprobes_and_updates_cache_on_version_change(monkeypatch, tmp_path):
    cache = ProbeCache(tmp_path / "probes.json")
    stale = ToolProbe(name="codex", path="/usr/bin/codex", version="0.20.0")
    cache.put(stale)

    _cli_present(monkeypatch, "codex", CODEX_HELP, CODEX_VERSION)  # version 0.21.0

    results = discover_all(["codex"], cache=cache)

    assert results["codex"].version == "0.21.0"
    # a fresh cache instance at the same path now serves the new version...
    reloaded = ProbeCache(tmp_path / "probes.json")
    assert reloaded.get("codex", "0.21.0") is not None
    # ...and asking for a version nobody has probed yet is still a clean miss
    # (the stale 0.20.0 entry lingering alongside it is harmless history, not
    # a false hit for a version that was never actually re-verified).
    assert reloaded.get("codex", "9.9.9") is None


# ================================================================ ProbeCache


def test_probe_cache_round_trips(tmp_path):
    path = tmp_path / "probes.json"
    probe = ToolProbe(
        name="codex", path="/usr/bin/codex", version="1.0.0",
        subcommands=["exec"], probed_at="2026-07-30T00:00:00+00:00",
        probe_source="codex --help",
    )
    ProbeCache(path).put(probe)

    reloaded = ProbeCache(path)
    got = reloaded.get("codex", "1.0.0")

    assert got == probe


def test_probe_cache_misses_on_version_change(tmp_path):
    path = tmp_path / "probes.json"
    probe = ToolProbe(name="codex", path="/usr/bin/codex", version="1.0.0")
    ProbeCache(path).put(probe)

    reloaded = ProbeCache(path)

    assert reloaded.get("codex", "2.0.0") is None
    assert reloaded.get("codex", "1.0.0") is not None


def test_probe_cache_falls_back_gracefully_on_corrupt_file(tmp_path):
    path = tmp_path / "probes.json"
    path.write_text("{not valid json at all", encoding="utf-8")

    cache = ProbeCache(path)  # must not raise

    assert cache.get("codex", "1.0.0") is None

    # and it must still be usable afterward, overwriting the corrupt file
    probe = ToolProbe(name="codex", path="/usr/bin/codex", version="1.0.0")
    cache.put(probe)
    assert ProbeCache(path).get("codex", "1.0.0") == probe


# ============================================================ capabilities


def test_capabilities_from_probe_only_sets_evidenced_fields():
    probe = ToolProbe(
        name="codex", path="/usr/bin/codex", version="0.21.0",
        supports_json=True, supports_resume=True, supports_mcp=True,
    )

    caps = capabilities_from_probe(probe)

    assert isinstance(caps, WorkerCapabilities)
    assert caps.structured_messages is True
    assert caps.resumable_sessions is True
    # Never evidenced by --help text alone — must stay at the honest default.
    assert caps.tool_calls is False
    assert caps.file_diffs is False
    assert caps.token_usage is False
    assert caps.images is False
    assert caps.reasoning_effort is False


def test_capabilities_from_probe_all_false_when_nothing_evidenced():
    probe = ToolProbe(name="plaintool", path="/usr/bin/plaintool", version="3.4.0")

    caps = capabilities_from_probe(probe)

    assert caps.structured_messages is False
    assert caps.resumable_sessions is False


# ======================================================= MCP server config


def test_discover_mcp_servers_reads_json_config(tmp_path):
    cfg = tmp_path / ".mcp.json"
    cfg.write_text(
        json.dumps({"mcpServers": {"filesystem": {"command": "npx", "args": ["-y", "@mcp/fs"]}}}),
        encoding="utf-8",
    )

    servers = discover_mcp_servers(paths=[cfg])

    assert len(servers) == 1
    s = servers[0]
    assert isinstance(s, McpServer)
    assert s.name == "filesystem"
    assert s.command == "npx"
    assert s.args == ["-y", "@mcp/fs"]
    assert s.source_file == str(cfg)
    assert s.transport == "stdio"


def test_discover_mcp_servers_redacts_credential_shaped_args(tmp_path):
    fake_token = "sk-test-abcdefghijklmnopqrstuvwxyz012345"  # planted fake secret
    cfg = tmp_path / ".mcp.json"
    cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "billing": {
                        "command": "billing-mcp",
                        "args": ["--api-key", fake_token],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    servers = discover_mcp_servers(paths=[cfg])

    dumped = json.dumps([s.model_dump() for s in servers])
    assert fake_token not in dumped
    assert fake_token not in repr(servers)
    s = servers[0]
    assert s.args[0] == "--api-key"
    assert s.args[1] == "[REDACTED]"


def test_discover_mcp_servers_redacts_url_query_secrets(tmp_path):
    fake_token = "abcdef0123456789ABCDEF0123456789"
    cfg = tmp_path / ".mcp.json"
    cfg.write_text(
        json.dumps(
            {"mcpServers": {"remote": {"type": "sse", "url": f"https://mcp.example.com/sse?token={fake_token}"}}}
        ),
        encoding="utf-8",
    )

    servers = discover_mcp_servers(paths=[cfg])

    assert len(servers) == 1
    assert fake_token not in json.dumps(servers[0].model_dump())
    assert servers[0].transport == "sse"
    assert servers[0].command == ""


def test_discover_mcp_servers_reads_codex_toml_config(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[mcp_servers.search]\ncommand = "npx"\nargs = ["-y", "@mcp/search"]\n',
        encoding="utf-8",
    )

    servers = discover_mcp_servers(paths=[cfg])

    assert len(servers) == 1
    assert servers[0].name == "search"
    assert servers[0].command == "npx"
    assert servers[0].source_file == str(cfg)


def test_discover_mcp_servers_skips_malformed_file_not_fatal(tmp_path):
    bad_json = tmp_path / "bad.mcp.json"
    bad_json.write_text("{not json at all", encoding="utf-8")

    good_json = tmp_path / "good.mcp.json"
    good_json.write_text(
        json.dumps({"mcpServers": {"ok": {"command": "ok-bin", "args": []}}}), encoding="utf-8"
    )

    bad_toml = tmp_path / "bad.toml"
    bad_toml.write_text("this is not [valid toml =", encoding="utf-8")

    servers = discover_mcp_servers(paths=[bad_json, good_json, bad_toml])

    assert len(servers) == 1
    assert servers[0].name == "ok"


def test_discover_mcp_servers_skips_missing_files_silently(tmp_path):
    missing = tmp_path / "does-not-exist.mcp.json"

    assert discover_mcp_servers(paths=[missing]) == []


def test_discover_mcp_servers_never_reads_env_values(tmp_path):
    """An entry's `env` mapping (env var VALUES) must never surface anywhere
    in the output, even though `env` is a completely normal, expected key in
    a real mcp.json server entry."""
    secret_env_value = "super-secret-env-value-should-never-leak"
    cfg = tmp_path / ".mcp.json"
    cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "svc": {
                        "command": "svc-bin",
                        "args": ["--flag"],
                        "env": {"SVC_API_KEY": secret_env_value},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    servers = discover_mcp_servers(paths=[cfg])

    assert secret_env_value not in json.dumps(servers[0].model_dump())
    assert not hasattr(servers[0], "env")
