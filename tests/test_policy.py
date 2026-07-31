"""Policy gate tests.

Half of these assert that the gate BLOCKS, and half assert that it gets out of the
way. The second half matters as much: a gate that hides data an agent needs is worse
than the waste it prevents, because the agent routes around it more expensively.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from forgeos.policy import (
    MAX_UNBOUNDED_BYTES,
    Decision,
    ReadRequest,
    Verdict,
    check_read,
    check_tool_call,
    looks_like_secret,
    ripper_command,
)

HOOK = Path(__file__).resolve().parent.parent / "hooks" / "ripper_gate.py"


def _req(path: str, **kw) -> ReadRequest:
    kw.setdefault("size_bytes", 1024)
    return ReadRequest(path=path, **kw)


# ------------------------------------------------------- the escape hatch


def test_bounded_read_of_a_huge_file_is_allowed():
    """The load-bearing exception. Naming a range IS the discipline the gate wants."""
    d = check_read(_req("core/huge.py", size_bytes=50 * 1024 * 1024, offset=1200, limit=80))
    assert d.allowed


def test_limit_alone_counts_as_bounded():
    assert check_read(_req("big.py", size_bytes=10**7, limit=50)).allowed


def test_offset_alone_counts_as_bounded():
    assert check_read(_req("big.py", size_bytes=10**7, offset=500)).allowed


def test_offset_zero_is_still_bounded():
    """0 is a real offset, not a missing one — falsy-checking this would break it."""
    assert check_read(_req("big.py", size_bytes=10**7, offset=0, limit=10)).allowed


def test_small_unbounded_read_is_allowed():
    assert check_read(_req("forgeos/policy.py", size_bytes=8000)).allowed


def test_file_exactly_at_the_limit_is_allowed():
    assert check_read(_req("edge.py", size_bytes=MAX_UNBOUNDED_BYTES)).allowed


def test_unknown_tools_are_allowed():
    """A gate that blocks what it does not recognise turns every new tool into an outage."""
    assert check_tool_call("SomeFutureTool", {"file_path": "x"}).allowed


def test_missing_path_is_allowed_not_crashed():
    assert check_tool_call("Read", {}).allowed


def test_unstattable_path_is_not_blocked_on_size():
    """Cannot measure it, so do not invent a reason to block it."""
    d = check_read(ReadRequest(path="does/not/exist/anywhere.py"))
    assert d.allowed


# --------------------------------------------------------------- the gate


def test_unbounded_read_over_the_limit_is_denied():
    d = check_read(_req("data/huge.json", size_bytes=MAX_UNBOUNDED_BYTES + 1))
    assert not d.allowed
    assert d.exit_code == 2


def test_denial_carries_a_copy_pasteable_replacement_command(tmp_path, monkeypatch):
    """A denial without a next step just gets worked around.

    The digest tool is optional-and-detected, so the test injects one rather
    than asserting this machine happens to have it — the original assertion
    passed only on the one machine with ~/.codex-brain and failed on every
    clean CI runner.
    """
    tool = tmp_path / "digest.py"
    tool.write_text("# fake digest tool")
    monkeypatch.setenv("FORGEOS_DIGEST_TOOL", str(tool))
    d = check_read(_req("core/big.py", size_bytes=5 * 1024 * 1024))
    assert "digest.py" in d.suggestion
    assert "core/big.py" in d.suggestion
    assert "offset" in d.suggestion


def test_denial_without_a_digest_tool_still_gives_a_runnable_next_step(tmp_path, monkeypatch):
    """On a machine with no external tool the fallback advice must stand alone —
    suggesting a command that does not exist reads as a broken tool, not a rule."""
    monkeypatch.setenv("FORGEOS_DIGEST_TOOL", "")
    monkeypatch.setattr("forgeos.policy.os.path.expanduser",
                        lambda p: str(tmp_path / "nohome"))
    d = check_read(_req("core/big.py", size_bytes=5 * 1024 * 1024))
    assert "digest.py" not in d.suggestion
    assert "core/big.py" in d.suggestion
    assert "offset" in d.suggestion


def test_denial_quantifies_the_waste_it_prevented():
    d = check_read(_req("big.py", size_bytes=400 * 1024))
    assert "KB" in d.reason
    assert "tokens" in d.reason


@pytest.mark.parametrize(
    "path",
    [
        "node_modules/react/index.js",
        "project/.git/COMMIT_EDITMSG",
        "app/__pycache__/mod.cpython-311.pyc",
        ".venv/lib/site-packages/pydantic/main.py",
        "frontend/dist/bundle.js",
        "rust/target/debug/build.rs",
        "vendor/OmniRoute/src/index.ts",
    ],
)
def test_generated_and_vendored_paths_are_denied_regardless_of_size(path):
    assert not check_read(_req(path, size_bytes=200)).allowed


@pytest.mark.parametrize(
    "name",
    ["package-lock.json", "yarn.lock", "poetry.lock", "Cargo.lock", "uv.lock"],
)
def test_lockfiles_are_denied_and_point_at_the_manifest(name):
    d = check_read(_req(f"repo/{name}", size_bytes=900))
    assert not d.allowed
    assert "manifest" in d.suggestion.lower()


@pytest.mark.parametrize("suffix", [".pyc", ".so", ".zip", ".png", ".mp4", ".sqlite"])
def test_binary_suffixes_are_denied(suffix):
    assert not check_read(_req(f"assets/thing{suffix}", size_bytes=100)).allowed


# ------------------------------------------------------------- secrets


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        "project/.env",
        "config/.env.local",
        "~/.ssh/id_rsa",
        "certs/server.pem",
        "keys/private_key",
        "home/.netrc",
        "app/credentials.json",
        "vault/secrets.yaml",
    ],
)
def test_credential_files_are_denied_regardless_of_size(path):
    """A small secret is still a secret, so size gating must not reach these."""
    d = check_read(_req(path, size_bytes=50))
    assert not d.allowed
    assert "secret" in d.reason.lower() or "credential" in d.reason.lower()


def test_secret_denial_beats_the_bounded_escape_hatch():
    """Naming a line range must not be a way to read a credential."""
    d = check_read(_req(".env", size_bytes=50, offset=0, limit=5))
    assert not d.allowed


def test_looks_like_secret_does_not_fire_on_ordinary_names():
    assert not looks_like_secret("forgeos/environment.py")
    assert not looks_like_secret("docs/keyboard.md")
    assert looks_like_secret("app/.env")


# ------------------------------------------------------ tool-call dispatch


def test_read_payload_is_gated():
    d = check_tool_call("Read", {"file_path": "node_modules/x/index.js"})
    assert not d.allowed


def test_read_payload_with_range_passes():
    d = check_tool_call("Read", {"file_path": "forgeos/ledger.py", "offset": 10, "limit": 40})
    assert d.allowed


def test_camel_case_payload_keys_are_accepted():
    """Harnesses differ on snake_case vs camelCase; the gate must not silently
    allow everything because it did not recognise the key names."""
    d = check_tool_call("Read", {"file_path": ".env"})
    assert not d.allowed


# --------------------------------------------------------- the hook itself


def _run_hook(payload: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.slow
def test_hook_exits_2_on_a_denied_read():
    """Exit 2 is the whole mechanism — anything else is merely advice."""
    r = _run_hook({"tool_name": "Read", "tool_input": {"file_path": "node_modules/a/b.js"}})
    assert r.returncode == 2
    assert "DENIED" in r.stderr
    assert "instead" in r.stderr.lower()


@pytest.mark.slow
def test_hook_exits_0_on_an_allowed_read():
    r = _run_hook({"tool_name": "Read", "tool_input": {"file_path": "forgeos/policy.py",
                                                       "offset": 1, "limit": 5}})
    assert r.returncode == 0


@pytest.mark.slow
def test_hook_fails_open_on_garbage_input():
    """A broken gate must not become an outage."""
    r = subprocess.run(
        [sys.executable, str(HOOK)], input="not json at all",
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0


@pytest.mark.slow
def test_hook_fails_open_on_empty_input():
    r = subprocess.run(
        [sys.executable, str(HOOK)], input="", capture_output=True, text=True, timeout=60
    )
    assert r.returncode == 0


@pytest.mark.slow
def test_hook_mentions_the_escape_hatch_when_denying():
    """The agent must learn the way through, not just the wall."""
    r = _run_hook({"tool_name": "Read", "tool_input": {"file_path": "x/package-lock.json"}})
    assert r.returncode == 2
    assert "offset" in r.stderr


# ------------------------------------------------------------- plumbing


def test_decision_exit_codes():
    assert Decision(verdict=Verdict.ALLOW).exit_code == 0
    assert Decision(verdict=Verdict.DENY).exit_code == 2


def test_ripper_command_quotes_paths_with_spaces():
    cmd = ripper_command("C:/Some Folder/file.py")
    assert '"C:/Some Folder/file.py"' in cmd
