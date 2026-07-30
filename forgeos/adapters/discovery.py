"""Capability discovery — introspect a CLI instead of hardcoding what it can do.

WHY THIS EXISTS: `settings.py`'s `_CLI_DEFAULTS` and every adapter's declared
capabilities are a snapshot, dated the day someone read `--help` by hand. The
moment a vendor ships an update that renames a flag or drops a subcommand,
that snapshot is silently wrong, and an adapter that trusts it invokes a
command that no longer exists — a confusing runtime failure, not a caught
one. This module replaces "someone remembered" with "we asked the binary,
just now, and cached the answer against the version that produced it."

Two rules this module holds itself to, matching AGENTS.md's "never
blind-read" and "nothing becomes a rule because someone said it":

1. **Never invent a flag.** A capability is `True` only if it is evidenced in
   the actual `--help`/`--version` output captured *this run* (or a prior run
   against the *same version*, via `ProbeCache`). If the evidence is not
   there, the field is `False` — that is the honest answer, not a gap to
   guess-fill. `ToolProbe.probe_source` names the exact command whose output
   justified the conclusions, so a wrong probe is checkable against real
   output, not trusted on faith.
2. **A version bump forces a re-probe.** `ProbeCache` keys on (name, version).
   Vendor ships an update -> version string changes -> cache miss -> fresh
   probe. There is no manual step where someone has to remember to refresh
   this file.

Subprocess isolation matches the rest of the adapters package
(`jcode.py`'s `_spawn`, `ollama.py`'s bare `subprocess.run` calls): every
shell-out in this module goes through the single `_run` seam below, so
tests substitute `discovery.subprocess.run` directly and never spawn a real
process. No CLI vendor's binary is ever actually invoked by the test suite.

This module never constructs an `env` mapping for a probed subprocess and
never reads a credential's value — mirroring the auth policy in `settings.py`
and `jcode.py`: a vendor CLI's login state is its own business, not
something this module inspects or stores.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tomllib
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from .base import WorkerCapabilities

PROBE_CACHE_PATH = Path(os.path.expanduser("~/.forgeos/probes.json"))

DEFAULT_TIMEOUT = 10  # seconds; --help/--version, never a real agent turn

# The exact roster from settings.py's _CLI_DEFAULTS, kept here as a convenience
# for callers of discover_all — not a hardcoded capability, just "which CLIs
# does forgeos know the *names* of", which is a much weaker and safer claim.
KNOWN_CLI_NAMES: tuple[str, ...] = (
    "claude", "codex", "opencode", "qwen", "kimi", "grok", "pi",
    "droid", "copilot", "ollama", "hermes",
)

_REDACTED = "[REDACTED]"


# ============================================================== ToolProbe


class ToolProbe(BaseModel):
    """What was actually observed about one CLI, and where it came from.

    Every boolean here answers "did the evidence say so", never "does this
    seem like the kind of tool that would have this". `probe_source` names
    the command whose stdout/stderr backs `subcommands`, `flags`,
    `has_headless`, `headless_invocation`, `supports_json`, `supports_resume`,
    and `supports_mcp` — all of those are derived from the same `--help`
    invocation. `version` comes from a separate `--version` call and is not
    itself a capability claim.
    """

    name: str
    path: str
    version: str = ""
    subcommands: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    has_headless: bool = False
    headless_invocation: str = ""
    supports_json: bool = False
    supports_resume: bool = False
    supports_mcp: bool = False
    probed_at: str = ""
    probe_source: str = ""


# ============================================================== McpServer


class McpServer(BaseModel):
    """One MCP server entry read from a config file already on this machine.

    Only `name`, `command`, `args` (redacted), `source_file`, and `transport`
    are ever carried. An entry's `env` mapping (env var *values*) and any
    `headers` mapping (auth headers) are never read into this model at all —
    not redacted-then-kept, simply never touched — because the instruction
    this module follows is "read the name and command only", and the safest
    way to honor "never a token/key/header value/env var value" is to never
    let that data enter the process's own memory as anything more than an
    unopened dict key we skip past.
    """

    name: str
    command: str = ""
    args: list[str] = Field(default_factory=list)
    source_file: str
    transport: str = "stdio"


# ========================================================= subprocess seam


def _run(cmd: list[str], *, timeout: int = DEFAULT_TIMEOUT) -> subprocess.CompletedProcess | None:
    """The one seam every subprocess call in this module goes through.

    Never raises: a missing dependency, a hung `--help`, or a binary that
    exits non-zero for `--help` (common — many CLIs treat --help as a usage
    error) must degrade discovery, not the caller. `None` return means "no
    evidence available", which every parser below already treats as empty
    input rather than a crash.
    """
    try:
        return subprocess.run(  # noqa: S603 - fixed argv, no shell
            cmd, capture_output=True, text=True, timeout=timeout
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _combined_output(result: subprocess.CompletedProcess | None) -> str:
    if result is None:
        return ""
    return f"{result.stdout or ''}\n{result.stderr or ''}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ================================================================ parsing

_VERSION_RE = re.compile(r"(\d+\.\d+\.\d+(?:[.\-][\w.]+)?)")

_COMMANDS_HEADERS = ("commands", "subcommands", "available commands")

_FLAG_TOKEN_RE = re.compile(r"(-{1,2}[A-Za-z][\w-]*)")

_HEADLESS_SUBCOMMANDS = ("exec", "run")
_HEADLESS_FLAGS = ("-p", "--print", "--non-interactive")
_HEADLESS_PHRASE = "non-interactively"


def _parse_version(text: str) -> str:
    if not text:
        return ""
    m = _VERSION_RE.search(text)
    return m.group(1) if m else ""


def _parse_subcommands(help_text: str) -> list[str]:
    """First token of each indented line inside a `Commands:`-style section.

    A section starts at a line whose stripped, colon-stripped text matches
    one of `_COMMANDS_HEADERS` and ends at the first blank line or the first
    line that is not indented (the next section header) — matching the
    layout every clap/argparse/commander-style `--help` actually produces.
    """
    lines = help_text.splitlines()
    out: list[str] = []
    seen: set[str] = set()
    in_section = False
    for line in lines:
        stripped = line.strip()
        if not in_section:
            header = stripped.rstrip(":").strip().lower()
            if stripped.endswith(":") and header in _COMMANDS_HEADERS:
                in_section = True
            continue
        if not stripped:
            break
        if not line[:1].isspace():
            break
        token = stripped.split(None, 1)[0].rstrip(",")
        if token and not token.startswith("-") and token not in seen:
            seen.add(token)
            out.append(token)
    return out


def _parse_flags(help_text: str) -> list[str]:
    """Every flag token on a line whose stripped text starts with `-`.

    Deliberately not scoped to an `Options:` header — some CLIs list flags
    under `Flags:`, `Global Options:`, or with no header at all next to usage
    text. "a line starting with `-`" is the one signal that holds everywhere.
    """
    out: list[str] = []
    seen: set[str] = set()
    for line in help_text.splitlines():
        if not line.strip().startswith("-"):
            continue
        for tok in _FLAG_TOKEN_RE.findall(line):
            if tok not in seen:
                seen.add(tok)
                out.append(tok)
    return out


def _detect_headless(
    name: str, help_text: str, subcommands: list[str], flags: list[str]
) -> tuple[bool, str]:
    """Headless mode is claimed True only if a known marker literally appears
    in the evidence: a subcommand named `exec`/`run`, a flag among
    `-p`/`--print`/`--non-interactive`, or the phrase "non-interactively"
    somewhere in the help text. No marker found means `False` — this
    function never assumes a coding-agent CLI has a headless mode just
    because most of them do.
    """
    invocation = ""
    found = False

    for sub in _HEADLESS_SUBCOMMANDS:
        if sub in subcommands:
            found = True
            if not invocation:
                invocation = f"{name} {sub}"

    for flag in _HEADLESS_FLAGS:
        if flag in flags:
            found = True
            if not invocation:
                invocation = f"{name} {flag}"

    if _HEADLESS_PHRASE in help_text.lower():
        found = True

    return found, invocation


# ============================================================== probe_cli


def probe_cli(name: str, *, timeout: int = DEFAULT_TIMEOUT) -> ToolProbe | None:
    """Introspect one CLI by actually running it. `None` if it is not on PATH
    — a missing tool is a normal, expected outcome here, never an exception.
    """
    path = shutil.which(name)
    if path is None:
        return None

    help_cmd = [name, "--help"]
    version_cmd = [name, "--version"]

    help_result = _run(help_cmd, timeout=timeout)
    version_result = _run(version_cmd, timeout=timeout)

    help_text = _combined_output(help_result)
    version_text = _combined_output(version_result)

    version = _parse_version(version_text) or _parse_version(help_text)
    subcommands = _parse_subcommands(help_text)
    flags = _parse_flags(help_text)
    has_headless, headless_invocation = _detect_headless(name, help_text, subcommands, flags)

    return ToolProbe(
        name=name,
        path=path,
        version=version,
        subcommands=subcommands,
        flags=flags,
        has_headless=has_headless,
        headless_invocation=headless_invocation,
        supports_json="--json" in flags,
        supports_resume=("resume" in subcommands) or ("--resume" in flags),
        supports_mcp=("mcp" in subcommands) or ("--mcp" in flags),
        probed_at=_now_iso(),
        probe_source=" ".join(help_cmd) if help_result is not None else "unavailable",
    )


def discover_all(
    names: Iterable[str], *, cache: "ProbeCache | None" = None
) -> dict[str, ToolProbe | None]:
    """Probe every name in `names`. A missing tool maps to `None`, never an
    exception — one CLI absent from this machine must not stop discovery of
    the rest.

    `cache` is optional. When given, a cheap `--version`-only call decides
    whether a full re-probe is needed at all: same version as last time means
    the cached `ToolProbe` is reused verbatim; a version bump (or first sight
    of this tool) triggers a full probe, which is then written back.
    """
    results: dict[str, ToolProbe | None] = {}
    for name in names:
        if cache is None:
            results[name] = probe_cli(name)
            continue

        path = shutil.which(name)
        if path is None:
            results[name] = None
            continue

        version = _parse_version(_combined_output(_run([name, "--version"])))
        cached = cache.get(name, version)
        if cached is not None:
            results[name] = cached
            continue

        probe = probe_cli(name)
        if probe is not None:
            cache.put(probe)
        results[name] = probe
    return results


# ======================================================= capability mapping


def capabilities_from_probe(probe: ToolProbe) -> WorkerCapabilities:
    """Map evidenced facts onto the adapter contract (`base.WorkerCapabilities`).

    Deliberately conservative: only `structured_messages` (from `--json`
    evidence) and `resumable_sessions` (from `--resume`/`resume` evidence) are
    set from the probe. `tool_calls`, `file_diffs`, `reasoning_effort`,
    `token_usage`, and `images` describe what a *live wire protocol* carries
    mid-session — `--help` text cannot honestly evidence any of that, the
    same gap `jcode.py` documents for its own `--trace`/`--ndjson` output. An
    adapter author who needs those fields still has to verify them against a
    real session, same as before; this function does not manufacture a
    confident-looking `True` to save that work.
    """
    return WorkerCapabilities(
        structured_messages=probe.supports_json,
        resumable_sessions=probe.supports_resume,
    )


# ==================================================================== cache


class ProbeCache:
    """Persists `ToolProbe`s to disk, keyed by (tool name, version).

    `get()` only ever returns a hit when the stored version matches the
    version asked for — that is the entire mechanism by which a vendor update
    invalidates the cache automatically: the version string changes, the key
    changes, `get()` misses, and the caller re-probes. A corrupt cache file
    is treated the same way `Settings.load()` treats a corrupt settings
    file (settings.py) — fall back to empty, never crash the caller.
    """

    def __init__(self, path: str | Path = PROBE_CACHE_PATH) -> None:
        self.path = Path(path)
        self._data: dict[str, dict] = self._load()

    def _load(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return raw if isinstance(raw, dict) else {}

    @staticmethod
    def _key(name: str, version: str) -> str:
        return f"{name}@{version}"

    def get(self, name: str, version: str) -> ToolProbe | None:
        if not version:
            return None
        entry = self._data.get(self._key(name, version))
        if entry is None:
            return None
        try:
            return ToolProbe.model_validate(entry)
        except Exception:
            return None

    def put(self, probe: ToolProbe) -> None:
        if not probe.version:
            return
        self._data[self._key(probe.name, probe.version)] = probe.model_dump(mode="json")
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")


# ========================================================= MCP server config

_CRED_KEY_RE = re.compile(
    r"(api[-_]?key|token|secret|password|passwd|auth(?:orization)?|bearer|credential|cookie)",
    re.IGNORECASE,
)
_CRED_PREFIX_RE = re.compile(r"^(sk-|gh[pousr]_|xox[baprs]-|eyJ|AIza|glpat-)")
_OPAQUE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_\-.]+$")

_MCP_SERVER_KEYS = ("mcpServers", "mcp_servers")


def _looks_like_secret(value: str) -> bool:
    if not value:
        return False
    if _CRED_PREFIX_RE.match(value):
        return True
    if (
        len(value) >= 20
        and _OPAQUE_TOKEN_RE.match(value)
        and any(c.isdigit() for c in value)
        and any(c.isalpha() for c in value)
    ):
        return True
    return False


def _redact_args(raw_args: list) -> list[str]:
    out: list[str] = []
    redact_next = False
    for raw in raw_args:
        a = "" if raw is None else str(raw)
        if redact_next:
            out.append(_REDACTED)
            redact_next = False
            continue
        if "=" in a and a.startswith("-"):
            flag, _, value = a.partition("=")
            if _CRED_KEY_RE.search(flag) or _looks_like_secret(value):
                out.append(f"{flag}={_REDACTED}")
            else:
                out.append(a)
            continue
        if a.startswith("-") and _CRED_KEY_RE.search(a):
            out.append(a)
            redact_next = True
            continue
        out.append(_REDACTED if _looks_like_secret(a) else a)
    return out


def _redact_url(url: str) -> str:
    return re.sub(
        r"(?i)([?&](?:api[-_]?key|token|secret|password|auth)=)[^&]+",
        lambda m: f"{m.group(1)}{_REDACTED}",
        url,
    )


def _find_mcp_server_dicts(obj: object) -> list[dict]:
    """Recursively find every `mcpServers`/`mcp_servers` mapping anywhere in a
    parsed config document and flatten them into raw `{name, **cfg}` dicts.

    Recursive rather than "look only at the top level" because
    `~/.claude.json` nests per-project server config at unpredictable depth
    (per-project blocks keyed by project path) — a fixed-depth lookup would
    silently miss those and under-report what is actually configured.
    """
    found: list[dict] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in _MCP_SERVER_KEYS and isinstance(value, dict):
                for server_name, server_cfg in value.items():
                    if isinstance(server_cfg, dict):
                        found.append({"name": server_name, **server_cfg})
            else:
                found.extend(_find_mcp_server_dicts(value))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(_find_mcp_server_dicts(item))
    return found


def _mcp_server_from_entry(entry: dict, source_file: str) -> McpServer:
    name = str(entry.get("name", "unknown"))
    command = entry.get("command")
    command = str(command) if command is not None else ""

    raw_args = entry.get("args")
    raw_args = raw_args if isinstance(raw_args, list) else []
    args = _redact_args(raw_args)

    transport = str(entry.get("type")) if entry.get("type") else "stdio"

    url = entry.get("url")
    if isinstance(url, str) and url:
        transport = str(entry.get("type") or "http")
        args = [_redact_url(url)]
        command = ""

    return McpServer(name=name, command=command, args=args, source_file=source_file, transport=transport)


def _default_mcp_config_paths() -> list[Path]:
    home = Path(os.path.expanduser("~"))
    return [
        home / ".claude" / ".mcp.json",
        home / ".claude.json",
        Path.cwd() / ".mcp.json",
        home / ".codex" / "config.toml",
    ]


def discover_mcp_servers(paths: list[str | Path] | None = None) -> list[McpServer]:
    """Read every MCP config file that actually exists among `paths` (default:
    the four real locations this method is documented to check) and return
    the servers declared in them.

    Only `name`/`command`/`args`/`source_file`/`transport` ever leave this
    function. `args` is redacted for anything credential-shaped. `env` and
    `headers` mappings inside a server entry are never read at all. A
    missing file is skipped silently; a file that exists but fails to parse
    (bad JSON, bad TOML, or a `.toml` file when `tomllib` is unavailable) is
    also skipped — one malformed config must not stop discovery of the rest.
    """
    candidates = [Path(p) for p in paths] if paths is not None else _default_mcp_config_paths()
    servers: list[McpServer] = []

    for p in candidates:
        if not p.exists() or not p.is_file():
            continue
        source = str(p)
        try:
            if p.suffix == ".toml":
                with p.open("rb") as fh:
                    data = tomllib.load(fh)
            else:
                data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        for raw in _find_mcp_server_dicts(data):
            try:
                servers.append(_mcp_server_from_entry(raw, source))
            except Exception:
                continue

    return servers


__all__ = [
    "DEFAULT_TIMEOUT",
    "KNOWN_CLI_NAMES",
    "PROBE_CACHE_PATH",
    "McpServer",
    "ProbeCache",
    "ToolProbe",
    "capabilities_from_probe",
    "discover_all",
    "discover_mcp_servers",
    "probe_cli",
]
