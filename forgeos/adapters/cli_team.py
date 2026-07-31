"""OMCTeamAdapter — the oh-my-claudecode (omc) team-runtime execution backend.

Real API surface, verified against a live `oh-my-claudecode` 4.15.7 install at
`<claude-config-dir>/plugins/cache/omc/oh-my-claudecode/<version>` (2026-07-30) — this is not guessed, and it corrects two claims that were
handed down as "verified" before this file was written:

- The caller's brief (and this repo's `AGENTS.md`) describes the job contract
  as reachable through `node .../bin/oh-my-claudecode.js`. That bin only
  imports `bridge/cli.cjs`, whose `team` subcommand dispatch
  (`src/cli/team.ts`) is what this adapter actually shells out to — so the
  invocation path is right, but the specific subcommand *names* needed
  independent confirmation (below) rather than reuse of the AGENTS.md prose.
- There are *two* different `team.ts`-shaped files in the plugin:
  `src/cli/commands/team.ts` (the interactive, tmux-pane, team-*name*-keyed
  surface: `omc team [N:agent-type[:role]] "task"`, `omc team status <name>`,
  `omc team shutdown <name>`, no `--json` on status/shutdown) and
  `src/cli/team.ts` (the one actually named in this task's brief) which
  additionally exports a **job-id-keyed, fully `--json`-capable** surface:
  `TeamStartInput`/`TeamStartResult`/`TeamJobStatus`/`TeamCleanupResult` are
  literal exported TypeScript interface names in that file, matching the
  contract this repo's `AGENTS.md` already documents field-for-field. This
  adapter uses that surface — it is the only one shaped like an async
  job/poll/cleanup lifecycle, which is what `WorkerAdapter` needs:

  - `omc team start --agent <type>[,<type>...] --task "<text>" [--count N]
    [--name TEAM] [--cwd DIR] [--new-window] [--auto-merge] --json`
    -> `TeamStartResult{jobId, status:"running", pid?}` on stdout.
  - `omc team status <job_id> --json` -> `TeamJobStatus{jobId, status,
    elapsedSeconds, result?, stderr?}`. (Passing a team *name* instead of a
    job id is also accepted by the real CLI but returns a differently-shaped
    tmux/worktree snapshot object — this adapter always starts from a job id
    it minted itself via `start`, so that branch is never exercised here.)
  - `omc team cleanup <job_id> [--grace-ms MS] --json` ->
    `TeamCleanupResult{jobId, message}`. Real cleanup: kills worker tmux
    panes, removes preserved worktrees, deletes team state — never a no-op.
  - `omc team wait <job_id> [--timeout-ms MS] --json` also exists (blocking,
    with its own internal backoff) but is deliberately not used: `send()`
    below needs its *own* configurable poll interval and the ability to stop
    early on `cancel()`, neither of which `wait`'s fixed internal loop gives.

  `--agent` is validated by the CLI (`normalizeAgentType`) against exactly
  six vendor CLI names: claude, codex, gemini, cursor, grok, antigravity —
  *not* the role vocabulary (`executor`, `architect`, `explore`, ...) this
  repo's `registry.py`/`AGENTS.md` use for `WorkerProfile.agent_type`. The
  structured `start` subcommand used here has no `--role` flag at all (only
  the legacy positional `N:agent-type:role` alias threads a role prompt
  through, and a separate code path at that — `commands/team.ts`, not
  `cli/team.ts`). So: whatever string the caller passes as `model_profile`
  is forwarded verbatim as `--agent` (never hardcoded, per the brief), but if
  it is actually a role name rather than one of the six vendor CLIs, the
  real CLI will reject it — surfaced honestly as an `EventKind.ERROR`, not
  silently coerced. Flagging this mismatch for review rather than papering
  over it: reconciling forgeos's role-shaped `agent_type` with omc's
  vendor-shaped `--agent` is a routing-layer decision, not this adapter's to
  make unilaterally.

  A separate MCP server, `bridge/team-mcp.cjs`, exposes near-identical tool
  names (`omc_run_team_start`/`_status`/`_wait`/`_cleanup`) over MCP JSON-RPC
  on stdio. It is real, functional code — but it is not registered in the
  plugin's own `.claude-plugin`/`package.json` `mcpServers` map (only a
  server named `"t"`, `bridge/mcp-server.cjs`, is), and nothing else in the
  bundle imports it. Speaking to it would mean hand-rolling an MCP JSON-RPC
  client for a server this plugin does not itself wire up — the CLI surface
  above is the simpler, actually-reachable path to the same contract.

Subscription vendor CLIs (claude, codex, gemini, ...) that omc's workers
spawn own their own credentials end to end (browser OAuth or their own env
var — see each vendor's own docs). This adapter never reads, copies, or logs
one: it spawns `node` with `env=None` (inherits the operator's own
environment) exactly like `acp.py`/`jcode.py`, and it never constructs an
`env` mapping of its own.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import EventKind, WorkerAdapter, WorkerCapabilities, WorkerEvent, WorkerUsage

_DEFAULT_NODE_COMMAND = "node"

# Discovered at import, never baked in — see `_discover_omc_bin`. Overridable
# via the constructor or $OMC_BIN_PATH.
def _discover_omc_bin() -> str:
    """Locate the omc entrypoint on THIS machine.

    Discovery rather than a baked-in path: the plugin lives under a version-stamped
    directory inside the user's own Claude config, so any literal default is correct
    on exactly one computer. Highest version wins; empty string when absent, which
    `health()` reports honestly rather than failing at spawn time.
    """
    import os as _os
    from pathlib import Path as _Path

    configured = _os.environ.get("OMC_BIN_PATH", "").strip()
    if configured:
        return configured

    config_dir = _os.environ.get("CLAUDE_CONFIG_DIR", "").strip() or _os.path.expanduser("~/.claude")
    cache = _Path(config_dir) / "plugins" / "cache"
    if not cache.is_dir():
        return ""

    candidates = sorted(
        cache.glob("*/oh-my-claudecode/*/bin/oh-my-claudecode.js"),
        key=lambda p: p.parent.parent.name,
        reverse=True,
    )
    return str(candidates[0]) if candidates else ""


_DEFAULT_BIN_PATH = _discover_omc_bin()

_DEFAULT_AGENT_TYPE = "claude"


async def _spawn(cmd: list[str]) -> Any:
    """Start `cmd` as a subprocess, return the `Process` handle immediately.

    Isolated in its own function exactly like `jcode.py`'s and `ollama.py`'s
    `_spawn` and `acp.py`'s `_import_acp`: the one seam this adapter uses to
    reach a real subprocess, so tests substitute it directly — no real
    `node`/omc invocation, ever, in the test suite.
    """
    return await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )


def _parse_json_object(text: str) -> dict | None:
    """Best-effort parse of a `--json` CLI result into a dict.

    Every command this adapter calls prints exactly one `JSON.stringify(...)`
    call on stdout when `--json` is passed (verified against `src/cli/team.ts`
    — `output()` is the only thing each `*Command` function calls). Anything
    that doesn't decode as a JSON object is treated as absent, matching
    `jcode.py`'s `_parse_json_result`, rather than crashing the adapter.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _stringify_result(result: Any) -> str:
    """`TeamJobStatus.result` is typed `unknown` on the TS side — render it as
    text for a canonical `WorkerEvent.text` without assuming a shape."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    return json.dumps(result)


def _slugify_team_name(task_id: str) -> str:
    """A team name omc will accept.

    `validateTeamName` (confirmed via `runtime-cli.cjs`) requires
    `/^[a-z0-9][a-z0-9-]{0,48}[a-z0-9]$/`: lowercase alnum/hyphen, starting
    and ending alnum, <= 50 chars. A random suffix keeps two forgeos tasks that
    happen to slugify identically (or run concurrently) from colliding on
    the same team name.
    """
    base = re.sub(r"[^a-z0-9]+", "-", task_id.lower()).strip("-")
    base = base[:32].rstrip("-") or "task"
    suffix = uuid.uuid4().hex[:8]
    return f"forgeos-{base}-{suffix}"


@dataclass
class _Session:
    """Live state for one forgeos-side omc team session.

    `job_id` starts empty: an omc team job cannot be started without a task
    description (`tasks` is a required, non-empty field of `TeamStartInput`),
    and `WorkerAdapter.start()` is never given one — only `send()`'s `prompt`
    supplies it. So, like `jcode.py`'s `jcode_session_id`, this is filled in
    lazily by the first `send()` call rather than by `start()`. Holds exactly
    what's needed to keep polling and to rebuild a checkpoint later — never
    message content.
    """

    task_id: str
    cwd: str
    model_profile: str
    agent_type: str
    team_name: str
    job_id: str = ""
    cancelled: bool = False
    last_usage: WorkerUsage = field(default_factory=WorkerUsage)


class OMCTeamAdapter(WorkerAdapter):
    """Runs a task through the omc team runtime's job-based CLI surface.

    `node_command`/`bin_path` name the entrypoint to spawn — nothing
    vendor/machine-specific is hardcoded beyond the documented fallback
    default (see `_DEFAULT_BIN_PATH`); both are constructor overrides, and
    `bin_path` also falls back to the `OMC_BIN_PATH` env var. `agentTypes`
    (here, one agent type per forgeos session) always comes from whatever the
    caller passes as `model_profile` to `start()` — the router picks it,
    this adapter never substitutes its own opinion except via the documented
    `default_agent_type` fallback for an empty string.
    """

    def __init__(
        self,
        node_command: str = _DEFAULT_NODE_COMMAND,
        bin_path: str | Path | None = None,
        *,
        default_agent_type: str = _DEFAULT_AGENT_TYPE,
        poll_interval_seconds: float = 2.0,
        cleanup_grace_ms: int = 10_000,
        new_window: bool = False,
        auto_merge: bool = False,
        max_parallel_sessions: int = 4,
    ) -> None:
        self._node_command = node_command
        self._bin_path = Path(bin_path or os.environ.get("OMC_BIN_PATH", _DEFAULT_BIN_PATH))
        self._default_agent_type = default_agent_type
        # "Poll interval must be configurable and default to something sane
        # (>=1s)" — a tight poll loop is wasted CPU on a machine that may
        # need it for a concurrent build (AGENTS.md), so this is clamped,
        # never silently allowed down to 0.
        self._poll_interval_seconds = max(1.0, poll_interval_seconds)
        self._cleanup_grace_ms = cleanup_grace_ms
        self._new_window = new_window
        self._auto_merge = auto_merge
        self._max_parallel_sessions = max_parallel_sessions
        self._sessions: dict[str, _Session] = {}

    # ------------------------------------------------------------- health

    def health(self) -> tuple[bool, str]:
        if not shutil.which(self._node_command):
            return False, f"'{self._node_command}' not found on PATH; install Node.js first"
        # `Path("")` is `Path(".")`, and "." always exists — so an empty discovery
        # result reported the adapter as healthy on every machine WITHOUT omc
        # installed, which is every machine but the author's. Require a real file.
        if not self._bin_path.name or not self._bin_path.is_file():
            return False, (
                f"omc plugin entrypoint not found at '{self._bin_path}'; "
                "install/update the oh-my-claudecode plugin or pass bin_path/OMC_BIN_PATH"
            )
        # On native Windows the omc team runtime hard-requires a tmux-compatible
        # binary (psmux provides one); without it the runtime prints a warning
        # and dies before running any task. Observed live: a "healthy" adapter
        # billed three tier-prior attempts producing empty event streams. A
        # backend that cannot execute must say so HERE, not at spawn time.
        if sys.platform == "win32" and not shutil.which("tmux"):
            return False, (
                "omc team runtime needs a tmux-compatible binary on native "
                "Windows and none is on PATH; install psmux "
                "(winget install psmux) or run under WSL2"
            )
        return True, f"'{self._node_command}' on PATH and omc entrypoint present at '{self._bin_path}'"

    def capabilities(self) -> WorkerCapabilities:
        return WorkerCapabilities(
            structured_messages=False,  # only coarse job status; no live per-turn stream through this surface
            resumable_sessions=True,  # job_id outlives any one forgeos session; checkpoint/resume rehydrate it
            tool_calls=False,  # no individual tool-call visibility through the job/status JSON
            file_diffs=True,  # workers run in isolated git worktrees (AGENTS.md); diffs are inspectable there
            reasoning_effort=False,
            token_usage=False,  # TeamStartResult/TeamJobStatus never carry a token or cost field
            images=False,
            max_parallel_sessions=self._max_parallel_sessions,
        )

    # ------------------------------------------------------------ sessions

    def _get_session(self, session_id: str) -> _Session:
        try:
            return self._sessions[session_id]
        except KeyError:
            raise KeyError(f"no active omc team session: {session_id}") from None

    async def start(self, task_id: str, cwd: str, model_profile: str) -> str:
        # No omc job exists yet (see `_Session`'s docstring) — this only
        # opens the local bookkeeping a forgeos session_id needs. `team_name`
        # can be resolved now (it only depends on task_id); `job_id` cannot.
        session_id = uuid.uuid4().hex
        agent_type = model_profile.strip() if model_profile and model_profile.strip() else self._default_agent_type
        self._sessions[session_id] = _Session(
            task_id=task_id,
            cwd=cwd,
            model_profile=model_profile,
            agent_type=agent_type,
            team_name=_slugify_team_name(task_id),
        )
        return session_id

    async def _run_cli(self, args: list[str]) -> tuple[bool, int, str, str]:
        """Spawn `node <bin_path> <args...>`, wait for completion.

        Returns `(ok, returncode, stdout, stderr)`. `ok` is False only when
        the subprocess itself could not be spawned or communicated with
        (missing binary, killed mid-flight); a clean non-zero exit from
        omc's own CLI is still `ok=True` with a non-zero `returncode`, so
        callers can tell "omc ran and reported failure" apart from "we never
        reached omc" — same split `jcode.py`/`ollama.py` make around
        `_spawn` vs. `communicate()`.
        """
        cmd = [self._node_command, str(self._bin_path), *args]
        try:
            proc = await _spawn(cmd)
        except OSError as exc:
            return False, -1, "", str(exc)
        try:
            stdout, stderr = await proc.communicate()
        except BaseException as exc:  # noqa: BLE001 - surfaced by the caller as a canonical event, not swallowed
            return False, -1, "", str(exc)
        returncode = proc.returncode if proc.returncode is not None else -1
        return (
            True,
            returncode,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
        )

    async def _start_job(self, sess: _Session, prompt: str) -> AsyncIterator[WorkerEvent]:
        args = [
            "team",
            "start",
            "--agent",
            sess.agent_type,
            "--task",
            prompt,
            "--name",
            sess.team_name,
            "--cwd",
            sess.cwd,
            "--json",
        ]
        if self._new_window:
            args.append("--new-window")
        if self._auto_merge:
            args.append("--auto-merge")

        ok, returncode, stdout, stderr = await self._run_cli(args)
        if not ok:
            yield WorkerEvent(kind=EventKind.ERROR, text=stderr or "failed to spawn the omc CLI")
            return
        if returncode != 0:
            yield WorkerEvent(kind=EventKind.ERROR, text=stderr.strip() or f"omc team start exited {returncode}")
            return

        result = _parse_json_object(stdout)
        if result is None or "jobId" not in result:
            yield WorkerEvent(kind=EventKind.ERROR, text=f"could not parse omc team start output: {stdout!r}")
            return

        sess.job_id = str(result["jobId"])
        yield WorkerEvent(
            kind=EventKind.META,
            data={"jobId": sess.job_id, "status": result.get("status", "running"), "pid": result.get("pid")},
        )

    async def _poll_until_terminal(self, sess: _Session) -> AsyncIterator[WorkerEvent]:
        while True:
            if sess.cancelled:
                yield WorkerEvent(kind=EventKind.DONE, status="cancelled")
                return

            ok, returncode, stdout, stderr = await self._run_cli(["team", "status", sess.job_id, "--json"])
            if not ok:
                yield WorkerEvent(kind=EventKind.ERROR, text=stderr or "failed to reach the omc CLI")
                return
            if returncode != 0:
                yield WorkerEvent(kind=EventKind.ERROR, text=stderr.strip() or f"omc team status exited {returncode}")
                return

            status = _parse_json_object(stdout)
            if status is None:
                yield WorkerEvent(kind=EventKind.ERROR, text=f"could not parse omc team status output: {stdout!r}")
                return

            job_status = str(status.get("status") or "unknown")
            if job_status == "running":
                yield WorkerEvent(
                    kind=EventKind.META,
                    data={"jobId": sess.job_id, "status": "running", "elapsedSeconds": status.get("elapsedSeconds")},
                )
                await asyncio.sleep(self._poll_interval_seconds)
                continue

            stderr_text = status.get("stderr")
            if job_status == "failed" and stderr_text:
                yield WorkerEvent(kind=EventKind.ERROR, text=str(stderr_text))

            yield WorkerEvent(kind=EventKind.MESSAGE, text=_stringify_result(status.get("result")))
            yield WorkerEvent(kind=EventKind.DONE, status=job_status)
            return

    async def send(self, session_id: str, prompt: str) -> AsyncIterator[WorkerEvent]:
        sess = self._get_session(session_id)

        if not sess.job_id:
            async for event in self._start_job(sess, prompt):
                yield event
            if not sess.job_id:
                return  # start failed; the ERROR event above already explains why
        elif prompt:
            # omc team jobs are a one-shot batch, not a multi-turn
            # conversation (see module docstring) — a second `send()` with
            # new text cannot be delivered mid-job. Say so instead of
            # silently discarding it.
            yield WorkerEvent(
                kind=EventKind.META,
                data={
                    "jobId": sess.job_id,
                    "note": "omc team jobs are one-shot; new prompt ignored, re-polling the existing job",
                },
            )

        async for event in self._poll_until_terminal(sess):
            yield event

    async def cancel(self, session_id: str) -> None:
        sess = self._get_session(session_id)
        sess.cancelled = True
        if sess.job_id:
            await self._cleanup_job(sess.job_id)

    async def _cleanup_job(self, job_id: str) -> None:
        """Release omc's own tmux panes / worktrees / job-state for `job_id`.

        `_run_cli` never raises (see its docstring), so there is nothing to
        guard here — `close()`/`cancel()` stay unconditionally safe to call.
        """
        await self._run_cli(["team", "cleanup", job_id, "--grace-ms", str(self._cleanup_grace_ms), "--json"])

    async def checkpoint(self, session_id: str) -> dict:
        """Canonical task state only — no message content of any kind."""
        sess = self._get_session(session_id)
        return {
            "adapter": "cli_team",
            "node_command": self._node_command,
            "bin_path": str(self._bin_path),
            "task_id": sess.task_id,
            "cwd": sess.cwd,
            "model_profile": sess.model_profile,
            "agent_type": sess.agent_type,
            "team_name": sess.team_name,
            "job_id": sess.job_id,
        }

    async def resume(self, checkpoint: dict) -> str:
        # A replacement worker resumes from this canonical state, never from
        # a transcript (none exists here to begin with — see module
        # docstring). The same `job_id` keeps being polled regardless of
        # which adapter instance/process asks `omc team status` about it.
        session_id = uuid.uuid4().hex
        task_id = checkpoint["task_id"]
        self._sessions[session_id] = _Session(
            task_id=task_id,
            cwd=checkpoint["cwd"],
            model_profile=checkpoint.get("model_profile", ""),
            agent_type=checkpoint.get("agent_type") or self._default_agent_type,
            team_name=checkpoint.get("team_name") or _slugify_team_name(task_id),
            job_id=checkpoint.get("job_id", ""),
        )
        return session_id

    async def usage(self, session_id: str) -> WorkerUsage:
        # Never set anywhere else in this adapter: the omc team CLI's
        # TeamStartResult/TeamJobStatus/TeamCleanupResult shapes (confirmed
        # against `src/cli/team.ts`) carry no token or dollar figure at all,
        # so an honest zero/`exact=False` is the only thing this backend is
        # entitled to report.
        return self._get_session(session_id).last_usage

    async def close(self, session_id: str) -> None:
        sess = self._sessions.pop(session_id, None)
        if sess is None:
            return
        if sess.job_id:
            await self._cleanup_job(sess.job_id)


__all__ = ["OMCTeamAdapter"]
