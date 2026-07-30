"""JcodeAdapter — the fast, free, local execution backend (`jcode.local`).

jcode is a Rust binary (`jcode.exe`, v0.58.0 verified 2026-07-30, on PATH as
`jcode`) that drives coding-agent turns against a Claude Max or ChatGPT Pro
*subscription* — not a metered per-token API. That is exactly why
`registry.default_registry()` pins `jcode.local` at `CostTier.FREE` and notes
it as "first choice for mechanical edits": anything this binary can finish
must never reach a metered model (AGENTS.md).

Real CLI surface, verified by running `jcode --help` and `jcode run --help`
on this machine (2026-07-30) — nothing below is guessed:

- `jcode run <MESSAGE> [OPTIONS]` — "Run a single message and exit". This
  *is* jcode's headless mode. Unlike ACP (`acp.py`), there is no separate
  long-lived "start a session, then prompt it repeatedly over one connection"
  primitive — every `send()` below is its own `jcode run ...` subprocess.
- `--json` — "Emit a machine-readable JSON result instead of streaming text".
  `--help` guarantees the *outer* shape is a JSON object; it does not document
  the field names inside it, and actually invoking a real coding-agent turn
  to reverse-engineer them was out of scope here (it would spend a real
  subscription turn and could write real files) — see `_parse_json_result`
  and `_extract_file_diffs` for how this adapter stays honest about that gap.
- `--cwd <CWD>` (`-C`) — working directory for the local client process.
- `--model <MODEL>` (`-m`) — model to use.
- `--resume [<RESUME>]` — "Resume a session by ID, or list sessions if no ID
  provided". This is the real, verified primitive `checkpoint()`/`resume()`
  are built on — jcode itself owns session identity and continuity, hive
  only carries the id.
- `--quiet` — "Suppress non-error CLI/status output for scripting and
  wrappers"; used so stdout is just the `--json` result.
- `--trace` exists ("Log tool inputs/outputs and token usage to stderr") but
  is deliberately not used: it is a human-readable trace log, not a
  documented structured schema, so this adapter does not parse it and does
  not claim token usage it cannot actually measure (see `capabilities()`).
- `--ndjson` ("Emit newline-delimited JSON events while the response
  streams") also exists but its per-event schema is undocumented and, same
  as above, was not reverse-engineered against a real run. `send()` therefore
  yields one MESSAGE (plus opportunistic FILE_DIFF events) and one DONE per
  call rather than a live multi-event stream.

jcode is a subscription CLI that owns its own auth end to end (browser OAuth
or its own credential store — see `jcode login`/`jcode account`). This
adapter never calls `jcode login`/`jcode auth`, never reads its credential
store, and never constructs an `env` mapping for the subprocess — the same
policy `ACPAdapter` (`acp.py`) applies to vendor ACP CLIs, for the same
reason: authentication is entirely the vendor CLI's problem.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from .base import EventKind, WorkerAdapter, WorkerCapabilities, WorkerEvent, WorkerUsage

# Plausible top-level keys for the response text / a resumable session id /
# a list of file edits inside `jcode run --json`'s result object. None of
# these are confirmed against a real result (see module docstring) — every
# lookup below tries each alias in turn and degrades gracefully rather than
# assuming one exact name is correct.
_TEXT_KEYS = ("text", "content", "message", "result", "output")
_SESSION_ID_KEYS = ("session_id", "sessionId", "resume_id", "resumeId")
_STATUS_KEYS = ("status", "stop_reason", "stopReason")
_DIFF_LIST_KEYS = ("file_diffs", "diffs", "files_changed", "edits")
_DIFF_PATH_KEYS = ("path", "file", "filename")
_DIFF_TEXT_KEYS = ("diff", "patch", "unified_diff", "unifiedDiff")


async def _spawn(cmd: list[str]) -> Any:
    """Start `cmd` as a subprocess, return the `Process` handle immediately.

    Isolated in its own function exactly like `acp.py`'s `_import_acp`: the
    one seam this adapter uses to reach a real subprocess, so tests
    substitute it directly instead of patching `asyncio` internals — no real
    `jcode` invocation, ever, in the test suite.
    """
    return await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )


def _first_present(data: dict, keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return value
    return None


def _parse_json_result(raw: bytes) -> dict | None:
    """Best-effort parse of `jcode run --json`'s stdout into a dict.

    Only the outer "it's a JSON object" shape is documented; anything that
    doesn't decode as one is treated as absent rather than crashing the
    adapter, and the caller falls back to the raw text.
    """
    try:
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _extract_file_diffs(data: dict) -> list[WorkerEvent]:
    """Opportunistic `FILE_DIFF` events from an unverified result shape.

    See the module docstring: none of `_DIFF_LIST_KEYS`/`_DIFF_PATH_KEYS`/
    `_DIFF_TEXT_KEYS` are confirmed against a real jcode result. This only
    ever *adds* events when one of the guessed shapes happens to match; when
    none do (the common case until this schema is verified against a real
    run) it returns an empty list and the plain MESSAGE text is all the
    caller gets. Nothing here is required for the adapter to work.
    """
    events: list[WorkerEvent] = []
    for list_key in _DIFF_LIST_KEYS:
        items = data.get(list_key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            path = _first_present(item, _DIFF_PATH_KEYS)
            diff_text = _first_present(item, _DIFF_TEXT_KEYS)
            if path is None or diff_text is None:
                continue
            events.append(WorkerEvent(kind=EventKind.FILE_DIFF, path=str(path), diff=str(diff_text)))
        break
    return events


@dataclass
class _Session:
    """Live state for one jcode session.

    `jcode_session_id` starts empty: jcode itself only assigns a session
    identity once a first `jcode run` actually happens (there is no separate
    "create a session" call to make ahead of the first message), so it is
    filled in by `send()` from whatever the result reports, and only ever
    carries that opaque id — never message content.
    """

    task_id: str
    cwd: str
    model_profile: str
    jcode_session_id: str = ""
    current_process: Any = None
    last_usage: WorkerUsage = field(default_factory=WorkerUsage)


class JcodeAdapter(WorkerAdapter):
    """Runs a task through the local `jcode` binary's headless `run` mode.

    `binary` names the executable to spawn (default `"jcode"`, the name it
    installs under on PATH). Nothing about which vendor model backs it is
    hardcoded here — that is `registry.WorkerProfile.model` / the `-m` flag
    at call time, same separation of concerns as `ACPAdapter`.
    """

    def __init__(self, binary: str = "jcode", *, max_parallel_sessions: int = 4) -> None:
        self._binary = binary
        self._max_parallel_sessions = max_parallel_sessions
        self._sessions: dict[str, _Session] = {}

    # ------------------------------------------------------------- health

    def health(self) -> tuple[bool, str]:
        if not shutil.which(self._binary):
            return False, f"'{self._binary}' not found on PATH; install jcode first"
        return True, f"'{self._binary}' found on PATH"

    def capabilities(self) -> WorkerCapabilities:
        return WorkerCapabilities(
            structured_messages=False,
            resumable_sessions=True,  # real `--resume <id>`, verified via `jcode run --help`
            tool_calls=False,  # this adapter does not parse individual tool-call events
            file_diffs=True,  # jcode edits files directly; see `_extract_file_diffs`
            reasoning_effort=False,
            token_usage=False,  # `--trace` is an unstructured stderr log, not measurable
            images=False,
            max_parallel_sessions=self._max_parallel_sessions,
        )

    # ------------------------------------------------------------ sessions

    def _get_session(self, session_id: str) -> _Session:
        try:
            return self._sessions[session_id]
        except KeyError:
            raise KeyError(f"no active jcode session: {session_id}") from None

    async def start(self, task_id: str, cwd: str, model_profile: str) -> str:
        # jcode has no "create a session, then prompt it" call to make ahead
        # of a first message (see module docstring) — this just opens the
        # local bookkeeping a hive session_id needs; the first `send()` is
        # what actually invokes jcode and learns its own session id.
        session_id = uuid.uuid4().hex
        self._sessions[session_id] = _Session(task_id=task_id, cwd=cwd, model_profile=model_profile)
        return session_id

    async def send(self, session_id: str, prompt: str) -> AsyncIterator[WorkerEvent]:
        sess = self._get_session(session_id)
        cmd = [self._binary, "run", prompt, "--json", "--quiet", "--cwd", sess.cwd]
        if sess.model_profile:
            cmd += ["--model", sess.model_profile]
        if sess.jcode_session_id:
            cmd += ["--resume", sess.jcode_session_id]

        try:
            proc = await _spawn(cmd)
        except OSError as exc:
            yield WorkerEvent(kind=EventKind.ERROR, text=str(exc))
            return

        sess.current_process = proc
        try:
            stdout, stderr = await proc.communicate()
        except BaseException as exc:  # noqa: BLE001 - surfaced as a canonical event, not swallowed
            yield WorkerEvent(kind=EventKind.ERROR, text=str(exc))
            return
        finally:
            sess.current_process = None

        if proc.returncode != 0:
            reason = stderr.decode("utf-8", errors="replace").strip() or f"jcode exited {proc.returncode}"
            yield WorkerEvent(kind=EventKind.ERROR, text=reason)
            return

        # jcode.local is pinned FREE (registry.py): a subscription CLI call
        # has no incremental metered dollar cost, so this is a fact this
        # adapter is entitled to report, not a guess dressed up as one.
        sess.last_usage = WorkerUsage(usd_micros=0, exact=True)

        data = _parse_json_result(stdout)
        if data is None:
            yield WorkerEvent(kind=EventKind.MESSAGE, text=stdout.decode("utf-8", errors="replace"))
            yield WorkerEvent(kind=EventKind.DONE, status="ok")
            return

        session_hint = _first_present(data, _SESSION_ID_KEYS)
        if session_hint:
            sess.jcode_session_id = str(session_hint)

        text = _first_present(data, _TEXT_KEYS)
        yield WorkerEvent(kind=EventKind.MESSAGE, text=str(text) if text is not None else "")

        for diff_event in _extract_file_diffs(data):
            yield diff_event

        status = _first_present(data, _STATUS_KEYS)
        yield WorkerEvent(kind=EventKind.DONE, status=str(status) if status is not None else "ok")

    async def cancel(self, session_id: str) -> None:
        sess = self._get_session(session_id)
        proc = sess.current_process
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()

    async def checkpoint(self, session_id: str) -> dict:
        """Canonical task state only — no message content of any kind."""
        sess = self._get_session(session_id)
        return {
            "adapter": "jcode",
            "binary": self._binary,
            "task_id": sess.task_id,
            "cwd": sess.cwd,
            "model_profile": sess.model_profile,
            "jcode_session_id": sess.jcode_session_id,
        }

    async def resume(self, checkpoint: dict) -> str:
        # No live connection to reconnect (unlike ACP): the actual
        # `--resume <id>` call happens lazily, inside the *next* `send()`.
        # This just rebuilds the local bookkeeping from canonical state.
        session_id = uuid.uuid4().hex
        self._sessions[session_id] = _Session(
            task_id=checkpoint["task_id"],
            cwd=checkpoint["cwd"],
            model_profile=checkpoint.get("model_profile", ""),
            jcode_session_id=checkpoint.get("jcode_session_id", ""),
        )
        return session_id

    async def usage(self, session_id: str) -> WorkerUsage:
        return self._get_session(session_id).last_usage

    async def close(self, session_id: str) -> None:
        sess = self._sessions.pop(session_id, None)
        if sess is None:
            return
        proc = sess.current_process
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()


__all__ = ["JcodeAdapter"]
