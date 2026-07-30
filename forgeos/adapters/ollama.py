"""OllamaAdapter — the free local *last-resort* backend (`ollama.local`).

Ollama is a local, unmetered model runner: no account, no per-token bill,
genuinely free. That is why `registry.default_registry()` gives it
`capabilities={"summarize", "classify", "triage"}` and not `"edit"` — it is
scoped to cheap preprocessing, never to difficult coding work. A local model
producing a plausible-but-wrong edit is false economy twice over: the retry
that fixes it costs more than routing to a real coding tier up front would
have, and while it runs it holds a CPU core a concurrent build on this same
machine may need. Route mechanical edits to `jcode.local` (`jcode.py`) or a
metered tier instead; use this adapter only for the cheap-preprocessing shape
its declared capabilities describe.

Real CLI surface, verified by running `ollama --help`, `ollama run --help`,
and `ollama list --help` on this machine (2026-07-30) — nothing below is
guessed:

- `ollama run MODEL [PROMPT] [flags]` — "Run a model". Passing `PROMPT`
  makes this one-shot and headless: it prints the response and exits, rather
  than opening the interactive REPL that omitting `PROMPT` would. That is
  the mode this adapter always uses.
- `--nowordwrap` — "Don't wrap words to the next line automatically"; used
  so stdout is the model's own text, not re-wrapped for a terminal.
- `--verbose` ("Show timings for response") and `--think` ("Enable thinking
  mode...") both exist but are deliberately not used: neither flag's output
  format is documented by `--help`, and actually running a real prompt to
  reverse-engineer either one was out of scope (see `capabilities()` —
  `token_usage` and `reasoning_effort` are both `False` for exactly this
  reason, not because the underlying feature doesn't exist).
- `ollama list` — "List models" — this is the health-check surface: if it
  fails outright, the daemon is unreachable; if it succeeds but the
  configured model tag is not in the output, the model has not been pulled.
  Either way `health()` reports `False` with that reason — it never runs
  `ollama pull` itself. Auto-pulling a multi-gigabyte model as a side effect
  of a health check would be exactly the kind of surprising, expensive
  action this harness's cost governance exists to prevent.

Local inference has no per-call dollar cost at all — there is no account,
no API key, nothing to meter — so this adapter never reads or constructs
credentials.
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import subprocess
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from .base import EventKind, WorkerAdapter, WorkerCapabilities, WorkerEvent, WorkerUsage

_DEFAULT_MODEL = "llava-phi3:3.8b"  # small vision-capable default; see registry.py's ollama.local


async def _spawn(cmd: list[str]) -> Any:
    """Start `cmd` as a subprocess, return the `Process` handle immediately.

    Isolated in its own function exactly like `jcode.py`'s `_spawn` and
    `acp.py`'s `_import_acp`: the one seam this adapter uses to reach a real
    subprocess for `send()`, so tests substitute it directly — no real
    `ollama run`, ever, in the test suite.
    """
    return await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )


@dataclass
class _Session:
    """Live state for one ollama session.

    Ollama's CLI has no session-resume primitive (no `--resume`-shaped flag
    across `run`/`list`/`ps`/`show` — verified against `--help` for each);
    every `send()` is an independent `ollama run` call, so there is no
    backend session identity to carry here at all, unlike `jcode.py`'s
    `jcode_session_id`.
    """

    task_id: str
    cwd: str
    model_profile: str
    current_process: Any = None
    last_usage: WorkerUsage = field(default_factory=WorkerUsage)


class OllamaAdapter(WorkerAdapter):
    """Runs a task through a local `ollama run <model> <prompt>` call.

    `model` is a constructor argument (small default) rather than hardcoded,
    since which local model is pulled is an operator/environment fact, not
    something this adapter should assume. `binary` names the executable to
    spawn (default `"ollama"`, its name on PATH).
    """

    def __init__(self, model: str = _DEFAULT_MODEL, *, binary: str = "ollama") -> None:
        self._model = model
        self._binary = binary
        self._sessions: dict[str, _Session] = {}

    # ------------------------------------------------------------- health

    def health(self) -> tuple[bool, str]:
        if not shutil.which(self._binary):
            return False, f"'{self._binary}' not found on PATH; install ollama first"

        try:
            result = subprocess.run(
                [self._binary, "list"], capture_output=True, text=True, timeout=10, check=False
            )
        except Exception as exc:  # noqa: BLE001 - health() must never raise (base.py contract)
            return False, f"could not reach the ollama daemon: {exc}"

        if result.returncode != 0:
            reason = (result.stderr or result.stdout or "").strip()
            return False, "ollama daemon not reachable" + (f" ({reason})" if reason else " (is `ollama serve` running?)")

        if not self._model_is_pulled(result.stdout):
            return False, f"model '{self._model}' not pulled; run `ollama pull {self._model}` to enable this adapter"

        return True, f"ollama daemon reachable and model '{self._model}' present"

    def _model_is_pulled(self, list_stdout: str) -> bool:
        """Best-effort check that `self._model` appears in `ollama list`'s
        output. The table's exact column layout isn't part of `--help`'s
        contract, so this only checks whether the model name is the first
        whitespace-separated field on any line (case-insensitive) rather
        than parsing named columns."""
        target = self._model.strip().lower()
        for line in list_stdout.splitlines():
            fields = line.split()
            if fields and fields[0].strip().lower() == target:
                return True
        return False

    def capabilities(self) -> WorkerCapabilities:
        return WorkerCapabilities(
            structured_messages=False,
            resumable_sessions=False,  # no `--resume`-shaped primitive exists for ollama
            tool_calls=False,
            file_diffs=False,  # cannot edit files — preprocessing only, never routed to edit tasks
            reasoning_effort=False,  # `--think` exists but its output isn't parsed (see module docstring)
            token_usage=False,  # `--verbose` exists but its output isn't parsed (see module docstring)
            images=False,
            max_parallel_sessions=1,  # a local model held in one process; don't contend with a concurrent build
        )

    # ------------------------------------------------------------ sessions

    def _get_session(self, session_id: str) -> _Session:
        try:
            return self._sessions[session_id]
        except KeyError:
            raise KeyError(f"no active ollama session: {session_id}") from None

    async def start(self, task_id: str, cwd: str, model_profile: str) -> str:
        session_id = uuid.uuid4().hex
        self._sessions[session_id] = _Session(task_id=task_id, cwd=cwd, model_profile=model_profile)
        return session_id

    async def send(self, session_id: str, prompt: str) -> AsyncIterator[WorkerEvent]:
        sess = self._get_session(session_id)
        cmd = [self._binary, "run", self._model, prompt, "--nowordwrap"]

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
            reason = stderr.decode("utf-8", errors="replace").strip() or f"ollama exited {proc.returncode}"
            yield WorkerEvent(kind=EventKind.ERROR, text=reason)
            return

        # Local inference: no account, no metered API, genuinely zero
        # dollar cost — this is a fact this adapter is entitled to report.
        sess.last_usage = WorkerUsage(usd_micros=0, exact=True)

        yield WorkerEvent(kind=EventKind.MESSAGE, text=stdout.decode("utf-8", errors="replace"))
        yield WorkerEvent(kind=EventKind.DONE, status="ok")

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
            "adapter": "ollama",
            "binary": self._binary,
            "model": self._model,
            "task_id": sess.task_id,
            "cwd": sess.cwd,
            "model_profile": sess.model_profile,
        }

    async def resume(self, checkpoint: dict) -> str:
        # No backend session identity ever existed to reconnect to (see
        # `_Session`'s docstring) — this honestly starts a fresh session
        # from canonical state rather than pretending continuity.
        session_id = uuid.uuid4().hex
        self._sessions[session_id] = _Session(
            task_id=checkpoint["task_id"],
            cwd=checkpoint["cwd"],
            model_profile=checkpoint.get("model_profile", ""),
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


__all__ = ["OllamaAdapter"]
