"""Local command worker adapter.

This is the small, provider-neutral execution seam used by local coding CLIs.
It follows mini-SWE-agent's local environment boundary: the command receives
the task prompt on stdin, output is surfaced as canonical events, and process
cleanup is owned by the adapter rather than the caller.
"""

from __future__ import annotations

import asyncio
import os
import signal
import shutil
from dataclasses import dataclass

from ..contracts import new_id
from .base import EventKind, WorkerAdapter, WorkerCapabilities, WorkerEvent, WorkerUsage


@dataclass
class _Session:
    task_id: str
    cwd: str
    model_profile: str
    process: asyncio.subprocess.Process | None = None


class LocalCommandAdapter(WorkerAdapter):
    """Run one configured local command per ForgeOS task.

    The command and arguments are configuration, never task text. The task
    prompt is written to stdin, which avoids shell interpolation and keeps the
    backend usable for CLIs that expose a non-interactive mode.
    """

    def __init__(self, command: str, args: list[str] | None = None, *, max_parallel_sessions: int = 1) -> None:
        if not command.strip():
            raise ValueError("command must not be blank")
        if max_parallel_sessions < 1:
            raise ValueError("max_parallel_sessions must be >= 1")
        self._command = command
        self._args = list(args or [])
        self._max_parallel_sessions = max_parallel_sessions
        self._sessions: dict[str, _Session] = {}

    def health(self) -> tuple[bool, str]:
        resolved = shutil.which(self._command)
        if resolved is None:
            return False, f"'{self._command}' not found on PATH"
        return True, f"local command available: {resolved}"

    def capabilities(self) -> WorkerCapabilities:
        return WorkerCapabilities(
            structured_messages=False,
            resumable_sessions=True,
            token_usage=False,
            max_parallel_sessions=self._max_parallel_sessions,
        )

    async def start(self, task_id: str, cwd: str, model_profile: str) -> str:
        session_id = new_id("local")
        self._sessions[session_id] = _Session(task_id, cwd, model_profile)
        return session_id

    def _get_session(self, session_id: str) -> _Session:
        try:
            return self._sessions[session_id]
        except KeyError:
            raise KeyError(f"no active local command session: {session_id}") from None

    async def send(self, session_id: str, prompt: str):
        session = self._get_session(session_id)
        if session.process is not None:
            raise RuntimeError(f"local command session already sent: {session_id}")
        session.process = await asyncio.create_subprocess_exec(
            self._command,
            *self._args,
            cwd=session.cwd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
        assert session.process.stdin is not None
        assert session.process.stdout is not None
        try:
            session.process.stdin.write(prompt.encode("utf-8"))
            await session.process.stdin.drain()
            session.process.stdin.close()
            while line := await session.process.stdout.readline():
                yield WorkerEvent(kind=EventKind.MESSAGE, text=line.decode("utf-8", errors="replace").rstrip())
            returncode = await session.process.wait()
            if returncode:
                yield WorkerEvent(
                    kind=EventKind.ERROR,
                    text=f"local command exited with code {returncode}",
                    data={"returncode": returncode},
                )
            yield WorkerEvent(kind=EventKind.DONE, status=str(returncode))
        except (BrokenPipeError, ConnectionResetError) as exc:
            yield WorkerEvent(kind=EventKind.ERROR, text=f"local command stdin failed: {exc}")

    async def cancel(self, session_id: str) -> None:
        session = self._get_session(session_id)
        await self._terminate(session)

    async def _terminate(self, session: _Session) -> None:
        process = session.process
        if process is None or process.returncode is not None:
            return
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.kill()
        await process.wait()

    async def checkpoint(self, session_id: str) -> dict:
        session = self._get_session(session_id)
        return {
            "task_id": session.task_id,
            "cwd": session.cwd,
            "model_profile": session.model_profile,
            "command": self._command,
            "args": list(self._args),
        }

    async def resume(self, checkpoint: dict) -> str:
        return await self.start(
            str(checkpoint["task_id"]),
            str(checkpoint["cwd"]),
            str(checkpoint.get("model_profile", "")),
        )

    async def usage(self, session_id: str) -> WorkerUsage:
        self._get_session(session_id)
        return WorkerUsage()

    async def close(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is not None:
            await self._terminate(session)


__all__ = ["LocalCommandAdapter"]
