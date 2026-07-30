"""The universal worker contract every execution backend implements.

The router picks a `WorkerProfile` (registry.py); everything above this module
then only ever talks to a `WorkerAdapter`. That seam is deliberate: an adapter
translates whatever its backend actually speaks — ACP JSON-RPC, an omc team
job, a raw HTTP completion, a local Rust binary's stdout — into the canonical
shapes below, so the manager, governor, and ledger never know or care which
kind of worker produced them. Leaking a backend's wire format upward (a PTY
blob, a provider-specific event dict) defeats the whole point of this file.

Money is integer microdollars everywhere in hive (see contracts.py). `exact`
on `WorkerUsage` exists because a backend that can only estimate its own spend
must never be allowed to present that estimate as a measured fact — that is
exactly the kind of drift that lets a budget silently stop holding.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from enum import Enum

from pydantic import BaseModel, Field


class WorkerCapabilities(BaseModel):
    """What a backend's wire protocol can carry, declared once per adapter.

    Distinct from `registry.WorkerProfile.capabilities` (task-shaped skills
    like "python" / "edit" that the router matches against a task's needs):
    this is protocol-shaped — what the connection itself is capable of,
    independent of any particular task.
    """

    structured_messages: bool = False
    resumable_sessions: bool = False
    tool_calls: bool = False
    file_diffs: bool = False
    reasoning_effort: bool = False
    token_usage: bool = False
    images: bool = False
    max_parallel_sessions: int = 1


class WorkerUsage(BaseModel):
    """Spend observed for one session.

    `exact` says whether `usd_micros` is a real figure the backend reported,
    as opposed to a guess (e.g. tokens times an assumed price). Never set it
    True unless the number came off the wire — a false `exact` is worse than
    an honest zero, because the ledger has no way to tell the difference
    afterward.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    usd_micros: int = 0
    exact: bool = False


class EventKind(str, Enum):
    """The canonical shape every backend's raw stream collapses into."""

    MESSAGE = "message"          # assistant/user text
    THOUGHT = "thought"          # reasoning/thinking content
    TOOL_CALL = "tool_call"      # a tool call started
    TOOL_UPDATE = "tool_update"  # a tool call's progress or result
    FILE_DIFF = "file_diff"      # a tool call surfaced as a file edit
    PLAN = "plan"                # a plan / todo-list update
    USAGE = "usage"              # a token/cost update mid-session
    META = "meta"                # anything else structured worth keeping
    ERROR = "error"              # backend or protocol error
    DONE = "done"                # the turn finished


class WorkerEvent(BaseModel):
    """One canonical event out of `WorkerAdapter.send()`.

    Never a raw terminal/PTY blob or a backend-specific message object — an
    adapter that leaks its wire format here breaks the "router never cares
    which kind" property this contract exists for. Fields are a superset
    across event kinds; each kind only populates the ones that apply.
    """

    kind: EventKind
    text: str = ""
    tool_call_id: str = ""
    tool_kind: str = ""
    status: str = ""
    path: str = ""
    diff: str = ""
    data: dict = Field(default_factory=dict)


class WorkerAdapter(ABC):
    """One execution backend. The router and manager only ever call this
    surface — never the backend's own client library directly.
    """

    @abstractmethod
    def health(self) -> tuple[bool, str]:
        """(usable, reason).

        Must never raise. A missing dependency, an unauthenticated CLI, or a
        binary not on PATH is a degraded health report, not a crash that
        takes the whole harness down with it.
        """
        raise NotImplementedError

    @abstractmethod
    def capabilities(self) -> WorkerCapabilities:
        """What this backend's protocol can carry, declared once."""
        raise NotImplementedError

    @abstractmethod
    async def start(self, task_id: str, cwd: str, model_profile: str) -> str:
        """Start a session for a task. Returns a session_id."""
        raise NotImplementedError

    @abstractmethod
    def send(self, session_id: str, prompt: str) -> AsyncIterator[WorkerEvent]:
        """Send a prompt; stream back canonical events as they arrive.

        Declared as a plain `def` returning `AsyncIterator[WorkerEvent]`
        because an abstract method has no body to `yield` from. Concrete
        adapters implement this as an async generator function
        (`async def send(...): ... yield ...`).
        """
        raise NotImplementedError

    @abstractmethod
    async def cancel(self, session_id: str) -> None:
        """Ask the backend to stop the in-flight turn for this session."""
        raise NotImplementedError

    @abstractmethod
    async def checkpoint(self, session_id: str) -> dict:
        """Canonical task state a *different* worker can resume from.

        Never a transcript. A replacement worker (possibly a different model
        entirely) resumes from state — task_id, cwd, model_profile, whatever
        handle the backend needs to reconnect — never from another model's
        conversation history.
        """
        raise NotImplementedError

    @abstractmethod
    async def resume(self, checkpoint: dict) -> str:
        """Resume a session from a checkpoint. Returns a new session_id."""
        raise NotImplementedError

    @abstractmethod
    async def usage(self, session_id: str) -> WorkerUsage:
        """Spend observed for this session so far."""
        raise NotImplementedError

    @abstractmethod
    async def close(self, session_id: str) -> None:
        """Release whatever resources this session holds. Idempotent."""
        raise NotImplementedError


__all__ = [
    "EventKind",
    "WorkerAdapter",
    "WorkerCapabilities",
    "WorkerEvent",
    "WorkerUsage",
]
