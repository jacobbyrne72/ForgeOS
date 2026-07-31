"""Schemas that cross a process boundary.

Everything here is validated at the edge, because the manager runs on a cheap
free-tier model and cheap models produce sloppy JSON. A tight schema plus a
retry is what makes a weak model reliable; a bigger model is not.

Money is integer microdollars everywhere. A float budget comparison is a budget
that silently does not hold.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum

from pydantic import BaseModel, Field, field_validator

USD_MICROS = 1_000_000


def to_micros(usd: float) -> int:
    """Dollars -> microdollars, rounded (never truncated toward zero)."""
    return int(round(usd * USD_MICROS))


def from_micros(micros: int) -> float:
    return micros / USD_MICROS


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now() -> float:
    return time.time()


class TaskState(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    BLOCKED = "blocked"
    VERIFYING = "verifying"
    DONE = "done"
    FAILED = "failed"
    PAUSED = "paused"


TERMINAL_STATES = frozenset({TaskState.DONE, TaskState.FAILED})


class EscalationKind(str, Enum):
    """The only reasons a worker interrupts the manager.

    Event-driven on purpose: a manager that reads every token costs more than
    the workers it supervises.
    """

    STUCK = "stuck"
    LOW_CONFIDENCE = "low_confidence"
    BUDGET = "budget"
    SCOPE_REQUEST = "scope_request"
    UNCLEAR = "unclear"
    LOOP = "loop"
    VERIFY_FAILED = "verify_failed"


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


class FailureClass(str, Enum):
    """Why the attempt failed — which decides what to do about it.

    This distinction is the largest single waste-saver in the system. The naive
    reflex on a failed task is "escalate to a stronger model", but that only helps
    for MODEL failures. Escalating an ENVIRONMENT failure buys premium tokens for a
    problem no model can solve, and it will fail again identically at every tier.
    """

    ENVIRONMENT = "environment"      # missing dep, broken venv, port in use → repair, do not escalate
    CONTEXT = "context"              # worker lacked a symbol/test → add that context, same tier
    MODEL = "model"                  # genuinely beyond this tier → escalate the ladder
    SPECIFICATION = "specification"  # the task itself is wrong/ambiguous → back to the manager/human
    POLICY = "policy"                # blocked by scope or approval rules → human decision
    TRANSIENT = "transient"          # rate limit, timeout, 5xx → back off and retry same tier

    @property
    def escalate_model(self) -> bool:
        """Only a MODEL failure justifies spending more per token."""
        return self is FailureClass.MODEL

    @property
    def retry_same_tier(self) -> bool:
        return self in (FailureClass.TRANSIENT, FailureClass.CONTEXT, FailureClass.ENVIRONMENT)

    @property
    def needs_human(self) -> bool:
        return self in (FailureClass.SPECIFICATION, FailureClass.POLICY)


class Scope(BaseModel):
    """A worker's default working set.

    Tight by default so workers stop hunting through a huge repo. `allow_expand`
    is the escape hatch that keeps tightness from becoming a bottleneck: the
    worker files a SCOPE_REQUEST and the manager approves in one hop.
    """

    paths: list[str] = Field(default_factory=list)
    allow_expand: bool = True
    note: str = ""

    def covers(self, path: str) -> bool:
        p = path.replace("\\", "/").lstrip("./")
        return any(p.startswith(root.replace("\\", "/").lstrip("./")) for root in self.paths)


class Budget(BaseModel):
    """Hard ceilings. The governor enforces these; nothing edits them upward."""

    max_usd: float = 2.0
    max_seconds: int = 900
    max_iterations: int = 12

    @field_validator("max_usd")
    @classmethod
    def _positive_usd(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("max_usd must be > 0")
        return v

    @field_validator("max_seconds", "max_iterations")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be > 0")
        return v

    @property
    def max_usd_micros(self) -> int:
        return to_micros(self.max_usd)


class AttemptSummary(BaseModel):
    """One prior attempt's outcome, compact enough to carry into a retry prompt
    without re-inflating the context `economy.reducer` already shrank.

    `reason` must be exact text lifted from what the worker/reducer already
    produced (blocker, reduced evidence, verbatim FAILED lines) -- never
    reworded and never truncated mid-token. Stripping or paraphrasing exactly
    this kind of anchor (error text, file:line, test node ids) has been
    measured to *raise* billed cost and *lower* task success, because the next
    attempt can no longer match its fix against the failure (arXiv:2607.12161).
    Compression happens only by dropping whole older `AttemptSummary` entries
    (see `forge._cap_attempt_history`), never by editing what one entry says.
    """

    attempt: int
    failure_class: str = ""
    reason: str = ""


class TaskSpec(BaseModel):
    """One unit of delegated work.

    `acceptance` is not decoration. A task is done when these pass against real
    command output, which is what stops the "should work" class of false green.
    """

    id: str = Field(default_factory=lambda: new_id("task"))
    job_id: str
    subject: str
    description: str
    acceptance: list[str] = Field(default_factory=list)
    scope: Scope = Field(default_factory=Scope)
    budget: Budget = Field(default_factory=Budget)
    capabilities: list[str] = Field(default_factory=list)
    parent_id: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=now)
    # Newest-first, hard-capped record of prior failed attempts on THIS task.
    # Empty on a first attempt, which is what keeps `_build_prompt` byte-identical
    # to a task with no retry history at all -- see forgeos/adapters/executor.py.
    attempt_history: list[AttemptSummary] = Field(default_factory=list)

    @field_validator("subject", "description")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v.strip()


class TestResults(BaseModel):
    # Not a pytest test class, despite the leading "Test". Without this, pytest
    # tries to collect it and warns on every run.
    __test__ = False

    passed: int = 0
    failed: int = 0

    @property
    def green(self) -> bool:
        return self.failed == 0 and self.passed > 0

    def __str__(self) -> str:
        return f"{self.passed}p/{self.failed}f"


class WorkerReport(BaseModel):
    """What a worker hands back. Evidence, not narration.

    The manager never reads a worker transcript — it reads `heartbeat()`. That
    single decision is the difference between supervision costing a few hundred
    tokens per check and costing more than the work being supervised.
    """

    task_id: str
    worker_id: str
    state: TaskState
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    goal: str = ""
    summary: str = ""
    files_touched: list[str] = Field(default_factory=list)
    commands_run: list[str] = Field(default_factory=list)
    evidence: str = ""
    tests: TestResults | None = None
    blocker: str = ""
    next_action: str = ""
    needs_manager: bool = False
    tokens_in: int = 0
    tokens_out: int = 0
    usd_micros: int = 0
    seconds: float = 0.0
    verdict: Verdict | None = None
    escalations: list[EscalationKind] = Field(default_factory=list)
    created_at: float = Field(default_factory=now)
    # The task generation this report's worker was issued at assignment time
    # (see `core.scheduler.Assignment.generation`). `Scheduler.report` fences
    # any report whose generation is not the task's current one -- that is how
    # an orphaned worker from a reclaimed attempt is stopped from overwriting
    # a result nobody is waiting on. `None` means the caller predates
    # generation fencing; see `ledger._generation_accepted` for exactly how
    # that is handled without breaking it.
    generation: int | None = None

    @property
    def succeeded(self) -> bool:
        return self.state is TaskState.DONE and self.verdict is not Verdict.FAIL

    def heartbeat(self) -> dict:
        """The compact record the manager actually sees.

        Deliberately excludes `summary`, `evidence` and `commands_run` — those
        are for the ledger and the human. Empty fields are dropped so a quiet
        worker costs almost nothing to supervise.
        """
        hb: dict = {
            "task_id": self.task_id,
            "state": self.state.value,
            "needs_manager": self.needs_manager,
        }
        if self.goal:
            hb["goal"] = self.goal
        if self.files_touched:
            hb["files_changed"] = self.files_touched
        if self.tests is not None:
            hb["tests"] = {"passed": self.tests.passed, "failed": self.tests.failed}
        if self.blocker:
            hb["blocker"] = self.blocker
        if self.next_action:
            hb["next_action"] = self.next_action
        if self.escalations:
            hb["escalations"] = [e.value for e in self.escalations]
        if self.tokens_in or self.tokens_out:
            hb["tokens_used"] = self.tokens_in + self.tokens_out
        return hb


class Escalation(BaseModel):
    id: str = Field(default_factory=lambda: new_id("esc"))
    job_id: str
    task_id: str
    worker_id: str = ""
    kind: EscalationKind
    detail: str = ""
    created_at: float = Field(default_factory=now)
    resolved_at: float | None = None
    resolution: str = ""

    @property
    def open(self) -> bool:
        return self.resolved_at is None


class GovernorTrip(BaseModel):
    """Why the brake engaged. Written to the ledger and surfaced to the human."""

    job_id: str
    task_id: str | None = None
    reason: str
    limit: str
    observed: str
    created_at: float = Field(default_factory=now)


class JobSpec(BaseModel):
    """A whole objective. Survives restarts; the manager resumes from the ledger."""

    id: str = Field(default_factory=lambda: new_id("job"))
    objective: str
    cwd: str
    budget: Budget = Field(default_factory=lambda: Budget(max_usd=20.0, max_seconds=7200, max_iterations=60))
    created_at: float = Field(default_factory=now)

    @field_validator("objective")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("objective must not be blank")
        return v.strip()
