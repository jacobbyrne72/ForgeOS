"""The manager — the only part of the harness that runs on a model, and the
part hardest to trust, because it runs on a *cheap* model.

A free-tier model produces sloppy prose and unreliable JSON. The fix is not a
bigger model; it is shrinking every decision to a closed set plus a few short
strings, validated by pydantic, with a retry and a deterministic fallback. A
weak model choosing from an enum is reliable. A weak model writing a prose
plan is not.

This is an exception handler, not a narrator. On a task that runs clean it is
called zero times — `wake_triggers` returns `[]` and nothing downstream ever
invokes `decide`. It wakes only on the defined triggers, and even then it
often does not need the model at all: the governor's `FailureClass` already
determines the remedy for every failure it can classify (see
`forgeos/core/governor.py::_REMEDY_BY_CLASS` — the same total mapping, expressed
here in terms of `ManagerDecision`). Spending a model call to re-derive what
that taxonomy already knows is exactly the waste this harness exists to
remove, so `decide()` pre-empts the model whenever the answer is already
deterministic: a clean heartbeat, an exhausted attempt budget, or a
classified failure. The model is invoked only for the genuinely ambiguous
case -- a worker escalation with no governor classification attached.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Callable

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from ..contracts import FailureClass


class ManagerDecision(str, Enum):
    """The only shapes a manager verdict can take. A closed set, on purpose:
    a weak model can pick one of these reliably; it cannot reliably write a
    free-form plan."""

    CONTINUE = "continue"
    FOLLOW_UP = "follow_up"
    ADD_CONTEXT = "add_context"
    SPLIT = "split"
    ESCALATE = "escalate"
    REVERT = "revert"
    BLOCK = "block"
    ASK_HUMAN = "ask_human"


class ManagerVerdict(BaseModel):
    """What the manager hands back. Small, schema-constrained, and validated
    at the edge -- see the module docstring for why."""

    decision: ManagerDecision
    reason_code: str
    instructions: list[str] = Field(default_factory=list)
    context_additions: list[str] = Field(default_factory=list)
    escalate_after_failed_attempts: int = 1
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("reason_code")
    @classmethod
    def _non_blank_reason(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("reason_code must not be blank")
        return v.strip()

    @field_validator("escalate_after_failed_attempts")
    @classmethod
    def _non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("escalate_after_failed_attempts must be >= 0")
        return v

    @model_validator(mode="after")
    def _instructions_match_decision(self) -> "ManagerVerdict":
        if self.decision is ManagerDecision.FOLLOW_UP and not self.instructions:
            raise ValueError("FOLLOW_UP requires at least one instruction")
        if self.decision is ManagerDecision.CONTINUE and self.instructions:
            raise ValueError("CONTINUE must not carry instructions")
        return self


# ------------------------------------------------------------------ triggers

# Heartbeat states (see contracts.TaskState) that mean the worker itself
# already flagged trouble -- checked as plain strings because heartbeat() is
# a dict, not a TaskState instance.
_TROUBLE_STATES = frozenset({"blocked", "failed"})


def wake_triggers(heartbeat: dict) -> list[str]:
    """Deterministic check for whether the manager should be woken at all.

    A clean heartbeat -- running or done, nothing flagged -- returns `[]` and
    the manager is never invoked. Everything here reads fields `heartbeat()`
    already puts in the compact record; there is no transcript to read.
    """
    triggers: list[str] = []
    if heartbeat.get("needs_manager"):
        triggers.append("needs_manager")
    for esc in heartbeat.get("escalations") or []:
        triggers.append(f"escalation:{esc}")
    if heartbeat.get("blocker"):
        triggers.append("blocker")
    tests = heartbeat.get("tests")
    if isinstance(tests, dict) and tests.get("failed", 0):
        triggers.append("tests_failed")
    state = heartbeat.get("state")
    if state in _TROUBLE_STATES:
        triggers.append(f"state:{state}")
    return triggers


# ------------------------------------------------------ deterministic remedy

# Every FailureClass the governor can produce already has a determined
# remedy (see governor._REMEDY_BY_CLASS). Re-expressed here as a
# ManagerDecision so `decide()` never has to ask the model something the
# taxonomy already answered. Instructions are only required for FOLLOW_UP
# (see ManagerVerdict validation), so every other row leaves them empty.
_FAILURE_DECISIONS: dict[FailureClass, tuple[ManagerDecision, str, tuple[str, ...]]] = {
    FailureClass.ENVIRONMENT: (
        ManagerDecision.FOLLOW_UP,
        "repair_environment",
        ("Repair the environment issue (dependency, venv, port, or config), then retry the same task at the same tier.",),
    ),
    FailureClass.CONTEXT: (
        ManagerDecision.ADD_CONTEXT,
        "add_missing_context",
        (),
    ),
    FailureClass.TRANSIENT: (
        ManagerDecision.FOLLOW_UP,
        "backoff_retry",
        ("Back off briefly and retry the same task at the same tier; the failure was transient.",),
    ),
    FailureClass.MODEL: (
        ManagerDecision.ESCALATE,
        "escalate_model_tier",
        (),
    ),
    FailureClass.SPECIFICATION: (
        ManagerDecision.ASK_HUMAN,
        "specification_unclear",
        (),
    ),
    FailureClass.POLICY: (
        ManagerDecision.ASK_HUMAN,
        "policy_blocked",
        (),
    ),
}

# Decisions that already mean "stop retrying at this tier" -- no budget left
# to advertise for further attempts.
_TERMINAL_DECISIONS = frozenset({ManagerDecision.ASK_HUMAN, ManagerDecision.ESCALATE, ManagerDecision.BLOCK})

_SCHEMA_HINT = (
    "Reply with ONLY a JSON object, no prose, no markdown fences. Keys: "
    "decision (one of " + ", ".join(d.value for d in ManagerDecision) + "), "
    "reason_code (short slug), instructions (list of imperative strings, "
    "required and non-empty only if decision is follow_up), "
    "context_additions (list of refs), escalate_after_failed_attempts (int), "
    "confidence (0..1)."
)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class ManagerUnavailable(RuntimeError):
    """Every manager in the pool failed. Surfaced rather than guessed around."""


class ManagerPool:
    """Primary manager with ordered backups, failing over on provider errors.

    Failover is clean *because* `decide()` takes a canonical heartbeat rather than a
    conversation. The backup receives exactly what the primary received — no
    transcript replay, no shared history to reconstruct — so a rate-limited primary
    costs one retry, not a restart of the reasoning.

    That property is the whole reason the manager was built around injected
    `complete` callables instead of an imported provider client.
    """

    def __init__(self, managers: list["Manager"], *, on_failover=None):
        if not managers:
            raise ValueError("ManagerPool needs at least one manager")
        self.managers = managers
        self.on_failover = on_failover
        self.failovers = 0
        self.active_index = 0

    @property
    def active(self) -> "Manager":
        return self.managers[self.active_index]

    @property
    def model_calls_made(self) -> int:
        return sum(m.model_calls_made for m in self.managers)

    def decide(self, heartbeat: dict, **kw) -> ManagerVerdict:
        """Try each manager in order on the SAME canonical heartbeat."""
        errors: list[str] = []
        for idx, manager in enumerate(self.managers):
            try:
                verdict = manager.decide(heartbeat, **kw)
            except Exception as exc:  # provider error: rate limit, timeout, 5xx
                errors.append(f"{type(exc).__name__}: {exc}")
                self.failovers += 1
                if self.on_failover is not None:
                    self.on_failover(idx, exc)
                continue
            self.active_index = idx
            return verdict

        raise ManagerUnavailable("; ".join(errors) or "no manager produced a verdict")


class Manager:
    """Decides what happens next when a worker's heartbeat needs judgement.

    `complete` is injected rather than imported so this class never talks to
    a provider directly -- that keeps it unit-testable with a fake and
    provider-agnostic in production.
    """

    def __init__(self, complete: Callable[[str], str]):
        self.complete = complete
        self.model_calls_made = 0

    # --------------------------------------------------------------- decide

    def decide(
        self,
        heartbeat: dict,
        *,
        failure: FailureClass | None,
        attempts: int,
        max_attempts: int,
    ) -> ManagerVerdict:
        triggers = wake_triggers(heartbeat)

        # Clean heartbeat, no governor classification: nothing to decide.
        if failure is None and not triggers:
            return ManagerVerdict(
                decision=ManagerDecision.CONTINUE,
                reason_code="clean_heartbeat",
                confidence=1.0,
                escalate_after_failed_attempts=max(max_attempts - attempts, 0),
            )

        # Out of attempts: a human decides, rather than escalating forever.
        # This checks before the failure-class table on purpose -- an
        # exhausted budget overrides what would otherwise be an ESCALATE.
        if attempts >= max_attempts:
            return ManagerVerdict(
                decision=ManagerDecision.ASK_HUMAN,
                reason_code="max_attempts_exhausted",
                confidence=1.0,
                escalate_after_failed_attempts=0,
            )

        # The governor already classified this failure; its remedy is a
        # deterministic function of the class, not a judgement call.
        if failure is not None:
            return self._deterministic_for_failure(failure, heartbeat, attempts, max_attempts)

        # Genuine ambiguity: a worker escalation with no governor
        # classification attached. This is the only path that costs a model
        # call.
        return self._decide_with_model(heartbeat, triggers, attempts, max_attempts)

    # ------------------------------------------------------- deterministic

    def _deterministic_for_failure(
        self,
        failure: FailureClass,
        heartbeat: dict,
        attempts: int,
        max_attempts: int,
    ) -> ManagerVerdict:
        decision, reason_code, instructions = _FAILURE_DECISIONS[failure]
        context_additions: list[str] = []
        if failure is FailureClass.CONTEXT:
            blocker = heartbeat.get("blocker") or ""
            if blocker:
                context_additions = [blocker]
        remaining = 0 if decision in _TERMINAL_DECISIONS else max(max_attempts - attempts, 0)
        return ManagerVerdict(
            decision=decision,
            reason_code=reason_code,
            instructions=list(instructions),
            context_additions=context_additions,
            confidence=1.0,
            escalate_after_failed_attempts=remaining,
        )

    # -------------------------------------------------------------- model

    def _decide_with_model(
        self,
        heartbeat: dict,
        triggers: list[str],
        attempts: int,
        max_attempts: int,
    ) -> ManagerVerdict:
        prompt = self._build_prompt(heartbeat, triggers, terse=False)
        verdict = self._call_and_parse(prompt)
        if verdict is None:
            prompt = self._build_prompt(heartbeat, triggers, terse=True)
            verdict = self._call_and_parse(prompt)
        if verdict is None:
            # Never fall back to CONTINUE: a manager that guesses "carry on"
            # after failing to parse twice is how a runaway loop keeps
            # running. Punt to a human instead.
            return ManagerVerdict(
                decision=ManagerDecision.ASK_HUMAN,
                reason_code="unparseable_model_output",
                confidence=0.0,
                escalate_after_failed_attempts=0,
            )
        return verdict

    def _call_and_parse(self, prompt: str) -> ManagerVerdict | None:
        raw = self.complete(prompt)
        self.model_calls_made += 1
        return self._parse_verdict(raw)

    def _build_prompt(self, heartbeat: dict, triggers: list[str], *, terse: bool) -> str:
        lines = [
            "You are the MANAGER. A worker heartbeat needs a decision.",
            f"heartbeat: {json.dumps(heartbeat, sort_keys=True)}",
            f"wake_triggers: {triggers}",
            _SCHEMA_HINT,
        ]
        if terse:
            lines.append(
                "Your previous reply could not be parsed as that JSON object. "
                "Reply with the JSON object ONLY -- nothing before or after it."
            )
        return "\n".join(lines)

    @staticmethod
    def _parse_verdict(raw: str) -> ManagerVerdict | None:
        """Parse strictly: valid JSON object matching the schema, optionally
        wrapped in prose a weak model tacked on. No leniency beyond that --
        malformed JSON syntax is not hand-repaired."""
        if not raw or not raw.strip():
            return None
        text = raw.strip()
        candidates = [text]
        match = _JSON_OBJECT.search(text)
        if match and match.group(0) != text:
            candidates.append(match.group(0))
        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(data, dict):
                continue
            try:
                return ManagerVerdict(**data)
            except ValidationError:
                continue
        return None


__all__ = [
    "Manager",
    "ManagerDecision",
    "ManagerVerdict",
    "wake_triggers",
]
