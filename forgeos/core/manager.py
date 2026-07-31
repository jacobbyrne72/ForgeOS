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

When the model's reply itself fails schema or format validation, that is a
mistake any tier could make -- not evidence this tier is out of its depth.
`_decide_with_model` retries exactly once, with the EXACT parse/validation
error appended to the SAME prompt so the model can self-correct against its
own specific mistake -- the `instructor` library's pattern (patch a client,
catch the `ValidationError`, retry with it in context). A violation that
survives seeing its own error is `SCHEMA_REPAIR_FAILURE_CLASS`
(`FailureClass.SPECIFICATION`), never `.MODEL`: `router.Router.escalate` only
escalates a MODEL failure, so this is what stops a formatting mistake from
ever buying a pricier tier. See `RepairAttempt` / `compress_repair_history`
for the inner loop's own (distinct from forge.py's cross-attempt
`attempt_history`) prior-iteration bookkeeping.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from ..contracts import FailureClass, TaskSpec
from ..economy.reducer import ToolResult, Turn, clear_tool_results
from ..leases import patterns_overlap


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


# ------------------------------------------------------ auto-repair on parse
#
# The `instructor` library's pattern for unreliable structured output: patch
# the client, validate the reply against a pydantic schema, and on failure
# retry with the EXACT error fed back so the model can self-correct -- "it
# also lets the model know what the error was so it can self-correct which
# can sometimes help you to correct a Json schema" (mined from a tutorial on
# the library; see docs cited in the auto-repair test module). A schema/format
# violation is never grounds to escalate: the next tier would make the exact
# same formatting mistake for the exact same reason, so this must resolve at
# the SAME tier or fail safe -- never buy a pricier one.


@dataclass
class ParseAttempt:
    """One call-and-parse cycle's outcome. `error` and `raw` are populated
    exactly when `verdict` is None -- there is nothing to repair once parsing
    already succeeded."""

    verdict: ManagerVerdict | None
    raw: str = ""
    error: str = ""


@dataclass
class RepairAttempt:
    """One failed iteration of an inner verify/repair loop -- e.g. one turn
    of `Manager._decide_with_model`'s schema-repair retry.

    Distinct from forge.py's cross-ATTEMPT `AttemptSummary`/`attempt_history`,
    which spans whole worker sessions across the OUTER per-task retry loop
    (see `forge.py::_cap_attempt_history`, tests/test_retry_context.py). This
    is one level further IN: iterations *within* a single validation cycle
    for one piece of model output.
    """

    iteration: int
    raw_reply: str
    error: str


def compress_repair_history(
    attempts: list[RepairAttempt],
    *,
    keep_last_n: int = 1,
    trigger_tokens: int = 200,
    clear_at_least_tokens: int = 1,
) -> list[RepairAttempt]:
    """Compress older repair iterations before the next prompt is built.

    Reuses `economy.reducer.clear_tool_results` -- the harness's one
    already-built "keep the newest N turns raw, clear older tool output to a
    placeholder once a token trigger is crossed" algorithm (a
    provider-agnostic port of Anthropic's `clear_tool_uses_20250919`; see
    docs/research/cost-reduction.md) -- rather than a second reducer. Each
    `RepairAttempt` becomes one `Turn` holding one `ToolResult`;
    `clear_tool_results` makes every clearing decision and this function only
    translates the shape back.

    Never rewords or trims a kept iteration: `clear_tool_results` either
    leaves an entry completely untouched or replaces its ENTIRE content with
    a placeholder -- nothing in between. That is what keeps an exact
    validation error, file:line reference, or test node id inside a
    still-relevant iteration byte-for-byte, per arXiv:2607.12161 (dropping
    whole low-value items is fine; rewording or truncating a kept one is not
    -- doing so was measured to raise billed cost and lower task success,
    because the next attempt can no longer match its fix against the
    failure).
    """
    turns = [
        Turn(
            content=f"repair iteration {a.iteration}",
            tool_results=[
                ToolResult(
                    tool_name=f"repair_attempt_{a.iteration}",
                    content=f"{a.raw_reply}\n{a.error}".strip("\n"),
                )
            ],
        )
        for a in attempts
    ]
    cleared, _saved_tokens = clear_tool_results(
        turns,
        trigger_tokens=trigger_tokens,
        keep_last_n=keep_last_n,
        clear_at_least_tokens=clear_at_least_tokens,
    )
    out: list[RepairAttempt] = []
    for original, turn in zip(attempts, cleared):
        result = turn.tool_results[0]
        if result.cleared:
            out.append(RepairAttempt(iteration=original.iteration, raw_reply="", error=result.content))
        else:
            out.append(original)
    return out


# A schema/format violation that survives the one repair retry in
# `Manager._decide_with_model` is a SPECIFICATION-class failure -- the
# schema/prompt combination did not work for this reply -- never a MODEL
# failure. `FailureClass.MODEL` is the only class `router.Router.escalate`
# acts on (`escalate_model` is False for every other class), so this constant
# is what keeps a formatting mistake from ever buying a pricier tier.
SCHEMA_REPAIR_FAILURE_CLASS = FailureClass.SPECIFICATION


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
        prompt = self._build_prompt(heartbeat, triggers)
        attempt = self._call_and_parse(prompt)
        if attempt.verdict is not None:
            return attempt.verdict

        # Auto-repair, not an escalation trigger (see the module docstring
        # and SCHEMA_REPAIR_FAILURE_CLASS): exactly one retry, with the exact
        # error appended to the SAME prompt -- never a fresh rebuild --
        # before this is treated as a real failure. `compress_repair_history`
        # is the same mechanism a deeper repair loop would reach for to keep
        # older iterations from re-inflating the prompt; here there is only
        # ever one prior iteration, so it is a no-op, but routing through it
        # means this one-retry path and any future multi-retry path share one
        # tested implementation rather than two.
        history = compress_repair_history(
            [RepairAttempt(iteration=1, raw_reply=attempt.raw, error=attempt.error)]
        )
        repair_prompt = self._repair_prompt(prompt, history[-1].error)
        attempt = self._call_and_parse(repair_prompt)
        if attempt.verdict is not None:
            return attempt.verdict

        # Never fall back to CONTINUE: a manager that guesses "carry on"
        # after failing to parse twice is how a runaway loop keeps running.
        # Punt to a human instead -- this is the SCHEMA_REPAIR_FAILURE_CLASS
        # (SPECIFICATION) outcome: the schema/prompt combination did not work
        # for this reply, which is never grounds to escalate to a pricier
        # tier.
        return ManagerVerdict(
            decision=ManagerDecision.ASK_HUMAN,
            reason_code="unparseable_model_output",
            confidence=0.0,
            escalate_after_failed_attempts=0,
        )

    def _call_and_parse(self, prompt: str) -> ParseAttempt:
        raw = self.complete(prompt)
        self.model_calls_made += 1
        return self._parse_verdict(raw)

    def _build_prompt(self, heartbeat: dict, triggers: list[str]) -> str:
        return "\n".join([
            "You are the MANAGER. A worker heartbeat needs a decision.",
            f"heartbeat: {json.dumps(heartbeat, sort_keys=True)}",
            f"wake_triggers: {triggers}",
            _SCHEMA_HINT,
        ])

    @staticmethod
    def _repair_prompt(prompt: str, error: str) -> str:
        """`prompt` reproduced in full, then the exact error appended in the
        tail -- never a reworded summary of it, and never a rebuild of the
        prefix (see `RepairAttempt` / `compress_repair_history`)."""
        return "\n".join([
            prompt,
            "",
            "Your previous reply failed validation with this exact error. "
            "Fix precisely this and reply with the corrected JSON object "
            "ONLY, nothing before or after it:",
            error,
        ])

    @staticmethod
    def _parse_verdict(raw: str) -> ParseAttempt:
        """Parse strictly: valid JSON object matching the schema, optionally
        wrapped in prose a weak model tacked on. No leniency beyond that --
        malformed JSON syntax is not hand-repaired.

        On failure, `error` is the most actionable message available. A
        schema `ValidationError` (real JSON was found; a field is wrong)
        always wins over an earlier JSON-syntax error, because it names the
        EXACT thing to fix -- which is what makes the one repair retry (see
        `Manager._decide_with_model`) a real self-correction loop rather than
        a blind resend.
        """
        if not raw or not raw.strip():
            return ParseAttempt(verdict=None, raw=raw, error="empty reply -- no JSON object found")
        text = raw.strip()
        candidates = [text]
        match = _JSON_OBJECT.search(text)
        if match and match.group(0) != text:
            candidates.append(match.group(0))

        best_error = ""
        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except (json.JSONDecodeError, ValueError) as exc:
                if not best_error:
                    best_error = f"JSON syntax error: {exc}"
                continue
            if not isinstance(data, dict):
                if not best_error:
                    best_error = f"parsed JSON is a {type(data).__name__}, not an object"
                continue
            try:
                return ParseAttempt(verdict=ManagerVerdict(**data), raw=raw)
            except ValidationError as exc:
                best_error = f"schema validation failed: {exc}"
        return ParseAttempt(verdict=None, raw=raw, error=best_error or "no JSON object found in reply")


# --------------------------------------------- parallel-safety pre-flight
#
# Anthropic's and Cognition's public write-ups on multi-agent decomposition
# disagree on whether to parallelize subagents at all, but converge on the
# same concrete failure mode: two subagents that never learn of each other
# silently diverge when they touch the same state without a shared reference
# (Cognition's Flappy-Bird example -- one agent draws the background, another
# the bird, neither holding the other's contract). Worktree-per-worker file
# isolation catches the file-level version of this for free; it does nothing
# for the semantic version, where the files never even overlap but one task's
# code is written against an interface the other task hasn't built yet. See
# docs/research/harness-workflows.md, top action #5.

# Verbs that mean "this task is the one bringing a name/schema/interface into
# existence", as opposed to merely mentioning something that already exists
# ("uses the existing Foo class"). Deliberately explicit rather than a
# stemmer -- a missed inflection just means a missed detection, which the
# conservative bias below already assumes will sometimes happen.
_INTRODUCE_VERB = re.compile(
    r"\b(?:create|creates|created|creating|"
    r"add|adds|added|adding|"
    r"introduce|introduces|introduced|introducing|"
    r"define|defines|defined|defining)\b",
    re.IGNORECASE,
)

# How far past an introduce-verb to look for the thing being introduced.
# Long enough for "creates the `Foo.bar` schema used by ..." to still catch
# `Foo.bar`; short enough that it does not sweep in unrelated identifiers
# from later in a long task description.
_INTRODUCE_WINDOW = 80

_QUOTED_NAME = re.compile(r"['\"`]([^'\"`]{1,80})['\"`]")
_DOTTED_PATH = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b")
_CAMEL_SYMBOL = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+\b")  # >=2 humps
_SNAKE_SYMBOL = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")  # >=1 underscore


def _task_text(spec: TaskSpec) -> str:
    return f"{spec.description}\n" + "\n".join(spec.acceptance)


def _introduced_symbols(text: str) -> set[str]:
    """Identifier-like tokens a task's own text claims to introduce.

    Deliberately simple, not a parser: only text following an introduce-style
    verb counts as a claim. An identifier mentioned elsewhere in a long
    description ("uses the existing Foo class") is not a claim of introducing
    anything -- counting it would flag near-total contract "sharing" between
    any two tasks that happen to name the same thing.
    """
    symbols: set[str] = set()
    for m in _INTRODUCE_VERB.finditer(text):
        window = text[m.end(): m.end() + _INTRODUCE_WINDOW]
        symbols.update(s.strip() for s in _QUOTED_NAME.findall(window) if s.strip())
        symbols.update(_DOTTED_PATH.findall(window))
        symbols.update(_CAMEL_SYMBOL.findall(window))
        symbols.update(_SNAKE_SYMBOL.findall(window))
    return symbols


def _mentions(text: str, symbol: str) -> bool:
    """Whether `symbol` appears in `text` as itself, not as a fragment of a
    longer identifier -- so "Foo" does not match inside "FooBar" or
    "Foo_bar" (`_` is a word character). Plain `\\b` boundaries, not a
    stricter non-word-or-dot lookaround: `PaymentGateway.charge()` -- a
    method call on the very thing being referenced -- is the ordinary way
    prose points at an introduced symbol, and a dot is not a word character,
    so `\\b` already treats it as a boundary without excluding it."""
    return re.search(r"\b" + re.escape(symbol) + r"\b", text) is not None


def _shared_contract(a: TaskSpec, b: TaskSpec) -> tuple[str, str, str] | None:
    """(introducer_id, dependent_id, symbol) if one task introduces something
    the other's own text references, else None. Checked both directions --
    either task could be the one introducing the shared piece. Sorted so the
    result is deterministic when a task's text happens to introduce more than
    one symbol the other references."""
    a_text, b_text = _task_text(a), _task_text(b)
    for symbol in sorted(_introduced_symbols(a_text)):
        if _mentions(b_text, symbol):
            return a.id, b.id, symbol
    for symbol in sorted(_introduced_symbols(b_text)):
        if _mentions(a_text, symbol):
            return b.id, a.id, symbol
    return None


class SafetyVerdict(BaseModel):
    """Whether two TaskSpecs may run in parallel, and why not if they can't.

    Conservative by design: a false-positive sequentialization costs a little
    wall-clock (one task waits for one it did not actually need to); a
    false-negative parallelization costs silent divergence between two
    workers that never see each other's decisions -- the exact failure mode
    both Anthropic's and Cognition's public write-ups independently reported.
    When genuinely unsure, this returns unsafe.
    """

    safe: bool
    reasons: list[str] = Field(default_factory=list)
    # Set only when a contract violation implies a direction: the introducing
    # task's id, then the dependent task's id. A pure path-overlap violation
    # has no natural direction -- either order sequences the pair correctly
    # -- so this stays None rather than implying an ordering the check does
    # not actually know.
    introduces_before: tuple[str, str] | None = None


def parallel_safety(a: TaskSpec, b: TaskSpec) -> SafetyVerdict:
    """Pre-flight check before two decomposed tasks are marked parallel-safe.

    Two independent, deterministic gates -- no model call, so the check
    cannot itself be gamed or drift between runs:

    1. `Scope.paths` must not overlap, reusing `leases.patterns_overlap`
       rather than a second glob matcher -- this is the same glob-overlap
       question a path lease answers, and a second implementation here would
       be one more place for the two to silently drift apart. An unscoped
       task (`scope.paths == []`) can touch anything, so non-overlap cannot
       be proven either way; treated as unsafe, not as "nothing to check".
    2. Neither task's description/acceptance may reference a not-yet-
       materialized symbol the OTHER task's own text claims to introduce --
       the semantic conflict file-level isolation cannot see.

    A caller that gets `safe=False` should add a `depends_on` edge, never
    drop the pair or run it anyway. `introduces_before` names which task
    should run first when the violation implies a direction; a pure scope
    overlap leaves it None since either order sequences the pair correctly.
    """
    reasons: list[str] = []
    introduces_before: tuple[str, str] | None = None

    if not a.scope.paths or not b.scope.paths:
        reasons.append(
            f'"{a.subject}" and "{b.subject}": at least one task has no declared '
            f"scope.paths -- an unscoped task can touch anything, so non-overlap "
            f"cannot be proven"
        )
    else:
        overlaps = [
            (pa, pb) for pa in a.scope.paths for pb in b.scope.paths
            if patterns_overlap(pa, pb)
        ]
        if overlaps:
            shown = ", ".join(f"{pa!r}~{pb!r}" for pa, pb in overlaps[:3])
            reasons.append(f'"{a.subject}" and "{b.subject}" share scope: {shown}')

    shared = _shared_contract(a, b)
    if shared:
        introducer_id, dependent_id, symbol = shared
        who = a.subject if introducer_id == a.id else b.subject
        other = b.subject if introducer_id == a.id else a.subject
        reasons.append(f'"{who}" introduces {symbol!r}, referenced by "{other}"')
        introduces_before = (introducer_id, dependent_id)

    return SafetyVerdict(safe=not reasons, reasons=reasons, introduces_before=introduces_before)


__all__ = [
    "Manager",
    "ManagerDecision",
    "ManagerVerdict",
    "ParseAttempt",
    "RepairAttempt",
    "SCHEMA_REPAIR_FAILURE_CLASS",
    "compress_repair_history",
    "parallel_safety",
    "SafetyVerdict",
    "wake_triggers",
]
