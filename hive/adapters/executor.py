"""Bridge an async `WorkerAdapter` stream into the Forge's synchronous `Executor`.

The mismatch this file resolves: `forge.Executor` is `Callable[[TaskSpec, str],
ExecutionResult]` — one blocking call, one final answer — while every real
backend implements `adapters.base.WorkerAdapter`, a long-running async session
that emits partial progress as `WorkerEvent`s. Nothing else converts one shape
into the other; without this module every `Forge.run` is driven by test fakes.

Aggregation is honest by construction:

- MESSAGE text becomes `evidence`; THOUGHT/PLAN/META chatter is dropped —
  reasoning is not evidence, and carrying it upward re-inflates exactly what
  `economy.reducer` exists to shrink.
- TOOL_CALL becomes `commands_run`; TOOL_UPDATE text becomes `raw_output`,
  which the Forge already reduces before any model sees it.
- FILE_DIFF paths become `files_touched` (deduplicated, order kept).
- Test counts are parsed from the worker's own output with
  `economy.reducer.reduce_pytest`, which only reports counts off a real pytest
  result line — a bare claim of success in prose parses to nothing, `tests`
  stays None, and the merge gate refuses for lack of verification. Never a
  fabricated green.
- Money comes only from `adapter.usage()`. `WorkerUsage.exact=False` means the
  dollar figure is a guess, so it is NOT copied into the result: `usd_micros`
  stays 0 and the Forge charges the routed tier's prior, flagged as an
  estimate (`forge._run_task`). Token counts are copied as reported, never
  invented here.
- The session is always closed — success, failure, timeout, or mid-stream
  exception — or worktrees and worker processes leak. On timeout the order is
  `cancel()` then `close()`, and the timeout is capped by the task's own
  `Budget.max_seconds`: the bridge can tighten a time contract, never widen it.

Failure classification reads only the error stream (ERROR events / the raised
exception), never ordinary output, and prefers ENVIRONMENT/TRANSIENT over MODEL
when uncertain: a wrong same-tier retry costs one cheap call, while a wrong
MODEL verdict escalates a broken venv to a premium tier that fails identically
and costs real money (see `contracts.FailureClass`).

Credentials are never read, logged, or forwarded: the bridge touches only
`TaskSpec` fields, canonical `WorkerEvent`s, and `WorkerUsage` numbers — no
environment access, no adapter internals, no free-form `data` pass-through.
The single carve-out is three DOCUMENTED boolean keys an adapter may set on
`event.data` — `spend_already_recorded`, `refused`, `transient` (see
`gateway_worker.py`) — read as booleans and nothing else; the rest of `data`
is never copied forward.

The result type is `BridgedExecutionResult`, an `ExecutionResult` subclass
adding `spend_already_recorded`: True means the adapter's own transport
already wrote this call's spend to the ledger before returning (the gateway's
invariant), so recording `usd_micros` again — or substituting a tier prior —
would double-charge. False, the default, means nobody has paid yet and the
Forge's existing recording rules are correct. The figure itself is still
reported honestly either way; only its *bankedness* is what this flag adds.
"""

from __future__ import annotations

import asyncio
import re
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field

from ..contracts import AttemptSummary, FailureClass, TaskSpec, TaskState, TestResults
from ..economy.reducer import reduce_pytest
from ..forge import ExecutionResult, Executor
from .base import EventKind, WorkerAdapter, WorkerEvent, WorkerUsage


class BridgedExecutionResult(ExecutionResult):
    """`ExecutionResult` plus the one fact the plain schema cannot express.

    `spend_already_recorded=True`: the adapter's transport wrote this call's
    spend to the ledger *before returning* (e.g. `Gateway.complete`'s
    invariant), so the `usd_micros` here is real but already banked — charging
    it again, or falling back to a tier prior because it reads as unmetered,
    double-counts the same call. `False` (default): nothing is banked yet;
    the Forge's existing recording rules apply unchanged. Consumers that only
    know `ExecutionResult` keep working — read it with
    `getattr(result, "spend_already_recorded", False)`.
    """

    spend_already_recorded: bool = False

# --------------------------------------------------------------- classification

# Signatures are matched against the ERROR stream only (never ordinary output),
# lowercased. Order of checks: ENVIRONMENT → TRANSIENT → POLICY → MODEL.
# ENVIRONMENT and TRANSIENT both mean "retry same tier" to the router
# (FailureClass.retry_same_tier), so a near-miss between them is cheap; what the
# ordering must prevent is a repairable failure reaching the MODEL fallback,
# because only MODEL escalates the price ladder.

_ENVIRONMENT_SIGNS = (
    "modulenotfounderror",
    "no module named",
    "importerror",
    "filenotfounderror",
    "no such file or directory",
    "command not found",
    "is not recognized as an internal or external command",  # Windows cmd's ENOENT
    "not found on path",
    "executable file not found",
    "cannot find module",  # node's ENOENT-for-imports
    "enoent",
    "missing dependency",
    "pip install",
    "virtualenv",
    "venv",
    "daemon not reachable",
)

_TRANSIENT_SIGNS = (
    "rate limit",
    "rate-limit",
    "rate_limit",
    "too many requests",
    "timed out",
    "timeout",
    "deadline exceeded",
    "connection reset",
    "econnreset",
    "connection refused",
    "econnrefused",
    "connection aborted",
    "broken pipe",
    "temporarily unavailable",
    "service unavailable",
    "internal server error",
    "bad gateway",
    "gateway timeout",
    "overloaded",
    "try again",
)

_POLICY_SIGNS = (
    "permission denied",
    "access denied",
    "operation not permitted",
    "not permitted",
    "approval required",
    "requires approval",
    "approval needed",
    "blocked by policy",
    "policy violation",
    "forbidden",
    "eacces",
    "eperm",
)

# HTTP-ish status codes as whole tokens only: "429" must match, "1429"/"5000"
# must not — a bare substring check here would misread ordinary numbers.
_TRANSIENT_CODE_RE = re.compile(r"\b(429|50[0-4])\b")


def classify_failure(error_text: str) -> FailureClass:
    """Map an error stream to the FailureClass that decides what happens next.

    Empty input is the genuinely-uncertain case: something failed and left no
    diagnostics at all. That is TRANSIENT, not MODEL — a same-tier retry costs
    one cheap call, while escalating on zero information buys premium tokens
    for a problem nobody has even seen yet.
    """
    low = (error_text or "").lower().strip()
    if not low:
        return FailureClass.TRANSIENT
    if any(sign in low for sign in _ENVIRONMENT_SIGNS):
        return FailureClass.ENVIRONMENT
    if any(sign in low for sign in _TRANSIENT_SIGNS) or _TRANSIENT_CODE_RE.search(low):
        return FailureClass.TRANSIENT
    if any(sign in low for sign in _POLICY_SIGNS):
        return FailureClass.POLICY
    return FailureClass.MODEL


# ------------------------------------------------------------------ aggregation

# DONE statuses that mean the backend itself is reporting failure. Anything
# else ("ok", a stop reason like "end_turn" or "max_tokens", "") means the turn
# finished — whether the *work* is done is the merge gate's call, not ours.
_FAILURE_DONE_STATUSES = frozenset(
    {"failed", "error", "cancelled", "canceled", "timeout", "timed_out", "refusal"}
)

_BLOCKER_CLIP = 2000  # a blocker is a headline for the ledger, not a transcript


@dataclass
class _StreamAggregate:
    """Everything folded out of one session's event stream."""

    evidence_parts: list[str] = field(default_factory=list)
    raw_parts: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    done_status: str | None = None  # None = the stream never said the turn finished
    # Documented boolean hints an adapter may attach to event.data — structured
    # facts from our own adapter layer, so they outrank text-signature guessing.
    spend_already_recorded: bool = False
    refused: bool = False         # preflight policy/budget refusal; no transport touched
    transient_hint: bool = False  # the backend itself says this failure is retryable


def _fold(agg: _StreamAggregate, event: WorkerEvent) -> None:
    """One canonical event into the aggregate. Only named fields are read —
    free-form `event.data` is never copied forward, so nothing an adapter
    stuffed in there (headers, tokens, backend internals) can leak upward."""
    kind = event.kind
    if kind is EventKind.MESSAGE:
        if event.text:
            agg.evidence_parts.append(event.text)
    elif kind is EventKind.TOOL_CALL:
        command = event.text.strip() or event.tool_kind.strip()
        if command:
            agg.commands.append(command)
    elif kind is EventKind.TOOL_UPDATE:
        if event.text:
            agg.raw_parts.append(event.text)
    elif kind is EventKind.FILE_DIFF:
        if event.path and event.path not in agg.files:
            agg.files.append(event.path)
    elif kind is EventKind.ERROR:
        if event.text:
            agg.errors.append(event.text)
        _fold_hints(agg, event)
    elif kind is EventKind.USAGE:
        # USAGE *numbers* are deliberately dropped — the authoritative figures
        # come from adapter.usage() exactly once, after the stream ends, and
        # folding both would double-count. Only the documented boolean flag is
        # read: whether the spend behind those numbers is already banked.
        _fold_hints(agg, event)
    elif kind is EventKind.DONE:
        agg.done_status = event.status or "ok"
    # THOUGHT / PLAN / META are deliberately dropped: reasoning and
    # coordination chatter are not evidence.


def _fold_hints(agg: _StreamAggregate, event: WorkerEvent) -> None:
    """The three documented boolean keys, and nothing else, out of event.data."""
    data = event.data
    if "spend_already_recorded" in data:
        agg.spend_already_recorded = bool(data["spend_already_recorded"])
    if data.get("refused"):
        agg.refused = True
    if data.get("transient"):
        agg.transient_hint = True


def _extract_tests(*candidates: str) -> TestResults | None:
    """Parse test counts the worker's own output actually contained.

    `reduce_pytest` only reports a passed count off a real pytest result line
    (`=== N passed in Xs ===`) and failures off real FAILED entries, so prose
    like "all tests pass, trust me" yields nothing and the merge gate refuses
    for lack of verification — which is the correct outcome for an unproven
    claim.
    """
    for text in candidates:
        if not text:
            continue
        low = text.lower()
        if "passed" not in low and "failed" not in low and "error" not in low:
            continue  # cheap gate before the tokenising reducer
        reduced = reduce_pytest(text)
        if reduced.passed or reduced.failed:
            return TestResults(passed=reduced.passed, failed=reduced.failed)
    return None


def _render_attempt_history(history: list[AttemptSummary]) -> list[str]:
    """Render prior-attempt records verbatim, newest first.

    Appended LAST in `_build_prompt`, after everything else, and only when
    `history` is non-empty -- so a first attempt (empty history) renders
    byte-identical to a task with no retry machinery at all, and a retry's
    prompt is exactly the first attempt's bytes plus this block tacked onto
    the end. That ordering is what lets a provider's prompt cache serve the
    unchanged portion at the cached rate instead of silently paying full price
    on every retry (see `hive/ledger.py` `cache_health`).

    Every word of `reason` is reproduced exactly as recorded -- never
    reworded, never truncated mid-token. Trimming exact error text or test
    node ids has been measured to raise billed cost and lower success rate,
    because the next attempt can no longer match its fix against the failure
    (arXiv:2607.12161). If this block ever needs to shrink, that happens by
    dropping whole older entries before they reach here (see
    `forge._cap_attempt_history`), not by editing what is rendered below.
    """
    n = len(history)
    lines = ["", f"This task has been attempted {n} time(s) before and failed. "
                 "Do not repeat the same mistake. Newest first:"]
    for item in history:
        tag = f" [{item.failure_class}]" if item.failure_class else ""
        lines.append(f"- attempt {item.attempt}{tag}: {item.reason}")
    return lines


def _build_prompt(spec: TaskSpec) -> str:
    """The worker's brief, from canonical TaskSpec fields only."""
    lines = [spec.subject, "", spec.description]
    if spec.acceptance:
        lines += ["", "Acceptance criteria — verify each by actually running something and "
                      "paste the real command output:"]
        lines += [f"- {item}" for item in spec.acceptance]
    if spec.scope.paths:
        lines += ["", "Stay within these paths (request a scope change rather than editing "
                      "outside them):"]
        lines += [f"- {path}" for path in spec.scope.paths]
    if spec.attempt_history:
        lines += _render_attempt_history(spec.attempt_history)
    return "\n".join(lines).strip()


# ---------------------------------------------------------------- session drive

@dataclass
class _SessionOutcome:
    aggregate: _StreamAggregate
    usage: WorkerUsage
    timed_out: bool = False
    stream_error: str = ""  # non-empty when send()/start() raised, vs. yielding ERROR


async def _run_session(
    adapter: WorkerAdapter, spec: TaskSpec, cwd: str, model_profile: str,
    timeout_seconds: float,
) -> _SessionOutcome:
    agg = _StreamAggregate()
    usage = WorkerUsage()  # zeros, exact=False — the honest default when nothing reports
    session_id: str | None = None
    timed_out = False
    stream_error = ""

    async def _drive_stream() -> None:
        nonlocal session_id, usage
        # An adapter that records its own spend was necessarily constructed before
        # a job existed, so it holds a placeholder job id. The spec knows the real
        # one; handing it over here is what keeps self-recorded spend attributable
        # to the job that incurred it instead of piling up under a placeholder.
        setter = getattr(adapter, "set_job_id", None)
        if callable(setter) and spec.job_id:
            setter(spec.job_id)
        session_id = await adapter.start(spec.id, cwd, model_profile)
        async for event in adapter.send(session_id, _build_prompt(spec)):
            _fold(agg, event)
        # Authoritative spend, asked for exactly once, after the stream ends.
        # Inside the wait_for on purpose: a backend hung in usage() must not
        # hang the job any more than one hung mid-stream. A usage() failure is
        # suppressed — zeros stand, and the Forge charges the tier prior.
        with suppress(Exception):
            usage = await adapter.usage(session_id)

    try:
        try:
            await asyncio.wait_for(_drive_stream(), timeout=timeout_seconds)
        except TimeoutError:  # asyncio.TimeoutError is this same class on 3.11+
            timed_out = True
            if session_id is not None:
                with suppress(Exception):
                    await adapter.cancel(session_id)
        except Exception as exc:  # noqa: BLE001 - folded into the result, never swallowed
            # usage() is deliberately NOT attempted here: the backend is in an
            # unknown state and a second call could hang or lie. Zeros are
            # honest; the Forge charges the tier prior for unmetered work.
            stream_error = f"{type(exc).__name__}: {exc}"
    finally:
        if session_id is not None:
            with suppress(Exception):
                await adapter.close(session_id)

    return _SessionOutcome(aggregate=agg, usage=usage, timed_out=timed_out,
                           stream_error=stream_error)


def _drive(coro) -> _SessionOutcome:
    """Run the session coroutine to completion from synchronous code.

    Detect, do not assume: with no loop running, `asyncio.run` owns the thread.
    Inside a running loop (an async dashboard handler driving the synchronous
    Forge, say) blocking this thread on its own loop would deadlock — so the
    session runs on a fresh loop in a single worker thread instead. The adapter
    session lives entirely inside that one coroutine, so nothing it creates is
    bound to the caller's loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _clip(text: str, limit: int = _BLOCKER_CLIP) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _classified(agg: _StreamAggregate, error_text: str) -> FailureClass:
    """Structured hints first, text signatures second.

    A boolean an adapter set deliberately (`refused`, `transient`) is a
    deterministic feature of the failure; the signature scan over prose is the
    fallback for backends that offer nothing better.
    """
    if agg.refused:
        return FailureClass.POLICY  # declined before/without doing the work; a human decision
    if agg.transient_hint:
        return FailureClass.TRANSIENT
    return classify_failure(error_text)


def _to_result(outcome: _SessionOutcome, timeout_seconds: float,
               seconds: float) -> BridgedExecutionResult:
    agg = outcome.aggregate
    evidence = "\n".join(agg.evidence_parts).strip()
    raw_output = "\n".join(agg.raw_parts).strip()
    error_text = "\n".join(agg.errors).strip()
    usage = outcome.usage

    base = dict(
        evidence=evidence,
        commands_run=list(agg.commands),
        files_touched=list(agg.files),
        tests=_extract_tests(raw_output, evidence),
        raw_output=raw_output,
        tokens_in=usage.input_tokens,
        tokens_out=usage.output_tokens,
        # The non-negotiable: an inexact dollar figure is a guess, and a guess
        # recorded as a measurement is how a budget silently stops holding.
        # Zero lets the Forge charge the tier prior, flagged as an estimate.
        usd_micros=usage.usd_micros if usage.exact else 0,
        seconds=seconds,
        spend_already_recorded=agg.spend_already_recorded,
    )

    if outcome.timed_out:
        return BridgedExecutionResult(
            state=TaskState.FAILED, failure=FailureClass.TRANSIENT,
            blocker=_clip(f"worker exceeded {timeout_seconds:g}s; session cancelled and closed"),
            **base,
        )

    if outcome.stream_error:
        combined = f"{outcome.stream_error}\n{error_text}".strip()
        failure = _classified(agg, combined)
        if failure is FailureClass.MODEL:
            # A raised exception is the harness or backend breaking, never the
            # model genuinely failing at the task — that arrives as ERROR
            # events. Escalating a crash to a premium tier re-runs the same
            # crash at a higher price, so an unclassified exception retries
            # at the same tier instead.
            failure = FailureClass.TRANSIENT
        return BridgedExecutionResult(
            state=TaskState.FAILED, failure=failure,
            blocker=_clip(outcome.stream_error), **base,
        )

    status = (agg.done_status or "").lower()
    if agg.done_status is not None and status not in _FAILURE_DONE_STATUSES:
        # The backend says the turn finished. Whether the WORK is done is not
        # this bridge's claim to make — evidence, tests, and the merge gate
        # decide that, which is why a clean DONE after a recovered mid-stream
        # ERROR still reports DONE here.
        return BridgedExecutionResult(state=TaskState.DONE, **base)

    # Failure: either an explicit failure status, or the stream ended without
    # ever saying the turn finished — the concrete failure paths in the real
    # adapters yield ERROR and return without a DONE.
    if status == "refusal":
        failure = FailureClass.POLICY  # the backend declined; a human decision, not a retry
    else:
        failure = _classified(agg, error_text)
    if error_text:
        blocker = _clip(error_text)
    elif agg.done_status is not None:
        blocker = f"backend reported terminal status '{agg.done_status}'"
    else:
        blocker = "worker stream ended without a DONE event"
    return BridgedExecutionResult(state=TaskState.FAILED, failure=failure,
                                  blocker=blocker, **base)


# --------------------------------------------------------------------- factory

def adapter_executor(
    adapter: WorkerAdapter,
    *,
    cwd: str,
    model_profile: str = "",
    timeout_seconds: float = 900.0,
) -> Executor:
    """A synchronous `Executor` that drives `adapter` to completion per task.

    The returned callable matches `forge.Executor` exactly —
    `(TaskSpec, worker_id) -> ExecutionResult` — so `Forge.run(executor=...)`
    accepts it unchanged. Each call is one full session lifecycle:
    start → send → aggregate events → usage → close, with close guaranteed.

    `timeout_seconds` is this bridge's own ceiling per task; the effective
    limit is `min(timeout_seconds, spec.budget.max_seconds)` — the task's
    budget is a contract with the human, and a bridge default must never
    outlive it (it may only tighten it).
    """
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be > 0")

    def execute(spec: TaskSpec, worker_id: str) -> ExecutionResult:  # noqa: ARG001 - routing id; the adapter is already bound
        effective_timeout = min(float(timeout_seconds), float(spec.budget.max_seconds))
        started = time.monotonic()
        outcome = _drive(_run_session(adapter, spec, cwd, model_profile, effective_timeout))
        return _to_result(outcome, effective_timeout, seconds=time.monotonic() - started)

    return execute


__all__ = ["BridgedExecutionResult", "adapter_executor", "classify_failure"]
