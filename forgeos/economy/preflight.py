"""Count and price a call BEFORE it is made, so the call can be refused.

A ledger check after the fact can only report an overspend; it can never
prevent one — the tokens are already gone. That asymmetry is the whole reason
this module exists. The preflight estimate is the one place where "no" still
costs nothing.

Counts are labelled honestly. `exact=True` means the target model's own
tokeniser counted the text; anything else is an estimate and is marked as
such, because a budget decision made on a number that pretends to be measured
is a budget decision made on a lie.

The same "no still costs nothing" logic applies one level up, before any
token is even counted: `task_fingerprint` / `prior_outcome` / `check_repeat_work`
refuse to re-spend on a task this exact ledger already settled, using an exact
content hash of the task's contract fields — never fuzzy similarity — against
the ledger's own history.
"""

from __future__ import annotations

import hashlib
import json
import time
from enum import Enum

from pydantic import BaseModel, Field

from ..catalog import ModelCard
from ..contracts import TaskSpec, TaskState, from_micros

try:  # tiktoken is expected, but its absence must degrade to estimates, not crash.
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None  # type: ignore[assignment]

# Rough rule of thumb for English/code when no tokeniser is available.
_CHARS_PER_TOKEN = 4
ESTIMATE_ENCODING = "estimate:chars/4"


class TokenCount(BaseModel):
    """A token count that admits how it was produced."""

    tokens: int = Field(ge=0)
    exact: bool
    encoding_used: str


def count_tokens(text: str, model: str = "") -> TokenCount:
    """Count tokens with the model's own tokeniser when tiktoken has it.

    Never raises: an unknown model, an empty model name, or a missing tiktoken
    all degrade to a ceil(chars/4) estimate flagged `exact=False`.
    """
    if tiktoken is not None and model:
        try:
            enc = tiktoken.encoding_for_model(model)
        except (KeyError, ValueError):
            enc = None
        if enc is not None:
            # disallowed_special=() so text QUOTING a special token (e.g. a doc
            # containing "<|endoftext|>") is counted, not crashed on.
            return TokenCount(
                tokens=len(enc.encode(text, disallowed_special=())),
                exact=True,
                encoding_used=enc.name,
            )
    return TokenCount(
        tokens=-(-len(text) // _CHARS_PER_TOKEN),  # ceil; 0 only for empty text
        exact=False,
        encoding_used=ESTIMATE_ENCODING,
    )


class CallEstimate(BaseModel):
    """What one prospective call is expected to cost, computed before it is made."""

    model_ref: str
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    usd_micros: int = Field(ge=0)
    fits_context: bool
    exact: bool  # was tokens_in counted by the model's tokeniser, or estimated?


def estimate_call(prompt: str, expected_output_tokens: int, card: ModelCard) -> CallEstimate:
    """Price a prospective call against a ModelCard.

    `expected_output_tokens` should be the caller's declared output ceiling
    (max_tokens), not a hope — pricing the cap is what makes the estimate an
    upper bound rather than wishful thinking.
    """
    counted = count_tokens(prompt, card.model_id)
    return CallEstimate(
        model_ref=card.ref,
        tokens_in=counted.tokens,
        tokens_out=expected_output_tokens,
        usd_micros=card.cost_micros(counted.tokens, expected_output_tokens),
        fits_context=card.fits(counted.tokens),
        exact=counted.exact,
    )


class Decision(str, Enum):
    ALLOW = "allow"
    REFUSE_BUDGET = "refuse_budget"
    REFUSE_CONTEXT = "refuse_context"
    REFUSE_DUPLICATE = "refuse_duplicate"


class CallRefused(RuntimeError):
    """Raised by `raise_if_refused()`. A refusal is a brake, not advice.

    Shared by `PreflightVerdict` (token/budget refusals) and `ReceiptVerdict`
    (repeat-work refusals) — both carry a `.reason` string, which is all this
    exception needs.
    """

    def __init__(self, verdict: PreflightVerdict | ReceiptVerdict):
        super().__init__(verdict.reason)
        self.verdict = verdict


class PreflightVerdict(BaseModel):
    decision: Decision
    reason: str
    estimate: CallEstimate

    @property
    def allowed(self) -> bool:
        return self.decision is Decision.ALLOW

    def raise_if_refused(self) -> None:
        """Turn a refusal into an exception so a call site cannot quietly ignore it."""
        if not self.allowed:
            raise CallRefused(self)


def check(estimate: CallEstimate, remaining_micros: int, max_context: int = 0) -> PreflightVerdict:
    """ALLOW or REFUSE a prospective call. A refusal means the call must not be made.

    Context is checked before budget — a payload that cannot physically fit is
    refused regardless of how much money remains. Spending exactly to the cap
    is allowed; exceeding it by one microdollar is not.
    """
    tag = "" if estimate.exact else " [token count estimated]"
    if not estimate.fits_context:
        return PreflightVerdict(
            decision=Decision.REFUSE_CONTEXT,
            reason=f"{estimate.tokens_in} input tokens exceed {estimate.model_ref}'s context window{tag}",
            estimate=estimate,
        )
    if max_context and estimate.tokens_in > max_context:
        return PreflightVerdict(
            decision=Decision.REFUSE_CONTEXT,
            reason=f"{estimate.tokens_in} input tokens exceed the {max_context}-token context cap{tag}",
            estimate=estimate,
        )
    if estimate.usd_micros > remaining_micros:
        return PreflightVerdict(
            decision=Decision.REFUSE_BUDGET,
            reason=(
                f"estimated cost {estimate.usd_micros} usd_micros exceeds remaining budget "
                f"{remaining_micros} usd_micros{tag}"
            ),
            estimate=estimate,
        )
    return PreflightVerdict(
        decision=Decision.ALLOW,
        reason=(
            f"estimated {estimate.usd_micros} usd_micros ({estimate.tokens_in} in / "
            f"{estimate.tokens_out} out) within remaining {remaining_micros} usd_micros{tag}"
        ),
        estimate=estimate,
    )


# --------------------------------------------------------- repeat-work check
#
# A ledger records every task ever run, but recording it does nothing on its
# own — the value only shows up if something reads it BEFORE the next call is
# made. This is that read. It answers one question: has this EXACT contract
# already been settled, and if so, don't spend on it again.

# Bounds every scan below to a fixed number of row decodes regardless of how
# large the ledger has grown. See `Ledger.recent_tasks`'s docstring for why
# this trades perfect recall (a match older than this many tasks is missed)
# for a fixed, small cost per check, rather than a stored fingerprint column.
DEFAULT_TASK_SCAN_LIMIT = 500

# "≥ this many prior attempts failed without reaching review" before the
# advisory fires. One such failure is not yet a pattern — a single bad
# attempt could be any transient thing. Two is the point where "maybe the
# environment is broken" stops being a guess.
DEFAULT_ENVIRONMENT_FAILURE_THRESHOLD = 2

# A receipt line, not a transcript — long enough to recognise the task by,
# short enough that a preflight refusal reason stays one line.
_EVIDENCE_CLIP = 300


def _normalize_subject(subject: str) -> str:
    """Collapse whitespace and case so typographic noise can't fork one contract in two.

    Subject is prose a human or manager model wrote; "Fix the bug" and
    "fix   the  bug" describe the same task. Scope paths and capabilities get
    a different, narrower normalization (see `_normalize_path`) rather than
    this treatment — casefolding a path could silently merge two genuinely
    different files on a case-sensitive filesystem, which prose has no
    equivalent risk of.
    """
    return " ".join(subject.split()).casefold()


def _normalize_path(path: str) -> str:
    """Mirror `contracts.Scope.covers`'s own normalization.

    A scope path and the fingerprint that describes it must agree on what
    counts as "the same path" — `forgeos/x.py`, `./forgeos/x.py`, and
    `forgeos\\x.py` all mean one path to `Scope.covers`, so they must collapse
    to one entry here too, or two functionally-identical scopes would
    fingerprint as different tasks.
    """
    return path.replace("\\", "/").lstrip("./")


def _canonical_fingerprint(
    *, subject: str, scope_paths: list[str], capabilities: list[str], acceptance: list[str]
) -> str:
    """The one place that decides what makes two tasks "the same task".

    Shared by `task_fingerprint` (a live `TaskSpec`) and `_fingerprint_from_row`
    (a stored ledger row) so a task submitted fresh and that same task read
    back off the ledger always hash identically — the entire repeat-work check
    depends on that agreement holding exactly, not approximately.

    `scope_paths` and `capabilities` are sorted before hashing: a `TaskSpec`
    built with the same paths/capabilities in a different list order is the
    same task, not a different one, and list order is not part of either
    field's meaning. `acceptance` is stripped but NOT sorted — acceptance
    lines are frequently sequential ("1. tests pass", "2. deploy") and are the
    part of the contract most likely to actually carry order-dependent
    meaning, unlike a bag of scope paths or capability tags.
    """
    payload = {
        "subject": _normalize_subject(subject),
        "scope_paths": sorted(_normalize_path(p) for p in scope_paths),
        "capabilities": sorted(capabilities),
        "acceptance": [a.strip() for a in acceptance],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"fp_{hashlib.sha256(canonical).hexdigest()}"


def task_fingerprint(spec: TaskSpec) -> str:
    """A deterministic identity for "is this the same contract" — not "the same words".

    Included — the fields that actually define the contract:
    `subject` (normalized), `scope.paths` (normalized, sorted), `capabilities`
    (sorted), `acceptance` (normalized, order preserved) — what the task is
    about, where it may touch, what it needs, and what "done" means.

    Excluded, deliberately:
    - `description` — free prose that churns between two submissions of the
      same intent with zero contract change; hashing it would fingerprint
      wording, not work.
    - `id` / `job_id` / `parent_id` / `created_at` — identifiers of this
      ATTEMPT, not the task. Including any of them would make every
      submission its own fingerprint, which defeats the entire point.
    - `budget` — a spending limit is not part of what the task IS.
    - `depends_on` — a scheduling detail of one job's task graph, not the work.
    - `attempt_history` — retry bookkeeping FROM a previous run of this same
      fingerprint. Including it would make a task's second attempt hash
      differently from its first, which is exactly backwards for a
      duplicate-work check.
    - `scope.allow_expand` / `scope.note` — scope enforcement metadata, not
      part of what the scope covers; only `scope.paths` is identity.
    """
    return _canonical_fingerprint(
        subject=spec.subject,
        scope_paths=spec.scope.paths,
        capabilities=spec.capabilities,
        acceptance=spec.acceptance,
    )


class PriorTask(BaseModel):
    """One previously-settled ledger task that fingerprints identically to a new one."""

    task_id: str
    fingerprint: str
    state: str  # ledger TaskState.value: "done" (merge-gate accepted) or "failed"
    usd_micros: int
    created_at: float  # of the latest report, i.e. when the outcome was reached
    evidence: str = ""
    verdict: str | None = None  # latest report's Verdict.value; None if never reviewed


def _fingerprint_from_row(row) -> str:
    """Recompute a task's fingerprint from its stored ledger row.

    There is no stored fingerprint column (see `task_fingerprint`'s docstring
    for why) — `tasks.scope` is `Scope.model_dump_json()`, the WHOLE model
    including `allow_expand`/`note`, so this reaches into the decoded JSON for
    `paths` only, exactly matching what `task_fingerprint` reads off a live
    `TaskSpec.scope`.
    """
    scope = json.loads(row["scope"])
    return _canonical_fingerprint(
        subject=row["subject"],
        scope_paths=scope.get("paths", []),
        capabilities=json.loads(row["capabilities"]),
        acceptance=json.loads(row["acceptance"]),
    )


def matching_tasks(ledger, fingerprint: str, *, scan_limit: int = DEFAULT_TASK_SCAN_LIMIT) -> list[PriorTask]:
    """Every SETTLED prior task with this exact fingerprint, newest first.

    "Settled" means the ledger's own `tasks.state` is DONE or FAILED — a task
    still pending/running/blocked has no outcome yet to report, so it is
    skipped rather than surfaced as a false "prior" result. Bounded to the
    `scan_limit` most recently created tasks across ALL jobs (see
    `Ledger.recent_tasks`); a malformed or pre-fingerprinting-era row is
    skipped rather than crashing the whole check on one bad record.

    This is the shared scan both `prior_outcome` (wants the latest match only)
    and `check_repeat_work` (wants to COUNT repeated failures, not just see
    the latest one) are built on, so the ledger is scanned once per question,
    not once per caller.
    """
    out: list[PriorTask] = []
    for row in ledger.recent_tasks(limit=scan_limit):
        if row["state"] not in (TaskState.DONE.value, TaskState.FAILED.value):
            continue
        try:
            row_fp = _fingerprint_from_row(row)
        except (ValueError, TypeError):
            continue
        if row_fp != fingerprint:
            continue
        reports = ledger.reports_for_task(row["id"])
        latest = reports[-1] if reports else None
        evidence = (latest["evidence"] or latest["summary"]) if latest else ""
        out.append(PriorTask(
            task_id=row["id"],
            fingerprint=row_fp,
            state=row["state"],
            usd_micros=ledger.task_spend_micros(row["id"]),
            created_at=float(latest["created_at"]) if latest else float(row["created_at"]),
            evidence=evidence[:_EVIDENCE_CLIP],
            verdict=latest["verdict"] if latest else None,
        ))
    return out


def prior_outcome(
    ledger, fingerprint: str, *, scan_limit: int = DEFAULT_TASK_SCAN_LIMIT
) -> PriorTask | None:
    """The most recent SETTLED prior task with this exact fingerprint, or None.

    A thin convenience over `matching_tasks` for a caller that only wants the
    latest outcome — the raw ledger lookup `check_repeat_work` is built on to
    decide whether a new submission of the same contract is worth spending on.
    """
    matches = matching_tasks(ledger, fingerprint, scan_limit=scan_limit)
    return matches[0] if matches else None


class ReceiptVerdict(BaseModel):
    """Refusal-shaped result of checking a task against its own ledger history.

    Mirrors `PreflightVerdict`'s `decision` / `reason` shape (see `check`
    above) so a caller already handling one preflight refusal handles both the
    same way, but carries `prior` ledger evidence instead of a `CallEstimate`
    — there is no prospective model call to price here, only a ledger fact.
    """

    decision: Decision
    reason: str
    prior: PriorTask | None = None

    @property
    def allowed(self) -> bool:
        return self.decision is Decision.ALLOW

    def raise_if_refused(self) -> None:
        if not self.allowed:
            raise CallRefused(self)


def check_repeat_work(
    ledger,
    spec: TaskSpec,
    *,
    skip: bool = False,
    scan_limit: int = DEFAULT_TASK_SCAN_LIMIT,
    environment_failure_threshold: int = DEFAULT_ENVIRONMENT_FAILURE_THRESHOLD,
) -> ReceiptVerdict:
    """Refuse to re-spend on a contract this exact ledger already settled.

    Keyed off an EXACT `task_fingerprint` match only — never fuzzy similarity.
    A similarity score is a judgment call this function has no deterministic
    basis to make, and a false "this looks like it" refusal would block
    genuinely new work; an exact hash match is the one thing that can be
    asserted honestly.

    Three outcomes, checked in this order:

    1. `skip=True` — the caller's opt-out. Nothing here can tell "deliberate
       re-run" apart from "wasteful repeat" on its own (a caller retrying
       after fixing the environment looks identical to one about to waste
       money), so the caller gets the final word. Always ALLOW.
    2. The most recent matching task's ledger state is DONE — `Forge.run`
       sets a task's ledger state to DONE only when its merge gate accepted
       it (see `forge.py`'s wave-collection loop), so this means the contract
       already merged. REFUSE_DUPLICATE, naming that task's id and receipt
       figures (cost, date).
    3. At least `environment_failure_threshold` matching tasks are FAILED
       with no recorded verdict. A report only gets `verdict='fail'` from the
       merge gate reviewing an actual patch (see the merge-gate report in
       `forge.py`); a report with no verdict at all means the attempt ended
       before ever reaching review. `FailureClass` itself (ENVIRONMENT vs.
       MODEL vs. ...) is NOT persisted per report in this ledger schema —
       `WorkerReport.blocker`, where that reason would live, is accepted by
       the model but silently dropped by `Ledger.record_report`'s column list,
       a real gap this function does not paper over. So "failed with no
       verdict" is an honest PROXY for an environment/setup-shaped failure,
       not an exact `FailureClass.ENVIRONMENT` match. This is ADVICE, never a
       refusal: the environment may since have been fixed, and treating an
       unmeasured guess as a budget-stopping fact would be the same lie
       `count_tokens` refuses to tell about an estimated token count.

    Anything else — no fingerprint match, or a failure count under the
    threshold — is a clean ALLOW.

    WIRED into `Forge.run`: called per `TaskSpec` immediately before
    `self.scheduler.submit(job, tasks)`. A `REFUSE_DUPLICATE` finishes that
    task as a refused `TaskOutcome` carrying `verdict.reason` — never routed,
    never executed, no lease taken — the same shape `Forge.run` already gives
    a governor trip. `Forge.run(allow_repeat_work=True)` is the caller's
    opt-out for a deliberate re-run.

    Note this is a DIFFERENT check from the "preflight" named in `forge.py`'s
    module docstring: that one is a per-model-call budget/context refusal made
    inside the Gateway once a task is already routed. This one is a
    pre-submission ledger check, and runs before routing exists at all.
    """
    if skip:
        return ReceiptVerdict(decision=Decision.ALLOW, reason="prior-work check skipped by caller")

    fingerprint = task_fingerprint(spec)
    matches = matching_tasks(ledger, fingerprint, scan_limit=scan_limit)
    if not matches:
        return ReceiptVerdict(
            decision=Decision.ALLOW,
            reason=f"no prior task with fingerprint {fingerprint} in the last {scan_limit} tasks",
        )

    done = next((m for m in matches if m.state == TaskState.DONE.value), None)
    if done is not None:
        date = time.strftime("%Y-%m-%d", time.gmtime(done.created_at))
        return ReceiptVerdict(
            decision=Decision.REFUSE_DUPLICATE,
            reason=(
                f"this exact contract already merged as {done.task_id} for "
                f"${from_micros(done.usd_micros):.4f} on {date}"
                + (f" — {done.evidence}" if done.evidence else "")
            ),
            prior=done,
        )

    unreviewed_failures = [
        m for m in matches if m.state == TaskState.FAILED.value and m.verdict is None
    ]
    if len(unreviewed_failures) >= environment_failure_threshold:
        latest_failure = unreviewed_failures[0]
        date = time.strftime("%Y-%m-%d", time.gmtime(latest_failure.created_at))
        return ReceiptVerdict(
            decision=Decision.ALLOW,
            reason=(
                f"{len(unreviewed_failures)} prior attempts on this exact contract failed "
                f"before reaching review (most recently {latest_failure.task_id} on {date}) "
                "— often environment/setup, not the task itself; not refusing, since that "
                "may since have been fixed"
            ),
            prior=latest_failure,
        )

    return ReceiptVerdict(
        decision=Decision.ALLOW,
        reason="no blocking prior outcome for this exact fingerprint",
    )


__all__ = [
    "DEFAULT_ENVIRONMENT_FAILURE_THRESHOLD",
    "DEFAULT_TASK_SCAN_LIMIT",
    "CallEstimate",
    "CallRefused",
    "Decision",
    "ESTIMATE_ENCODING",
    "PreflightVerdict",
    "PriorTask",
    "ReceiptVerdict",
    "TokenCount",
    "check",
    "check_repeat_work",
    "count_tokens",
    "estimate_call",
    "matching_tasks",
    "prior_outcome",
    "task_fingerprint",
]
