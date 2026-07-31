"""Auto Memory — mine what already happened into durable, reviewed lessons.

Every task the fleet has ever run is already sitting in `ledger.py` (tasks,
reports, spend) and `events.py` (the append-only event log). Nobody reads
that history back FOR patterns unless something does it deliberately — this
module is that something.

It is deliberately NOT a model call. AGENTS.md rule 10 draws a hard line:
"Nothing becomes a rule because someone said it" — an outside claim enters
the knowledge base labelled unverified and stays that way until evidence and
a contradiction check promote it. A model musing "the fleet seems bad at X"
is exactly the kind of confident, unfalsifiable opinion rule 10 exists to
keep out. So this module does only what counting can do honestly: find
EXACT repeats — the same merge-refusal reason, the same (worker, capability)
that has never once won, the same cost blowing out relative to the fleet,
the same contract resubmitted — and hand each one to `knowledge.claims`'s
`ClaimStore` as an unverified candidate with its receipts attached.
Corroboration, contradiction-checking, and promotion are `claims.py`'s job;
this module never calls `ClaimStore.promote()` or `mark_verified()`.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import defaultdict
from enum import Enum

from pydantic import BaseModel, Field

from ..contracts import TaskState, Verdict
from ..events import EventLog, EventType
from .claims import Claim, ClaimStore, ClaimType, claim_key

# `_fingerprint_from_row` is `economy.preflight`'s private row -> fingerprint
# reader, built specifically so "a task read back off the ledger" and "a
# freshly submitted TaskSpec" hash identically (see its docstring). This
# module needs exactly that: turn stored `tasks` rows back into the same
# identity `task_fingerprint` would produce, to group resubmissions of the
# same contract. Reusing it is the same private-attribute-reuse call
# `__main__.cmd_receipts` already makes against `ledger._conn` — the public
# surface has no grouping accessor for this yet, and re-deriving the same
# normalization rules here would risk them silently drifting from the ones
# `check_repeat_work` depends on.
from ..economy.preflight import _fingerprint_from_row

# Below this many distinct tasks/workers, a repeat is not yet a pattern — one
# rejection for the same reason could be any transient thing. Mirrors
# `claims.MIN_SOURCES_TO_PROMOTE` and
# `preflight.DEFAULT_ENVIRONMENT_FAILURE_THRESHOLD`: two is the point where
# "coincidence" stops being the more likely explanation.
DEFAULT_MIN_OCCURRENCES = 2

# Bounds every scan to a fixed number of row/event decodes regardless of how
# large the ledger/event log has grown — the same trade `Ledger.recent_tasks`
# and `preflight.DEFAULT_TASK_SCAN_LIMIT` already make, at the honest cost of
# missing a pattern older than the `scan_limit`-th most recent record.
DEFAULT_SCAN_LIMIT = 500

# A worker's average accepted-task cost on a capability must exceed the fleet
# median by this multiple before it is flagged. Below this, cost differences
# are the ordinary spread between models/tiers, not a candidate worth a
# human's eyes.
COST_OUTLIER_MULTIPLIER = 2.0


class PatternKind(str, Enum):
    MERGE_REFUSAL_REPEATED = "merge_refusal_repeated"
    CAPABILITY_MISCAST = "capability_miscast"
    CAPABILITY_EXPENSIVE = "capability_expensive"
    DUPLICATE_CONTRACT_RESUBMITTED = "duplicate_contract_resubmitted"


class Receipt(BaseModel):
    """One piece of evidence behind a candidate — a task id a human can look up."""

    task_id: str
    detail: str = ""
    created_at: float = 0.0


class LessonCandidate(BaseModel):
    """A deterministically-mined pattern, not yet a claim.

    `file_candidates` is what turns this into a `Claim` — always through
    `ClaimStore.add`, always starting `VerificationStatus.UNVERIFIED`.
    Nothing in this module ever promotes one.
    """

    kind: PatternKind
    subject: str
    occurrences: int
    receipts: list[Receipt] = Field(default_factory=list)
    claim_text: str
    claim_type: ClaimType = ClaimType.WARNING
    source_ref: str
    topics: list[str] = Field(default_factory=list)


class _Outcome(BaseModel):
    """One settled task, decoded once and shared by every pattern miner below."""

    task_id: str
    subject: str
    capabilities: list[str]
    worker_id: str
    succeeded: bool
    usd_micros: int
    created_at: float
    fingerprint: str | None


def _settled_outcomes(ledger, scan_limit: int) -> list[_Outcome]:
    """Decode the `scan_limit` most recent settled tasks once, for every miner to share.

    "Settled" mirrors `preflight.matching_tasks`: DONE or FAILED only — a
    task still in flight has no outcome yet to mine a pattern from.
    """
    out: list[_Outcome] = []
    for row in ledger.recent_tasks(limit=scan_limit):
        if row["state"] not in (TaskState.DONE.value, TaskState.FAILED.value):
            continue

        reports = ledger.reports_for_task(row["id"])
        final = reports[-1] if reports else None
        worker_id = final["worker_id"] if final else ""
        verdict = final["verdict"] if final else None
        # Mirrors `WorkerReport.succeeded`: DONE and not explicitly FAILed.
        # A missing verdict (a report predating the merge gate, or one that
        # never went through review) does not count against the worker.
        succeeded = row["state"] == TaskState.DONE.value and verdict != Verdict.FAIL.value
        created_at = float(final["created_at"]) if final else float(row["created_at"])

        try:
            capabilities = json.loads(row["capabilities"])
        except (ValueError, TypeError):
            capabilities = []

        try:
            fingerprint = _fingerprint_from_row(row)
        except (ValueError, TypeError, KeyError):
            fingerprint = None

        out.append(
            _Outcome(
                task_id=row["id"],
                subject=row["subject"],
                capabilities=capabilities,
                worker_id=worker_id,
                succeeded=succeeded,
                usd_micros=ledger.task_spend_micros(row["id"]),
                created_at=created_at,
                fingerprint=fingerprint,
            )
        )
    return out


# --------------------------------------------------------- pattern 1: merge refusals


def _mine_merge_refusals(
    events: EventLog, *, min_occurrences: int, scan_limit: int
) -> list[LessonCandidate]:
    """"This fleet keeps failing X" — the same merge-gate refusal reason recurring.

    Reads `EventType.TASK_REJECTED`'s `reason` payload, which `forge.py` sets
    to `"; ".join(verdict.reasons)` (see its merge-gate block) — the exact
    per-reason strings `MergeGate.evaluate` produces, e.g. "no independent
    review". Split back into individual reasons and counted by the number of
    DISTINCT tasks that hit each one, not raw event count: a single task
    retried across several attempts emits one `TASK_REJECTED` per attempt,
    and that is one flaky task, not several.
    """
    after = max(0, events.last_seq() - scan_limit)
    reason_tasks: dict[str, set[str]] = defaultdict(set)
    for ev in events.replay(after_seq=after):
        if ev.type is not EventType.TASK_REJECTED or not ev.task_id:
            continue
        for part in str(ev.payload.get("reason", "")).split("; "):
            reason = part.strip()
            if reason:
                reason_tasks[reason].add(ev.task_id)

    out: list[LessonCandidate] = []
    for reason in sorted(reason_tasks):
        task_ids = reason_tasks[reason]
        if len(task_ids) < min_occurrences:
            continue
        out.append(
            LessonCandidate(
                kind=PatternKind.MERGE_REFUSAL_REPEATED,
                subject=reason,
                occurrences=len(task_ids),
                receipts=[Receipt(task_id=tid) for tid in sorted(task_ids)],
                claim_text=(
                    f"The merge gate has refused {len(task_ids)} distinct task(s) for "
                    f"the same reason: {reason!r}. That looks like a recurring failure "
                    f"mode in this fleet, not isolated bad luck."
                ),
                source_ref=f"automemory:merge_refusal:{claim_key(reason)}",
                topics=["merge_refusal"],
            )
        )
    return out


# ------------------------------------------------------ pattern 2: worker miscast


def _mine_capability_miscast(
    outcomes: list[_Outcome], *, min_occurrences: int
) -> list[LessonCandidate]:
    """"W is miscast for C" — a worker that has never once won on a capability."""
    groups: dict[tuple[str, str], list[_Outcome]] = defaultdict(list)
    for o in outcomes:
        if not o.worker_id:
            continue
        for cap in o.capabilities:
            groups[(o.worker_id, cap)].append(o)

    out: list[LessonCandidate] = []
    for key in sorted(groups):
        worker_id, cap = key
        items = groups[key]
        if len(items) < min_occurrences or any(o.succeeded for o in items):
            continue
        out.append(
            LessonCandidate(
                kind=PatternKind.CAPABILITY_MISCAST,
                subject=f"{worker_id} / {cap}",
                occurrences=len(items),
                receipts=[Receipt(task_id=o.task_id, created_at=o.created_at) for o in items],
                claim_text=(
                    f"Worker {worker_id!r} has attempted {len(items)} task(s) tagged "
                    f"capability {cap!r} and has not won a single one. This worker "
                    f"looks miscast for this capability."
                ),
                source_ref=f"automemory:miscast:{worker_id}:{cap}",
                topics=["capability_miscast", cap],
            )
        )
    return out


# ----------------------------------------------------- pattern 3: cost outliers


def _mine_capability_expensive(
    outcomes: list[_Outcome], *, min_occurrences: int
) -> list[LessonCandidate]:
    """"W is expensive for C" — accepted-task cost far above the fleet median."""
    by_cap: dict[str, dict[str, list[_Outcome]]] = defaultdict(lambda: defaultdict(list))
    for o in outcomes:
        if not o.succeeded or not o.worker_id:
            continue
        for cap in o.capabilities:
            by_cap[cap][o.worker_id].append(o)

    out: list[LessonCandidate] = []
    for cap in sorted(by_cap):
        by_worker = by_cap[cap]
        avg_by_worker = {
            w: sum(o.usd_micros for o in items) / len(items) for w, items in by_worker.items()
        }
        # Need at least two workers to have a fleet to compare against — one
        # worker's average cannot be an outlier relative to itself.
        if len(avg_by_worker) < 2:
            continue
        fleet_median = statistics.median(avg_by_worker.values())
        if fleet_median <= 0:
            continue

        for worker_id in sorted(by_worker):
            items = by_worker[worker_id]
            if len(items) < min_occurrences:
                continue
            avg = avg_by_worker[worker_id]
            if avg <= fleet_median * COST_OUTLIER_MULTIPLIER:
                continue
            out.append(
                LessonCandidate(
                    kind=PatternKind.CAPABILITY_EXPENSIVE,
                    subject=f"{worker_id} / {cap}",
                    occurrences=len(items),
                    receipts=[
                        Receipt(
                            task_id=o.task_id,
                            created_at=o.created_at,
                            detail=f"${o.usd_micros / 1_000_000:.4f}",
                        )
                        for o in items
                    ],
                    claim_text=(
                        f"Worker {worker_id!r} averages ${avg / 1_000_000:.4f} per "
                        f"accepted task on capability {cap!r} across {len(items)} "
                        f"task(s), against a fleet median of "
                        f"${fleet_median / 1_000_000:.4f}. This worker looks expensive "
                        f"for this capability."
                    ),
                    source_ref=f"automemory:expensive:{worker_id}:{cap}",
                    topics=["capability_expensive", cap],
                )
            )
    return out


# -------------------------------------------------- pattern 4: duplicate contracts


def _mine_duplicate_contracts(
    outcomes: list[_Outcome], *, min_occurrences: int
) -> list[LessonCandidate]:
    """"Callers keep resubmitting X" — the exact same task contract, more than once.

    Grouped by the same fingerprint `economy.preflight.check_repeat_work`
    would use to refuse a resubmission — each `_Outcome` here is one `tasks`
    row (one real submission), never a retry of an existing row, so the
    count is genuinely how many times this contract was submitted, not how
    many attempts one submission took.
    """
    groups: dict[str, list[_Outcome]] = defaultdict(list)
    for o in outcomes:
        if o.fingerprint:
            groups[o.fingerprint].append(o)

    out: list[LessonCandidate] = []
    for fp in sorted(groups):
        items = groups[fp]
        if len(items) < min_occurrences:
            continue
        subject = items[0].subject
        out.append(
            LessonCandidate(
                kind=PatternKind.DUPLICATE_CONTRACT_RESUBMITTED,
                subject=subject,
                occurrences=len(items),
                receipts=[Receipt(task_id=o.task_id, created_at=o.created_at) for o in items],
                claim_text=(
                    f"The exact same task contract ({subject!r}) has been submitted "
                    f"{len(items)} times. Callers keep resubmitting this instead of "
                    f"reusing the settled result."
                ),
                source_ref=f"automemory:duplicate_contract:{fp}",
                topics=["duplicate_contract"],
            )
        )
    return out


# --------------------------------------------------------------------- public


def mine_lessons(
    ledger,
    events: EventLog,
    *,
    min_occurrences: int = DEFAULT_MIN_OCCURRENCES,
    scan_limit: int = DEFAULT_SCAN_LIMIT,
) -> list[LessonCandidate]:
    """Deterministic patterns only — pattern extraction is counting, not judgment.

    Read-only against `ledger` and `events`; never mutates either. Every
    candidate returned carries its receipts (task ids, dates, counts) and a
    proposed claim TEXT — filing it is a separate, explicit step
    (`file_candidates`), and nothing here decides whether the pattern is
    TRUE, only that it recurred at least `min_occurrences` times.
    """
    outcomes = _settled_outcomes(ledger, scan_limit)
    candidates: list[LessonCandidate] = []
    candidates += _mine_merge_refusals(events, min_occurrences=min_occurrences, scan_limit=scan_limit)
    candidates += _mine_capability_miscast(outcomes, min_occurrences=min_occurrences)
    candidates += _mine_capability_expensive(outcomes, min_occurrences=min_occurrences)
    candidates += _mine_duplicate_contracts(outcomes, min_occurrences=min_occurrences)
    return candidates


def file_candidates(claims_store: ClaimStore, candidates: list[LessonCandidate]) -> list[Claim]:
    """Enter each candidate through `ClaimStore.add` — the ONLY door into the knowledge base.

    `ClaimStore.add` is itself idempotent per `(claim_key(text), source_ref)`:
    a repeated `source_ref` is `INSERT OR IGNORE`, never a second
    corroborating source (see its docstring). Re-mining an unchanged ledger
    reproduces the same `claim_text`/`source_ref` for the same pattern, so
    re-filing it is a no-op, not a duplicate. Every claim lands
    `VerificationStatus.UNVERIFIED` (or `CORROBORATED` if a second, genuinely
    different source already exists) — this function never calls
    `promote()` or `mark_verified()`.
    """
    filed: list[Claim] = []
    for c in candidates:
        evidence = json.dumps([r.model_dump() for r in c.receipts], sort_keys=True)
        source_hash = hashlib.sha256(evidence.encode("utf-8")).hexdigest()
        claim = claims_store.add(
            c.claim_text,
            c.claim_type,
            source_ref=c.source_ref,
            source_hash=source_hash,
            locator=f"occurrences={c.occurrences}",
            topics=[c.kind.value, *c.topics],
        )
        filed.append(claim)
    return filed


__all__ = [
    "COST_OUTLIER_MULTIPLIER",
    "DEFAULT_MIN_OCCURRENCES",
    "DEFAULT_SCAN_LIMIT",
    "LessonCandidate",
    "PatternKind",
    "Receipt",
    "file_candidates",
    "mine_lessons",
]
