"""The context capsule: a ranked, budgeted, content-addressed handoff to a worker.

Passing a whole repository is the single largest token waste in agent harnesses,
and it is not just expensive -- it makes results worse. Models retrieve less
reliably from the middle of a long context ("Lost in the Middle"), so a small
ranked capsule beats a big dump on cost AND accuracy at once. This module is
the assembler for that capsule: it does no vector search and no repo scanning.
It only ranks and budgets `ContextRef`s the caller already resolved.

Silent truncation is the failure mode this exists to prevent. A worker that
never learns something was withheld cannot ask for it back -- so every ref
that does not make the cut is recorded in `Capsule.excluded` with a reason,
and every worker request for more is recorded too (`expansion_request`), so
the next capsule for the same kind of task can be built better.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

from ..contracts import new_id, now
from ..policy import ReadRequest, check_read
from .avoidance import AvoidanceLog, AvoidanceMethod
from .preflight import count_tokens


class RefKind(str, Enum):
    """What kind of thing a ContextRef points at."""

    SYMBOL = "symbol"
    FILE_SLICE = "file_slice"
    TEST = "test"
    DIFF = "diff"
    ARTIFACT = "artifact"
    CARD = "card"


# Ref kinds that may name a filesystem path and so must clear the policy gate.
# ARTIFACT and CARD are content-addressed / knowledge-base entries, never a
# path on disk -- there is nothing for check_read to check.
_PATH_KINDS = frozenset({RefKind.SYMBOL, RefKind.FILE_SLICE, RefKind.TEST, RefKind.DIFF})


class ContextRef(BaseModel):
    """One content-addressed reference into a capsule.

    `ref` is a small URI naming what this is and where: `symbol://path#name`,
    `file_slice://path#L10-L50`, `test://path::test_name`, `diff://path`,
    `artifact://sha256/<hash>`, `card://<id>`. `sha256` is the hash of the
    exact content this ref stood for at build time, so a worker (or a later
    audit) can tell whether the underlying file has since moved on.
    """

    kind: RefKind
    ref: str
    sha256: str
    reason: str
    token_estimate: int = Field(ge=0)

    @field_validator("ref", "reason")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v.strip()

    @field_validator("sha256")
    @classmethod
    def _valid_digest(cls, v: str) -> str:
        v = v.strip().lower()
        if len(v) != 64 or any(c not in "0123456789abcdef" for c in v):
            raise ValueError("sha256 must be a 64-char hex digest")
        return v


def make_ref(kind: RefKind, ref: str, content: str, reason: str, *, model: str = "") -> ContextRef:
    """Build a ContextRef from real content: hash it, price it, done.

    This is the only place this module touches raw text. Everywhere else
    moves already-built ContextRefs, so the same slice is never re-counted
    twice. Reuses `preflight.count_tokens` rather than a second tokenizer --
    one counting rule for the whole economy layer.
    """
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    counted = count_tokens(content, model)
    return ContextRef(kind=kind, ref=ref, sha256=digest, reason=reason, token_estimate=counted.tokens)


def _extract_path(ref: ContextRef) -> str | None:
    """Pull a filesystem path out of a ref string, when it names one.

    ARTIFACT/CARD refs and any ref pointing at a bare artifact
    (`...://sha256/<hash>`) have nothing to check against the filesystem.
    """
    if ref.kind not in _PATH_KINDS:
        return None
    _, _, rest = ref.ref.partition("://")
    if not rest or rest.startswith("sha256/"):
        return None
    cut = len(rest)
    for sep in ("::", "#"):
        idx = rest.find(sep)
        if idx != -1:
            cut = min(cut, idx)
    path = rest[:cut]
    return path or None


class ExpansionRequest(BaseModel):
    """A worker's first-class ask for more than its default scope carried.

    Scope is a default, not a cage: this is the escape hatch, and every
    request is kept so future capsule-building can learn what workers
    actually needed instead of guessing again next time.
    """

    id: str = Field(default_factory=lambda: new_id("exp"))
    kind: RefKind
    detail: str
    created_at: float = Field(default_factory=now)

    @field_validator("detail")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("detail must not be blank -- an unexplained expansion request can't be acted on")
        return v.strip()


class Capsule(BaseModel):
    """The finished, budgeted handoff. `manifest()` is what the worker sees.

    `total_tokens` is self-correcting: it is always recomputed from `items`
    after construction, so it cannot drift from what it claims to summarize
    even if a caller hand-builds a Capsule with a stale number.
    """

    objective: str
    acceptance: list[str] = Field(default_factory=list)
    write_scope: list[str] = Field(default_factory=list)
    read_scope: list[str] = Field(default_factory=list)
    items: list[ContextRef] = Field(default_factory=list)
    excluded: dict[str, str] = Field(default_factory=dict)
    total_tokens: int = Field(default=0, ge=0)

    @field_validator("objective")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("objective must not be blank")
        return v.strip()

    @model_validator(mode="after")
    def _recompute_total(self) -> "Capsule":
        self.total_tokens = sum(i.token_estimate for i in self.items)
        return self

    def manifest(self) -> dict:
        """The compact dict a worker actually receives.

        Same discipline as `WorkerReport.heartbeat()`: everything the worker
        needs to act, nothing it has to pay to read past.
        """
        return {
            "objective": self.objective,
            "acceptance": list(self.acceptance),
            "write_scope": list(self.write_scope),
            "read_scope": list(self.read_scope),
            "items": [
                {
                    "kind": i.kind.value,
                    "ref": i.ref,
                    "sha256": i.sha256,
                    "reason": i.reason,
                    "tokens": i.token_estimate,
                }
                for i in self.items
            ],
            "excluded": dict(self.excluded),
            "total_tokens": self.total_tokens,
        }

    def record_savings(
        self,
        avoidance_log: AvoidanceLog,
        job_id: str,
        task_id: str | None,
        baseline_tokens: int,
        *,
        baseline_source: str = "tiktoken count of the full unranked context this capsule replaced",
    ) -> str:
        """Log this capsule's cost against a measured baseline.

        `baseline_tokens` must already be a measured count -- e.g. `count_tokens()`
        run over the full file(s)/repo slice that would have been sent without
        this capsule -- never a guess. `AvoidanceLog` rejects a blank
        `baseline_source` at the edge, but only the caller knows whether the
        number itself was actually measured, so say how here rather than
        accepting the vague default when a real number is available.
        """
        return avoidance_log.record(
            job_id,
            AvoidanceMethod.PACK,
            baseline_tokens=baseline_tokens,
            actual_tokens=self.total_tokens,
            baseline_source=baseline_source,
            task_id=task_id,
        )


class CapsuleBuilder:
    """Assembles a Capsule against a hard token budget.

    `add()` never raises and never silently drops: a refusal returns False
    and is recorded in `excluded` with why, whether the reason was policy
    (the ref is denied outright, regardless of budget) or budget (it would
    have fit the rules but not the wallet).
    """

    def __init__(self, budget: int):
        if budget <= 0:
            raise ValueError("budget must be > 0")
        self.budget = budget
        self.items: list[ContextRef] = []
        self.excluded: dict[str, str] = {}
        self.expansion_requests: list[ExpansionRequest] = []

    @property
    def total_tokens(self) -> int:
        return sum(i.token_estimate for i in self.items)

    def add(self, ref: ContextRef) -> bool:
        """Try to include one ref. Policy is checked before budget: a denied
        ref is refused outright, not just when the wallet happens to be tight.
        """
        path = _extract_path(ref)
        if path is not None:
            decision = check_read(ReadRequest(path=path, offset=0, limit=1))
            if not decision.allowed:
                self.excluded[ref.ref] = f"policy denied: {decision.reason}"
                return False

        projected = self.total_tokens + ref.token_estimate
        if projected > self.budget:
            self.excluded[ref.ref] = (
                f"budget: {ref.token_estimate} tokens would bring the capsule to {projected}, "
                f"over the {self.budget}-token budget ({self.budget - self.total_tokens} remaining)"
            )
            return False

        self.items.append(ref)
        return True

    def expansion_request(self, ref_kind: RefKind, detail: str) -> ExpansionRequest:
        """Record a worker's ask for context beyond what the capsule carried."""
        req = ExpansionRequest(kind=ref_kind, detail=detail)
        self.expansion_requests.append(req)
        return req

    def finish(self, *, objective: str, acceptance: Sequence[str], write_scope: Sequence[str]) -> Capsule:
        """Turn the accumulated adds into a Capsule. `read_scope` is derived,
        not asked for: it is the set of paths actually present in `items`, so
        it can never claim to cover more than the capsule really carries.
        """
        read_scope = sorted({p for r in self.items if (p := _extract_path(r)) is not None})
        return Capsule(
            objective=objective,
            acceptance=list(acceptance),
            write_scope=list(write_scope),
            read_scope=read_scope,
            items=list(self.items),
            excluded=dict(self.excluded),
        )

    @classmethod
    def build(
        cls,
        objective: str,
        acceptance: Sequence[str],
        primary_symbols: Sequence[ContextRef] = (),
        callers: Sequence[ContextRef] = (),
        related_tests: Sequence[ContextRef] = (),
        recent_failures: Sequence[ContextRef] = (),
        write_scope: Sequence[str] = (),
        budget: int = 8_000,
    ) -> Capsule:
        """Ranked assembly: cheapest-and-most-relevant first.

        Order: primary symbols (the thing being changed) -> closest tests
        (the fastest signal on whether a change worked) -> callers (blast
        radius) -> history (recent_failures -- what has already been tried,
        so it is not retried blind). No embeddings, no search: this merges
        lists the caller already ranked, stopping the instant the budget
        would be exceeded.
        """
        builder = cls(budget=budget)
        for ref in (*primary_symbols, *related_tests, *callers, *recent_failures):
            builder.add(ref)
        return builder.finish(objective=objective, acceptance=acceptance, write_scope=write_scope)


__all__ = [
    "Capsule",
    "CapsuleBuilder",
    "ContextRef",
    "ExpansionRequest",
    "RefKind",
    "make_ref",
]
