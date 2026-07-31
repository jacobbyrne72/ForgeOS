"""ForgeOS v2 contracts — the typed objects the kernel treats as truth.

Implements the four schemas shipped in the v2 blueprint pack
(`schemas/mission-contract.schema.json`, `job-card.schema.json`,
`secure-artifact.schema.json`, `entitlement-sample.schema.json`).

The design law these encode: **the contract, not the transcript, is the source
of truth.** A model proposes; the deterministic kernel owns state, budgets,
permissions and merge authority. Everything here is therefore immutable once
issued — a revision is a new object with a bumped `revision`, never an edit —
because a contract that can be quietly rewritten mid-run cannot constrain the
run it was supposed to govern.

Kept separate from `contracts.py` on purpose. That module is the live v1 shape
the running Forge depends on; these are the v2 planes being built alongside it.
Merging them before the v2 execution path exists would break a working system to
serve a design that has not yet earned its place.
"""

from __future__ import annotations

import hashlib
from enum import Enum

from pydantic import BaseModel, Field, field_validator

from .contracts import new_id, now, to_micros


class RiskClass(str, Enum):
    """Drives verification depth and how much deliberation a merge is worth.

    Not a label for humans to read: `critical` should cost more to accept than
    `low`, and the router is expected to price that difference.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Trust(str, Enum):
    """Where an artifact's content came from, which decides what it may do.

    The load-bearing distinction in the Secure Artifact Bus: repository files,
    web pages and tool output are DATA. They may inform a decision; they may
    never alter a MissionContract, grant a permission, or install a skill.
    `model_generated` is likewise a claim, not a fact, until a proof closes.
    """

    HUMAN = "human"                      # a person authored or approved it
    DETERMINISTIC = "deterministic"      # a program produced it; reproducible
    MODEL_GENERATED = "model_generated"  # a model asserted it
    UNTRUSTED = "untrusted"              # came from outside the trust boundary


class Criterion(BaseModel):
    """One acceptance criterion and the proof route that can close it.

    `proof_requirements` is mandatory and non-empty by construction. A criterion
    with no proof route is a wish: nothing can ever discharge it, so a task
    carrying one can never legitimately complete, and the system would be forced
    to either hang or lie.
    """

    id: str
    statement: str
    proof_requirements: list[str] = Field(min_length=1)

    @field_validator("id", "statement")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v.strip()


class MissionBudget(BaseModel):
    """Hard ceilings. Never widened to make a job finish — see AGENTS.md rule 1.

    Cash is carried as a float here because the schema says so, and converted to
    integer microdollars at the boundary via `cash_limit_micros`. Float money is
    fine to *declare* and never fine to *accumulate*: a budget check summing
    floats silently stops holding.
    """

    cash_limit: float = Field(gt=0)
    currency: str = "USD"
    token_limit: int = Field(gt=0)
    manager_token_limit: int | None = None
    deadline_seconds: int = Field(gt=0)

    @property
    def cash_limit_micros(self) -> int:
        return to_micros(self.cash_limit)


class Permissions(BaseModel):
    """What a mission may touch. Enforced mechanically, not by instruction.

    `forbidden_actions` is required and may be empty — but it must be *stated*,
    because "we never said it couldn't" is how scope violations get argued after
    the fact. An empty list is an explicit claim that nothing is forbidden.
    """

    read_scope: list[str] = Field(default_factory=list)
    write_scope: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    forbidden_actions: list[str] = Field(default_factory=list)
    network_scope: list[str] = Field(default_factory=list)


class MissionContract(BaseModel):
    """The compiled objective. Immutable; a change means a new revision.

    `non_goals` earns its place: the failures that hurt most are rarely "the
    model did nothing", they are "the model did something nobody asked for".
    Stating the boundary is what makes a scope violation detectable rather than
    arguable.
    """

    mission_id: str = Field(default_factory=lambda: new_id("M"))
    revision: int = Field(default=1, ge=1)
    goal: str
    non_goals: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    invariants: list[str] = Field(default_factory=list)
    risk_class: RiskClass = RiskClass.MEDIUM
    criteria: list[Criterion] = Field(min_length=1)
    budget: MissionBudget
    permissions: Permissions = Field(default_factory=Permissions)
    created_at: float = Field(default_factory=now)

    model_config = {"frozen": True}

    @field_validator("goal")
    @classmethod
    def _goal_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("goal must not be blank")
        return v.strip()

    @field_validator("criteria")
    @classmethod
    def _unique_ids(cls, v: list[Criterion]) -> list[Criterion]:
        ids = [c.id for c in v]
        if len(set(ids)) != len(ids):
            # Duplicate ids would make proof attribution ambiguous: closing "C1"
            # could satisfy either criterion, and the gate could not tell.
            raise ValueError(f"criterion ids must be unique, got {ids}")
        return v

    @property
    def criterion_ids(self) -> list[str]:
        return [c.id for c in self.criteria]

    def revise(self, **changes) -> MissionContract:
        """A new contract at the next revision. The original is never mutated.

        Replanning is allowed — the blueprint requires it — but a revision is an
        event with a number, not an in-place edit, so an audit can always ask
        what the contract said at the moment a decision was taken.
        """
        data = self.model_dump()
        data.update(changes)
        data["revision"] = self.revision + 1
        return MissionContract(**data)

    def digest(self) -> str:
        """Stable hash of the contract's semantic content.

        Excludes `created_at` only. A timestamp would make every comparison
        unequal and the digest useless for detecting real change.

        `mission_id` is deliberately included: two different missions that happen
        to share a goal are not the same contract, and collapsing them would let
        one mission's audit trail answer for another's.
        """
        payload = self.model_dump(exclude={"created_at"}, mode="json")
        import json

        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()


class JobBudget(BaseModel):
    """Per-worker ceilings. Every field optional; absent means "inherit".

    Deliberately not defaulted to a number: a job that silently invents its own
    budget is a job the mission's ceiling does not actually bound.
    """

    input_tokens: int | None = Field(default=None, gt=0)
    output_tokens: int | None = Field(default=None, gt=0)
    cash: float | None = Field(default=None, gt=0)
    wall_seconds: int | None = Field(default=None, gt=0)

    @property
    def cash_micros(self) -> int | None:
        return to_micros(self.cash) if self.cash is not None else None


class JobCard(BaseModel):
    """The compact contract handed to one worker — the Drift Firewall's input.

    This is what a worker receives *instead of* a conversation. It names the
    criteria the worker is accountable for, the paths it may write, the tools it
    may call, and the conditions under which it must stop. Role enforcement is
    mechanical here precisely because research on multi-agent failures found
    stronger role *prompts* were not sufficient.
    """

    job_id: str = Field(default_factory=lambda: new_id("J"))
    mission_id: str
    role: str
    objective: str
    criterion_ids: list[str] = Field(min_length=1)
    read_scope: list[str] = Field(default_factory=list)
    write_scope: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    maximum_attempts: int = Field(default=3, ge=1)
    budget: JobBudget = Field(default_factory=JobBudget)
    stop_conditions: list[str] = Field(min_length=1)

    model_config = {"frozen": True}

    @field_validator("role", "objective")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v.strip()

    def covers(self, contract: MissionContract) -> list[str]:
        """Criterion ids this card claims that the contract does not define.

        Returns the offenders rather than a bool: a card accountable for a
        criterion nobody wrote can never be discharged, and the operator needs to
        know *which* one to fix.
        """
        known = set(contract.criterion_ids)
        return [c for c in self.criterion_ids if c not in known]


class SecureArtifact(BaseModel):
    """A typed, hashed, trust-labelled output — not free-form agent chat.

    Workers publish these instead of messaging each other. The point is that
    `trust` and `taint` travel WITH the content: a downstream consumer can see
    that a claim came from an untrusted repository file before deciding what it
    is allowed to influence.
    """

    artifact_id: str = Field(default_factory=lambda: new_id("A"))
    kind: str
    sha256: str
    producer: str
    mission_id: str
    job_id: str | None = None
    criterion_ids: list[str] = Field(default_factory=list)
    trust: Trust = Trust.MODEL_GENERATED
    taint: list[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=now)
    signature: str | None = None

    model_config = {"frozen": True}

    @field_validator("sha256")
    @classmethod
    def _valid_digest(cls, v: str) -> str:
        v = v.strip().lower()
        if len(v) != 64 or any(c not in "0123456789abcdef" for c in v):
            raise ValueError("sha256 must be a 64-char hex digest")
        return v

    @property
    def may_grant_authority(self) -> bool:
        """Whether this artifact's content may change policy, scope or skills.

        Only human-approved or deterministically-produced content may. This is
        the single rule that stops prompt injection through repository files and
        tool output: content arriving from outside is data, and data does not get
        to rewrite the contract that governs it.
        """
        return self.trust in (Trust.HUMAN, Trust.DETERMINISTIC) and not self.taint

    @classmethod
    def of(cls, content: str, *, kind: str, producer: str, mission_id: str, **kw):
        """Build an artifact whose digest is of the content it actually carries."""
        return cls(
            kind=kind,
            producer=producer,
            mission_id=mission_id,
            sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            **kw,
        )


class EntitlementSample(BaseModel):
    """One observation of a resource's remaining allowance, with its uncertainty.

    `confidence` and `error_bound` are mandatory-in-spirit because the whole
    point is that ForgeOS must never claim precise allowance data a provider does
    not expose. A subscription window that can only be inferred is reported as
    "0.64 +/- 0.08", never as "0.64" — and `is_measured` is what lets a consumer
    tell the two apart instead of guessing.
    """

    resource_id: str
    remaining: float = Field(ge=0.0)
    capacity: float | None = None
    unit: str
    resets_at: float | None = None
    source: str
    measured_at: float = Field(default_factory=now)
    confidence: float = Field(ge=0.0, le=1.0)
    error_bound: float = Field(default=0.0, ge=0.0)

    @property
    def is_measured(self) -> bool:
        """True only when the provider itself reported this.

        An inferred sample with high confidence is still inferred. Collapsing
        that distinction is how an estimate becomes a fact nobody can audit.
        """
        return self.source in ("official_api", "official_cli", "provider_header")

    def render(self) -> str:
        """Human-readable, and honest about which kind of number this is."""
        pct = self.remaining * 100
        if self.is_measured and self.error_bound == 0.0:
            return f"{pct:.0f}% measured"
        return f"estimated {pct:.0f}% +/- {self.error_bound * 100:.0f}%"


__all__ = [
    "Criterion",
    "EntitlementSample",
    "JobBudget",
    "JobCard",
    "MissionBudget",
    "MissionContract",
    "Permissions",
    "RiskClass",
    "SecureArtifact",
    "Trust",
]
