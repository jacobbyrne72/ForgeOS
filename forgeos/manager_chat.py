"""The manager you talk to — the only thing a human should need to address.

Everything else in ForgeOS is machinery. This is the surface: you describe what
you want, the manager asks what it genuinely cannot infer, proposes a blueprint,
and **nothing executes until you approve it**. After that it runs the workers,
watches them, and writes their follow-up prompts itself.

The point is not convenience. It is that *you should never have to write a prompt
for a worker*. Prompt quality is a skill with a cost attached — a vague brief
buys a wrong answer at full price, and the person paying is rarely the person
best placed to phrase it. So the manager owns that translation: a human writes
intent, and the manager compiles intent into `JobCard`s with scopes, budgets,
stop conditions and acceptance criteria attached. Workers never see raw user
prose.

Three rules make this safe rather than merely pleasant:

1. **Approval is mechanical, not conversational.** `ManagerSession.approve()`
   must be called with the digest of the exact blueprint shown. A model cannot
   talk its way past it, and a blueprint that changed after you read it will not
   match the digest you approved.
2. **Clarification is cheap; execution is not.** The conversation runs on the
   cheapest capable worker. Only after approval does anything touch a
   subscription or a metered API.
3. **Supervision reads heartbeats, never transcripts.** A manager that re-reads
   a worker's conversation to decide whether it is stuck costs more than the
   work it is supervising. Follow-ups are generated from a ~90-byte status, and
   the intervention is a JobCard revision, not a chat message.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

from pydantic import BaseModel, Field

from .contracts import new_id, now
from .contracts_v2 import (
    Criterion,
    JobCard,
    MissionBudget,
    MissionContract,
    Permissions,
    RiskClass,
)


class Phase(str, Enum):
    """Where a session is. Ordered; a session never moves backwards except by
    an explicit revision, which returns it to AWAITING_APPROVAL."""

    GATHERING = "gathering"              # asking what cannot be inferred
    AWAITING_APPROVAL = "awaiting_approval"  # blueprint proposed, human deciding
    EXECUTING = "executing"              # approved; workers running
    DONE = "done"
    ABANDONED = "abandoned"


class Question(BaseModel):
    """One thing the manager genuinely cannot infer.

    `why` is required. A question without a stated reason is how an interview
    turns into an interrogation: the human cannot tell which answers matter, so
    they either over-answer (expensive) or disengage (worse).
    """

    id: str = Field(default_factory=lambda: new_id("Q"))
    text: str
    why: str
    options: list[str] = Field(default_factory=list)
    answer: str = ""

    @property
    def answered(self) -> bool:
        return bool(self.answer.strip())


class Blueprint(BaseModel):
    """What the manager proposes to do, in terms a human can actually check.

    Deliberately not a rendering of the MissionContract's JSON. A human
    approving work needs to see the goal, what is explicitly excluded, what it
    will cost at most, and how each promise will be proven — not a schema dump.
    `non_goals` is shown because the expensive surprises are almost never "it
    did nothing", they are "it also did this".
    """

    contract: MissionContract
    plan: list[str] = Field(min_length=1)
    estimated_usd: float = 0.0
    estimate_is_measured: bool = False

    def render(self) -> str:
        c = self.contract
        lines = [
            f"GOAL      {c.goal}",
            f"RISK      {c.risk_class.value}",
            "",
            "PLAN",
            *[f"  {i}. {s}" for i, s in enumerate(self.plan, 1)],
            "",
            "WILL NOT",
            *([f"  - {n}" for n in c.non_goals] or ["  (nothing excluded — say so if that is wrong)"]),
            "",
            "PROOF",
            *[f"  {cr.id}  {cr.statement}\n       via {', '.join(cr.proof_requirements)}"
              for cr in c.criteria],
            "",
            f"CEILING   ${c.budget.cash_limit:.2f} hard cap, "
            f"{c.budget.token_limit:,} tokens, {c.budget.deadline_seconds}s",
        ]
        # An estimate and a measurement are different claims and must not look
        # alike at the moment someone is deciding whether to spend money.
        if self.estimated_usd:
            kind = "measured" if self.estimate_is_measured else "estimated"
            lines.append(f"EXPECTED  ${self.estimated_usd:.4f} ({kind})")
        lines += ["", f"DIGEST    {c.digest()[:16]}  (approval is bound to this)"]
        return "\n".join(lines)


class ApprovalError(RuntimeError):
    """Raised when execution is attempted without a matching approval."""


class ManagerSession(BaseModel):
    """One conversation, from "I want X" to accepted work.

    Holds no model client. The caller supplies `ask` — anything that turns a
    prompt into text — so the whole session is testable without a network, and
    so the clarify phase can run on a free worker while execution runs on a paid
    one.
    """

    session_id: str = Field(default_factory=lambda: new_id("S"))
    objective: str
    phase: Phase = Phase.GATHERING
    questions: list[Question] = Field(default_factory=list)
    blueprint: Blueprint | None = None
    approved_digest: str = ""
    approved_at: float = 0.0
    revisions: int = 0
    transcript: list[dict] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    # ----------------------------------------------------------- conversation

    def say(self, who: str, text: str) -> None:
        """Append to the human-facing log.

        This is a record for the person, NOT context for a worker. Workers get
        JobCards; nothing here is ever replayed into a model prompt, which is
        what stops a long conversation quietly becoming an expensive one.
        """
        self.transcript.append({"who": who, "text": text, "at": now()})

    def ask_questions(self, questions: list[Question]) -> list[Question]:
        """Record what the manager needs to know. Returns only the unanswered.

        Questions are additive across rounds: an answered one is never asked
        again, so a second round costs only the genuinely new ambiguity.
        """
        known = {q.text.strip().lower() for q in self.questions}
        for q in questions:
            if q.text.strip().lower() not in known:
                self.questions.append(q)
        return self.open_questions()

    def open_questions(self) -> list[Question]:
        return [q for q in self.questions if not q.answered]

    def answer(self, question_id: str, text: str) -> None:
        for q in self.questions:
            if q.id == question_id:
                q.answer = text
                self.say("human", f"{q.text} -> {text}")
                return
        raise KeyError(f"no such question: {question_id}")

    # ------------------------------------------------------------- blueprint

    def propose(self, blueprint: Blueprint) -> str:
        """Put a blueprint up for approval and return what the human should read.

        Moves to AWAITING_APPROVAL. Proposing again (after a revision) is fine
        and increments `revisions` — replanning is allowed, it just has to be
        visible and re-approved rather than slipped in.
        """
        if self.blueprint is not None:
            self.revisions += 1
        self.blueprint = blueprint
        # Any prior approval is void: it was for a different plan.
        self.approved_digest = ""
        self.approved_at = 0.0
        self.phase = Phase.AWAITING_APPROVAL
        rendered = blueprint.render()
        self.say("manager", rendered)
        return rendered

    def approve(self, digest: str) -> None:
        """Approve the exact blueprint that was shown.

        The digest is required and must match. This is the whole safety
        property: a blueprint revised after you read it does not carry your
        approval forward, and no amount of conversation substitutes for the
        check. `approve("")` or a stale digest raises.
        """
        if self.blueprint is None:
            raise ApprovalError("nothing has been proposed yet")
        expected = self.blueprint.contract.digest()
        if not digest or digest not in (expected, expected[:16]):
            raise ApprovalError(
                "approval digest does not match the proposed blueprint — "
                "it changed after you read it, so the approval does not carry"
            )
        self.approved_digest = expected
        self.approved_at = now()
        self.phase = Phase.EXECUTING
        self.say("human", f"approved {expected[:16]}")

    @property
    def is_approved(self) -> bool:
        return (
            self.blueprint is not None
            and bool(self.approved_digest)
            and self.approved_digest == self.blueprint.contract.digest()
        )

    # ------------------------------------------------------------- execution

    def job_cards(self, *, roles: dict[str, list[str]] | None = None) -> list[JobCard]:
        """Compile the approved contract into per-worker JobCards.

        Refuses without approval. This is the function that means a human never
        writes a worker prompt: scope, tools, budget, stop conditions and the
        criteria a worker is accountable for are all derived from the contract
        they already read and agreed to.

        `roles` maps a role name to the criterion ids it owns; the default puts
        every criterion on one implementer, which is the right shape for small
        work and the cheapest topology that can produce a result.
        """
        if not self.is_approved:
            raise ApprovalError(
                "no approved blueprint — nothing may execute. This is the gate, "
                "not a formality."
            )
        c = self.blueprint.contract
        roles = roles or {"implementer": c.criterion_ids}

        cards: list[JobCard] = []
        for role, criterion_ids in roles.items():
            cards.append(
                JobCard(
                    mission_id=c.mission_id,
                    role=role,
                    objective=c.goal,
                    criterion_ids=list(criterion_ids),
                    read_scope=list(c.permissions.read_scope),
                    write_scope=list(c.permissions.write_scope),
                    allowed_tools=list(c.permissions.allowed_tools),
                    stop_conditions=[
                        "criteria_pass",
                        "budget_exhausted",
                        "specification_conflict",
                    ],
                )
            )
        return cards

    # ------------------------------------------------------------ supervision

    def follow_up(self, heartbeat: dict) -> str | None:
        """A corrective instruction derived from a compact status, or None.

        Reads a heartbeat — not a transcript. Returns None when the worker is
        fine, because the cheapest supervision is the intervention that does not
        happen: waking a manager on every heartbeat costs more than the work it
        watches.

        Deliberately mechanical rather than a model call. Every trigger here is
        a fact already in the heartbeat, and paying a model to notice
        "tokens went up, evidence did not" would be paying for arithmetic.
        """
        if not heartbeat:
            return None

        state = str(heartbeat.get("state", "")).lower()
        blocker = str(heartbeat.get("blocker", "")).strip()
        needs_manager = bool(heartbeat.get("needs_manager"))

        if blocker:
            # A stated blocker is the cheapest signal there is: the worker has
            # already done the diagnosis, so answer it rather than re-derive it.
            return f"You reported: {blocker}. Resolve it within your current scope, or stop and say precisely what you need."
        if needs_manager:
            return "You asked for a decision. State the options you see and the one you recommend, in under 80 words."
        if state == "blocked":
            return "You are blocked with no stated reason. Say what you tried and what stopped you."

        tokens = int(heartbeat.get("tokens_used", 0) or 0)
        evidence = heartbeat.get("evidence_count", None)
        if tokens > 0 and evidence == 0:
            # Spend without evidence is the drift signature that matters: it is
            # how a worker burns a budget looking busy.
            return "Spend is accumulating with no evidence recorded. Produce one concrete piece of evidence for your criteria, or stop."
        return None

    def close(self, *, accepted: bool) -> None:
        self.phase = Phase.DONE if accepted else Phase.ABANDONED


def draft_blueprint(
    objective: str,
    *,
    plan: list[str],
    criteria: list[tuple[str, str, list[str]]],
    max_usd: float,
    token_limit: int = 100_000,
    deadline_seconds: int = 1800,
    non_goals: list[str] | None = None,
    write_scope: list[str] | None = None,
    risk: RiskClass = RiskClass.MEDIUM,
    estimated_usd: float = 0.0,
) -> Blueprint:
    """Build a Blueprint from plain parts.

    A helper, not a policy: the manager decides *what* to propose, this only
    assembles it. `criteria` is (id, statement, proof_requirements) — the proof
    list is required by `Criterion`, so a blueprint literally cannot be drafted
    with a promise nobody can check.
    """
    contract = MissionContract(
        goal=objective,
        non_goals=list(non_goals or []),
        risk_class=risk,
        criteria=[Criterion(id=i, statement=s, proof_requirements=p) for i, s, p in criteria],
        budget=MissionBudget(
            cash_limit=max_usd,
            token_limit=token_limit,
            deadline_seconds=deadline_seconds,
        ),
        permissions=Permissions(write_scope=list(write_scope or [])),
    )
    return Blueprint(
        contract=contract,
        plan=plan,
        estimated_usd=estimated_usd,
        estimate_is_measured=False,
    )


# The caller supplies this: anything mapping a prompt to text. Keeping it a
# plain callable is what lets the clarify phase run on a free worker while
# execution runs on a paid one, and lets every test here run without a network.
Ask = Callable[[str], str]


__all__ = [
    "ApprovalError",
    "Ask",
    "Blueprint",
    "ManagerSession",
    "Phase",
    "Question",
    "draft_blueprint",
]
