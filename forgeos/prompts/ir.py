"""The Model-Native Prompt Compiler's intermediate representation.

Every provider gets the same flattened text today, which throws away exactly
the capabilities that make each model cheap and accurate: Anthropic's
XML-structured cache breakpoints, OpenAI's terse single-statement
instructions, Gemini's long-context "query last" rule, DeepSeek's ~50-120x
cached-vs-uncached price gap. `PromptIR` is the fix: the provider-neutral
semantics of a prompt, stated exactly ONCE, so a per-family renderer
(`forgeos.prompts.renderers`) can turn it into each vendor's documented house
style without the semantics themselves ever being duplicated per provider.

`PromptIR` is built FROM a `TaskSpec` (`ir_from_task`) rather than replacing
it -- `acceptance`, `scope`, and `budget` already exist on `TaskSpec` and this
module reads them, it does not reinvent them. A `Capsule`
(`forgeos.economy.capsule`) supplies the ranked, budgeted context detail a
bare `TaskSpec` does not carry: which symbols, tests, and evidence a worker
actually gets, already policy-checked and token-budgeted.

Nothing in this module renders text. That is `renderers.py`'s job, and the
split matters: this file is the one place the *meaning* of a prompt is
decided, so it is decided once, not once per vendor.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from forgeos.contracts import TaskSpec
from forgeos.economy.capsule import Capsule, RefKind
from forgeos.settings import Role


class Mission(BaseModel):
    """What this task is for, and what it deliberately does not cover.

    `non_goals` are THIS TASK's exclusions -- they vary task to task, unlike
    `Authority.forbidden` below, which is the standing, cross-task policy
    (no push, no live orders, ...). Keeping them separate lets a renderer put
    the standing policy in the byte-stable prefix and the task-specific
    exclusions in the volatile tail, instead of one list mixing both.
    """

    goal: str
    non_goals: list[str] = Field(default_factory=list)


class Job(BaseModel):
    """Who is being asked to do this, and what "done" is keyed to.

    `criterion_ids` are short, stable labels ("AC1", "AC2", ...) for the
    entries in `Verification.completion_requires`, in the same order. The
    job section can then say "must satisfy AC1, AC2" without repeating the
    full criterion text a second time -- the OpenAI renderer's "stated
    exactly once" rule applies to the IR itself, not only to its output.
    """

    role: Role
    objective: str
    criterion_ids: list[str] = Field(default_factory=list)


class Authority(BaseModel):
    """What this task may read, may write, and must never do.

    `forbidden` is the standing policy (irreversible/outward-facing actions,
    hard rules) -- see `Mission.non_goals` for why it is a separate list from
    the task-specific exclusions.
    """

    read_scope: list[str] = Field(default_factory=list)
    write_scope: list[str] = Field(default_factory=list)
    forbidden: list[str] = Field(default_factory=list)


class ContextIR(BaseModel):
    """The evidence a worker gets, already ranked and budgeted upstream.

    Each list holds one short line per `Capsule` item (`"ref: reason"`) --
    `ir_from_task` is the only place that does this formatting, so every
    renderer reads the same three lists instead of re-deriving them from a
    `Capsule` each time.
    """

    symbols: list[str] = Field(default_factory=list)
    tests: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class Execution(BaseModel):
    """How the worker should act, and how many tries it gets."""

    preferred_action: str = "implement"
    max_attempts: int = 1


class Verification(BaseModel):
    """How "done" is checked. `completion_requires` is `TaskSpec.acceptance`
    verbatim -- the acceptance list is not decoration there and it is not
    decoration here either.
    """

    commands: list[str] = Field(default_factory=list)
    completion_requires: list[str] = Field(default_factory=list)


class Output(BaseModel):
    """The contract for what the worker hands back.

    The field is `schema_`, not `schema`: pydantic v2 warns on a field named
    `schema` because it shadows `BaseModel.schema`. Trailing underscore is
    the same convention Python already uses for `type_`/`id_` in exactly
    this situation.
    """

    schema_: str = ""
    verbosity: str = "normal"


class PromptIR(BaseModel):
    """The provider-neutral semantics of one prompt, stated exactly once.

    Nothing here is vendor-specific -- no XML, no section headers, no
    markdown. `forgeos.prompts.renderers.render(ir, family)` is what turns
    this into a specific vendor's documented house style.
    """

    mission: Mission
    job: Job
    authority: Authority = Field(default_factory=Authority)
    context: ContextIR = Field(default_factory=ContextIR)
    execution: Execution = Field(default_factory=Execution)
    verification: Verification = Field(default_factory=Verification)
    output: Output = Field(default_factory=Output)


def _criterion_ids(count: int) -> list[str]:
    return [f"AC{i + 1}" for i in range(count)]


def _capsule_lines(capsule: Capsule, kinds: tuple[RefKind, ...]) -> list[str]:
    return [f"{item.ref}: {item.reason}" for item in capsule.items if item.kind in kinds]


def ir_from_task(
    spec: TaskSpec,
    *,
    role: Role = Role.IMPLEMENTER,
    non_goals: Sequence[str] = (),
    forbidden: Sequence[str] = (),
    capsule: Capsule | None = None,
    preferred_action: str = "implement",
    verification_commands: Sequence[str] = (),
    output_schema: str = "",
    verbosity: str = "normal",
) -> PromptIR:
    """Build a `PromptIR` from a `TaskSpec`, composing with what already exists.

    `spec.subject`, `spec.description`, and `spec.acceptance` map straight
    onto `mission.goal`, `job.objective`, and `verification.completion_requires`
    -- `TaskSpec` already owns this data, so it is read, not re-asked-for.
    `job.criterion_ids` are generated ("AC1".."ACn") in lockstep with
    `spec.acceptance`, one id per criterion, in order.

    A `TaskSpec` alone has no notion of ranked context, a per-task policy
    exclusion list, or an output contract -- those are supplied by the
    caller (`capsule`, `non_goals`, `forbidden`, `output_schema`) rather than
    guessed at, because inventing a default for a caller's own policy would
    be a guess this module has no business making.

    When `capsule` is given, `authority.read_scope`/`write_scope` and
    `context.*` come from it -- a `Capsule` already carries the actually-used,
    policy-checked, budgeted paths (`Capsule.read_scope` is derived from what
    is really in `items`, which is strictly more precise than `spec.scope`,
    the worker's *default* working set). Without a capsule, `spec.scope.paths`
    is the only signal available and is used for both.
    """
    if capsule is not None:
        read_scope = list(capsule.read_scope)
        write_scope = list(capsule.write_scope)
        symbols = _capsule_lines(capsule, (RefKind.SYMBOL,))
        tests = _capsule_lines(capsule, (RefKind.TEST,))
        evidence = _capsule_lines(
            capsule, (RefKind.FILE_SLICE, RefKind.DIFF, RefKind.ARTIFACT, RefKind.CARD)
        )
    else:
        read_scope = list(spec.scope.paths)
        write_scope = list(spec.scope.paths)
        symbols = []
        tests = []
        evidence = []

    return PromptIR(
        mission=Mission(goal=spec.subject, non_goals=list(non_goals)),
        job=Job(
            role=role,
            objective=spec.description,
            criterion_ids=_criterion_ids(len(spec.acceptance)),
        ),
        authority=Authority(
            read_scope=read_scope, write_scope=write_scope, forbidden=list(forbidden)
        ),
        context=ContextIR(symbols=symbols, tests=tests, evidence=evidence),
        execution=Execution(
            preferred_action=preferred_action, max_attempts=spec.budget.max_iterations
        ),
        verification=Verification(
            commands=list(verification_commands), completion_requires=list(spec.acceptance)
        ),
        output=Output(schema_=output_schema, verbosity=verbosity),
    )


__all__ = [
    "Authority",
    "ContextIR",
    "Execution",
    "Job",
    "Mission",
    "Output",
    "PromptIR",
    "Verification",
    "ir_from_task",
]
