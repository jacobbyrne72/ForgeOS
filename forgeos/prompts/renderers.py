"""Per-family renderers: turn one `PromptIR` into each vendor's documented
house style.

Every renderer returns `(prefix, tail)`, shaped exactly like
`forgeos.prompts.prefix.StablePrefix.text` and `Prompt.full_text`'s tail:
`prefix` holds what stays byte-identical across calls for the same IR (role
contract, standing policy, output schema -- nothing task-specific), `tail`
holds what varies (the objective, ranked context, acceptance criteria). That
split is not a per-vendor stylistic choice -- `forgeos.prompts.prefix`
already models Anthropic AND OpenAI as requiring a byte-identical prefix to
ever earn a cache hit (`PROVIDER_CACHE_RULES`), so every renderer here
benefits from it, not only DeepSeek's explicitly-cache-driven one.

House styles implemented, one function per family:

    anthropic  XML-tagged sections, explicit "implement, do not merely
               suggest" action intent, no blanket-thoroughness language,
               long material (context) placed before the ask.
    openai     lean Goal/Acceptance/Constraints/Context/Evidence/Output
               contract sections, each instruction stated exactly once, no
               persona biography.
    google     one consistent "## Section" header style throughout, stable
               material first, the query LAST (their documented long-context
               placement rule).
    deepseek   the split IS the optimisation: cached input is roughly
               50-120x cheaper than uncached, so the prefix is maximised
               with everything that plausibly repeats across calls to this
               role, and the tail is explicitly labelled JOB_CARD.
    generic    plain vendor-neutral fallback for any other family string.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from pydantic import BaseModel

from forgeos.prompts.ir import PromptIR
from forgeos.prompts.roles import role_prefix
from forgeos.registry import WorkerProfile

_CHARS_PER_TOKEN = 4  # same rough approximation forgeos.prompts.prefix uses


def _bullets(items: Sequence[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "(none)"


def _numbered_criteria(ir: PromptIR) -> list[str]:
    """Pair each `verification.completion_requires` entry with its
    `job.criterion_ids` label, positionally. Falls back to a freshly minted
    "AC<n>" label if the two lists were built to different lengths by hand
    (`ir_from_task` never does this -- both come from the same
    `spec.acceptance` there) rather than raising on a caller-constructed IR.
    """
    ids = ir.job.criterion_ids
    reqs = ir.verification.completion_requires
    return [f"{ids[i] if i < len(ids) else f'AC{i + 1}'}: {req}" for i, req in enumerate(reqs)]


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


# ------------------------------------------------------------------ anthropic


def _render_anthropic(ir: PromptIR) -> tuple[str, str]:
    """XML-tagged sections; long material (context) before the ask; explicit
    "implement, do not merely suggest" action intent; no
    blanket-thoroughness language -- it overtriggers newer models.
    """
    prefix = "\n\n".join(
        [
            f"<role>\n{role_prefix(ir.job.role).text}\n</role>",
            f"<forbidden>\n{_bullets(ir.authority.forbidden)}\n</forbidden>",
            f"<output_schema>\n{ir.output.schema_ or '(none specified)'}\n</output_schema>",
        ]
    )
    tail = "\n\n" + "\n\n".join(
        [
            "<context>\n<symbols>\n{}\n</symbols>\n<tests>\n{}\n</tests>\n"
            "<evidence>\n{}\n</evidence>\n</context>".format(
                _bullets(ir.context.symbols), _bullets(ir.context.tests), _bullets(ir.context.evidence)
            ),
            f"<non_goals>\n{_bullets(ir.mission.non_goals)}\n</non_goals>",
            f"<mission>\n{ir.mission.goal}\n</mission>",
            "<job>\n{}\nImplement, do not merely suggest. Preferred action: {}. "
            "Max attempts: {}. Criteria: {}.\n</job>".format(
                ir.job.objective,
                ir.execution.preferred_action,
                ir.execution.max_attempts,
                ", ".join(ir.job.criterion_ids) or "(none)",
            ),
            "<verification>\nCommands:\n{}\nCompletion requires:\n{}\n</verification>".format(
                _bullets(ir.verification.commands), _bullets(_numbered_criteria(ir))
            ),
            "<completion_contract>\nDone only when every item in <verification> passes "
            "against real command output. Do not report done otherwise.\n</completion_contract>",
        ]
    )
    return prefix, tail


# --------------------------------------------------------------------- openai


def _render_openai(ir: PromptIR) -> tuple[str, str]:
    """Lean six-section layout, each instruction stated exactly once.

    Unlike the other three renderers this does NOT embed roles.py's full
    "You are the X" contract prose -- `Role: <name>` is the whole role
    statement, matching OpenAI's documented preference for direct
    instructions over persona/role-play framing.
    """
    prefix = "\n".join(
        [
            f"Role: {ir.job.role.value}",
            "Constraints:",
            _bullets(ir.authority.forbidden),
            f"Output contract: {ir.output.schema_ or '(unspecified, plain text)'} "
            f"(verbosity: {ir.output.verbosity})",
        ]
    )
    tail = "\n" + "\n".join(
        [
            f"Goal: {ir.mission.goal}",
            f"Objective: {ir.job.objective}",
            f"Action: {ir.execution.preferred_action} (max {ir.execution.max_attempts} attempt(s))",
            "Non-goals:",
            _bullets(ir.mission.non_goals),
            "Acceptance:",
            _bullets(_numbered_criteria(ir)),
            "Verify by running:",
            _bullets(ir.verification.commands),
            "Context:",
            _bullets(ir.context.symbols + ir.context.tests),
            "Evidence:",
            _bullets(ir.context.evidence),
        ]
    )
    return prefix, tail


# --------------------------------------------------------------------- google


def _render_google(ir: PromptIR) -> tuple[str, str]:
    """One consistent "## Section" header style throughout; stable material
    first, the query LAST -- Gemini's documented long-context placement rule.
    """
    prefix = "\n\n".join(
        [
            f"## Role\n{role_prefix(ir.job.role).text}",
            f"## Constraints\n{_bullets(ir.authority.forbidden)}",
            f"## Output contract\n{ir.output.schema_ or '(none specified)'} "
            f"(verbosity: {ir.output.verbosity})",
        ]
    )
    tail = "\n\n" + "\n\n".join(
        [
            "## Context\nSymbols:\n{}\nTests:\n{}\nEvidence:\n{}".format(
                _bullets(ir.context.symbols), _bullets(ir.context.tests), _bullets(ir.context.evidence)
            ),
            f"## Non-goals\n{_bullets(ir.mission.non_goals)}",
            "## Verification\nCommands:\n{}\nCompletion requires:\n{}".format(
                _bullets(ir.verification.commands), _bullets(_numbered_criteria(ir))
            ),
            f"## Query\nGoal: {ir.mission.goal}\nObjective: {ir.job.objective}\n"
            f"Action: {ir.execution.preferred_action} (max {ir.execution.max_attempts} attempt(s))",
        ]
    )
    return prefix, tail


# ------------------------------------------------------------------ deepseek


def _render_deepseek(ir: PromptIR) -> tuple[str, str]:
    """Maximise the byte-stable cached prefix: role contract, policy, and
    output schema -- everything that plausibly repeats across calls to this
    role -- go in the prefix; the volatile per-task material is pushed into
    a tail explicitly labelled JOB_CARD so the boundary stays visible to
    whoever reads a rendered prompt later. Nothing timestamp-like or
    otherwise dynamic belongs in the prefix; this renderer only ever writes
    role/policy/schema text there, never a task-specific value.
    """
    prefix = "\n".join(
        [
            "STABLE_PREFIX",
            f"ROLE: {ir.job.role.value}",
            role_prefix(ir.job.role).text,
            "POLICY (forbidden):",
            _bullets(ir.authority.forbidden),
            f"OUTPUT_SCHEMA: {ir.output.schema_ or '(unspecified, plain text)'}",
            f"OUTPUT_VERBOSITY: {ir.output.verbosity}",
        ]
    )
    tail = "\n" + "\n".join(
        [
            "JOB_CARD",
            f"GOAL: {ir.mission.goal}",
            f"OBJECTIVE: {ir.job.objective}",
            "NON_GOALS:",
            _bullets(ir.mission.non_goals),
            f"ACTION: {ir.execution.preferred_action} (max {ir.execution.max_attempts} attempt(s))",
            "SYMBOLS:",
            _bullets(ir.context.symbols),
            "TESTS:",
            _bullets(ir.context.tests),
            "EVIDENCE:",
            _bullets(ir.context.evidence),
            "VERIFY_COMMANDS:",
            _bullets(ir.verification.commands),
            "COMPLETION_REQUIRES:",
            _bullets(_numbered_criteria(ir)),
        ]
    )
    return prefix, tail


# --------------------------------------------------------------------- generic


def _render_generic(ir: PromptIR) -> tuple[str, str]:
    """Plain, vendor-neutral rendering for any family with no dedicated
    backend -- a caller that got the family string wrong, or is talking to a
    vendor this module has not modeled yet, still gets a complete prompt.
    """
    prefix = "\n".join(
        [
            f"Role: {ir.job.role.value}",
            f"Role contract: {role_prefix(ir.job.role).text}",
            "Forbidden:",
            _bullets(ir.authority.forbidden),
            f"Output schema: {ir.output.schema_ or '(none specified)'} "
            f"(verbosity: {ir.output.verbosity})",
        ]
    )
    tail = "\n" + "\n".join(
        [
            f"Goal: {ir.mission.goal}",
            "Non-goals:",
            _bullets(ir.mission.non_goals),
            f"Objective: {ir.job.objective}",
            f"Action: {ir.execution.preferred_action} (max {ir.execution.max_attempts} attempt(s))",
            "Symbols:",
            _bullets(ir.context.symbols),
            "Tests:",
            _bullets(ir.context.tests),
            "Evidence:",
            _bullets(ir.context.evidence),
            "Verify by running:",
            _bullets(ir.verification.commands),
            "Completion requires:",
            _bullets(_numbered_criteria(ir)),
        ]
    )
    return prefix, tail


_RENDERERS: dict[str, Callable[[PromptIR], tuple[str, str]]] = {
    "anthropic": _render_anthropic,
    "openai": _render_openai,
    "google": _render_google,
    "deepseek": _render_deepseek,
}


class RenderedPrompt(BaseModel):
    """One family's rendering of a `PromptIR`.

    `prefix`/`tail` are shaped exactly like `StablePrefix.text` and
    `Prompt.full_text`'s tail (`forgeos.prompts.prefix`): prefix first,
    unmodified, every call for the same IR; only the tail carries what
    varies. A caller wanting an actual cache breakpoint wraps `prefix` in a
    `StablePrefix(role=ir.job.role, version=..., text=prefix)` and calls
    `build_prompt(that, tail)` exactly as today -- this model does not
    duplicate that machinery, it produces text already shaped to fit it.
    """

    prefix: str
    tail: str
    family: str
    est_tokens: int


def render(ir: PromptIR, family: str) -> RenderedPrompt:
    """Render `ir` in `family`'s documented house style.

    An unrecognised `family` falls back to `_render_generic` instead of
    raising: a caller that got the family string wrong, or is talking to a
    vendor with no dedicated backend yet, still gets a complete, working
    prompt rather than an exception. `RenderedPrompt.family` records the
    family actually used ("generic" on fallback), not whatever string the
    caller passed in, so a downstream cache-key or routing decision keyed on
    `.family` reflects the template shape that was really applied.
    """
    renderer = _RENDERERS.get(family)
    prefix, tail = (renderer or _render_generic)(ir)
    return RenderedPrompt(
        prefix=prefix,
        tail=tail,
        family=family if renderer is not None else "generic",
        est_tokens=_estimate_tokens(prefix) + _estimate_tokens(tail),
    )


def resolve_family(profile: WorkerProfile | None) -> str:
    """Provider family for `render()`, reusing forge.py's own `_family_of`
    rather than a second vendor-detection table.

    The import is deferred to inside this function rather than at module
    scope: `forgeos.forge` pulls in the full execution stack (governor,
    ledger, scheduler, worktrees, ...) and measured ~3s to import on this
    machine. `forgeos.prompts` must stay cheap to import for every caller
    that only wants `PromptIR`/`render` and never resolves a family --
    exactly the reason `forgeos.prompts.prefix` already defers its own
    `import tiktoken` inside `_cache_floor_encoding` instead of importing it
    at module scope. Reusing `_family_of` outright (instead of copying
    `_FAMILY_HINTS`) keeps the vendor-hint table at exactly one copy, so it
    cannot drift the moment either one gains a new vendor hint.
    """
    from forgeos.forge import _family_of

    return _family_of(profile)


__all__ = [
    "RenderedPrompt",
    "render",
    "resolve_family",
]
