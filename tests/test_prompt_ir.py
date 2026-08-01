"""Tests for the Model-Native Prompt Compiler: `PromptIR` + per-family `render`.

Every provider getting the same flattened text throws away the exact
capabilities that make each model cheap and accurate. These tests hold two
things to account: that `ir_from_task` actually reads what `TaskSpec` (and a
`Capsule`, when given one) already carries rather than re-asking for it, and
that the cost lever the whole prompt layer rests on -- a byte-identical
prefix, no matter how the volatile tail changes -- genuinely holds across
every family renderer, not only the one DeepSeek's docs make explicit.
"""

from __future__ import annotations

from forgeos.contracts import Budget, Scope, TaskSpec
from forgeos.economy.capsule import CapsuleBuilder, RefKind, make_ref
from forgeos.prompts.ir import ir_from_task
from forgeos.prompts.renderers import RenderedPrompt, render, resolve_family
from forgeos.registry import Adapter, WorkerProfile
from forgeos.settings import Role

FAMILIES = ("anthropic", "openai", "google", "deepseek")


def _task(**overrides) -> TaskSpec:
    fields = dict(
        job_id="job_test",
        subject="Add prompt IR",
        description="Build PromptIR and per-family renderers for the model-native prompt compiler.",
        acceptance=["pytest tests/test_prompt_ir.py passes", "ruff check forgeos/prompts passes"],
        scope=Scope(paths=["forgeos/prompts/"]),
        budget=Budget(max_iterations=5),
    )
    fields.update(overrides)
    return TaskSpec(**fields)


def _capsule():
    builder = CapsuleBuilder(budget=10_000)
    builder.add(
        make_ref(
            RefKind.SYMBOL,
            "symbol://forgeos/prompts/ir.py#PromptIR",
            "class PromptIR(BaseModel): ...",
            "primary symbol",
        )
    )
    builder.add(
        make_ref(
            RefKind.TEST,
            "test://tests/test_prompt_ir.py::test_ir_from_task_maps_every_field",
            "def test_ir_from_task_maps_every_field(): ...",
            "closest test",
        )
    )
    builder.add(
        make_ref(
            RefKind.FILE_SLICE,
            "file_slice://forgeos/contracts.py#L167-L196",
            "class TaskSpec(BaseModel): ...",
            "acceptance/scope/budget shape",
        )
    )
    return builder.finish(
        objective="wire the prompt IR",
        acceptance=["pytest passes"],
        write_scope=["forgeos/prompts/"],
    )


# --------------------------------------------------------------- ir_from_task


def test_ir_from_task_maps_every_field_without_a_capsule():
    spec = _task()
    ir = ir_from_task(
        spec,
        role=Role.REVIEWER,
        non_goals=["do not touch unrelated files"],
        forbidden=["never push to a default branch"],
        preferred_action="review",
        verification_commands=["rtk proxy python -m pytest tests/test_prompt_ir.py -q"],
        output_schema="WorkerReport",
        verbosity="terse",
    )

    assert ir.mission.goal == spec.subject
    assert ir.mission.non_goals == ["do not touch unrelated files"]

    assert ir.job.role is Role.REVIEWER
    assert ir.job.objective == spec.description
    assert ir.job.criterion_ids == ["AC1", "AC2"]

    assert ir.authority.read_scope == spec.scope.paths
    assert ir.authority.write_scope == spec.scope.paths
    assert ir.authority.forbidden == ["never push to a default branch"]

    assert ir.context.symbols == []
    assert ir.context.tests == []
    assert ir.context.evidence == []

    assert ir.execution.preferred_action == "review"
    assert ir.execution.max_attempts == spec.budget.max_iterations

    assert ir.verification.commands == ["rtk proxy python -m pytest tests/test_prompt_ir.py -q"]
    assert ir.verification.completion_requires == spec.acceptance

    assert ir.output.schema_ == "WorkerReport"
    assert ir.output.verbosity == "terse"


def test_ir_from_task_reads_context_and_scope_from_a_capsule():
    spec = _task()
    capsule = _capsule()
    ir = ir_from_task(spec, capsule=capsule)

    assert ir.authority.read_scope == capsule.read_scope
    assert ir.authority.write_scope == capsule.write_scope
    assert ir.authority.read_scope != spec.scope.paths  # the capsule is strictly more precise

    assert len(ir.context.symbols) == 1
    assert "PromptIR" in ir.context.symbols[0]
    assert len(ir.context.tests) == 1
    assert len(ir.context.evidence) == 1


def test_ir_from_task_criterion_ids_track_acceptance_length():
    spec = _task(acceptance=["one criterion only"])
    ir = ir_from_task(spec)
    assert ir.job.criterion_ids == ["AC1"]
    assert ir.verification.completion_requires == ["one criterion only"]


# --------------------------------------------------------------- anthropic


def test_anthropic_renders_documented_xml_tags():
    ir = ir_from_task(_task(), non_goals=["skip the dashboard"], forbidden=["no live orders"])
    rp = render(ir, "anthropic")
    full = rp.prefix + rp.tail

    for tag in ("<mission>", "<job>", "<non_goals>", "<context>", "<verification>", "<completion_contract>"):
        assert tag in full

    assert "implement, do not merely suggest" in full.lower()
    assert "thorough" not in full.lower()  # no blanket-thoroughness language


# ------------------------------------------------------------------- openai


def test_openai_states_each_instruction_exactly_once_with_no_persona_bio():
    ir = ir_from_task(_task(), non_goals=["skip the dashboard"], forbidden=["no live orders"])
    rp = render(ir, "openai")
    full = rp.prefix + rp.tail

    for header in ("Goal:", "Acceptance:", "Constraints:", "Context:", "Evidence:", "Output contract:"):
        assert full.count(header) == 1, f"{header!r} should appear exactly once, got {full.count(header)}"

    assert "you are the" not in full.lower()  # no persona biography


# ------------------------------------------------------------------- google


def test_google_places_the_query_last_with_consistent_headers():
    ir = ir_from_task(_task(), non_goals=["skip the dashboard"], forbidden=["no live orders"])
    rp = render(ir, "google")
    full = rp.prefix + rp.tail

    headers = ("## Role", "## Constraints", "## Output contract", "## Context", "## Non-goals", "## Verification", "## Query")
    for header in headers:
        assert header in full

    # Every section uses the same "## " style -- no XML section tags mixed in.
    assert "<mission>" not in full

    last_header_index = max(full.rindex(h) for h in headers)
    assert last_header_index == full.rindex("## Query"), "the query must be the last section"


# ----------------------------------------------------------------- deepseek


def test_deepseek_maximises_the_stable_prefix():
    ir = ir_from_task(_task(), non_goals=["skip the dashboard"], forbidden=["no live orders"])
    rp = render(ir, "deepseek")

    assert "STABLE_PREFIX" in rp.prefix
    assert "STABLE_PREFIX" not in rp.tail
    assert "JOB_CARD" in rp.tail
    assert "JOB_CARD" not in rp.prefix

    # Role contract, policy, and output schema are stable -- they belong in the prefix.
    assert "no live orders" in rp.prefix
    assert ir.job.objective not in rp.prefix

    # The volatile per-task material belongs in the tail.
    assert ir.mission.goal in rp.tail
    assert ir.job.objective in rp.tail


# ------------------------------------------------------------- byte-stability


def test_rendering_the_same_ir_twice_is_byte_identical():
    ir = ir_from_task(_task(), non_goals=["skip the dashboard"], forbidden=["no live orders"])
    for family in (*FAMILIES, "some-unmodeled-vendor"):
        a = render(ir, family)
        b = render(ir, family)
        assert a.prefix == b.prefix
        assert a.tail == b.tail
        assert a.prefix.encode("utf-8") == b.prefix.encode("utf-8")


def test_changing_only_the_tail_never_perturbs_the_prefix():
    ir = ir_from_task(_task(), non_goals=["skip the dashboard"], forbidden=["no live orders"])

    for family in FAMILIES:
        before = render(ir, family)

        changed = ir.model_copy(deep=True)
        changed.job.objective = "a completely different objective than before"
        changed.mission.non_goals.append("also skip the reports page")
        changed.context.evidence.append("evidence://forgeos/prompts/renderers.py")
        changed.verification.completion_requires.append("a brand new acceptance criterion")

        after = render(changed, family)

        assert after.prefix == before.prefix, f"{family}: tail-only change perturbed the prefix"
        assert after.tail != before.tail, f"{family}: tail did not actually change"


# ------------------------------------------------------------ unknown family


def test_unknown_family_falls_back_without_raising():
    ir = ir_from_task(_task())
    rp = render(ir, "some-unmodeled-vendor")

    assert isinstance(rp, RenderedPrompt)
    assert rp.family == "generic"
    assert "Role contract:" in rp.prefix  # the generic renderer's own marker


# ------------------------------------------------------- constraints survive


def test_forbidden_and_non_goals_survive_every_family():
    ir = ir_from_task(
        _task(),
        forbidden=["never touch forgeos/leases.py"],
        non_goals=["do not refactor unrelated code"],
    )
    for family in (*FAMILIES, "totally-unmodeled-vendor"):
        rp = render(ir, family)
        full = rp.prefix + rp.tail
        assert "never touch forgeos/leases.py" in full, f"{family} dropped a forbidden entry"
        assert "do not refactor unrelated code" in full, f"{family} dropped a non-goal"


# ------------------------------------------------------------- family reuse


def test_resolve_family_reuses_forge_family_of():
    from forgeos.forge import _family_of

    profile = WorkerProfile(worker_id="w1", adapter=Adapter.CLI_TEAM, vendor="claude", model="claude-opus-5")
    assert resolve_family(profile) == _family_of(profile) == "anthropic"


def test_resolve_family_none_profile_is_empty_string():
    assert resolve_family(None) == ""


def test_resolve_family_falls_back_to_local_for_an_unhinted_local_adapter():
    profile = WorkerProfile(worker_id="w2", adapter=Adapter.OLLAMA, model="my-local-model")
    assert resolve_family(profile) == "local"


def test_resolve_family_unknown_gateway_profile_is_empty_string():
    profile = WorkerProfile(worker_id="w3", adapter=Adapter.GATEWAY, model="totally-unknown-thing")
    assert resolve_family(profile) == ""
