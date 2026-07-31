"""Token economy — the layers that make work cheap, ordered by leverage.

    lowerer     don't call a model at all; this is algorithmic work    (100%)
    capsule     send the smallest sufficient context                   (65-82%)
    packing     rank, bound and diversify it; surface contradictions
    reducer     shrink tool output before a model ever reads it        (~91%)
    preflight   count tokens BEFORE the call, and refuse it
    testselect  climb the verification ladder instead of running everything
    avoidance   record the counterfactual against a MEASURED baseline
    savings     prove it, keeping measurement and estimate distinct

The ordering is the point: each layer is cheaper than the one below it. Routing a
request to a cheaper model saves a fraction; not making the request saves all of it.
"""

from .avoidance import AVOIDANCE_SCHEMA, Avoidance, AvoidanceLog, AvoidanceMethod
from .capsule import Capsule, CapsuleBuilder, ContextRef, ExpansionRequest, RefKind, make_ref
from .lowerer import (
    EXCEPTION_RATE_MAX,
    STRATEGIES,
    LoweringVerdict,
    Operation,
    OperationKind,
    classify,
    exception_rate_policy,
    record_lowering,
    samples_look_structured,
    savings_estimate,
)
from .packing import (
    MAX_SOURCES,
    MAX_SOURCES_PER_DIR,
    SECTION_SPLIT,
    Excerpt,
    Overflow,
    PackReport,
    SectionBudget,
    Tension,
    allocate_sections,
    diversify,
    expand_to_boundary,
    find_tensions,
    next_reads,
    spend,
)
from .preflight import (
    ESTIMATE_ENCODING,
    CallEstimate,
    CallRefused,
    Decision,
    PreflightVerdict,
    TokenCount,
    check,
    count_tokens,
    estimate_call,
)
from .reducer import Failure, ReducedOutput, reduce_generic, reduce_pytest
from .savings import (
    STANDARD_METRICS,
    Baseline,
    BilledCostBreakdown,
    CounterfactualPlan,
    Figure,
    Outcome,
    Provenance,
    SavingsProof,
    billed_cost_breakdown,
    build_proof,
    cost_per_accepted_task,
    render_receipt,
    savings_pct,
    verify_proof,
)
from .testselect import Level, Selection, TestGraph, next_level

__all__ = [
    "AVOIDANCE_SCHEMA",
    "EXCEPTION_RATE_MAX",
    "ESTIMATE_ENCODING",
    "MAX_SOURCES",
    "MAX_SOURCES_PER_DIR",
    "SECTION_SPLIT",
    "STANDARD_METRICS",
    "STRATEGIES",
    "Avoidance",
    "AvoidanceLog",
    "AvoidanceMethod",
    "Baseline",
    "BilledCostBreakdown",
    "CallEstimate",
    "CallRefused",
    "Capsule",
    "CapsuleBuilder",
    "ContextRef",
    "CounterfactualPlan",
    "Decision",
    "Excerpt",
    "ExpansionRequest",
    "Failure",
    "Figure",
    "Level",
    "LoweringVerdict",
    "Operation",
    "OperationKind",
    "Outcome",
    "Overflow",
    "PackReport",
    "PreflightVerdict",
    "Provenance",
    "ReducedOutput",
    "RefKind",
    "SavingsProof",
    "SectionBudget",
    "Selection",
    "Tension",
    "TestGraph",
    "TokenCount",
    "allocate_sections",
    "billed_cost_breakdown",
    "build_proof",
    "check",
    "classify",
    "cost_per_accepted_task",
    "count_tokens",
    "diversify",
    "estimate_call",
    "exception_rate_policy",
    "expand_to_boundary",
    "find_tensions",
    "make_ref",
    "next_level",
    "next_reads",
    "record_lowering",
    "reduce_generic",
    "reduce_pytest",
    "render_receipt",
    "samples_look_structured",
    "savings_estimate",
    "savings_pct",
    "spend",
    "verify_proof",
]
