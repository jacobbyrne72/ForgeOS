"""Forge Lowerer tests: shape detection, safety properties, honest accounting.

No network, no model calls — classify is pure keyword/structure analysis and
these tests prove it stays that way (same input, same verdict, every time).
"""

from __future__ import annotations

import pytest

from forgeos.economy.avoidance import AvoidanceLog, AvoidanceMethod
from forgeos.economy.lowerer import (
    EXCEPTION_RATE_MAX,
    HANDFUL,
    MIN_ITEMS_TO_LOWER,
    PROCEED,
    REVERT,
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

REAL_TOOLS = (
    "ripgrep", "rg", "ast-grep", "tree-sitter", "json", "csv",
    "git diff", "difflib", "sqlite", "hashlib", "python",
)


def _op(**kw) -> Operation:
    base = dict(
        description="Scan 20,000 source files and find every import statement",
        item_count=20_000,
        per_item_tokens_estimate=800,
        has_deterministic_structure=True,
    )
    base.update(kw)
    return Operation(**base)


# ------------------------------------------------------------ shape detection


def test_20k_file_import_scan_is_search_and_lowerable():
    v = classify(_op())
    assert v.kind is OperationKind.SEARCH
    assert v.lowerable is True
    assert v.requires_sample_validation is True
    assert "rg" in v.strategy or "ripgrep" in v.strategy
    # O(n) -> O(1): the deterministic path must be a tiny fraction of the baseline.
    assert v.estimated_deterministic_tokens < v.estimated_model_tokens // 100


def test_architecture_judgment_is_reason_and_not_lowerable():
    v = classify(
        _op(
            description="Decide whether this architecture is sound and recommend improvements",
            item_count=10,
        )
    )
    assert v.kind is OperationKind.REASON
    assert v.lowerable is False
    assert v.estimated_deterministic_tokens == v.estimated_model_tokens  # no cheaper path exists


def test_field_extraction_from_records_is_extract():
    v = classify(
        _op(
            description="Parse 50,000 JSON records and extract seven fields from each",
            item_count=50_000,
            per_item_tokens_estimate=400,
        )
    )
    assert v.kind is OperationKind.EXTRACT
    assert v.lowerable is True
    assert "json" in v.strategy


def test_counting_failures_by_root_cause_is_count_aggregate():
    v = classify(
        _op(
            description="Count test failures grouped by root cause across the logs",
            item_count=5_000,
            per_item_tokens_estimate=300,
        )
    )
    assert v.kind is OperationKind.COUNT_AGGREGATE
    assert v.lowerable is True


def test_duplicate_removal_is_dedupe_with_hashing():
    v = classify(
        _op(
            description="Remove duplicate documents that have identical content",
            item_count=10_000,
            per_item_tokens_estimate=500,
        )
    )
    assert v.kind is OperationKind.DEDUPE
    assert v.lowerable is True
    assert "hash" in v.strategy


def test_config_comparison_is_diff_compare():
    v = classify(
        _op(
            description="Compare the two exported config trees and list every changed key",
            item_count=500,
            per_item_tokens_estimate=300,
        )
    )
    assert v.kind is OperationKind.DIFF_COMPARE
    assert v.lowerable is True


def test_case_conversion_rename_is_mechanical_transform():
    v = classify(
        _op(
            description="Rename every fixture file from camelCase to snake_case",
            item_count=2_000,
            per_item_tokens_estimate=200,
        )
    )
    assert v.kind is OperationKind.TRANSFORM_MECHANICAL
    assert v.lowerable is True


def test_rule_based_classification_lowers_but_semantic_does_not():
    semantic = classify(
        _op(
            description="Classify each customer support message",
            item_count=10_000,
            per_item_tokens_estimate=100,
            has_deterministic_structure=False,
        )
    )
    assert semantic.kind is OperationKind.REASON
    assert semantic.lowerable is False

    ruled = classify(
        _op(
            description="Classify each log line by error code into known buckets",
            item_count=10_000,
            per_item_tokens_estimate=100,
        )
    )
    assert ruled.kind is OperationKind.CLASSIFY_RULES
    assert ruled.lowerable is True


def test_unrecognised_shape_defaults_to_reason_with_low_confidence():
    v = classify(_op(description="Do the thing with the stuff", item_count=100))
    assert v.kind is OperationKind.REASON
    assert v.lowerable is False
    assert v.confidence <= 0.5


def test_classify_is_deterministic():
    op = _op()
    assert classify(op) == classify(op)  # no model, no randomness, no state


# --------------------------------------------- sample validation (the big one)


@pytest.mark.parametrize("count", [101, 500, 20_000, 1_000_000])
def test_lowering_more_than_100_items_always_demands_sample_validation(count):
    v = classify(_op(item_count=count))
    assert v.lowerable is True
    assert v.requires_sample_validation is True


def test_validation_demanded_above_a_handful_even_when_not_worth_lowering():
    # 6 items: uneconomic to lower, but if someone lowers anyway, validate first.
    v = classify(_op(item_count=6))
    assert v.lowerable is False
    assert v.requires_sample_validation is True
    assert 6 > HANDFUL


def test_handful_or_fewer_does_not_demand_validation():
    v = classify(_op(item_count=3, per_item_tokens_estimate=800))
    assert v.lowerable is True  # 2400 model tokens vs 1000 synthesis overhead
    assert v.requires_sample_validation is False


# ------------------------------------------------------- overhead economics


def test_single_item_is_not_worth_lowering():
    v = classify(
        _op(
            description="Find every reference in one document",
            item_count=1,
            per_item_tokens_estimate=5_000,
        )
    )
    assert v.lowerable is False
    assert "overhead" in v.reason
    assert v.requires_sample_validation is False
    assert 1 < MIN_ITEMS_TO_LOWER


def test_zero_items_do_not_crash_or_divide_by_zero():
    op = _op(item_count=0)
    v = classify(op)
    assert v.lowerable is False
    assert v.estimated_model_tokens == 0
    s = savings_estimate(op, v)
    assert s["saved"] == 0
    assert s["saved_pct"] == 0.0


def test_zero_per_item_tokens_do_not_crash():
    op = _op(item_count=10_000, per_item_tokens_estimate=0)
    v = classify(op)
    assert v.lowerable is False  # nothing to save from a zero baseline
    s = savings_estimate(op, v)
    assert s["saved"] == 0
    assert s["saved_pct"] == 0.0


def test_deterministic_path_must_undercut_model_baseline():
    # 20 items x 800: baseline 16,000 vs synthesis 1,000 + 20-sample x 800 = 17,000.
    v = classify(_op(item_count=20))
    assert v.lowerable is False
    assert v.estimated_deterministic_tokens >= v.estimated_model_tokens


# ------------------------------------------------------------------- savings


def test_savings_are_labelled_estimated_never_measured():
    op = _op()
    v = classify(op)
    s = savings_estimate(op, v)
    assert s["estimated"] is True
    assert "ESTIMATED" in s["basis"]
    assert "not measured" in s["basis"]
    assert s["model_tokens"] == 20_000 * 800
    assert s["saved"] == s["model_tokens"] - s["deterministic_tokens"]
    assert s["saved_pct"] > 99.0  # O(n) -> O(1) is not a 20% saving


def test_unlowerable_operation_saves_nothing():
    op = _op(description="Decide whether the design is sound", item_count=50)
    v = classify(op)
    assert v.lowerable is False
    s = savings_estimate(op, v)
    assert s["saved"] == 0
    assert s["saved_pct"] == 0.0
    assert s["estimated"] is True


# ----------------------------------------------------------------- tripwire


def test_exception_rate_above_threshold_reverts_to_model():
    assert exception_rate_policy(1_000, 51) == REVERT   # 5.1% > 5%
    assert exception_rate_policy(100, 6) == REVERT
    assert exception_rate_policy(10, 10) == REVERT


def test_exception_rate_at_or_below_threshold_proceeds():
    assert exception_rate_policy(1_000, 50) == PROCEED  # exactly 5% still proceeds
    assert exception_rate_policy(1_000, 0) == PROCEED
    assert 50 / 1_000 == EXCEPTION_RATE_MAX


def test_zero_processed_is_no_evidence_not_a_crash():
    assert exception_rate_policy(0, 0) == PROCEED


def test_exception_rate_rejects_nonsense_counts():
    with pytest.raises(ValueError):
        exception_rate_policy(-1, 0)
    with pytest.raises(ValueError):
        exception_rate_policy(10, -1)
    with pytest.raises(ValueError):
        exception_rate_policy(5, 6)  # more exceptions than items processed


# ------------------------------------------------------------------- ledger


def test_record_lowering_writes_deterministic_row_with_named_baseline():
    log = AvoidanceLog()
    try:
        op = _op()
        v = classify(op)
        rid = record_lowering(log, "job-1", "task-9", op, v, actual_tokens=17_000)
        rows = log.for_job("job-1")
        assert len(rows) == 1
        row = rows[0]
        assert row.id == rid
        assert row.method is AvoidanceMethod.DETERMINISTIC
        assert row.task_id == "task-9"
        assert row.baseline_tokens == v.estimated_model_tokens
        assert row.actual_tokens == 17_000
        assert row.saved_tokens == v.estimated_model_tokens - 17_000
        # The ledger contract: baseline_source names HOW the baseline was counted.
        assert "items x" in row.baseline_source
        assert "estimate" in row.baseline_source
    finally:
        log.close()


def test_record_lowering_refuses_unlowered_operations():
    log = AvoidanceLog()
    try:
        op = _op(description="Decide whether this design is sound", item_count=10)
        v = classify(op)
        assert v.lowerable is False
        with pytest.raises(ValueError):
            record_lowering(log, "job-1", None, op, v, actual_tokens=0)
        assert log.for_job("job-1") == []
    finally:
        log.close()


# --------------------------------------------------------------- strategies


def test_every_kind_has_a_strategy_naming_a_real_tool():
    for kind in OperationKind:
        assert kind in STRATEGIES
        if kind is not OperationKind.REASON:
            assert any(t in STRATEGIES[kind] for t in REAL_TOOLS), STRATEGIES[kind]
    assert "model" in STRATEGIES[OperationKind.REASON]


# --------------------------------------------------------------- confidence


def test_high_item_count_is_a_stronger_lowering_signal():
    small = classify(_op(item_count=30, has_deterministic_structure=False))
    big = classify(_op(item_count=100_000, has_deterministic_structure=False))
    assert small.lowerable and big.lowerable
    assert big.confidence > small.confidence


def test_confidence_is_bounded_and_never_certain():
    for op in (
        _op(),
        _op(item_count=1),
        _op(description="Decide whether it is sound", item_count=10),
        _op(description="Do the thing", item_count=0),
    ):
        v = classify(op)
        assert 0.0 <= v.confidence <= 0.95 or v.kind is OperationKind.REASON
        assert v.confidence < 1.0


def test_structured_samples_raise_confidence():
    plain = _op(has_deterministic_structure=False)
    with_samples = _op(
        has_deterministic_structure=False,
        sample_items=['{"a": 1}', '{"b": 2}'],
    )
    assert classify(with_samples).confidence > classify(plain).confidence


def test_samples_look_structured_detection():
    assert samples_look_structured(['{"a": 1}', '{"b": 2}']) is True
    assert samples_look_structured(["a,b,c", "d,e,f"]) is True
    assert samples_look_structured(["plain prose here", "no structure at all"]) is False
    assert samples_look_structured([]) is False


# ----------------------------------------------------------------- contract


def test_verdict_shape():
    v = classify(_op())
    assert isinstance(v, LoweringVerdict)
    assert isinstance(v.kind, OperationKind)
    assert isinstance(v.strategy, str) and v.strategy
    assert isinstance(v.reason, str) and v.reason
    assert v.estimated_model_tokens >= 0
    assert v.estimated_deterministic_tokens >= 0
