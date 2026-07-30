"""Context capsule tests: ranked assembly, budget enforcement, policy refusal.

No LLM calls, no network, no real repo scanning -- content and sizes are
passed explicitly, exactly like test_economy.py does for preflight/avoidance.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from hive.economy import AvoidanceLog, AvoidanceMethod
from hive.economy.capsule import (
    Capsule,
    CapsuleBuilder,
    ContextRef,
    ExpansionRequest,
    RefKind,
    make_ref,
)


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _ref(kind: RefKind, ref: str, *, reason: str = "relevant to the objective", tokens: int = 10) -> ContextRef:
    return ContextRef(kind=kind, ref=ref, sha256=_sha(ref), reason=reason, token_estimate=tokens)


# ------------------------------------------------------------------ make_ref


def test_make_ref_hashes_content_and_counts_tokens():
    ref = make_ref(RefKind.FILE_SLICE, "file_slice://a.py#L1-L2", "def f():\n    return 1\n", "primary target")
    assert ref.sha256 == hashlib.sha256(b"def f():\n    return 1\n").hexdigest()
    assert ref.token_estimate > 0
    assert ref.kind is RefKind.FILE_SLICE


def test_make_ref_reuses_preflight_counting_for_known_model():
    ref = make_ref(RefKind.SYMBOL, "symbol://a.py#f", "x" * 40, "reason", model="gpt-4o")
    # gpt-4o has a real tiktoken encoding; 40 identical chars is well under 40 tokens.
    assert 0 < ref.token_estimate < 40


# --------------------------------------------------------------- ContextRef


def test_context_ref_rejects_blank_ref_and_reason():
    with pytest.raises(ValueError):
        ContextRef(kind=RefKind.SYMBOL, ref="  ", sha256=_sha("x"), reason="r", token_estimate=1)
    with pytest.raises(ValueError):
        ContextRef(kind=RefKind.SYMBOL, ref="symbol://a.py#f", sha256=_sha("x"), reason=" ", token_estimate=1)


def test_context_ref_rejects_malformed_sha256():
    with pytest.raises(ValueError):
        ContextRef(kind=RefKind.SYMBOL, ref="symbol://a.py#f", sha256="not-a-hash", reason="r", token_estimate=1)


def test_context_ref_rejects_negative_token_estimate():
    with pytest.raises(ValueError):
        ContextRef(kind=RefKind.SYMBOL, ref="symbol://a.py#f", sha256=_sha("x"), reason="r", token_estimate=-1)


# ----------------------------------------------------------- builder.add()


def test_add_accepts_ref_within_budget():
    b = CapsuleBuilder(budget=100)
    ref = _ref(RefKind.SYMBOL, "symbol://a.py#f", tokens=40)
    assert b.add(ref) is True
    assert b.total_tokens == 40
    assert ref in b.items


def test_add_refuses_over_budget_and_records_excluded_with_reason():
    b = CapsuleBuilder(budget=50)
    fits = _ref(RefKind.SYMBOL, "symbol://a.py#f", tokens=40)
    overflows = _ref(RefKind.SYMBOL, "symbol://b.py#g", tokens=20)

    assert b.add(fits) is True
    assert b.add(overflows) is False

    assert overflows.ref in b.excluded
    assert "budget" in b.excluded[overflows.ref]
    assert b.total_tokens == 40  # the budget was never exceeded


def test_budget_is_never_exceeded_across_many_refs():
    b = CapsuleBuilder(budget=100)
    refs = [_ref(RefKind.SYMBOL, f"symbol://f{i}.py#x", tokens=30) for i in range(10)]
    for r in refs:
        b.add(r)
    assert b.total_tokens <= 100
    assert len(b.items) == 3  # 30*3 = 90 <= 100 < 30*4
    assert len(b.excluded) == 7


def test_builder_budget_must_be_positive():
    with pytest.raises(ValueError):
        CapsuleBuilder(budget=0)
    with pytest.raises(ValueError):
        CapsuleBuilder(budget=-5)


# ------------------------------------------------------------- policy gate


def test_credential_path_refused_even_when_explicitly_requested():
    b = CapsuleBuilder(budget=10_000)
    ref = _ref(RefKind.FILE_SLICE, "file_slice://config/.env#L1-L3", tokens=5)

    assert b.add(ref) is False
    assert ref.ref in b.excluded
    assert "policy denied" in b.excluded[ref.ref]
    assert ref not in b.items


def test_vendored_and_lockfile_paths_are_also_refused():
    b = CapsuleBuilder(budget=10_000)
    vendored = _ref(RefKind.FILE_SLICE, "file_slice://vendor/thirdparty.py#L1-L2", tokens=5)
    lockfile = _ref(RefKind.FILE_SLICE, "file_slice://package-lock.json#L1-L2", tokens=5)

    assert b.add(vendored) is False
    assert b.add(lockfile) is False
    assert "policy denied" in b.excluded[vendored.ref]
    assert "policy denied" in b.excluded[lockfile.ref]


def test_artifact_and_card_refs_have_no_path_and_skip_policy():
    b = CapsuleBuilder(budget=10_000)
    artifact = _ref(RefKind.ARTIFACT, f"artifact://sha256/{_sha('blob')}", tokens=5)
    card = _ref(RefKind.CARD, "card://kb/some-finding", tokens=5)

    assert b.add(artifact) is True
    assert b.add(card) is True


def test_bare_artifact_diff_ref_skips_policy_even_though_diff_is_path_capable():
    b = CapsuleBuilder(budget=10_000)
    ref = _ref(RefKind.DIFF, f"diff://sha256/{_sha('patch')}", tokens=5)
    assert b.add(ref) is True


# ------------------------------------------------------------ expansion_request


def test_expansion_request_is_recorded():
    b = CapsuleBuilder(budget=10)
    req = b.expansion_request(RefKind.SYMBOL, "need the caller of validate() for impact analysis")
    assert isinstance(req, ExpansionRequest)
    assert b.expansion_requests == [req]
    assert req.kind is RefKind.SYMBOL
    assert "impact analysis" in req.detail


def test_expansion_request_rejects_blank_detail():
    b = CapsuleBuilder(budget=10)
    with pytest.raises(ValueError):
        b.expansion_request(RefKind.TEST, "   ")


# ------------------------------------------------------------------- build()


def test_ranked_build_orders_primary_symbols_before_history():
    primary = [_ref(RefKind.SYMBOL, "symbol://a.py#f", reason="primary", tokens=10)]
    tests_ = [_ref(RefKind.TEST, "test://tests/test_a.py::test_f", reason="closest test", tokens=10)]
    callers = [_ref(RefKind.SYMBOL, "symbol://b.py#caller", reason="caller", tokens=10)]
    history = [_ref(RefKind.DIFF, "diff://a.py", reason="recent failure", tokens=10)]

    capsule = CapsuleBuilder.build(
        objective="fix the thing",
        acceptance=["pytest tests/test_a.py passes"],
        primary_symbols=primary,
        callers=callers,
        related_tests=tests_,
        recent_failures=history,
        write_scope=["a.py"],
        budget=1_000,
    )

    refs_in_order = [i.ref for i in capsule.items]
    assert refs_in_order == [primary[0].ref, tests_[0].ref, callers[0].ref, history[0].ref]
    assert capsule.items.index(primary[0]) < capsule.items.index(history[0])


def test_ranked_build_stops_at_budget_and_drops_never_vanish():
    primary = [_ref(RefKind.SYMBOL, "symbol://a.py#f", tokens=10)]
    tests_ = [_ref(RefKind.TEST, "test://tests/test_a.py::test_f", tokens=10)]
    history = [_ref(RefKind.DIFF, "diff://a.py", reason="prior failed attempt", tokens=10)]

    capsule = CapsuleBuilder.build(
        objective="fix the thing",
        acceptance=["tests pass"],
        primary_symbols=primary,
        related_tests=tests_,
        recent_failures=history,
        write_scope=["a.py"],
        budget=15,  # only the first (10-token) ref fits
    )

    assert capsule.total_tokens <= 15
    assert [i.ref for i in capsule.items] == [primary[0].ref]
    assert tests_[0].ref in capsule.excluded
    assert history[0].ref in capsule.excluded
    assert capsule.excluded[history[0].ref]  # never an empty/absent reason


def test_build_end_to_end_filters_policy_denied_refs_into_capsule_excluded():
    primary = [_ref(RefKind.SYMBOL, "symbol://a.py#f", tokens=10)]
    denied = [_ref(RefKind.FILE_SLICE, "file_slice://.env#L1-L1", tokens=5)]

    capsule = CapsuleBuilder.build(
        objective="fix the thing",
        acceptance=["tests pass"],
        primary_symbols=primary,
        callers=denied,
        write_scope=["a.py"],
        budget=1_000,
    )

    assert denied[0].ref not in [i.ref for i in capsule.items]
    assert "policy denied" in capsule.excluded[denied[0].ref]


# --------------------------------------------------------- Capsule / manifest


def test_read_scope_is_derived_and_excludes_non_path_kinds():
    primary = [_ref(RefKind.SYMBOL, "symbol://a.py#f", tokens=10)]
    tests_ = [_ref(RefKind.TEST, "test://tests/test_a.py::test_f", tokens=10)]
    artifact = [_ref(RefKind.ARTIFACT, f"artifact://sha256/{_sha('x')}", tokens=10)]

    capsule = CapsuleBuilder.build(
        objective="obj",
        acceptance=["a"],
        primary_symbols=primary + artifact,
        related_tests=tests_,
        write_scope=["a.py"],
        budget=1_000,
    )

    assert capsule.read_scope == sorted({"a.py", "tests/test_a.py"})


def test_total_tokens_matches_sum_of_items():
    refs = [_ref(RefKind.SYMBOL, f"symbol://f{i}.py#x", tokens=7) for i in range(4)]
    capsule = CapsuleBuilder.build(objective="o", acceptance=["a"], primary_symbols=refs, budget=1_000)
    assert capsule.total_tokens == sum(i.token_estimate for i in capsule.items) == 28


def test_total_tokens_is_self_correcting_against_a_stale_value():
    ref = _ref(RefKind.SYMBOL, "symbol://a.py#f", tokens=9)
    capsule = Capsule(objective="o", items=[ref], total_tokens=9999)
    assert capsule.total_tokens == 9  # recomputed, not trusted from the constructor call


def test_empty_capsule_is_valid_and_does_not_divide_by_zero():
    capsule = Capsule(objective="do nothing yet")
    assert capsule.items == []
    assert capsule.total_tokens == 0
    assert capsule.excluded == {}
    manifest = capsule.manifest()
    assert manifest["total_tokens"] == 0
    assert manifest["items"] == []
    json.dumps(manifest)  # must not raise


def test_manifest_is_compact_and_json_serializable():
    ref = _ref(RefKind.SYMBOL, "symbol://a.py#f", reason="primary target", tokens=12)
    capsule = CapsuleBuilder.build(
        objective="fix bug",
        acceptance=["pytest tests/test_a.py -q passes"],
        primary_symbols=[ref],
        write_scope=["a.py"],
        budget=1_000,
    )
    manifest = capsule.manifest()
    raw = json.dumps(manifest)  # must not raise
    assert json.loads(raw) == manifest

    assert manifest["objective"] == "fix bug"
    assert manifest["items"] == [
        {
            "kind": "symbol",
            "ref": "symbol://a.py#f",
            "sha256": ref.sha256,
            "reason": "primary target",
            "tokens": 12,
        }
    ]
    assert "summary" not in manifest  # compact: no field the worker didn't need


def test_capsule_rejects_blank_objective():
    with pytest.raises(ValueError):
        Capsule(objective="   ")


# ------------------------------------------------------------- record_savings


@pytest.fixture()
def avd():
    log = AvoidanceLog(":memory:")
    yield log
    log.close()


def test_record_savings_writes_a_pack_row(avd):
    ref = _ref(RefKind.SYMBOL, "symbol://a.py#f", tokens=200)
    capsule = CapsuleBuilder.build(objective="o", acceptance=["a"], primary_symbols=[ref], budget=1_000)

    rid = capsule.record_savings(
        avd, "job1", "task1", baseline_tokens=10_000, baseline_source="tiktoken count of the full repo file"
    )

    rows = avd.for_job("job1")
    assert [r.id for r in rows] == [rid]
    row = rows[0]
    assert row.method is AvoidanceMethod.PACK
    assert row.baseline_tokens == 10_000
    assert row.actual_tokens == capsule.total_tokens == 200
    assert row.saved_tokens == 9_800


def test_record_savings_default_baseline_source_is_not_blank(avd):
    capsule = Capsule(objective="o")
    rid = capsule.record_savings(avd, "job2", None, baseline_tokens=500)
    row = avd.for_job("job2")[0]
    assert rid == row.id
    assert row.baseline_source.strip() != ""
    assert row.actual_tokens == 0
    assert row.saved_tokens == 500
