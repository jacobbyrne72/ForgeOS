"""Verification ladder and merge gate tests.

The theme: every way a change could reach the base branch without being genuinely
verified. Passing tests, a missing scanner, zero tests run, self-review, absent
evidence — each is a route to a false green, and each has a test here.
"""

from __future__ import annotations

import shutil

import pytest

from forgeos.core.verify import (
    Finding,
    Gate,
    GateResult,
    GateStatus,
    MergeGate,
    detect_test_tampering,
    next_gate,
    run_gitleaks,
    run_lint,
    run_security,
    run_semgrep,
)


def _ok(gate=Gate.SECURITY) -> GateResult:
    return GateResult(gate=gate, status=GateStatus.PASS, command="scan", evidence="0 findings")


def _fail(gate=Gate.SECURITY, n=1) -> GateResult:
    return GateResult(
        gate=gate, status=GateStatus.FAIL, command="scan", evidence=f"{n} finding(s)",
        findings=[Finding(rule="r", path="p.py", line=1, severity="high", message="m")] * n,
    )


def _unavailable(gate=Gate.SECURITY) -> GateResult:
    return GateResult(gate=gate, status=GateStatus.UNAVAILABLE, evidence="tool not on PATH")


GREEN = dict(tests_passed=13, tests_failed=0, evidence="13 passed in 1.2s",
             commands_run=["python -m pytest -q"], reviewer_verdict="pass",
             reviewer_worker="reviewer.a", implementer_worker="coder.b")


# ------------------------------------------------------------- the ladder


def test_ladder_climbs_one_rung_at_a_time():
    assert next_gate(Gate.SYNTAX, True) is Gate.LINT
    assert next_gate(Gate.LINT, True) is Gate.DIRECT_TESTS
    assert next_gate(Gate.DIRECT_TESTS, True) is Gate.PACKAGE_TESTS
    assert next_gate(Gate.PACKAGE_TESTS, True) is Gate.SECURITY
    assert next_gate(Gate.SECURITY, True) is Gate.FULL_SUITE


def test_ladder_refuses_to_climb_past_a_failure():
    """Running the full suite after a lint failure wastes minutes to learn nothing."""
    assert next_gate(Gate.LINT, False) is None
    assert next_gate(Gate.DIRECT_TESTS, False) is None


def test_ladder_terminates_at_the_top():
    assert next_gate(Gate.FULL_SUITE, True) is None


def test_security_sits_before_the_full_suite():
    """Cheap security beats an expensive suite run on a patch that cannot merge."""
    assert Gate.SECURITY < Gate.FULL_SUITE


# ------------------------------------------------------- unavailable != pass


def test_missing_tool_reports_unavailable_not_pass(monkeypatch):
    """'We could not check' must never read as 'it is fine'."""
    monkeypatch.setattr(shutil, "which", lambda _: None)
    for res in (run_lint(["a.py"]), run_semgrep(["a.py"]), run_gitleaks(["a.py"])):
        assert res.status is GateStatus.UNAVAILABLE
        assert not res.passed
        assert res.blocks_merge


def test_unavailable_gate_blocks_the_merge():
    v = MergeGate().evaluate(gates=[_unavailable()], **GREEN)
    assert not v.allowed
    assert any("could not be checked" in r for r in v.reasons)


def test_no_paths_is_skipped_not_failed():
    """A change with nothing to scan is not a security failure."""
    assert run_lint([]).status is GateStatus.SKIPPED
    assert run_semgrep([]).status is GateStatus.SKIPPED


# ---------------------------------------------------------------- merge gate


def test_green_change_with_clean_security_is_allowed():
    v = MergeGate().evaluate(gates=[_ok()], **GREEN)
    assert v.allowed, v.reasons


def test_a7_functionally_correct_but_insecure_patch_is_blocked():
    """THE acceptance criterion: all tests green, security finding present.

    Published work shows agents produce patches that work and are still unsafe, so
    green tests alone must never be sufficient evidence for merge.
    """
    v = MergeGate().evaluate(gates=[_fail(n=2)], **GREEN)
    assert not v.allowed
    assert any("security failed" in r for r in v.reasons)
    assert v.blocking


def test_leaked_secret_blocks_the_merge():
    leak = GateResult(
        gate=Gate.SECURITY, status=GateStatus.FAIL, evidence="1 secret(s) detected",
        findings=[Finding(rule="generic-api-key", path="cfg.py", line=3, severity="high",
                          message="API key committed")],
    )
    assert not MergeGate().evaluate(gates=[leak], **GREEN).allowed


def test_failing_tests_block():
    args = {**GREEN, "tests_failed": 1}
    v = MergeGate().evaluate(gates=[_ok()], **args)
    assert not v.allowed
    assert any("failing" in r for r in v.reasons)


def test_zero_tests_passed_is_not_green():
    """'No tests ran' reported as success is the most common false green there is."""
    args = {**GREEN, "tests_passed": 0}
    v = MergeGate().evaluate(gates=[_ok()], **args)
    assert not v.allowed
    assert any("nothing was actually verified" in r for r in v.reasons)


def test_missing_evidence_blocks():
    args = {**GREEN, "evidence": "   "}
    assert not MergeGate().evaluate(gates=[_ok()], **args).allowed


def test_evidence_without_a_command_blocks():
    """Evidence has to trace to something that actually ran."""
    args = {**GREEN, "commands_run": []}
    v = MergeGate().evaluate(gates=[_ok()], **args)
    assert not v.allowed
    assert any("no command" in r for r in v.reasons)


def test_missing_independent_review_blocks():
    args = {**GREEN, "reviewer_verdict": None}
    assert not MergeGate().evaluate(gates=[_ok()], **args).allowed


def test_reviewer_rejection_blocks():
    args = {**GREEN, "reviewer_verdict": "fail"}
    assert not MergeGate().evaluate(gates=[_ok()], **args).allowed


def test_self_review_is_refused():
    """The model that wrote the bug shares the blind spot that produced it."""
    args = {**GREEN, "reviewer_worker": "same.worker", "implementer_worker": "same.worker"}
    v = MergeGate().evaluate(gates=[_ok()], **args)
    assert not v.allowed
    assert any("must not be the implementer" in r for r in v.reasons)


def test_all_blocking_reasons_are_reported_together():
    """One fix at a time would mean one full verification cycle per problem."""
    v = MergeGate().evaluate(
        gates=[_fail(), _unavailable(gate=Gate.LINT)],
        tests_passed=0, tests_failed=3, evidence="", commands_run=[],
        reviewer_verdict=None, reviewer_worker="w", implementer_worker="w",
    )
    assert not v.allowed
    assert len(v.reasons) >= 6


# ------------------------------------------------------ real scanners, real diff


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("gitleaks") is None, reason="gitleaks not installed")
def test_gitleaks_integration_returns_a_definite_verdict(tmp_path):
    """One genuinely live scan, proving the integration and parsing work.

    Deliberately NOT asserting a detection here. The obvious fake credentials to
    plant are the vendor documentation examples, and scanners allowlist those on
    purpose — a scanner that flagged `AKIAIOSFODNN7EXAMPLE` would be generating false
    positives on every AWS tutorial in existence. Planting a *convincing* fake secret
    to defeat the allowlist would be a worse idea than the test is worth.

    So: this test proves gitleaks runs, its JSON parses, and it yields a definite
    verdict. The FAIL path — a finding blocking the merge — is covered by the mocked
    tests above, which is the right place to assert gate behaviour anyway.
    """
    import subprocess

    (tmp_path / "cfg.py").write_text(
        'AWS_ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"  # vendor doc example, allowlisted\n',
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x"],
        cwd=tmp_path, capture_output=True,
    )

    res = run_gitleaks(cwd=str(tmp_path))
    # A definite verdict, never a crash and never an empty/unknown state.
    assert res.status in (GateStatus.PASS, GateStatus.FAIL, GateStatus.UNAVAILABLE)
    assert "gitleaks" in res.command and "detect" in res.command
    if res.status is GateStatus.FAIL:
        assert res.findings
        assert MergeGate().evaluate(gates=[res], **GREEN).allowed is False


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("gitleaks") is None, reason="gitleaks not installed")
def test_gitleaks_passes_a_clean_repo(tmp_path):
    import subprocess

    (tmp_path / "clean.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x"],
        cwd=tmp_path, capture_output=True,
    )
    res = run_gitleaks(cwd=str(tmp_path))
    assert res.status in (GateStatus.PASS, GateStatus.UNAVAILABLE)


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff not installed")
def test_ruff_gate_runs_for_real(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("import os\n", encoding="utf-8")  # unused import
    res = run_lint([str(bad)])
    assert res.status in (GateStatus.PASS, GateStatus.FAIL)
    assert "ruff" in res.command and "check" in res.command


# ------------------------------------- the gate judges the change, not the past


def _gitleaks_returning(payload: str, monkeypatch):
    """Substitute the subprocess so the parsing and scoping can be tested without
    a real scanner or a real repository."""
    import forgeos.core.verify as v

    captured: dict = {}

    class _Result:
        stdout = payload
        stderr = ""
        returncode = 1

    def fake_run(cmd, *, cwd=None, timeout=None):
        captured["cmd"] = cmd
        return _Result()

    monkeypatch.setattr(v, "_run", fake_run)
    monkeypatch.setattr(v, "_tool_available", lambda _: True)
    monkeypatch.setattr(v, "resolve_tool", lambda name: f"/usr/bin/{name}")
    return captured


_LEAK = (
    '[{"RuleID":"generic-api-key","File":"%s","StartLine":9,'
    '"Description":"Detected a Generic API Key"}]'
)


def test_gitleaks_scans_the_working_tree_not_the_whole_history(monkeypatch):
    """History scanning makes the verdict independent of the change under review,
    and lets one old finding block every future merge with no way to clear it."""
    captured = _gitleaks_returning("[]", monkeypatch)
    run_gitleaks(["a.py"], cwd="/repo")
    assert "--no-git" in captured["cmd"]
    assert captured["cmd"][-1] == "a.py"


def test_a_finding_on_a_changed_path_still_fails_the_gate(monkeypatch):
    _gitleaks_returning(_LEAK % "src/app.py", monkeypatch)
    res = run_gitleaks(["src/app.py"], cwd="/repo")
    assert res.status is GateStatus.FAIL
    assert res.findings


def test_a_finding_outside_the_changed_paths_does_not_fail_the_gate(monkeypatch):
    """Real, but not this change's to answer for. A gate that cannot be satisfied
    is one people learn to bypass."""
    _gitleaks_returning(_LEAK % "tests/fixtures/legacy.py", monkeypatch)
    res = run_gitleaks(["src/app.py"], cwd="/repo")
    assert res.status is GateStatus.PASS


def test_a_filtered_finding_is_still_counted_out_loud(monkeypatch):
    """Silently dropping it would make the gate look cleaner than the repo is."""
    _gitleaks_returning(_LEAK % "other/thing.py", monkeypatch)
    res = run_gitleaks(["src/app.py"], cwd="/repo")
    assert "1 outside the changed paths" in res.evidence


def test_findings_under_a_changed_directory_are_in_scope(monkeypatch):
    _gitleaks_returning(_LEAK % "src/deep/nested/app.py", monkeypatch)
    assert run_gitleaks(["src/"], cwd="/repo").status is GateStatus.FAIL


def test_windows_separators_do_not_smuggle_a_finding_past_the_scope_check(monkeypatch):
    """gitleaks reports forward slashes; a caller on Windows may not. A finding
    that slipped through on a separator mismatch would be a silent miss."""
    _gitleaks_returning(_LEAK % "src/app.py", monkeypatch)
    assert run_gitleaks(["src\\app.py"], cwd="/repo").status is GateStatus.FAIL


def test_no_paths_means_no_scoping_rather_than_nothing_in_scope(monkeypatch):
    """The dangerous default. 'The caller did not say what changed' must never
    quietly become 'nothing is in scope' — that turns the filter into a mute."""
    _gitleaks_returning(_LEAK % "anywhere.py", monkeypatch)
    assert run_gitleaks(cwd="/repo").status is GateStatus.FAIL


def test_hives_own_source_is_clean_of_secrets():
    """forgeos scans other people's code; it should survive its own scanner.

    Scoped to the directories forgeos actually authors. `--no-git` walks the
    filesystem rather than the index, so an unscoped scan also covers untracked
    and gitignored trees — `vendor/` here is full of third-party code with
    documentation examples that trip `generic-api-key`, and those are neither
    forgeos's secrets nor forgeos's to fix. That is exactly why the Forge always passes
    `files_touched` to this gate rather than letting it scan freely.

    Skipped when gitleaks is absent — an unavailable scanner is reported, never
    treated as a pass, and that rule applies to forgeos's own source too.
    """
    if shutil.which("gitleaks") is None:
        pytest.skip("gitleaks not installed")
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    ours = [d for d in ("forgeos", "tests", "hooks", "tools") if (repo / d).exists()]
    res = run_gitleaks(ours, cwd=str(repo))
    assert res.status is not GateStatus.FAIL, f"forgeos's own source leaks: {res.findings}"


def test_combined_security_fails_if_either_scanner_fails(monkeypatch):
    import forgeos.core.verify as v

    monkeypatch.setattr(v, "run_semgrep", lambda p, cwd=None: _ok())
    monkeypatch.setattr(v, "run_gitleaks", lambda p=None, cwd=None: _fail())
    assert run_security(["a.py"]).status is GateStatus.FAIL


def test_combined_security_is_unavailable_if_either_tool_is_missing(monkeypatch):
    import forgeos.core.verify as v

    monkeypatch.setattr(v, "run_semgrep", lambda p, cwd=None: _ok())
    monkeypatch.setattr(v, "run_gitleaks", lambda p=None, cwd=None: _unavailable())
    assert run_security(["a.py"]).status is GateStatus.UNAVAILABLE


# --------------------------------- the gate must enforce its own contract
# Each of these was proven merge-able by execution during review. They are the
# difference between a gate and a ritual.


def test_no_gates_at_all_cannot_merge():
    """`gates=[]` merged clean: the gate enforced its headline contract only when a
    caller volunteered the evidence against itself."""
    v = MergeGate().evaluate(gates=[], **GREEN)
    assert not v.allowed
    assert any("no security gate" in r for r in v.reasons)


@pytest.mark.parametrize("verdict", ["FAIL", "failed", "reject", "REJECTED", "no", ""])
def test_only_the_exact_token_pass_counts_as_approval(verdict):
    """Comparing against the literal "fail" meant every other spelling of rejection
    read as approval."""
    v = MergeGate().evaluate(gates=[_ok()], **{**GREEN, "reviewer_verdict": verdict})
    assert not v.allowed, f"{verdict!r} was accepted as approval"


def test_a_derived_reviewer_id_is_refused():
    """'reviewer::a' reviewing 'a' is one worker wearing a label."""
    v = MergeGate().evaluate(gates=[_ok()], **{**GREEN, "reviewer_worker": "reviewer::a",
                                               "implementer_worker": "a"})
    assert not v.allowed
    assert any("derived from implementer" in r for r in v.reasons)


def test_missing_reviewer_identity_is_refused_not_skipped():
    """A truthiness guard meant empty ids skipped the independence check entirely."""
    v = MergeGate().evaluate(gates=[_ok()], **{**GREEN, "reviewer_worker": "",
                                               "implementer_worker": ""})
    assert not v.allowed
    assert any("independence unverifiable" in r for r in v.reasons)


def test_a_genuinely_independent_clean_change_still_merges():
    """The gate must not become unpassable — that would just get it bypassed."""
    assert MergeGate().evaluate(gates=[_ok()], **GREEN).allowed


# ------------------------------------------------- test-tampering detection
# Per SpecBench and the RLVR reward-hacking survey (docs/research/verification-
# economy.md): an agent under a pass/fail loop can make tests go green by
# editing the test instead of the code. Both tiers of the heuristic, and the
# wiring into MergeGate, are covered here.


def test_detect_test_tampering_flags_a_named_pair():
    reasons = detect_test_tampering(["tests/test_verify.py", "forgeos/core/verify.py"])
    assert len(reasons) == 1
    assert "test-tampering:" in reasons[0]
    assert "tests/test_verify.py" in reasons[0] and "forgeos/core/verify.py" in reasons[0]


def test_detect_test_tampering_matches_the_suffix_convention_too():
    reasons = detect_test_tampering(["verify_test.py", "verify.py"])
    assert len(reasons) == 1
    assert "test-tampering:" in reasons[0]


def test_detect_test_tampering_flags_the_general_case_as_a_warn():
    """No name match, but a test and an unrelated source moved together."""
    reasons = detect_test_tampering(["tests/test_verify.py", "forgeos/core/router.py"])
    assert len(reasons) == 1
    assert "WARN" in reasons[0]


def test_detect_test_tampering_reports_both_tiers_separately():
    reasons = detect_test_tampering(
        ["tests/test_verify.py", "forgeos/core/verify.py",  # named pair
         "tests/test_router.py"]  # no matching source touched -> general case
    )
    assert len(reasons) == 2
    assert any("test-tampering:" in r and "WARN" not in r for r in reasons)
    assert any("WARN" in r for r in reasons)


def test_detect_test_tampering_is_silent_when_only_the_test_changed():
    assert detect_test_tampering(["tests/test_verify.py"]) == []


def test_detect_test_tampering_is_silent_when_only_source_changed():
    assert detect_test_tampering(["forgeos/core/verify.py"]) == []


def test_detect_test_tampering_is_silent_on_an_empty_changeset():
    assert detect_test_tampering([]) == []


def test_detect_test_tampering_ignores_non_python_source():
    """A test touched alongside a doc or config file is not a tampering signal."""
    assert detect_test_tampering(["tests/test_verify.py", "README.md"]) == []


def test_detect_test_tampering_handles_windows_separators():
    reasons = detect_test_tampering(["tests\\test_verify.py", "forgeos\\core\\verify.py"])
    assert len(reasons) == 1
    assert "test-tampering:" in reasons[0]


def test_merge_gate_without_files_touched_is_backward_compatible():
    """None (the default) is how every pre-existing caller behaves — this must
    not start blocking forge.py's own call site or any of the 30+ tests above
    that never mention files_touched at all."""
    assert MergeGate().evaluate(gates=[_ok()], **GREEN).allowed


def test_merge_gate_blocks_a_named_test_source_pair():
    args = {**GREEN, "files_touched": ["tests/test_verify.py", "forgeos/core/verify.py"]}
    v = MergeGate().evaluate(gates=[_ok()], **args)
    assert not v.allowed
    assert any("test-tampering:" in r for r in v.reasons)


def test_merge_gate_blocks_the_general_case_too():
    args = {**GREEN, "files_touched": ["tests/test_verify.py", "forgeos/core/router.py"]}
    v = MergeGate().evaluate(gates=[_ok()], **args)
    assert not v.allowed
    assert any("WARN" in r for r in v.reasons)


def test_tamper_reviewed_unblocks_but_still_surfaces_as_a_warning():
    """A human acknowledgment is an override, not a silencer — the coincidence
    is still worth recording even once someone has signed off on it."""
    args = {
        **GREEN,
        "files_touched": ["tests/test_verify.py", "forgeos/core/verify.py"],
        "tamper_reviewed": True,
    }
    v = MergeGate().evaluate(gates=[_ok()], **args)
    assert v.allowed, v.reasons
    assert any("test-tampering:" in w for w in v.warnings)
    assert any("tamper_reviewed=True" in w for w in v.warnings)


def test_files_touched_empty_list_is_honoured_not_skipped():
    """An explicit empty list means the caller checked and found nothing — that
    is different from not being able to check at all (None), and it must not
    be treated as a tampering signal either way."""
    args = {**GREEN, "files_touched": []}
    assert MergeGate().evaluate(gates=[_ok()], **args).allowed


def test_files_touched_with_no_tampering_still_merges():
    args = {**GREEN, "files_touched": ["forgeos/core/verify.py", "forgeos/core/router.py"]}
    assert MergeGate().evaluate(gates=[_ok()], **args).allowed


# --------------------------------------------- reviewer provider-family WARN
# Panickssery et al. (NeurIPS 2024, arXiv 2404.13076): LLM evaluators recognize
# and favor their own generations. A different worker id from the same model
# family is weaker evidence of independence than it looks.


def test_same_reviewer_family_warns_but_does_not_block():
    args = {**GREEN, "reviewer_family": "anthropic", "implementer_family": "anthropic"}
    v = MergeGate().evaluate(gates=[_ok()], **args)
    assert v.allowed, v.reasons
    assert any("anthropic" in w and "self-preference" in w for w in v.warnings)


def test_different_reviewer_family_has_no_warning():
    args = {**GREEN, "reviewer_family": "anthropic", "implementer_family": "openai"}
    v = MergeGate().evaluate(gates=[_ok()], **args)
    assert v.allowed
    assert v.warnings == []


def test_reviewer_family_omitted_by_default_has_no_warning():
    """Backward compatible: no existing caller passes these, and their absence
    must not be treated as a match."""
    v = MergeGate().evaluate(gates=[_ok()], **GREEN)
    assert v.warnings == []


def test_reviewer_family_warning_is_independent_of_the_worker_id_block():
    """Same-family review is a WARN even when worker-id independence already
    passed cleanly — the two checks answer different questions."""
    args = {
        **GREEN,
        "reviewer_worker": "reviewer.a", "implementer_worker": "coder.b",
        "reviewer_family": "anthropic", "implementer_family": "anthropic",
    }
    v = MergeGate().evaluate(gates=[_ok()], **args)
    assert v.allowed
    assert v.warnings != []
