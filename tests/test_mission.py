"""Tests for the mission compiler (hive.core.mission) and operating modes
(hive.modes). Pure pytest: no network, no model calls, fake repo layouts via
tmp_path.
"""

from __future__ import annotations

import json

import pytest

from hive.contracts import Budget
from hive.core.mission import MissionContract, RepoRef, compile_mission, detect_acceptance_commands
from hive.modes import (
    BLITZ_PROFILE,
    DEEP_PROFILE,
    PROFILES,
    Mode,
    ModeProfile,
    profile_for,
    suggest_mode,
)

# ---------------------------------------------------------------------------
# acceptance-command detection
# ---------------------------------------------------------------------------


def test_pyproject_repo_yields_pytest_command(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = \"x\"\n", encoding="utf-8")

    contract = compile_mission("ship the fix", str(tmp_path))

    assert "python -m pytest -q" in contract.acceptance_commands
    assert contract.acceptance_unverifiable is False


def test_package_json_with_test_script_yields_npm_test(tmp_path):
    (tmp_path / "package.json").write_text(
        json.dumps({"scripts": {"test": "jest"}}), encoding="utf-8"
    )

    contract = compile_mission("ship the fix", str(tmp_path))

    assert "npm test" in contract.acceptance_commands
    assert contract.acceptance_unverifiable is False


def test_package_json_default_no_test_script_is_not_detected(tmp_path):
    # `npm init -y` default — a real script that isn't actually a test suite.
    (tmp_path / "package.json").write_text(
        json.dumps({"scripts": {"test": 'echo "Error: no test specified" && exit 1'}}),
        encoding="utf-8",
    )

    commands = detect_acceptance_commands(tmp_path)

    assert "npm test" not in commands


def test_bare_directory_is_acceptance_unverifiable(tmp_path):
    contract = compile_mission("do something", str(tmp_path))

    assert contract.acceptance_commands == []
    assert contract.acceptance_unverifiable is True


def test_ruff_toml_yields_lint_command_without_pytest(tmp_path):
    (tmp_path / "ruff.toml").write_text("line-length = 100\n", encoding="utf-8")

    commands = detect_acceptance_commands(tmp_path)

    assert commands == ["ruff check ."]


def test_pyproject_with_ruff_section_yields_both_commands(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = \"x\"\n\n[tool.ruff]\nline-length = 100\n", encoding="utf-8"
    )

    commands = detect_acceptance_commands(tmp_path)

    assert commands == ["python -m pytest -q", "ruff check ."]


def test_cargo_toml_yields_cargo_test(tmp_path):
    (tmp_path / "Cargo.toml").write_text("[package]\nname = \"x\"\n", encoding="utf-8")

    commands = detect_acceptance_commands(tmp_path)

    assert commands == ["cargo test"]


def test_polyglot_repo_yields_every_detected_command(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = \"x\"\n", encoding="utf-8")
    (tmp_path / "package.json").write_text(
        json.dumps({"scripts": {"test": "jest"}}), encoding="utf-8"
    )

    commands = detect_acceptance_commands(tmp_path)

    assert "python -m pytest -q" in commands
    assert "npm test" in commands


def test_oversized_manifest_is_never_read_for_content(tmp_path):
    # Existence still yields the pytest command (filename-based), but content
    # detection (ruff section) must not fire — the file is never opened
    # because it exceeds the manifest size guard. This is the "never a large
    # file" rule from AGENTS.md rule 3, applied to manifest detection.
    huge = "[project]\nname = \"x\"\n# padding\n" + ("x" * 200_000) + "\n[tool.ruff]\n"
    (tmp_path / "pyproject.toml").write_text(huge, encoding="utf-8")

    commands = detect_acceptance_commands(tmp_path)

    assert commands == ["python -m pytest -q"]
    assert "ruff check ." not in commands


def test_compile_mission_overrides_are_honored(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = \"x\"\n", encoding="utf-8")

    contract = compile_mission(
        "ship the fix",
        str(tmp_path),
        acceptance_commands=["make test"],
        acceptance_unverifiable=False,
        budget=Budget(max_usd=5.0, max_seconds=300, max_iterations=4),
    )

    assert contract.acceptance_commands == ["make test"]
    assert contract.budget.max_usd == 5.0


def test_compile_mission_default_repositories_point_at_cwd(tmp_path):
    contract = compile_mission("do something", str(tmp_path))

    assert contract.repositories == [RepoRef(path=str(tmp_path))]


# ---------------------------------------------------------------------------
# immutability / revise()
# ---------------------------------------------------------------------------


def test_revise_increments_revision_and_records_reason_leaves_original_untouched():
    original = MissionContract(objective="build the thing", acceptance_commands=["pytest"])

    revised = original.revise("acceptance criteria were wrong", acceptance_commands=["pytest -q"])

    assert revised.revision == original.revision + 1
    assert revised.revision_reason == "acceptance criteria were wrong"
    assert revised.parent_id == original.id
    assert revised.acceptance_commands == ["pytest -q"]
    # original is completely untouched
    assert original.revision == 0
    assert original.revision_reason == ""
    assert original.acceptance_commands == ["pytest"]
    assert revised.id != original.id


def test_revise_requires_non_blank_reason():
    original = MissionContract(objective="build the thing")

    with pytest.raises(ValueError):
        original.revise("   ")


def test_revise_works_even_after_contract_has_started():
    original = MissionContract(objective="build the thing")
    original.mark_started()

    revised = original.revise("scope grew")

    assert revised.revision == 1
    assert revised.started is False


def test_mutating_started_contract_raises():
    contract = MissionContract(objective="build the thing")
    contract.mark_started()

    with pytest.raises(RuntimeError):
        contract.acceptance_commands = ["rm -rf /"]

    with pytest.raises(RuntimeError):
        contract.objective = "something else entirely"


def test_mutating_unstarted_contract_is_allowed():
    contract = MissionContract(objective="build the thing")

    contract.acceptance_commands = ["pytest -q"]  # should not raise

    assert contract.acceptance_commands == ["pytest -q"]


def test_mark_started_is_idempotent():
    contract = MissionContract(objective="build the thing")
    contract.mark_started()
    contract.mark_started()

    assert contract.started is True


def test_objective_must_not_be_blank():
    with pytest.raises(Exception):
        MissionContract(objective="   ")


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------


def test_suggest_mode_security_capability_is_deep_even_at_low_risk():
    assert suggest_mode(["security"], risk="low") == Mode.DEEP


def test_suggest_mode_architecture_capability_is_deep_even_at_low_risk():
    assert suggest_mode(["architecture"], risk="low") == Mode.DEEP


def test_suggest_mode_auth_and_migration_keywords_are_deep():
    assert suggest_mode(["auth_flow"], risk="low") == Mode.DEEP
    assert suggest_mode(["database_migration"], risk="low") == Mode.DEEP


def test_suggest_mode_high_risk_alone_is_deep():
    assert suggest_mode(["mechanical"], risk="high") == Mode.DEEP


def test_suggest_mode_trivial_low_risk_is_blitz():
    assert suggest_mode(["trivial_edit"], risk="low") == Mode.BLITZ
    assert suggest_mode(["mechanical"], risk="medium") == Mode.BLITZ


def test_profile_for_returns_matching_profile():
    assert profile_for(Mode.BLITZ) is BLITZ_PROFILE
    assert profile_for(Mode.DEEP) is DEEP_PROFILE
    assert PROFILES[Mode.BLITZ] is BLITZ_PROFILE


def test_blitz_and_deep_have_identical_budget_and_approval_semantics():
    # Neither profile carries any field that could widen a budget ceiling or
    # bypass the human-approval gate for irreversible actions — the only
    # dial modes are allowed to turn is verification breadth.
    forbidden = {
        "budget",
        "max_usd",
        "max_seconds",
        "max_iterations",
        "approval",
        "requires_approval",
        "skip_approval",
        "bypass_approval",
    }
    assert forbidden.isdisjoint(set(ModeProfile.model_fields))

    # Both modes leave a mission's budget completely untouched — mode
    # selection is orthogonal to spend governance.
    mission_blitz = MissionContract(objective="trivial fix", budget=Budget(max_usd=3.0))
    mission_deep = MissionContract(objective="auth overhaul", budget=Budget(max_usd=3.0))
    suggest_mode(["trivial_edit"], risk="low")
    suggest_mode(["security"], risk="low")
    assert mission_blitz.budget.max_usd == 3.0
    assert mission_deep.budget.max_usd == 3.0
    assert mission_blitz.budget == mission_deep.budget


def test_blitz_narrows_verification_breadth_not_budget():
    assert BLITZ_PROFILE.targeted_tests_only_during_iteration is True
    assert DEEP_PROFILE.targeted_tests_only_during_iteration is False
    assert BLITZ_PROFILE.full_review_count <= DEEP_PROFILE.full_review_count
    assert BLITZ_PROFILE.require_independent_review is False
    assert DEEP_PROFILE.require_independent_review is True
    assert BLITZ_PROFILE.require_security_checks is False
    assert DEEP_PROFILE.require_security_checks is True


def test_mode_profile_rejects_non_positive_counts():
    with pytest.raises(Exception):
        ModeProfile(
            parallel_readers=0,
            initial_tier=BLITZ_PROFILE.initial_tier,
            reasoning_effort=BLITZ_PROFILE.reasoning_effort,
            targeted_tests_only_during_iteration=True,
            full_review_count=1,
            max_manager_interventions=1,
            parallel_hypotheses=1,
            require_independent_review=False,
            require_security_checks=False,
        )
