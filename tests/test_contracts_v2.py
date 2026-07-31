"""ForgeOS v2 contracts — the properties, not the field list.

Each test here corresponds to a design law from the v2 blueprint, and each law
exists because its absence is a known failure mode:

- A contract that can be edited in place cannot govern the run it was written
  for, so revision is an event with a number.
- A criterion with no proof route can never be discharged, so the system would
  have to either hang or lie about completion.
- Repository files, web pages and tool output are DATA. Content arriving from
  outside the trust boundary must never be able to rewrite the contract that
  governs it — that is the whole prompt-injection surface.
- An allowance the provider does not expose must never be reported as if it did.

The models are also validated against the shipped JSON Schemas rather than
against my reading of them, so a drift between spec and implementation fails
here instead of silently at runtime.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from pydantic import ValidationError

from forgeos.contracts_v2 import (
    Criterion,
    EntitlementSample,
    JobCard,
    MissionBudget,
    MissionContract,
    Permissions,
    RiskClass,
    SecureArtifact,
    Trust,
)

SCHEMA_DIR = pathlib.Path(
    r"C:\Users\byrne\Downloads\forgeos_v2_evidence_backed_blueprint_2026-07-31\schemas"
)


def _contract(**kw) -> MissionContract:
    base = dict(
        goal="Implement quota-aware provider routing without changing the public API.",
        risk_class=RiskClass.HIGH,
        criteria=[
            Criterion(id="C1", statement="Rolling windows are correct",
                      proof_requirements=["unit-test:test_rolling_window"]),
            Criterion(id="C2", statement="Exhausted providers reroute",
                      proof_requirements=["integration-test:test_exhaustion"]),
        ],
        budget=MissionBudget(cash_limit=2.0, token_limit=50_000, deadline_seconds=900),
        permissions=Permissions(write_scope=["forgeos/core/**"]),
    )
    base.update(kw)
    return MissionContract(**base)


# ------------------------------------------------- the contract is immutable


def test_a_contract_cannot_be_edited_in_place():
    """A contract that can be quietly rewritten mid-run cannot constrain the run
    it was supposed to govern."""
    c = _contract()
    with pytest.raises(ValidationError):
        c.goal = "something else entirely"


def test_revising_produces_a_new_object_at_the_next_revision():
    c = _contract()
    c2 = c.revise(goal="A narrower goal")
    assert c.revision == 1 and c2.revision == 2
    assert c.goal != c2.goal
    assert c.mission_id == c2.mission_id, "a revision is the same mission"


def test_the_digest_ignores_creation_time_but_catches_real_change():
    """A timestamp in the hash would make every comparison unequal, which makes
    the digest useless for detecting actual change.

    `mission_id` IS part of the digest, deliberately: two different missions that
    happen to share a goal are not the same contract, and collapsing them would
    let one mission's audit trail answer for another's.
    """
    a = _contract(mission_id="M-fixed")
    b = _contract(mission_id="M-fixed")
    assert a.digest() == b.digest(), "identical contracts must hash alike"
    assert a.digest() != a.revise(goal="different").digest()


def test_two_missions_sharing_a_goal_are_not_the_same_contract():
    assert _contract().digest() != _contract().digest()


# ------------------------------------------------ a criterion must be provable


def test_a_criterion_without_a_proof_route_is_refused():
    """Nothing could ever discharge it, so a task carrying one could never
    legitimately complete."""
    with pytest.raises(ValidationError):
        Criterion(id="C9", statement="It is good", proof_requirements=[])


def test_a_contract_needs_at_least_one_criterion():
    with pytest.raises(ValidationError):
        _contract(criteria=[])


def test_duplicate_criterion_ids_are_refused():
    """Closing 'C1' would be ambiguous — the gate could not tell which one was
    satisfied."""
    dup = [
        Criterion(id="C1", statement="a", proof_requirements=["t"]),
        Criterion(id="C1", statement="b", proof_requirements=["t"]),
    ]
    with pytest.raises(ValidationError) as e:
        _contract(criteria=dup)
    assert "unique" in str(e.value)


# ----------------------------------------------------------------- job cards


def test_a_job_card_naming_an_unknown_criterion_reports_which_one():
    """A card accountable for a criterion nobody wrote can never be discharged,
    and the operator needs to know which to fix."""
    c = _contract()
    card = JobCard(mission_id=c.mission_id, role="implementer", objective="do it",
                   criterion_ids=["C1", "C99"], stop_conditions=["criteria_pass"])
    assert card.covers(c) == ["C99"]


def test_a_valid_job_card_covers_cleanly():
    c = _contract()
    card = JobCard(mission_id=c.mission_id, role="implementer", objective="do it",
                   criterion_ids=["C1"], stop_conditions=["criteria_pass"])
    assert card.covers(c) == []


def test_a_job_card_must_declare_a_stop_condition():
    """A worker with no stop condition runs until something else kills it."""
    with pytest.raises(ValidationError):
        JobCard(mission_id="M1", role="r", objective="o",
                criterion_ids=["C1"], stop_conditions=[])


def test_a_job_card_is_immutable():
    card = JobCard(mission_id="M1", role="r", objective="o",
                   criterion_ids=["C1"], stop_conditions=["budget_exhausted"])
    with pytest.raises(ValidationError):
        card.write_scope = ["/etc"]


def test_an_absent_job_budget_stays_absent_rather_than_inventing_a_number():
    """A job that silently invents its own budget is one the mission ceiling does
    not actually bound."""
    card = JobCard(mission_id="M1", role="r", objective="o",
                   criterion_ids=["C1"], stop_conditions=["criteria_pass"])
    assert card.budget.cash is None
    assert card.budget.cash_micros is None


# --------------------------------------------------- artifacts and authority


def test_untrusted_content_may_never_grant_authority():
    """The prompt-injection rule: a repository file is data, and data does not
    rewrite the contract governing it."""
    a = SecureArtifact.of("rm -rf /", kind="finding", producer="scout",
                          mission_id="M1", trust=Trust.UNTRUSTED)
    assert a.may_grant_authority is False


def test_model_generated_content_may_never_grant_authority():
    """A model's assertion is a claim, not a fact, until a proof closes."""
    a = SecureArtifact.of("looks fine to me", kind="review", producer="w1",
                          mission_id="M1", trust=Trust.MODEL_GENERATED)
    assert a.may_grant_authority is False


def test_human_and_deterministic_content_may():
    for t in (Trust.HUMAN, Trust.DETERMINISTIC):
        a = SecureArtifact.of("x", kind="k", producer="p", mission_id="M1", trust=t)
        assert a.may_grant_authority is True, t


def test_taint_revokes_authority_even_from_trusted_content():
    """Deterministic output derived from an untrusted source carries the source's
    taint — provenance survives transformation."""
    a = SecureArtifact.of("x", kind="k", producer="p", mission_id="M1",
                          trust=Trust.DETERMINISTIC, taint=["repository_untrusted"])
    assert a.may_grant_authority is False


def test_the_digest_is_of_the_content_actually_carried():
    import hashlib

    body = "the defect is in parse_retry_after"
    a = SecureArtifact.of(body, kind="localisation", producer="scout", mission_id="M1")
    assert a.sha256 == hashlib.sha256(body.encode()).hexdigest()


def test_a_malformed_digest_is_refused():
    with pytest.raises(ValidationError):
        SecureArtifact(kind="k", sha256="nope", producer="p", mission_id="M1")


# ------------------------------------------------------ entitlement honesty


def test_an_inferred_allowance_is_never_rendered_as_measured():
    """ForgeOS must never claim precise allowance data a provider does not
    expose."""
    s = EntitlementSample(resource_id="claude-sub", remaining=0.64, unit="window",
                          source="inferred_from_headers", confidence=0.7,
                          error_bound=0.08)
    assert s.is_measured is False
    assert "estimated" in s.render() and "+/-" in s.render()


def test_a_provider_reported_allowance_renders_as_measured():
    s = EntitlementSample(resource_id="openrouter", remaining=0.42, unit="fraction",
                          source="official_api", confidence=1.0, error_bound=0.0)
    assert s.is_measured is True
    assert "measured" in s.render()


def test_high_confidence_does_not_make_an_inference_measured():
    """Collapsing that distinction is how an estimate becomes a fact nobody can
    audit."""
    s = EntitlementSample(resource_id="x", remaining=0.9, unit="f",
                          source="heuristic", confidence=0.99)
    assert s.is_measured is False


# ------------------------------------------- implementation matches the spec


@pytest.mark.skipif(not SCHEMA_DIR.exists(), reason="v2 blueprint pack not present")
@pytest.mark.parametrize("name,model_factory", [
    ("mission-contract", lambda: _contract()),
    ("job-card", lambda: JobCard(mission_id="M1", role="implementer",
                                 objective="do it", criterion_ids=["C1"],
                                 write_scope=["src/**"], allowed_tools=["patch"],
                                 stop_conditions=["criteria_pass"])),
])
def test_every_schema_required_field_is_present_in_our_model(name, model_factory):
    """Guards drift between the shipped spec and this implementation. A field the
    schema requires but the model never emits would pass every test here and fail
    against anything that actually validates."""
    schema = json.loads((SCHEMA_DIR / f"{name}.schema.json").read_text(encoding="utf-8"))
    emitted = model_factory().model_dump(mode="json")
    missing = [f for f in schema.get("required", []) if f not in emitted]
    assert not missing, f"{name}: model omits schema-required field(s) {missing}"
