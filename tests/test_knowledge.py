"""Tests for the claim promotion gate.

These are the tests that matter most in the repo. Every one of them describes a way
an unverified outside claim could have become a standing agent instruction, and
asserts that it cannot.
"""

from __future__ import annotations

import pytest

from forgeos.knowledge import (
    MIN_SOURCES_TO_PROMOTE,
    ClaimStore,
    ClaimType,
    VerificationStatus,
    claim_key,
)


@pytest.fixture()
def store():
    s = ClaimStore(":memory:")
    yield s
    s.close()


def _add(store, text="isolate subagent context", ctype=ClaimType.RECOMMENDATION, **kw):
    kw.setdefault("source_ref", "video://abc123")
    kw.setdefault("source_hash", "sha256:aaa")
    return store.add(text, ctype, **kw)


# ------------------------------------------------------------------ identity


def test_claim_key_normalises_whitespace_and_case():
    assert claim_key("Isolate  Subagent   Context") == claim_key("isolate subagent context")


def test_same_claim_from_two_sources_collapses_to_one_row():
    s = ClaimStore(":memory:")
    a = _add(s, source_ref="video://one")
    b = _add(s, source_ref="video://two")
    assert a.id == b.id
    assert s.source_count(a.id) == 2
    s.close()


def test_same_source_twice_does_not_fake_corroboration():
    """Re-reading one video must not look like two independent sightings."""
    s = ClaimStore(":memory:")
    c = _add(s, source_ref="video://one")
    _add(s, source_ref="video://one")
    assert s.source_count(c.id) == 1
    assert s.get(c.id).status is VerificationStatus.UNVERIFIED
    s.close()


def test_second_independent_source_upgrades_to_corroborated(store):
    c = _add(store, source_ref="video://one")
    assert store.get(c.id).status is VerificationStatus.UNVERIFIED
    _add(store, source_ref="blog://two")
    assert store.get(c.id).status is VerificationStatus.CORROBORATED


# ------------------------------------------------------------------- the gate


def test_single_source_claim_cannot_be_promoted(store):
    c = _add(store, source_ref="video://only")
    r = store.promote(c.id)
    assert r.promoted is False
    assert "source" in r.reason


def test_corroborated_claim_can_be_promoted(store):
    c = _add(store, source_ref="video://one")
    _add(store, source_ref="blog://two")
    r = store.promote(c.id)
    assert r.promoted is True
    assert store.get(c.id).agents_may_follow


def test_self_verification_alone_is_enough(store):
    """We reproduced it. That outranks any number of people asserting it."""
    c = _add(store, source_ref="video://one")
    store.mark_verified(c.id)
    assert store.promote(c.id).promoted is True


def test_claim_without_source_hash_is_refused(store):
    """Evidence has to be content-addressed or it cannot be re-checked later."""
    c = _add(store, source_ref="video://one", source_hash="")
    _add(store, source_ref="blog://two", source_hash="")
    r = store.promote(c.id)
    assert r.promoted is False
    assert "hash" in r.reason


def test_opinions_are_never_promotable_however_popular(store):
    """Ten people repeating an opinion does not make it a fact."""
    c = _add(store, "framework X is the best", ClaimType.OPINION, source_ref="v://1")
    for i in range(2, 12):
        _add(store, "framework X is the best", ClaimType.OPINION, source_ref=f"v://{i}")
    assert store.source_count(c.id) == 11
    r = store.promote(c.id)
    assert r.promoted is False
    assert "never promotable" in r.reason


def test_predictions_are_never_promotable(store):
    c = _add(store, "agents will replace IDEs", ClaimType.PREDICTION, source_ref="v://1")
    _add(store, "agents will replace IDEs", ClaimType.PREDICTION, source_ref="v://2")
    assert store.promote(c.id).promoted is False


def test_time_sensitive_claim_needs_self_verification_not_corroboration(store):
    """A stale tool instruction fails confidently, which is worse than absent."""
    c = _add(store, "use --flag to enable caching", ClaimType.TOOL_INSTRUCTION,
             source_ref="v://1", time_sensitive=True)
    _add(store, "use --flag to enable caching", ClaimType.TOOL_INSTRUCTION,
         source_ref="v://2", time_sensitive=True)
    r = store.promote(c.id)
    assert r.promoted is False
    assert "time-sensitive" in r.reason

    store.mark_verified(c.id)
    assert store.promote(c.id).promoted is True


def test_contradicted_claim_cannot_be_promoted(store):
    a = _add(store, "always compress the prefix", source_ref="v://1")
    _add(store, "always compress the prefix", source_ref="v://2")
    b = _add(store, "never compress the prefix", source_ref="v://3")
    store.add_conflict(a.id, b.id, detail="prefix compression destroys cache hits")
    r = store.promote(a.id)
    assert r.promoted is False
    assert store.get(a.id).status is VerificationStatus.CONTRADICTED


def test_promoting_unknown_claim_is_refused_not_crashed(store):
    assert store.promote("clm_nope").promoted is False


# --------------------------------------------------------------- consumption


def test_agents_only_ever_see_promoted_claims(store):
    good = _add(store, "verify with real command output", source_ref="v://1")
    _add(store, "verify with real command output", source_ref="v://2")
    store.promote(good.id)

    _add(store, "rewrite everything in rust", ClaimType.OPINION, source_ref="v://9")
    _add(store, "unreplicated tip", source_ref="v://10")

    instructions = store.instructions()
    assert [c.text for c in instructions] == ["verify with real command output"]


def test_fresh_store_yields_no_instructions(store):
    """Default state is "agents follow nothing outside their own rules"."""
    assert store.instructions() == []
    assert store.stats() == {}


def test_min_sources_constant_is_at_least_two():
    """An unreplicated claim is an anecdote; the floor must never drop to 1."""
    assert MIN_SOURCES_TO_PROMOTE >= 2
