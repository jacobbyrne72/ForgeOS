"""Skipping orchestration on work that does not need it — without opening a hole.

Motivation, measured: a one-line docstring edit submitted through the dashboard
went through compile -> route -> execute -> review -> merge gate and drew four
model calls plus a tier prior, where one raw call answered the same request.

The danger is the mirror image: a bypass that fires on ambiguous work trades a
small saving for unreviewed wrong code merged fast. So most of this file is
about what must NOT be bypassed.
"""

from __future__ import annotations

import pytest

from forgeos.core.bypass import BypassDecision, should_bypass
from forgeos.core.effort import Difficulty


def _mech(**kw):
    base = dict(objective="rename _weaker to _less_certain", scope_paths=1,
                capabilities={"edit"}, acceptance=["the symbol is renamed"])
    base.update(kw)
    return should_bypass(**base)


# ------------------------------------------------------------ eligible work


def test_a_fully_specified_one_file_edit_bypasses():
    d = _mech()
    assert d.bypass
    assert d.difficulty is Difficulty.MECHANICAL


def test_a_lookup_bypasses():
    d = should_bypass("which function records spend to the ledger?",
                      acceptance=["names the function"])
    assert d.bypass


def test_the_reason_is_stated_when_it_bypasses():
    """A silent bypass is unauditable, and this decision changes what a receipt
    means."""
    assert "orchestration would add cost" in _mech().reason


# ------------------------------------------------- what must never bypass


@pytest.mark.parametrize("cap", ["security", "migration", "auth", "deploy", "release"])
def test_risky_capabilities_never_bypass_however_small(cap):
    """A one-line auth edit is still an auth edit."""
    d = _mech(capabilities={"edit", cap})
    assert not d.bypass
    assert cap in d.reason


def test_multi_file_work_never_bypasses():
    """Once a change spans files, choosing among them IS the planning."""
    d = _mech(scope_paths=4)
    assert not d.bypass and "4 paths" in d.reason


@pytest.mark.parametrize("objective", [
    "why does the lease occasionally grant two workers the same path?",
    "design the retry policy for provider failover",
    "debug the intermittent scheduler failure",
    "add provider failover to the gateway",
])
def test_open_ended_work_never_bypasses(objective):
    assert not should_bypass(objective, acceptance=["it works"]).bypass


def test_a_task_with_dependencies_never_bypasses():
    """Whatever the task is alone, its ORDER matters, and ordering is the
    scheduler's job."""
    d = _mech(depends_on=["task_1"])
    assert not d.bypass and "dependency" in d.reason


def test_no_acceptance_criterion_means_no_bypass():
    """The safety argument is that the merge gate still runs. With nothing to
    check mechanically, there is no argument."""
    d = _mech(acceptance=[])
    assert not d.bypass and "acceptance" in d.reason


def test_the_caller_can_always_force_the_full_path():
    d = _mech(force_full=True)
    assert not d.bypass and "caller requested" in d.reason


def test_force_full_outranks_every_eligibility_signal():
    assert not _mech(force_full=True, scope_paths=1, acceptance=["x"]).bypass


# ------------------------------------------------------------- precedence


def test_the_strongest_objection_is_the_one_reported():
    """A multi-file security task should say 'security', not 'too many paths' --
    the reason is what tells a maintainer whether the policy is behaving."""
    d = should_bypass("rename the token", scope_paths=9,
                      capabilities={"security"}, acceptance=["done"])
    assert not d.bypass and "security" in d.reason


def test_a_dependency_outranks_a_risky_capability_report():
    d = should_bypass("rename the token", capabilities={"security"},
                      acceptance=["done"], depends_on=["t1"])
    assert "dependency" in d.reason


# ------------------------------------------------------------ spec adapter


def test_bypass_for_spec_reads_a_real_task_spec():
    from forgeos.contracts import Budget, Scope, TaskSpec
    from forgeos.core.bypass import bypass_for_spec

    spec = TaskSpec(job_id="j", subject="rename the helper", description="d",
                    scope=Scope(paths=["a.py"]), capabilities=["edit"],
                    acceptance=["renamed"], budget=Budget(max_usd=1.0))
    assert bypass_for_spec(spec).bypass


def test_bypass_for_spec_never_raises_on_a_broken_spec():
    """A bypass decision is an optimisation. Failing a real task because
    eligibility could not be computed trades the work for a routing detail."""
    from forgeos.core.bypass import bypass_for_spec

    class Broken:
        @property
        def subject(self):
            raise RuntimeError("boom")

    d = bypass_for_spec(Broken())
    assert isinstance(d, BypassDecision) and not d.bypass


def test_a_spec_with_no_acceptance_does_not_bypass():
    from forgeos.contracts import Budget, Scope, TaskSpec
    from forgeos.core.bypass import bypass_for_spec

    spec = TaskSpec(job_id="j", subject="rename the helper", description="d",
                    scope=Scope(paths=["a.py"]), capabilities=["edit"],
                    acceptance=[], budget=Budget(max_usd=1.0))
    assert not bypass_for_spec(spec).bypass


# --------------------------------------------------------------- determinism


def test_the_decision_is_deterministic():
    """Routing that varies between identical calls makes a cost regression
    impossible to attribute."""
    first = _mech()
    for _ in range(5):
        again = _mech()
        assert (again.bypass, again.reason) == (first.bypass, first.reason)


def test_no_model_call_is_made(monkeypatch):
    import socket

    def _boom(*a, **k):  # pragma: no cover - only runs if the rule is broken
        raise AssertionError("should_bypass attempted network I/O")

    monkeypatch.setattr(socket, "socket", _boom)
    monkeypatch.setattr(socket, "create_connection", _boom)
    assert _mech().bypass
