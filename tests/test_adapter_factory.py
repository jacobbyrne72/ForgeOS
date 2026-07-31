"""Tests for `forgeos/adapters/factory.py`.

The property under test is refusal, not construction. Anyone can build an object;
the thing that matters is that a machine which cannot run a backend gets told so,
rather than being handed a different worker that happens to be available.

A silent substitution here would corrupt the ledger in a way nothing downstream
could detect: the router's chosen profile gets the attribution, the substitute
does the work, and every win-rate the router later learns from is evidence about
a worker that never ran the task.
"""

from __future__ import annotations

from forgeos.adapters.factory import build_adapter, runnable_workers
from forgeos.registry import Adapter, CostTier, WorkerProfile, acp_profiles, default_registry


def _profile(adapter: Adapter, **kw) -> WorkerProfile:
    base = dict(worker_id=f"w.{adapter.value}", adapter=adapter, tier=CostTier.FREE)
    base.update(kw)
    return WorkerProfile(**base)


class _FakeCatalog:
    def get(self, ref):
        return object()


class _FakeGateway:
    _transports = [object()]
    _catalog = _FakeCatalog()


# ============================================================ gateway-backed


def test_a_gateway_profile_builds_when_given_a_gateway_and_a_live_budget():
    p = _profile(Adapter.GATEWAY, model="openrouter/free-model")
    adapter, reason = build_adapter(
        p, gateway=_FakeGateway(), job_id="job-1", remaining_micros=lambda: 1_000_000
    )
    assert adapter is not None
    assert "ready" in reason or "built" in reason


def test_a_gateway_profile_refuses_to_build_without_a_gateway():
    """Constructing one here would mean inventing a Ledger, and the spend would
    land somewhere nobody reads."""
    adapter, reason = build_adapter(
        _profile(Adapter.GATEWAY), remaining_micros=lambda: 1_000
    )
    assert adapter is None
    assert "Gateway" in reason


def test_a_gateway_profile_refuses_to_build_without_a_live_budget():
    """A snapshot budget cannot refuse an over-budget call, so there is no safe
    default to substitute."""
    adapter, reason = build_adapter(_profile(Adapter.GATEWAY), gateway=_FakeGateway())
    assert adapter is None
    assert "remaining_micros" in reason


def test_the_gateway_adapter_is_built_for_the_profiles_own_worker_id():
    """Spend must be attributed to the routed profile, not to a generic label."""
    p = _profile(Adapter.GATEWAY, worker_id="gateway.free", model="openrouter/free-model")
    adapter, _ = build_adapter(
        p, gateway=_FakeGateway(), job_id="j", remaining_micros=lambda: 10
    )
    assert adapter._worker_id == "gateway.free"


# =============================================================== unavailable


def test_an_unhealthy_backend_is_reported_unavailable_not_substituted():
    p = _profile(Adapter.OLLAMA, model="nope:1b")
    adapter, reason = build_adapter(p)
    # Ollama may or may not be installed on the machine running this. Either
    # answer is correct; silently returning a *different* backend is not.
    if adapter is None:
        assert p.worker_id in reason or "ollama" in reason.lower()
    else:
        assert adapter.__class__.__name__ == "OllamaAdapter"


def test_health_is_skipped_when_asked_so_construction_can_be_tested_alone():
    adapter, _ = build_adapter(_profile(Adapter.OLLAMA), check_health=False)
    assert adapter is not None
    assert adapter.__class__.__name__ == "OllamaAdapter"


def test_a_backend_whose_health_check_raises_is_unavailable_not_a_crash():
    """`health()` is contractually forbidden from raising, but a third-party
    adapter can still break that promise and must not take the fleet down."""

    class Exploding:
        adapter = Adapter.OLLAMA
        worker_id = "boom"

        def health(self):
            raise RuntimeError("kaboom")

    import forgeos.adapters.factory as factory

    original = factory._construct
    factory._construct = lambda p, **kw: (Exploding(), "built")
    try:
        adapter, reason = build_adapter(_profile(Adapter.OLLAMA))
    finally:
        factory._construct = original

    assert adapter is None
    assert "RuntimeError" in reason
    assert "kaboom" in reason


# ======================================================================= acp


def test_an_acp_profile_refuses_to_build_without_a_command():
    p = _profile(Adapter.ACP)
    adapter, reason = build_adapter(p, check_health=False)
    assert adapter is None
    assert p.worker_id in reason


def test_an_acp_profile_is_unavailable_when_the_sdk_is_not_installed(monkeypatch):
    from forgeos.adapters import acp as acp_adapter

    def boom():
        raise ImportError("no module named acp")

    monkeypatch.setattr(acp_adapter, "_import_acp", boom)
    p = _profile(Adapter.ACP, command="claude-agent-acp")

    adapter, reason = build_adapter(p)

    assert adapter is None
    assert "acp SDK not installed" in reason


def test_an_acp_profile_is_unavailable_when_the_cli_is_not_on_path(monkeypatch):
    from forgeos.adapters import acp as acp_adapter

    monkeypatch.setattr(acp_adapter, "_import_acp", lambda: object())
    monkeypatch.setattr(acp_adapter.shutil, "which", lambda cmd: None)
    p = _profile(Adapter.ACP, command="nonexistent-cli")

    adapter, reason = build_adapter(p)

    assert adapter is None
    assert "nonexistent-cli" in reason


def test_an_acp_profile_builds_with_a_fake_sdk_and_cli_on_path(monkeypatch):
    from forgeos.adapters import acp as acp_adapter

    monkeypatch.setattr(acp_adapter, "_import_acp", lambda: object())
    monkeypatch.setattr(acp_adapter.shutil, "which", lambda cmd: f"/usr/bin/{cmd}")
    p = _profile(Adapter.ACP, command="claude-agent-acp", args=["--flag"])

    adapter, reason = build_adapter(p)

    assert adapter is not None
    assert adapter.__class__.__name__ == "ACPAdapter"
    assert "importable" in reason


def test_acp_example_profiles_are_valid_and_constructible():
    """Not enabled by default (see registry.acp_profiles docstring), but each
    one must still name a real, buildable adapter."""
    profiles = acp_profiles()
    assert len(profiles) == 2
    for p in profiles:
        assert p.adapter is Adapter.ACP
        assert p.command
        adapter, reason = build_adapter(p, check_health=False)
        assert adapter is not None, reason
        assert adapter.__class__.__name__ == "ACPAdapter"


def test_acp_example_profiles_are_not_in_the_default_registry():
    default_ids = {p.worker_id for p in default_registry().all()}
    assert not default_ids & {p.worker_id for p in acp_profiles()}


# ================================================================= coverage


def test_every_adapter_enum_member_has_an_implementation_branch():
    """The defect this module exists to fix: an enum member no code can build.

    A profile naming such an adapter is routable, leasable, and unrunnable — it
    loses the task after the scheduling slot has already been spent on it.
    """
    unimplemented = []
    for member in Adapter:
        _, reason = build_adapter(
            _profile(member),
            gateway=_FakeGateway(),
            job_id="j",
            remaining_micros=lambda: 1_000,
            check_health=False,
        )
        if "no implementation registered" in reason:
            unimplemented.append(member.value)
    assert not unimplemented, f"adapters with no implementation: {unimplemented}"


def test_every_worker_in_the_default_fleet_names_a_buildable_adapter():
    """Not that every worker runs here — that every worker *could*, given its
    backend. A fleet entry whose adapter has no code is a promise forgeos cannot
    keep regardless of what is installed."""
    orphans = []
    for p in default_registry().all():
        _, reason = build_adapter(
            p,
            gateway=_FakeGateway(),
            job_id="j",
            remaining_micros=lambda: 1_000,
            check_health=False,
        )
        if "no implementation registered" in reason:
            orphans.append((p.worker_id, p.adapter.value))
    assert not orphans, f"registered workers with no adapter implementation: {orphans}"


# ================================================================== doctor


def test_runnable_workers_reports_one_verdict_per_registered_profile():
    profiles = default_registry().all()
    report = runnable_workers(profiles, gateway=_FakeGateway(), remaining_micros=lambda: 1_000)
    assert set(report) == {p.worker_id for p in profiles}
    assert all(v.startswith(("ok: ", "unavailable: ")) for v in report.values())


def test_runnable_workers_explains_every_unavailable_worker():
    """An empty reason is the same failure as no report: the caller cannot route
    around a problem it has not been told about."""
    report = runnable_workers(default_registry().all())
    for worker_id, verdict in report.items():
        detail = verdict.split(": ", 1)[1]
        assert detail.strip(), f"{worker_id} was judged with no reason given"
