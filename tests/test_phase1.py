"""Phase 1 tests: contracts, ledger, registry. No LLM calls, no network."""

from __future__ import annotations

import pytest

from forgeos.contracts import (
    Budget,
    Escalation,
    EscalationKind,
    GovernorTrip,
    JobSpec,
    Scope,
    TaskSpec,
    TaskState,
    Verdict,
    WorkerReport,
    from_micros,
    to_micros,
)
from forgeos.ledger import Ledger
from forgeos.registry import (
    MIN_ATTEMPTS_TO_TRUST,
    TIER_PRIOR_MICROS,
    Adapter,
    CostTier,
    Registry,
    WorkerProfile,
    default_registry,
)


# ---------------------------------------------------------------- contracts


def test_money_roundtrip_is_exact_at_cent_boundaries():
    for usd in (0.0, 0.01, 0.1, 1.0, 2.5, 19.99, 400.0):
        assert from_micros(to_micros(usd)) == pytest.approx(usd)


def test_money_rounds_not_truncates():
    # 0.0000005 must not silently vanish; truncation is how caps leak.
    assert to_micros(0.0000015) == 2
    assert to_micros(1.9999995) == 2000000


def test_budget_rejects_nonpositive():
    for kwargs in ({"max_usd": 0}, {"max_usd": -1}, {"max_seconds": 0}, {"max_iterations": 0}):
        with pytest.raises(ValueError):
            Budget(**kwargs)


def test_budget_exposes_micros():
    assert Budget(max_usd=2.5).max_usd_micros == 2_500_000


def test_taskspec_rejects_blank_text():
    with pytest.raises(ValueError):
        TaskSpec(job_id="j", subject="   ", description="d")
    with pytest.raises(ValueError):
        TaskSpec(job_id="j", subject="s", description="")


def test_jobspec_rejects_blank_objective():
    with pytest.raises(ValueError):
        JobSpec(objective="  ", cwd=".")


def test_scope_covers_handles_windows_and_posix_separators():
    scope = Scope(paths=["forgeos/core", "tests"])
    assert scope.covers("forgeos/core/router.py")
    assert scope.covers("forgeos\\core\\router.py")
    assert scope.covers("./tests/test_phase1.py")
    assert not scope.covers("forgeos/dashboard/app.py")


def test_empty_scope_covers_nothing():
    assert not Scope().covers("anything.py")


def test_report_succeeded_requires_done_and_not_failed_verdict():
    base = {"task_id": "t", "worker_id": "w"}
    assert WorkerReport(**base, state=TaskState.DONE).succeeded
    assert WorkerReport(**base, state=TaskState.DONE, verdict=Verdict.PASS).succeeded
    assert not WorkerReport(**base, state=TaskState.DONE, verdict=Verdict.FAIL).succeeded
    assert not WorkerReport(**base, state=TaskState.RUNNING).succeeded


def test_confidence_is_bounded():
    with pytest.raises(ValueError):
        WorkerReport(task_id="t", worker_id="w", state=TaskState.DONE, confidence=1.5)


def test_escalation_open_until_resolved():
    esc = Escalation(job_id="j", task_id="t", kind=EscalationKind.STUCK)
    assert esc.open
    assert Escalation(job_id="j", task_id="t", kind=EscalationKind.LOOP, resolved_at=1.0).open is False


# ------------------------------------------------------------------- ledger


@pytest.fixture()
def led():
    ledger = Ledger(":memory:")
    yield ledger
    ledger.close()


@pytest.fixture()
def job(led):
    j = JobSpec(objective="build the thing", cwd=".")
    led.open_job(j)
    return j


def test_open_job_is_readable_and_active(led, job):
    row = led.job(job.id)
    assert row["objective"] == "build the thing"
    assert row["closed_at"] is None
    assert [r["id"] for r in led.active_jobs()] == [job.id]


def test_open_job_persists_execution_isolation_contract(led):
    j = JobSpec(objective="preserve worktree mode", cwd=".")
    led.open_job(j, isolate_worktrees=True, base_ref="abc123")
    row = led.job(j.id)
    assert row["isolate_worktrees"] == 1
    assert row["base_ref"] == "abc123"


def test_close_job_removes_it_from_active(led, job):
    led.close_job(job.id, TaskState.DONE)
    assert led.active_jobs() == []
    assert led.job(job.id)["state"] == "done"


def test_add_task_persists_scope_and_budget(led, job):
    t = TaskSpec(
        job_id=job.id,
        subject="s",
        description="d",
        scope=Scope(paths=["forgeos"]),
        budget=Budget(max_usd=1.5),
        capabilities=["python", "edit"],
    )
    led.add_task(t)
    row = led.task(t.id)
    assert row["state"] == "pending"
    assert "forgeos" in row["scope"]
    assert "1.5" in row["budget"]


def test_unfinished_tasks_excludes_terminal_states(led, job):
    a = TaskSpec(job_id=job.id, subject="a", description="d")
    b = TaskSpec(job_id=job.id, subject="b", description="d")
    c = TaskSpec(job_id=job.id, subject="c", description="d")
    for t in (a, b, c):
        led.add_task(t)
    led.set_task_state(a.id, TaskState.DONE)
    led.set_task_state(b.id, TaskState.FAILED)
    assert [r["id"] for r in led.unfinished_tasks(job.id)] == [c.id]


def test_record_report_also_advances_task_state(led, job):
    t = TaskSpec(job_id=job.id, subject="s", description="d")
    led.add_task(t)
    led.record_report(
        WorkerReport(task_id=t.id, worker_id="w1", state=TaskState.DONE, verdict=Verdict.PASS)
    )
    assert led.task(t.id)["state"] == "done"


def test_spend_sums_per_job_and_per_task(led, job):
    t = TaskSpec(job_id=job.id, subject="s", description="d")
    led.add_task(t)
    led.record_spend(job.id, "w1", "m", 1_000, task_id=t.id)
    led.record_spend(job.id, "w1", "m", 2_500, task_id=t.id)
    led.record_spend(job.id, "w2", "m", 500)  # job-level, no task
    assert led.job_spend_micros(job.id) == 4_000
    assert led.task_spend_micros(t.id) == 3_500


def test_spend_of_unknown_job_is_zero_not_error(led):
    assert led.job_spend_micros("nope") == 0


def test_cache_stats_reports_hit_rate(led, job):
    led.record_spend(job.id, "w", "m", 100, tokens_in=200, tokens_cached_in=800, tokens_out=50)
    stats = led.cache_stats(job.id)
    assert stats == {
        "fresh_in": 200,
        "cached_in": 800,
        "tokens_out": 50,
        "cache_hit_pct": 80.0,
    }


def test_cache_stats_empty_does_not_divide_by_zero(led, job):
    assert led.cache_stats(job.id)["cache_hit_pct"] == 0.0


def test_escalation_lifecycle(led, job):
    t = TaskSpec(job_id=job.id, subject="s", description="d")
    led.add_task(t)
    esc = Escalation(job_id=job.id, task_id=t.id, kind=EscalationKind.SCOPE_REQUEST, detail="need core/")
    led.open_escalation(esc)
    assert len(led.open_escalations(job.id)) == 1
    led.resolve_escalation(esc.id, "approved")
    assert led.open_escalations(job.id) == []


def test_trips_are_recorded_for_the_human(led, job):
    led.record_trip(
        GovernorTrip(job_id=job.id, reason="usd cap", limit="$2.00", observed="$2.13")
    )
    trips = led.trips_for_job(job.id)
    assert len(trips) == 1
    assert trips[0]["observed"] == "$2.13"


def test_events_return_newest_first(led, job):
    led.record_event(job.id, "assigned", "a")
    led.record_event(job.id, "assigned", "b")
    assert [e["detail"] for e in led.events_for_job(job.id)][0] == "b"


def test_capability_stats_credits_only_the_matching_capability(led, job):
    py = TaskSpec(job_id=job.id, subject="py", description="d", capabilities=["python"])
    rs = TaskSpec(job_id=job.id, subject="rs", description="d", capabilities=["rust"])
    led.add_task(py)
    led.add_task(rs)
    led.record_report(
        WorkerReport(task_id=py.id, worker_id="w1", state=TaskState.DONE, usd_micros=100, seconds=10)
    )
    led.record_report(
        WorkerReport(task_id=rs.id, worker_id="w1", state=TaskState.DONE, usd_micros=900, seconds=90)
    )
    stats = led.capability_stats("python")
    assert len(stats) == 1
    assert stats[0]["attempts"] == 1
    assert stats[0]["avg_usd_micros"] == 100


def test_capability_stats_win_rate_counts_failures(led, job):
    t1 = TaskSpec(job_id=job.id, subject="a", description="d", capabilities=["edit"])
    t2 = TaskSpec(job_id=job.id, subject="b", description="d", capabilities=["edit"])
    led.add_task(t1)
    led.add_task(t2)
    led.record_report(WorkerReport(task_id=t1.id, worker_id="w", state=TaskState.DONE))
    led.record_report(WorkerReport(task_id=t2.id, worker_id="w", state=TaskState.FAILED))
    stats = led.capability_stats("edit")[0]
    assert stats["attempts"] == 2
    assert stats["wins"] == 1
    assert stats["win_rate"] == 0.5


def test_done_but_failed_verdict_is_not_a_win(led, job):
    t = TaskSpec(job_id=job.id, subject="a", description="d", capabilities=["edit"])
    led.add_task(t)
    led.record_report(
        WorkerReport(task_id=t.id, worker_id="w", state=TaskState.DONE, verdict=Verdict.FAIL)
    )
    assert led.capability_stats("edit")[0]["wins"] == 0


def test_min_attempts_filter_hides_thin_evidence(led, job):
    t = TaskSpec(job_id=job.id, subject="a", description="d", capabilities=["edit"])
    led.add_task(t)
    led.record_report(WorkerReport(task_id=t.id, worker_id="w", state=TaskState.DONE))
    assert led.capability_stats("edit", min_attempts=2) == []


def test_file_touch_counts_feed_the_loop_detector(led, job):
    t = TaskSpec(job_id=job.id, subject="a", description="d")
    led.add_task(t)
    for _ in range(3):
        led.record_report(
            WorkerReport(
                task_id=t.id,
                worker_id="w",
                state=TaskState.RUNNING,
                files_touched=["a.py", "b.py"],
                evidence="1 failed",
            )
        )
    assert led.file_touch_counts(t.id) == {"a.py": 3, "b.py": 3}
    # Identical evidence three times over is the churn signature.
    assert len(set(led.evidence_fingerprints(t.id))) == 1


def test_foreign_key_blocks_orphan_task(led):
    with pytest.raises(Exception):
        led.add_task(TaskSpec(job_id="ghost", subject="s", description="d"))


# ----------------------------------------------------------------- registry


def _w(worker_id: str, tier: CostTier, caps: set[str], **kw) -> WorkerProfile:
    return WorkerProfile(
        worker_id=worker_id, adapter=Adapter.CLI_TEAM, tier=tier, capabilities=caps, **kw
    )


def test_fit_is_fraction_of_required_capabilities():
    w = _w("a", CostTier.FREE, {"python", "edit"})
    assert w.fit(["python"]) == 1.0
    assert w.fit(["python", "rust"]) == 0.5
    assert w.fit([]) == 1.0


def test_rank_drops_workers_that_cannot_cover_every_capability():
    reg = Registry([_w("partial", CostTier.FREE, {"python"}), _w("full", CostTier.PREMIUM, {"python", "rust"})])
    ranked = reg.rank(["python", "rust"])
    assert [c.worker.worker_id for c in ranked] == ["full"]


def test_rank_prefers_free_when_fit_and_win_rate_are_equal():
    reg = Registry(
        [
            _w("paid", CostTier.PREMIUM, {"edit"}, prior_win_rate=0.7, est_seconds=60),
            _w("free", CostTier.FREE, {"edit"}, prior_win_rate=0.7, est_seconds=60),
        ]
    )
    assert reg.pick(["edit"]).worker.worker_id == "free"


def test_capability_fit_outranks_cost():
    """A cheap worker that cannot do the job must not win. Cheap failure costs more."""
    reg = Registry(
        [
            _w("cheap_wrong", CostTier.FREE, {"edit"}),
            _w("pricey_right", CostTier.PREMIUM, {"edit", "security"}),
        ]
    )
    assert reg.pick(["edit", "security"]).worker.worker_id == "pricey_right"


def test_needs_file_edits_filters_read_only_workers():
    reg = Registry(
        [
            _w("readonly", CostTier.FREE, {"edit"}, can_edit_files=False),
            _w("writer", CostTier.PREMIUM, {"edit"}, can_edit_files=True),
        ]
    )
    assert reg.pick(["edit"], needs_file_edits=True).worker.worker_id == "writer"


def test_no_eligible_worker_returns_none_not_a_bad_pick():
    reg = Registry([_w("a", CostTier.FREE, {"python"})])
    assert reg.pick(["kubernetes"]) is None
    assert reg.rank(["kubernetes"]) == []


def test_measured_stats_override_declared_prior_once_trusted():
    """A worker that claims 0.9 but measures 0.1 must lose to an honest rival."""
    reg = Registry(
        [
            _w("liar", CostTier.FREE, {"edit"}, prior_win_rate=0.95),
            _w("honest", CostTier.FREE, {"edit"}, prior_win_rate=0.6),
        ]
    )
    assert reg.pick(["edit"]).worker.worker_id == "liar"  # before evidence

    stats = {
        "liar": {"attempts": MIN_ATTEMPTS_TO_TRUST, "win_rate": 0.1, "avg_usd_micros": 0, "avg_seconds": 10},
        "honest": {"attempts": MIN_ATTEMPTS_TO_TRUST, "win_rate": 0.9, "avg_usd_micros": 0, "avg_seconds": 10},
    }
    best = reg.pick(["edit"], stats=stats)
    assert best.worker.worker_id == "honest"
    assert best.measured is True


def test_thin_evidence_is_ignored_and_labelled():
    reg = Registry([_w("a", CostTier.FREE, {"edit"}, prior_win_rate=0.6)])
    stats = {"a": {"attempts": 1, "win_rate": 0.0, "avg_usd_micros": 0, "avg_seconds": 1}}
    cand = reg.pick(["edit"], stats=stats)
    assert cand.measured is False
    assert cand.win_rate == 0.6
    assert "declared prior" in cand.reason


def test_default_fleet_sends_mechanical_edits_to_the_cheapest_worker_that_can_edit():
    """The cheapest-capable-first rule, checked against the real installed fleet.

    Note what this pins down: the default fleet has **no free file-editing worker**.
    The free workers here summarise and classify; they cannot write to disk. So the
    cheapest honest answer for a real edit is a metered one, and the test asserts
    that rather than a specific id — routing an edit to a worker that cannot edit
    would buy a guaranteed failure and a retry one tier up, which is worse than
    starting at the cheapest worker that can actually finish.
    """
    reg = default_registry()
    pick = reg.pick(["mechanical", "edit"], needs_file_edits=True)
    assert pick.worker.can_edit_files

    cheaper = [
        w.worker_id
        for w in reg.all()
        if w.can_edit_files
        and TIER_PRIOR_MICROS[w.tier] < TIER_PRIOR_MICROS[pick.worker.tier]
        and {"mechanical", "edit"} <= w.capabilities
    ]
    assert not cheaper, f"a cheaper capable editor was passed over: {cheaper}"


def test_default_fleet_still_reaches_premium_for_architecture():
    reg = default_registry()
    pick = reg.pick(["architecture", "design"])
    assert pick.worker.worker_id == "forgeos.architect"


def test_default_fleet_ids_are_unique():
    reg = default_registry()
    ids = [w.worker_id for w in reg.all()]
    assert len(ids) == len(set(ids))


# ------------------------------------------------------ cached-input pricing


def test_cached_input_is_priced_at_the_cache_rate_not_the_fresh_rate():
    """The whole savings claim depends on this. Pricing cached tokens at the fresh
    rate would make the biggest lever in the harness report zero dollars saved."""
    from forgeos.catalog import ModelCard

    card = ModelCard(
        model_id="m", provider="p",
        input_cost_per_1m=3.0, output_cost_per_1m=15.0, cache_read_cost_per_1m=0.3,
    )
    fresh = card.cost_micros(tokens_in=100_000, tokens_out=0)
    cached = card.cost_micros(tokens_in=0, tokens_out=0, tokens_cached_in=100_000)
    assert cached * 10 == fresh
    assert card.cache_discount == pytest.approx(0.9)


def test_missing_cache_price_falls_back_to_fresh_and_never_understates():
    """A conservative fallback can only make an estimate too high, never let a
    budget check pass when it should have failed."""
    from forgeos.catalog import ModelCard

    card = ModelCard(model_id="m", provider="p", input_cost_per_1m=2.0, output_cost_per_1m=4.0)
    assert card.cache_read_rate == 2.0
    assert card.cache_discount == 0.0
    assert card.cost_micros(0, 0, 1_000_000) == card.cost_micros(1_000_000, 0)


def test_real_catalog_actually_carries_cache_prices():
    """Guards against a schema change silently zeroing the discount."""
    from forgeos.catalog import default_catalog

    cat = default_catalog()
    if len(cat) == 0:
        pytest.skip("no local models.dev cache on this machine")
    discounted = [c for c in cat.all() if c.cache_discount > 0.5]
    assert len(discounted) > 100


# ------------------------------- win rate scores tasks, not check-ins


def test_a_worker_that_checks_in_before_succeeding_scores_100_percent(led, job):
    """A BLOCKED report is a worker asking for help, not a failure. Scoring per
    report would teach the router to prefer workers that stay silent — the exact
    opposite of what the event-driven escalation design needs."""
    t = TaskSpec(job_id=job.id, subject="a", description="d", capabilities=["edit"])
    led.add_task(t)
    led.record_report(
        WorkerReport(task_id=t.id, worker_id="w", state=TaskState.BLOCKED,
                     blocker="need a decision", usd_micros=400, seconds=10)
    )
    led.record_report(
        WorkerReport(task_id=t.id, worker_id="w", state=TaskState.DONE,
                     verdict=Verdict.PASS, usd_micros=300, seconds=8)
    )
    s = led.capability_stats("edit")[0]
    assert s["attempts"] == 1          # one task, not two reports
    assert s["wins"] == 1
    assert s["win_rate"] == 1.0
    # Spend is summed across the whole attempt, not averaged per report.
    assert s["avg_usd_micros"] == 700


def test_many_heartbeats_then_success_is_still_one_win(led, job):
    t = TaskSpec(job_id=job.id, subject="a", description="d", capabilities=["edit"])
    led.add_task(t)
    for _ in range(5):
        led.record_report(
            WorkerReport(task_id=t.id, worker_id="w", state=TaskState.RUNNING, usd_micros=100)
        )
    led.record_report(
        WorkerReport(task_id=t.id, worker_id="w", state=TaskState.DONE, verdict=Verdict.PASS)
    )
    s = led.capability_stats("edit")[0]
    assert s["attempts"] == 1
    assert s["win_rate"] == 1.0


def test_a_genuine_failure_still_counts_against_the_worker(led, job):
    """The fix must not make every worker look perfect."""
    good = TaskSpec(job_id=job.id, subject="a", description="d", capabilities=["edit"])
    bad = TaskSpec(job_id=job.id, subject="b", description="d", capabilities=["edit"])
    led.add_task(good)
    led.add_task(bad)
    led.record_report(
        WorkerReport(task_id=good.id, worker_id="w", state=TaskState.DONE, verdict=Verdict.PASS)
    )
    led.record_report(
        WorkerReport(task_id=bad.id, worker_id="w", state=TaskState.RUNNING)
    )
    led.record_report(
        WorkerReport(task_id=bad.id, worker_id="w", state=TaskState.FAILED)
    )
    s = led.capability_stats("edit")[0]
    assert s["attempts"] == 2
    assert s["wins"] == 1
    assert s["win_rate"] == 0.5


def test_two_workers_on_one_task_are_scored_separately(led, job):
    """A reassigned task must credit each worker for its own outcome."""
    t = TaskSpec(job_id=job.id, subject="a", description="d", capabilities=["edit"])
    led.add_task(t)
    led.record_report(WorkerReport(task_id=t.id, worker_id="first", state=TaskState.FAILED))
    led.record_report(
        WorkerReport(task_id=t.id, worker_id="second", state=TaskState.DONE, verdict=Verdict.PASS)
    )
    by_worker = {s["worker_id"]: s for s in led.capability_stats("edit")}
    assert by_worker["first"]["win_rate"] == 0.0
    assert by_worker["second"]["win_rate"] == 1.0


# --------------------- multi-capability stats must dedupe by task


def test_worker_stats_counts_a_multi_capability_task_once(led, job):
    """A task tagged with three capabilities is ONE attempt. Summing per-capability
    rows would call it three, inflating attempts and skewing the win rate and the
    average spend that routing is based on."""
    t = TaskSpec(job_id=job.id, subject="a", description="d",
                 capabilities=["edit", "python", "mechanical"])
    led.add_task(t)
    led.record_report(
        WorkerReport(task_id=t.id, worker_id="w", state=TaskState.DONE,
                     verdict=Verdict.PASS, usd_micros=500, seconds=12)
    )
    s = led.worker_stats(["edit", "python", "mechanical"])[0]
    assert s["attempts"] == 1
    assert s["win_rate"] == 1.0
    assert s["avg_usd_micros"] == 500

    # The naive merge this replaces would have produced 3.
    naive = sum(
        row["attempts"]
        for cap in ("edit", "python", "mechanical")
        for row in led.capability_stats(cap)
        if row["worker_id"] == "w"
    )
    assert naive == 3


def test_worker_stats_still_averages_across_genuinely_distinct_tasks(led, job):
    a = TaskSpec(job_id=job.id, subject="a", description="d", capabilities=["edit"])
    b = TaskSpec(job_id=job.id, subject="b", description="d", capabilities=["python"])
    led.add_task(a)
    led.add_task(b)
    led.record_report(WorkerReport(task_id=a.id, worker_id="w", state=TaskState.DONE,
                                   verdict=Verdict.PASS, usd_micros=200))
    led.record_report(WorkerReport(task_id=b.id, worker_id="w", state=TaskState.FAILED,
                                   usd_micros=400))
    s = led.worker_stats(["edit", "python"])[0]
    assert s["attempts"] == 2
    assert s["win_rate"] == 0.5
    assert s["avg_usd_micros"] == 300


def test_worker_stats_with_no_capabilities_returns_empty(led):
    assert led.worker_stats([]) == []


def test_worker_stats_respects_check_in_semantics(led, job):
    """Same rule as capability_stats: a BLOCKED check-in is not a failure."""
    t = TaskSpec(job_id=job.id, subject="a", description="d", capabilities=["edit", "python"])
    led.add_task(t)
    led.record_report(WorkerReport(task_id=t.id, worker_id="w", state=TaskState.BLOCKED))
    led.record_report(WorkerReport(task_id=t.id, worker_id="w", state=TaskState.DONE,
                                   verdict=Verdict.PASS))
    s = led.worker_stats(["edit", "python"])[0]
    assert s["attempts"] == 1
    assert s["win_rate"] == 1.0
