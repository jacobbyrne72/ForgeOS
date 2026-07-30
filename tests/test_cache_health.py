"""Cache-hit regression detection (`Ledger.cache_health`).

The failure this guards against is silent: hive assembles every prompt
prefix-then-tail so a provider can serve the cached portion at roughly a
tenth the price. If anything perturbs the prefix, the cache-hit rate drops
to zero with no error and no test failure -- the bill just gets bigger.
`cache_stats` already reports today's number; these tests prove
`cache_health` can tell a REGRESSION (it used to hit, now it does not) apart
from a provider that never cached in the first place, and refuses to guess
off too few calls.

No LLM calls, no network -- spend rows are seeded directly, exactly the way
`Gateway.complete` and `Forge._run_task` record them.
"""

from __future__ import annotations

import pytest

from hive.contracts import JobSpec
from hive.ledger import (
    CACHE_HEALTH_FLOOR_PCT,
    CACHE_HEALTH_MIN_CALLS,
    CACHE_HEALTH_RECENT_CALLS,
    CACHE_HEALTH_REGRESSION_RATIO,
    Ledger,
)


@pytest.fixture()
def led():
    ledger = Ledger(":memory:")
    yield ledger
    ledger.close()


@pytest.fixture()
def job(led):
    j = JobSpec(objective="prove cache_health tells regression from no-support", cwd=".")
    led.open_job(j)
    return j


def _spend(led, job_id, worker="w1", model="m", n=1, *, tokens_in, tokens_cached_in, kind="call"):
    """Seed `n` identical spend rows -- one real call's shape, repeated."""
    for _ in range(n):
        led.record_spend(job_id, worker, model, 10, tokens_in=tokens_in,
                         tokens_cached_in=tokens_cached_in, kind=kind)


# ------------------------------------------------------- minimum-calls gate


def test_no_spend_at_all_is_insufficient_data_not_a_false_ok(led, job):
    health = led.cache_health(job.id)
    assert health == {"status": "insufficient_data", "group_by": "worker", "groups": []}


def test_fewer_than_min_calls_is_insufficient_data_even_if_it_looks_healthy(led, job):
    """Two calls is noise, not a rate -- a real cache regression must never be
    inferred (or ruled out) from a handful of calls."""
    assert CACHE_HEALTH_MIN_CALLS > 2
    _spend(led, job.id, n=2, tokens_in=10, tokens_cached_in=90)  # looks like 90% hit rate
    health = led.cache_health(job.id)
    assert health["status"] == "insufficient_data"
    assert health["groups"][0]["status"] == "insufficient_data"


def test_baseline_below_minimum_is_insufficient_even_with_a_full_recent_window(led, job):
    """Enough TOTAL calls to fill the recent window, but nothing left over for
    a baseline -- there is nothing to compare "recent" against yet."""
    total = CACHE_HEALTH_RECENT_CALLS + (CACHE_HEALTH_MIN_CALLS - 1)
    _spend(led, job.id, n=total, tokens_in=10, tokens_cached_in=90)
    health = led.cache_health(job.id)
    assert health["groups"][0]["baseline_calls"] == CACHE_HEALTH_MIN_CALLS - 1
    assert health["status"] == "insufficient_data"


# --------------------------------------------------- the three real verdicts


def test_provider_that_never_cached_is_no_cache_support_not_regressed(led, job):
    """The false alarm this whole feature exists to prevent: a provider with
    zero cache support, at every point in its history, must never read as a
    regression -- there was no working state to regress FROM."""
    n = CACHE_HEALTH_RECENT_CALLS + CACHE_HEALTH_MIN_CALLS
    _spend(led, job.id, n=n, tokens_in=100, tokens_cached_in=0)
    health = led.cache_health(job.id)
    assert health["status"] == "no_cache_support"
    g = health["groups"][0]
    assert g["status"] == "no_cache_support"
    assert g["baseline_hit_pct"] <= CACHE_HEALTH_FLOOR_PCT
    assert g["status"] != "regressed"


def test_steady_healthy_cache_is_ok(led, job):
    n = CACHE_HEALTH_RECENT_CALLS + CACHE_HEALTH_MIN_CALLS
    _spend(led, job.id, n=n, tokens_in=10, tokens_cached_in=90)  # 90% throughout
    health = led.cache_health(job.id)
    assert health["status"] == "ok"
    g = health["groups"][0]
    assert g["baseline_hit_pct"] == pytest.approx(90.0)
    assert g["recent_hit_pct"] == pytest.approx(90.0)


def test_regression_is_detected_when_recent_hit_rate_collapses(led, job):
    """It used to hit, and now it does not -- the actionable case."""
    baseline_n = CACHE_HEALTH_MIN_CALLS + 2
    _spend(led, job.id, n=baseline_n, tokens_in=10, tokens_cached_in=90)  # healthy baseline
    _spend(led, job.id, n=CACHE_HEALTH_RECENT_CALLS, tokens_in=100, tokens_cached_in=0)  # cache went dead

    health = led.cache_health(job.id)
    assert health["status"] == "regressed"
    g = health["groups"][0]
    assert g["status"] == "regressed"
    assert g["baseline_calls"] == baseline_n
    assert g["recent_calls"] == CACHE_HEALTH_RECENT_CALLS
    assert g["baseline_hit_pct"] == pytest.approx(90.0)
    assert g["recent_hit_pct"] == pytest.approx(0.0)


def test_a_partial_drop_within_the_regression_ratio_still_reads_ok(led, job):
    """Recent falling a little short of baseline is ordinary noise, not a
    regression -- only a fall below CACHE_HEALTH_REGRESSION_RATIO counts."""
    baseline_n = CACHE_HEALTH_MIN_CALLS + 2
    _spend(led, job.id, n=baseline_n, tokens_in=20, tokens_cached_in=80)  # 80%
    recent_pct = 80.0 * CACHE_HEALTH_REGRESSION_RATIO + 5  # comfortably above the regression floor
    cached = int(round(recent_pct))
    _spend(led, job.id, n=CACHE_HEALTH_RECENT_CALLS, tokens_in=100 - cached, tokens_cached_in=cached)

    health = led.cache_health(job.id)
    assert health["groups"][0]["status"] == "ok"


# ------------------------------------------------------------------ grouping


def test_group_by_must_be_worker_or_model(led, job):
    with pytest.raises(ValueError):
        led.cache_health(job.id, group_by="nope")


def test_group_by_worker_is_the_default_and_separates_workers(led, job):
    n = CACHE_HEALTH_RECENT_CALLS + CACHE_HEALTH_MIN_CALLS
    _spend(led, job.id, worker="alice", n=n, tokens_in=10, tokens_cached_in=90)
    _spend(led, job.id, worker="bob", n=n, tokens_in=100, tokens_cached_in=0)

    health = led.cache_health(job.id)
    assert health["group_by"] == "worker"
    by_group = {g["group"]: g for g in health["groups"]}
    assert by_group["alice"]["status"] == "ok"
    assert by_group["bob"]["status"] == "no_cache_support"
    # One worker's dead cache is real signal, but it must not drown out the
    # other worker's healthy one in the per-group breakdown.
    assert set(by_group) == {"alice", "bob"}


def test_group_by_model_separates_models_a_worker_rotated_through(led, job):
    """A single worker that switches models mid-job must not have a
    model-specific regression diluted by grouping on worker alone."""
    n = CACHE_HEALTH_RECENT_CALLS + CACHE_HEALTH_MIN_CALLS
    _spend(led, job.id, worker="w1", model="model-a", n=n, tokens_in=10, tokens_cached_in=90)
    _spend(led, job.id, worker="w1", model="model-b", n=n, tokens_in=100, tokens_cached_in=0)

    by_worker = led.cache_health(job.id, group_by="worker")
    assert len(by_worker["groups"]) == 1  # both models blended under one worker

    by_model = led.cache_health(job.id, group_by="model")
    assert {g["group"] for g in by_model["groups"]} == {"model-a", "model-b"}


# ------------------------------------------------------------ overall rollup


def test_overall_status_is_regressed_if_any_single_group_regressed(led, job):
    baseline_n = CACHE_HEALTH_MIN_CALLS + 2
    _spend(led, job.id, worker="healthy", n=CACHE_HEALTH_RECENT_CALLS + CACHE_HEALTH_MIN_CALLS,
          tokens_in=10, tokens_cached_in=90)
    _spend(led, job.id, worker="broken", n=baseline_n, tokens_in=10, tokens_cached_in=90)
    _spend(led, job.id, worker="broken", n=CACHE_HEALTH_RECENT_CALLS, tokens_in=100, tokens_cached_in=0)

    health = led.cache_health(job.id)
    assert health["status"] == "regressed"


def test_job_id_filters_to_that_job_only(led):
    j1 = JobSpec(objective="job one", cwd=".")
    j2 = JobSpec(objective="job two", cwd=".")
    led.open_job(j1)
    led.open_job(j2)
    n = CACHE_HEALTH_RECENT_CALLS + CACHE_HEALTH_MIN_CALLS
    _spend(led, j1.id, n=n, tokens_in=10, tokens_cached_in=90)
    _spend(led, j2.id, n=n, tokens_in=100, tokens_cached_in=0)

    assert led.cache_health(j1.id)["status"] == "ok"
    assert led.cache_health(j2.id)["status"] == "no_cache_support"


# ----------------------------------------------------- estimate rows excluded


def test_estimate_kind_rows_do_not_influence_the_verdict(led, job):
    """`kind='estimate'` rows are the tier-prior charge for an unmetered
    subscription worker (hive/forge.py `_run_task`) -- they never carry real
    provider usage and must not drag a genuine cache signal toward false
    alarm or false comfort."""
    n = CACHE_HEALTH_RECENT_CALLS + CACHE_HEALTH_MIN_CALLS
    _spend(led, job.id, n=n, tokens_in=10, tokens_cached_in=90)  # real, healthy
    # A pile of zero-cache "estimate" rows that would read as a regression if
    # they were ever mixed into the real signal.
    _spend(led, job.id, n=50, tokens_in=1000, tokens_cached_in=0, kind="estimate")

    health = led.cache_health(job.id)
    assert health["status"] == "ok"
    assert health["groups"][0]["baseline_calls"] + health["groups"][0]["recent_calls"] == n


# ------------------------------------------------------------- dashboard wiring


def test_dashboard_economy_endpoint_reports_cache_health_zeroed_not_absent(tmp_path):
    from hive.dashboard.app import LEDGER_DB, create_app
    from fastapi.testclient import TestClient

    state_dir = tmp_path / "state"
    app = create_app(state_dir)
    client = TestClient(app)

    body = client.get("/api/economy").json()
    assert "cache_health" in body
    assert body["cache_health"]["status"] == "insufficient_data"
    assert body["cache_health"]["groups"] == []
    # The ledger the app opened is the same file a manager process would write to.
    assert (state_dir / LEDGER_DB).exists()


def test_dashboard_economy_endpoint_reports_a_seeded_regression(tmp_path):
    from hive.dashboard.app import LEDGER_DB, create_app
    from fastapi.testclient import TestClient

    state_dir = tmp_path / "state"
    ledger = Ledger(state_dir / LEDGER_DB)
    try:
        j = JobSpec(objective="seed a regression for the dashboard", cwd=str(state_dir))
        ledger.open_job(j)
        baseline_n = CACHE_HEALTH_MIN_CALLS + 2
        _spend(ledger, j.id, n=baseline_n, tokens_in=10, tokens_cached_in=90)
        _spend(ledger, j.id, n=CACHE_HEALTH_RECENT_CALLS, tokens_in=100, tokens_cached_in=0)
    finally:
        ledger.close()

    app = create_app(state_dir)
    client = TestClient(app)
    body = client.get("/api/economy", params={"job_id": j.id}).json()
    assert body["cache_health"]["status"] == "regressed"
    assert body["cache_health"]["groups"][0]["status"] == "regressed"
