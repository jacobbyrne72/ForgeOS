"""Latency attribution tests.

Every timestamp here is passed explicitly (`started_at=`/`at=`, or a
monkeypatched `now`) rather than produced by sleeping — a test suite that
burns real wall clock to test a wall-clock module is exactly the kind of
waste this module exists to catch.
"""

from __future__ import annotations

import pytest

import hive.core.timing as timing
from hive.core.timing import Attribution, Phase, PhaseTotal, Span, SpanStore, TaskTotal

T0 = 1_700_000_000.0


@pytest.fixture()
def store():
    s = SpanStore(":memory:")
    yield s
    s.close()


def _phase_seconds(attribution: Attribution, phase: Phase) -> float:
    for pt in attribution.by_phase:
        if pt.phase is phase:
            return pt.seconds
    raise AssertionError(f"phase {phase} not present in by_phase: {attribution.by_phase}")


# --------------------------------------------------------------- start/end


def test_start_returns_open_span_with_explicit_timestamp(store):
    span = store.start("job1", Phase.TOOL_CALL, task_id="t1", detail="grep", started_at=T0)
    assert span.job_id == "job1"
    assert span.task_id == "t1"
    assert span.phase is Phase.TOOL_CALL
    assert span.detail == "grep"
    assert span.started_at == T0
    assert span.ended_at is None
    assert span.open is True


def test_end_closes_span_and_removes_from_open_spans(store):
    span = store.start("job1", Phase.TOOL_CALL, started_at=T0)
    assert len(store.open_spans("job1")) == 1

    closed = store.end(span.id, at=T0 + 4.0)
    assert closed.ended_at == T0 + 4.0
    assert closed.open is False
    assert closed.duration == pytest.approx(4.0)
    assert store.open_spans("job1") == []


def test_spans_for_job_ordered_by_start(store):
    store.start("job1", Phase.TEST_RUN, started_at=T0 + 10.0)
    store.start("job1", Phase.TOOL_CALL, started_at=T0)
    spans = store.spans_for_job("job1")
    assert [s.phase for s in spans] == [Phase.TOOL_CALL, Phase.TEST_RUN]


def test_open_spans_filters_by_job(store):
    store.start("job1", Phase.TOOL_CALL, started_at=T0)
    store.start("job2", Phase.TEST_RUN, started_at=T0)
    assert len(store.open_spans("job1")) == 1
    assert len(store.open_spans("job2")) == 1
    assert len(store.open_spans()) == 2


# --------------------------------------------------------------- measure()


def test_measure_context_manager_closes_span_normally(store):
    with store.measure("job1", Phase.MODEL_REASONING, task_id="t1") as span:
        span_id = span.id
        assert span.open is True

    spans = store.spans_for_job("job1")
    assert len(spans) == 1
    assert spans[0].id == span_id
    assert spans[0].open is False
    assert store.open_spans("job1") == []


def test_measure_context_manager_closes_span_on_exception(store):
    captured_id = None
    with pytest.raises(RuntimeError, match="boom"):
        with store.measure("job1", Phase.TOOL_CALL) as span:
            captured_id = span.id
            raise RuntimeError("boom")

    # The exception propagated (asserted above by pytest.raises), but the
    # span must not still be open — losing that timing data silently is
    # exactly the failure mode this module exists to prevent.
    spans = store.spans_for_job("job1")
    assert len(spans) == 1
    assert spans[0].id == captured_id
    assert spans[0].open is False
    assert store.open_spans("job1") == []


# --------------------------------------------------------------- attribution


def test_phase_totals_and_bottleneck(store):
    s1 = store.start("job1", Phase.TOOL_CALL, started_at=T0)
    store.end(s1.id, at=T0 + 2.0)
    s2 = store.start("job1", Phase.TEST_RUN, started_at=T0 + 2.0)
    store.end(s2.id, at=T0 + 10.0)

    attr = store.attribution("job1")

    assert attr.wall_clock == pytest.approx(10.0)
    assert attr.measured == pytest.approx(10.0)
    assert attr.unmeasured == pytest.approx(0.0)
    assert attr.parallelism_ratio == pytest.approx(1.0)
    assert attr.bottleneck is Phase.TEST_RUN

    assert _phase_seconds(attr, Phase.TOOL_CALL) == pytest.approx(2.0)
    assert _phase_seconds(attr, Phase.TEST_RUN) == pytest.approx(8.0)

    pcts = {pt.phase: pt.pct_of_measured for pt in attr.by_phase}
    assert pcts[Phase.TOOL_CALL] == pytest.approx(20.0)
    assert pcts[Phase.TEST_RUN] == pytest.approx(80.0)
    assert sum(pcts.values()) == pytest.approx(100.0)


def test_unmeasured_time_reported_for_gap_between_spans(store):
    # 0-3 instrumented, 3-7 is a genuine gap, 7-10 instrumented.
    s1 = store.start("job1", Phase.ENV_STARTUP, started_at=T0)
    store.end(s1.id, at=T0 + 3.0)
    s2 = store.start("job1", Phase.TEST_RUN, started_at=T0 + 7.0)
    store.end(s2.id, at=T0 + 10.0)

    attr = store.attribution("job1")

    assert attr.wall_clock == pytest.approx(10.0)
    assert attr.measured == pytest.approx(6.0)
    assert attr.unmeasured == pytest.approx(4.0)
    assert attr.parallelism_ratio == pytest.approx(0.6)


def test_overlapping_spans_produce_parallelism_ratio_above_one_unclamped(store):
    # Two workers, fully overlapping 10-second spans: 20s of work in 10s of
    # wall clock. The ratio must show 2.0, not be capped at 1.0.
    s1 = store.start("job1", Phase.TOOL_CALL, task_id="worker-a", started_at=T0)
    store.end(s1.id, at=T0 + 10.0)
    s2 = store.start("job1", Phase.TOOL_CALL, task_id="worker-b", started_at=T0)
    store.end(s2.id, at=T0 + 10.0)

    attr = store.attribution("job1")

    assert attr.wall_clock == pytest.approx(10.0)
    assert attr.measured == pytest.approx(20.0)
    assert attr.parallelism_ratio == pytest.approx(2.0)
    # Full overlap covers the entire wall-clock window twice over — there is
    # no leftover gap, so unmeasured floors at zero rather than going negative.
    assert attr.unmeasured == pytest.approx(0.0)


def test_open_span_counted_as_elapsed_so_far(store, monkeypatch):
    monkeypatch.setattr(timing, "now", lambda: T0 + 5.0)

    store.start("job1", Phase.ENV_STARTUP, started_at=T0)  # left open

    attr = store.attribution("job1")

    assert attr.wall_clock == pytest.approx(5.0)
    assert attr.measured == pytest.approx(5.0)
    assert attr.unmeasured == pytest.approx(0.0)
    assert _phase_seconds(attr, Phase.ENV_STARTUP) == pytest.approx(5.0)


def test_span_duration_property_elapsed_so_far_when_open(store, monkeypatch):
    span = store.start("job1", Phase.QUEUE_WAIT, started_at=T0)
    monkeypatch.setattr(timing, "now", lambda: T0 + 7.5)
    assert span.open is True
    assert span.duration == pytest.approx(7.5)


def test_empty_store_returns_zeros_without_dividing_by_zero(store):
    attr = store.attribution("no-such-job")

    assert attr.wall_clock == 0.0
    assert attr.measured == 0.0
    assert attr.unmeasured == 0.0
    assert attr.parallelism_ratio == 0.0
    assert attr.bottleneck is None
    assert attr.by_phase == []


# --------------------------------------------------------------- slowest_tasks


def test_slowest_tasks_ranks_by_total_duration(store):
    # task A: 5s + 7s = 12s across two spans
    a1 = store.start("job1", Phase.TOOL_CALL, task_id="A", started_at=T0)
    store.end(a1.id, at=T0 + 5.0)
    a2 = store.start("job1", Phase.MODEL_REASONING, task_id="A", started_at=T0 + 5.0)
    store.end(a2.id, at=T0 + 12.0)

    # task B: 3s
    b1 = store.start("job1", Phase.TOOL_CALL, task_id="B", started_at=T0)
    store.end(b1.id, at=T0 + 3.0)

    # job-level span with no task_id — must not be attributable to a task
    q = store.start("job1", Phase.QUEUE_WAIT, started_at=T0)
    store.end(q.id, at=T0 + 100.0)

    ranked = store.slowest_tasks("job1", n=5)

    assert [t.task_id for t in ranked] == ["A", "B"]
    assert ranked[0].seconds == pytest.approx(12.0)
    assert ranked[1].seconds == pytest.approx(3.0)


def test_slowest_tasks_respects_n(store):
    for i, dur in enumerate([1.0, 5.0, 3.0]):
        sp = store.start("job1", Phase.TOOL_CALL, task_id=f"t{i}", started_at=T0)
        store.end(sp.id, at=T0 + dur)

    ranked = store.slowest_tasks("job1", n=1)
    assert len(ranked) == 1
    assert ranked[0].task_id == "t1"
    assert ranked[0].seconds == pytest.approx(5.0)


def test_slowest_tasks_empty_job_returns_empty_list(store):
    assert store.slowest_tasks("no-such-job", n=5) == []


# --------------------------------------------------------------- model sanity


def test_phase_total_and_task_total_are_plain_models():
    pt = PhaseTotal(phase=Phase.TOOL_CALL, seconds=1.0, pct_of_measured=50.0)
    assert pt.phase is Phase.TOOL_CALL
    tt = TaskTotal(task_id="x", seconds=2.0)
    assert tt.task_id == "x"


def test_span_model_open_property_true_until_ended():
    span = Span(job_id="j", phase=Phase.MANAGER, started_at=T0)
    assert span.open is True
    span2 = Span(job_id="j", phase=Phase.MANAGER, started_at=T0, ended_at=T0 + 1.0)
    assert span2.open is False
    assert span2.duration == pytest.approx(1.0)
