"""Degradation recorder tests.

The load-bearing property: a subsystem that catches `except Exception` and
carries on must not carry on *silently*. `record_degradation` is the one
mechanism forgeos has for making that visible, so it must (a) actually capture
enough for an operator to act on, (b) never grow without bound under a flood,
(c) never itself become the failure, and (d) actually get called from a real
swallow site rather than just existing unused.
"""

from __future__ import annotations

import threading

import pytest

from forgeos import diagnostics
from forgeos.diagnostics import (
    _MAX_DEGRADATIONS,
    clear,
    degradations,
    record_degradation,
)


@pytest.fixture(autouse=True)
def _clean_state():
    """Every test starts and ends with an empty recorder -- it is module-global
    state, and a leftover degradation from one test must not leak into the next."""
    clear()
    yield
    clear()


# --------------------------------------------------------------------- basics


def test_record_captures_subsystem_consequence_and_exception_text():
    record_degradation(
        "resources", "psutil hardware detection failed", ValueError("bad value"),
        consequence="pool sizing may be wrong for this machine",
    )
    degs = degradations()
    assert len(degs) == 1
    d = degs[0]
    assert d.subsystem == "resources"
    assert d.what_failed == "psutil hardware detection failed"
    assert d.consequence == "pool sizing may be wrong for this machine"
    assert "ValueError" in d.reason
    assert "bad value" in d.reason
    assert d.count == 1


def test_record_without_an_exception_still_captures_subsystem_and_consequence():
    record_degradation("adapter_discovery", "entry-point scan failed", consequence="fleet may be smaller")
    degs = degradations()
    assert len(degs) == 1
    assert degs[0].reason == ""


def test_long_exception_text_is_truncated():
    huge = RuntimeError("x" * 5000)
    record_degradation("resources", "disk usage probe failed", huge, consequence="disk_free_gib is 0")
    d = degradations()[0]
    assert len(d.reason) <= diagnostics._REASON_MAX_LEN
    assert d.reason.endswith("...")


def test_clear_empties_the_recorder():
    record_degradation("x", "y", consequence="z")
    assert degradations()
    clear()
    assert degradations() == []


# ---------------------------------------------------------------- deduplication


def test_repeat_of_the_same_degradation_increments_count_not_the_list():
    for _ in range(5):
        record_degradation(
            "resources", "psutil hardware detection failed", OSError("no permission"),
            consequence="pool sizing may be wrong for this machine",
        )
    degs = degradations()
    assert len(degs) == 1
    assert degs[0].count == 5


def test_different_what_failed_in_the_same_subsystem_are_separate_entries():
    record_degradation("resources", "psutil hardware detection failed", consequence="a")
    record_degradation("resources", "disk usage probe failed", consequence="b")
    degs = degradations()
    assert len(degs) == 2
    assert {d.what_failed for d in degs} == {"psutil hardware detection failed", "disk usage probe failed"}


def test_same_what_failed_in_different_subsystems_are_separate_entries():
    record_degradation("resources", "probe failed", consequence="a")
    record_degradation("adapter_discovery", "probe failed", consequence="b")
    assert len(degradations()) == 2


# ------------------------------------------------------------------- bounding


def test_ring_buffer_bound_holds_under_a_flood_of_distinct_degradations():
    for i in range(_MAX_DEGRADATIONS * 3):
        record_degradation("flood", f"failure {i}", consequence="c")
    assert len(degradations()) <= _MAX_DEGRADATIONS


def test_flood_of_repeats_does_not_grow_the_list_at_all():
    """The opposite failure mode: identical degradations must dedupe, not just
    get capped -- otherwise a tight failure loop evicts everything else."""
    for _ in range(_MAX_DEGRADATIONS * 3):
        record_degradation("hot_loop", "same failure every time", consequence="c")
    degs = degradations()
    assert len(degs) == 1
    assert degs[0].count == _MAX_DEGRADATIONS * 3


# -------------------------------------------------------------- thread safety


def test_concurrent_writers_do_not_lose_updates_or_raise():
    errors: list[BaseException] = []

    def _write(worker_id: int) -> None:
        try:
            for i in range(50):
                record_degradation(
                    "concurrent", f"failure {i % 5}", consequence=f"from worker {worker_id}",
                )
        except BaseException as exc:  # pragma: no cover - assertion happens below
            errors.append(exc)

    threads = [threading.Thread(target=_write, args=(w,)) for w in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors
    degs = {d.what_failed: d for d in degradations() if d.subsystem == "concurrent"}
    assert set(degs) == {f"failure {i}" for i in range(5)}
    # 8 workers x 50 writes x 1/5 land on each key = 80 per key, and no
    # increment may be lost to a race.
    assert sum(d.count for d in degs.values()) == 8 * 50


# ------------------------------------------------------------- never-raises


def test_recorder_broken_internally_never_raises_into_the_caller(monkeypatch):
    """If the recorder's own bookkeeping breaks, that must not become the
    caller's exception -- the caller is already mid-`except Exception`."""

    class _ExplodingDict(dict):
        def get(self, *a, **kw):
            raise RuntimeError("diagnostics storage is broken")

    monkeypatch.setattr(diagnostics, "_degradations", _ExplodingDict())
    record_degradation("anything", "anything", ValueError("x"), consequence="anything")  # must not raise


def test_recorder_survives_a_hostile_exception_str(monkeypatch):
    class _Hostile(Exception):
        def __str__(self):
            raise RuntimeError("even __str__ is broken")

    record_degradation("resources", "probe failed", _Hostile(), consequence="c")  # must not raise


# -------------------------------------------------------------------- doctor


def test_doctor_prints_an_honest_none_line_when_nothing_degraded(capsys):
    from forgeos import __main__ as cli

    cli._print_degradations()
    out = capsys.readouterr().out
    assert "Degradations" in out
    assert "none recorded" in out


def test_doctor_prints_subsystem_consequence_and_count(capsys):
    from forgeos import __main__ as cli

    record_degradation("resources", "psutil hardware detection failed", consequence="pool sizing may be wrong")
    record_degradation("resources", "psutil hardware detection failed", consequence="pool sizing may be wrong")
    record_degradation("adapter_discovery", "entry-point scan failed", consequence="fleet may be smaller")

    cli._print_degradations()
    out = capsys.readouterr().out
    assert "[resources] pool sizing may be wrong (x2)" in out
    assert "[adapter_discovery] fleet may be smaller" in out
    assert "(x1)" not in out  # count of 1 is not worth calling out


# --------------------------------------------------------- real swallow site


def test_auto_discover_entry_point_scan_failure_is_recorded_not_swallowed(monkeypatch):
    """The concrete failure the whole module exists for: entry-point discovery
    raising must no longer disappear into a bare `except: pass`."""
    import importlib.metadata

    from forgeos.adapter.auto_discover import _scan_entry_points

    def _raise(**kwargs):
        raise RuntimeError("corrupt entry-points metadata")

    monkeypatch.setattr(importlib.metadata, "entry_points", _raise)

    results: list = []
    seen: set = set()
    _scan_entry_points(results, seen)  # must not raise

    assert results == []  # control flow unchanged: discovery still yields nothing
    degs = degradations()
    assert len(degs) == 1
    assert degs[0].subsystem == "adapter_discovery"
    assert "corrupt entry-points metadata" in degs[0].reason
    assert "smaller" in degs[0].consequence
