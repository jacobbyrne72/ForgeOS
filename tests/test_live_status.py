"""Live session cost, and progress during a fan-out.

Both gaps were confirmed absent before this module. The second one prevents a
specific waste: a fan-out that prints nothing until the last call lands is
indistinguishable from a hang, and the natural response to a suspected hang is
to kill it -- throwing away every call already paid for.
"""

from __future__ import annotations

import time

from forgeos.core.live_status import (
    CRITICAL_FRACTION,
    WARN_FRACTION,
    Progress,
    SessionStatus,
    from_ledger,
)


# ------------------------------------------------------------ session line


def test_sub_cent_spend_is_never_rendered_as_free():
    """`savings.py` learned this the hard way: at 2dp every sub-cent call
    printed "$0.00", and a line reporting real spend as free is worse than one
    reporting nothing."""
    line = SessionStatus(spent_micros=4_000).render()
    assert "$0.00 " not in line
    assert "0.004" in line


def test_ordinary_amounts_read_like_money():
    assert "$1.50" in SessionStatus(spent_micros=1_500_000).render()


def test_modelled_spend_is_always_labelled():
    """An unlabelled number beside a measured one reads as measured."""
    line = SessionStatus(spent_micros=800, modelled_micros=60_000).render()
    assert "modelled" in line


def test_no_modelled_spend_means_no_modelled_text():
    assert "modelled" not in SessionStatus(spent_micros=800).render()


def test_the_burn_bar_tracks_the_budget():
    line = SessionStatus(spent_micros=500_000, budget_micros=1_000_000).render()
    assert "#" in line and "." in line


def test_burn_levels_escalate():
    budget = 1_000_000
    assert SessionStatus(spent_micros=10, budget_micros=budget).level == "ok"
    assert SessionStatus(spent_micros=int(budget * WARN_FRACTION),
                         budget_micros=budget).level == "warn"
    assert SessionStatus(spent_micros=int(budget * CRITICAL_FRACTION),
                         budget_micros=budget).level == "critical"


def test_remaining_budget_never_goes_negative():
    """A negative remaining figure reads as a refund."""
    s = SessionStatus(spent_micros=2_000_000, budget_micros=1_000_000)
    assert s.remaining_micros == 0


def test_no_budget_set_still_reports_spend():
    line = SessionStatus(spent_micros=1_234).render()
    assert "spent" in line and "/" not in line.split("spent")[1][:12]


def test_the_line_is_plain_ascii():
    """This goes to terminals that mangle anything else, and a status line
    rendering as mojibake is worse than none."""
    line = SessionStatus(model="deepseek/deepseek-chat", reasoning_effort="none",
                         spent_micros=4_000, modelled_micros=60_000,
                         budget_micros=1_000_000, tasks_done=2, tasks_total=6,
                         cache_hit_pct=41.0).render()
    assert line.isascii()


def test_model_and_effort_are_shown():
    line = SessionStatus(model="deepseek/deepseek-chat", reasoning_effort="low").render()
    assert "deepseek" in line and "effort=low" in line


# ------------------------------------------------------------- progress


def test_progress_counts_and_renders():
    p = Progress(total=10, label="rating chunks")
    for _ in range(3):
        p.tick()
    line = p.render()
    assert "3/10" in line and "rating chunks" in line


def test_failures_are_shown_separately_from_completions():
    p = Progress(total=5)
    p.tick(ok=True)
    p.tick(ok=False)
    assert p.done == 2 and p.failed == 1
    assert "failed" in p.render()


def test_no_failures_means_no_failure_text():
    p = Progress(total=3)
    p.tick()
    assert "failed" not in p.render()


def test_eta_refuses_to_guess_from_one_sample():
    """One sample is not a rate. A wildly wrong "3 seconds remaining" invites
    someone to wait for something that takes ten minutes."""
    p = Progress(total=100)
    assert p.eta_seconds is None
    p.tick()
    assert p.eta_seconds is None


def test_eta_appears_once_there_is_a_rate():
    p = Progress(total=100)
    time.sleep(0.02)
    p.tick()
    p.tick()
    assert p.eta_seconds is not None and p.eta_seconds > 0


def test_a_finished_run_has_no_eta():
    p = Progress(total=2)
    p.tick()
    p.tick()
    assert p.eta_seconds is None


def test_progress_attaches_to_the_session_line():
    s = SessionStatus(spent_micros=100, progress=Progress(total=4, label="fan-out"))
    s.progress.tick()
    assert "fan-out" in s.render() and "1/4" in s.render()


def test_an_empty_fanout_does_not_divide_by_zero():
    assert Progress(total=0).fraction == 1.0


# --------------------------------------------------------------- ledger


class _Ledger:
    def job_spend_split(self, job_id):
        return (788, 60_000)


class _Broken:
    def job_spend_split(self, job_id):
        raise RuntimeError("db gone")

    def job_spend_micros(self, job_id, *, measured_only=False):
        raise RuntimeError("also gone")


def test_from_ledger_keeps_measured_and_modelled_apart():
    s = from_ledger(_Ledger(), "j", model="m", budget_micros=1_000_000)
    assert s.spent_micros == 788 and s.modelled_micros == 60_000


def test_a_broken_ledger_never_breaks_the_status_line():
    """This is redrawn constantly; a status line that can break a run is worse
    than no status line."""
    s = from_ledger(_Broken(), "j")
    assert s.spent_micros == 0
    assert s.render()
