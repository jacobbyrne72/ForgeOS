"""What this session is costing, while it is still costing it.

ForgeOS records spend precisely and reports it afterwards. `forge receipts`
answers "what did that cost" perfectly — once the job is over, which is exactly
too late to do anything about it. The moment a budget matters is the moment
before it is spent.

Two gaps, both confirmed absent before this module and both named independently
by people reviewing other harnesses as the thing they wished every agent had:

  1. LIVE SESSION LINE. Cost so far, budget remaining, model, reasoning effort,
     in one place, during the run. A cost figure you can only read afterwards
     cannot change a decision.

  2. n-OF-m PROGRESS. A fan-out of 117 parallel calls with no output until the
     last one lands is indistinguishable from a hang, and the natural response
     to a suspected hang is to kill it — throwing away every call already paid
     for. Progress is not decoration here; it prevents a specific waste.

Rendering only. This module computes and formats; it never spends, never
decides, and never touches the ledger it reads from. A status display that can
influence execution is one nobody can trust to be honest about it.

Money is integer microdollars throughout, matching the ledger. Floats are
introduced only in the final formatted string.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# Bar width tuned for an 80-column terminal alongside the rest of a status line.
_BAR_WIDTH = 20

# Past this fraction of budget, the line says so. Not a limit -- the governor
# owns limits -- just the point where a person would want to know.
WARN_FRACTION = 0.75
CRITICAL_FRACTION = 0.90


def _usd(micros: int) -> str:
    """Microdollars to a string that never renders real spend as free.

    Below a cent, widen the precision. `savings.py` learned this the hard way:
    at two decimals every sub-cent call printed "$0.00", and a receipt reporting
    real spend as free is worse than one reporting nothing.
    """
    dollars = micros / 1_000_000
    if micros and abs(dollars) < 0.01:
        return f"${dollars:.4f}"
    return f"${dollars:,.2f}"


def _bar(fraction: float, width: int = _BAR_WIDTH) -> str:
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    return "#" * filled + "." * (width - filled)


@dataclass
class Progress:
    """n-of-m for a fan-out, with an ETA that refuses to guess early.

    The ETA is None until at least two items have finished. One sample is not a
    rate, and a wildly wrong "3 seconds remaining" is worse than no estimate --
    it invites someone to wait for something that will take ten minutes.
    """

    total: int
    done: int = 0
    failed: int = 0
    started_at: float = field(default_factory=time.monotonic)
    label: str = ""

    def tick(self, *, ok: bool = True) -> None:
        self.done += 1
        if not ok:
            self.failed += 1

    @property
    def fraction(self) -> float:
        return self.done / self.total if self.total else 1.0

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def eta_seconds(self) -> float | None:
        if self.done < 2 or self.done >= self.total:
            return None
        rate = self.done / self.elapsed if self.elapsed > 0 else 0.0
        if rate <= 0:
            return None
        return (self.total - self.done) / rate

    def render(self) -> str:
        eta = self.eta_seconds
        eta_text = f"  eta {eta:.0f}s" if eta is not None else ""
        failed = f"  {self.failed} failed" if self.failed else ""
        label = f"{self.label} " if self.label else ""
        return (f"{label}[{_bar(self.fraction)}] {self.done}/{self.total}"
                f"{failed}{eta_text}")


@dataclass
class SessionStatus:
    """One line describing the run, cheap enough to redraw constantly.

    `spent_micros` is MEASURED spend only -- provider-billed calls. A modelled
    tier prior for a flat-rate seat is real information but it is not money
    anyone was invoiced, and mixing the two is the exact overstatement
    `ledger.job_spend_split` exists to prevent. Modelled spend is carried
    separately and labelled when shown.
    """

    model: str = ""
    reasoning_effort: str = ""
    spent_micros: int = 0
    modelled_micros: int = 0
    budget_micros: int = 0
    tasks_done: int = 0
    tasks_total: int = 0
    cache_hit_pct: float | None = None
    started_at: float = field(default_factory=time.monotonic)
    progress: Progress | None = None

    @property
    def remaining_micros(self) -> int:
        """Never negative. A budget cannot be less than exhausted, and a
        negative remaining figure reads as a refund."""
        return max(0, self.budget_micros - self.spent_micros)

    @property
    def burn_fraction(self) -> float:
        return self.spent_micros / self.budget_micros if self.budget_micros else 0.0

    @property
    def level(self) -> str:
        if not self.budget_micros:
            return "ok"
        if self.burn_fraction >= CRITICAL_FRACTION:
            return "critical"
        if self.burn_fraction >= WARN_FRACTION:
            return "warn"
        return "ok"

    def render(self) -> str:
        """The line itself. Plain ASCII: this goes to terminals that mangle
        anything else, and a status line that renders as mojibake is worse than
        none."""
        parts: list[str] = []
        if self.model:
            parts.append(self.model)
        if self.reasoning_effort:
            parts.append(f"effort={self.reasoning_effort}")

        if self.budget_micros:
            marker = {"critical": " !!", "warn": " !", "ok": ""}[self.level]
            parts.append(
                f"[{_bar(self.burn_fraction)}] {_usd(self.spent_micros)}"
                f"/{_usd(self.budget_micros)}{marker}"
            )
        else:
            parts.append(f"spent {_usd(self.spent_micros)}")

        if self.modelled_micros:
            # Named as modelled every time it appears. An unlabelled number
            # beside a measured one reads as measured.
            parts.append(f"+{_usd(self.modelled_micros)} modelled")
        if self.tasks_total:
            parts.append(f"tasks {self.tasks_done}/{self.tasks_total}")
        if self.cache_hit_pct is not None:
            parts.append(f"cache {self.cache_hit_pct:.0f}%")
        parts.append(f"{self.elapsed_text}")
        line = "  ".join(parts)
        if self.progress is not None:
            line += "\n  " + self.progress.render()
        return line

    @property
    def elapsed_text(self) -> str:
        seconds = time.monotonic() - self.started_at
        if seconds < 60:
            return f"{seconds:.0f}s"
        return f"{seconds / 60:.0f}m{seconds % 60:02.0f}s"


def from_ledger(ledger, job_id: str, *, model: str = "", reasoning_effort: str = "",
                budget_micros: int = 0) -> SessionStatus:
    """Build a status from the ledger, measured and modelled kept apart.

    Reads only. Never raises: a status line that can break a run is worse than
    no status line, and this exists to be redrawn constantly.
    """
    measured = modelled = 0
    try:
        measured, modelled = ledger.job_spend_split(job_id)
    except Exception:
        try:
            measured = ledger.job_spend_micros(job_id, measured_only=True)
        except Exception:
            measured = 0
    return SessionStatus(
        model=model, reasoning_effort=reasoning_effort,
        spent_micros=measured, modelled_micros=modelled,
        budget_micros=budget_micros,
    )
