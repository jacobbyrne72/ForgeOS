"""Worker capability registry.

Workers declare what they can do. The ledger records what they actually did.
Ranking blends the two: a declared prior until there is evidence, measured
win-rate and cost once there is. That is the whole "data-driven throttle" idea —
the router stops guessing as soon as the ledger can answer.

Ordering is cheapest-capable-first, never habit-first. A task that a local Rust
binary can finish should never reach a metered model.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

# Below this many attempts, measured stats are noise and the declared prior wins.
MIN_ATTEMPTS_TO_TRUST = 4

# Ranking weights. Capability fit dominates; cost breaks ties. Never let cost
# outrank fit, or the router will keep sending hard work to a model that fails it
# cheaply and then burns three retries.
W_FIT = 1.0
W_WIN = 0.6
W_COST = 0.35
W_LATENCY = 0.1


class CostTier(str, Enum):
    FREE = "free"
    CHEAP = "cheap"
    STANDARD = "standard"
    PREMIUM = "premium"


# Rough per-task priors in microdollars, used only until the ledger has evidence.
TIER_PRIOR_MICROS: dict[CostTier, int] = {
    CostTier.FREE: 0,
    CostTier.CHEAP: 3_000,
    CostTier.STANDARD: 60_000,
    CostTier.PREMIUM: 400_000,
}


class Adapter(str, Enum):
    CLI_TEAM = "cli_team"
    GATEWAY = "gateway"
    OLLAMA = "ollama"


class WorkerProfile(BaseModel):
    """One routable worker: an adapter plus a pinned configuration.

    Pinned on purpose. Effort is not switched mid-task — the router sends work to
    a *different* profile instead, which is what makes the pool idea work.
    """

    worker_id: str
    adapter: Adapter
    model: str = ""
    # `agent_type` is hive's ROLE vocabulary (executor, architect, explore...).
    # `vendor` is which backing CLI actually runs it (claude, codex, gemini...).
    # These are different namespaces and conflating them is a real bug: the omc
    # team runtime's `--agent` flag accepts only vendor names and rejects roles, so
    # passing `agent_type` there fails at spawn time with a confusing error.
    agent_type: str = ""
    vendor: str = ""
    effort: str = "medium"
    tier: CostTier = CostTier.STANDARD
    capabilities: set[str] = Field(default_factory=set)
    can_edit_files: bool = False
    prior_win_rate: float = 0.6
    est_seconds: float = 60.0
    notes: str = ""

    @property
    def prior_micros(self) -> int:
        return TIER_PRIOR_MICROS[self.tier]

    def fit(self, required: list[str]) -> float:
        """Fraction of required capabilities this worker declares."""
        if not required:
            return 1.0
        have = sum(1 for c in required if c in self.capabilities)
        return have / len(required)


class Candidate(BaseModel):
    worker: WorkerProfile
    score: float
    fit: float
    win_rate: float
    usd_micros: int
    seconds: float
    measured: bool
    reason: str


class Registry:
    def __init__(self, workers: list[WorkerProfile] | None = None):
        self._workers: dict[str, WorkerProfile] = {}
        for w in workers or []:
            self.add(w)

    def add(self, worker: WorkerProfile) -> None:
        self._workers[worker.worker_id] = worker

    def get(self, worker_id: str) -> WorkerProfile | None:
        return self._workers.get(worker_id)

    def all(self) -> list[WorkerProfile]:
        return list(self._workers.values())

    def rank(
        self,
        required: list[str],
        *,
        stats: dict[str, dict] | None = None,
        needs_file_edits: bool = False,
        min_fit: float = 1.0,
    ) -> list[Candidate]:
        """Rank workers for a task.

        `stats` is keyed by worker_id, shaped like Ledger.capability_stats rows.
        Workers that cannot cover every required capability are dropped rather
        than ranked low — a partial worker produces a partial result and a retry.
        """
        stats = stats or {}
        eligible: list[Candidate] = []

        for w in self._workers.values():
            fit = w.fit(required)
            if fit < min_fit:
                continue
            if needs_file_edits and not w.can_edit_files:
                continue

            s = stats.get(w.worker_id)
            measured = bool(s and s.get("attempts", 0) >= MIN_ATTEMPTS_TO_TRUST)
            if measured:
                win = float(s["win_rate"])
                usd = int(s["avg_usd_micros"])
                secs = float(s["avg_seconds"])
                reason = f"measured over {s['attempts']} tasks"
            else:
                win = w.prior_win_rate
                usd = w.prior_micros
                secs = w.est_seconds
                attempts = (s or {}).get("attempts", 0)
                reason = f"declared prior ({attempts} attempts, need {MIN_ATTEMPTS_TO_TRUST})"

            eligible.append(
                Candidate(
                    worker=w,
                    score=0.0,
                    fit=fit,
                    win_rate=win,
                    usd_micros=usd,
                    seconds=secs,
                    measured=measured,
                    reason=reason,
                )
            )

        if not eligible:
            return []

        max_usd = max(c.usd_micros for c in eligible) or 1
        max_secs = max(c.seconds for c in eligible) or 1.0

        for c in eligible:
            c.score = (
                W_FIT * c.fit
                + W_WIN * c.win_rate
                - W_COST * (c.usd_micros / max_usd)
                - W_LATENCY * (c.seconds / max_secs)
            )

        # Ties break toward the cheaper worker, then the faster one.
        eligible.sort(key=lambda c: (-c.score, c.usd_micros, c.seconds))
        return eligible

    def pick(self, required: list[str], **kw) -> Candidate | None:
        ranked = self.rank(required, **kw)
        return ranked[0] if ranked else None


# --------------------------------------------------------------------------
# Default fleet. Reflects what is actually installed on this machine, verified
# 2026-07-30: a CLI worker team, OmniRoute v3.8.50, Ollama.
#
# Worker ids are hive's own. They name a ROLE in this fleet, not whichever
# third-party runtime happens to execute it today — baking another project's
# brand into an id makes the id a lie the moment the backend is swapped.
#
# Not every fast local binary belongs here. A tool can be worth reading the source
# of without being worth routing work to; those are separate judgements, and
# conflating them puts a worker in the fleet that nobody actually wants to run.
# --------------------------------------------------------------------------

_TEAM = Adapter.CLI_TEAM


def default_registry() -> Registry:
    return Registry(
        [
            # Free / local first. Anything these can finish must never reach a metered model.
            WorkerProfile(
                worker_id="ollama.local",
                adapter=Adapter.OLLAMA,
                model="llava-phi3:3.8b",
                tier=CostTier.FREE,
                capabilities={"summarize", "classify", "triage"},
                prior_win_rate=0.4,
                est_seconds=20.0,
                notes="Local last resort for cheap preprocessing.",
            ),
            WorkerProfile(
                worker_id="gateway.free",
                adapter=Adapter.GATEWAY,
                model="auto:free",
                tier=CostTier.FREE,
                capabilities={"summarize", "classify", "triage", "review", "plan"},
                prior_win_rate=0.5,
                est_seconds=25.0,
                notes="Free-tier pool via OmniRoute. Manager decisions must be enum-constrained here.",
            ),
            # Metered Claude workers via the omc team runtime. Real tool use, git worktrees.
            WorkerProfile(
                worker_id="hive.executor",
                adapter=_TEAM,
                vendor="claude",
                agent_type="executor",
                tier=CostTier.STANDARD,
                capabilities={"edit", "implement", "python", "typescript", "mechanical", "refactor"},
                can_edit_files=True,
                prior_win_rate=0.8,
                est_seconds=180.0,
            ),
            WorkerProfile(
                worker_id="hive.explore",
                adapter=_TEAM,
                vendor="claude",
                agent_type="explore",
                tier=CostTier.CHEAP,
                capabilities={"search", "locate", "map"},
                prior_win_rate=0.75,
                est_seconds=60.0,
            ),
            WorkerProfile(
                worker_id="hive.test-engineer",
                adapter=_TEAM,
                vendor="claude",
                agent_type="test-engineer",
                tier=CostTier.STANDARD,
                capabilities={"test", "verify", "python", "typescript"},
                can_edit_files=True,
                prior_win_rate=0.78,
                est_seconds=200.0,
            ),
            WorkerProfile(
                worker_id="hive.debugger",
                adapter=_TEAM,
                vendor="claude",
                agent_type="debugger",
                tier=CostTier.STANDARD,
                capabilities={"debug", "diagnose", "test"},
                can_edit_files=True,
                prior_win_rate=0.7,
                est_seconds=240.0,
            ),
            WorkerProfile(
                worker_id="hive.verifier",
                adapter=_TEAM,
                vendor="claude",
                agent_type="verifier",
                tier=CostTier.CHEAP,
                capabilities={"verify", "review"},
                prior_win_rate=0.8,
                est_seconds=90.0,
            ),
            WorkerProfile(
                worker_id="hive.code-reviewer",
                adapter=_TEAM,
                vendor="claude",
                agent_type="code-reviewer",
                tier=CostTier.STANDARD,
                capabilities={"review", "security"},
                prior_win_rate=0.75,
                est_seconds=120.0,
            ),
            WorkerProfile(
                worker_id="hive.architect",
                adapter=_TEAM,
                vendor="claude",
                agent_type="architect",
                effort="high",
                tier=CostTier.PREMIUM,
                capabilities={"design", "architecture", "plan", "diagnose"},
                prior_win_rate=0.85,
                est_seconds=300.0,
                notes="Expensive. Reserve for genuinely hard design calls.",
            ),
            WorkerProfile(
                worker_id="hive.critic",
                adapter=_TEAM,
                vendor="claude",
                agent_type="critic",
                effort="high",
                tier=CostTier.PREMIUM,
                capabilities={"review", "critique", "verify"},
                prior_win_rate=0.85,
                est_seconds=240.0,
                notes="Expensive. Use for adversarial checks that must not be wrong.",
            ),
        ]
    )


__all__ = [
    "Adapter",
    "Candidate",
    "CostTier",
    "MIN_ATTEMPTS_TO_TRUST",
    "Registry",
    "WorkerProfile",
    "default_registry",
]
