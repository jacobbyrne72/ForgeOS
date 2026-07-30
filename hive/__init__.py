"""hive — a cost-governed agent harness.

Layers, cheapest first:

    economy/   eliminate the request      (atlas pack, dedup, cache reuse)
    gateway/   make the request cheap     (OmniRoute free tiers, litellm fallback)
    registry   pick the cheapest worker that can actually finish
    adapters/  run it                     (jcode, omc team, OmniRoute, Ollama)
    core/      supervise it               (manager, governor, router)
    ledger     prove what it cost

The ordering is the whole thesis. Routing a request to a cheaper model saves a
fraction; not making the request saves all of it.
"""

__version__ = "0.1.0"

from .forge import ExecutionResult, Forge, ForgeResult, TaskOutcome
from .contracts import (
    Budget,
    Escalation,
    EscalationKind,
    GovernorTrip,
    JobSpec,
    Scope,
    TaskSpec,
    TaskState,
    TestResults,
    Verdict,
    WorkerReport,
    from_micros,
    to_micros,
)
from .ledger import Ledger, open_ledger
from .registry import Adapter, CostTier, Registry, WorkerProfile, default_registry

__all__ = [
    "Adapter",
    "ExecutionResult",
    "Forge",
    "ForgeResult",
    "Budget",
    "CostTier",
    "Escalation",
    "EscalationKind",
    "GovernorTrip",
    "JobSpec",
    "Ledger",
    "Registry",
    "Scope",
    "TaskSpec",
    "TaskState",
    "TestResults",
    "Verdict",
    "WorkerProfile",
    "TaskOutcome",
    "WorkerReport",
    "default_registry",
    "from_micros",
    "open_ledger",
    "to_micros",
]
