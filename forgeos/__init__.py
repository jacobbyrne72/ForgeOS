"""forgeos — the cost-governed AI harness. Eliminate requests before you make them."""
from __future__ import annotations
__version__ = "0.2.0"
from .compiler import CompilerError, Mission, compile_mission
from .circuit_breaker import BreakerState, CircuitBreaker
from .prompt_cache import CacheEntry, PromptCache
from .security_diff import DiffResult, DiffLine, get_diff, run_diff_security
from .forge import ExecutionResult, Forge, ForgeResult, TaskOutcome
from .contracts import (
    Budget, Escalation, EscalationKind, GovernorTrip, JobSpec, Scope,
    TaskSpec, TaskState, TestResults, Verdict, WorkerReport,
    from_micros, to_micros,
)
from .ledger import Ledger, open_ledger
from .registry import Adapter, CostTier, Registry, WorkerProfile, default_registry
from .adapter.auto_discover import DiscoveredAdapter, discover_adapters
__all__ = [
    "Forge", "ForgeResult", "ExecutionResult", "TaskOutcome",
    "compile_mission", "Mission", "CompilerError",
    "CircuitBreaker", "BreakerState",
    "PromptCache", "CacheEntry",
    "DiffResult", "DiffLine", "get_diff", "run_diff_security",
    "Adapter", "CostTier", "Registry", "WorkerProfile", "default_registry",
    "discover_adapters", "DiscoveredAdapter",
    "Ledger", "open_ledger",
    "Budget", "Escalation", "EscalationKind", "GovernorTrip",
    "JobSpec", "Scope", "TaskSpec", "TaskState", "TestResults",
    "Verdict", "WorkerReport", "from_micros", "to_micros",
    "__version__",
]
