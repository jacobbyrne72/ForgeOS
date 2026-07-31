"""forgeos — the cost-governed AI harness. Eliminate requests before you make them."""

from __future__ import annotations

__version__ = "0.4.5"
from .compiler import CompilerError, Mission, compile_mission
from .circuit_breaker import BreakerState, CircuitBreaker
from .prompt_cache import CacheEntry, PromptCache
from .security_diff import DiffResult, DiffLine, get_diff, run_diff_security
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
from .adapter.auto_discover import DiscoveredAdapter, discover_adapters
from .bench import BenchmarkResult, Layer, bench, format_benchmark
from .context_compress import compress_context
from .watch import watch
from .adapt import AdapterProfiler, AdapterDecision, AdapterProfile
from .model_select import ModelSelector, ModelRecommendation
from .profile import ModelProfiler, ModelProfile
from .auto_optimize import AutoOptimizer, AutoOptimizeResult
from .token_budget import TokenBudget, BudgetExceeded
from .optimizer import CostOptimizer, OptimizationPlan
from .batch_optimize import BatchOptimizer
from .cost_replacer import CostReplacer

__all__ = [
    "Forge",
    "ForgeResult",
    "ExecutionResult",
    "TaskOutcome",
    "compile_mission",
    "Mission",
    "CompilerError",
    "CircuitBreaker",
    "BreakerState",
    "PromptCache",
    "CacheEntry",
    "DiffResult",
    "DiffLine",
    "get_diff",
    "run_diff_security",
    "Adapter",
    "CostTier",
    "Registry",
    "WorkerProfile",
    "default_registry",
    "discover_adapters",
    "DiscoveredAdapter",
    "Ledger",
    "open_ledger",
    "Budget",
    "Escalation",
    "EscalationKind",
    "GovernorTrip",
    "JobSpec",
    "Scope",
    "TaskSpec",
    "TaskState",
    "TestResults",
    "Verdict",
    "WorkerReport",
    "from_micros",
    "to_micros",
    "__version__",
    "bench",
    "BenchmarkResult",
    "Layer",
    "format_benchmark",
    "compress_context",
    "watch",
    "ModelSelector", "ModelRecommendation",
    "AdapterProfiler",
    "AdapterDecision",
    "AdapterProfile",
    "ModelProfiler",
    "ModelProfile",
    "AutoOptimizer",
    "AutoOptimizeResult",
    "TokenBudget",
    "BudgetExceeded",
    "CostOptimizer",
    "OptimizationPlan",
    "BatchOptimizer",
    "CostReplacer",
    "AdaptiveBatch",
    "SmartBatchPredictor",
    "CostAuditor",
]

from .adaptive_batch import AdaptiveBatch
from .smart_batch import SmartBatchPredictor

from .cost_audit import CostAuditor

from .cost_router import CostRouter, Route
