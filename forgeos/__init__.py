"""forgeos — the cost-governed AI harness. Eliminate requests before you make them."""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.6.10"

# Name -> submodule. Every one of these used to be imported eagerly at
# `import forgeos`, all 39 modules of them, on EVERY invocation including
# `--help`. Measured on this machine: 12.1s to import the package and up to
# 22s for a CLI that had not yet done any work. A tool whose whole pitch is
# "N agents, zero interruptions" cannot cost ten seconds of process startup
# per agent -- that is a tax on exactly the workload it exists to serve.
#
# PEP 562 module __getattr__ defers each import to first ACCESS. The public
# API is unchanged: every name below still resolves as `forgeos.Name`, and
# the resolved value is written back into globals() so the cost is paid once
# per name per process, never per lookup.
_LAZY: dict[str, str] = {
    'Adapter': 'registry',
    'AdapterDecision': 'adapt',
    'AdapterProfile': 'adapt',
    'AdapterProfiler': 'adapt',
    'AdaptiveBatch': 'adaptive_batch',
    'AutoOptimizeResult': 'auto_optimize',
    'AutoOptimizer': 'auto_optimize',
    'BatchOptimizer': 'batch_optimize',
    'BatchProjection': 'batch_projection',
    'BenchmarkResult': 'bench',
    'BreakerState': 'circuit_breaker',
    'Budget': 'contracts',
    'BudgetExceeded': 'token_budget',
    'CacheEntry': 'prompt_cache',
    'CachePurge': 'cache_purge',
    'CircuitBreaker': 'circuit_breaker',
    'CompilerError': 'compiler',
    'CostAuditor': 'cost_audit',
    'CostEfficiency': 'cost_efficiency',
    'CostForecast': 'cost_forecast',
    'CostGuard': 'cost_guard',
    'CostOptimizer': 'optimizer',
    'CostReplacer': 'cost_replacer',
    'CostReplayer': 'cost_replay',
    'CostRetry': 'cost_retry',
    'CostRouter': 'cost_router',
    'CostScheduler': 'cost_scheduler',
    'CostTier': 'registry',
    'CostTracker': 'cost_tracker',
    'DiffLine': 'security_diff',
    'DiffResult': 'security_diff',
    'DiscoveredAdapter': 'adapter.auto_discover',
    'Escalation': 'contracts',
    'EscalationKind': 'contracts',
    'ExecutionResult': 'forge',
    'Forge': 'forge',
    'ForgeResult': 'forge',
    'GovernorTrip': 'contracts',
    'JobSpec': 'contracts',
    'Layer': 'bench',
    'Ledger': 'ledger',
    'LocalExecutor': 'local_exec',
    'LocalFormatter': 'local_formatter',
    'Mission': 'compiler',
    'ModelProfile': 'profile',
    'ModelProfiler': 'profile',
    'ModelRanker': 'model_ranker',
    'ModelRecommendation': 'model_select',
    'ModelSelector': 'model_select',
    'OptimizationPlan': 'optimizer',
    'PromptCache': 'prompt_cache',
    'Registry': 'registry',
    'Route': 'cost_router',
    'Scope': 'contracts',
    'SmartBatchPredictor': 'smart_batch',
    'TaskOutcome': 'forge',
    'TaskSpec': 'contracts',
    'TaskState': 'contracts',
    'TestResults': 'contracts',
    'TokenBudget': 'token_budget',
    'TokenRouter': 'token_router',
    'Verdict': 'contracts',
    'WorkerProfile': 'registry',
    'WorkerReport': 'contracts',
    'bench': 'bench',
    'compile_mission': 'compiler',
    'compress_context': 'context_compress',
    'compress_output': 'output_compressor',
    'default_registry': 'registry',
    'discover_adapters': 'adapter.auto_discover',
    'estimate_output_tokens': 'output_compressor',
    'estimate_prompt_tokens': 'prompt_optimizer',
    'estimate_tokens': 'prompt_optimizer',
    'format_benchmark': 'bench',
    'from_micros': 'contracts',
    'get_diff': 'security_diff',
    'open_ledger': 'ledger',
    'optimize_prompt': 'prompt_optimizer',
    'remove_filler_words': 'prompt_optimizer',
    'run_diff_security': 'security_diff',
    'shrink_prompt': 'prompt_optimizer',
    'to_micros': 'contracts',
    'truncate_response': 'output_compressor',
    'watch': 'watch',
}


def __getattr__(name: str) -> object:
    """Import the submodule owning `name` on first access (PEP 562)."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(f'.{module}', __name__), name)
    globals()[name] = value  # subsequent lookups skip this function entirely
    return value


def __dir__() -> list[str]:
    """Keep tab-completion and `dir(forgeos)` showing the full API, which a
    bare module __getattr__ would otherwise hide."""
    return sorted(set(globals()) | set(_LAZY))


if TYPE_CHECKING:  # type checkers and IDEs need the real bindings
    from .adapt import AdapterDecision, AdapterProfile, AdapterProfiler  # noqa: F401
    from .adapter.auto_discover import DiscoveredAdapter, discover_adapters  # noqa: F401
    from .adaptive_batch import AdaptiveBatch  # noqa: F401
    from .auto_optimize import AutoOptimizeResult, AutoOptimizer  # noqa: F401
    from .batch_optimize import BatchOptimizer  # noqa: F401
    from .batch_projection import BatchProjection  # noqa: F401
    from .bench import BenchmarkResult, Layer, bench, format_benchmark  # noqa: F401
    from .cache_purge import CachePurge  # noqa: F401
    from .circuit_breaker import BreakerState, CircuitBreaker  # noqa: F401
    from .compiler import CompilerError, Mission, compile_mission  # noqa: F401
    from .context_compress import compress_context  # noqa: F401
    from .contracts import Budget, Escalation, EscalationKind, GovernorTrip, JobSpec, Scope, TaskSpec, TaskState, TestResults, Verdict, WorkerReport, from_micros, to_micros  # noqa: F401
    from .cost_audit import CostAuditor  # noqa: F401
    from .cost_efficiency import CostEfficiency  # noqa: F401
    from .cost_forecast import CostForecast  # noqa: F401
    from .cost_guard import CostGuard  # noqa: F401
    from .cost_replacer import CostReplacer  # noqa: F401
    from .cost_replay import CostReplayer  # noqa: F401
    from .cost_retry import CostRetry  # noqa: F401
    from .cost_router import CostRouter, Route  # noqa: F401
    from .cost_scheduler import CostScheduler  # noqa: F401
    from .cost_tracker import CostTracker  # noqa: F401
    from .forge import ExecutionResult, Forge, ForgeResult, TaskOutcome  # noqa: F401
    from .ledger import Ledger, open_ledger  # noqa: F401
    from .local_exec import LocalExecutor  # noqa: F401
    from .local_formatter import LocalFormatter  # noqa: F401
    from .model_ranker import ModelRanker  # noqa: F401
    from .model_select import ModelRecommendation, ModelSelector  # noqa: F401
    from .optimizer import CostOptimizer, OptimizationPlan  # noqa: F401
    from .output_compressor import compress_output, estimate_output_tokens, truncate_response  # noqa: F401
    from .profile import ModelProfile, ModelProfiler  # noqa: F401
    from .prompt_cache import CacheEntry, PromptCache  # noqa: F401
    from .prompt_optimizer import estimate_prompt_tokens, estimate_tokens, optimize_prompt, remove_filler_words, shrink_prompt  # noqa: F401
    from .registry import Adapter, CostTier, Registry, WorkerProfile, default_registry  # noqa: F401
    from .security_diff import DiffLine, DiffResult, get_diff, run_diff_security  # noqa: F401
    from .smart_batch import SmartBatchPredictor  # noqa: F401
    from .token_budget import BudgetExceeded, TokenBudget  # noqa: F401
    from .token_router import TokenRouter  # noqa: F401
    from .watch import watch  # noqa: F401

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
    "TokenRouter",
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
    "CostRouter",
    "Route",
    "CostTracker",
    "CostReplayer",
    "optimize_prompt",
    "estimate_tokens",
    "LocalExecutor",
    "CostRetry",
    "CostScheduler",
    "CostGuard",
    "CostForecast",
    "CostEfficiency",
    "compress_output",
    "estimate_output_tokens",
    "LocalFormatter",
    "CachePurge",
    "shrink_prompt",
    "estimate_prompt_tokens",
    "remove_filler_words",
    "ModelRanker",
    "truncate_response",
    "BatchProjection",
]
