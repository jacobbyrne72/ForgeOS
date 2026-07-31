"""The routed executor: worker_id → registry profile → adapter → result.

This is the default `Forge.run(executor=None)` path, so the tests prove three
things and nothing else: unavailability is reported (never substituted),
availability flows the bridge's result through untouched, and a classified
MODEL failure buys exactly one rung of escalation on the next attempt.
"""

from __future__ import annotations

from forgeos.adapters import routed
from forgeos.adapters.base import (
    EventKind,
    WorkerAdapter,
    WorkerCapabilities,
    WorkerEvent,
    WorkerUsage,
)
from forgeos.adapters.routed import routed_executor
from forgeos.contracts import (
    Budget,
    FailureClass,
    Scope,
    TaskSpec,
    TaskState,
    TestResults,
)
from forgeos.forge import ExecutionResult, Forge
from forgeos.registry import Adapter, CostTier, Registry, WorkerProfile
from forgeos.settings import Settings


class FakeAdapter(WorkerAdapter):
    """The base.py protocol in memory (same shape as test_executor_bridge)."""

    def __init__(self, events: list[WorkerEvent] | None = None) -> None:
        self._events = list(events or [])
        self.calls: list[str] = []
        self.prompts: list[str] = []
        self.started: list[tuple[str, str, str]] = []

    def health(self) -> tuple[bool, str]:
        return True, "fake adapter"

    def capabilities(self) -> WorkerCapabilities:
        return WorkerCapabilities()

    async def start(self, task_id: str, cwd: str, model_profile: str) -> str:
        self.calls.append("start")
        self.started.append((task_id, cwd, model_profile))
        return "sess-1"

    async def send(self, session_id: str, prompt: str):
        self.calls.append("send")
        self.prompts.append(prompt)
        for event in self._events:
            yield event

    async def cancel(self, session_id: str) -> None:
        self.calls.append("cancel")

    async def checkpoint(self, session_id: str) -> dict:
        return {"task_id": "t"}

    async def resume(self, checkpoint: dict) -> str:
        return "sess-resumed"

    async def usage(self, session_id: str) -> WorkerUsage:
        return WorkerUsage()

    async def close(self, session_id: str) -> None:
        self.calls.append("close")


def _green_events() -> list[WorkerEvent]:
    return [
        WorkerEvent(kind=EventKind.MESSAGE, text="Implemented the retry fix."),
        WorkerEvent(kind=EventKind.TOOL_CALL, text="python -m pytest tests/test_retry.py",
                    tool_kind="execute"),
        WorkerEvent(kind=EventKind.TOOL_UPDATE, status="completed",
                    text="=================== 13 passed in 1.21s ==================="),
        WorkerEvent(kind=EventKind.FILE_DIFF, path="src/retry.py",
                    diff="--- a/src/retry.py\n+++ b/src/retry.py"),
        WorkerEvent(kind=EventKind.DONE, status="ok"),
    ]


def _spec() -> TaskSpec:
    return TaskSpec(job_id="job_pending", subject="normalise retry parsing",
                    description="d", capabilities=["edit", "python"],
                    scope=Scope(paths=["src/retry.py"]),
                    acceptance=["targeted tests pass"],
                    budget=Budget(max_usd=2.0, max_seconds=900))


def _profile(worker_id="local.fake", *, tier=CostTier.FREE, model="") -> WorkerProfile:
    return WorkerProfile(worker_id=worker_id, adapter=Adapter.OLLAMA, tier=tier,
                         capabilities={"edit", "python", "review"},
                         can_edit_files=True, prior_win_rate=0.9, model=model)


# ======================================================= unavailable is honest


def test_an_unknown_worker_id_is_an_environment_failure_not_a_crash():
    execute = routed_executor(Registry([]))
    result = execute(_spec(), "ghost.worker")
    assert result.state is TaskState.FAILED
    assert result.failure is FailureClass.ENVIRONMENT
    assert "ghost.worker" in result.blocker


def test_an_unbuildable_backend_reports_the_factorys_reason_verbatim():
    registry = Registry([
        WorkerProfile(worker_id="api.paid", adapter=Adapter.GATEWAY,
                      tier=CostTier.CHEAP, capabilities={"edit", "python"},
                      can_edit_files=True, prior_win_rate=0.9),
    ])
    result = routed_executor(registry)(_spec(), "api.paid")
    assert result.state is TaskState.FAILED
    assert result.failure is FailureClass.ENVIRONMENT
    assert "Gateway" in result.blocker  # the factory's reason, not a paraphrase


# ========================================================== available flows on


def test_a_buildable_backend_runs_and_the_bridged_result_flows_through(monkeypatch):
    adapter = FakeAdapter(_green_events())
    monkeypatch.setattr(routed, "build_adapter", lambda p, **kw: (adapter, "built"))
    execute = routed_executor(Registry([_profile(model="qwen-7b")]))

    result = execute(_spec(), "local.fake")

    assert result.state is TaskState.DONE
    assert "src/retry.py" in result.files_touched
    # The profile's model reached the adapter session unchanged.
    assert adapter.started[0][2] == "qwen-7b"
    assert "close" in adapter.calls


def test_forge_run_with_no_executor_uses_the_routed_path_end_to_end(tmp_path, monkeypatch):
    """`Forge.run(executor=None)` must reach a real adapter via the registry —
    the default that makes the harness a tool rather than a library."""
    adapters: list[FakeAdapter] = []

    def build(profile, **kwargs):
        a = FakeAdapter(_green_events())
        adapters.append(a)
        return a, "built"

    monkeypatch.setattr(routed, "build_adapter", build)

    def approve(spec: TaskSpec, worker_id: str) -> ExecutionResult:
        return ExecutionResult(state=TaskState.DONE, evidence="review: no unresolved risk",
                               commands_run=["git diff --stat"],
                               tests=TestResults(passed=1, failed=0))

    forge = Forge(home=tmp_path / "state",
                  registry=Registry([_profile(), _profile("cloud.fake", tier=CostTier.PREMIUM)]),
                  max_attempts=2)
    try:
        result = forge.run("prove the routed default", [_spec()], reviewer=approve)
    finally:
        forge.close()

    assert result.accepted == 1, result.outcomes
    assert adapters, "no adapter was ever built — the routed default did not run"
    assert "normalise retry parsing" in adapters[0].prompts[0]


def test_default_executor_lazily_builds_a_gateway_on_the_forges_ledger(tmp_path, monkeypatch):
    from forgeos.catalog import Catalog

    monkeypatch.setattr("forgeos.catalog.default_catalog", lambda: Catalog([]))
    forge = Forge(
        home=tmp_path / "state",
        registry=Registry([]),
        settings=Settings(providers={}),
    )
    try:
        assert forge._gateway is None
        forge.default_executor()
        assert forge._gateway is not None
        assert forge._gateway._ledger is forge.ledger
        assert forge._gateway._dead_models.path.endswith("dead_models.db")
    finally:
        forge.close()


# ============================================================= tier escalation


def test_a_model_failure_escalates_the_tier_on_the_next_attempt(tmp_path):
    cheap = _profile("local.cheap", tier=CostTier.FREE)
    strong = _profile("cloud.strong", tier=CostTier.PREMIUM)
    ran: list[str] = []

    def execute(spec: TaskSpec, worker_id: str) -> ExecutionResult:
        ran.append(worker_id)
        if worker_id == "local.cheap":
            return ExecutionResult(state=TaskState.FAILED, failure=FailureClass.MODEL,
                                   blocker="beyond this tier")
        return ExecutionResult(state=TaskState.DONE, evidence="fixed",
                               commands_run=["python -m pytest -q"],
                               files_touched=["src/retry.py"],
                               tests=TestResults(passed=13, failed=0))

    def approve(spec: TaskSpec, worker_id: str) -> ExecutionResult:
        return ExecutionResult(state=TaskState.DONE, evidence="review pass",
                               commands_run=["git diff"],
                               tests=TestResults(passed=1, failed=0))

    forge = Forge(home=tmp_path / "state", registry=Registry([cheap, strong]),
                  max_attempts=3)
    try:
        result = forge.run("escalate on model failure", [_spec()],
                           executor=execute, reviewer=approve)
    finally:
        forge.close()

    outcome = result.outcomes[0]
    assert ran[0] == "local.cheap", ran
    assert "cloud.strong" in ran, (
        f"MODEL failure never escalated past the cheap tier: {ran} / {outcome.reason}")
    assert outcome.attempts >= 2


def test_an_environment_failure_never_escalates(tmp_path):
    """The other half of the contract: a broken venv fails identically at every
    price point, so the router must never be asked to buy a stronger model."""
    cheap = _profile("local.cheap", tier=CostTier.FREE)
    strong = _profile("cloud.strong", tier=CostTier.PREMIUM)
    ran: list[str] = []

    def execute(spec: TaskSpec, worker_id: str) -> ExecutionResult:
        ran.append(worker_id)
        return ExecutionResult(state=TaskState.FAILED, failure=FailureClass.ENVIRONMENT,
                               blocker="ModuleNotFoundError: pytest")

    forge = Forge(home=tmp_path / "state", registry=Registry([cheap, strong]),
                  max_attempts=3)
    try:
        result = forge.run("do not escalate a broken venv", [_spec()], executor=execute)
    finally:
        forge.close()

    assert "cloud.strong" not in ran, ran
    assert result.outcomes[0].accepted is False
