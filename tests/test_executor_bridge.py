"""The adapter→executor bridge, proven against a fake WorkerAdapter.

Every test drives `forgeos.adapters.executor.adapter_executor` through the real
async protocol from `adapters/base.py` — start → send → usage → close — with a
fake adapter: no subprocess, no network, no real CLI. The final test hands the
bridged callable to a real `Forge.run` and asserts the whole machine accepts
the work, which is exactly the integration the bridge exists to enable.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from forgeos.adapters.base import (
    EventKind,
    WorkerAdapter,
    WorkerCapabilities,
    WorkerEvent,
    WorkerUsage,
)
from forgeos.adapters.executor import adapter_executor, classify_failure
from forgeos.contracts import Budget, FailureClass, Scope, TaskSpec, TaskState
from forgeos.forge import ExecutionResult, Forge
from forgeos.registry import Adapter, CostTier, Registry, WorkerProfile


class FakeAdapter(WorkerAdapter):
    """The full base.py protocol in memory, recording the call order."""

    def __init__(
        self,
        events: list[WorkerEvent] | None = None,
        *,
        usage: WorkerUsage | None = None,
        send_exc: Exception | None = None,
        hang_seconds: float = 0.0,
        resumable: bool = False,
        checkpoint_exc: Exception | None = None,
    ) -> None:
        self._events = list(events or [])
        self._usage = usage or WorkerUsage()
        self._send_exc = send_exc
        self._hang_seconds = hang_seconds
        self._resumable = resumable
        self._checkpoint_exc = checkpoint_exc
        self.calls: list[str] = []
        self.prompts: list[str] = []
        self.started: list[tuple[str, str, str]] = []
        self.resumed_from: list[dict] = []

    def health(self) -> tuple[bool, str]:
        return True, "fake adapter"

    def capabilities(self) -> WorkerCapabilities:
        return WorkerCapabilities(resumable_sessions=self._resumable)

    async def start(self, task_id: str, cwd: str, model_profile: str) -> str:
        self.calls.append("start")
        self.started.append((task_id, cwd, model_profile))
        return "sess-1"

    async def send(self, session_id: str, prompt: str):
        self.calls.append("send")
        self.prompts.append(prompt)
        if self._send_exc is not None:
            raise self._send_exc
        if self._hang_seconds:
            await asyncio.sleep(self._hang_seconds)
        for event in self._events:
            yield event

    async def cancel(self, session_id: str) -> None:
        self.calls.append("cancel")

    async def checkpoint(self, session_id: str) -> dict:
        self.calls.append("checkpoint")
        if self._checkpoint_exc is not None:
            raise self._checkpoint_exc
        return {"task_id": "t", "session_id": session_id}

    async def resume(self, checkpoint: dict) -> str:
        self.calls.append("resume")
        self.resumed_from.append(checkpoint)
        return "sess-resumed"

    async def usage(self, session_id: str) -> WorkerUsage:
        self.calls.append("usage")
        return self._usage

    async def close(self, session_id: str) -> None:
        self.calls.append("close")


def _spec(subject="normalise retry parsing", *, max_seconds=900,
          paths=("src/retry.py",)) -> TaskSpec:
    return TaskSpec(job_id="job_pending", subject=subject, description="d",
                    capabilities=["edit", "python"], scope=Scope(paths=list(paths)),
                    acceptance=["targeted tests pass"],
                    budget=Budget(max_usd=2.0, max_seconds=max_seconds))


def _green_events() -> list[WorkerEvent]:
    return [
        WorkerEvent(kind=EventKind.THOUGHT, text="privately reasoning about retries"),
        WorkerEvent(kind=EventKind.MESSAGE, text="Implemented the retry fix."),
        WorkerEvent(kind=EventKind.TOOL_CALL, text="python -m pytest tests/test_retry.py",
                    tool_kind="execute"),
        WorkerEvent(kind=EventKind.TOOL_UPDATE, status="completed",
                    text="=================== 13 passed in 1.21s ==================="),
        WorkerEvent(kind=EventKind.FILE_DIFF, path="src/retry.py",
                    diff="--- a/src/retry.py\n+++ b/src/retry.py"),
        WorkerEvent(kind=EventKind.DONE, status="ok"),
    ]


# =============================================================== happy path


def test_a_green_stream_maps_to_done_with_evidence_files_and_commands():
    adapter = FakeAdapter(_green_events())
    execute = adapter_executor(adapter, cwd="/repo", model_profile="small")
    spec = _spec()

    result = execute(spec, "worker-1")

    assert result.state is TaskState.DONE
    assert result.failure is None
    assert "Implemented the retry fix." in result.evidence
    assert "privately reasoning" not in result.evidence, "THOUGHT chatter is not evidence"
    assert result.files_touched == ["src/retry.py"]
    assert result.commands_run == ["python -m pytest tests/test_retry.py"]
    assert result.tests is not None
    assert result.tests.passed == 13 and result.tests.failed == 0
    assert "13 passed" in result.raw_output
    assert result.seconds >= 0.0
    assert isinstance(result, ExecutionResult), "a subclass — Forge.run accepts it unchanged"
    assert result.spend_already_recorded is False, "no transport banked anything here"
    # Full session lifecycle, in order, ending in close.
    assert adapter.calls == ["start", "send", "usage", "close"]
    assert adapter.started[0] == (spec.id, "/repo", "small")
    # The prompt is built from canonical TaskSpec fields.
    assert spec.subject in adapter.prompts[0]
    assert "targeted tests pass" in adapter.prompts[0]


def test_a_success_claim_without_a_real_pytest_summary_parses_to_no_tests():
    events = [
        WorkerEvent(kind=EventKind.MESSAGE, text="All the tests pass, trust me."),
        WorkerEvent(kind=EventKind.DONE, status="ok"),
    ]
    result = adapter_executor(FakeAdapter(events), cwd=".")(_spec(), "w")
    assert result.state is TaskState.DONE
    assert result.tests is None, "prose is not verification; the merge gate must refuse this"


# ============================================================ money honesty


def test_inexact_usage_never_becomes_a_measured_dollar_figure():
    usage = WorkerUsage(input_tokens=900, output_tokens=150, usd_micros=123_456, exact=False)
    adapter = FakeAdapter(_green_events(), usage=usage)

    result = adapter_executor(adapter, cwd=".")(_spec(), "w")

    assert result.usd_micros == 0, "an inexact figure must be left for the tier prior"
    assert result.tokens_in == 900 and result.tokens_out == 150, \
        "token counts are copied as reported, never invented"


def test_exact_usage_passes_straight_through():
    usage = WorkerUsage(input_tokens=10, output_tokens=5, usd_micros=4200, exact=True)
    adapter = FakeAdapter(_green_events(), usage=usage)
    result = adapter_executor(adapter, cwd=".")(_spec(), "w")
    assert result.usd_micros == 4200


def test_gateway_style_usage_event_marks_spend_already_recorded():
    """A gateway-backed worker banks its spend in the ledger before returning
    (`gateway_worker.py`). The flag must survive the mapping so the Forge can
    tell "nobody has paid yet" apart from "real spend, already recorded" —
    without it the same call is billed twice."""
    events = [
        WorkerEvent(kind=EventKind.MESSAGE, text="summary of the diff"),
        WorkerEvent(kind=EventKind.USAGE, data={
            "input_tokens": 1200, "output_tokens": 300, "usd_micros": 4200,
            "exact": True, "spend_already_recorded": True,
        }),
        WorkerEvent(kind=EventKind.DONE),
    ]
    usage = WorkerUsage(input_tokens=1200, output_tokens=300, usd_micros=4200, exact=True)
    result = adapter_executor(FakeAdapter(events, usage=usage), cwd=".")(_spec(), "w")

    assert result.state is TaskState.DONE
    assert result.spend_already_recorded is True
    assert result.usd_micros == 4200, "the figure is real and stays reported honestly"
    assert result.tokens_in == 1200, "numbers come from usage(), never re-folded from events"


def test_a_preflight_refusal_hint_beats_text_guessing_and_banks_nothing():
    """The gateway refuses before touching the transport: no money moved. The
    structured `refused` hint must classify POLICY even though the prose
    matches no policy signature."""
    events = [WorkerEvent(
        kind=EventKind.ERROR,
        text="refused before the call: projected cost exceeds remaining headroom",
        data={"refused": True, "spend_already_recorded": False},
    )]
    result = adapter_executor(FakeAdapter(events), cwd=".")(_spec(), "w")

    assert result.state is TaskState.FAILED
    assert result.failure is FailureClass.POLICY
    assert result.spend_already_recorded is False


def test_a_transport_failure_hint_is_transient_with_spend_banked():
    """A call that died mid-flight was still charged (the gateway records the
    estimate on the way out) and is retryable — both facts arrive as
    structured hints, not prose."""
    events = [WorkerEvent(
        kind=EventKind.ERROR,
        text="transport failed: upstream socket closed unexpectedly",
        data={"transient": True, "spend_already_recorded": True},
    )]
    result = adapter_executor(FakeAdapter(events), cwd=".")(_spec(), "w")

    assert result.state is TaskState.FAILED
    assert result.failure is FailureClass.TRANSIENT
    assert result.spend_already_recorded is True, \
        "charging this call again would double-count the banked estimate"


# ==================================================== failure classification


def test_module_not_found_in_the_error_stream_is_environment_not_model():
    events = [WorkerEvent(kind=EventKind.ERROR,
                          text="ModuleNotFoundError: No module named 'pytest'")]
    result = adapter_executor(FakeAdapter(events), cwd=".")(_spec(), "w")

    assert result.state is TaskState.FAILED
    assert result.failure is not FailureClass.MODEL, \
        "escalating a broken venv buys premium tokens for a problem no model can fix"
    assert result.failure is FailureClass.ENVIRONMENT
    assert "ModuleNotFoundError" in result.blocker


def test_a_429_in_the_error_stream_is_transient():
    events = [WorkerEvent(kind=EventKind.ERROR,
                          text="provider returned HTTP 429 Too Many Requests")]
    result = adapter_executor(FakeAdapter(events), cwd=".")(_spec(), "w")
    assert result.state is TaskState.FAILED
    assert result.failure is FailureClass.TRANSIENT


def test_permission_denied_is_policy():
    events = [WorkerEvent(kind=EventKind.ERROR,
                          text="permission denied: this action needs human sign-off")]
    result = adapter_executor(FakeAdapter(events), cwd=".")(_spec(), "w")
    assert result.failure is FailureClass.POLICY


def test_backend_reported_failure_status_classifies_from_its_error_stream():
    events = [
        WorkerEvent(kind=EventKind.ERROR,
                    text="worker crashed: ModuleNotFoundError: No module named 'requests'"),
        WorkerEvent(kind=EventKind.MESSAGE, text=""),
        WorkerEvent(kind=EventKind.DONE, status="failed"),
    ]
    result = adapter_executor(FakeAdapter(events), cwd=".")(_spec(), "w")
    assert result.state is TaskState.FAILED, "DONE(status=failed) must never read as success"
    assert result.failure is FailureClass.ENVIRONMENT


def test_a_refusal_stop_is_a_policy_failure_not_a_retry():
    events = [WorkerEvent(kind=EventKind.DONE, status="refusal")]
    result = adapter_executor(FakeAdapter(events), cwd=".")(_spec(), "w")
    assert result.state is TaskState.FAILED
    assert result.failure is FailureClass.POLICY


def test_a_stream_that_ends_without_done_is_failed_transient():
    events = [WorkerEvent(kind=EventKind.MESSAGE, text="partial answer, then silence")]
    result = adapter_executor(FakeAdapter(events), cwd=".")(_spec(), "w")
    assert result.state is TaskState.FAILED
    assert result.failure is FailureClass.TRANSIENT, \
        "zero diagnostics: a cheap same-tier retry, never a blind escalation"
    assert "without a DONE" in result.blocker


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("bash: gitleaks: command not found", FailureClass.ENVIRONMENT),
        ("'ruff' is not recognized as an internal or external command", FailureClass.ENVIRONMENT),
        ("ENOENT: no such file or directory, open 'package.json'", FailureClass.ENVIRONMENT),
        ("upstream returned 503 Service Unavailable", FailureClass.TRANSIENT),
        ("read timed out after 30s", FailureClass.TRANSIENT),
        ("", FailureClass.TRANSIENT),
        ("the proposed patch does not satisfy the acceptance criteria", FailureClass.MODEL),
    ],
)
def test_classification_table(text: str, expected: FailureClass):
    assert classify_failure(text) is expected


# ========================================================== timeout and close


def test_a_hung_worker_is_cancelled_closed_and_reported_transient():
    adapter = FakeAdapter(hang_seconds=30.0)
    execute = adapter_executor(adapter, cwd=".", timeout_seconds=0.2)

    started = time.monotonic()
    result = execute(_spec(), "w")

    assert time.monotonic() - started < 10.0, "a hung worker must not hang the job"
    assert result.state is TaskState.FAILED
    assert result.failure is FailureClass.TRANSIENT
    assert "cancel" in adapter.calls and "close" in adapter.calls
    assert adapter.calls.index("cancel") < adapter.calls.index("close"), \
        "cancel the in-flight turn first, then release the session"
    assert "usage" not in adapter.calls, "a hung backend's usage() could hang too"
    assert "exceeded" in result.blocker


def test_the_task_budget_caps_the_timeout_never_the_reverse():
    """`Budget.max_seconds` is a contract with the human; a bridge default must
    never outlive it."""
    adapter = FakeAdapter(hang_seconds=30.0)
    execute = adapter_executor(adapter, cwd=".", timeout_seconds=900.0)

    started = time.monotonic()
    result = execute(_spec(max_seconds=1), "w")

    assert time.monotonic() - started < 10.0
    assert result.state is TaskState.FAILED
    assert result.failure is FailureClass.TRANSIENT


def test_close_is_called_even_when_send_raises():
    adapter = FakeAdapter(send_exc=RuntimeError("adapter blew up mid-stream"))
    result = adapter_executor(adapter, cwd=".")(_spec(), "w")

    assert "close" in adapter.calls, "an unclosed session leaks worktrees and processes"
    assert result.state is TaskState.FAILED
    assert "adapter blew up" in result.blocker
    # A raised exception is the harness breaking, not the model failing the
    # task — escalating it re-runs the same crash at a higher price.
    assert result.failure is FailureClass.TRANSIENT


def test_a_raised_connection_error_still_classifies_from_its_text():
    adapter = FakeAdapter(send_exc=ConnectionResetError("connection reset by peer"))
    result = adapter_executor(adapter, cwd=".")(_spec(), "w")
    assert result.failure is FailureClass.TRANSIENT
    assert "close" in adapter.calls


def test_a_zero_timeout_is_rejected_at_construction():
    with pytest.raises(ValueError):
        adapter_executor(FakeAdapter(), cwd=".", timeout_seconds=0)


# ==================================================== checkpoint and resume


def test_a_resumable_failure_is_checkpointed_and_the_next_call_resumes_it():
    """The whole point of the seam: a FAILED-but-resumable attempt leaves a
    checkpoint behind, and the NEXT adapter built for the SAME task resumes
    from it instead of cold-starting -- proving continuity lives in the store,
    not in any one adapter instance (routed.py builds a fresh adapter per
    attempt in production)."""
    store: dict[str, dict] = {}
    spec = _spec()
    failing = FakeAdapter(
        [WorkerEvent(kind=EventKind.ERROR,
                     text="the proposed patch does not satisfy the acceptance criteria"),
         WorkerEvent(kind=EventKind.DONE, status="failed")],
        resumable=True,
    )

    result1 = adapter_executor(failing, cwd=".", checkpoint_store=store)(spec, "w")

    assert result1.state is TaskState.FAILED
    # usage() is asked for inside the same try block, before the finally that
    # captures the checkpoint -- it only skips when send()/start() raises.
    assert failing.calls == ["start", "send", "usage", "checkpoint", "close"]
    assert store[spec.id] == {"task_id": "t", "session_id": "sess-1"}

    resuming = FakeAdapter(_green_events(), resumable=True)
    result2 = adapter_executor(resuming, cwd=".", checkpoint_store=store)(spec, "w")

    assert result2.state is TaskState.DONE
    assert resuming.calls[0] == "resume", "a stored checkpoint must be resumed, not cold-started"
    assert "start" not in resuming.calls
    assert resuming.resumed_from == [store[spec.id]]


def test_a_non_resumable_adapter_cold_starts_even_with_a_checkpoint_on_file():
    store: dict[str, dict] = {}
    spec = _spec()
    failing = FakeAdapter(
        [WorkerEvent(kind=EventKind.ERROR, text="boom"),
         WorkerEvent(kind=EventKind.DONE, status="failed")],
        resumable=False,
    )
    adapter_executor(failing, cwd=".", checkpoint_store=store)(spec, "w")

    assert "checkpoint" not in failing.calls, \
        "a non-resumable adapter's checkpoint is never even asked for"
    assert spec.id not in store

    again = FakeAdapter(_green_events(), resumable=False)
    result = adapter_executor(again, cwd=".", checkpoint_store=store)(spec, "w")

    assert again.calls[0] == "start"
    assert "resume" not in again.calls
    assert result.state is TaskState.DONE


def test_a_raising_checkpoint_never_masks_the_real_failure():
    store: dict[str, dict] = {}
    spec = _spec()
    adapter = FakeAdapter(
        [WorkerEvent(kind=EventKind.ERROR, text="ModuleNotFoundError: No module named 'pytest'"),
         WorkerEvent(kind=EventKind.DONE, status="failed")],
        resumable=True, checkpoint_exc=RuntimeError("checkpoint backend unreachable"),
    )

    result = adapter_executor(adapter, cwd=".", checkpoint_store=store)(spec, "w")

    assert result.state is TaskState.FAILED
    assert result.failure is FailureClass.ENVIRONMENT
    assert "ModuleNotFoundError" in result.blocker
    assert spec.id not in store, "a raising checkpoint() must leave nothing behind"
    assert "close" in adapter.calls, "close must still run after a raising checkpoint()"


def test_different_tasks_never_share_a_checkpoint_slot():
    store: dict[str, dict] = {}
    spec_a, spec_b = _spec(subject="task a"), _spec(subject="task b")
    fail_events = [WorkerEvent(kind=EventKind.ERROR, text="boom"),
                   WorkerEvent(kind=EventKind.DONE, status="failed")]

    adapter_executor(FakeAdapter(fail_events, resumable=True),
                     cwd=".", checkpoint_store=store)(spec_a, "w")

    assert spec_a.id in store
    assert spec_b.id not in store

    resuming_b = FakeAdapter(_green_events(), resumable=True)
    adapter_executor(resuming_b, cwd=".", checkpoint_store=store)(spec_b, "w")

    assert "start" in resuming_b.calls
    assert "resume" not in resuming_b.calls, \
        "task b must cold-start; it has no checkpoint of its own"


def test_a_green_attempt_never_bothers_calling_checkpoint():
    store: dict[str, dict] = {}
    adapter = FakeAdapter(_green_events(), resumable=True)
    result = adapter_executor(adapter, cwd=".", checkpoint_store=store)(_spec(), "w")

    assert result.state is TaskState.DONE
    assert "checkpoint" not in adapter.calls, \
        "a successful attempt has nothing to resume; capturing one would be a wasted call"


def test_no_checkpoint_store_behaves_exactly_like_before():
    adapter = FakeAdapter(
        [WorkerEvent(kind=EventKind.ERROR, text="boom"),
         WorkerEvent(kind=EventKind.DONE, status="failed")],
        resumable=True,
    )
    result = adapter_executor(adapter, cwd=".")(_spec(), "w")
    assert result.state is TaskState.FAILED
    assert "checkpoint" not in adapter.calls
    assert adapter.calls[0] == "start"


# ===================================================== event-loop coexistence


def test_the_executor_works_from_inside_a_running_event_loop():
    """An async caller (dashboard handler, adapter test harness) must be able
    to invoke the synchronous executor without deadlocking its own loop."""
    adapter = FakeAdapter(_green_events())
    execute = adapter_executor(adapter, cwd=".", model_profile="m")

    async def call_from_async():
        return execute(_spec(), "w")

    result = asyncio.run(call_from_async())
    assert result.state is TaskState.DONE
    assert "close" in adapter.calls


# ============================================================ the real Forge


def _fleet() -> Registry:
    return Registry([
        WorkerProfile(worker_id="local.fake", adapter=Adapter.OLLAMA, tier=CostTier.FREE,
                      capabilities={"edit", "python", "mechanical"}, can_edit_files=True,
                      prior_win_rate=0.85),
        WorkerProfile(worker_id="cloud.fake", adapter=Adapter.CLI_TEAM,
                      tier=CostTier.PREMIUM,
                      capabilities={"edit", "python", "mechanical", "review"},
                      can_edit_files=True, prior_win_rate=0.95),
    ])


def test_the_bridged_executor_satisfies_forge_run_end_to_end(tmp_path):
    """The whole point of the bridge: `Forge.run(executor=...)` accepts the
    returned callable unchanged and the machine runs to an accepted merge —
    routing, leases, spend recording, reduction, the real security gate
    (scoped to `files_touched`, exactly the paths the bridge collected from
    FILE_DIFF events), independent review, receipt."""
    worker = FakeAdapter(_green_events())
    reviewer = FakeAdapter([
        WorkerEvent(kind=EventKind.MESSAGE, text="no unresolved risk"),
        WorkerEvent(kind=EventKind.TOOL_CALL, text="git diff --stat"),
        WorkerEvent(kind=EventKind.DONE, status="ok"),
    ])

    forge = Forge(home=tmp_path / "forgeos", registry=_fleet(), max_attempts=3)
    try:
        result = forge.run(
            "prove the adapter bridge", [_spec()],
            executor=adapter_executor(worker, cwd=".", model_profile="small"),
            reviewer=adapter_executor(reviewer, cwd=".", model_profile="small"),
        )
    finally:
        forge.close()

    assert result.accepted == 1, result.outcomes
    assert result.rejected == 0
    assert result.outcomes[0].reason == "merged"
    # Both sessions — implementer and independent reviewer — were closed.
    assert "close" in worker.calls
    assert "close" in reviewer.calls
    # The worker actually received the task, not an empty prompt.
    assert "normalise retry parsing" in worker.prompts[0]
