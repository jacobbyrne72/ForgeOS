"""Job-scoped gateway seat affinity stays a preference, never a mandate."""

from __future__ import annotations

from forgeos.catalog import Catalog, ModelCard
from forgeos.contracts import JobSpec
from forgeos.gateway.client import Gateway, GatewayRequest, RawCallResult
from forgeos.ledger import Ledger
from forgeos.core.quota import QuotaTracker


class _Transport:
    def __init__(self, name: str, calls: list[str]) -> None:
        self.name = name
        self.serves = {"test"}
        self.calls = calls

    def complete(self, *, model_id, prompt, max_output_tokens, reasoning_effort,
                 tools_schema, prompt_prefix=""):
        del prompt, max_output_tokens, reasoning_effort, tools_schema, prompt_prefix
        self.calls.append(self.name)
        return RawCallResult(text="ok", tokens_in=2, tokens_out=1, model_used=model_id)


def _gateway():
    calls: list[str] = []
    ledger = Ledger(":memory:")
    job = JobSpec(objective="exercise gateway affinity", cwd=".")
    ledger.open_job(job)
    gateway = Gateway(
        catalog=Catalog([ModelCard(
            model_id="model", provider="test", context=10_000,
            input_cost_per_1m=1.0, output_cost_per_1m=1.0,
        )]),
        ledger=ledger,
        transports=[_Transport("a", calls), _Transport("b", calls)],
    )
    return gateway, ledger, job, calls


def _request(tail: str) -> GatewayRequest:
    return GatewayRequest(model_ref="test/model", prompt_tail=tail, max_output_tokens=10)


def test_affinity_reorders_a_health_better_candidate_and_reports_truthfully():
    gateway, ledger, job, calls = _gateway()
    try:
        first = gateway.complete(
            _request("first"), job_id=job.id, task_id=None, worker_id="w",
            remaining_micros=1_000_000, affinity_key=job.id,
        )
        assert first.affinity_applied is False
        assert calls == ["a"]

        # Make the non-pinned seat the health winner. Affinity should move the
        # already-legal warm seat back to the front, not resurrect anything.
        gateway._health.record_success("a", latency_ms=100)
        gateway._health.record_success("b", latency_ms=1)
        second = gateway.complete(
            _request("second"), job_id=job.id, task_id=None, worker_id="w",
            remaining_micros=1_000_000, affinity_key=job.id,
        )
        assert calls == ["a", "a"]
        assert second.affinity_applied is True
    finally:
        gateway.close()
        ledger.close()


def test_exhausted_model_quota_releases_affinity_pin():
    gateway, ledger, job, calls = _gateway()
    try:
        gateway.complete(
            _request("first"), job_id=job.id, task_id=None, worker_id="w",
            remaining_micros=1_000_000, affinity_key=job.id,
        )
        gateway._health.record_success("a", latency_ms=100)
        gateway._health.record_success("b", latency_ms=1)
        quota = QuotaTracker()
        quota.record_exhaustion("test", model="model", at=1_800_000_000)

        response = gateway.complete(
            _request("after quota"), job_id=job.id, task_id=None, worker_id="w",
            remaining_micros=1_000_000, affinity_key=job.id, quota=quota,
        )
        assert calls == ["a", "b"]
        assert response.affinity_applied is False
    finally:
        gateway.close()
        ledger.close()
