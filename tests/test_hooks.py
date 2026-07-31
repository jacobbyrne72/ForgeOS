"""Lifecycle hooks — HookRunner unit tests plus the privacy floor on payloads.

Real-subprocess tests (anything that actually spawns `sys.executable`) are
`slow`-marked, matching this project's convention that spawning a real
external process is the expensive tier of the suite (see pyproject.toml).
The empty-config and payload-shape tests spawn nothing and stay in the fast
default run.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from forgeos.contracts import Budget, JobSpec, Scope, TaskSpec
from forgeos.hooks import HookRunner, job_end_payload, task_payload
from forgeos.settings import HookDef, HookEvent


def _job() -> JobSpec:
    return JobSpec(objective="ship failover", cwd=".")


def _task(job: JobSpec, **kw) -> TaskSpec:
    base = dict(
        job_id=job.id, subject="normalise retry parsing",
        description="SECRET internal notes: do not leak this text",
        capabilities=["edit", "python"], scope=Scope(paths=["src/retry.py"]),
        budget=Budget(max_usd=2.0),
    )
    base.update(kw)
    return TaskSpec(**base)


# Generous on purpose. These tests assert BEHAVIOUR (veto, warning, argv shape),
# never latency, and every one of them spawns a real Python interpreter. At the
# old 5s default they passed alone and failed inside the full suite: interpreter
# startup is ~2s on an idle machine here and blows past 5s once the suite
# saturates the box, so six tests failed for reasons none of them were testing.
# A loaded CI runner would hit the same thing intermittently, which is the worst
# kind of failure to ship in a public repo -- it teaches people to re-run until
# green. The one test that genuinely needs a short timeout passes `timeout=0.2`
# explicitly.
def _hook(event: HookEvent, code: str, timeout: float = 60.0, args: list[str] | None = None) -> HookDef:
    return HookDef(event=event, command=sys.executable, args=["-c", code, *(args or [])],
                   timeout_seconds=timeout)


# ============================================================ empty config


def test_empty_config_never_calls_subprocess(monkeypatch):
    """Zero hooks configured must mean zero subprocess invocations — the
    guarantee `HookRunner.run` makes by construction, verified here by making
    any call to subprocess.run a hard failure."""

    def _boom(*a, **kw):
        raise AssertionError("subprocess.run must not be called with no hooks configured")

    monkeypatch.setattr(subprocess, "run", _boom)

    runner = HookRunner()
    job, spec = _job(), _task(_job())
    for event in HookEvent:
        result = runner.run(event, task_payload(job, spec))
        assert result.outcomes == ()
        assert not result.vetoed
        assert result.warnings == []


def test_configured_reports_only_registered_events():
    runner = HookRunner([_hook(HookEvent.PRE_ROUTE, "pass")])
    assert runner.configured(HookEvent.PRE_ROUTE)
    assert not runner.configured(HookEvent.PRE_EXECUTE)
    assert not runner.configured(HookEvent.POST_GATE)
    assert not runner.configured(HookEvent.JOB_END)


# ============================================================ payload privacy floor


def test_task_payload_excludes_description_text():
    job = _job()
    spec = _task(job)
    payload = task_payload(job, spec)

    assert "description" not in payload
    assert "SECRET" not in str(payload)
    assert payload["subject"] == spec.subject
    assert payload["scope_paths"] == ["src/retry.py"]
    assert payload["budget_max_usd"] == 2.0


def test_job_end_payload_has_no_objective_text():
    job = _job()
    payload = job_end_payload(job, accepted=1, rejected=0, spend_usd=0.01, halted_reason="")

    assert "objective" not in payload
    assert "ship failover" not in str(payload)
    assert payload["job_id"] == job.id
    assert payload["accepted"] == 1


# ============================================================ real subprocess: veto shape


@pytest.mark.slow
def test_veto_refuses_with_reason():
    code = (
        "import json,sys; sys.stdin.read(); "
        "print(json.dumps({'veto': 'budget policy: over daily cap'})); "
        "sys.exit(2)"
    )
    runner = HookRunner([_hook(HookEvent.PRE_ROUTE, code)])
    job, spec = _job(), _task(_job())

    result = runner.run(HookEvent.PRE_ROUTE, task_payload(job, spec))

    assert result.vetoed
    assert result.veto_reason == "budget policy: over daily cap"
    assert result.warnings == []


@pytest.mark.slow
def test_exit_2_without_veto_shape_proceeds_with_warning():
    """A hook that exits 2 without the documented {"veto": reason} payload is
    malformed, not a clean refusal — it must proceed, not silently deny."""
    code = "import sys; sys.stdin.read(); sys.exit(2)"
    runner = HookRunner([_hook(HookEvent.PRE_EXECUTE, code)])
    job, spec = _job(), _task(_job())

    result = runner.run(HookEvent.PRE_EXECUTE, task_payload(job, spec))

    assert not result.vetoed
    assert len(result.warnings) == 1
    assert "veto" in result.warnings[0]


@pytest.mark.slow
def test_crash_proceeds_with_warning():
    code = "import sys; sys.stdin.read(); raise RuntimeError('boom')"
    runner = HookRunner([_hook(HookEvent.PRE_ROUTE, code)])
    job, spec = _job(), _task(_job())

    result = runner.run(HookEvent.PRE_ROUTE, task_payload(job, spec))

    assert not result.vetoed
    assert len(result.warnings) == 1
    assert "exited" in result.warnings[0]


@pytest.mark.slow
def test_timeout_proceeds_with_warning():
    code = "import sys, time; sys.stdin.read(); time.sleep(5)"
    runner = HookRunner([_hook(HookEvent.PRE_ROUTE, code, timeout=0.2)])
    job, spec = _job(), _task(_job())

    result = runner.run(HookEvent.PRE_ROUTE, task_payload(job, spec))

    assert not result.vetoed
    assert len(result.warnings) == 1
    assert "timed out" in result.warnings[0]


@pytest.mark.slow
def test_nonzero_exit_proceeds_with_warning():
    code = "import sys; sys.stdin.read(); sys.exit(1)"
    runner = HookRunner([_hook(HookEvent.PRE_ROUTE, code)])
    job, spec = _job(), _task(_job())

    result = runner.run(HookEvent.PRE_ROUTE, task_payload(job, spec))

    assert not result.vetoed
    assert len(result.warnings) == 1
    assert "exited 1" in result.warnings[0]


@pytest.mark.slow
def test_success_produces_no_warning_and_no_veto():
    code = "import sys; sys.stdin.read(); sys.exit(0)"
    runner = HookRunner([_hook(HookEvent.PRE_ROUTE, code)])
    job, spec = _job(), _task(_job())

    result = runner.run(HookEvent.PRE_ROUTE, task_payload(job, spec))

    assert not result.vetoed
    assert result.warnings == []


@pytest.mark.slow
def test_never_uses_shell(monkeypatch):
    """subprocess.run must be called with an argv list, never shell=True —
    a hook command is external config, not something to string-interpolate
    into a shell."""
    seen = {}
    real_run = subprocess.run

    def _spy(*args, **kwargs):
        seen["shell"] = kwargs.get("shell", False)
        seen["args_is_list"] = isinstance(args[0], list)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", _spy)
    code = "import sys; sys.stdin.read(); sys.exit(0)"
    runner = HookRunner([_hook(HookEvent.PRE_ROUTE, code)])
    runner.run(HookEvent.PRE_ROUTE, task_payload(_job(), _task(_job())))

    assert seen["args_is_list"]
    assert not seen["shell"]
