"""ForgeBench tests: the Release 0.1 exit gate harness.

No network anywhere in this file. Two distinct test doubles, matching the
two things the task spec asked to be faked separately:

- `FakeExecutor` -- a canned `ArmExecutor` double that bypasses prompt
  building, `Gateway`, and the ledger entirely. Used for testing ForgeBench's
  own orchestration logic (acceptance evaluation, budget aborts, aggregate
  maths) in isolation.
- `FakeTransport` -- the same scripted-`Transport` double `tests/test_gateway.py`
  already uses, wired into a REAL `Gateway` and a REAL `Ledger(":memory:")`.
  Used for testing `GatewayExecutor`'s integration with the real pricing/
  ledger code path, with zero network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from forgeos.catalog import Catalog, ModelCard
from forgeos.contracts import JobSpec, Scope
from forgeos.economy.savings import Provenance
from forgeos.gateway.client import Gateway, RawCallResult
from forgeos.ledger import Ledger

from forgeos import forgebench
from forgeos.forgebench import (
    AcceptanceCheck,
    ArmCallResult,
    ArmKind,
    BudgetExhausted,
    DEFAULT_SUITE,
    GatewayExecutor,
    PinnedTask,
    SavingsClass,
    build_parser,
    build_report,
    keyword_groups,
    run_suite,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------- doubles


class FakeExecutor:
    """Canned, deterministic results per (task_id, arm). No network, no
    gateway, no ledger -- exercises ForgeBench's own orchestration logic."""

    def __init__(self, results: dict[tuple[str, ArmKind], ArmCallResult | Exception]) -> None:
        self._results = results
        self.calls: list[tuple[str, ArmKind]] = []

    def run(self, task: PinnedTask, arm: ArmKind) -> ArmCallResult:
        self.calls.append((task.id, arm))
        outcome = self._results[(task.id, arm)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class NeverCallExecutor:
    """Fails the test outright if invoked -- proves the pre-flight budget
    guard aborted before touching the executor at all."""

    def run(self, task: PinnedTask, arm: ArmKind) -> ArmCallResult:
        pytest.fail(f"executor called for {task.id}/{arm.value} -- pre-flight guard should have aborted")


class FakeTransport:
    """A transport double whose behavior is fully scripted; never touches a
    network. Same shape as tests/test_gateway.py's own FakeTransport."""

    def __init__(self, name: str, responder) -> None:
        self.name = name
        self._responder = responder
        self.calls: list[dict] = []

    def complete(self, *, model_id, prompt, max_output_tokens, reasoning_effort, tools_schema):
        self.calls.append(dict(model_id=model_id, prompt=prompt, max_output_tokens=max_output_tokens))
        return self._responder(prompt)


def _call(
    text: str = "ok", tokens_in: int = 10, tokens_out: int = 5,
    tokens_cached_in: int = 0, usd_micros: int = 100, seconds: float = 0.01,
) -> ArmCallResult:
    return ArmCallResult(
        text=text, tokens_in=tokens_in, tokens_out=tokens_out,
        tokens_cached_in=tokens_cached_in, usd_micros=usd_micros, seconds=seconds,
    )


def _task(task_id: str, *, predicate=None, command=None) -> PinnedTask:
    if predicate is None and command is None:
        predicate = keyword_groups(("ok",))  # a harmless, always-constructible default
    return PinnedTask(
        id=task_id,
        objective=f"objective for {task_id}",
        scope=Scope(paths=["forgeos/contracts.py"]),
        acceptance=AcceptanceCheck(
            description=f"check for {task_id}", predicate=predicate, command=command,
        ),
    )


PRICED_CARD = ModelCard(
    model_id="m", provider="p", input_cost_per_1m=3.0, output_cost_per_1m=15.0,
)


# -------------------------------------------------------------- AcceptanceCheck


def test_acceptance_check_requires_exactly_one_of_predicate_or_command():
    with pytest.raises(ValueError):
        AcceptanceCheck(description="neither")
    with pytest.raises(ValueError):
        AcceptanceCheck(description="both", predicate=lambda t: True, command=("true",))


def test_predicate_acceptance_evaluates_text():
    check = AcceptanceCheck(description="d", predicate=keyword_groups(("yes",)))
    assert check.evaluate("Yes, that is correct") is True
    assert check.evaluate("no") is False


def test_keyword_groups_requires_all_groups_to_hit():
    pred = keyword_groups(("record_spend",), ("dedup", "coalesce"))
    assert pred("record_spend prevents dedup") is True
    assert pred("record_spend only") is False
    assert pred("dedup only") is False


def test_command_acceptance_runs_against_temp_file_and_checks_exit_code():
    check = AcceptanceCheck(
        description="contains OK",
        command=(sys.executable, "-c",
                 "import sys; sys.exit(0 if 'OK' in open(sys.argv[1]).read() else 1)",
                 "{output}"),
    )
    assert check.evaluate("the answer is OK") is True
    assert check.evaluate("the answer is not right") is False


def test_command_acceptance_survives_a_broken_command():
    check = AcceptanceCheck(description="broken", command=("this-executable-does-not-exist-xyz",))
    assert check.evaluate("anything") is False


def test_json_array_task_accepts_and_rejects_correctly():
    json_task = next(t for t in DEFAULT_SUITE if t.id == "contracts-micros-json")
    assert json_task.acceptance.evaluate('["to_micros", "from_micros"]') is True
    assert json_task.acceptance.evaluate('["to_micros"]') is False
    assert json_task.acceptance.evaluate("to_micros and from_micros") is False  # not JSON


# ------------------------------------------------------------------ PinnedTask


def test_pinned_task_rejects_blank_id_or_objective():
    with pytest.raises(ValueError):
        _task("")
    with pytest.raises(ValueError):
        PinnedTask(
            id="x", objective="   ", scope=Scope(paths=[]),
            acceptance=AcceptanceCheck(description="d", predicate=lambda t: True),
        )


def test_contract_dict_is_stable_and_json_safe():
    import json

    task = _task("t1", predicate=lambda t: True)
    blob1 = json.dumps(task.contract_dict(), sort_keys=True)
    blob2 = json.dumps(task.contract_dict(), sort_keys=True)
    assert blob1 == blob2
    assert task.contract_dict()["id"] == "t1"


def test_default_suite_has_five_to_eight_tasks_all_mechanical():
    assert 5 <= len(DEFAULT_SUITE) <= 8
    kinds = [
        "command" if t.acceptance.command is not None else "predicate"
        for t in DEFAULT_SUITE
    ]
    assert "command" in kinds  # at least one exit-code-style check
    assert "predicate" in kinds  # at least one deterministic-assertion-style check
    assert len({t.id for t in DEFAULT_SUITE}) == len(DEFAULT_SUITE)  # unique ids


# --------------------------------------------------------- acceptance differs


def test_acceptance_differs_voids_the_comparison():
    """Task A: baseline passes, forgeos fails. Task B: both pass. The
    per-task comparison must flag A as void; the aggregate must void the
    whole suite's savings claim rather than print a ratio."""
    t_a = _task("a", predicate=lambda text: text == "right")
    t_b = _task("b", predicate=lambda text: text == "right")
    executor = FakeExecutor({
        ("a", ArmKind.BASELINE): _call("right", usd_micros=100),
        ("a", ArmKind.FORGEOS): _call("wrong", usd_micros=10),
        ("b", ArmKind.BASELINE): _call("right", usd_micros=100),
        ("b", ArmKind.FORGEOS): _call("right", usd_micros=10),
    })
    run = run_suite(
        [t_a, t_b], suite_name="void-test", executor=executor,
        savings_class=SavingsClass.A_PAIRED_MEASURED, budget_usd_micros=10_000_000,
    )
    report = build_report(run, mission_id="m", repo_revision="rev")

    comp_a = next(c for c in report.comparisons if c.task_id == "a")
    comp_b = next(c for c in report.comparisons if c.task_id == "b")
    assert comp_a.comparison_voided is True
    assert comp_b.comparison_voided is False

    assert report.comparison_voided is True
    assert report.proof.savings == {}
    assert report.exit_gate_passed is False

    rendered = forgebench.render_report(report)
    assert "COST COMPARISON VOID" in rendered
    assert "VOID (acceptance differs)" in rendered


def test_verify_proof_clean_even_when_voided_by_disagreement():
    """Voiding is an honest PARTIAL outcome with an empty savings dict, not a
    receipt-integrity violation -- verify_proof must have nothing to say."""
    t_a = _task("a", predicate=lambda text: text == "right")
    executor = FakeExecutor({
        ("a", ArmKind.BASELINE): _call("right"),
        ("a", ArmKind.FORGEOS): _call("wrong"),
    })
    run = run_suite(
        [t_a], suite_name="void-clean", executor=executor,
        savings_class=SavingsClass.A_PAIRED_MEASURED, budget_usd_micros=10_000_000,
    )
    report = build_report(run, mission_id="m", repo_revision="rev")
    assert report.proof_complaints == []


# --------------------------------------------------------------------- class D


def test_class_d_refuses_savings_claim():
    t_a = _task("a", predicate=lambda text: text == "right")
    executor = FakeExecutor({("a", ArmKind.FORGEOS): _call("right", usd_micros=10)})
    run = run_suite(
        [t_a], suite_name="class-d", executor=executor,
        savings_class=SavingsClass.D_NO_BASELINE, budget_usd_micros=10_000_000,
    )
    # only the forgeos arm should ever be attempted
    assert executor.calls == [("a", ArmKind.FORGEOS)]

    report = build_report(run, mission_id="m", repo_revision="rev")
    assert report.proof.baseline.figures == {}
    assert report.proof.savings == {}
    assert report.exit_gate_passed is None
    assert report.proof.baseline.baseline_provenance is Provenance.UNKNOWN

    assert report.proof_complaints == [
        "missing evidence hash: baseline_trace_hash",
    ]

    rendered = forgebench.render_report(report)
    assert "class=D" in rendered
    assert "no saving is claimed" in rendered
    assert "N/A" in rendered  # exit gate is not evaluable without a baseline
    # The SAVINGS section (between its header and EVIDENCE) must be empty --
    # no cash-cost percentage is ever printed when there is no baseline.
    savings_section = rendered.split("\nSAVINGS")[-1].split("\nEVIDENCE")[0]
    assert "cash_cost" not in savings_section


# --------------------------------------------------------------- budget ceiling


def test_budget_ceiling_aborts_mid_run():
    t1, t2, t3 = _task("t1", predicate=lambda t: True), _task("t2", predicate=lambda t: True), _task(
        "t3", predicate=lambda t: True
    )
    executor = FakeExecutor({
        ("t1", ArmKind.BASELINE): _call("x"),
        ("t1", ArmKind.FORGEOS): _call("x"),
        ("t2", ArmKind.BASELINE): _call("x"),
        ("t2", ArmKind.FORGEOS): BudgetExhausted("t2/forgeos: fake budget exhaustion"),
        # t3 must never be attempted
        ("t3", ArmKind.BASELINE): _call("x"),
        ("t3", ArmKind.FORGEOS): _call("x"),
    })
    run = run_suite(
        [t1, t2, t3], suite_name="abort-mid", executor=executor,
        savings_class=SavingsClass.A_PAIRED_MEASURED, budget_usd_micros=10_000_000,
    )
    assert run.aborted_reason != ""
    assert run.aborted_at_task == "t2"
    assert ("t3", ArmKind.BASELINE) not in executor.calls
    assert ("t3", ArmKind.FORGEOS) not in executor.calls
    assert len(run.outcomes) == 3  # t1 both arms + t2 baseline only

    report = build_report(run, mission_id="m", repo_revision="rev")
    assert report.exit_gate_passed is None


def test_budget_ceiling_aborts_before_any_call_when_preflight_exceeds_budget():
    tasks = list(DEFAULT_SUITE)
    run = run_suite(
        tasks, suite_name="preflight-abort", executor=NeverCallExecutor(),
        savings_class=SavingsClass.A_PAIRED_MEASURED,
        budget_usd_micros=1,  # $0.000001 -- unaffordable for any real task
        root=REPO_ROOT, card=PRICED_CARD,
    )
    assert run.aborted_reason != ""
    assert run.outcomes == []


def test_dry_run_prices_the_whole_suite_and_makes_zero_calls():
    tasks = list(DEFAULT_SUITE)
    run = run_suite(
        tasks, suite_name="dry-run-test", executor=None,
        savings_class=SavingsClass.A_PAIRED_MEASURED, budget_usd_micros=10_000_000,
        dry_run=True, root=REPO_ROOT, card=PRICED_CARD,
    )
    assert run.dry_run is True
    assert run.dry_run_estimate_usd_micros > 0
    assert run.outcomes == []
    report = build_report(run, mission_id="m", repo_revision="rev")
    rendered = forgebench.render_report(report)
    assert "DRY RUN" in rendered


# --------------------------------------------------------- per-task / aggregate maths


def test_per_task_and_aggregate_math_is_exact():
    tasks = [_task(f"t{i}", predicate=lambda text: text == "PASS") for i in range(3)]
    results = {}
    expected_baseline_micros = 0
    expected_forgeos_micros = 0
    for i, task in enumerate(tasks):
        b_micros, h_micros = 1000 + i, 100 + i
        results[(task.id, ArmKind.BASELINE)] = _call(
            "PASS", tokens_in=500 + i, tokens_out=50, usd_micros=b_micros
        )
        results[(task.id, ArmKind.FORGEOS)] = _call(
            "PASS" if i != 2 else "FAIL", tokens_in=50 + i, tokens_out=20, usd_micros=h_micros
        )
        expected_baseline_micros += b_micros
        expected_forgeos_micros += h_micros

    executor = FakeExecutor(results)
    run = run_suite(
        tasks, suite_name="math-test", executor=executor,
        savings_class=SavingsClass.A_PAIRED_MEASURED, budget_usd_micros=10_000_000,
    )
    assert len(run.outcomes) == 6

    baseline_totals = forgebench.totals_for(run, ArmKind.BASELINE)
    forgeos_totals = forgebench.totals_for(run, ArmKind.FORGEOS)

    assert baseline_totals.usd_micros == expected_baseline_micros
    assert forgeos_totals.usd_micros == expected_forgeos_micros
    assert baseline_totals.accepted_count == 3  # all "PASS"
    assert forgeos_totals.accepted_count == 2  # task index 2 answered "FAIL"
    assert baseline_totals.attempted_count == 3
    assert forgeos_totals.attempted_count == 3

    report = build_report(run, mission_id="m", repo_revision="rev")
    # task 2 disagrees (baseline PASS, forgeos FAIL) -> voided aggregate
    assert report.comparison_voided is True
    assert report.proof.cost_per_accepted_task is not None
    assert report.proof.cost_per_accepted_task.value == pytest.approx(
        forgebench.from_micros(expected_forgeos_micros) / 2
    )


def test_aggregate_math_when_every_task_agrees_and_forgeos_is_cheaper():
    tasks = [_task(f"u{i}", predicate=lambda text: text == "PASS") for i in range(2)]
    results = {}
    for task in tasks:
        results[(task.id, ArmKind.BASELINE)] = _call("PASS", usd_micros=1_000_000)
        results[(task.id, ArmKind.FORGEOS)] = _call("PASS", usd_micros=100_000)
    run = run_suite(
        tasks, suite_name="clean-win", executor=FakeExecutor(results),
        savings_class=SavingsClass.A_PAIRED_MEASURED, budget_usd_micros=10_000_000,
    )
    report = build_report(run, mission_id="m", repo_revision="rev")

    assert report.comparison_voided is False
    assert report.exit_gate_passed is True
    assert report.proof_complaints == []
    assert report.proof.savings["cash_cost"].value == pytest.approx(90.0)  # (2.0-0.2)/2.0*100
    rendered = forgebench.render_report(report)
    assert "RELEASE 0.1 EXIT GATE: PASS" in rendered


def test_regression_is_reported_honestly_not_hidden():
    """forgeos MORE expensive than baseline: exit gate must FAIL, and the
    savings figure must show the real, negative number -- never clamped."""
    tasks = [_task("r1", predicate=lambda text: text == "PASS")]
    results = {
        ("r1", ArmKind.BASELINE): _call("PASS", usd_micros=100_000),
        ("r1", ArmKind.FORGEOS): _call("PASS", usd_micros=500_000),
    }
    run = run_suite(
        tasks, suite_name="regression", executor=FakeExecutor(results),
        savings_class=SavingsClass.A_PAIRED_MEASURED, budget_usd_micros=10_000_000,
    )
    report = build_report(run, mission_id="m", repo_revision="rev")
    assert report.comparison_voided is False
    assert report.exit_gate_passed is False
    assert report.proof.savings["cash_cost"].value < 0
    assert report.proof_complaints == []


# ------------------------------------------------------------- GatewayExecutor


@pytest.fixture()
def ledger():
    led = Ledger(":memory:")
    yield led
    led.close()


def test_gateway_executor_spends_via_real_ledger_and_packs_a_smaller_forgeos_prompt(ledger):
    task = next(t for t in DEFAULT_SUITE if t.id == "contracts-micros-json")

    def responder(prompt: str) -> RawCallResult:
        return RawCallResult(text='["to_micros", "from_micros"]', tokens_in=len(prompt) // 4, tokens_out=10)

    transport = FakeTransport("fake", responder)
    catalog = Catalog([PRICED_CARD])
    gateway = Gateway(catalog=catalog, ledger=ledger, transports=[transport])
    job_id = ledger.open_job(JobSpec(objective="test", cwd=str(REPO_ROOT)))

    executor = GatewayExecutor(
        gateway=gateway, job_id=job_id, model_ref="p/m", root=REPO_ROOT,
        remaining_micros_fn=lambda: 10_000_000,
    )

    baseline_result = executor.run(task, ArmKind.BASELINE)
    forgeos_result = executor.run(task, ArmKind.FORGEOS)

    assert len(transport.calls) == 2
    baseline_prompt = transport.calls[0]["prompt"]
    forgeos_prompt = transport.calls[1]["prompt"]
    assert len(forgeos_prompt) < len(baseline_prompt)  # capsule packing actually engaged

    assert ledger.job_spend_micros(job_id) == baseline_result.usd_micros + forgeos_result.usd_micros
    assert ledger.job_spend_micros(job_id) > 0


def test_gateway_executor_raises_budget_exhausted_on_refusal(ledger):
    task = next(t for t in DEFAULT_SUITE if t.id == "contracts-micros-json")

    def responder(prompt: str) -> RawCallResult:
        return RawCallResult(text="anything", tokens_in=10, tokens_out=10)

    transport = FakeTransport("fake", responder)
    catalog = Catalog([PRICED_CARD])
    gateway = Gateway(catalog=catalog, ledger=ledger, transports=[transport])
    job_id = ledger.open_job(JobSpec(objective="test", cwd=str(REPO_ROOT)))
    executor = GatewayExecutor(
        gateway=gateway, job_id=job_id, model_ref="p/m", root=REPO_ROOT,
        remaining_micros_fn=lambda: 0,  # nothing left -- every call must be refused
    )
    with pytest.raises(BudgetExhausted):
        executor.run(task, ArmKind.BASELINE)


# --------------------------------------------------------------------------- CLI


def test_build_parser_defaults():
    args = build_parser().parse_args([])
    assert args.budget_usd == pytest.approx(0.50)
    assert args.dry_run is False
    assert args.skip_baseline is False
    assert args.ledger_path == ":memory:"
    assert args.model == ""


def test_build_parser_parses_flags():
    args = build_parser().parse_args(
        ["--model", "p/m", "--budget-usd", "1.25", "--dry-run", "--skip-baseline"]
    )
    assert args.model == "p/m"
    assert args.budget_usd == pytest.approx(1.25)
    assert args.dry_run is True
    assert args.skip_baseline is True


def test_main_dry_run_end_to_end_with_injected_catalog(capsys):
    catalog = Catalog([PRICED_CARD])
    rc = forgebench.main(
        ["--dry-run", "--model", "p/m", "--budget-usd", "50"],
        catalog=catalog, root=REPO_ROOT,
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out


def test_main_reports_no_usable_model_when_none_resolvable(capsys):
    rc = forgebench.main(
        ["--dry-run", "--model", "nope/does-not-exist"], catalog=Catalog([]), root=REPO_ROOT,
    )
    assert rc == 1
    assert "no usable model" in capsys.readouterr().out
