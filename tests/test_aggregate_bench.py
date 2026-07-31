from __future__ import annotations

import json
from pathlib import Path

from forgeos import forgebench_table as aggregate_bench


def _receipt(
    *,
    mode: str = "live",
    provenance: str = "measured",
    savings_class: str = "A",
    baseline_accepted: int = 2,
    forgeos_accepted: int = 2,
    baseline_usd_micros: int = 1_000_000,
    forgeos_usd_micros: int = 700_000,
    comparison_voided: bool = False,
    exit_gate_passed: bool | None = True,
    aborted_reason: str = "",
) -> dict:
    return {
        "schema": "forgeos.forgebench.v1",
        "mode": mode,
        "provenance": provenance,
        "suite": {
            "name": "forgeos-default",
            "savings_class": savings_class,
            "tasks": [{"id": "one"}, {"id": "two"}],
        },
        "aborted": {"reason": aborted_reason, "at_task": ""},
        "comparison_voided": comparison_voided,
        "exit_gate_passed": exit_gate_passed,
        "totals": {
            "baseline": {
                "accepted_count": baseline_accepted,
                "attempted_count": 2,
                "usd_micros": baseline_usd_micros,
            },
            "forgeos": {
                "accepted_count": forgeos_accepted,
                "attempted_count": 2,
                "usd_micros": forgeos_usd_micros,
            },
        },
        "proof": {"mission_id": "mission-1", "repo_revision": "abc123"},
    }


def test_mixed_receipts_preserve_excluded_runs_and_gate_savings(tmp_path: Path) -> None:
    receipts = [
        _receipt(),
        _receipt(
            comparison_voided=True,
            forgeos_accepted=1,
            exit_gate_passed=False,
        ),
        _receipt(mode="dry-run", provenance="modelled"),
    ]
    inputs = []
    for index, receipt in enumerate(receipts):
        path = tmp_path / f"receipt-{index}.json"
        path.write_text(json.dumps(receipt), encoding="utf-8")
        inputs.append((path, receipt))

    table = aggregate_bench.aggregate_receipts(inputs)

    assert table["schema"] == "forgeos.forgebench_table.v1"
    assert table["summary"]["total_runs"] == 3
    assert table["summary"]["eligible_runs"] == 1
    assert table["summary"]["excluded_runs"] == 2
    assert table["summary"]["status_counts"] == {
        "DRY-RUN": 1,
        "PASS": 1,
        "VOID": 1,
    }
    assert table["summary"]["eligible_totals"]["savings_pct"] == 30.0
    assert [row["status"] for row in table["runs"]] == ["PASS", "VOID", "DRY-RUN"]
    assert table["runs"][1]["eligibility_reason"] == (
        "acceptance differed, so the cost comparison is void"
    )


def test_markdown_shows_statuses_and_exclusion_boundary() -> None:
    table = aggregate_bench.aggregate_receipts(
        [("live.json", _receipt()), ("void.json", _receipt(comparison_voided=True))]
    )

    markdown = aggregate_bench.render_markdown(table)

    assert "**VOID**" in markdown
    assert "DRY-RUN" not in markdown
    assert "30.0%" in markdown
    assert "Excluded runs are shown above" in markdown


def test_invalid_schema_is_rejected_without_partial_table(tmp_path: Path) -> None:
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps({"schema": "other"}), encoding="utf-8")

    assert aggregate_bench.main([str(path)]) == 2
