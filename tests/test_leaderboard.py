from __future__ import annotations

import json
from pathlib import Path

from forgeos import leaderboard


def _receipt(
    *,
    model_ref: str,
    baseline_accepted: int = 2,
    forgeos_accepted: int = 2,
    baseline_usd_micros: int = 2_000_000,
    forgeos_usd_micros: int = 1_000_000,
    mode: str = "live",
    provenance: str = "measured",
    comparison_voided: bool = False,
) -> dict:
    return {
        "schema": "forgeos.forgebench.v1",
        "model_ref": model_ref,
        "mode": mode,
        "provenance": provenance,
        "suite": {
            "name": "pinned-suite",
            "savings_class": "A",
            "tasks": [{"id": "one"}, {"id": "two"}],
        },
        "aborted": {"reason": "", "at_task": ""},
        "comparison_voided": comparison_voided,
        "exit_gate_passed": True,
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
        "proof": {
            "mission_id": f"mission-{model_ref}",
            "repo_revision": "abc123",
            "contract_hash": "contract-v1",
        },
    }


def test_leaderboard_aggregates_same_model_and_ranks_unit_cost() -> None:
    board = leaderboard.build_leaderboard([
        ("model-a-1.json", _receipt(model_ref="provider/model-a")),
        ("model-a-2.json", _receipt(
            model_ref="provider/model-a", baseline_accepted=1, forgeos_accepted=1,
            baseline_usd_micros=800_000, forgeos_usd_micros=400_000,
        )),
        ("model-b.json", _receipt(model_ref="provider/model-b", forgeos_usd_micros=2_000_000)),
    ])

    assert board["schema"] == "forgeos.leaderboard.v1"
    assert board["summary"] == {
        "total_runs": 3,
        "eligible_runs": 3,
        "excluded_runs": 0,
        "ranked_entries": 2,
        "note": board["summary"]["note"],
    }
    assert [entry["label"] for entry in board["entries"]] == [
        "provider/model-a", "provider/model-b",
    ]
    assert board["entries"][0]["rank"] == 1
    assert board["entries"][0]["accepted_count"] == 3
    assert board["entries"][0]["cost_per_accepted_usd"] == 0.4666666666666667


def test_leaderboard_preserves_unrankable_receipts_with_reasons() -> None:
    board = leaderboard.build_leaderboard([
        ("dry.json", _receipt(model_ref="dry", mode="dry-run", provenance="modelled")),
        ("void.json", _receipt(model_ref="void", comparison_voided=True)),
        ("zero.json", _receipt(model_ref="zero", baseline_accepted=0, forgeos_accepted=0)),
    ])

    assert board["entries"] == []
    assert board["summary"]["excluded_runs"] == 3
    assert [row["leaderboard_reason"] for row in board["runs"]] == [
        "not a measured live receipt",
        "acceptance differed, so the cost comparison is void",
        "no accepted ForgeOS tasks",
    ]


def test_leaderboard_main_writes_machine_receipt_and_markdown(tmp_path: Path, capsys) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(_receipt(model_ref="provider/model")), encoding="utf-8")
    output = tmp_path / "leaderboard.json"

    assert leaderboard.main([str(receipt), "--json-out", str(output)]) == 0
    text = capsys.readouterr().out
    assert "cost-per-accepted-task leaderboard" in text
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["entries"][0]["label"] == "provider/model"
