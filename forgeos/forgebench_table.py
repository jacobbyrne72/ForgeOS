"""Build a correctness-gated public table from ForgeBench JSON receipts.

This module is deliberately model-free and read-only. Every input remains
visible, while savings are aggregated only from measured live Class-A runs
whose acceptance counts match. Dry-runs, failures, aborts, and voided
comparisons are retained with an exclusion reason.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

RECEIPT_SCHEMA = "forgeos.forgebench.v1"
TABLE_SCHEMA = "forgeos.forgebench_table.v1"


def _arm_totals(receipt: dict[str, Any], arm: str) -> dict[str, int]:
    totals = receipt.get("totals", {}).get(arm)
    if not isinstance(totals, dict):
        raise ValueError(f"receipt has no totals.{arm} object")
    required = ("accepted_count", "attempted_count", "usd_micros")
    if any(key not in totals for key in required):
        raise ValueError(f"receipt totals.{arm} is missing a required counter")
    return {key: int(totals[key]) for key in required}


def _status(receipt: dict[str, Any]) -> str:
    if receipt.get("mode") == "dry-run":
        return "DRY-RUN"
    aborted = receipt.get("aborted", {})
    if isinstance(aborted, dict) and aborted.get("reason"):
        return "ABORTED"
    if receipt.get("comparison_voided"):
        return "VOID"
    if receipt.get("exit_gate_passed") is True:
        return "PASS"
    if receipt.get("exit_gate_passed") is False:
        return "FAIL"
    return "INCONCLUSIVE"


def _eligibility(
    receipt: dict[str, Any], totals: dict[str, dict[str, int]]
) -> tuple[bool, str]:
    if receipt.get("mode") != "live" or receipt.get("provenance") != "measured":
        return False, "not a measured live receipt"
    if receipt.get("suite", {}).get("savings_class") != "A":
        return False, "not a paired Class-A receipt"
    if receipt.get("comparison_voided"):
        return False, "acceptance differed, so the cost comparison is void"
    if totals["baseline"]["attempted_count"] == 0 or totals["forgeos"]["attempted_count"] == 0:
        return False, "one arm has no completed calls"
    if totals["baseline"]["accepted_count"] != totals["forgeos"]["accepted_count"]:
        return False, "accepted counts differ"
    return True, "paired measured run with matching acceptance"


def _cost_per_accepted_usd(totals: dict[str, int]) -> float | None:
    accepted = totals["accepted_count"]
    return None if accepted <= 0 else totals["usd_micros"] / accepted / 1_000_000


def row_for_receipt(path: str | Path, receipt: dict[str, Any]) -> dict[str, Any]:
    """Normalize one validated ForgeBench receipt into a table row."""
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise ValueError(
            f"{path}: expected schema {RECEIPT_SCHEMA!r}, got {receipt.get('schema')!r}"
        )
    suite = receipt.get("suite")
    if not isinstance(suite, dict):
        raise ValueError(f"{path}: missing suite object")
    totals = {arm: _arm_totals(receipt, arm) for arm in ("baseline", "forgeos")}
    eligible, reason = _eligibility(receipt, totals)
    proof = receipt.get("proof") if isinstance(receipt.get("proof"), dict) else {}
    return {
        "source": str(path),
        "run_id": proof.get("mission_id") or suite.get("name") or Path(path).stem,
        "repo_revision": proof.get("repo_revision", ""),
        "suite": suite.get("name", ""),
        "mode": receipt.get("mode", "unknown"),
        "savings_class": suite.get("savings_class", ""),
        "status": _status(receipt),
        "tasks": len(suite.get("tasks", [])),
        **{
            arm: {
                **totals[arm],
                "cost_per_accepted_usd": _cost_per_accepted_usd(totals[arm]),
            }
            for arm in ("baseline", "forgeos")
        },
        "savings_eligible": eligible,
        "eligibility_reason": reason,
    }


def aggregate_receipts(receipts: list[tuple[str | Path, dict[str, Any]]]) -> dict[str, Any]:
    """Build a table and a correctness-gated aggregate summary."""
    rows = [row_for_receipt(path, receipt) for path, receipt in receipts]
    eligible = [row for row in rows if row["savings_eligible"]]
    baseline_usd = sum(row["baseline"]["usd_micros"] for row in eligible)
    forgeos_usd = sum(row["forgeos"]["usd_micros"] for row in eligible)
    baseline_accepted = sum(row["baseline"]["accepted_count"] for row in eligible)
    forgeos_accepted = sum(row["forgeos"]["accepted_count"] for row in eligible)
    savings_pct = None if baseline_usd == 0 else round((1 - forgeos_usd / baseline_usd) * 100, 6)
    return {
        "schema": TABLE_SCHEMA,
        "runs": rows,
        "summary": {
            "total_runs": len(rows),
            "eligible_runs": len(eligible),
            "excluded_runs": len(rows) - len(eligible),
            "status_counts": {
                status: sum(row["status"] == status for row in rows)
                for status in sorted({row["status"] for row in rows})
            },
            "eligible_totals": {
                "baseline_usd_micros": baseline_usd,
                "forgeos_usd_micros": forgeos_usd,
                "baseline_accepted": baseline_accepted,
                "forgeos_accepted": forgeos_accepted,
                "baseline_cost_per_accepted_usd": (
                    baseline_usd / baseline_accepted / 1_000_000 if baseline_accepted else None
                ),
                "forgeos_cost_per_accepted_usd": (
                    forgeos_usd / forgeos_accepted / 1_000_000 if forgeos_accepted else None
                ),
                "savings_pct": savings_pct,
            },
            "note": (
                "Savings aggregate uses only measured live Class-A receipts with "
                "matching acceptance; excluded runs remain in runs[]."
            ),
        },
    }


def load_receipt(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{target}: cannot read JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{target}: root JSON value must be an object")
    return value


def _expand_paths(paths: list[str]) -> list[Path]:
    expanded: list[Path] = []
    for raw in paths:
        target = Path(raw)
        expanded.extend(sorted(target.glob("*.json")) if target.is_dir() else [target])
    return expanded


def render_markdown(table: dict[str, Any]) -> str:
    summary = table["summary"]
    totals = summary["eligible_totals"]
    savings = "n/a" if totals["savings_pct"] is None else f"{totals['savings_pct']:.1f}%"
    lines = [
        "# ForgeBench receipt table",
        "",
        f"Runs: {summary['total_runs']} | eligible: {summary['eligible_runs']} | excluded: {summary['excluded_runs']}",
        "",
        "| Run | Mode | Status | Baseline accepted / USD | ForgeOS accepted / USD | Eligible |",
        "|---|---|---|---:|---:|:---:|",
    ]
    for row in table["runs"]:
        lines.append(
            f"| `{row['run_id']}` | {row['mode']} | **{row['status']}** | "
            f"{row['baseline']['accepted_count']} / ${row['baseline']['usd_micros'] / 1e6:.6f} | "
            f"{row['forgeos']['accepted_count']} / ${row['forgeos']['usd_micros'] / 1e6:.6f} | "
            f"{'yes' if row['savings_eligible'] else 'no'} |"
        )
    baseline = totals["baseline_cost_per_accepted_usd"]
    forgeos = totals["forgeos_cost_per_accepted_usd"]
    lines.extend(
        [
            "",
            "## Eligible aggregate",
            "",
            f"- Baseline cost per accepted task: {f'${baseline:.6f}' if baseline is not None else 'n/a'}",
            f"- ForgeOS cost per accepted task: {f'${forgeos:.6f}' if forgeos is not None else 'n/a'}",
            f"- Correctness-gated savings: **{savings}**",
            "- Excluded runs are shown above and do not contribute to the savings figure.",
        ]
    )
    return "\n".join(lines)


def write_json(path: str | Path, table: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="ForgeBench JSON files or directories")
    parser.add_argument("--json-out", default="", help="write the aggregate table as JSON")
    args = parser.parse_args(argv)
    paths = _expand_paths(args.paths)
    if not paths:
        parser.error("no receipt files found")
    try:
        table = aggregate_receipts([(path, load_receipt(path)) for path in paths])
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    print(render_markdown(table))
    if args.json_out:
        print(f"JSON table: {write_json(args.json_out, table)}")
    return 0
