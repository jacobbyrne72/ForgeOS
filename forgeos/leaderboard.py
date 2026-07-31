"""Build a correctness-gated cost-per-accepted-task leaderboard.

This is an aggregation/reporting surface only: it reads ForgeBench receipts,
never calls a provider, and ranks only measured live Class-A runs with matching
acceptance and at least one accepted ForgeOS task. Runs that cannot support a
fair cost comparison remain visible with an exclusion reason.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .forgebench_table import _expand_paths, load_receipt, row_for_receipt

LEADERBOARD_SCHEMA = "forgeos.leaderboard.v1"


def _label(row: dict[str, Any], receipt: dict[str, Any], path: str | Path) -> str:
    """Choose an explicit model/fleet label, with a truthful file fallback."""
    return str(
        receipt.get("label")
        or receipt.get("model_ref")
        or row.get("model_ref")
        or Path(path).stem
    )


def build_leaderboard(
    receipts: list[tuple[str | Path, dict[str, Any]]],
) -> dict[str, Any]:
    """Aggregate comparable receipts and rank their measured unit cost."""
    runs: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "runs": 0,
            "accepted_count": 0,
            "usd_micros": 0,
            "gate_passed_runs": 0,
            "sources": [],
        }
    )
    for path, receipt in receipts:
        row = row_for_receipt(path, receipt)
        label = _label(row, receipt, path)
        accepted = int(row["forgeos"]["accepted_count"])
        eligible = bool(row["savings_eligible"]) and accepted > 0
        row["leaderboard_label"] = label
        row["leaderboard_eligible"] = eligible
        if not row["savings_eligible"]:
            row["leaderboard_reason"] = row["eligibility_reason"]
        elif accepted <= 0:
            row["leaderboard_reason"] = "no accepted ForgeOS tasks"
        else:
            row["leaderboard_reason"] = "measured live run with accepted work"
        runs.append(row)
        if not eligible:
            continue
        key = (label, row["suite"], row["contract_hash"])
        group = groups[key]
        group["runs"] += 1
        group["accepted_count"] += accepted
        group["usd_micros"] += int(row["forgeos"]["usd_micros"])
        group["gate_passed_runs"] += int(row["status"] == "PASS")
        group["sources"].append(str(path))

    entries = []
    for (label, suite, contract_hash), group in groups.items():
        entries.append({
            "label": label,
            "suite": suite,
            "contract_hash": contract_hash,
            "runs": group["runs"],
            "gate_passed_runs": group["gate_passed_runs"],
            "accepted_count": group["accepted_count"],
            "usd_micros": group["usd_micros"],
            "cost_per_accepted_usd": (
                group["usd_micros"] / group["accepted_count"] / 1_000_000
            ),
            "sources": group["sources"],
        })
    entries.sort(key=lambda item: (item["cost_per_accepted_usd"], item["label"], item["suite"]))
    for rank, entry in enumerate(entries, start=1):
        entry["rank"] = rank

    eligible_runs = sum(1 for row in runs if row["leaderboard_eligible"])
    fleet_accepted_count = sum(
        int(row["forgeos"]["accepted_count"])
        for row in runs
        if row["leaderboard_eligible"]
    )
    fleet_usd_micros = sum(
        int(row["forgeos"]["usd_micros"])
        for row in runs
        if row["leaderboard_eligible"]
    )
    fleet_rollup = {
        "eligible_runs": eligible_runs,
        "accepted_count": fleet_accepted_count,
        "usd_micros": fleet_usd_micros,
        "cost_per_accepted_usd": (
            fleet_usd_micros / fleet_accepted_count / 1_000_000
            if fleet_accepted_count
            else None
        ),
        "scope": (
            "Descriptive cumulative total across eligible measured live Class-A "
            "receipts; not used as a cross-suite ranking."
        ),
    }
    return {
        "schema": LEADERBOARD_SCHEMA,
        "entries": entries,
        "runs": runs,
        "fleet_rollup": fleet_rollup,
        "summary": {
            "total_runs": len(runs),
            "eligible_runs": eligible_runs,
            "excluded_runs": len(runs) - eligible_runs,
            "ranked_entries": len(entries),
            "note": (
                "Ranks measured live Class-A receipts with matching acceptance and "
                "at least one accepted ForgeOS task; excluded runs remain in runs[]."
            ),
        },
    }


def render_markdown(board: dict[str, Any]) -> str:
    summary = board["summary"]
    lines = [
        "# ForgeOS cost-per-accepted-task leaderboard",
        "",
        (
            f"Runs: {summary['total_runs']} | eligible: {summary['eligible_runs']} | "
            f"excluded: {summary['excluded_runs']}"
        ),
        (
            f"Fleet rollup (eligible receipts): {board['fleet_rollup']['accepted_count']} "
            f"accepted | ${board['fleet_rollup']['usd_micros'] / 1_000_000:.6f} measured "
            f"ForgeOS spend | ${board['fleet_rollup']['cost_per_accepted_usd']:.6f} / accepted"
            if board["fleet_rollup"]["cost_per_accepted_usd"] is not None
            else "Fleet rollup (eligible receipts): no accepted work"
        ),
        "",
        "| Rank | Model / fleet | Suite | Runs | Accepted | Cost / accepted | Gate passes |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for entry in board["entries"]:
        lines.append(
            f"| {entry['rank']} | `{entry['label']}` | `{entry['suite']}` | "
            f"{entry['runs']} | {entry['accepted_count']} | "
            f"${entry['cost_per_accepted_usd']:.6f} | "
            f"{entry['gate_passed_runs']} |"
        )
    lines.extend(["", "Excluded runs:"])
    excluded = [row for row in board["runs"] if not row["leaderboard_eligible"]]
    if excluded:
        for row in excluded:
            lines.append(f"- `{row['leaderboard_label']}`: {row['leaderboard_reason']}")
    else:
        lines.append("- none")
    return "\n".join(lines)


def write_json(path: str | Path, board: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(board, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="ForgeBench JSON files or directories")
    parser.add_argument("--json-out", default="", help="write the leaderboard as JSON")
    args = parser.parse_args(argv)
    paths = _expand_paths(args.paths)
    if not paths:
        parser.error("no receipt files found")
    try:
        board = build_leaderboard([(path, load_receipt(path)) for path in paths])
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    print(render_markdown(board))
    if args.json_out:
        print(f"JSON leaderboard: {write_json(args.json_out, board)}")
    return 0


__all__ = ["LEADERBOARD_SCHEMA", "build_leaderboard", "main", "render_markdown", "write_json"]
