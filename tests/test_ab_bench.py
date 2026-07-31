from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from forgeos.catalog import Catalog, ModelCard
from forgeos.settings import AuthMode, Provider, ProviderKind, Settings


_PATH = Path(__file__).parents[1] / "tools" / "ab_bench.py"
_SPEC = importlib.util.spec_from_file_location("forgeos_ab_bench", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
ab_bench = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = ab_bench
_SPEC.loader.exec_module(ab_bench)


def _settings() -> Settings:
    return Settings(
        providers={
            "deepseek": Provider(
                name="deepseek", kind=ProviderKind.API, auth=AuthMode.NONE
            )
        }
    )


def test_ab_bench_defaults_to_a_no_call_dry_run(tmp_path, monkeypatch, capsys):
    card = ModelCard(
        model_id="deepseek-chat",
        provider="deepseek",
        input_cost_per_1m=1.0,
        output_cost_per_1m=2.0,
        context=100_000,
    )
    monkeypatch.setattr(ab_bench, "default_catalog", lambda: Catalog([card]))
    monkeypatch.setattr(ab_bench.Settings, "load", classmethod(lambda cls: _settings()))
    monkeypatch.setattr(ab_bench, "load_env", lambda *args, **kwargs: [])

    class NeverGateway:
        def __init__(self, *args, **kwargs):  # pragma: no cover - assertion is the test
            raise AssertionError("dry-run must not construct a Gateway")

    monkeypatch.setattr(ab_bench, "Gateway", NeverGateway)
    receipt_path = tmp_path / "ab-bench.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ab_bench.py",
            "--model",
            "deepseek/deepseek-chat",
            "--repeat",
            "2",
            "--json-out",
            str(receipt_path),
        ],
    )

    assert ab_bench.main() == 0
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    out = capsys.readouterr().out

    assert "DRY RUN" in out
    assert receipt["mode"] == "dry-run"
    assert receipt["provenance"] == "modelled"
    assert receipt["ledger"] == {"calls": 0, "spend_usd_micros": 0}
    assert receipt["arms"]["baseline"]["planned_calls"] == 2
    assert receipt["arms"]["forgeos"]["accepted"] is None


def test_dry_receipt_marks_correctness_as_unmeasured():
    card = ModelCard(
        model_id="m",
        provider="p",
        input_cost_per_1m=1.0,
        output_cost_per_1m=1.0,
        context=10_000,
    )
    receipt = ab_bench.build_dry_receipt(
        model_ref="p/m",
        card=card,
        baseline_prompt="baseline",
        capsule_prompt="capsule",
        capsule_stats={"blocks_sent": 1},
        repeat=1,
        max_output=20,
        budget_usd=1.0,
        baseline_tokens=1,
        forgeos_tokens=1,
    )

    assert receipt["comparison"]["savings_class"] == "D"
    assert receipt["comparison"]["correctness_gated_saving"] is None
