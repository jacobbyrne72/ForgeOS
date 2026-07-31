import json

from forgeos import cli
from forgeos.prompt_cache import PromptCache


def test_stats_reports_entries_capacity_and_savings(tmp_path):
    cache = PromptCache(tmp_path / "cache.db")
    try:
        cache.store("local", "model", "prompt", "answer", cost_usd=0.25, tokens_out=12)
        assert cache.stats() == {
            "total_saved": 0.25,
            "total_tokens_saved": 12,
            "entries": 1,
            "utilization_pct": 0.1,
        }
    finally:
        cache.close()


def test_cache_stats_cli_is_json_and_closes_cleanly(tmp_path, monkeypatch, capsys):
    cache_path = tmp_path / "cache.db"
    monkeypatch.setattr(cli, "PromptCache", lambda: PromptCache(cache_path))

    assert cli.main(["cache", "stats"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["entries"] == 0
    assert payload["utilization_pct"] == 0.0
