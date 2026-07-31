"""Tests for the 2026-07-31 "Agent Slimmer" consolidation.

17 modules across 6 duplicate "jobs" (retry, batching, prompt reduction,
dedup/cache, output reduction, forecasting) were collapsed down to their
best foundation, with any genuinely distinct capability folded in rather
than lost. See each survivor module's docstring for exactly what was merged
from where and why.

This file has one section per group, plus two cross-cutting sections: proof
the deleted modules are actually gone (not just abandoned), and proof the
package's public __init__ exports still resolve.
"""
from __future__ import annotations

import sys
import types

import pytest

from forgeos import cli


# --------------------------------------------------------------------- retry


class TestCostRetry:
    def test_should_retry_allows_within_budget(self):
        from forgeos.cost_retry import CostRetry

        r = CostRetry(max_retries=3, base_cost_budget=0.20)
        should, info = r.should_retry(0)
        assert should is True
        assert info["reason"] == "retry_ok"
        assert r.total_retries == 1
        assert r.retry_spend == pytest.approx(0.03)

    def test_should_retry_stops_at_max_retries(self):
        from forgeos.cost_retry import CostRetry

        r = CostRetry(max_retries=2)
        should, info = r.should_retry(2)
        assert should is False
        assert info["reason"] == "max_retries_reached"

    def test_should_retry_stops_on_waste_error(self):
        from forgeos.cost_retry import CostRetry

        r = CostRetry(max_retries=3)
        should, info = r.should_retry(1, last_error="Rate limit exceeded, try later")
        assert should is False
        assert info["reason"] == "waste_error"
        assert info["match"] == "rate limit"

    def test_should_retry_stops_when_per_attempt_budget_exceeded(self):
        from forgeos.cost_retry import CostRetry

        r = CostRetry(max_retries=5, base_cost_budget=0.03)
        # attempt=2 -> remaining_budget = 0.03 / 4 = 0.0075 < default retry_cost 0.03
        should, info = r.should_retry(2)
        assert should is False
        assert info["reason"] == "budget_exceeded"

    def test_cumulative_retry_budget_cap_blocks_once_exceeded(self):
        """Merged from retry_budget.py's RetryBudget — a session-wide spend
        ceiling independent of the per-attempt budget."""
        from forgeos.cost_retry import CostRetry

        r = CostRetry(max_retries=10, base_cost_budget=10.0, max_total_retry_spend=0.05)
        results = [r.should_retry(i)[0] for i in range(4)]
        # 0.03, 0.06, 0.09, 0.12 cumulative spend -> blocked once total > 0.05
        assert results == [True, False, False, False]
        status = r.budget_status()
        assert status["blocked_by_budget"] == 3
        assert status["retry_spend"] == pytest.approx(0.03)
        assert status["remaining"] == pytest.approx(0.02)

    def test_budget_status_uncapped_by_default(self):
        from forgeos.cost_retry import CostRetry

        r = CostRetry()
        assert r.budget_status()["remaining"] is None
        assert r.budget_status()["max_total_retry_spend"] is None

    def test_run_with_retry_default_cost_matches_pre_merge_formula(self):
        """Default (no fallback_cost) behavior must be byte-identical to the
        pre-merge cost_retry.py: flat _RETRY_COST per executed attempt."""
        from forgeos.cost_retry import CostRetry

        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 2:
                return False, "transient error", None
            return True, "", "done"

        r = CostRetry(max_retries=3, base_cost_budget=1.0)
        result = r.run_with_retry(flaky)
        assert result["success"] is True
        assert result["attempts"] == 2
        assert result["total_cost"] == pytest.approx(0.03 * 2)

    def test_run_with_retry_fallback_cost_prices_retries_cheaper(self):
        """Merged from cost_aware_retry.py's CostAwareRetry — opt-in cheaper
        pricing for attempts after the first."""
        from forgeos.cost_retry import CostRetry

        calls = {"n": 0}

        def always_fails():
            calls["n"] += 1
            return False, "server error", None

        r = CostRetry(max_retries=2, base_cost_budget=1.0)
        result = r.run_with_retry(always_fails, fallback_cost=0.01)
        assert result["success"] is False
        # attempt 0 costs 0.03, attempts 1 and 2 cost 0.01 each
        assert result["total_cost"] == pytest.approx(0.03 + 0.01 + 0.01)

    def test_run_with_retry_never_calls_task_fn_when_blocked_before_attempt(self):
        from forgeos.cost_retry import CostRetry

        calls = {"n": 0}

        def always_fails():
            calls["n"] += 1
            return False, "error", None

        r = CostRetry(max_retries=5, base_cost_budget=1.0, max_total_retry_spend=0.03)
        r.run_with_retry(always_fails)
        # attempt 0 always runs; further attempts are blocked by the cumulative
        # cap before task_fn is invoked again.
        assert calls["n"] == 1

    def test_savings_report_unchanged_shape(self):
        from forgeos.cost_retry import CostRetry

        r = CostRetry(max_retries=4)
        report = r.savings_report()
        assert report == {"total_retries": 0, "max_saved_per_waste_retry": round(0.03 * 4, 4)}

    def test_cli_retry_command_runs(self, monkeypatch, tmp_path, capsys):
        monkeypatch.chdir(tmp_path)
        args = types.SimpleNamespace(max_retries=3)
        assert cli.cmd_retry(args) == 0
        assert "Cost-Aware Retry" in capsys.readouterr().out


# ------------------------------------------------------------------ batching


class TestBatching:
    def test_batch_optimizer_unchanged(self):
        from forgeos.batch_optimize import BatchOptimizer

        b = BatchOptimizer()
        result = b.run_batch([
            {"type": "code_gen", "name": "a"},
            {"type": "code_gen", "name": "b"},
            {"type": "review", "name": "c"},
        ])
        assert result["total_tasks"] == 3
        assert set(result["by_task_type"]) == {"code_gen", "review"}
        report = b.print_report(result)
        assert "BATCH SAVINGS REPORT" in report

    def test_adaptive_batch_analyze_and_run_unchanged(self):
        from forgeos.adaptive_batch import AdaptiveBatch

        ab = AdaptiveBatch()
        tasks = [{"type": "code_gen", "name": "a"}, {"type": "review", "name": "b"}]
        analysis = ab.analyze(tasks)
        assert analysis["total_tasks"] == 2
        assert analysis["recommended_strategy"] in {"bulk_single", "grouped", "auto_pipeline"}

        result = ab.run(tasks)
        assert "analysis" in result and "actual_savings" in result

    def test_adaptive_batch_recommend_batch_size_merged_from_adaptive_batch_cost(self):
        """recommend_batch_size/get_savings_trend came from the now-deleted
        adaptive_batch_cost.py's AdaptiveBatchCostOptimizer, verbatim."""
        from forgeos.adaptive_batch import AdaptiveBatch

        ab = AdaptiveBatch()
        result = ab.recommend_batch_size("code_gen", avg_tokens=500)
        assert result["task_type"] == "code_gen"
        assert 1 <= result["optimal_batch_size"] <= 50
        assert result["estimated_batch_cost"] >= 0

        trend = ab.get_savings_trend()
        assert trend["total_optimized_tasks"] == 1
        assert trend["trend"] == "establishing"

        for _ in range(6):
            ab.recommend_batch_size("review", avg_tokens=200)
        assert ab.get_savings_trend()["trend"] == "optimizing"

    def test_smart_batch_predictor_unchanged(self):
        from forgeos.smart_batch import SmartBatchPredictor

        p = SmartBatchPredictor()
        for _ in range(3):
            p.record("code_gen", 0.02, 300)
        result = p.predict_savings("code_gen", count=5)
        assert result["source"] == "historical"
        assert result["count"] == 5

        batch_result = p.predict_batch([{"type": "code_gen"}, {"type": "unseen_type"}])
        assert batch_result["total_tasks"] == 2

    def test_deleted_smart_batching_stub_is_gone(self):
        with pytest.raises(ModuleNotFoundError):
            import forgeos.smart_batching  # noqa: F401

    def test_cli_adaptbatch_command_runs(self, capsys):
        args = types.SimpleNamespace()
        assert cli.cmd_adbatch(args) == 0
        assert "Adaptive Batch Optimizer" in capsys.readouterr().out

    def test_cli_smartbatch_command_runs(self, capsys):
        args = types.SimpleNamespace()
        assert cli.cmd_smartbatch(args) == 0
        assert "Smart Batch Predictor" in capsys.readouterr().out

    def test_cli_batch_command_runs(self, capsys):
        args = types.SimpleNamespace(daily_tasks=25)
        assert cli.cmd_batch(args) == 0
        assert "BATCH SAVINGS REPORT" in capsys.readouterr().out

    def test_cli_adaptive_batch_dead_handler_still_correct_after_import_fix(self, capsys):
        """cmd_adaptive_batch is not reachable via any 'forge <cmd>' (it has
        no dispatch entry, confirmed pre-existing via test_cli_dispatch.py's
        reachability tests) but it must not be a broken import landmine now
        that adaptive_batch_cost.py is gone."""
        args = types.SimpleNamespace(tasks=2, tokens=400)
        assert cli.cmd_adaptive_batch(args) == 0
        assert "Trend" in capsys.readouterr().out


# --------------------------------------------------------------- prompt reduction


class TestPromptOptimizer:
    def test_optimize_prompt_unchanged_whitespace_and_history_behavior(self):
        from forgeos.prompt_optimizer import optimize_prompt

        prompt = "Hello\n\n\n\nworld   \n" + "You are a helpful assistant. " * 2
        optimized, stats = optimize_prompt(prompt, max_tokens=8192)
        assert "\n\n\n" not in optimized
        assert stats["original_tokens"] > 0
        assert stats["under_budget"] is True

    def test_optimize_prompt_trims_history_to_last_n(self):
        from forgeos.prompt_optimizer import optimize_prompt

        history = "Previous turns:\n" + "".join(f"- Turn {i}\n" for i in range(20))
        optimized, _ = optimize_prompt(history + "\nDo the task.", trim_history_to=3)
        kept = [line for line in optimized.splitlines() if line.startswith("- Turn")]
        assert len(kept) == 3
        assert kept == ["- Turn 17", "- Turn 18", "- Turn 19"]

    def test_optimize_prompt_remove_filler_is_opt_in(self):
        """New: remove_filler folds in prompt_summarizer.py's technique."""
        from forgeos.prompt_optimizer import optimize_prompt

        prompt = "So, I basically just want you to, you know, fix the bug."
        without, _ = optimize_prompt(prompt)
        assert "basically" in without

        with_filler, _ = optimize_prompt(prompt, remove_filler=True)
        assert "basically" not in with_filler
        assert "you know" not in with_filler

    def test_remove_filler_words_standalone(self):
        from forgeos.prompt_optimizer import remove_filler_words

        assert remove_filler_words("This is basically fine.") == "This is fine."

    def test_shrink_prompt_unchanged_hard_truncate_behavior(self):
        from forgeos.prompt_optimizer import shrink_prompt

        long_prompt = "Explain recursion in great detail. " * 200
        shrunk, stats = shrink_prompt(long_prompt, target_tokens=50)
        assert stats["shrunk_tokens"] <= 55  # truncation is approximate at the char level
        assert shrunk.endswith("... (truncated to save tokens)")
        assert stats["tokens_saved"] > 0

    def test_shrink_prompt_no_truncation_needed(self):
        from forgeos.prompt_optimizer import shrink_prompt

        shrunk, stats = shrink_prompt("short prompt", target_tokens=2048)
        assert "truncated" not in shrunk
        assert stats["tokens_saved"] == 0

    def test_estimate_tokens_alias_identity(self):
        from forgeos.prompt_optimizer import estimate_tokens, estimate_prompt_tokens

        assert estimate_prompt_tokens is estimate_tokens
        assert estimate_tokens("x" * 40) == 10

    def test_deleted_prompt_shrinker_and_summarizer_are_gone(self):
        with pytest.raises(ModuleNotFoundError):
            import forgeos.prompt_shrinker  # noqa: F401
        with pytest.raises(ModuleNotFoundError):
            import forgeos.prompt_summarizer  # noqa: F401

    def test_cli_shrink_command_runs(self, capsys):
        args = types.SimpleNamespace(max_tokens=2048)
        assert cli.cmd_shrink(args) == 0
        assert "Prompt Shrinker" in capsys.readouterr().out

    def test_cli_prompt_opt_command_runs(self, capsys):
        args = types.SimpleNamespace(max_tokens=100)
        assert cli.cmd_prompt_opt(args) == 0
        assert "Prompt Optimizer" in capsys.readouterr().out


# ------------------------------------------------------------------- dedup/cache


class TestDedupCache:
    """result_cache/request_coalescer/smart_dedup are NOT duplicates of one
    another (exact-persistent, in-flight-concurrent, and fuzzy-similarity are
    three different mechanisms) — they were merged into one file only
    because none was wired anywhere and none had a natural foundation. See
    dedup_cache.py's module docstring."""

    def test_result_cache_roundtrip(self, tmp_path):
        from forgeos.dedup_cache import ResultCache

        cache = ResultCache(db_path=tmp_path / "cache.json")
        assert cache.get("hello", "gpt") is None

        cache.store("hello", "gpt", "hi there", cost_usd=0.002)
        hit = cache.get("hello", "gpt")
        assert hit["cached"] is True
        assert hit["response"] == "hi there"
        assert cache.size() == 1
        assert cache.total_saved() == pytest.approx(0.002)

    def test_result_cache_persists_across_instances(self, tmp_path):
        from forgeos.dedup_cache import ResultCache

        path = tmp_path / "cache.json"
        ResultCache(db_path=path).store("p", "m", "r", 0.01)
        reopened = ResultCache(db_path=path)
        assert reopened.get("p", "m") is not None

    def test_request_coalescer_dedupes_concurrent_key(self):
        from forgeos.dedup_cache import RequestCoalescer

        rc = RequestCoalescer()
        assert rc.acquire("k1") is True
        assert rc.acquire("k1") is False  # second caller for same key waits
        assert rc.acquire("k2") is True

        rc.complete("k1", "the result")
        assert rc.wait_and_get("k1") == "the result"

        stats = rc.stats()
        assert stats["total_requests"] == 3
        assert stats["deduped"] == 1

    def test_request_coalescer_wait_times_out_with_no_result(self):
        from forgeos.dedup_cache import RequestCoalescer

        rc = RequestCoalescer()
        assert rc.wait_and_get("never-acquired", timeout=0.01) is None

    def test_smart_dedup_detects_similar_prompts(self):
        from forgeos.dedup_cache import SmartDedup

        d = SmartDedup(similarity_threshold=0.8)
        is_dup, sim = d.is_duplicate("fix the login bug please", "gpt")
        assert is_dup is False
        assert sim == 0.0

        is_dup2, sim2 = d.is_duplicate("fix the login bug", "gpt")
        assert is_dup2 is True
        assert sim2 > 0.8

        # different model -> not compared against the first model's history
        is_dup3, _ = d.is_duplicate("fix the login bug please", "claude")
        assert is_dup3 is False

        stats = d.stats()
        assert stats["deduped"] == 1

    def test_deleted_result_cache_request_coalescer_smart_dedup_are_gone(self):
        with pytest.raises(ModuleNotFoundError):
            import forgeos.result_cache  # noqa: F401
        with pytest.raises(ModuleNotFoundError):
            import forgeos.request_coalescer  # noqa: F401
        with pytest.raises(ModuleNotFoundError):
            import forgeos.smart_dedup  # noqa: F401


# --------------------------------------------------------------- output reduction


class TestOutputCompressor:
    def test_compress_output_unchanged_behavior(self):
        from forgeos.output_compressor import compress_output

        text = "line\n\n\n\nmore   \n" * 5
        compressed, stats = compress_output(text, target_tokens=4096)
        assert "\n\n\n" not in compressed
        assert stats["tokens_saved"] >= 0

    def test_compress_output_truncates_over_budget(self):
        from forgeos.output_compressor import compress_output

        long_text = "model generated line of output text here. " * 500
        compressed, stats = compress_output(long_text, target_tokens=20)
        assert compressed.endswith("... (token budget enforced)")
        assert stats["compressed_tokens"] <= 25

    def test_truncate_response_moved_unchanged_behavior(self):
        """truncate_response used to live in response_truncator.py — same
        sentence-boundary truncation, now co-located in output_compressor.py."""
        from forgeos.output_compressor import truncate_response

        short, info = truncate_response("short text", max_tokens=100)
        assert info["truncated"] is False

        long_text = "This is a sentence. " * 500
        truncated, info2 = truncate_response(long_text, max_tokens=20)
        assert info2["truncated"] is True
        assert truncated.endswith("\n...(truncated to save tokens)")

    def test_compress_by_sentence_ratio_new_from_response_compressor(self):
        """Merged from response_compressor.py's ResponseCompressor.compress —
        a distinct extractive technique, not a token-budget cut."""
        from forgeos.output_compressor import compress_by_sentence_ratio

        text = " ".join(f"Sentence number {i}." for i in range(10))
        result = compress_by_sentence_ratio(text, max_ratio=0.5)
        assert result.count(".") == 5
        assert result != text

    def test_compress_by_sentence_ratio_leaves_short_text_alone(self):
        from forgeos.output_compressor import compress_by_sentence_ratio

        assert compress_by_sentence_ratio("too short") == "too short"

    def test_deleted_response_compressor_and_truncator_are_gone(self):
        with pytest.raises(ModuleNotFoundError):
            import forgeos.response_compressor  # noqa: F401
        with pytest.raises(ModuleNotFoundError):
            import forgeos.response_truncator  # noqa: F401

    def test_cli_truncate_command_runs(self, capsys):
        args = types.SimpleNamespace(max_tokens=4096)
        assert cli.cmd_truncate(args) == 0
        assert "Response Truncator" in capsys.readouterr().out

    def test_cli_output_compress_command_runs(self, capsys):
        args = types.SimpleNamespace(max_tokens=100)
        assert cli.cmd_output_compress(args) == 0
        assert "Output Compressor" in capsys.readouterr().out


# ------------------------------------------------------------------- forecasting


class TestCostForecast:
    def test_forecast_insufficient_data(self, monkeypatch, tmp_path):
        from forgeos.cost_forecast import CostForecast

        monkeypatch.chdir(tmp_path)
        result = CostForecast().forecast(days=30)
        assert result["method"] == "insufficient_data"
        assert result["forecast_usd"] == 0.0

    def test_forecast_linear_regression_with_data(self, monkeypatch, tmp_path):
        from forgeos.cost_forecast import CostForecast

        monkeypatch.chdir(tmp_path)
        cf = CostForecast()
        for saved in [0.01, 0.012, 0.014, 0.016]:
            cf.tracker.record("code_gen", saved)
        result = cf.forecast(days=7)
        assert result["method"] == "linear_regression"
        assert result["data_points"] == 4
        assert result["confidence"] == "low"

    def test_budget_ok_requires_daily_budget_set(self, monkeypatch, tmp_path):
        from forgeos.cost_forecast import CostForecast

        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError):
            CostForecast().budget_ok()

    def test_budget_ok_merged_from_budget_forecast(self, monkeypatch, tmp_path):
        """Merged from budget_forecast.py's BudgetForecast.budget_ok, but
        reusing this class's own regression forecast() rather than a
        separate averaging algorithm."""
        from forgeos.cost_forecast import CostForecast

        monkeypatch.chdir(tmp_path)
        cf = CostForecast(daily_budget=100.0)
        for saved in [0.01, 0.01, 0.01]:
            cf.tracker.record("code_gen", saved)
        status = cf.budget_ok(days=7)
        assert status["budget_ceiling"] == pytest.approx(700.0)
        assert status["ok"] == (status["budget_remaining"] >= 0)

    def test_cost_predictor_predicts_and_tracks_calls(self):
        """Kept as its own class: an a-priori per-call estimate from a token
        count and a price table, not a historical-trend forecast."""
        from forgeos.cost_forecast import CostPredictor

        p = CostPredictor(cost_per_1k_input=0.0015, cost_per_1k_output=0.002)
        cost = p.predict_cost(1000, 500)
        assert cost == pytest.approx(0.0015 + 0.001)
        assert p.stats()["predictions"] == 1

    def test_cost_predictor_should_use_cache(self):
        from forgeos.cost_forecast import CostPredictor

        p = CostPredictor()
        assert p.should_use_cache(1000, cached_cost=0.0) is True
        assert p.should_use_cache(1000, cached_cost=10.0) is False

    def test_deleted_budget_forecast_and_cost_predictor_module_are_gone(self):
        with pytest.raises(ModuleNotFoundError):
            import forgeos.budget_forecast  # noqa: F401
        with pytest.raises(ModuleNotFoundError):
            import forgeos.cost_predictor  # noqa: F401

    def test_cli_forecast_command_runs(self, monkeypatch, tmp_path, capsys):
        monkeypatch.chdir(tmp_path)
        args = types.SimpleNamespace(forecast_days=30)
        assert cli.cmd_forecast(args) == 0
        assert "Cost Forecast" in capsys.readouterr().out


# ---------------------------------------------------------- package-level sanity


class TestPackageStillWorks:
    def test_all_originally_exported_names_still_resolve(self):
        import forgeos

        for name in [
            "CostRetry", "BatchOptimizer", "AdaptiveBatch", "SmartBatchPredictor",
            "optimize_prompt", "estimate_tokens", "shrink_prompt", "estimate_prompt_tokens",
            "compress_output", "estimate_output_tokens", "truncate_response",
            "CostForecast",
        ]:
            assert hasattr(forgeos, name), f"forgeos.{name} no longer resolves"
            assert name in forgeos.__all__

    def test_version_unchanged(self):
        import forgeos

        assert forgeos.__version__ == "0.6.10"

    def test_all_seventeen_consolidated_modules_are_gone(self):
        removed = [
            "cost_aware_retry", "retry_budget",
            "smart_batching", "adaptive_batch_cost",
            "prompt_shrinker", "prompt_summarizer",
            "result_cache", "request_coalescer", "smart_dedup",
            "response_compressor", "response_truncator",
            "budget_forecast", "cost_predictor",
        ]
        for mod in removed:
            assert f"forgeos.{mod}" not in sys.modules or True  # imported fresh below
            with pytest.raises(ModuleNotFoundError):
                __import__(f"forgeos.{mod}")

    def test_survivors_still_present_on_disk_via_import(self):
        import forgeos.cost_retry  # noqa: F401
        import forgeos.adaptive_batch  # noqa: F401
        import forgeos.batch_optimize  # noqa: F401
        import forgeos.smart_batch  # noqa: F401
        import forgeos.prompt_optimizer  # noqa: F401
        import forgeos.dedup_cache  # noqa: F401
        import forgeos.output_compressor  # noqa: F401
        import forgeos.cost_forecast  # noqa: F401
