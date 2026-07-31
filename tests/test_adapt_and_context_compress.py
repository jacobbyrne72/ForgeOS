from forgeos.adapt import AdapterProfiler
from forgeos.context_compress import compress_context


def test_profiler_selects_the_cheapest_capable_successful_adapter():
    profiler = AdapterProfiler()
    profiler.record_task("slow", 20_000, 2.0, True, {"python"})
    profiler.record_task("fast", 5_000, 1.0, True, {"python"})

    decision = profiler.best_adapter({"python"})

    assert decision is not None
    assert decision.adapter_name == "fast"
    assert decision.estimated_cost_usd == 0.005


def test_profiler_rejects_adapters_without_required_capabilities():
    profiler = AdapterProfiler()
    profiler.record_task("docs", 1_000, 1.0, True, {"writing"})

    assert profiler.best_adapter({"python"}) is None


def test_context_compression_keeps_the_matching_symbol_and_drops_the_rest():
    """Asserts the CONTENT kept, not the exact bytes.

    This previously pinned `"def parse_retry():\\n    return 3\\n"` exactly --
    a trailing newline that was an artefact of slicing lines, from back when
    compression returned line windows rather than whole symbols. Pinning the
    artefact meant the test failed when the module started returning
    syntactically complete functions, which was the improvement.
    """
    files = [("retry.py", "def parse_retry():\n    return 3\n\ndef unrelated():\n    return 0")]

    compressed = compress_context("retry parsing", files)

    assert len(compressed) == 1
    path, body = compressed[0]
    assert path == "retry.py"
    assert "def parse_retry():" in body and "return 3" in body
    assert "unrelated" not in body
    compile(body, "<kept>", "exec")  # whole symbol, not a fragment
